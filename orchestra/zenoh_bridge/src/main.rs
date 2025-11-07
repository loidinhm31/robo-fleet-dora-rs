use dora_node_api::{
    arrow::array::{Array, BinaryArray, Float32Array, UInt8Array},
    dora_core::config::DataId,
    DoraNode, Event, Parameter,
};
use eyre::Result;
use robo_rover_lib::{init_tracing, FleetSelectCommand, FleetSubscriptionCommand};
use serde_json;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex;
use zenoh::Config;

// Type alias for Zenoh subscriber (default handler)
type ZenohSubscriber = zenoh::pubsub::Subscriber<zenoh::handlers::FifoChannelHandler<zenoh::sample::Sample>>;

/// Subscriptions for a single rover
struct RoverSubscriptions {
    entity_id: String,

    // Data subscribers (FROM rover)
    video_sub: ZenohSubscriber,
    audio_sub: ZenohSubscriber,
    rover_telemetry_sub: ZenohSubscriber,
    arm_telemetry_sub: ZenohSubscriber,
    servo_telemetry_sub: ZenohSubscriber,
    tracked_detections_sub: ZenohSubscriber,
    tracking_telemetry_sub: ZenohSubscriber,
    metrics_sub: ZenohSubscriber,
}

/// Subscribe to all topics for a specific rover
async fn subscribe_to_rover(
    session: &Arc<zenoh::Session>,
    entity_id: &str,
) -> Result<RoverSubscriptions> {
    tracing::info!("Subscribing to rover: {}", entity_id);

    let video_topic = format!("rover/{}/video/raw", entity_id);
    let video_sub = session.declare_subscriber(&video_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to subscribe to {}: {}", video_topic, e))?;
    tracing::info!("{}", video_topic);

    let audio_topic = format!("rover/{}/audio/raw", entity_id);
    let audio_sub = session.declare_subscriber(&audio_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to subscribe to {}: {}", audio_topic, e))?;
    tracing::info!("{}", audio_topic);

    let rover_telemetry_topic = format!("rover/{}/telemetry/rover", entity_id);
    let rover_telemetry_sub = session.declare_subscriber(&rover_telemetry_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to subscribe to {}: {}", rover_telemetry_topic, e))?;
    tracing::info!("{}", rover_telemetry_topic);

    let arm_telemetry_topic = format!("rover/{}/telemetry/arm", entity_id);
    let arm_telemetry_sub = session.declare_subscriber(&arm_telemetry_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to subscribe to {}: {}", arm_telemetry_topic, e))?;
    tracing::info!("{}", arm_telemetry_topic);

    let servo_telemetry_topic = format!("rover/{}/telemetry/servo", entity_id);
    let servo_telemetry_sub = session.declare_subscriber(&servo_telemetry_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to subscribe to {}: {}", servo_telemetry_topic, e))?;
    tracing::info!("{}", servo_telemetry_topic);

    let tracked_detections_topic = format!("rover/{}/video/detections", entity_id);
    let tracked_detections_sub = session.declare_subscriber(&tracked_detections_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to subscribe to {}: {}", tracked_detections_topic, e))?;
    tracing::info!("{}", tracked_detections_topic);

    let tracking_telemetry_topic = format!("rover/{}/telemetry/tracking", entity_id);
    let tracking_telemetry_sub = session.declare_subscriber(&tracking_telemetry_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to subscribe to {}: {}", tracking_telemetry_topic, e))?;
    tracing::info!("{}", tracking_telemetry_topic);

    let metrics_topic = format!("rover/{}/metrics", entity_id);
    let metrics_sub = session.declare_subscriber(&metrics_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to subscribe to {}: {}", metrics_topic, e))?;
    tracing::info!("{}", metrics_topic);

    Ok(RoverSubscriptions {
        entity_id: entity_id.to_string(),
        video_sub,
        audio_sub,
        rover_telemetry_sub,
        arm_telemetry_sub,
        servo_telemetry_sub,
        tracked_detections_sub,
        tracking_telemetry_sub,
        metrics_sub,
    })
}

/// Unsubscribe from a rover (cleanup)
fn unsubscribe_from_rover(subs: RoverSubscriptions) {
    tracing::info!("Unsubscribing from rover: {}", subs.entity_id);
    // Subscriptions are dropped automatically
    drop(subs);
}

/// Handle fleet subscription commands (activate/deactivate rovers)
async fn handle_fleet_subscription_command(
    active_rovers: &mut HashMap<String, RoverSubscriptions>,
    session: &Arc<zenoh::Session>,
    data: dora_node_api::ArrowData,
) -> Result<()> {
    if let Some(binary_array) = data.0.as_any().downcast_ref::<BinaryArray>() {
        if binary_array.len() > 0 {
            let bytes = binary_array.value(0);
            let cmd: FleetSubscriptionCommand = serde_json::from_slice(bytes)?;

            match cmd {
                FleetSubscriptionCommand::ActivateRover { entity_id, .. } => {
                    if !active_rovers.contains_key(&entity_id) {
                        tracing::info!("Activating rover: {}", entity_id);
                        let subs = subscribe_to_rover(session, &entity_id).await?;
                        active_rovers.insert(entity_id, subs);
                    } else {
                        tracing::warn!("Rover {} already active", entity_id);
                    }
                }

                FleetSubscriptionCommand::DeactivateRover { entity_id, .. } => {
                    if let Some(subs) = active_rovers.remove(&entity_id) {
                        tracing::info!("Deactivating rover: {}", entity_id);
                        unsubscribe_from_rover(subs);
                    } else {
                        tracing::warn!("Rover {} not active", entity_id);
                    }
                }

                FleetSubscriptionCommand::SetActiveRovers { entity_ids, .. } => {
                    tracing::info!("Setting active rovers: {:?}", entity_ids);

                    // Remove rovers not in new list
                    let to_remove: Vec<String> = active_rovers.keys()
                        .filter(|k| !entity_ids.contains(k))
                        .cloned()
                        .collect();

                    for rover_id in to_remove {
                        if let Some(subs) = active_rovers.remove(&rover_id) {
                            tracing::info!("  - Removing: {}", rover_id);
                            unsubscribe_from_rover(subs);
                        }
                    }

                    // Add new rovers
                    for rover_id in entity_ids {
                        if !active_rovers.contains_key(&rover_id) {
                            tracing::info!("  + Adding: {}", rover_id);
                            let subs = subscribe_to_rover(session, &rover_id).await?;
                            active_rovers.insert(rover_id, subs);
                        }
                    }
                }
            }

            tracing::info!("Active rovers: {:?}", active_rovers.keys().collect::<Vec<_>>());
        }
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let _guard = init_tracing();

    tracing::info!("Starting Orchestra Zenoh Bridge (Multi-Rover)");

    // Get entity IDs from environment
    let entity_id = std::env::var("ENTITY_ID").unwrap_or_else(|_| "orchestra".to_string());
    tracing::info!("Orchestra ID: {}", entity_id);

    // Get initial active rovers from environment
    let active_rovers_env = std::env::var("ACTIVE_ROVERS")
        .unwrap_or_else(|_| "rover-kiwi".to_string());
    let initial_rovers: Vec<String> = active_rovers_env
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    tracing::info!("Initial active rovers: {:?}", initial_rovers);

    // Video frame configuration
    let frame_width = std::env::var("VIDEO_WIDTH")
        .ok()
        .and_then(|v| v.parse::<i64>().ok())
        .unwrap_or(640);
    let frame_height = std::env::var("VIDEO_HEIGHT")
        .ok()
        .and_then(|v| v.parse::<i64>().ok())
        .unwrap_or(480);

    tracing::info!("Expected video frame dimensions: {}x{}", frame_width, frame_height);

    // Audio configuration
    let audio_sample_rate = std::env::var("AUDIO_SAMPLE_RATE")
        .ok()
        .and_then(|v| v.parse::<i64>().ok())
        .unwrap_or(16000);
    let audio_channels = std::env::var("AUDIO_CHANNELS")
        .ok()
        .and_then(|v| v.parse::<i64>().ok())
        .unwrap_or(1);

    tracing::info!("Expected audio format: {}Hz, {} channels", audio_sample_rate, audio_channels);

    // Initialize Dora node
    let (mut node, mut events) = DoraNode::init_from_env()?;

    // Initialize Zenoh session
    let config_path = std::env::var("ZENOH_CONFIG")
        .unwrap_or_else(|_| "orchestra/zenoh_bridge/zenoh_config.json5".to_string());

    // Log current working directory for debugging
    if let Ok(cwd) = std::env::current_dir() {
        tracing::info!("Current working directory: {}", cwd.display());
    }
    tracing::info!("Loading Zenoh config from: {}", config_path);

    let config = if std::path::Path::new(&config_path).exists() {
        tracing::info!("Config file found");
        Config::from_file(&config_path)
            .map_err(|e| eyre::eyre!("Failed to load Zenoh config from {}: {}", config_path, e))?
    } else {
        tracing::warn!("Config file not found at {}", config_path);
        tracing::warn!("Using default config with peer mode");
        let mut config = Config::default();
        config.insert_json5("mode", "\"peer\"")
            .map_err(|e| eyre::eyre!("Failed to set Zenoh mode: {}", e))?;
        config
    };

    let session = Arc::new(zenoh::open(config).await
        .map_err(|e| eyre::eyre!("Failed to open Zenoh session: {}", e))?);

    tracing::info!("Zenoh session ID: {}", session.zid());

    // =========================================================================
    // Initialize subscriptions for active rovers
    // =========================================================================

    let mut active_rovers: HashMap<String, RoverSubscriptions> = HashMap::new();
    let mut selected_entity: Option<String> = None;

    for rover_id in initial_rovers {
        let subs = subscribe_to_rover(&session, &rover_id).await?;
        active_rovers.insert(rover_id.clone(), subs);
        // Select first rover by default
        if selected_entity.is_none() {
            selected_entity = Some(rover_id);
        }
    }

    if let Some(ref entity) = selected_entity {
        tracing::info!("Selected entity for commands: {}", entity);
    }

    // =========================================================================
    // PUBLISHERS: Send commands TO rovers via Zenoh
    // Note: Publishers are now created dynamically per command
    // =========================================================================

    // =========================================================================
    // Dora output DataIds
    // =========================================================================

    let video_frame_output = DataId::from("video_frame".to_owned());
    let audio_frame_output = DataId::from("audio_frame".to_owned());
    let rover_telemetry_output = DataId::from("rover_telemetry".to_owned());
    let arm_telemetry_output = DataId::from("arm_telemetry".to_owned());
    let servo_telemetry_output = DataId::from("servo_telemetry".to_owned());
    let tracked_detections_output = DataId::from("tracked_detections".to_owned());
    let tracking_telemetry_output = DataId::from("tracking_telemetry".to_owned());
    let performance_metrics_output = DataId::from("performance_metrics".to_owned());

    // Statistics per rover
    let video_counts: Arc<Mutex<HashMap<String, u64>>> = Arc::new(Mutex::new(HashMap::new()));
    let audio_counts: Arc<Mutex<HashMap<String, u64>>> = Arc::new(Mutex::new(HashMap::new()));

    // Create channel to bridge Dora's sync events to async
    let (dora_tx, dora_rx) = flume::unbounded();

    // Spawn task to read Dora events
    std::thread::spawn(move || {
        while let Some(event) = events.recv() {
            if dora_tx.send(event).is_err() {
                break;
            }
        }
    });

    tracing::info!("Entering main event loop...");

    // =========================================================================
    // Main event loop - Dynamic multi-rover subscription
    // =========================================================================

    loop {
        // Build select! branches dynamically for all active rovers
        tokio::select! {
            // Handle Dora events (commands and fleet management)
            Ok(event) = dora_rx.recv_async() => {
                match event {
                    Event::Input { id, data, .. } => {
                        match id.as_str() {
                            // Fleet subscription management
                            "fleet_subscription_command" => {
                                if let Err(e) = handle_fleet_subscription_command(
                                    &mut active_rovers,
                                    &session,
                                    data
                                ).await {
                                    tracing::error!("Fleet subscription error: {}", e);
                                }
                            }

                            // Fleet selection (which rover to send commands to)
                            "fleet_select_command" => {
                                if let Some(binary_array) = data.as_any().downcast_ref::<BinaryArray>() {
                                    if binary_array.len() > 0 {
                                        let bytes = binary_array.value(0);
                                        if let Ok(cmd) = serde_json::from_slice::<FleetSelectCommand>(bytes) {
                                            if active_rovers.contains_key(&cmd.entity_id) {
                                                selected_entity = Some(cmd.entity_id.clone());
                                                tracing::info!("Selected entity for commands: {}", cmd.entity_id);
                                            } else {
                                                tracing::warn!("Cannot select inactive rover: {}", cmd.entity_id);
                                            }
                                        }
                                    }
                                }
                            }

                            // Audio stream (web UI walkie-talkie mode)
                            "audio_stream_web" => {
                                if let Some(float32_array) = data.as_any().downcast_ref::<Float32Array>() {
                                    if float32_array.len() > 0 {
                                        // Route to currently selected rover
                                        if let Some(ref entity_id) = selected_entity {
                                            if active_rovers.contains_key(entity_id) {
                                                let audio_stream_topic = format!("rover/{}/cmd/audio_stream", entity_id);

                                                let float_slice = float32_array.values().as_ref();
                                                let bytes: &[u8] = unsafe {
                                                    std::slice::from_raw_parts(
                                                        float_slice.as_ptr() as *const u8,
                                                        float_slice.len() * std::mem::size_of::<f32>()
                                                    )
                                                };
                                                let _ = session.put(audio_stream_topic, bytes).await;
                                            } else {
                                                tracing::warn!("Selected rover {} is not active", entity_id);
                                            }
                                        } else {
                                            tracing::warn!("No rover selected for audio stream");
                                        }
                                    }
                                }
                            }

                            // Other commands (BinaryArray - JSON serialized)
                            _ if data.as_any().is::<BinaryArray>() => {
                                if let Some(binary_array) = data.as_any().downcast_ref::<BinaryArray>() {
                                    if binary_array.len() > 0 {
                                        let bytes = binary_array.value(0);

                                        // Route command to currently selected rover
                                        if let Some(ref entity_id) = selected_entity {
                                            if active_rovers.contains_key(entity_id) {
                                                let topic = match id.as_str() {
                                                    "rover_command_web" | "rover_command_parser" => {
                                                        Some(format!("rover/{}/cmd/movement", entity_id))
                                                    }
                                                    "arm_command_web" | "arm_command_parser" => {
                                                        Some(format!("rover/{}/cmd/arm", entity_id))
                                                    }
                                                    "camera_command_web" | "camera_control_parser" => {
                                                        Some(format!("rover/{}/cmd/camera", entity_id))
                                                    }
                                                    "audio_command_web" => {
                                                        Some(format!("rover/{}/cmd/audio", entity_id))
                                                    }
                                                    "tracking_command_web" | "tracking_command_parser" => {
                                                        Some(format!("rover/{}/cmd/tracking", entity_id))
                                                    }
                                                    "tts_command_web" | "tts_command_parser" => {
                                                        Some(format!("rover/{}/cmd/tts", entity_id))
                                                    }
                                                    _ => None,
                                                };

                                                if let Some(topic) = topic {
                                                    tracing::debug!("Routing command to {}: {}", entity_id, topic);
                                                    let _ = session.put(&topic, bytes).await;
                                                }
                                            } else {
                                                tracing::warn!("Selected rover {} is not active", entity_id);
                                            }
                                        } else {
                                            tracing::warn!("No rover selected for command: {}", id.as_str());
                                        }
                                    }
                                }
                            }

                            _ => {}
                        }
                    }
                    Event::Stop(_) => {
                        tracing::info!("Stop signal received");
                        let video_counts_map = video_counts.lock().await;
                        let audio_counts_map = audio_counts.lock().await;
                        for (rover_id, count) in video_counts_map.iter() {
                            tracing::info!("  {}: video={}", rover_id, count);
                        }
                        for (rover_id, count) in audio_counts_map.iter() {
                            tracing::info!("  {}: audio={}", rover_id, count);
                        }
                        break;
                    }
                    _ => {}
                }
            }

            // Receive from all active rovers' video subscriptions
            result = receive_from_rovers(&active_rovers, |subs| &subs.video_sub) => {
                if let Some((entity_id, sample)) = result {
                    let payload = sample.payload().to_bytes();

                    // Forward raw RGB8 data as UInt8Array with entity_id in metadata
                    let video_array = UInt8Array::from(payload.to_vec());

                    // Add entity_id and video metadata to parameters
                    let mut params = BTreeMap::new();
                    params.insert("entity_id".to_owned(), Parameter::String(entity_id.clone()));
                    params.insert("width".to_owned(), Parameter::Integer(frame_width));
                    params.insert("height".to_owned(), Parameter::Integer(frame_height));
                    params.insert("encoding".to_owned(), Parameter::String("rgb8".to_string()));
                    params.insert("timestamp".to_owned(),
                        Parameter::Integer(SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as i64)
                    );

                    let _ = node.send_output(video_frame_output.clone(), params, video_array);

                    let mut counts = video_counts.lock().await;
                    *counts.entry(entity_id).or_insert(0) += 1;
                    if counts.values().sum::<u64>() % 30 == 0 {
                        tracing::debug!("Video frames: {:?}", counts);
                    }
                }
            }

            // Receive from all active rovers' audio subscriptions
            result = receive_from_rovers(&active_rovers, |subs| &subs.audio_sub) => {
                if let Some((entity_id, sample)) = result {
                    let payload = sample.payload().to_bytes();

                    // Convert raw bytes to Float32Array
                    let float_slice: &[f32] = unsafe {
                        std::slice::from_raw_parts(
                            payload.as_ref().as_ptr() as *const f32,
                            payload.len() / std::mem::size_of::<f32>()
                        )
                    };
                    let audio_array = Float32Array::from(float_slice.to_vec());

                    // Create metadata with entity_id
                    let mut params = BTreeMap::new();
                    params.insert("sample_rate".to_owned(), Parameter::Integer(audio_sample_rate));
                    params.insert("channels".to_owned(), Parameter::Integer(audio_channels));
                    params.insert("format".to_owned(), Parameter::String("f32le".to_string()));
                    params.insert("entity_id".to_owned(), Parameter::String(entity_id.clone()));

                    let _ = node.send_output(audio_frame_output.clone(), params, audio_array);

                    let mut counts = audio_counts.lock().await;
                    *counts.entry(entity_id).or_insert(0) += 1;
                }
            }

            // Receive from all active rovers' rover telemetry
            result = receive_from_rovers(&active_rovers, |subs| &subs.rover_telemetry_sub) => {
                if let Some((entity_id, sample)) = result {
                    forward_telemetry_with_entity_id(
                        &mut node,
                        &rover_telemetry_output,
                        entity_id,
                        sample
                    );
                }
            }

            // Receive from all active rovers' arm telemetry
            result = receive_from_rovers(&active_rovers, |subs| &subs.arm_telemetry_sub) => {
                if let Some((entity_id, sample)) = result {
                    forward_telemetry_with_entity_id(
                        &mut node,
                        &arm_telemetry_output,
                        entity_id,
                        sample
                    );
                }
            }

            // Receive from all active rovers' servo telemetry
            result = receive_from_rovers(&active_rovers, |subs| &subs.servo_telemetry_sub) => {
                if let Some((entity_id, sample)) = result {
                    forward_telemetry_with_entity_id(
                        &mut node,
                        &servo_telemetry_output,
                        entity_id,
                        sample
                    );
                }
            }

            // Receive from all active rovers' tracked detections
            result = receive_from_rovers(&active_rovers, |subs| &subs.tracked_detections_sub) => {
                if let Some((entity_id, sample)) = result {
                    forward_telemetry_with_entity_id(
                        &mut node,
                        &tracked_detections_output,
                        entity_id,
                        sample
                    );
                }
            }

            // Receive from all active rovers' tracking telemetry
            result = receive_from_rovers(&active_rovers, |subs| &subs.tracking_telemetry_sub) => {
                if let Some((entity_id, sample)) = result {
                    forward_telemetry_with_entity_id(
                        &mut node,
                        &tracking_telemetry_output,
                        entity_id,
                        sample
                    );
                }
            }

            // Receive from all active rovers' metrics
            result = receive_from_rovers(&active_rovers, |subs| &subs.metrics_sub) => {
                if let Some((entity_id, sample)) = result {
                    forward_telemetry_with_entity_id(
                        &mut node,
                        &performance_metrics_output,
                        entity_id,
                        sample
                    );
                }
            }
        }
    }

    Ok(())
}

/// Helper function to receive from any rover's specific subscriber
async fn receive_from_rovers<'a, F>(
    active_rovers: &'a HashMap<String, RoverSubscriptions>,
    get_sub: F,
) -> Option<(String, zenoh::sample::Sample)>
where
    F: Fn(&'a RoverSubscriptions) -> &'a ZenohSubscriber,
{
    if active_rovers.is_empty() {
        // No active rovers, sleep briefly
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        return None;
    }

    // Create pinned futures for all active rovers
    let mut futures = Vec::new();
    for (entity_id, subs) in active_rovers.iter() {
        let sub = get_sub(subs);
        let entity_id = entity_id.clone();
        let fut = async move {
            let result = sub.recv_async().await;
            (entity_id, result)
        };
        futures.push(Box::pin(fut));
    }

    // Use select_all to wait for first completion
    if futures.is_empty() {
        return None;
    }

    let (result, _index, _remaining) = futures::future::select_all(futures).await;

    match result.1 {
        Ok(sample) => Some((result.0, sample)),
        Err(e) => {
            tracing::error!("Receive error from {}: {}", result.0, e);
            None
        }
    }
}

/// Forward telemetry data with entity_id tag
fn forward_telemetry_with_entity_id(
    node: &mut DoraNode,
    output_id: &DataId,
    entity_id: String,
    sample: zenoh::sample::Sample,
) {
    let payload = sample.payload().to_bytes();

    // Deserialize as JSON, inject entity_id, re-serialize
    if let Ok(mut telemetry_json) = serde_json::from_slice::<serde_json::Value>(&payload) {
        // Add entity_id field to the telemetry JSON
        if let Some(obj) = telemetry_json.as_object_mut() {
            obj.insert("entity_id".to_string(), serde_json::Value::String(entity_id));
        }

        // Re-serialize with entity_id included
        if let Ok(serialized) = serde_json::to_vec(&telemetry_json) {
            let arrow_data = BinaryArray::from_vec(vec![serialized.as_slice()]);
            let _ = node.send_output(output_id.clone(), Default::default(), arrow_data);
        }
    }
}
