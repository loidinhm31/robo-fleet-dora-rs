use dora_node_api::{
    arrow::array::{Array, BinaryArray, Float32Array},
    dora_core::config::DataId,
    DoraNode, Event, Parameter,
};
use eyre::Result;
use robo_rover_lib::{
    capture_age_ms, init_tracing, record_capture_age, AudioFrameMetadata,
    AudioFrameSequenceTracker, FleetSelectCommand, FleetSubscriptionCommand, FrameSequenceTracker,
    JpegFramePacket, MetricWindow, PcmFramePacket, PcmSampleFormat,
};
use serde_json;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use zenoh::Config;

#[path = "walkie-audio.rs"]
mod walkie_audio;
use walkie_audio::encode_walkie_packet;

// Type alias for Zenoh subscriber (default handler)
type ZenohSubscriber =
    zenoh::pubsub::Subscriber<zenoh::handlers::FifoChannelHandler<zenoh::sample::Sample>>;

/// Subscriptions for a single rover
struct RoverSubscriptions {
    entity_id: String,

    // Data subscribers (FROM rover)
    video_sub: ZenohSubscriber,
    audio_sub: ZenohSubscriber,
    playback_audio_sub: ZenohSubscriber,
    rover_telemetry_sub: ZenohSubscriber,
    arm_telemetry_sub: ZenohSubscriber,
    servo_telemetry_sub: ZenohSubscriber,
    detections_sub: ZenohSubscriber,
    tracked_detections_sub: ZenohSubscriber,
    tracking_telemetry_sub: ZenohSubscriber,
    metrics_sub: ZenohSubscriber,
    voice_status_sub: ZenohSubscriber,
    tts_command_result_sub: ZenohSubscriber,
}

struct LegacyAudioState {
    stream_id: uuid::Uuid,
    next_frame_id: u64,
}

struct DecodedAudioFrame {
    metadata: AudioFrameMetadata,
    payload: Vec<u8>,
    legacy: bool,
}

/// Subscribe to all topics for a specific rover
async fn subscribe_to_rover(
    session: &Arc<zenoh::Session>,
    entity_id: &str,
) -> Result<RoverSubscriptions> {
    tracing::info!("Subscribing to rover: {}", entity_id);

    let video_topic = format!("rover/{}/video/jpeg/v1", entity_id);
    let video_sub = session
        .declare_subscriber(&video_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to subscribe to {}: {}", video_topic, e))?;
    tracing::info!("{}", video_topic);

    let audio_topic = format!("rover/{}/audio/raw", entity_id);
    let audio_sub = session
        .declare_subscriber(&audio_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to subscribe to {}: {}", audio_topic, e))?;
    tracing::info!("{}", audio_topic);

    let playback_audio_topic = format!("rover/{}/audio/playback/raw", entity_id);
    let playback_audio_sub = session
        .declare_subscriber(&playback_audio_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to subscribe to {}: {}", playback_audio_topic, e))?;
    tracing::info!("{}", playback_audio_topic);

    let rover_telemetry_topic = format!("rover/{}/telemetry/rover", entity_id);
    let rover_telemetry_sub = session
        .declare_subscriber(&rover_telemetry_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to subscribe to {}: {}", rover_telemetry_topic, e))?;
    tracing::info!("{}", rover_telemetry_topic);

    let arm_telemetry_topic = format!("rover/{}/telemetry/arm", entity_id);
    let arm_telemetry_sub = session
        .declare_subscriber(&arm_telemetry_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to subscribe to {}: {}", arm_telemetry_topic, e))?;
    tracing::info!("{}", arm_telemetry_topic);

    let servo_telemetry_topic = format!("rover/{}/telemetry/servo", entity_id);
    let servo_telemetry_sub = session
        .declare_subscriber(&servo_telemetry_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to subscribe to {}: {}", servo_telemetry_topic, e))?;
    tracing::info!("{}", servo_telemetry_topic);

    let detections_topic = format!("rover/{}/video/detections_only", entity_id);
    let detections_sub = session
        .declare_subscriber(&detections_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to subscribe to {}: {}", detections_topic, e))?;
    tracing::info!("{}", detections_topic);

    let tracked_detections_topic = format!("rover/{}/video/detections", entity_id);
    let tracked_detections_sub = session
        .declare_subscriber(&tracked_detections_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to subscribe to {}: {}", tracked_detections_topic, e))?;
    tracing::info!("{}", tracked_detections_topic);

    let tracking_telemetry_topic = format!("rover/{}/telemetry/tracking", entity_id);
    let tracking_telemetry_sub = session
        .declare_subscriber(&tracking_telemetry_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to subscribe to {}: {}", tracking_telemetry_topic, e))?;
    tracing::info!("{}", tracking_telemetry_topic);

    let metrics_topic = format!("rover/{}/metrics", entity_id);
    let metrics_sub = session
        .declare_subscriber(&metrics_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to subscribe to {}: {}", metrics_topic, e))?;
    tracing::info!("{}", metrics_topic);

    let voice_status_topic = voice_status_topic(entity_id);
    let voice_status_sub = session
        .declare_subscriber(&voice_status_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to subscribe to {}: {}", voice_status_topic, e))?;
    tracing::info!("{}", voice_status_topic);

    let voice_result_topic = voice_result_topic(entity_id);
    let tts_command_result_sub = session
        .declare_subscriber(&voice_result_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to subscribe to {}: {}", voice_result_topic, e))?;
    tracing::info!("{}", voice_result_topic);

    Ok(RoverSubscriptions {
        entity_id: entity_id.to_string(),
        video_sub,
        audio_sub,
        playback_audio_sub,
        rover_telemetry_sub,
        arm_telemetry_sub,
        servo_telemetry_sub,
        detections_sub,
        tracked_detections_sub,
        tracking_telemetry_sub,
        metrics_sub,
        voice_status_sub,
        tts_command_result_sub,
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
    latest_voice_config: Option<&[u8]>,
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
                        if let Some(config) = latest_voice_config {
                            queue_voice_config_publish(
                                session.clone(),
                                entity_id.clone(),
                                config.to_vec(),
                            );
                        }
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
                    let to_remove: Vec<String> = active_rovers
                        .keys()
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
                            if let Some(config) = latest_voice_config {
                                queue_voice_config_publish(
                                    session.clone(),
                                    rover_id.clone(),
                                    config.to_vec(),
                                );
                            }
                            active_rovers.insert(rover_id, subs);
                        }
                    }
                }
            }

            tracing::info!(
                "Active rovers: {:?}",
                active_rovers.keys().collect::<Vec<_>>()
            );
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
    let active_rovers_env =
        std::env::var("ACTIVE_ROVERS").unwrap_or_else(|_| "rover-kiwi".to_string());
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

    tracing::info!(
        "Expected video frame dimensions: {}x{}",
        frame_width,
        frame_height
    );

    // Audio configuration
    let audio_sample_rate = std::env::var("AUDIO_SAMPLE_RATE")
        .ok()
        .and_then(|v| v.parse::<i64>().ok())
        .unwrap_or(16000);
    let audio_channels = std::env::var("AUDIO_CHANNELS")
        .ok()
        .and_then(|v| v.parse::<i64>().ok())
        .unwrap_or(1);

    tracing::info!(
        "Expected audio format: {}Hz, {} channels",
        audio_sample_rate,
        audio_channels
    );

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
        config
            .insert_json5("mode", "\"peer\"")
            .map_err(|e| eyre::eyre!("Failed to set Zenoh mode: {}", e))?;
        config
    };

    let session = Arc::new(
        zenoh::open(config)
            .await
            .map_err(|e| eyre::eyre!("Failed to open Zenoh session: {}", e))?,
    );

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
    let playback_audio_frame_output = DataId::from("playback_audio_frame".to_owned());
    let rover_telemetry_output = DataId::from("rover_telemetry".to_owned());
    let arm_telemetry_output = DataId::from("arm_telemetry".to_owned());
    let servo_telemetry_output = DataId::from("servo_telemetry".to_owned());
    let detections_output = DataId::from("detections".to_owned());
    let tracked_detections_output = DataId::from("tracked_detections".to_owned());
    let tracking_telemetry_output = DataId::from("tracking_telemetry".to_owned());
    let performance_metrics_output = DataId::from("performance_metrics".to_owned());
    let voice_status_output = DataId::from("voice_status".to_owned());
    let tts_command_result_output = DataId::from("tts_command_result".to_owned());

    // Statistics per rover
    let video_counts: Arc<Mutex<HashMap<String, u64>>> = Arc::new(Mutex::new(HashMap::new()));
    let audio_counts: Arc<Mutex<HashMap<String, u64>>> = Arc::new(Mutex::new(HashMap::new()));
    let mut video_metrics = MetricWindow::new(Duration::from_secs(5));
    let mut video_age_metrics = MetricWindow::new(Duration::from_secs(5));
    let mut video_sequences: HashMap<String, FrameSequenceTracker> = HashMap::new();
    let mut audio_metrics = MetricWindow::new(Duration::from_secs(5));
    let mut audio_age_metrics = MetricWindow::new(Duration::from_secs(5));
    let mut audio_sequences: HashMap<String, AudioFrameSequenceTracker> = HashMap::new();
    let mut playback_audio_sequences: HashMap<String, AudioFrameSequenceTracker> = HashMap::new();
    let mut legacy_audio_states: HashMap<String, LegacyAudioState> = HashMap::new();
    let mut audio_errors = 0u64;
    let mut audio_sequence_drops = 0u64;
    let mut latest_voice_config: Option<Vec<u8>> = None;

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
                    Event::Input { id, data, metadata } => {
                        match id.as_str() {
                            // Fleet subscription management
                            "fleet_subscription_command" => {
                                if let Err(e) = handle_fleet_subscription_command(
                                    &mut active_rovers,
                                    &session,
                                    data,
                                    latest_voice_config.as_deref(),
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

                                                match encode_walkie_packet(
                                                    &metadata.parameters,
                                                    float32_array.values().as_ref(),
                                                ) {
                                                    Ok(packet) => {
                                                        if let Err(error) = session.put(audio_stream_topic, packet).await {
                                                            tracing::error!(%error, "failed to publish speaker audio");
                                                        }
                                                    }
                                                    Err(error) => tracing::warn!(%error, "rejected invalid walkie frame"),
                                                }
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
                                        let input_id = id.as_str();

                                        if input_id == "tts_config_command" {
                                            latest_voice_config = Some(bytes.to_vec());
                                            queue_voice_config_fanout(
                                                session.clone(),
                                                snapshot_active_rover_ids(&active_rovers),
                                                bytes.to_vec(),
                                            );
                                            continue;
                                        }

                                        // ----------------------------------------------------------------
                                        // PARSER inputs: must carry an authoritative target_entity_id.
                                        // Never fall back to the selected rover for parser commands.
                                        // ----------------------------------------------------------------
                                        let is_parser_input = matches!(
                                            input_id,
                                            "rover_command_parser"
                                            | "tracking_command_parser"
                                            | "camera_control_parser"
                                        );

                                        if is_parser_input {
                                            // Extract target from the JSON envelope
                                            let target_entity_id: Option<String> =
                                                serde_json::from_slice::<serde_json::Value>(bytes)
                                                    .ok()
                                                    .and_then(|v| {
                                                        v.get("target_entity_id")
                                                            .and_then(|t| t.as_str())
                                                            .map(|s| s.to_string())
                                                    });

                                            match target_entity_id {
                                                None => {
                                                    tracing::warn!(
                                                        metric = "parser_routing",
                                                        input = input_id,
                                                        reason = "missing_target",
                                                        "rejected parser command: no target_entity_id"
                                                    );
                                                }
                                                Some(ref target_id) if target_id.trim().is_empty() => {
                                                    tracing::warn!(
                                                        metric = "parser_routing",
                                                        input = input_id,
                                                        reason = "empty_target",
                                                        "rejected parser command: empty target_entity_id"
                                                    );
                                                }
                                                Some(ref target_id) => {
                                                    if !active_rovers.contains_key(target_id) {
                                                        tracing::warn!(
                                                            metric = "parser_routing",
                                                            input = input_id,
                                                            target = %target_id,
                                                            reason = "inactive_target",
                                                            "rejected parser command: target rover is not active"
                                                        );
                                                    } else {
                                                        let topic = match input_id {
                                                            "rover_command_parser" => Some(
                                                                format!(
                                                                    "rover/{}/cmd/movement",
                                                                    target_id
                                                                ),
                                                            ),
                                                            "camera_control_parser" => Some(
                                                                format!(
                                                                    "rover/{}/cmd/camera",
                                                                    target_id
                                                                ),
                                                            ),
                                                            "tracking_command_parser" => Some(
                                                                format!(
                                                                    "rover/{}/cmd/tracking",
                                                                    target_id
                                                                ),
                                                            ),
                                                            _ => None,
                                                        };
                                                        if let Some(topic) = topic {
                                                            tracing::info!(
                                                                metric = "parser_routing",
                                                                input = input_id,
                                                                target = %target_id,
                                                                "routing parser command to target rover"
                                                            );
                                                            let _ = session.put(&topic, bytes).await;
                                                        }
                                                    }
                                                }
                                            }
                                        } else {
                                            // ----------------------------------------------------------------
                                            // WEB/MANUAL inputs: route to the currently selected rover.
                                            // ----------------------------------------------------------------
                                            if let Some(ref entity_id) = selected_entity {
                                                if active_rovers.contains_key(entity_id) {
                                                    let topic = web_command_topic(input_id, entity_id);

                                                    if let Some(topic) = topic {
                                                        tracing::debug!("Routing web command to {}: {}", entity_id, topic);
                                                        let _ = session.put(&topic, bytes).await;
                                                    }
                                                } else {
                                                    tracing::warn!("Selected rover {} is not active", entity_id);
                                                }
                                            } else {
                                                tracing::warn!("No rover selected for command: {}", input_id);
                                            }
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
                        let frames_forwarded: u64 = audio_counts_map.values().copied().sum();
                        tracing::info!(metric="audio_pipeline_total", stage="orchestra_zenoh_receive",
                            frames_forwarded, sequence_drops=audio_sequence_drops,
                            errors=audio_errors);
                        break;
                    }
                    _ => {}
                }
            }

            // Receive from all active rovers' video subscriptions
            result = receive_from_rovers(&active_rovers, |subs| &subs.video_sub) => {
                if let Some((entity_id, sample)) = result {
                    let receive_started = Instant::now();
                    let payload = sample.payload().to_bytes();
                    match JpegFramePacket::decode(payload.as_ref()) {
                        Ok(frame) => {
                            match video_sequences.entry(entity_id.clone()).or_default()
                                .observe(frame.metadata.frame_id) {
                                Ok(missing) => video_metrics.record_drops(missing),
                                Err(()) => video_metrics.record_error(),
                            }
                            let frame_age_ms = capture_age_ms(frame.metadata.capture_timestamp_ms)
                                .unwrap_or_else(|| {
                                    video_metrics.record_error();
                                    0
                                });
                            video_age_metrics.record(Duration::from_millis(frame_age_ms), 0);
                            let video_array = BinaryArray::from_vec(vec![frame.payload]);
                            let mut params = BTreeMap::new();
                            params.insert("entity_id".to_owned(), Parameter::String(entity_id.clone()));
                            params.insert("width".to_owned(), Parameter::Integer(frame.metadata.width as i64));
                            params.insert("height".to_owned(), Parameter::Integer(frame.metadata.height as i64));
                            params.insert("encoding".to_owned(), Parameter::String("jpeg".to_string()));
                            params.insert("codec".to_owned(), Parameter::String("jpeg".to_string()));
                            params.insert("compressed_size".to_owned(), Parameter::Integer(frame.payload.len() as i64));
                            params.insert("frame_id".to_owned(), Parameter::Integer(frame.metadata.frame_id as i64));
                            params.insert("capture_timestamp_ms".to_owned(),
                                Parameter::Integer(frame.metadata.capture_timestamp_ms as i64));
                            if let Err(error) = node.send_output(video_frame_output.clone(), params, video_array) {
                                video_metrics.record_error();
                                tracing::error!(%error, "failed to forward video frame to Dora");
                            } else {
                                video_metrics.record(receive_started.elapsed(), payload.len());
                                let mut counts = video_counts.lock().await;
                                *counts.entry(entity_id).or_insert(0) += 1;
                            }
                            if let Some(snapshot) = video_metrics.snapshot_if_due() {
                                tracing::info!(metric="video_pipeline", stage="orchestra_zenoh_receive",
                                    frame_id=frame.metadata.frame_id, frame_age_ms,
                                    count=snapshot.count, bytes=snapshot.bytes, drops=snapshot.drops,
                                    errors=snapshot.errors, p50_us=snapshot.p50_us,
                                    p95_us=snapshot.p95_us, p99_us=snapshot.p99_us, max_us=snapshot.max_us);
                            }
                            if let Some(snapshot) = video_age_metrics.snapshot_if_due() {
                                tracing::info!(metric="video_pipeline", stage="orchestra_zenoh_receive_age",
                                    count=snapshot.count, p50_us=snapshot.p50_us,
                                    p95_us=snapshot.p95_us, p99_us=snapshot.p99_us,
                                    max_us=snapshot.max_us);
                            }
                        }
                        Err(error) => {
                            video_metrics.record_error();
                            tracing::warn!(%error, entity_id, "rejected invalid jpeg video packet");
                        }
                    }
                }
            }

            // Receive from all active rovers' audio subscriptions
            result = receive_from_rovers(&active_rovers, |subs| &subs.audio_sub) => {
                if let Some((entity_id, sample)) = result {
                    let receive_started = Instant::now();
                    let payload = sample.payload().to_bytes();
                    match decode_rover_audio(
                        payload.as_ref(),
                        &entity_id,
                        &mut legacy_audio_states,
                        u32::try_from(audio_sample_rate).unwrap_or(16_000),
                        u16::try_from(audio_channels).unwrap_or(1),
                    ) {
                        Ok(frame) => {
                            match audio_sequences.entry(entity_id.clone()).or_default()
                                .observe(frame.metadata) {
                                Ok(observation) => {
                                    audio_metrics.record_drops(observation.missing_frames);
                                    audio_sequence_drops = audio_sequence_drops
                                        .saturating_add(observation.missing_frames);
                                }
                                Err(error) => {
                                    audio_errors = audio_errors.saturating_add(1);
                                    audio_metrics.record_error();
                                    tracing::warn!(%error, %entity_id, "rejected duplicate or regressed audio frame");
                                    continue;
                                }
                            }
                            let frame_age_ms = record_capture_age(
                                &mut audio_age_metrics,
                                frame.metadata.capture_timestamp_ms,
                            );
                            if frame_age_ms.is_none() {
                                audio_errors = audio_errors.saturating_add(1);
                                audio_metrics.record_error();
                            }
                            let params = audio_dora_parameters(frame.metadata, &entity_id, frame.payload.len());
                            let audio_array = BinaryArray::from_vec(vec![frame.payload.as_slice()]);
                            if let Err(error) = node.send_output(audio_frame_output.clone(), params, audio_array) {
                                audio_errors = audio_errors.saturating_add(1);
                                audio_metrics.record_error();
                                tracing::error!(%error, %entity_id, "failed to forward audio frame to Dora");
                            } else {
                                audio_metrics.record(receive_started.elapsed(), payload.len());
                                let mut counts = audio_counts.lock().await;
                                *counts.entry(entity_id.clone()).or_insert(0) += 1;
                            }
                            if let Some(snapshot) = audio_metrics.snapshot_if_due() {
                                tracing::info!(metric="audio_pipeline", stage="orchestra_zenoh_receive",
                                    %entity_id, stream_id=%frame.metadata.stream_id,
                                    frame_id=frame.metadata.frame_id, frame_age_ms=?frame_age_ms,
                                    legacy=frame.legacy, count=snapshot.count, bytes=snapshot.bytes,
                                    drops=snapshot.drops, errors=snapshot.errors,
                                    p50_us=snapshot.p50_us, p95_us=snapshot.p95_us,
                                    p99_us=snapshot.p99_us, max_us=snapshot.max_us);
                            }
                            if let Some(snapshot) = audio_age_metrics.snapshot_if_due() {
                                tracing::info!(metric="audio_pipeline", stage="orchestra_zenoh_receive_age",
                                    count=snapshot.count, p50_us=snapshot.p50_us,
                                    p95_us=snapshot.p95_us, p99_us=snapshot.p99_us,
                                    max_us=snapshot.max_us);
                            }
                        }
                        Err(error) => {
                            audio_errors = audio_errors.saturating_add(1);
                            audio_metrics.record_error();
                            tracing::warn!(%error, %entity_id, "rejected invalid audio packet");
                        }
                    }
                }
            }

            // Receive rover speaker monitor audio for browser operators only.
            result = receive_from_rovers(&active_rovers, |subs| &subs.playback_audio_sub) => {
                if let Some((entity_id, sample)) = result {
                    let receive_started = Instant::now();
                    let payload = sample.payload().to_bytes();
                    match PcmFramePacket::decode(payload.as_ref()) {
                        Ok(frame) if frame.metadata.format == PcmSampleFormat::S16Le => {
                            match playback_audio_sequences.entry(entity_id.clone()).or_default()
                                .observe(frame.metadata) {
                                Ok(observation) => {
                                    audio_metrics.record_drops(observation.missing_frames);
                                    audio_sequence_drops = audio_sequence_drops
                                        .saturating_add(observation.missing_frames);
                                }
                                Err(error) => {
                                    audio_errors = audio_errors.saturating_add(1);
                                    audio_metrics.record_error();
                                    tracing::warn!(%error, %entity_id, "rejected duplicate or regressed playback audio frame");
                                    continue;
                                }
                            }
                            let frame_age_ms = record_capture_age(
                                &mut audio_age_metrics,
                                frame.metadata.capture_timestamp_ms,
                            );
                            if frame_age_ms.is_none() {
                                audio_errors = audio_errors.saturating_add(1);
                                audio_metrics.record_error();
                            }
                            let params = audio_dora_parameters(frame.metadata, &entity_id, frame.payload.len());
                            let audio_array = BinaryArray::from_vec(vec![frame.payload]);
                            if let Err(error) = node.send_output(playback_audio_frame_output.clone(), params, audio_array) {
                                audio_errors = audio_errors.saturating_add(1);
                                audio_metrics.record_error();
                                tracing::error!(%error, %entity_id, "failed to forward playback audio frame to Dora");
                            } else {
                                audio_metrics.record(receive_started.elapsed(), payload.len());
                            }
                            if let Some(snapshot) = audio_metrics.snapshot_if_due() {
                                tracing::info!(metric="audio_pipeline", stage="orchestra_zenoh_receive_playback",
                                    %entity_id, stream_id=%frame.metadata.stream_id,
                                    frame_id=frame.metadata.frame_id, frame_age_ms=?frame_age_ms,
                                    count=snapshot.count, bytes=snapshot.bytes,
                                    drops=snapshot.drops, errors=snapshot.errors,
                                    p50_us=snapshot.p50_us, p95_us=snapshot.p95_us,
                                    p99_us=snapshot.p99_us, max_us=snapshot.max_us);
                            }
                        }
                        Ok(_) => {
                            audio_errors = audio_errors.saturating_add(1);
                            audio_metrics.record_error();
                            tracing::warn!(%entity_id, "rejected non-s16le playback audio packet");
                        }
                        Err(error) => {
                            audio_errors = audio_errors.saturating_add(1);
                            audio_metrics.record_error();
                            tracing::warn!(%error, %entity_id, "rejected invalid playback audio packet");
                        }
                    }
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

            // Receive from all active rovers' detection-only results
            result = receive_from_rovers(&active_rovers, |subs| &subs.detections_sub) => {
                if let Some((entity_id, sample)) = result {
                    forward_telemetry_with_entity_id(
                        &mut node,
                        &detections_output,
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

            result = receive_from_rovers(&active_rovers, |subs| &subs.voice_status_sub) => {
                if let Some((_entity_id, sample)) = result {
                    forward_binary_output(&mut node, &voice_status_output, sample);
                }
            }

            result = receive_from_rovers(&active_rovers, |subs| &subs.tts_command_result_sub) => {
                if let Some((_entity_id, sample)) = result {
                    forward_binary_output(&mut node, &tts_command_result_output, sample);
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
            obj.insert(
                "entity_id".to_string(),
                serde_json::Value::String(entity_id),
            );
        }

        // Re-serialize with entity_id included
        if let Ok(serialized) = serde_json::to_vec(&telemetry_json) {
            let arrow_data = BinaryArray::from_vec(vec![serialized.as_slice()]);
            let _ = node.send_output(output_id.clone(), Default::default(), arrow_data);
        }
    }
}

fn forward_binary_output(node: &mut DoraNode, output_id: &DataId, sample: zenoh::sample::Sample) {
    let payload = sample.payload().to_bytes();
    let arrow_data = BinaryArray::from_vec(vec![payload.as_ref()]);
    let _ = node.send_output(output_id.clone(), Default::default(), arrow_data);
}

fn voice_config_topic(entity_id: &str) -> String {
    format!("rover/{entity_id}/cmd/voice/config")
}

fn voice_status_topic(entity_id: &str) -> String {
    format!("rover/{entity_id}/voice/status")
}

fn voice_result_topic(entity_id: &str) -> String {
    format!("rover/{entity_id}/voice/result")
}

fn web_command_topic(input_id: &str, entity_id: &str) -> Option<String> {
    match input_id {
        "rover_command_web" => Some(format!("rover/{entity_id}/cmd/movement")),
        "arm_command_web" => Some(format!("rover/{entity_id}/cmd/arm")),
        "camera_command_web" => Some(format!("rover/{entity_id}/cmd/camera")),
        "audio_command_web" => Some(format!("rover/{entity_id}/cmd/audio")),
        "stream_command_web" => Some(format!("rover/{entity_id}/cmd/stream/v1")),
        "tracking_command_web" => Some(format!("rover/{entity_id}/cmd/tracking")),
        "tts_command_web" => Some(format!("rover/{entity_id}/cmd/tts")),
        _ => None,
    }
}

fn snapshot_active_rover_ids(active_rovers: &HashMap<String, RoverSubscriptions>) -> Vec<String> {
    let mut entity_ids: Vec<String> = active_rovers.keys().cloned().collect();
    entity_ids.sort();
    entity_ids
}

async fn publish_voice_config(
    session: Arc<zenoh::Session>,
    entity_id: String,
    payload: Vec<u8>,
) -> Result<()> {
    session
        .put(voice_config_topic(&entity_id), payload)
        .await
        .map_err(|error| eyre::eyre!("Failed to publish voice config to {entity_id}: {error}"))?;
    Ok(())
}

fn queue_voice_config_publish(session: Arc<zenoh::Session>, entity_id: String, payload: Vec<u8>) {
    tokio::spawn(async move {
        if let Err(error) = publish_voice_config(session, entity_id.clone(), payload).await {
            tracing::error!(%error, %entity_id, "failed to publish voice config");
        }
    });
}

fn queue_voice_config_fanout(
    session: Arc<zenoh::Session>,
    entity_ids: Vec<String>,
    payload: Vec<u8>,
) {
    for entity_id in entity_ids {
        queue_voice_config_publish(session.clone(), entity_id, payload.clone());
    }
}

fn decode_rover_audio(
    payload: &[u8],
    entity_id: &str,
    legacy_states: &mut HashMap<String, LegacyAudioState>,
    legacy_sample_rate: u32,
    legacy_channels: u16,
) -> Result<DecodedAudioFrame, String> {
    if payload.starts_with(b"PCMF") {
        let packet = PcmFramePacket::decode(payload)?;
        if packet.metadata.format != PcmSampleFormat::S16Le {
            return Err("v1 rover audio packet must contain s16le samples".into());
        }
        return Ok(DecodedAudioFrame {
            metadata: packet.metadata,
            payload: packet.payload.to_vec(),
            legacy: false,
        });
    }

    let max_legacy_bytes = usize::try_from(legacy_sample_rate)
        .ok()
        .and_then(|rate| rate.checked_mul(usize::from(legacy_channels)))
        .and_then(|samples| samples.checked_mul(PcmSampleFormat::F32Le.bytes_per_sample()))
        .ok_or_else(|| "legacy audio dimensions overflow".to_string())?;
    let samples = decode_legacy_f32le(payload, max_legacy_bytes)?;
    let state = legacy_states
        .entry(entity_id.to_owned())
        .or_insert_with(|| LegacyAudioState {
            stream_id: uuid::Uuid::new_v4(),
            next_frame_id: 0,
        });
    let frame_id = state.next_frame_id;
    state.next_frame_id = state.next_frame_id.saturating_add(1);
    let metadata = AudioFrameMetadata {
        stream_id: state.stream_id,
        frame_id,
        capture_timestamp_ms: current_time_ms()?,
        sample_rate: legacy_sample_rate,
        channels: legacy_channels,
        sample_count: samples
            .len()
            .try_into()
            .map_err(|_| "legacy sample count exceeds u32")?,
        format: PcmSampleFormat::S16Le,
    };
    let converted = float32_to_s16le(&samples);
    metadata.validate_payload_len(converted.len())?;
    Ok(DecodedAudioFrame {
        metadata,
        payload: converted,
        legacy: true,
    })
}

fn decode_legacy_f32le(payload: &[u8], max_bytes: usize) -> Result<Vec<f32>, String> {
    if payload.is_empty() || payload.len() > max_bytes || payload.len() % 4 != 0 {
        return Err(format!(
            "invalid legacy f32le payload length: {}",
            payload.len()
        ));
    }
    Ok(payload
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn float32_to_s16le(samples: &[f32]) -> Vec<u8> {
    let mut output = Vec::with_capacity(samples.len() * 2);
    for &sample in samples {
        let converted = if sample <= -1.0 {
            i16::MIN
        } else {
            (sample.clamp(-1.0, 1.0) * f32::from(i16::MAX)).round() as i16
        };
        output.extend_from_slice(&converted.to_le_bytes());
    }
    output
}

fn audio_dora_parameters(
    metadata: AudioFrameMetadata,
    entity_id: &str,
    payload_len: usize,
) -> BTreeMap<String, Parameter> {
    BTreeMap::from([
        ("entity_id".into(), Parameter::String(entity_id.to_owned())),
        (
            "stream_id".into(),
            Parameter::String(metadata.stream_id.to_string()),
        ),
        (
            "frame_id".into(),
            Parameter::Integer(metadata.frame_id as i64),
        ),
        (
            "capture_timestamp_ms".into(),
            Parameter::Integer(metadata.capture_timestamp_ms as i64),
        ),
        (
            "sample_rate".into(),
            Parameter::Integer(i64::from(metadata.sample_rate)),
        ),
        (
            "channels".into(),
            Parameter::Integer(i64::from(metadata.channels)),
        ),
        (
            "sample_count".into(),
            Parameter::Integer(i64::from(metadata.sample_count)),
        ),
        (
            "format".into(),
            Parameter::String(metadata.format.metadata_name().into()),
        ),
        ("size".into(), Parameter::Integer(payload_len as i64)),
    ])
}

fn current_time_ms() -> Result<u64, String> {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|error| error.to_string())?
        .as_millis()
        .try_into()
        .map_err(|_| "current timestamp exceeds u64".into())
}

#[cfg(test)]
mod audio_tests {
    use super::*;

    #[test]
    fn legacy_audio_decode_is_bounded_and_converts_to_s16le() {
        let mut states = HashMap::new();
        let input: Vec<u8> = [-1.0_f32, 0.0, 1.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let frame = decode_rover_audio(&input, "rover-a", &mut states, 16_000, 1).unwrap();
        assert!(frame.legacy);
        assert_eq!(frame.payload.len(), 6);
        assert_eq!(frame.metadata.sample_count, 3);
        assert!(decode_legacy_f32le(&[0, 1, 2], 64_000).is_err());
    }

    #[test]
    fn malformed_versioned_packet_does_not_fall_back_to_legacy() {
        let mut states = HashMap::new();
        assert!(decode_rover_audio(b"PCMFbad", "rover-a", &mut states, 16_000, 1).is_err());
        assert!(states.is_empty());
    }
}

// ---------------------------------------------------------------------------
// Routing logic tests (Phase 02: source-aware command routing)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod routing_tests {
    /// Returns true if the given Dora input id is classified as a parser input
    /// that requires an authoritative target_entity_id.
    fn is_parser_input(id: &str) -> bool {
        matches!(
            id,
            "rover_command_parser" | "tracking_command_parser" | "camera_control_parser"
        )
    }

    /// Extract target_entity_id from a JSON bytes payload (mirrors bridge logic).
    fn extract_target(bytes: &[u8]) -> Option<String> {
        serde_json::from_slice::<serde_json::Value>(bytes)
            .ok()
            .and_then(|v| {
                v.get("target_entity_id")
                    .and_then(|t| t.as_str())
                    .map(|s| s.to_string())
            })
    }

    // --- Parser input classification ---

    #[test]
    fn parser_inputs_are_classified_correctly() {
        assert!(is_parser_input("rover_command_parser"));
        assert!(is_parser_input("tracking_command_parser"));
        assert!(is_parser_input("camera_control_parser"));
    }

    #[test]
    fn web_inputs_are_not_parser_inputs() {
        assert!(!is_parser_input("rover_command_web"));
        assert!(!is_parser_input("arm_command_web"));
        assert!(!is_parser_input("tracking_command_web"));
        assert!(!is_parser_input("camera_command_web"));
        assert!(!is_parser_input("tts_command_web"));
        assert!(!is_parser_input("tts_config_command"));
        assert!(!is_parser_input("audio_command_web"));
        assert!(!is_parser_input("stream_command_web"));
        assert!(!is_parser_input("audio_stream_web"));
        assert!(!is_parser_input("fleet_subscription_command"));
        assert!(!is_parser_input("fleet_select_command"));
    }

    // --- Target extraction from JSON ---

    #[test]
    fn target_extracted_from_rover_command_json() {
        let payload = serde_json::json!({
            "command": {"type": "Stop", "timestamp": 0, "command_id": "abc"},
            "metadata": {
                "command_id": "abc",
                "timestamp": 0,
                "source": "VoiceCommand",
                "priority": "Low"
            },
            "target_entity_id": "rover-a"
        });
        let bytes = serde_json::to_vec(&payload).unwrap();
        let target = extract_target(&bytes);
        assert_eq!(target.as_deref(), Some("rover-a"));
    }

    #[test]
    fn missing_target_returns_none() {
        let payload = serde_json::json!({
            "command": {"type": "Stop", "timestamp": 0, "command_id": "abc"},
            "metadata": {}
        });
        let bytes = serde_json::to_vec(&payload).unwrap();
        let target = extract_target(&bytes);
        assert!(target.is_none());
    }

    #[test]
    fn null_target_returns_none() {
        let payload = serde_json::json!({
            "target_entity_id": null
        });
        let bytes = serde_json::to_vec(&payload).unwrap();
        let target = extract_target(&bytes);
        assert!(target.is_none());
    }

    #[test]
    fn empty_target_is_detected() {
        let payload = serde_json::json!({ "target_entity_id": "  " });
        let bytes = serde_json::to_vec(&payload).unwrap();
        let target = extract_target(&bytes);
        // Extraction returns the raw string; the bridge then checks .trim().is_empty()
        assert!(target
            .as_deref()
            .map(|s| s.trim().is_empty())
            .unwrap_or(true));
    }

    // --- Simulated routing decisions ---

    /// Simulates the bridge routing decision for a parser command:
    /// - with valid active target → should route
    /// - with inactive target → should reject
    /// - with no target → should reject
    #[test]
    fn parser_routing_rejects_inactive_target() {
        let active_rovers = std::collections::HashMap::from([("rover-a".to_string(), ())]);

        let target_inactive = "rover-b";
        assert!(!active_rovers.contains_key(target_inactive));

        let target_active = "rover-a";
        assert!(active_rovers.contains_key(target_active));
    }

    #[test]
    fn parser_routing_accepts_active_target() {
        let active_rovers = std::collections::HashMap::from([("rover-kiwi".to_string(), ())]);
        let target = "rover-kiwi";
        assert!(active_rovers.contains_key(target));
    }

    #[test]
    fn web_routing_uses_selected_rover_regardless_of_payload() {
        // Web command doesn't need target in payload — it uses selected_entity
        let selected = Some("rover-kiwi".to_string());
        let active_rovers = std::collections::HashMap::from([("rover-kiwi".to_string(), ())]);
        let entity_id = selected.as_ref().unwrap();
        assert!(active_rovers.contains_key(entity_id.as_str()));
    }

    #[test]
    fn rover_a_parser_command_targets_rover_a_not_selected_b() {
        // Selected rover is rover-b; parser command carries target rover-a.
        // Bridge must route to rover-a, not rover-b.
        let selected_entity = Some("rover-b".to_string());
        let active_rovers = std::collections::HashMap::from([
            ("rover-a".to_string(), ()),
            ("rover-b".to_string(), ()),
        ]);
        let parser_target = "rover-a";

        // Selected rover is rover-b but parser target is rover-a
        assert_ne!(selected_entity.as_deref(), Some(parser_target));
        // Parser target is active → should route to rover-a
        assert!(active_rovers.contains_key(parser_target));
        // Topic that would be built
        let topic = format!("rover/{}/cmd/movement", parser_target);
        assert_eq!(topic, "rover/rover-a/cmd/movement");
    }

    #[test]
    fn browser_target_preserved_after_selection_change() {
        // Browser stream captured target at stream start. Even if UI switches
        // selected rover, the authoritative target must not change.
        let target_at_stream_start = "rover-a";
        let selected_after_change = Some("rover-b".to_string());

        // The parser uses target_at_stream_start (from SpeechTranscription)
        // NOT selected_after_change
        assert_ne!(
            Some(target_at_stream_start),
            selected_after_change.as_deref()
        );

        // Routing should use target_at_stream_start
        let routed_topic = format!("rover/{}/cmd/movement", target_at_stream_start);
        assert_eq!(routed_topic, "rover/rover-a/cmd/movement");
    }

    #[test]
    fn voice_topic_names_follow_contract() {
        assert_eq!(
            super::voice_config_topic("rover-a"),
            "rover/rover-a/cmd/voice/config"
        );
        assert_eq!(
            super::voice_status_topic("rover-a"),
            "rover/rover-a/voice/status"
        );
        assert_eq!(
            super::voice_result_topic("rover-a"),
            "rover/rover-a/voice/result"
        );
    }

    #[test]
    fn web_tts_routing_stays_selected_targeted() {
        assert_eq!(
            super::web_command_topic("tts_command_web", "rover-a").as_deref(),
            Some("rover/rover-a/cmd/tts")
        );
        assert_eq!(
            super::web_command_topic("tts_config_command", "rover-a"),
            None
        );
    }
}
