use dora_node_api::{
    arrow::array::{Array, BinaryArray, UInt8Array, Float32Array},
    dora_core::config::DataId,
    DoraNode, Event,
};
use eyre::Result;
use robo_rover_lib::{
    init_tracing,
    types::{RoverCommandWithMetadata, ArmCommandWithMetadata, InputSource},
};
use zenoh::Config;

#[tokio::main]
async fn main() -> Result<()> {
    let _guard = init_tracing();

    tracing::info!("Starting Rover Zenoh Bridge");

    // Get entity ID from environment
    let entity_id = std::env::var("ENTITY_ID").unwrap_or_else(|_| "rover-kiwi".to_string());
    tracing::info!("Rover ID: {}", entity_id);

    // Initialize Dora node
    let (mut node, mut events) = DoraNode::init_from_env()?;

    // Initialize Zenoh session with config file
    let config_path = std::env::var("ZENOH_CONFIG")
        .unwrap_or_else(|_| "rover-kiwi/zenoh_bridge/zenoh_config.json5".to_string());

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

    let session = zenoh::open(config).await
        .map_err(|e| eyre::eyre!("Failed to open Zenoh session: {}", e))?;

    tracing::info!("Zenoh session ID: {}", session.zid());

    // =========================================================================
    // PUBLISHERS: Send data TO orchestra via Zenoh
    // =========================================================================

    let video_topic = format!("rover/{}/video/raw", entity_id);
    let video_pub = session
        .declare_publisher(&video_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to declare publisher {}: {}", video_topic, e))?;
    tracing::info!("Publisher: {}", video_topic);

    let audio_topic = format!("rover/{}/audio/raw", entity_id);
    let audio_pub = session
        .declare_publisher(&audio_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to declare publisher {}: {}", audio_topic, e))?;
    tracing::info!("Publisher: {}", audio_topic);

    let rover_telemetry_topic = format!("rover/{}/telemetry/rover", entity_id);
    let rover_telemetry_pub = session
        .declare_publisher(&rover_telemetry_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to declare publisher {}: {}", rover_telemetry_topic, e))?;
    tracing::info!("Publisher: {}", rover_telemetry_topic);

    let arm_telemetry_topic = format!("rover/{}/telemetry/arm", entity_id);
    let arm_telemetry_pub = session
        .declare_publisher(&arm_telemetry_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to declare publisher {}: {}", arm_telemetry_topic, e))?;
    tracing::info!("Publisher: {}", arm_telemetry_topic);

    let servo_telemetry_topic = format!("rover/{}/telemetry/servo", entity_id);
    let servo_telemetry_pub = session
        .declare_publisher(&servo_telemetry_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to declare publisher {}: {}", servo_telemetry_topic, e))?;
    tracing::info!("Publisher: {}", servo_telemetry_topic);

    let metrics_topic = format!("rover/{}/metrics", entity_id);
    let metrics_pub = session
        .declare_publisher(&metrics_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to declare publisher {}: {}", metrics_topic, e))?;
    tracing::info!("Publisher: {}", metrics_topic);

    let tracked_detections_topic = format!("rover/{}/video/detections", entity_id);
    let tracked_detections_pub = session
        .declare_publisher(&tracked_detections_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to declare publisher {}: {}", tracked_detections_topic, e))?;
    tracing::info!("Publisher: {}", tracked_detections_topic);

    let tracking_telemetry_topic = format!("rover/{}/telemetry/tracking", entity_id);
    let tracking_telemetry_pub = session
        .declare_publisher(&tracking_telemetry_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to declare publisher {}: {}", tracking_telemetry_topic, e))?;
    tracing::info!("Publisher: {}", tracking_telemetry_topic);

    // =========================================================================
    // SUBSCRIBERS: Receive commands FROM orchestra via Zenoh
    // =========================================================================

    let rover_cmd_topic = format!("rover/{}/cmd/movement", entity_id);
    let rover_cmd_sub = session
        .declare_subscriber(&rover_cmd_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to declare subscriber {}: {}", rover_cmd_topic, e))?;
    tracing::info!("Subscriber: {}", rover_cmd_topic);

    let arm_cmd_topic = format!("rover/{}/cmd/arm", entity_id);
    let arm_cmd_sub = session
        .declare_subscriber(&arm_cmd_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to declare subscriber {}: {}", arm_cmd_topic, e))?;
    tracing::info!("Subscriber: {}", arm_cmd_topic);

    let camera_cmd_topic = format!("rover/{}/cmd/camera", entity_id);
    let camera_cmd_sub = session
        .declare_subscriber(&camera_cmd_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to declare subscriber {}: {}", camera_cmd_topic, e))?;
    tracing::info!("Subscriber: {}", camera_cmd_topic);

    let audio_cmd_topic = format!("rover/{}/cmd/audio", entity_id);
    let audio_cmd_sub = session
        .declare_subscriber(&audio_cmd_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to declare subscriber {}: {}", audio_cmd_topic, e))?;
    tracing::info!("Subscriber: {}", audio_cmd_topic);

    let tracking_cmd_topic = format!("rover/{}/cmd/tracking", entity_id);
    let tracking_cmd_sub = session
        .declare_subscriber(&tracking_cmd_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to declare subscriber {}: {}", tracking_cmd_topic, e))?;
    tracing::info!("Subscriber: {}", tracking_cmd_topic);

    let tts_cmd_topic = format!("rover/{}/cmd/tts", entity_id);
    let tts_cmd_sub = session
        .declare_subscriber(&tts_cmd_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to declare subscriber {}: {}", tts_cmd_topic, e))?;
    tracing::info!("Subscriber: {}", tts_cmd_topic);

    let audio_stream_topic = format!("rover/{}/cmd/audio_stream", entity_id);
    let audio_stream_sub = session
        .declare_subscriber(&audio_stream_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to declare subscriber {}: {}", audio_stream_topic, e))?;
    tracing::info!("Subscriber: {}", audio_stream_topic);

    // =========================================================================
    // Dora output DataIds
    // =========================================================================

    let rover_command_output = DataId::from("rover_command".to_owned());
    let arm_command_output = DataId::from("arm_command".to_owned());
    let camera_command_output = DataId::from("camera_command".to_owned());
    let audio_command_output = DataId::from("audio_command".to_owned());
    let tracking_command_output = DataId::from("tracking_command".to_owned());
    let tts_command_output = DataId::from("tts_command".to_owned());
    let audio_stream_output = DataId::from("audio_stream".to_owned());

    // Statistics
    let mut video_count: u64 = 0;
    let mut telemetry_count: u64 = 0;
    let mut cmd_count: u64 = 0;

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
    // Main event loop
    // =========================================================================

    loop {
        tokio::select! {
            // Handle Dora events (data FROM local dataflow TO publish to Zenoh)
            Ok(event) = dora_rx.recv_async() => {
                match event {
                    Event::Input { id, data, .. } => {
                        match id.as_str() {
                            "video_frame" => {
                                // Video frames are UInt8Array (raw RGB8 bytes)
                                if let Some(uint8_array) = data.as_any().downcast_ref::<UInt8Array>() {
                                    if uint8_array.len() > 0 {
                                        let bytes = uint8_array.values().as_ref();
                                        let _ = video_pub.put(bytes).await;
                                        video_count += 1;
                                        if video_count % 30 == 0 {
                                            tracing::info!("Published {} video frames", video_count);
                                        }
                                    }
                                }
                            }
                            "audio_frame" => {
                                // Audio frames are Float32Array (raw Float32 audio samples)
                                if let Some(float32_array) = data.as_any().downcast_ref::<Float32Array>() {
                                    if float32_array.len() > 0 {
                                        // Convert Float32Array to bytes for Zenoh transport
                                        let float_slice = float32_array.values().as_ref();
                                        let bytes: &[u8] = unsafe {
                                            std::slice::from_raw_parts(
                                                float_slice.as_ptr() as *const u8,
                                                float_slice.len() * std::mem::size_of::<f32>()
                                            )
                                        };
                                        let _ = audio_pub.put(bytes).await;
                                    }
                                }
                            }
                            _ => {
                                // Other data types are BinaryArray (JSON serialized)
                                if let Some(binary_array) = data.as_any().downcast_ref::<BinaryArray>() {
                                    if binary_array.len() > 0 {
                                        let bytes = binary_array.value(0);

                                        match id.as_str() {
                                            "rover_telemetry" => {
                                                let _ = rover_telemetry_pub.put(bytes).await;
                                                telemetry_count += 1;
                                            }
                                            "arm_telemetry" => {
                                                let _ = arm_telemetry_pub.put(bytes).await;
                                            }
                                            "servo_telemetry" => {
                                                let _ = servo_telemetry_pub.put(bytes).await;
                                            }
                                            "performance_metrics" => {
                                                let _ = metrics_pub.put(bytes).await;
                                            }
                                            "tracked_detections" => {
                                                let _ = tracked_detections_pub.put(bytes).await;
                                            }
                                            "tracking_telemetry" => {
                                                let _ = tracking_telemetry_pub.put(bytes).await;
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Event::Stop(_) => {
                        tracing::info!("Stop signal received");
                        tracing::info!("Stats: video={}, telemetry={}, commands={}", video_count, telemetry_count, cmd_count);
                        break;
                    }
                    _ => {}
                }
            }

            // Handle Zenoh rover command subscription
            Ok(sample) = rover_cmd_sub.recv_async() => {
                let payload = sample.payload().to_bytes();
                match serde_json::from_slice::<RoverCommandWithMetadata>(&payload) {
                    Ok(mut rover_cmd) => {
                        if matches!(rover_cmd.metadata.source, InputSource::WebBridge) {
                            rover_cmd.metadata.source = InputSource::Zenoh;
                        }
                        if let Ok(serialized) = serde_json::to_vec(&rover_cmd) {
                            let arrow_data = BinaryArray::from_vec(vec![serialized.as_slice()]);
                            let _ = node.send_output(rover_command_output.clone(), Default::default(), arrow_data);
                            cmd_count += 1;
                        }
                    }
                    Err(e) => tracing::error!("Failed to parse rover command: {}", e),
                }
            }

            // Handle Zenoh arm command subscription
            Ok(sample) = arm_cmd_sub.recv_async() => {
                let payload = sample.payload().to_bytes();
                match serde_json::from_slice::<ArmCommandWithMetadata>(&payload) {
                    Ok(mut arm_cmd) => {
                        if matches!(arm_cmd.metadata.source, InputSource::WebBridge) {
                            arm_cmd.metadata.source = InputSource::Zenoh;
                        }
                        if let Ok(serialized) = serde_json::to_vec(&arm_cmd) {
                            let arrow_data = BinaryArray::from_vec(vec![serialized.as_slice()]);
                            let _ = node.send_output(arm_command_output.clone(), Default::default(), arrow_data);
                            cmd_count += 1;
                        }
                    }
                    Err(e) => tracing::error!("Failed to parse arm command: {}", e),
                }
            }

            // Handle other command subscriptions (pass-through)
            Ok(sample) = camera_cmd_sub.recv_async() => {
                let payload = sample.payload().to_bytes();
                let arrow_data = BinaryArray::from_vec(vec![payload.as_ref()]);
                let _ = node.send_output(camera_command_output.clone(), Default::default(), arrow_data);
            }

            Ok(sample) = audio_cmd_sub.recv_async() => {
                let payload = sample.payload().to_bytes();
                let arrow_data = BinaryArray::from_vec(vec![payload.as_ref()]);
                let _ = node.send_output(audio_command_output.clone(), Default::default(), arrow_data);
            }

            Ok(sample) = tracking_cmd_sub.recv_async() => {
                let payload = sample.payload().to_bytes();
                let arrow_data = BinaryArray::from_vec(vec![payload.as_ref()]);
                let _ = node.send_output(tracking_command_output.clone(), Default::default(), arrow_data);
            }

            Ok(sample) = tts_cmd_sub.recv_async() => {
                let payload = sample.payload().to_bytes();
                let arrow_data = BinaryArray::from_vec(vec![payload.as_ref()]);
                let _ = node.send_output(tts_command_output.clone(), Default::default(), arrow_data);
            }

            Ok(sample) = audio_stream_sub.recv_async() => {
                let payload = sample.payload().to_bytes();

                // Convert raw bytes back to Float32Array
                let float_slice: &[f32] = unsafe {
                    std::slice::from_raw_parts(
                        payload.as_ref().as_ptr() as *const f32,
                        payload.len() / std::mem::size_of::<f32>()
                    )
                };
                let audio_array = Float32Array::from(float_slice.to_vec());

                // Send as Float32Array
                let _ = node.send_output(audio_stream_output.clone(), Default::default(), audio_array);
            }
        }
    }

    Ok(())
}
