use dora_node_api::{
    arrow::array::{Array, BinaryArray, Float32Array},
    dora_core::config::DataId,
    DoraNode, Event,
};
use eyre::Result;
use power_coordinator::JournalRecord;
use robo_rover_lib::{
    capture_age_ms, init_tracing, power_v1_topic, record_capture_age,
    types::{ArmCommandWithMetadata, InputSource, RoverCommandWithMetadata},
    AudioFrameMetadata, AudioFrameSequenceTracker, FrameSequenceTracker, JpegFramePacket,
    LifecycleCommand, LifecycleCommandResult, LifecycleRole, LifecycleStatus, LifecycleWakeLease,
    MetricWindow, PcmFramePacket, PcmSampleFormat, PowerAuthoritySnapshot, PowerCommandResult,
    PowerStatus, PowerTopic, PowerTransition, ProtectedWorkRelayBody, ProtectedWorkRelayEnvelope,
    ProtectedWorkSnapshotRequest, RecordingOccurrence, SignedPowerCommand, SignedPowerEnvelope,
    SignedPowerEnvelopeKind, SignedPowerJournalAcknowledgement, VideoFrameMetadata,
    PROTECTED_WORK_REQUEST_TTL_MS,
};
use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use zenoh::Config;

#[path = "walkie-audio.rs"]
mod walkie_audio;
use walkie_audio::WalkieDecoder;

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
    let protected_work_key = protected_work_key_from_env()?;
    let power_command_key = power_command_key_from_env()?;
    let power_deployment_id = power_deployment_id_from_env()?;
    let (media_publish_tx, media_publish_rx) =
        flume::bounded::<MediaPublish>(MEDIA_PUBLISH_CAPACITY);
    let media_session = Arc::clone(&session);
    tokio::spawn(async move {
        while let Ok(publish) = media_publish_rx.recv_async().await {
            if let Err(error) = media_session.put(publish.topic, publish.payload).await {
                tracing::debug!(%error, "dropped failed rover media publish");
            }
        }
    });

    // =========================================================================
    // PUBLISHERS: Send data TO orchestra via Zenoh
    // =========================================================================

    let video_topic = format!("rover/{}/video/jpeg/v1", entity_id);
    let _video_pub = session
        .declare_publisher(&video_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to declare publisher {}: {}", video_topic, e))?;
    tracing::info!("Publisher: {}", video_topic);

    let audio_topic = format!("rover/{}/audio/raw", entity_id);
    let _audio_pub = session
        .declare_publisher(&audio_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to declare publisher {}: {}", audio_topic, e))?;
    tracing::info!("Publisher: {}", audio_topic);

    let playback_audio_topic = format!("rover/{}/audio/playback/raw", entity_id);
    let _playback_audio_pub = session
        .declare_publisher(&playback_audio_topic)
        .await
        .map_err(|e| {
            eyre::eyre!(
                "Failed to declare publisher {}: {}",
                playback_audio_topic,
                e
            )
        })?;
    tracing::info!("Publisher: {}", playback_audio_topic);

    let rover_telemetry_topic = format!("rover/{}/telemetry/rover", entity_id);
    let rover_telemetry_pub = session
        .declare_publisher(&rover_telemetry_topic)
        .await
        .map_err(|e| {
            eyre::eyre!(
                "Failed to declare publisher {}: {}",
                rover_telemetry_topic,
                e
            )
        })?;
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
        .map_err(|e| {
            eyre::eyre!(
                "Failed to declare publisher {}: {}",
                servo_telemetry_topic,
                e
            )
        })?;
    tracing::info!("Publisher: {}", servo_telemetry_topic);

    let resource_snapshot_topic = format!("rover/{}/resources/v1", entity_id);
    let resource_snapshot_pub = session
        .declare_publisher(&resource_snapshot_topic)
        .await
        .map_err(|e| {
            eyre::eyre!(
                "Failed to declare publisher {}: {}",
                resource_snapshot_topic,
                e
            )
        })?;
    tracing::info!("Publisher: {}", resource_snapshot_topic);

    let lifecycle_status_topic = format!("rover/{}/lifecycle/status/v1", entity_id);
    let lifecycle_status_pub = session
        .declare_publisher(&lifecycle_status_topic)
        .await
        .map_err(|e| {
            eyre::eyre!(
                "Failed to declare publisher {}: {}",
                lifecycle_status_topic,
                e
            )
        })?;
    let lifecycle_result_topic = format!("rover/{}/lifecycle/result/v1", entity_id);
    let lifecycle_result_pub = session
        .declare_publisher(&lifecycle_result_topic)
        .await
        .map_err(|e| {
            eyre::eyre!(
                "Failed to declare publisher {}: {}",
                lifecycle_result_topic,
                e
            )
        })?;
    let lifecycle_capabilities_topic = format!("rover/{}/lifecycle/capabilities/v1", entity_id);
    let lifecycle_capabilities_pub = session
        .declare_publisher(&lifecycle_capabilities_topic)
        .await
        .map_err(|e| {
            eyre::eyre!(
                "Failed to declare publisher {}: {}",
                lifecycle_capabilities_topic,
                e
            )
        })?;

    let power_status_topic = power_v1_topic(&entity_id, PowerTopic::Status);
    let power_status_pub = session
        .declare_publisher(&power_status_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to declare publisher {}: {}", power_status_topic, e))?;
    let power_transition_topic = power_v1_topic(&entity_id, PowerTopic::Transition);
    let power_transition_pub = session
        .declare_publisher(&power_transition_topic)
        .await
        .map_err(|e| {
            eyre::eyre!(
                "Failed to declare publisher {}: {}",
                power_transition_topic,
                e
            )
        })?;
    let power_command_result_topic = power_v1_topic(&entity_id, PowerTopic::CommandResult);
    let power_command_result_pub = session
        .declare_publisher(&power_command_result_topic)
        .await
        .map_err(|e| {
            eyre::eyre!(
                "Failed to declare publisher {}: {}",
                power_command_result_topic,
                e
            )
        })?;
    let power_snapshot_topic = power_v1_topic(&entity_id, PowerTopic::Snapshot);
    let power_snapshot_pub = session
        .declare_publisher(&power_snapshot_topic)
        .await
        .map_err(|e| {
            eyre::eyre!(
                "Failed to declare publisher {}: {}",
                power_snapshot_topic,
                e
            )
        })?;
    let power_event_topic = power_v1_topic(&entity_id, PowerTopic::Event);
    let power_event_pub = session
        .declare_publisher(&power_event_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to declare publisher {}: {}", power_event_topic, e))?;

    // Detection-only mode (YOLO, no tracking IDs) — separate from tracked_detections
    let detections_topic = format!("rover/{}/video/detections_only", entity_id);
    let _detections_pub = session
        .declare_publisher(&detections_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to declare publisher {}: {}", detections_topic, e))?;
    tracing::info!("Publisher: {}", detections_topic);

    // Full tracking mode (YOLO + ReID + BoTSORT, with tracking IDs)
    let tracked_detections_topic = format!("rover/{}/video/detections", entity_id);
    let _tracked_detections_pub = session
        .declare_publisher(&tracked_detections_topic)
        .await
        .map_err(|e| {
            eyre::eyre!(
                "Failed to declare publisher {}: {}",
                tracked_detections_topic,
                e
            )
        })?;
    tracing::info!("Publisher: {}", tracked_detections_topic);

    let tracking_telemetry_topic = format!("rover/{}/telemetry/tracking", entity_id);
    let _tracking_telemetry_pub = session
        .declare_publisher(&tracking_telemetry_topic)
        .await
        .map_err(|e| {
            eyre::eyre!(
                "Failed to declare publisher {}: {}",
                tracking_telemetry_topic,
                e
            )
        })?;
    tracing::info!("Publisher: {}", tracking_telemetry_topic);

    let voice_status_topic = format!("rover/{}/voice/status", entity_id);
    let voice_status_pub = session
        .declare_publisher(&voice_status_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to declare publisher {}: {}", voice_status_topic, e))?;
    tracing::info!("Publisher: {}", voice_status_topic);

    let voice_result_topic = format!("rover/{}/voice/result", entity_id);
    let tts_command_result_pub = session
        .declare_publisher(&voice_result_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to declare publisher {}: {}", voice_result_topic, e))?;
    tracing::info!("Publisher: {}", voice_result_topic);

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

    let stream_cmd_topic = format!("rover/{}/cmd/stream/v1", entity_id);
    let stream_cmd_sub = session
        .declare_subscriber(&stream_cmd_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to declare subscriber {}: {}", stream_cmd_topic, e))?;
    tracing::info!("Subscriber: {}", stream_cmd_topic);

    let tts_cmd_topic = format!("rover/{}/cmd/tts", entity_id);
    let tts_cmd_sub = session
        .declare_subscriber(&tts_cmd_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to declare subscriber {}: {}", tts_cmd_topic, e))?;
    tracing::info!("Subscriber: {}", tts_cmd_topic);

    let tts_config_topic = format!("rover/{}/cmd/voice/config", entity_id);
    let tts_config_sub = session
        .declare_subscriber(&tts_config_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to declare subscriber {}: {}", tts_config_topic, e))?;
    tracing::info!("Subscriber: {}", tts_config_topic);

    let audio_stream_topic = format!("rover/{}/cmd/audio_stream", entity_id);
    let audio_stream_sub = session
        .declare_subscriber(&audio_stream_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to declare subscriber {}: {}", audio_stream_topic, e))?;
    tracing::info!("Subscriber: {}", audio_stream_topic);

    let lifecycle_command_topic = format!("rover/{}/cmd/lifecycle/v1", entity_id);
    let lifecycle_command_sub = session
        .declare_subscriber(&lifecycle_command_topic)
        .await
        .map_err(|e| {
            eyre::eyre!(
                "Failed to declare subscriber {}: {}",
                lifecycle_command_topic,
                e
            )
        })?;
    let lifecycle_wake_lease_topic = format!("rover/{}/cmd/lifecycle-wake-lease/v1", entity_id);
    let lifecycle_wake_lease_sub = session
        .declare_subscriber(&lifecycle_wake_lease_topic)
        .await
        .map_err(|e| {
            eyre::eyre!(
                "Failed to declare subscriber {}: {}",
                lifecycle_wake_lease_topic,
                e
            )
        })?;
    let lifecycle_query_topic = format!("rover/{}/cmd/lifecycle-query/v1", entity_id);
    let lifecycle_query_sub = session
        .declare_subscriber(&lifecycle_query_topic)
        .await
        .map_err(|e| {
            eyre::eyre!(
                "Failed to declare subscriber {}: {}",
                lifecycle_query_topic,
                e
            )
        })?;
    let power_command_topic = power_v1_topic(&entity_id, PowerTopic::Command);
    let power_command_sub = session
        .declare_subscriber(&power_command_topic)
        .await
        .map_err(|e| {
            eyre::eyre!(
                "Failed to declare subscriber {}: {}",
                power_command_topic,
                e
            )
        })?;
    let power_snapshot_request_topic = power_v1_topic(&entity_id, PowerTopic::SnapshotRequest);
    let power_snapshot_request_sub = session
        .declare_subscriber(&power_snapshot_request_topic)
        .await
        .map_err(|e| {
            eyre::eyre!(
                "Failed to declare subscriber {}: {}",
                power_snapshot_request_topic,
                e
            )
        })?;
    let power_event_ack_topic = power_v1_topic(&entity_id, PowerTopic::EventAck);
    let power_event_ack_sub = session
        .declare_subscriber(&power_event_ack_topic)
        .await
        .map_err(|e| {
            eyre::eyre!(
                "Failed to declare subscriber {}: {}",
                power_event_ack_topic,
                e
            )
        })?;
    let protected_work_occurrence_topic =
        format!("rover/{}/power/protected-work/occurrence/v1", entity_id);
    let protected_work_occurrence_sub = session
        .declare_subscriber(&protected_work_occurrence_topic)
        .await
        .map_err(|e| {
            eyre::eyre!(
                "Failed to declare subscriber {}: {}",
                protected_work_occurrence_topic,
                e
            )
        })?;
    let protected_work_snapshot_topic =
        format!("rover/{}/power/protected-work/snapshot/v1", entity_id);
    let protected_work_snapshot_sub = session
        .declare_subscriber(&protected_work_snapshot_topic)
        .await
        .map_err(|e| {
            eyre::eyre!(
                "Failed to declare subscriber {}: {}",
                protected_work_snapshot_topic,
                e
            )
        })?;

    // =========================================================================
    // Dora output DataIds
    // =========================================================================

    let rover_command_output = DataId::from("rover_command".to_owned());
    let arm_command_output = DataId::from("arm_command".to_owned());
    let camera_command_output = DataId::from("camera_command".to_owned());
    let audio_command_output = DataId::from("audio_command".to_owned());
    let stream_command_output = DataId::from("stream_command".to_owned());
    let tracking_command_output = DataId::from("tracking_command".to_owned());
    let tts_command_output = DataId::from("tts_command".to_owned());
    let tts_config_command_output = DataId::from("tts_config_command".to_owned());
    let audio_stream_output = DataId::from("audio_stream".to_owned());
    let lifecycle_command_output = DataId::from("lifecycle_command_relay".to_owned());
    let lifecycle_wake_lease_output = DataId::from("lifecycle_wake_lease_relay".to_owned());
    let lifecycle_status_query_output = DataId::from("lifecycle_status_query".to_owned());
    let recording_occurrence_status_output = DataId::from("recording_occurrence_status".to_owned());
    let protected_work_snapshot_output = DataId::from("protected_work_snapshot".to_owned());
    let power_command_output = DataId::from("power_command".to_owned());
    let power_snapshot_request_output = DataId::from("power_snapshot_request".to_owned());
    let power_event_ack_output = DataId::from("power_event_ack".to_owned());

    // Statistics
    let mut video_count: u64 = 0;
    let mut telemetry_count: u64 = 0;
    let mut cmd_count: u64 = 0;
    let mut video_metrics = MetricWindow::new(Duration::from_secs(5));
    let mut video_age_metrics = MetricWindow::new(Duration::from_secs(5));
    let mut video_sequence = FrameSequenceTracker::default();
    let mut audio_count: u64 = 0;
    let mut audio_errors: u64 = 0;
    let mut audio_sequence_drops: u64 = 0;
    let mut audio_metrics = MetricWindow::new(Duration::from_secs(5));
    let mut walkie_decoder = WalkieDecoder::new();
    let mut walkie_metrics_interval = tokio::time::interval(Duration::from_secs(5));
    walkie_metrics_interval.tick().await;
    let mut protected_work_request_interval = tokio::time::interval(Duration::from_secs(15));
    protected_work_request_interval.tick().await;
    let mut walkie_received: u64 = 0;
    let mut walkie_invalid: u64 = 0;
    let mut walkie_missing_frames: u64 = 0;
    let mut walkie_missing_samples: u64 = 0;
    let mut walkie_forwarded: u64 = 0;
    let mut walkie_forward_failures: u64 = 0;
    let mut audio_age_metrics = MetricWindow::new(Duration::from_secs(5));
    let mut audio_sequence = AudioFrameSequenceTracker::default();
    let mut playback_audio_sequence = AudioFrameSequenceTracker::default();

    // Media ingress is explicitly lossy and isolated. A camera/audio burst
    // therefore cannot delay a snapshot request, power command, event record
    // or signed acknowledgement on the bounded control plane.
    let (control_tx, control_rx) = flume::bounded(CONTROL_INGRESS_CAPACITY);
    let (media_tx, media_rx) = flume::bounded(MEDIA_INGRESS_CAPACITY);
    let (power_update_tx, power_update_rx) = flume::bounded(1);
    let latest_power_updates = Arc::new(Mutex::new(LatestPowerUpdates::default()));
    let power_updates_for_thread = Arc::clone(&latest_power_updates);
    let power_update_tx_for_thread = power_update_tx.clone();

    // Spawn task to read Dora events
    std::thread::spawn(move || {
        while let Some(event) = events.recv() {
            if is_coalesced_power_update(&event) {
                let mut updates = power_updates_for_thread
                    .lock()
                    .expect("power update mutex poisoned");
                match &event {
                    Event::Input { id, .. } if id.as_str() == "power_snapshot" => {
                        updates.snapshot = Some(event);
                    }
                    Event::Input { id, .. } if id.as_str() == "power_status" => {
                        updates.status = Some(event);
                    }
                    _ => unreachable!("only power updates are coalesced"),
                }
                let _ = power_update_tx_for_thread.try_send(());
            } else if is_high_rate_dora_ingress(&event) {
                if media_tx.try_send(event).is_err() {
                    tracing::debug!("dropped saturated high-rate Dora media ingress");
                }
            } else if control_tx.send(event).is_err() {
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
            Some(event) = receive_dora_event(&control_rx, &power_update_rx, &power_update_tx, &latest_power_updates, &media_rx) => {
                match event {
                    Event::Input { id, data, metadata } => {
                        match id.as_str() {
                            "video_frame" => {
                                // Video frames are rover-side encoded JPEG BinaryArray payloads.
                                if let Some(binary_array) = data.as_any().downcast_ref::<BinaryArray>() {
                                    if binary_array.len() > 0 {
                                        let started = Instant::now();
                                        let bytes = binary_array.value(0);
                                        match frame_metadata(&metadata.parameters).and_then(|metadata| {
                                            match video_sequence.observe(metadata.frame_id) {
                                                Ok(missing) => video_metrics.record_drops(missing),
                                                Err(()) => {
                                                    video_metrics.record_error();
                                                    return Err("duplicate or regressed frame_id".into());
                                                }
                                            }
                                            let age_ms = capture_age_ms(metadata.capture_timestamp_ms)
                                                .unwrap_or_else(|| {
                                                    video_metrics.record_error();
                                                    0
                                            });
                                            video_age_metrics.record(Duration::from_millis(age_ms), 0);
                                            JpegFramePacket { metadata, payload: bytes }.encode()
                                        }) {
                                            Ok(packet) => {
                                                let packet_len = packet.len();
                                                if queue_media_publish(&media_publish_tx, &video_topic, packet) {
                                                    video_count += 1;
                                                    video_metrics.record(started.elapsed(), packet_len);
                                                } else {
                                                    video_metrics.record_drops(1);
                                                }
                                            }
                                            Err(error) => {
                                                video_metrics.record_error();
                                                tracing::error!(%error, "invalid video frame metadata");
                                            }
                                        }
                                        if let Some(snapshot) = video_metrics.snapshot_if_due() {
                                            let capture_timestamp_ms = frame_metadata(&metadata.parameters)
                                                .map(|value| value.capture_timestamp_ms).unwrap_or_default();
                                            let frame_age_ms = capture_age_ms(capture_timestamp_ms)
                                                .unwrap_or_default();
                                            tracing::info!(metric="video_pipeline", stage="rover_zenoh_publish",
                                                frame_age_ms, count=snapshot.count, bytes=snapshot.bytes,
                                                drops=snapshot.drops, errors=snapshot.errors,
                                                p50_us=snapshot.p50_us, p95_us=snapshot.p95_us,
                                                p99_us=snapshot.p99_us, max_us=snapshot.max_us);
                                        }
                                        if let Some(snapshot) = video_age_metrics.snapshot_if_due() {
                                            tracing::info!(metric="video_pipeline", stage="rover_zenoh_publish_age",
                                                count=snapshot.count, p50_us=snapshot.p50_us,
                                                p95_us=snapshot.p95_us, p99_us=snapshot.p99_us,
                                                max_us=snapshot.max_us);
                                        }
                                    }
                                }
                            }
                            "audio_frame" => {
                                let started = Instant::now();
                                let result = (|| -> Result<_, String> {
                                    let binary = data.as_any().downcast_ref::<BinaryArray>()
                                        .ok_or_else(|| "audio frame must be BinaryArray".to_string())?;
                                    if binary.len() != 1 {
                                        return Err("audio frame must contain exactly one payload".into());
                                    }
                                    let payload = binary.value(0);
                                    let audio_metadata = audio_frame_metadata(&metadata.parameters)?;
                                    if audio_metadata.format != PcmSampleFormat::S16Le {
                                        return Err("rover Zenoh bridge requires s16le audio".into());
                                    }
                                    audio_metadata.validate_payload_len(payload.len())?;
                                    let observation = audio_sequence.observe(audio_metadata)?;
                                    audio_metrics.record_drops(observation.missing_frames);
                                    audio_sequence_drops = audio_sequence_drops
                                        .saturating_add(observation.missing_frames);
                                    let age_ms = record_capture_age(
                                        &mut audio_age_metrics,
                                        audio_metadata.capture_timestamp_ms,
                                    );
                                    if age_ms.is_none() {
                                        audio_errors = audio_errors.saturating_add(1);
                                        audio_metrics.record_error();
                                    }
                                    let packet = PcmFramePacket { metadata: audio_metadata, payload }.encode()?;
                                    Ok((audio_metadata, age_ms, packet))
                                })();

                                match result {
                                    Ok((audio_metadata, frame_age_ms, packet)) => {
                                        let packet_len = packet.len();
                                        if queue_media_publish(&media_publish_tx, &audio_topic, packet) {
                                            audio_count = audio_count.saturating_add(1);
                                            audio_metrics.record(started.elapsed(), packet_len);
                                        } else {
                                            audio_metrics.record_drops(1);
                                        }
                                        if let Some(snapshot) = audio_metrics.snapshot_if_due() {
                                            tracing::info!(metric="audio_pipeline", stage="rover_zenoh_publish",
                                                stream_id=%audio_metadata.stream_id, frame_id=audio_metadata.frame_id,
                                                frame_age_ms=?frame_age_ms, count=snapshot.count, bytes=snapshot.bytes,
                                                drops=snapshot.drops, errors=snapshot.errors,
                                                p50_us=snapshot.p50_us, p95_us=snapshot.p95_us,
                                                p99_us=snapshot.p99_us, max_us=snapshot.max_us);
                                        }
                                        if let Some(snapshot) = audio_age_metrics.snapshot_if_due() {
                                            tracing::info!(metric="audio_pipeline", stage="rover_zenoh_publish_age",
                                                count=snapshot.count, p50_us=snapshot.p50_us,
                                                p95_us=snapshot.p95_us, p99_us=snapshot.p99_us,
                                                max_us=snapshot.max_us);
                                        }
                                    }
                                    Err(error) => {
                                        audio_errors = audio_errors.saturating_add(1);
                                        audio_metrics.record_error();
                                        tracing::warn!(%error, "rejected invalid audio frame");
                                    }
                                }
                            }
                            "playback_audio" => {
                                let started = Instant::now();
                                let result = (|| -> Result<_, String> {
                                    let binary = data.as_any().downcast_ref::<BinaryArray>()
                                        .ok_or_else(|| "playback audio frame must be BinaryArray".to_string())?;
                                    if binary.len() != 1 {
                                        return Err("playback audio frame must contain exactly one payload".into());
                                    }
                                    let payload = binary.value(0);
                                    let audio_metadata = audio_frame_metadata(&metadata.parameters)?;
                                    if audio_metadata.format != PcmSampleFormat::S16Le {
                                        return Err("rover Zenoh bridge requires s16le playback audio".into());
                                    }
                                    audio_metadata.validate_payload_len(payload.len())?;
                                    let observation = playback_audio_sequence.observe(audio_metadata)?;
                                    audio_metrics.record_drops(observation.missing_frames);
                                    audio_sequence_drops = audio_sequence_drops
                                        .saturating_add(observation.missing_frames);
                                    let age_ms = record_capture_age(
                                        &mut audio_age_metrics,
                                        audio_metadata.capture_timestamp_ms,
                                    );
                                    if age_ms.is_none() {
                                        audio_errors = audio_errors.saturating_add(1);
                                        audio_metrics.record_error();
                                    }
                                    let packet = PcmFramePacket { metadata: audio_metadata, payload }.encode()?;
                                    Ok((audio_metadata, age_ms, packet))
                                })();

                                match result {
                                    Ok((audio_metadata, frame_age_ms, packet)) => {
                                        let packet_len = packet.len();
                                        if queue_media_publish(&media_publish_tx, &playback_audio_topic, packet) {
                                            audio_count = audio_count.saturating_add(1);
                                            audio_metrics.record(started.elapsed(), packet_len);
                                        } else {
                                            audio_metrics.record_drops(1);
                                        }
                                        if let Some(snapshot) = audio_metrics.snapshot_if_due() {
                                            tracing::info!(metric="audio_pipeline", stage="rover_zenoh_publish_playback",
                                                stream_id=%audio_metadata.stream_id, frame_id=audio_metadata.frame_id,
                                                frame_age_ms=?frame_age_ms, count=snapshot.count, bytes=snapshot.bytes,
                                                drops=snapshot.drops, errors=snapshot.errors,
                                                p50_us=snapshot.p50_us, p95_us=snapshot.p95_us,
                                                p99_us=snapshot.p99_us, max_us=snapshot.max_us);
                                        }
                                    }
                                    Err(error) => {
                                        audio_errors = audio_errors.saturating_add(1);
                                        audio_metrics.record_error();
                                        tracing::warn!(%error, "rejected invalid playback audio frame");
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
                                            "resource_snapshot" => {
                                                let _ = resource_snapshot_pub.put(bytes).await;
                                            }
                                            "lifecycle_status" => {
                                                if serde_json::from_slice::<LifecycleStatus>(bytes).is_ok() {
                                                    let _ = lifecycle_status_pub.put(bytes).await;
                                                }
                                            }
                                            "lifecycle_command_result" => {
                                                if serde_json::from_slice::<LifecycleCommandResult>(bytes).is_ok() {
                                                    let _ = lifecycle_result_pub.put(bytes).await;
                                                }
                                            }
                                            "lifecycle_capabilities" => {
                                                let _ = lifecycle_capabilities_pub.put(bytes).await;
                                            }
                                            "power_status" => {
                                                if serde_json::from_slice::<PowerStatus>(bytes)
                                                    .is_ok_and(|status| status.validates_for(LifecycleRole::Rover, &entity_id).is_ok())
                                                {
                                                    let _ = power_status_pub.put(bytes).await;
                                                }
                                            }
                                            "power_transition" => {
                                                if serde_json::from_slice::<PowerTransition>(bytes)
                                                    .is_ok_and(|transition| transition.validates_for(LifecycleRole::Rover, &entity_id).is_ok())
                                                {
                                                    let now_ms = current_time_ms().unwrap_or_default();
                                                    match SignedPowerEnvelope::new(
                                                        SignedPowerEnvelopeKind::Transition,
                                                        LifecycleRole::Rover,
                                                        entity_id.clone(),
                                                        now_ms,
                                                        serde_json::from_slice::<PowerTransition>(bytes)?,
                                                    ).sign(&power_command_key) {
                                                        Ok(transition) => { let _ = power_transition_pub.put(serde_json::to_vec(&transition)?).await; }
                                                        Err(error) => tracing::warn!(%error, "failed to sign rover power transition"),
                                                    }
                                                }
                                            }
                                            "power_command_result" => {
                                                if serde_json::from_slice::<PowerCommandResult>(bytes)
                                                    .is_ok_and(|result| result.validate().is_ok())
                                                {
                                                    let now_ms = current_time_ms().unwrap_or_default();
                                                    match SignedPowerEnvelope::new(
                                                        SignedPowerEnvelopeKind::CommandResult,
                                                        LifecycleRole::Rover,
                                                        entity_id.clone(),
                                                        now_ms,
                                                        serde_json::from_slice::<PowerCommandResult>(bytes)?,
                                                    ).sign(&power_command_key) {
                                                        Ok(result) => {
                                                            let _ = power_command_result_pub.put(serde_json::to_vec(&result)?).await;
                                                        }
                                                        Err(error) => tracing::warn!(%error, "failed to sign rover power command result"),
                                                    }
                                                }
                                            }
                                            "power_snapshot" => {
                                                if serde_json::from_slice::<PowerAuthoritySnapshot>(bytes)
                                                    .is_ok_and(|snapshot| snapshot.validates_for(LifecycleRole::Rover, &entity_id).is_ok())
                                                {
                                                    let now_ms = current_time_ms().unwrap_or_default();
                                                    match SignedPowerEnvelope::new(
                                                        SignedPowerEnvelopeKind::Snapshot,
                                                        LifecycleRole::Rover,
                                                        entity_id.clone(),
                                                        now_ms,
                                                        serde_json::from_slice::<PowerAuthoritySnapshot>(bytes)?,
                                                    ).sign(&power_command_key) {
                                                        Ok(snapshot) => {
                                                            let _ = power_snapshot_pub.put(serde_json::to_vec(&snapshot)?).await;
                                                        }
                                                        Err(error) => tracing::warn!(%error, "failed to sign rover power snapshot"),
                                                    }
                                                }
                                            }
                                            "power_journal_record" => {
                                                if serde_json::from_slice::<JournalRecord>(bytes)
                                                    .is_ok_and(|record| record.validate().is_ok() && record.event.role == LifecycleRole::Rover && record.event.entity_id == entity_id)
                                                {
                                                    let _ = power_event_pub.put(bytes).await;
                                                }
                                            }
                                            "detections" => {
                                                let _ = queue_media_publish(
                                                    &media_publish_tx,
                                                    &detections_topic,
                                                    bytes.to_vec(),
                                                );
                                            }
                                            "tracked_detections" => {
                                                let _ = queue_media_publish(
                                                    &media_publish_tx,
                                                    &tracked_detections_topic,
                                                    bytes.to_vec(),
                                                );
                                            }
                                            "tracking_telemetry" => {
                                                let _ = queue_media_publish(
                                                    &media_publish_tx,
                                                    &tracking_telemetry_topic,
                                                    bytes.to_vec(),
                                                );
                                            }
                                            "voice_status" => {
                                                if let Err(error) = voice_status_pub.put(bytes).await {
                                                    tracing::error!(%error, "failed to publish voice status");
                                                }
                                            }
                                            "tts_command_result" => {
                                                if let Err(error) =
                                                    tts_command_result_pub.put(bytes).await
                                                {
                                                    tracing::error!(
                                                        %error,
                                                        "failed to publish TTS command result"
                                                    );
                                                }
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
                        tracing::info!(metric="audio_pipeline_total", stage="rover_zenoh_publish",
                            frames_published=audio_count, sequence_drops=audio_sequence_drops,
                            errors=audio_errors);
                        tracing::info!(metric="walkie_transport_total", stage="rover_zenoh_shutdown",
                            received_frames=walkie_received, invalid_frames=walkie_invalid,
                            missing_frames=walkie_missing_frames,
                            missing_samples=walkie_missing_samples,
                            forwarded_frames=walkie_forwarded,
                            forward_failures=walkie_forward_failures);
                        tracing::info!("Stats: video={}, audio={}, telemetry={}, commands={}", video_count, audio_count, telemetry_count, cmd_count);
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

            Ok(sample) = stream_cmd_sub.recv_async() => {
                let payload = sample.payload().to_bytes();
                let arrow_data = BinaryArray::from_vec(vec![payload.as_ref()]);
                let _ = node.send_output(stream_command_output.clone(), Default::default(), arrow_data);
            }

            Ok(sample) = tts_cmd_sub.recv_async() => {
                let payload = sample.payload().to_bytes();
                let arrow_data = BinaryArray::from_vec(vec![payload.as_ref()]);
                let _ = node.send_output(tts_command_output.clone(), Default::default(), arrow_data);
            }

            Ok(sample) = tts_config_sub.recv_async() => {
                let payload = sample.payload().to_bytes();
                let arrow_data = BinaryArray::from_vec(vec![payload.as_ref()]);
                let _ = node.send_output(
                    tts_config_command_output.clone(),
                    Default::default(),
                    arrow_data,
                );
            }

            Ok(sample) = audio_stream_sub.recv_async() => {
                let payload = sample.payload().to_bytes();
                match walkie_decoder.decode(payload.as_ref()) {
                    Ok(frame) => {
                        walkie_received = walkie_received.saturating_add(1);
                        walkie_missing_frames = walkie_missing_frames
                            .saturating_add(frame.missing_frames);
                        walkie_missing_samples = walkie_missing_samples
                            .saturating_add(frame.missing_samples);
                        let audio_array = Float32Array::from(frame.samples);
                        if let Err(error) = node.send_output(
                            audio_stream_output.clone(), frame.parameters, audio_array)
                        {
                            walkie_forward_failures = walkie_forward_failures.saturating_add(1);
                            tracing::error!(%error, "failed to forward speaker audio to Dora");
                        } else {
                            walkie_forwarded = walkie_forwarded.saturating_add(1);
                        }
                    }
                    Err(error) => {
                        walkie_invalid = walkie_invalid.saturating_add(1);
                        tracing::warn!(%error, "rejected invalid speaker audio payload");
                    }
                }
            }

            Ok(sample) = lifecycle_command_sub.recv_async() => {
                let payload = sample.payload().to_bytes();
                if let Some(command) = lifecycle_command_for_manager(payload.as_ref()) {
                    tracing::info!(request_id = %command.request_id, target = ?command.target, "received validated lifecycle command payload from Zenoh; forwarding to lifecycle manager");
                    let _ = node.send_output(lifecycle_command_output.clone(), Default::default(), BinaryArray::from_vec(vec![payload.as_ref()]));
                } else { tracing::warn!("rejected malformed lifecycle command"); }
            }

            Ok(sample) = lifecycle_wake_lease_sub.recv_async() => {
                let payload = sample.payload().to_bytes();
                if serde_json::from_slice::<LifecycleWakeLease>(&payload).is_ok() {
                    let _ = node.send_output(lifecycle_wake_lease_output.clone(), Default::default(), BinaryArray::from_vec(vec![payload.as_ref()]));
                } else { tracing::warn!("rejected malformed lifecycle wake lease"); }
            }

            Ok(sample) = lifecycle_query_sub.recv_async() => {
                let payload = sample.payload().to_bytes();
                let _ = node.send_output(lifecycle_status_query_output.clone(), Default::default(), BinaryArray::from_vec(vec![payload.as_ref()]));
            }

            Ok(sample) = power_command_sub.recv_async() => {
                let payload = sample.payload().to_bytes();
                match serde_json::from_slice::<SignedPowerCommand>(&payload) {
                    Ok(envelope)
                        if envelope.verify(&power_command_key, current_time_ms().unwrap_or_default()).is_ok()
                            && envelope.validates_for(SignedPowerEnvelopeKind::Command, LifecycleRole::Rover, &entity_id).is_ok()
                            && envelope.payload.validates_for(LifecycleRole::Rover, &entity_id).is_ok() => {
                        let command = serde_json::to_vec(&envelope.payload)?;
                        let arrow_data = BinaryArray::from_vec(vec![command.as_slice()]);
                        if let Err(error) = node.send_output(power_command_output.clone(), Default::default(), arrow_data) {
                            tracing::warn!(%error, "failed to forward rover power command");
                        }
                    }
                    _ => tracing::warn!("rejected invalid or misrouted power command"),
                }
            }

            Ok(_sample) = power_snapshot_request_sub.recv_async() => {
                // The coordinator emits its authoritative snapshot after every input,
                // including this explicit reconnect request.
                let arrow_data = BinaryArray::from_vec(vec![b"{}".as_slice()]);
                if let Err(error) = node.send_output(power_snapshot_request_output.clone(), Default::default(), arrow_data) {
                    tracing::warn!(%error, "failed to forward power snapshot request");
                }
            }

            Ok(sample) = power_event_ack_sub.recv_async() => {
                let payload = sample.payload().to_bytes();
                match serde_json::from_slice::<SignedPowerJournalAcknowledgement>(&payload) {
                    Ok(envelope)
                        if envelope.verify(&power_command_key, current_time_ms().unwrap_or_default()).is_ok()
                            && envelope.validates_for(
                                SignedPowerEnvelopeKind::JournalAcknowledgement,
                                LifecycleRole::Rover,
                                &entity_id,
                            ).is_ok()
                            && envelope
                                .payload
                                .validates_for(&entity_id, Some(&power_deployment_id))
                                .is_ok() => {
                        let acknowledgement = serde_json::to_vec(&envelope.payload)?;
                        let _ = node.send_output(
                            power_event_ack_output.clone(),
                            Default::default(),
                            BinaryArray::from_vec(vec![acknowledgement.as_slice()]),
                        );
                    }
                    _ => tracing::warn!("rejected invalid or stale remote power journal acknowledgement"),
                }
            }

            Ok(sample) = protected_work_occurrence_sub.recv_async() => {
                let payload = sample.payload().to_bytes();
                if let Some(occurrence) = verified_protected_work_occurrence(
                    payload.as_ref(),
                    &entity_id,
                    &protected_work_key,
                ) {
                    let serialized = serde_json::to_vec(&occurrence)?;
                    let _ = node.send_output(
                        recording_occurrence_status_output.clone(),
                        Default::default(),
                        BinaryArray::from_vec(vec![serialized.as_slice()]),
                    );
                } else {
                    tracing::warn!("rejected unauthenticated or misrouted protected recording work");
                }
            }

            Ok(sample) = protected_work_snapshot_sub.recv_async() => {
                let payload = sample.payload().to_bytes();
                if let Some(snapshot) = verified_protected_work_snapshot(
                    payload.as_ref(),
                    &entity_id,
                    &protected_work_key,
                ) {
                    let serialized = serde_json::to_vec(&snapshot)?;
                    let _ = node.send_output(
                        protected_work_snapshot_output.clone(),
                        Default::default(),
                        BinaryArray::from_vec(vec![serialized.as_slice()]),
                    );
                } else {
                    tracing::warn!("rejected unauthenticated or stale protected-work snapshot");
                }
            }

            _ = protected_work_request_interval.tick() => {
                if let Ok(envelope) = protected_work_snapshot_request(&entity_id, &protected_work_key) {
                    let topic = format!("rover/{entity_id}/power/protected-work/request/v1");
                    if let Err(error) = session.put(topic, serde_json::to_vec(&envelope)?).await {
                        tracing::warn!(%error, "failed to request protected-work snapshot");
                    }
                }
            }

            _ = walkie_metrics_interval.tick() => {
                tracing::info!(metric="walkie_transport_total", stage="rover_zenoh_receive",
                    received_frames=walkie_received, invalid_frames=walkie_invalid,
                    missing_frames=walkie_missing_frames,
                    missing_samples=walkie_missing_samples,
                    forwarded_frames=walkie_forwarded,
                    forward_failures=walkie_forward_failures);
            }
        }
    }

    Ok(())
}

fn lifecycle_command_for_manager(payload: &[u8]) -> Option<LifecycleCommand> {
    serde_json::from_slice::<LifecycleCommand>(payload)
        .ok()
        .filter(|command| command.validate().is_ok())
}

fn protected_work_key_from_env() -> Result<Vec<u8>> {
    let key = std::env::var("POWER_PROTECTED_WORK_HMAC_KEY")
        .map_err(|_| eyre::eyre!("POWER_PROTECTED_WORK_HMAC_KEY is required"))?;
    if key.len() < 32 {
        return Err(eyre::eyre!(
            "POWER_PROTECTED_WORK_HMAC_KEY must be at least 32 bytes"
        ));
    }
    Ok(key.into_bytes())
}

fn power_command_key_from_env() -> Result<Vec<u8>> {
    let key = std::env::var("POWER_COMMAND_HMAC_KEY")
        .map_err(|_| eyre::eyre!("POWER_COMMAND_HMAC_KEY is required"))?;
    if key.len() < 32 {
        return Err(eyre::eyre!(
            "POWER_COMMAND_HMAC_KEY must be at least 32 bytes"
        ));
    }
    Ok(key.into_bytes())
}

fn power_deployment_id_from_env() -> Result<String> {
    let deployment_id = std::env::var("POWER_DEPLOYMENT_ID").unwrap_or_else(|_| "default".into());
    if deployment_id.is_empty()
        || deployment_id.len() > 128
        || !deployment_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(eyre::eyre!("invalid POWER_DEPLOYMENT_ID"));
    }
    Ok(deployment_id)
}

fn protected_work_snapshot_request(
    entity_id: &str,
    key: &[u8],
) -> Result<ProtectedWorkRelayEnvelope, String> {
    let issued_at_ms = current_time_ms()?;
    ProtectedWorkRelayEnvelope::new(
        entity_id.into(),
        issued_at_ms,
        PROTECTED_WORK_REQUEST_TTL_MS,
        ProtectedWorkRelayBody::SnapshotRequest {
            request: ProtectedWorkSnapshotRequest {
                protocol_version: robo_rover_lib::PROTECTED_WORK_RELAY_PROTOCOL_VERSION,
                request_id: uuid::Uuid::new_v4().to_string(),
                entity_id: entity_id.into(),
            },
        },
    )
    .sign(key)
}

fn verified_protected_work_occurrence(
    payload: &[u8],
    entity_id: &str,
    key: &[u8],
) -> Option<RecordingOccurrence> {
    let envelope = serde_json::from_slice::<ProtectedWorkRelayEnvelope>(payload).ok()?;
    envelope.verify(key, current_time_ms().ok()?).ok()?;
    match envelope.body {
        ProtectedWorkRelayBody::Occurrence { occurrence }
            if envelope.target_entity_id == entity_id && occurrence.entity_id == entity_id =>
        {
            Some(occurrence)
        }
        _ => None,
    }
}

fn verified_protected_work_snapshot(
    payload: &[u8],
    entity_id: &str,
    key: &[u8],
) -> Option<robo_rover_lib::ProtectedWorkSnapshot> {
    let envelope = serde_json::from_slice::<ProtectedWorkRelayEnvelope>(payload).ok()?;
    envelope.verify(key, current_time_ms().ok()?).ok()?;
    match envelope.body {
        ProtectedWorkRelayBody::Snapshot { snapshot }
            if envelope.target_entity_id == entity_id && snapshot.entity_id == entity_id =>
        {
            Some(snapshot)
        }
        _ => None,
    }
}

fn current_time_ms() -> Result<u64, String> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| "system clock is before Unix epoch".into())
        .map(|duration| duration.as_millis() as u64)
}

const CONTROL_INGRESS_CAPACITY: usize = 64;
const MEDIA_INGRESS_CAPACITY: usize = 8;
const MEDIA_PUBLISH_CAPACITY: usize = 8;

struct MediaPublish {
    topic: String,
    payload: Vec<u8>,
}

fn queue_media_publish(
    sender: &flume::Sender<MediaPublish>,
    topic: &str,
    payload: Vec<u8>,
) -> bool {
    if sender
        .try_send(MediaPublish {
            topic: topic.into(),
            payload,
        })
        .is_err()
    {
        tracing::debug!("dropped saturated rover media publish");
        false
    } else {
        true
    }
}

#[derive(Default)]
struct LatestPowerUpdates {
    status: Option<Event>,
    snapshot: Option<Event>,
}

fn is_high_rate_dora_ingress(event: &Event) -> bool {
    matches!(event, Event::Input { id, .. } if is_high_rate_input_id(id.as_str()))
}

fn is_high_rate_input_id(id: &str) -> bool {
    matches!(
        id,
        "video_frame"
            | "audio_frame"
            | "playback_audio"
            | "detections"
            | "tracked_detections"
            | "tracking_telemetry"
    )
}

fn is_coalesced_power_update(event: &Event) -> bool {
    matches!(event, Event::Input { id, .. } if is_coalesced_power_update_id(id.as_str()))
}

fn is_coalesced_power_update_id(id: &str) -> bool {
    matches!(id, "power_status" | "power_snapshot")
}

async fn receive_dora_event(
    control_rx: &flume::Receiver<Event>,
    power_update_rx: &flume::Receiver<()>,
    power_update_tx: &flume::Sender<()>,
    latest_power_updates: &Arc<Mutex<LatestPowerUpdates>>,
    media_rx: &flume::Receiver<Event>,
) -> Option<Event> {
    loop {
        tokio::select! {
            biased;
            event = control_rx.recv_async() => return event.ok(),
            _ = power_update_rx.recv_async() => {
                let mut updates = latest_power_updates.lock().expect("power update mutex poisoned");
                let event = updates.snapshot.take().or_else(|| updates.status.take());
                if updates.snapshot.is_some() || updates.status.is_some() {
                    let _ = power_update_tx.try_send(());
                }
                if event.is_some() {
                    return event;
                }
            }
            event = media_rx.recv_async() => return event.ok(),
        }
    }
}

fn frame_metadata(
    parameters: &std::collections::BTreeMap<String, dora_node_api::Parameter>,
) -> Result<VideoFrameMetadata, String> {
    let integer = |key: &str| {
        parameters
            .get(key)
            .and_then(|value| match value {
                dora_node_api::Parameter::Integer(value) => u64::try_from(*value).ok(),
                _ => None,
            })
            .ok_or_else(|| format!("missing or invalid {key}"))
    };
    Ok(VideoFrameMetadata {
        frame_id: integer("frame_id")?,
        capture_timestamp_ms: integer("capture_timestamp_ms")?,
        width: integer("width")?
            .try_into()
            .map_err(|_| "width exceeds u32")?,
        height: integer("height")?
            .try_into()
            .map_err(|_| "height exceeds u32")?,
    })
}

fn audio_frame_metadata(
    parameters: &std::collections::BTreeMap<String, dora_node_api::Parameter>,
) -> Result<AudioFrameMetadata, String> {
    let integer = |key: &str| {
        parameters
            .get(key)
            .and_then(|value| match value {
                dora_node_api::Parameter::Integer(value) => u64::try_from(*value).ok(),
                _ => None,
            })
            .ok_or_else(|| format!("missing or invalid {key}"))
    };
    let string = |key: &str| {
        parameters
            .get(key)
            .and_then(|value| match value {
                dora_node_api::Parameter::String(value) => Some(value.as_str()),
                _ => None,
            })
            .ok_or_else(|| format!("missing or invalid {key}"))
    };
    Ok(AudioFrameMetadata {
        stream_id: uuid::Uuid::parse_str(string("stream_id")?).map_err(|e| e.to_string())?,
        frame_id: integer("frame_id")?,
        capture_timestamp_ms: integer("capture_timestamp_ms")?,
        sample_rate: integer("sample_rate")?
            .try_into()
            .map_err(|_| "sample_rate exceeds u32")?,
        channels: integer("channels")?
            .try_into()
            .map_err(|_| "channels exceeds u16")?,
        sample_count: integer("sample_count")?
            .try_into()
            .map_err(|_| "sample_count exceeds u32")?,
        format: PcmSampleFormat::from_metadata_name(string("format")?)?,
    })
}

#[cfg(test)]
mod lifecycle_command_tests {
    use super::*;
    use robo_rover_lib::{
        LifecycleDesiredState, LifecycleRole, LifecycleTarget, LIFECYCLE_PROTOCOL_VERSION,
    };

    fn valid_command() -> LifecycleCommand {
        LifecycleCommand {
            protocol_version: LIFECYCLE_PROTOCOL_VERSION,
            request_id: "550e8400-e29b-41d4-a716-446655440000".into(),
            manager_epoch: 1,
            target: LifecycleTarget {
                role: LifecycleRole::Rover,
                entity_id: "rover-kiwi".into(),
                node_id: "edge-voice".into(),
            },
            desired_state: LifecycleDesiredState::Quiesced,
            expected_revision: 0,
            issued_at_ms: 1,
            expires_at_ms: 2,
            origin: Default::default(),
            transition_id: None,
        }
    }

    #[test]
    fn malformed_zenoh_lifecycle_payload_is_not_forwarded_to_manager() {
        assert!(lifecycle_command_for_manager(br#"{\"target\":\"not-a-command\"}"#).is_none());
    }

    #[test]
    fn invalid_zenoh_lifecycle_payload_is_not_forwarded_to_manager() {
        let mut command = valid_command();
        command.expires_at_ms = command.issued_at_ms;

        assert!(lifecycle_command_for_manager(&serde_json::to_vec(&command).unwrap()).is_none());
    }

    #[test]
    fn valid_zenoh_lifecycle_payload_remains_forwardable() {
        assert!(
            lifecycle_command_for_manager(&serde_json::to_vec(&valid_command()).unwrap()).is_some()
        );
    }

    #[test]
    fn protected_work_requires_a_valid_occurrence_for_this_rover() {
        let occurrence = serde_json::json!({
            "occurrence_id": "f4f3e2d1-c0b9-48a7-9615-141312111000",
            "schedule_id": "f4f3e2d1-c0b9-48a7-9615-141312111001",
            "schedule_revision": 1,
            "entity_id": "rover-kiwi",
            "planned_start_ms": 1,
            "planned_end_ms": 2,
            "dst_resolution": "exact",
            "state": "active",
            "retry_count": 0,
            "next_retry_at_ms": null,
            "group_id": null,
            "start_request_id": "f4f3e2d1-c0b9-48a7-9615-141312111002",
            "attempts": [],
            "last_error": null,
            "suppressed_by_manual": false,
            "created_at_ms": 1,
            "updated_at_ms": 1,
            "terminal_at_ms": null,
            "expires_at_ms": null
        });
        let key = b"12345678901234567890123456789012";
        let envelope = ProtectedWorkRelayEnvelope::new(
            "rover-kiwi".into(),
            super::current_time_ms().unwrap(),
            robo_rover_lib::PROTECTED_WORK_RELAY_TTL_MS,
            ProtectedWorkRelayBody::Occurrence {
                occurrence: serde_json::from_value(occurrence).unwrap(),
            },
        )
        .sign(key)
        .unwrap();
        let payload = serde_json::to_vec(&envelope).unwrap();
        assert!(super::verified_protected_work_occurrence(&payload, "rover-kiwi", key).is_some());
        assert!(super::verified_protected_work_occurrence(&payload, "rover-other", key).is_none());
    }

    #[test]
    fn saturated_media_ingress_cannot_take_the_control_plane() {
        assert!(super::is_high_rate_input_id("video_frame"));
        assert!(super::is_high_rate_input_id("audio_frame"));
        assert!(!super::is_high_rate_input_id("power_command"));
        assert!(!super::is_high_rate_input_id("power_snapshot_request"));
        assert!(!super::is_high_rate_input_id("power_journal_record"));
        assert!(super::is_coalesced_power_update_id("power_status"));
        assert!(super::is_coalesced_power_update_id("power_snapshot"));
        assert!(!super::is_coalesced_power_update_id("power_command"));
    }

    #[test]
    fn a_stalled_media_publisher_cannot_fill_the_control_queue() {
        let (control_tx, control_rx) = flume::bounded(1);
        let (media_tx, media_rx) = flume::bounded(1);
        assert!(super::queue_media_publish(
            &media_tx,
            "rover/test/media",
            vec![1]
        ));
        assert!(!super::queue_media_publish(
            &media_tx,
            "rover/test/media",
            vec![2]
        ));
        assert!(control_tx.try_send(()).is_ok());
        assert_eq!(control_rx.len(), 1);
        assert_eq!(media_rx.len(), 1);
    }
}
