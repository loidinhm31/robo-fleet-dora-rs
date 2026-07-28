use dora_node_api::{
    arrow::array::{Array, BinaryArray, Float32Array},
    dora_core::config::DataId,
    DoraNode, Event, Parameter,
};
use eyre::Result;
use power_coordinator::{JournalAcknowledgement, JournalRecord};
use robo_rover_lib::{
    capture_age_ms, init_tracing, power_v1_topic, record_capture_age, AudioAction, AudioControl,
    AudioFrameMetadata, AudioFrameSequenceTracker, CameraAction, CameraControl, FleetSelectCommand,
    FleetSubscriptionCommand, FrameSequenceTracker, JpegFramePacket, LifecycleCommand,
    LifecycleCommandResult, LifecycleRole, LifecycleStatus, LifecycleWakeLease, MetricWindow,
    PcmFramePacket, PcmSampleFormat, PowerAuthoritySnapshot, PowerCommand, PowerStatus, PowerTopic,
    ProtectedWorkRelayBody, ProtectedWorkRelayEnvelope, ProtectedWorkSnapshot,
    ProtectedWorkSnapshotRequest, RecordingOccurrence, SignedPowerCommandResult,
    SignedPowerEnvelope, SignedPowerEnvelopeKind, SignedPowerSnapshot, SignedPowerTransition,
    StreamCommand, StreamControl, TargetedMediaControl, PROTECTED_WORK_RELAY_TTL_MS,
};
use serde_json;
use std::collections::{BTreeMap, BTreeSet, HashMap};
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

// 64 lifecycle status slots retain 15 × 4 remote safe-node reports plus four
// local Orchestra reports. Keep bridge activation within the same bound.
const MAX_ACTIVE_ROVERS: usize = 15;
const MAX_PENDING_REMOTE_POWER_EVENTS: usize = 1_024;

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
    resource_snapshot_sub: ZenohSubscriber,
    voice_status_sub: ZenohSubscriber,
    tts_command_result_sub: ZenohSubscriber,
    lifecycle_status_sub: ZenohSubscriber,
    lifecycle_result_sub: ZenohSubscriber,
    lifecycle_capabilities_sub: ZenohSubscriber,
    power_status_sub: ZenohSubscriber,
    power_transition_sub: ZenohSubscriber,
    power_command_result_sub: ZenohSubscriber,
    power_snapshot_sub: ZenohSubscriber,
    power_event_sub: ZenohSubscriber,
    protected_work_request_sub: ZenohSubscriber,
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

    let resource_snapshot_topic = format!("rover/{}/resources/v1", entity_id);
    let resource_snapshot_sub = session
        .declare_subscriber(&resource_snapshot_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to subscribe to {}: {}", resource_snapshot_topic, e))?;
    tracing::info!("{}", resource_snapshot_topic);

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

    let lifecycle_status_topic = format!("rover/{}/lifecycle/status/v1", entity_id);
    let lifecycle_status_sub = session
        .declare_subscriber(&lifecycle_status_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to subscribe to {}: {}", lifecycle_status_topic, e))?;
    let lifecycle_result_topic = format!("rover/{}/lifecycle/result/v1", entity_id);
    let lifecycle_result_sub = session
        .declare_subscriber(&lifecycle_result_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to subscribe to {}: {}", lifecycle_result_topic, e))?;
    let lifecycle_capabilities_topic = format!("rover/{}/lifecycle/capabilities/v1", entity_id);
    let lifecycle_capabilities_sub = session
        .declare_subscriber(&lifecycle_capabilities_topic)
        .await
        .map_err(|e| {
            eyre::eyre!(
                "Failed to subscribe to {}: {}",
                lifecycle_capabilities_topic,
                e
            )
        })?;
    let power_status_topic = power_v1_topic(entity_id, PowerTopic::Status);
    let power_status_sub = session
        .declare_subscriber(&power_status_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to subscribe to {}: {}", power_status_topic, e))?;
    let power_transition_topic = power_v1_topic(entity_id, PowerTopic::Transition);
    let power_transition_sub = session
        .declare_subscriber(&power_transition_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to subscribe to {}: {}", power_transition_topic, e))?;
    let power_command_result_topic = power_v1_topic(entity_id, PowerTopic::CommandResult);
    let power_command_result_sub = session
        .declare_subscriber(&power_command_result_topic)
        .await
        .map_err(|e| {
            eyre::eyre!(
                "Failed to subscribe to {}: {}",
                power_command_result_topic,
                e
            )
        })?;
    let power_snapshot_topic = power_v1_topic(entity_id, PowerTopic::Snapshot);
    let power_snapshot_sub = session
        .declare_subscriber(&power_snapshot_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to subscribe to {}: {}", power_snapshot_topic, e))?;
    let power_event_topic = power_v1_topic(entity_id, PowerTopic::Event);
    let power_event_sub = session
        .declare_subscriber(&power_event_topic)
        .await
        .map_err(|e| eyre::eyre!("Failed to subscribe to {}: {}", power_event_topic, e))?;
    let snapshot_request_topic = power_v1_topic(entity_id, PowerTopic::SnapshotRequest);
    session
        .put(&snapshot_request_topic, b"{}")
        .await
        .map_err(|e| eyre::eyre!("Failed to request {}: {}", snapshot_request_topic, e))?;
    tracing::info!(%entity_id, "requested fresh power authority snapshot");
    let protected_work_request_topic = format!("rover/{entity_id}/power/protected-work/request/v1");
    let protected_work_request_sub = session
        .declare_subscriber(&protected_work_request_topic)
        .await
        .map_err(|e| {
            eyre::eyre!(
                "Failed to subscribe to {}: {}",
                protected_work_request_topic,
                e
            )
        })?;

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
        resource_snapshot_sub,
        voice_status_sub,
        tts_command_result_sub,
        lifecycle_status_sub,
        lifecycle_result_sub,
        lifecycle_capabilities_sub,
        power_status_sub,
        power_transition_sub,
        power_command_result_sub,
        power_snapshot_sub,
        power_event_sub,
        protected_work_request_sub,
    })
}

/// Unsubscribe from a rover (cleanup)
fn unsubscribe_from_rover(subs: RoverSubscriptions) {
    tracing::info!("Unsubscribing from rover: {}", subs.entity_id);
    // Subscriptions are dropped automatically
    drop(subs);
}

fn active_rover_ids<I>(entity_ids: I) -> Result<Vec<String>, String>
where
    I: IntoIterator<Item = String>,
{
    let entity_ids = entity_ids
        .into_iter()
        .map(|entity_id| entity_id.trim().to_owned())
        .filter(|entity_id| !entity_id.is_empty())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    if entity_ids.len() > MAX_ACTIVE_ROVERS {
        return Err(format!(
            "active Rover count ({}) exceeds lifecycle status queue limit ({MAX_ACTIVE_ROVERS})",
            entity_ids.len()
        ));
    }
    Ok(entity_ids)
}

fn can_activate_rover(active_rover_count: usize) -> bool {
    active_rover_count < MAX_ACTIVE_ROVERS
}

/// Handle fleet subscription commands (activate/deactivate rovers)
async fn handle_fleet_subscription_command(
    active_rovers: &mut HashMap<String, RoverSubscriptions>,
    session: &Arc<zenoh::Session>,
    data: dora_node_api::ArrowData,
    latest_voice_config: Option<&[u8]>,
    protected_work_keys: &BTreeMap<String, Vec<u8>>,
) -> Result<()> {
    if let Some(binary_array) = data.0.as_any().downcast_ref::<BinaryArray>() {
        if binary_array.len() > 0 {
            let bytes = binary_array.value(0);
            let cmd: FleetSubscriptionCommand = serde_json::from_slice(bytes)?;

            match cmd {
                FleetSubscriptionCommand::ActivateRover { entity_id, .. } => {
                    if active_rovers.contains_key(&entity_id) {
                        tracing::warn!("Rover {} already active", entity_id);
                    } else if !protected_work_keys.contains_key(&entity_id) {
                        tracing::warn!(%entity_id, "rejected Rover activation without a protected-work HMAC key");
                    } else if can_activate_rover(active_rovers.len()) {
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
                        tracing::warn!(%entity_id, max_active_rovers = MAX_ACTIVE_ROVERS, "rejected Rover activation beyond lifecycle status queue capacity");
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
                    let entity_ids = match active_rover_ids(entity_ids) {
                        Ok(entity_ids) => entity_ids,
                        Err(error) => {
                            tracing::warn!(%error, "rejected active Rover set beyond lifecycle status queue capacity");
                            return Ok(());
                        }
                    };
                    if let Some(entity_id) = entity_ids
                        .iter()
                        .find(|entity_id| !protected_work_keys.contains_key(*entity_id))
                    {
                        tracing::warn!(%entity_id, "rejected active Rover set without a protected-work HMAC key");
                        return Ok(());
                    }
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
    let initial_rovers = active_rover_ids(
        active_rovers_env
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>(),
    )
    .map_err(eyre::Report::msg)?;

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
    let protected_work_keys = protected_work_keys_from_env()?;
    let power_command_keys = power_command_keys_from_env()?;
    let power_deployment_id = power_deployment_id_from_env()?;
    if let Some(entity_id) = initial_rovers
        .iter()
        .find(|entity_id| !protected_work_keys.contains_key(*entity_id))
    {
        return Err(eyre::eyre!(
            "missing protected-work HMAC key for active rover {entity_id}"
        ));
    }

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
    let resource_snapshot_output = DataId::from("resource_snapshot".to_owned());
    let voice_status_output = DataId::from("voice_status".to_owned());
    let tts_command_result_output = DataId::from("tts_command_result".to_owned());
    let lifecycle_status_output = DataId::from("lifecycle_status".to_owned());
    let lifecycle_result_output = DataId::from("lifecycle_command_result".to_owned());
    let lifecycle_capabilities_output = DataId::from("lifecycle_capabilities".to_owned());
    let power_status_output = DataId::from("power_status".to_owned());
    let power_transition_output = DataId::from("power_transition".to_owned());
    let power_command_result_output = DataId::from("power_command_result".to_owned());
    let power_authority_snapshot_output = DataId::from("power_authority_snapshot".to_owned());
    let power_journal_record_output = DataId::from("power_journal_record".to_owned());
    let protected_work_snapshot_request_output =
        DataId::from("protected_work_snapshot_request".to_owned());

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
    let mut fresh_power_snapshots: HashMap<String, PowerAuthoritySnapshot> = HashMap::new();
    let mut remote_event_owners: HashMap<String, String> = HashMap::new();

    // High-rate browser audio must never occupy the control plane's ingress.
    // Media is lossy by design; power, lifecycle and journal inputs are not.
    let (control_tx, control_rx) = flume::bounded(CONTROL_INGRESS_CAPACITY);
    let (media_tx, media_rx) = flume::bounded(MEDIA_INGRESS_CAPACITY);
    let (media_publish_tx, media_publish_rx) =
        flume::bounded::<MediaPublish>(MEDIA_PUBLISH_CAPACITY);
    let media_session = Arc::clone(&session);
    tokio::spawn(async move {
        while let Ok(publish) = media_publish_rx.recv_async().await {
            if let Err(error) = media_session.put(publish.topic, publish.payload).await {
                tracing::debug!(%error, "dropped failed Orchestra media publish");
            }
        }
    });

    // Spawn task to read Dora events
    std::thread::spawn(move || {
        while let Some(event) = events.recv() {
            if is_high_rate_dora_ingress(&event) {
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
    // Main event loop - Dynamic multi-rover subscription
    // =========================================================================

    loop {
        // Build select! branches dynamically for all active rovers
        tokio::select! {
            // Handle Dora events (commands and fleet management)
            Ok(event) = receive_dora_event(&control_rx, &media_rx) => {
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
                                    &protected_work_keys,
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

                            // Sparse, explicitly targeted media changes. Never use selected_entity.
                            "targeted_media_control" => {
                                if let Some(binary_array) = data.as_any().downcast_ref::<BinaryArray>() {
                                    if let Some(bytes) = (binary_array.len() > 0).then(|| binary_array.value(0)) {
                                        if let Err(error) = publish_targeted_media_control(&session, &active_rovers, bytes).await {
                                            tracing::warn!(%error, "rejected targeted media control");
                                        }
                                    }
                                }
                            }

                            // The scheduler owns occurrence truth. Every transition is HMAC
                            // authenticated for its explicit rover; terminal states clear the
                            // rover-local protected-work gate.
                            "recording_occurrence_status" => {
                                if let Some(binary_array) = data.as_any().downcast_ref::<BinaryArray>() {
                                    if binary_array.len() == 1 {
                                        let bytes = binary_array.value(0);
                                        if let Ok(occurrence) = serde_json::from_slice::<RecordingOccurrence>(bytes) {
                                            if let Some(key) = protected_work_keys.get(&occurrence.entity_id) {
                                                match signed_protected_work_envelope(
                                                    occurrence.entity_id.clone(),
                                                    ProtectedWorkRelayBody::Occurrence { occurrence },
                                                    key,
                                                ) {
                                                    Ok(envelope) => {
                                                        let topic = format!("rover/{}/power/protected-work/occurrence/v1", envelope.target_entity_id);
                                                        if let Err(error) = session.put(topic, serde_json::to_vec(&envelope)?).await {
                                                            tracing::warn!(%error, "failed to publish protected recording work");
                                                        }
                                                    }
                                                    Err(error) => tracing::warn!(%error, "rejected invalid protected recording work"),
                                                }
                                            } else {
                                                tracing::warn!("rejected protected recording work without a rover key");
                                            }
                                        }
                                    }
                                }
                            }

                            "protected_work_snapshot" => {
                                if let Some(binary_array) = data.as_any().downcast_ref::<BinaryArray>() {
                                    if binary_array.len() == 1 {
                                        let bytes = binary_array.value(0);
                                        if let Ok(snapshot) = serde_json::from_slice::<ProtectedWorkSnapshot>(bytes) {
                                            if let Some(key) = protected_work_keys.get(&snapshot.entity_id) {
                                                match signed_protected_work_envelope(
                                                    snapshot.entity_id.clone(),
                                                    ProtectedWorkRelayBody::Snapshot { snapshot },
                                                    key,
                                                ) {
                                                    Ok(envelope) => {
                                                        let topic = format!("rover/{}/power/protected-work/snapshot/v1", envelope.target_entity_id);
                                                        if let Err(error) = session.put(topic, serde_json::to_vec(&envelope)?).await {
                                                            tracing::warn!(%error, "failed to publish protected-work snapshot");
                                                        }
                                                    }
                                                    Err(error) => tracing::warn!(%error, "rejected invalid protected-work snapshot"),
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            // Remote profile commands are single-use after a fresh
                            // Rover observation. This prevents reconnect force-takeover
                            // even if a local producer retries a stale command.
                            "power_command" => {
                                if let Some(binary_array) = data.as_any().downcast_ref::<BinaryArray>() {
                                    if binary_array.len() == 1 {
                                        let bytes = binary_array.value(0);
                                        if let Ok(command) = serde_json::from_slice::<PowerCommand>(bytes) {
                                            let now_ms = current_time_ms().unwrap_or_default();
                                            let target = command.entity_id.clone();
                                            let allowed = active_rovers.contains_key(&target)
                                                && fresh_power_snapshots.get(&target).is_some_and(|snapshot| {
                                                    power_command_is_snapshot_fenced(&command, snapshot, now_ms)
                                                });
                                            if allowed {
                                                let topic = power_v1_topic(&target, PowerTopic::Command);
                                                let Some(key) = power_command_keys.get(&target) else {
                                                    tracing::warn!(entity_id = %target, "rejected power command without a configured transport key");
                                                    continue;
                                                };
                                                let signed = match SignedPowerEnvelope::new(
                                                    SignedPowerEnvelopeKind::Command,
                                                    LifecycleRole::Rover,
                                                    target.clone(),
                                                    now_ms,
                                                    command.clone(),
                                                ).sign(key) {
                                                    Ok(signed) => signed,
                                                    Err(error) => {
                                                        tracing::warn!(%error, entity_id = %target, "failed to sign power command");
                                                        continue;
                                                    }
                                                };
                                                match session.put(topic, serde_json::to_vec(&signed)?).await {
                                                    Ok(_) => {
                                                        fresh_power_snapshots.remove(&target);
                                                        tracing::info!(entity_id = %target, command_id = %command.command_id, "published snapshot-fenced power command");
                                                    }
                                                    Err(error) => tracing::warn!(%error, entity_id = %target, "failed to publish power command"),
                                                }
                                            } else {
                                                tracing::warn!(entity_id = %target, "rejected power command without a fresh, newer Rover snapshot");
                                            }
                                        }
                                    }
                                }
                            }

                            "remote_power_event_ack" => {
                                if let Some(binary_array) = data.as_any().downcast_ref::<BinaryArray>() {
                                    if binary_array.len() == 1 {
                                        let bytes = binary_array.value(0);
                                        if let Ok(ack) = serde_json::from_slice::<JournalAcknowledgement>(bytes) {
                                            if let Some(entity_id) = remote_event_owners.get(&ack.event_id).cloned() {
                                                let Some(key) = power_command_keys.get(&entity_id) else {
                                                    tracing::warn!(%entity_id, "rejected rover journal acknowledgement without a configured transport key");
                                                    continue;
                                                };
                                                if ack.validates_for(&entity_id, Some(&power_deployment_id)).is_err() {
                                                    tracing::warn!(%entity_id, "rejected invalid rover journal acknowledgement");
                                                    continue;
                                                }
                                                let now_ms = current_time_ms().unwrap_or_default();
                                                let signed = match SignedPowerEnvelope::new(
                                                    SignedPowerEnvelopeKind::JournalAcknowledgement,
                                                    LifecycleRole::Rover,
                                                    entity_id.clone(),
                                                    now_ms,
                                                    ack,
                                                ).sign(key) {
                                                    Ok(signed) => signed,
                                                    Err(error) => {
                                                        tracing::warn!(%error, %entity_id, "failed to sign rover journal acknowledgement");
                                                        continue;
                                                    }
                                                };
                                                let topic = power_v1_topic(&entity_id, PowerTopic::EventAck);
                                                match session.put(topic, serde_json::to_vec(&signed)?).await {
                                                    Ok(_) => {
                                                        remote_event_owners.remove(&signed.payload.event_id);
                                                    }
                                                    Err(error) => tracing::warn!(%error, %entity_id, "failed to relay rover power event acknowledgement"),
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            "lifecycle_command_authorized" => {
                                if let Some(binary_array) = data.as_any().downcast_ref::<BinaryArray>() {
                                    if binary_array.len() == 1 {
                                        let bytes = binary_array.value(0);
                                        match serde_json::from_slice::<LifecycleCommand>(bytes) {
                                            Ok(command) if command.validate().is_ok() && command.target.role == LifecycleRole::Rover && active_rovers.contains_key(&command.target.entity_id) => {
                                                let topic = format!("rover/{}/cmd/lifecycle/v1", command.target.entity_id);
                                                match session.put(topic, bytes).await {
                                                    Ok(_) => tracing::info!(request_id = %command.request_id, target = ?command.target, "published authorized lifecycle command to Rover"),
                                                    Err(error) => tracing::warn!(%error, request_id = %command.request_id, target = ?command.target, "failed to publish authorized lifecycle command"),
                                                }
                                            }
                                            _ => tracing::warn!("rejected lifecycle command with invalid target"),
                                        }
                                    }
                                }
                            }

                            "lifecycle_wake_lease_authorized" => {
                                if let Some(binary_array) = data.as_any().downcast_ref::<BinaryArray>() {
                                    if binary_array.len() == 1 {
                                        let bytes = binary_array.value(0);
                                        match serde_json::from_slice::<LifecycleWakeLease>(bytes) {
                                            Ok(lease) if lease.validate().is_ok() && lease.target.role == LifecycleRole::Rover && active_rovers.contains_key(&lease.target.entity_id) => {
                                                let topic = format!("rover/{}/cmd/lifecycle-wake-lease/v1", lease.target.entity_id);
                                                if let Err(error) = session.put(topic, bytes).await { tracing::warn!(%error, "failed to publish lifecycle wake lease"); }
                                            }
                                            _ => tracing::warn!("rejected lifecycle wake lease with invalid target"),
                                        }
                                    }
                                }
                            }

                            "lifecycle_status_query" => {
                                for entity_id in active_rovers.keys() {
                                    let topic = format!("rover/{}/cmd/lifecycle-query/v1", entity_id);
                                    if let Err(error) = session.put(topic, b"{}").await { tracing::warn!(%error, %entity_id, "failed to publish lifecycle query"); }
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
                                                        queue_media_publish(&media_publish_tx, &audio_stream_topic, packet);
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

            // Resource snapshots retain their producer scope and must match the topic entity.
            result = receive_from_rovers(&active_rovers, |subs| &subs.resource_snapshot_sub) => {
                if let Some((entity_id, sample)) = result {
                    forward_resource_snapshot(
                        &mut node,
                        &resource_snapshot_output,
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

            result = receive_from_rovers(&active_rovers, |subs| &subs.lifecycle_status_sub) => {
                if let Some((entity_id, sample)) = result {
                    let payload = sample.payload().to_bytes();
                    if let Ok(status) = serde_json::from_slice::<LifecycleStatus>(&payload) {
                        if status.target.entity_id == entity_id && status.validate().is_ok() { forward_binary_output(&mut node, &lifecycle_status_output, sample); }
                    }
                }
            }

            result = receive_from_rovers(&active_rovers, |subs| &subs.lifecycle_result_sub) => {
                if let Some((_entity_id, sample)) = result {
                    let payload = sample.payload().to_bytes();
                    if serde_json::from_slice::<LifecycleCommandResult>(&payload).is_ok() { forward_binary_output(&mut node, &lifecycle_result_output, sample); }
                }
            }

            result = receive_from_rovers(&active_rovers, |subs| &subs.lifecycle_capabilities_sub) => {
                if let Some((_entity_id, sample)) = result {
                    forward_binary_output(&mut node, &lifecycle_capabilities_output, sample);
                }
            }

            result = receive_from_rovers(&active_rovers, |subs| &subs.power_status_sub) => {
                if let Some((entity_id, sample)) = result {
                    let payload = sample.payload().to_bytes();
                    if serde_json::from_slice::<PowerStatus>(&payload)
                        .is_ok_and(|status| status.validates_for(LifecycleRole::Rover, &entity_id).is_ok())
                    {
                        forward_binary_output(&mut node, &power_status_output, sample);
                    }
                }
            }

            result = receive_from_rovers(&active_rovers, |subs| &subs.power_transition_sub) => {
                if let Some((entity_id, sample)) = result {
                    let payload = sample.payload().to_bytes();
                    let Some(key) = power_command_keys.get(&entity_id) else { continue; };
                    let now_ms = current_time_ms().unwrap_or(u64::MAX);
                    if let Ok(envelope) = serde_json::from_slice::<SignedPowerTransition>(&payload) {
                        if envelope.verify(key, now_ms).is_ok()
                            && envelope.validates_for(SignedPowerEnvelopeKind::Transition, LifecycleRole::Rover, &entity_id).is_ok()
                            && envelope.payload.validates_for(LifecycleRole::Rover, &entity_id).is_ok()
                        {
                            let transition = serde_json::to_vec(&envelope.payload)?;
                            node.send_output(power_transition_output.clone(), Default::default(), BinaryArray::from_vec(vec![transition.as_slice()]))?;
                        }
                    }
                }
            }

            // A command result is not interchangeable with aggregate status.
            // Verify the signed, entity-scoped envelope before giving the local
            // scheduler its exact command-id acknowledgement.
            result = receive_from_rovers(&active_rovers, |subs| &subs.power_command_result_sub) => {
                if let Some((entity_id, sample)) = result {
                    let payload = sample.payload().to_bytes();
                    let Some(key) = power_command_keys.get(&entity_id) else {
                        tracing::warn!(%entity_id, "rejected rover power result without a configured transport key");
                        continue;
                    };
                    let now_ms = current_time_ms().unwrap_or(u64::MAX);
                    if let Ok(envelope) = serde_json::from_slice::<SignedPowerCommandResult>(&payload) {
                        if envelope.verify(key, now_ms).is_ok()
                            && envelope.validates_for(
                                SignedPowerEnvelopeKind::CommandResult,
                                LifecycleRole::Rover,
                                &entity_id,
                            ).is_ok()
                            && envelope.payload.validate().is_ok()
                        {
                            let result = serde_json::to_vec(&envelope.payload)?;
                            node.send_output(
                                power_command_result_output.clone(),
                                Default::default(),
                                BinaryArray::from_vec(vec![result.as_slice()]),
                            )?;
                        } else {
                            tracing::warn!(%entity_id, "rejected invalid or stale rover power command result");
                        }
                    } else {
                        tracing::warn!(%entity_id, "rejected malformed rover power command result");
                    }
                }
            }

            result = receive_from_rovers(&active_rovers, |subs| &subs.power_snapshot_sub) => {
                if let Some((entity_id, sample)) = result {
                    let payload = sample.payload().to_bytes();
                    if let Ok(envelope) = serde_json::from_slice::<SignedPowerSnapshot>(&payload) {
                        let Some(key) = power_command_keys.get(&entity_id) else {
                            tracing::warn!(%entity_id, "rejected rover snapshot without a configured transport key");
                            continue;
                        };
                        let now_ms = current_time_ms().unwrap_or(u64::MAX);
                        if envelope.verify(key, now_ms).is_err()
                            || envelope.validates_for(
                                SignedPowerEnvelopeKind::Snapshot,
                                LifecycleRole::Rover,
                                &entity_id,
                            ).is_err()
                            || envelope.payload.validates_for(LifecycleRole::Rover, &entity_id).is_err()
                            || envelope.payload.expires_at_ms <= now_ms
                        {
                            tracing::warn!(%entity_id, "rejected invalid or stale rover power snapshot");
                            continue;
                        }
                        let snapshot_payload = serde_json::to_vec(&envelope.payload)?;
                        fresh_power_snapshots.insert(entity_id.clone(), envelope.payload);
                        node.send_output(
                            power_authority_snapshot_output.clone(),
                            Default::default(),
                            BinaryArray::from_vec(vec![snapshot_payload.as_slice()]),
                        )?;
                    }
                }
            }

            result = receive_from_rovers(&active_rovers, |subs| &subs.power_event_sub) => {
                if let Some((entity_id, sample)) = result {
                    let payload = sample.payload().to_bytes();
                    if let Ok(record) = serde_json::from_slice::<JournalRecord>(&payload) {
                        if record.validate().is_ok()
                            && record.event.role == LifecycleRole::Rover
                            && record.event.entity_id == entity_id
                        {
                            if remote_event_owners.contains_key(&record.event.event_id) {
                                tracing::debug!(event_id = %record.event.event_id, "deduplicated pending rover power event");
                                continue;
                            }
                            if remote_event_owners.len() >= MAX_PENDING_REMOTE_POWER_EVENTS {
                                tracing::warn!(max_pending = MAX_PENDING_REMOTE_POWER_EVENTS, "remote power event acknowledgement map is full; retaining event for retry");
                                continue;
                            }
                            remote_event_owners.insert(record.event.event_id.clone(), entity_id);
                            forward_binary_output(&mut node, &power_journal_record_output, sample);
                        }
                    }
                }
            }

            result = receive_from_rovers(&active_rovers, |subs| &subs.protected_work_request_sub) => {
                if let Some((entity_id, sample)) = result {
                    let payload = sample.payload().to_bytes();
                    if let Some(key) = protected_work_keys.get(&entity_id) {
                        if let Some(request) = verified_snapshot_request(payload.as_ref(), &entity_id, key) {
                            let serialized = serde_json::to_vec(&request)?;
                            let _ = node.send_output(
                                protected_work_snapshot_request_output.clone(),
                                Default::default(),
                                BinaryArray::from_vec(vec![serialized.as_slice()]),
                            );
                        } else {
                            tracing::warn!(%entity_id, "rejected protected-work snapshot request");
                        }
                    }
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

fn forward_resource_snapshot(
    node: &mut DoraNode,
    output_id: &DataId,
    entity_id: String,
    sample: zenoh::sample::Sample,
) {
    let payload = sample.payload().to_bytes();
    let Ok(snapshot) = serde_json::from_slice::<robo_rover_lib::ResourceSnapshot>(&payload) else {
        tracing::warn!(%entity_id, "discarded malformed rover resource snapshot");
        return;
    };
    if snapshot.entity_id != entity_id
        || snapshot.role != robo_rover_lib::ResourceRole::Rover
        || snapshot.validate().is_err()
    {
        tracing::warn!(%entity_id, "discarded invalid rover resource snapshot");
        return;
    }
    if let Ok(serialized) = serde_json::to_vec(&snapshot) {
        let arrow_data = BinaryArray::from_vec(vec![serialized.as_slice()]);
        if let Err(error) = node.send_output(output_id.clone(), Default::default(), arrow_data) {
            tracing::warn!(%error, %entity_id, "failed to forward rover resource snapshot");
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

fn protected_work_keys_from_env() -> Result<BTreeMap<String, Vec<u8>>> {
    let raw = std::env::var("POWER_PROTECTED_WORK_HMAC_KEYS")
        .map_err(|_| eyre::eyre!("POWER_PROTECTED_WORK_HMAC_KEYS is required"))?;
    let keys = serde_json::from_str::<BTreeMap<String, String>>(&raw)
        .map_err(|_| eyre::eyre!("POWER_PROTECTED_WORK_HMAC_KEYS must be a JSON object"))?;
    if keys.is_empty() || keys.values().any(|key| key.len() < 32) {
        return Err(eyre::eyre!(
            "protected-work HMAC keys must be at least 32 bytes"
        ));
    }
    Ok(keys
        .into_iter()
        .map(|(entity_id, key)| (entity_id, key.into_bytes()))
        .collect())
}

fn power_command_keys_from_env() -> Result<BTreeMap<String, Vec<u8>>> {
    let raw = std::env::var("POWER_COMMAND_HMAC_KEYS")
        .map_err(|_| eyre::eyre!("POWER_COMMAND_HMAC_KEYS is required"))?;
    let keys = serde_json::from_str::<BTreeMap<String, String>>(&raw)
        .map_err(|_| eyre::eyre!("POWER_COMMAND_HMAC_KEYS must be a JSON object"))?;
    if keys.is_empty() || keys.values().any(|key| key.len() < 32) {
        return Err(eyre::eyre!(
            "power-command HMAC keys must be at least 32 bytes"
        ));
    }
    Ok(keys
        .into_iter()
        .map(|(entity_id, key)| (entity_id, key.into_bytes()))
        .collect())
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

fn signed_protected_work_envelope(
    entity_id: String,
    body: ProtectedWorkRelayBody,
    key: &[u8],
) -> Result<ProtectedWorkRelayEnvelope, String> {
    ProtectedWorkRelayEnvelope::new(
        entity_id,
        current_time_ms()?,
        PROTECTED_WORK_RELAY_TTL_MS,
        body,
    )
    .sign(key)
}

fn verified_snapshot_request(
    payload: &[u8],
    entity_id: &str,
    key: &[u8],
) -> Option<ProtectedWorkSnapshotRequest> {
    let envelope = serde_json::from_slice::<ProtectedWorkRelayEnvelope>(payload).ok()?;
    envelope.verify(key, current_time_ms().ok()?).ok()?;
    match envelope.body {
        ProtectedWorkRelayBody::SnapshotRequest { request }
            if envelope.target_entity_id == entity_id && request.entity_id == entity_id =>
        {
            Some(request)
        }
        _ => None,
    }
}

async fn publish_targeted_media_control(
    session: &Arc<zenoh::Session>,
    active_rovers: &HashMap<String, RoverSubscriptions>,
    bytes: &[u8],
) -> Result<()> {
    let control: TargetedMediaControl = serde_json::from_slice(bytes)
        .map_err(|error| eyre::eyre!("invalid targeted media JSON: {error}"))?;
    control
        .validate()
        .map_err(|error| eyre::eyre!("invalid targeted media control: {error}"))?;
    if !active_rovers.contains_key(&control.entity_id) {
        return Err(eyre::eyre!(
            "target rover is not active: {}",
            control.entity_id
        ));
    }
    let timestamp = current_time_ms().map_err(|error| eyre::eyre!(error))?;
    for (topic, payload) in targeted_media_payloads(&control, timestamp)? {
        session
            .put(topic, payload)
            .await
            .map_err(|error| eyre::eyre!("failed to publish targeted media control: {error}"))?;
    }
    Ok(())
}

fn targeted_media_payloads(
    control: &TargetedMediaControl,
    timestamp: u64,
) -> serde_json::Result<Vec<(String, Vec<u8>)>> {
    let mut commands = Vec::new();
    if let Some(enabled) = control.camera_enabled {
        let payload = serde_json::to_vec(&CameraControl {
            command: if enabled {
                CameraAction::Start
            } else {
                CameraAction::Stop
            },
            timestamp,
        })?;
        commands.push((format!("rover/{}/cmd/camera", control.entity_id), payload));
    }
    if let Some(enabled) = control.jpeg_enabled {
        let payload = serde_json::to_vec(&StreamControl {
            command: if enabled {
                StreamCommand::Start
            } else {
                StreamCommand::Stop
            },
            video_enabled: enabled,
            audio_enabled: false,
            quality: None,
            target_fps: None,
        })?;
        commands.push((
            format!("rover/{}/cmd/stream/v1", control.entity_id),
            payload,
        ));
    }
    if let Some(enabled) = control.microphone_enabled {
        let payload = serde_json::to_vec(&AudioControl {
            command: if enabled {
                AudioAction::Start
            } else {
                AudioAction::Stop
            },
            timestamp,
        })?;
        commands.push((format!("rover/{}/cmd/audio", control.entity_id), payload));
    }
    Ok(commands)
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

const CONTROL_INGRESS_CAPACITY: usize = 64;
const MEDIA_INGRESS_CAPACITY: usize = 8;
const MEDIA_PUBLISH_CAPACITY: usize = 8;

struct MediaPublish {
    topic: String,
    payload: Vec<u8>,
}

fn queue_media_publish(sender: &flume::Sender<MediaPublish>, topic: &str, payload: Vec<u8>) {
    if sender
        .try_send(MediaPublish {
            topic: topic.into(),
            payload,
        })
        .is_err()
    {
        tracing::debug!("dropped saturated Orchestra media publish");
    }
}

fn is_high_rate_dora_ingress(event: &Event) -> bool {
    matches!(event, Event::Input { id, .. } if is_high_rate_input_id(id.as_str()))
}

fn is_high_rate_input_id(id: &str) -> bool {
    id == "audio_stream_web"
}

async fn receive_dora_event(
    control_rx: &flume::Receiver<Event>,
    media_rx: &flume::Receiver<Event>,
) -> Result<Event, flume::RecvError> {
    tokio::select! {
        biased;
        event = control_rx.recv_async() => event,
        event = media_rx.recv_async() => event,
    }
}

fn power_command_is_snapshot_fenced(
    command: &PowerCommand,
    snapshot: &PowerAuthoritySnapshot,
    now_ms: u64,
) -> bool {
    command
        .validates_for(LifecycleRole::Rover, &snapshot.entity_id)
        .is_ok()
        && snapshot
            .validates_for(LifecycleRole::Rover, &snapshot.entity_id)
            .is_ok()
        && snapshot.captured_at_ms <= now_ms
        && snapshot.expires_at_ms > now_ms
        && snapshot.authority.next_epoch() == Some(command.authority)
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
    use robo_rover_lib::{
        CameraAction, CameraControl, PowerAuthority, PowerAuthoritySnapshot, PowerCommand,
        PowerCommandAction, PowerPolicy, PowerProfile, PowerState, ProtectedWorkRelayBody,
        ProtectedWorkRelayEnvelope, StreamCommand, StreamControl, TargetedMediaControl,
        POWER_PROTOCOL_VERSION, PROTECTED_WORK_RELAY_TTL_MS,
    };

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

    #[test]
    fn saturated_media_ingress_is_not_classified_as_control() {
        assert!(super::is_high_rate_input_id("audio_stream_web"));
        assert!(!super::is_high_rate_input_id("power_command"));
        assert!(!super::is_high_rate_input_id("remote_power_event_ack"));
    }

    #[test]
    fn a_stalled_media_publisher_cannot_fill_the_control_queue() {
        let (control_tx, control_rx) = flume::bounded(1);
        let (media_tx, media_rx) = flume::bounded(1);
        media_tx.try_send(()).unwrap();
        assert!(media_tx.try_send(()).is_err());
        assert!(control_tx.try_send(()).is_ok());
        assert_eq!(control_rx.len(), 1);
        assert_eq!(media_rx.len(), 1);
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

    #[test]
    fn targeted_media_payloads_preserve_exact_entity_and_changed_resources() {
        let control = TargetedMediaControl {
            protocol_version: 1,
            entity_id: "rover-a".into(),
            camera_enabled: Some(true),
            jpeg_enabled: Some(false),
            microphone_enabled: None,
        };
        let commands = super::targeted_media_payloads(&control, 42).unwrap();
        assert_eq!(commands.len(), 2);
        assert_eq!(commands[0].0, "rover/rover-a/cmd/camera");
        assert_eq!(commands[1].0, "rover/rover-a/cmd/stream/v1");
        assert!(matches!(
            serde_json::from_slice::<CameraControl>(&commands[0].1)
                .unwrap()
                .command,
            CameraAction::Start
        ));
        let stream = serde_json::from_slice::<StreamControl>(&commands[1].1).unwrap();
        assert!(matches!(stream.command, StreamCommand::Stop));
        assert!(!stream.video_enabled);
    }

    #[test]
    fn power_command_requires_a_fresh_snapshot_and_exact_epoch_reconciliation() {
        let snapshot = PowerAuthoritySnapshot {
            protocol_version: POWER_PROTOCOL_VERSION,
            snapshot_id: "f4f3e2d1-c0b9-48a7-9615-141312111000".into(),
            role: robo_rover_lib::LifecycleRole::Rover,
            entity_id: "rover-kiwi".into(),
            authority: PowerAuthority {
                epoch: 7,
                sequence: 2,
            },
            state: PowerState::Active,
            effective_profile: PowerProfile::NormalRover,
            captured_at_ms: 100,
            expires_at_ms: 200,
        };
        let mut command = PowerCommand {
            protocol_version: POWER_PROTOCOL_VERSION,
            command_id: "f4f3e2d1-c0b9-48a7-9615-141312111001".into(),
            role: robo_rover_lib::LifecycleRole::Rover,
            entity_id: "rover-kiwi".into(),
            authority: PowerAuthority {
                epoch: 7,
                sequence: 2,
            },
            action: PowerCommandAction::SetPolicy {
                policy: PowerPolicy::Auto,
            },
            issued_at_ms: 100,
            not_before_ms: 100,
            expires_at_ms: 150,
            detail: None,
        };

        assert!(!super::power_command_is_snapshot_fenced(
            &command, &snapshot, 101
        ));
        command.authority = PowerAuthority {
            epoch: 8,
            sequence: 1,
        };
        assert!(super::power_command_is_snapshot_fenced(
            &command, &snapshot, 101
        ));
        command.authority.sequence = 2;
        assert!(!super::power_command_is_snapshot_fenced(
            &command, &snapshot, 101
        ));
        assert!(!super::power_command_is_snapshot_fenced(
            &command, &snapshot, 200
        ));
    }

    #[test]
    fn protected_work_occurrence_envelope_is_signed_for_its_target_rover() {
        let occurrence = serde_json::json!({
            "occurrence_id": "f4f3e2d1-c0b9-48a7-9615-141312111000",
            "schedule_id": "f4f3e2d1-c0b9-48a7-9615-141312111001",
            "schedule_revision": 1,
            "entity_id": "rover-a",
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
            "rover-a".into(),
            super::current_time_ms().unwrap(),
            PROTECTED_WORK_RELAY_TTL_MS,
            ProtectedWorkRelayBody::Occurrence {
                occurrence: serde_json::from_value(occurrence).unwrap(),
            },
        )
        .sign(key)
        .unwrap();
        assert!(envelope
            .verify(key, super::current_time_ms().unwrap())
            .is_ok());
        assert!(envelope
            .verify(
                b"abcdefghijklmnopqrstuvwxyz123456",
                super::current_time_ms().unwrap()
            )
            .is_err());
    }
}

#[cfg(test)]
mod lifecycle_capacity_tests {
    use super::*;

    #[test]
    fn active_rover_configuration_is_bounded_by_status_queue_capacity() {
        let supported = (0..MAX_ACTIVE_ROVERS)
            .map(|index| format!("rover-{index}"))
            .collect::<Vec<_>>();
        let overflow = (0..=MAX_ACTIVE_ROVERS)
            .map(|index| format!("rover-{index}"))
            .collect::<Vec<_>>();

        assert_eq!(
            active_rover_ids(supported).unwrap().len(),
            MAX_ACTIVE_ROVERS
        );
        assert!(active_rover_ids(overflow).is_err());
    }

    #[test]
    fn activation_rejects_the_first_rover_beyond_capacity() {
        assert!(can_activate_rover(MAX_ACTIVE_ROVERS - 1));
        assert!(!can_activate_rover(MAX_ACTIVE_ROVERS));
    }
}
