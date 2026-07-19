use dora_node_api::{
    arrow::array::{Array, BinaryArray, Float32Array},
    dora_core::config::DataId,
    DoraNode, Event,
};
use eyre::Result;
use robo_rover_lib::types::{
    ActiveRoversStatus, DetectionFrame, FleetSelectCommand, FleetStatus, FleetSubscriptionCommand,
    RecordingClipQuery, RecordingDeleteRequest, RecordingDeleteResult,
    RecordingPlaybackTicketRequest, RecordingSessionAction, RecordingSessionCommand,
    RecordingSessionState, SpeechTranscription, SttStatus, SystemMetrics, TrackingCommand,
    TrackingTelemetry,
};
use robo_rover_lib::{
    capture_age_ms, init_tracing, record_capture_age, ArmCommand, ArmCommandWithMetadata,
    AudioAction, AudioControl, AudioFrameMetadata, AudioFrameSequenceTracker, CameraAction,
    CameraControl, CommandMetadata, CommandPriority, FrameSequenceTracker, InputSource,
    MetricWindow, PcmSampleFormat, RecordingClipQueryResult, RecordingReasonCode,
    RecordingSessionCommandResult, RecordingSessionStatus, RoverCommand, RoverCommandWithMetadata,
    StreamControl, TargetedMediaControl, TtsAckState, TtsCommand, TtsCommandAck, TtsCommandResult,
    TtsConfigState, TtsConfigUpdate, TtsLanguage, TtsRuntimeConfig, VoiceReasonCode, VoiceStatus,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use uuid;

use axum::http::{HeaderValue, Method};
use axum::{body::Body, middleware, response::Response};
use log::info;
use mongodb::Collection;
use serde_json::Value;
use socketioxide::{
    extract::{Bin, Data, SocketRef, TryData},
    SocketIo,
};
use std::env;
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;

mod security;
use security::{
    extract_client_ip, load_or_generate_jwt_secret, log_auth_attempt, log_rate_limit_exceeded,
    log_validation_error, parse_allowed_origins, warn_http_origins, AuthRateLimiter,
    CommandRateLimiter, IpRateLimiter, SessionRegistry,
};

mod audio_counters;
use audio_counters::AudioDeliveryCounters;

#[path = "recording-access.rs"]
mod recording_access;
#[path = "recording-playback.rs"]
mod recording_playback;
#[path = "recording-socket.rs"]
mod recording_socket;
use recording_access::RecordingAccess;
use recording_socket::{PendingRequest, RecordingState, RequestKind};

#[path = "media-demand-registry.rs"]
mod media_demand_registry;
use media_demand_registry::{MediaDemandRegistry, MediaDemandTransition, MediaResource};

mod stt_bridge;
mod stt_ingress;
mod stt_protocol;
mod stt_socket_delivery;
mod stt_stream_registry;
mod stt_stream_state;
mod stt_transcript_routing;
mod voice_runtime;
#[path = "walkie-audio.rs"]
mod walkie_audio;
use stt_bridge::{SttBridge, SttBridgeConfig, TranscriptRoute};
use stt_protocol::{send_dora_message, SttOutputIds, VoiceCommandAudioFrame, VoiceCommandControl};
use stt_socket_delivery::{emit_authenticated, AUTHENTICATED_ROOM};
use voice_runtime::{ConfigUpdateOutcome, VoiceRuntimeState};
use walkie_audio::{WalkieAudioFrameMetadata, WalkieIngress};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct JointPositions {
    pub shoulder_pan: f64,
    pub shoulder_lift: f64,
    pub elbow_flex: f64,
    pub wrist_flex: f64,
    pub wrist_roll: f64,
    pub gripper: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WebArmCommand {
    pub command_type: String, // "joint_position", "cartesian", "home", "stop"
    pub joint_positions: Option<JointPositions>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WebRoverCommand {
    pub command_type: String,
    pub wheel1: Option<f64>,
    pub wheel2: Option<f64>,
    pub wheel3: Option<f64>,
    pub wheel4: Option<f64>,
}

// Client state for video and audio streaming
#[derive(Clone)]
struct ClientState {
    socket_id: String,
    video_enabled: bool,
    audio_enabled: bool,
    target_fps: u8,
    last_video_sent: Arc<Mutex<SystemTime>>,
    last_audio_sent: Arc<Mutex<SystemTime>>,
    last_activity: Arc<Mutex<Instant>>,
    video_frames_sent: Arc<Mutex<u64>>,
    audio_frames_sent: Arc<Mutex<u64>>,
    audio_frames_dropped: Arc<Mutex<u64>>,
    frames_dropped: Arc<Mutex<u64>>,
}

impl ClientState {
    fn new(socket_id: String) -> Self {
        Self {
            socket_id,
            video_enabled: false,
            audio_enabled: true,
            target_fps: 15,
            last_video_sent: Arc::new(Mutex::new(SystemTime::now())),
            last_audio_sent: Arc::new(Mutex::new(SystemTime::now())),
            last_activity: Arc::new(Mutex::new(Instant::now())),
            video_frames_sent: Arc::new(Mutex::new(0)),
            audio_frames_sent: Arc::new(Mutex::new(0)),
            audio_frames_dropped: Arc::new(Mutex::new(0)),
            frames_dropped: Arc::new(Mutex::new(0)),
        }
    }

    fn should_send_video(&self) -> bool {
        if !self.video_enabled {
            return false;
        }

        let last_sent = self.last_video_sent.lock().unwrap();
        let elapsed = last_sent.elapsed().unwrap_or(Duration::from_secs(1));
        let min_interval = Duration::from_millis((1000 / self.target_fps as u64).max(1));

        elapsed >= min_interval
    }

    fn mark_video_sent(&self) {
        *self.last_video_sent.lock().unwrap() = SystemTime::now();
        *self.video_frames_sent.lock().unwrap() += 1;
    }

    fn should_send_audio(&self) -> bool {
        if !self.audio_enabled {
            return false;
        }
        // Audio is less frequent, so we send every frame
        true
    }

    fn mark_audio_sent(&self) {
        *self.last_audio_sent.lock().unwrap() = SystemTime::now();
        *self.audio_frames_sent.lock().unwrap() += 1;
    }

    fn mark_frame_dropped(&self) {
        *self.frames_dropped.lock().unwrap() += 1;
    }

    fn mark_audio_dropped(&self) {
        *self.audio_frames_dropped.lock().unwrap() += 1;
    }
}

fn record_audio_error(metrics: &mut MetricWindow, total_errors: &mut u64) {
    metrics.record_error();
    *total_errors = total_errors.saturating_add(1);
}

#[derive(Default)]
struct AudioDeliveryErrorCounts {
    input: u64,
    socket_id: u64,
    socket_missing: u64,
    routing: u64,
    emit: u64,
}

impl AudioDeliveryErrorCounts {
    fn total(&self) -> u64 {
        self.input
            .saturating_add(self.socket_id)
            .saturating_add(self.socket_missing)
            .saturating_add(self.routing)
            .saturating_add(self.emit)
    }
}

fn touch_activity(clients: &Mutex<Vec<ClientState>>, socket_id: &str) {
    if let Ok(clients) = clients.lock() {
        if let Some(client) = clients.iter().find(|c| c.socket_id == socket_id) {
            *client.last_activity.lock().unwrap() = Instant::now();
        }
    }
}

fn browser_consumer_prefix(socket_id: &str) -> String {
    format!("browser:{socket_id}:")
}

fn browser_consumer_id(socket_id: &str, intent: &str) -> String {
    format!("{}{}", browser_consumer_prefix(socket_id), intent)
}

fn enqueue_media_transitions(
    queue: &Mutex<Vec<TargetedMediaControl>>,
    transitions: impl IntoIterator<Item = MediaDemandTransition>,
) {
    if let Ok(mut queue) = queue.lock() {
        queue.extend(
            transitions
                .into_iter()
                .map(|transition| transition.targeted_control()),
        );
    }
}

fn selected_browser_target(shared_state: &SharedState) -> Option<String> {
    let selected = shared_state
        .fleet_status
        .lock()
        .ok()?
        .selected_entity
        .clone();
    shared_state
        .active_rovers_status
        .lock()
        .ok()?
        .active_rovers
        .contains(&selected)
        .then_some(selected)
}

fn active_rover_status_includes(status: &ActiveRoversStatus, entity_id: &str) -> bool {
    status
        .active_rovers
        .iter()
        .any(|active| active == entity_id)
}

fn set_browser_media_demand(
    shared_state: &SharedState,
    socket_id: &str,
    intent: &str,
    resource: MediaResource,
    enabled: bool,
) {
    let consumer_id = browser_consumer_id(socket_id, intent);
    let transitions = shared_state
        .media_demand_registry
        .lock()
        .ok()
        .and_then(|mut registry| {
            let transition = if enabled {
                let Some(entity_id) = selected_browser_target(shared_state) else {
                    tracing::warn!(
                        socket_id,
                        "rejected browser media demand without an active selected rover"
                    );
                    return None;
                };
                registry.acquire(entity_id, consumer_id, resource)
            } else {
                registry.release_consumer_resource(&consumer_id, resource)
            };
            transition.map(|transition| vec![transition])
        })
        .unwrap_or_default();
    enqueue_media_transitions(&shared_state.targeted_media_control_queue, transitions);
}

fn move_browser_media_demand(shared_state: &SharedState, socket_id: &str, entity_id: &str) {
    let transitions = shared_state
        .media_demand_registry
        .lock()
        .ok()
        .map(|mut registry| {
            registry.move_consumer_prefix(&browser_consumer_prefix(socket_id), entity_id)
        })
        .unwrap_or_default();
    enqueue_media_transitions(&shared_state.targeted_media_control_queue, transitions);
}

fn remove_browser_media_demand(shared_state: &SharedState, socket_id: &str) {
    if let Ok(mut clients) = shared_state.video_clients.lock() {
        clients.retain(|client| client.socket_id != socket_id);
    }
    let transitions = shared_state
        .media_demand_registry
        .lock()
        .ok()
        .map(|mut registry| registry.release_consumer_prefix(&browser_consumer_prefix(socket_id)))
        .unwrap_or_default();
    enqueue_media_transitions(&shared_state.targeted_media_control_queue, transitions);
}

fn browser_video_frame_payload(
    capture_timestamp_ms: u64,
    frame_id: u64,
    width: u32,
    height: u32,
    codec: &str,
) -> Value {
    serde_json::json!({
        "timestamp": capture_timestamp_ms,
        "capture_timestamp_ms": capture_timestamp_ms,
        "frame_id": frame_id,
        "width": width,
        "height": height,
        "codec": codec,
    })
}

fn validate_browser_jpeg_payload(payload: &[u8]) -> Result<(), &'static str> {
    if payload.len() < 4 {
        return Err("jpeg payload too short");
    }
    if payload[0..2] != [0xff, 0xd8] {
        return Err("jpeg payload missing SOI marker");
    }
    if payload[payload.len() - 2..] != [0xff, 0xd9] {
        return Err("jpeg payload missing EOI marker");
    }
    Ok(())
}

fn audio_frame_metadata(
    parameters: &std::collections::BTreeMap<String, dora_node_api::Parameter>,
    payload_len: usize,
) -> Result<(AudioFrameMetadata, Option<String>), String> {
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
    let metadata = AudioFrameMetadata {
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
    };
    validate_browser_pcm_payload(metadata, payload_len)?;
    let entity_id = parameters.get("entity_id").and_then(|value| match value {
        dora_node_api::Parameter::String(value) => Some(value.clone()),
        _ => None,
    });
    Ok((metadata, entity_id))
}

fn browser_audio_frame_payload(
    metadata: AudioFrameMetadata,
    entity_id: Option<String>,
    source_kind: &'static str,
) -> Value {
    let duration_ms = f64::from(metadata.sample_count) * 1_000.0
        / (f64::from(metadata.sample_rate) * f64::from(metadata.channels));
    serde_json::json!({
        "protocol_version": 1,
        "timestamp": metadata.capture_timestamp_ms,
        "capture_timestamp_ms": metadata.capture_timestamp_ms,
        "stream_id": metadata.stream_id.to_string(),
        "frame_id": metadata.frame_id,
        "sample_rate": metadata.sample_rate,
        "channels": metadata.channels,
        "sample_count": metadata.sample_count,
        "duration_ms": duration_ms,
        "format": metadata.format.metadata_name(),
        "entity_id": entity_id,
        "source_kind": source_kind,
    })
}

fn validate_browser_pcm_payload(
    metadata: AudioFrameMetadata,
    payload_len: usize,
) -> Result<(), String> {
    if metadata.format != PcmSampleFormat::S16Le {
        return Err("browser audio input must be s16le".into());
    }

    let expected_len = (metadata.sample_count as usize)
        .checked_mul(PcmSampleFormat::S16Le.bytes_per_sample())
        .ok_or_else(|| "browser PCM payload length overflow".to_string())?;
    if payload_len != expected_len {
        return Err(format!(
            "browser PCM payload length mismatch: expected {expected_len}, got {payload_len}"
        ));
    }

    metadata.validate_payload_len(payload_len)
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WebCameraCommand {
    pub command: String, // "start" or "stop"
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WebAudioCommand {
    pub command: String, // "start" or "stop"
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WebStreamControlCommand {
    pub command: String, // "start" or "stop"
    pub video_enabled: Option<bool>,
    pub audio_enabled: Option<bool>,
    pub target_fps: Option<u8>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AuthCredentials {
    pub username: String,
    pub password: String,
    pub token: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WebTrackingCommand {
    pub command_type: String, // "enable", "disable", "select_target", "clear_target"
    pub tracking_id: Option<u32>, // For "select_target"
    pub detection_index: Option<usize>, // For "select_target" by index
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WebTtsCommand {
    pub text: String,
}

const WALKIE_ACTIVITY_TTL: Duration = Duration::from_millis(250);

#[derive(Debug, Default)]
struct VoiceAdmissionState {
    walkie_activity: HashMap<String, Instant>,
}

impl VoiceAdmissionState {
    fn note_walkie_frame(&mut self, entity_id: &str, now: Instant) {
        self.prune_expired(now);
        self.walkie_activity.insert(entity_id.to_owned(), now);
    }

    fn is_walkie_active(&mut self, entity_id: &str, now: Instant) -> bool {
        self.prune_expired(now);
        self.walkie_activity.contains_key(entity_id)
    }

    fn prune_expired(&mut self, now: Instant) {
        self.walkie_activity
            .retain(|_, last_seen| now.duration_since(*last_seen) <= WALKIE_ACTIVITY_TTL);
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WebFleetSubscriptionCommand {
    pub action: String,                  // "activate", "deactivate", "set_active"
    pub entity_id: Option<String>,       // For activate/deactivate
    pub entity_ids: Option<Vec<String>>, // For set_active
}

#[derive(Clone)]
struct SharedState {
    pub arm_command_queue: Arc<Mutex<Vec<WebArmCommand>>>,
    pub rover_command_queue: Arc<Mutex<Vec<WebRoverCommand>>>,
    pub camera_command_queue: Arc<Mutex<Vec<WebCameraCommand>>>,
    pub audio_command_queue: Arc<Mutex<Vec<WebAudioCommand>>>,
    pub stream_command_queue: Arc<Mutex<Vec<StreamControl>>>,
    pub targeted_media_control_queue: Arc<Mutex<Vec<TargetedMediaControl>>>,
    pub tracking_command_queue: Arc<Mutex<Vec<WebTrackingCommand>>>,
    pub tts_command_queue: Arc<Mutex<Vec<TtsCommand>>>,
    pub walkie_ingress: Arc<Mutex<WalkieIngress>>,
    pub stt_bridge: Arc<SttBridge>,
    pub fleet_subscription_command_queue: Arc<Mutex<Vec<WebFleetSubscriptionCommand>>>,
    pub fleet_select_command_queue: Arc<Mutex<Vec<FleetSelectCommand>>>,
    pub video_clients: Arc<Mutex<Vec<ClientState>>>,
    pub media_demand_registry: Arc<Mutex<MediaDemandRegistry>>,
    pub performance_monitoring_enabled: Arc<Mutex<bool>>,
    pub auth_rate_limiter: Arc<AuthRateLimiter>,
    pub ip_rate_limiter: Arc<IpRateLimiter>,
    pub command_rate_limiter: Arc<CommandRateLimiter>,
    pub tts_config_rate_limiter: Arc<CommandRateLimiter>,
    pub session_registry: Arc<SessionRegistry>,
    pub fleet_status: Arc<Mutex<FleetStatus>>,
    pub active_rovers_status: Arc<Mutex<ActiveRoversStatus>>,
    pub voice_runtime: Arc<Mutex<VoiceRuntimeState>>,
    pub voice_admission: Arc<Mutex<VoiceAdmissionState>>,
    /// Process-level cumulative audio delivery counters. Lives in
    /// `SharedState` (not on `ClientState`) so lifetime totals survive
    /// client disconnects. See `audio_counters::AudioDeliveryCounters`.
    pub audio_counters: Arc<AudioDeliveryCounters>,
    pub recording: RecordingState,
    pub recording_access: Arc<RecordingAccess>,
}

impl SharedState {
    fn new() -> Self {
        // Read fleet configuration from environment variables
        let selected_entity =
            env::var("SELECTED_ENTITY_ID").unwrap_or_else(|_| "rover-kiwi".to_string());
        let fleet_roster_str =
            env::var("FLEET_ROSTER").unwrap_or_else(|_| "rover-kiwi".to_string());
        let fleet_roster: Vec<String> = fleet_roster_str
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        // Read active rovers configuration (defaults to selected entity)
        let active_rovers_str =
            env::var("ACTIVE_ROVERS").unwrap_or_else(|_| selected_entity.clone());
        let active_rovers: Vec<String> = active_rovers_str
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        let fleet_status = FleetStatus::new(selected_entity, fleet_roster);
        let active_rovers_status = ActiveRoversStatus::new(active_rovers);
        let voice_runtime = VoiceRuntimeState::new(
            active_rovers_status.active_rovers.clone(),
            default_tts_runtime_from_env(),
        );

        tracing::info!("Fleet roster: {:?}", fleet_status.fleet_roster);
        tracing::info!("Active rovers: {:?}", active_rovers_status.active_rovers);

        Self {
            arm_command_queue: Arc::new(Mutex::new(Vec::new())),
            rover_command_queue: Arc::new(Mutex::new(Vec::new())),
            camera_command_queue: Arc::new(Mutex::new(Vec::new())),
            audio_command_queue: Arc::new(Mutex::new(Vec::new())),
            stream_command_queue: Arc::new(Mutex::new(Vec::new())),
            targeted_media_control_queue: Arc::new(Mutex::new(Vec::new())),
            tracking_command_queue: Arc::new(Mutex::new(Vec::new())),
            tts_command_queue: Arc::new(Mutex::new(Vec::new())),
            walkie_ingress: Arc::new(Mutex::new(WalkieIngress::default())),
            stt_bridge: Arc::new(SttBridge::new(stt_bridge_config())),
            fleet_subscription_command_queue: Arc::new(Mutex::new(Vec::new())),
            fleet_select_command_queue: Arc::new(Mutex::new(Vec::new())),
            video_clients: Arc::new(Mutex::new(Vec::new())),
            media_demand_registry: Arc::new(Mutex::new(MediaDemandRegistry::default())),
            performance_monitoring_enabled: Arc::new(Mutex::new(true)),
            auth_rate_limiter: Arc::new(AuthRateLimiter::new()),
            ip_rate_limiter: Arc::new(IpRateLimiter::new()),
            command_rate_limiter: Arc::new(CommandRateLimiter::new()),
            tts_config_rate_limiter: Arc::new(CommandRateLimiter::new_tts_config()),
            session_registry: Arc::new(SessionRegistry::new()),
            fleet_status: Arc::new(Mutex::new(fleet_status)),
            active_rovers_status: Arc::new(Mutex::new(active_rovers_status)),
            voice_runtime: Arc::new(Mutex::new(voice_runtime)),
            voice_admission: Arc::new(Mutex::new(VoiceAdmissionState::default())),
            audio_counters: Arc::new(AudioDeliveryCounters::new()),
            recording: RecordingState::from_env(),
            recording_access: Arc::new(RecordingAccess::from_env()),
        }
    }
}

fn stt_bridge_config() -> SttBridgeConfig {
    let decode_capacity = env::var("STT_DECODE_QUEUE_CAPACITY")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(8);
    let queue_capacity = env::var("WEB_STT_QUEUE_CAPACITY")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or_else(|| decode_capacity.saturating_mul(8))
        .clamp(4, 4_096);
    let stream_idle_seconds = env::var("WEB_STT_STREAM_IDLE_SECONDS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(30)
        .clamp(5, 3_600);
    let closing_seconds = env::var("WEB_STT_CLOSING_SECONDS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(30)
        .clamp(5, 300);

    tracing::info!(
        queue_capacity,
        stream_idle_seconds,
        closing_seconds,
        "Browser STT transport configured"
    );
    SttBridgeConfig {
        queue_capacity,
        stream_idle_ttl: Duration::from_secs(stream_idle_seconds),
        closing_ttl: Duration::from_secs(closing_seconds),
    }
}

fn default_tts_runtime_from_env() -> TtsRuntimeConfig {
    let default = TtsRuntimeConfig::default();
    let language = match env::var("TTS_DEFAULT_LANGUAGE") {
        Ok(value) if value.eq_ignore_ascii_case("vi") => TtsLanguage::Vi,
        Ok(value) if value.eq_ignore_ascii_case("en") => TtsLanguage::En,
        Ok(_) | Err(env::VarError::NotUnicode(_)) => default.language,
        Err(env::VarError::NotPresent) => default.language,
    };

    let parse_or_default =
        |name: &str, fallback: String| -> String { env::var(name).unwrap_or(fallback) };

    let config = TtsRuntimeConfig {
        language,
        speaker_id: parse_or_default("TTS_DEFAULT_SPEAKER_ID", default.speaker_id.to_string())
            .parse()
            .unwrap_or(default.speaker_id),
        speed: parse_or_default("TTS_DEFAULT_SPEED", default.speed.to_string())
            .parse()
            .unwrap_or(default.speed),
        num_steps: parse_or_default("TTS_DEFAULT_STEPS", default.num_steps.to_string())
            .parse()
            .unwrap_or(default.num_steps),
        volume: parse_or_default("TTS_DEFAULT_VOLUME", default.volume.to_string())
            .parse()
            .unwrap_or(default.volume),
    };

    if let Err(error) = config.validate() {
        tracing::warn!(%error, "invalid web_bridge TTS defaults; falling back to code defaults");
        default
    } else {
        config
    }
}

fn setup_socketio(
    shared_state: SharedState,
    node_handle: Arc<Mutex<DoraNode>>,
    tts_config_command_output: DataId,
    user_collection: Arc<Collection<security::User>>,
    jwt_secret: Arc<String>,
    session_ttl_secs: u64,
    trust_proxy: bool,
) -> (SocketIo, socketioxide::layer::SocketIoLayer) {
    let (layer, io) = SocketIo::new_layer();

    tracing::info!(
        "Authentication: MongoDB + bcrypt + JWT (TTL {}s)",
        session_ttl_secs
    );
    tracing::info!(
        "Security features: Rate limiting enabled, Input validation enabled, JWT sessions enabled"
    );
    tracing::info!("Proxy trust: {}", trust_proxy);

    // Clone io for use inside the closure
    let io_for_fleet = io.clone();
    let io_for_active_rovers = io.clone();
    let io_for_voice = io.clone();

    io.ns("/", move |socket: SocketRef, TryData::<AuthCredentials>(auth)| {
        let user_collection = user_collection.clone();
        let jwt_secret = jwt_secret.clone();
        let shared_state = shared_state.clone();
        let node_handle = node_handle.clone();
        let tts_config_command_output = tts_config_command_output.clone();

        async move {
        let socket_id = socket.id.to_string();

        // Per-IP rate limit (in addition to per-socket)
        let client_ip = extract_client_ip(&socket.req_parts().headers, trust_proxy);
        if !shared_state.ip_rate_limiter.check_auth_attempt_ip(&client_ip) {
            log_rate_limit_exceeded(&socket_id, "auth_ip");
            tracing::warn!(security_event = "ip_rate_limit_exceeded", client_ip = %client_ip, "Per-IP auth rate limit exceeded");
            socket.emit("auth_error", serde_json::json!({"reason": "rate_limited"})).ok();
            socket.disconnect().ok();
            return;
        }

        // Check rate limit for authentication attempts
        if !shared_state.auth_rate_limiter.check_auth_attempt(&socket_id) {
            log_rate_limit_exceeded(&socket_id, "auth");
            socket.emit("auth_error", serde_json::json!({"reason": "rate_limited"})).ok();
            socket.disconnect().ok();
            return;
        }

        let credentials = match auth {
            Ok(c) => c,
            Err(_) => {
                log_auth_attempt(&socket_id, "unknown", false);
                socket.emit("auth_error", serde_json::json!({"reason": "invalid_credentials"})).ok();
                socket.disconnect().ok();
                return;
            }
        };

        let username_for_log = credentials.username.clone();
        let token_ref = credentials.token.as_deref();

        match security::authenticate_and_issue_token(
            token_ref,
            &credentials.username,
            &credentials.password,
            &user_collection,
            &jwt_secret,
            session_ttl_secs,
        ).await {
            Ok((token, claims)) => {
                let sub = claims.sub.clone();
                log_auth_attempt(&socket_id, &sub, true);
                socket.emit("auth_token", token).ok();
                shared_state.session_registry.register(&socket_id, claims);
                if let Err(error) = socket.join(AUTHENTICATED_ROOM) {
                    tracing::error!(%error, "Failed to join authenticated Socket.IO room");
                    shared_state.session_registry.remove(&socket_id);
                    socket.disconnect().ok();
                    return;
                }
                shared_state.auth_rate_limiter.reset(&socket_id);
                tracing::info!("Client authenticated and connected: {} (user={})", socket_id, sub);
            }
            Err(reason) => {
                log_auth_attempt(&socket_id, &username_for_log, false);
                socket.emit("auth_error", serde_json::json!({"reason": reason.as_str()})).ok();
                socket.disconnect().ok();
                return;
            }
        }

        // Add client to video streaming list
        let client_state = ClientState::new(socket_id.clone());
        shared_state.video_clients.lock().unwrap().push(client_state);

        // Send fleet status to newly connected client
        let fleet_status = shared_state.fleet_status.lock().unwrap().clone();
        socket.emit("fleet_status", fleet_status).ok();
        emit_tts_config_state(&socket, &current_tts_config_state(&shared_state));
        for status in current_voice_statuses(&shared_state) {
            emit_voice_status(&socket, &status);
        }

        // STT status is authoritative and process-wide. Reconnects either receive
        // the cached lifecycle state or trigger a fresh response from central STT.
        if let Some(status) = shared_state.stt_bridge.cached_status() {
            socket.emit("stt_status", status).ok();
        } else {
            shared_state.stt_bridge.request_status();
        }

        for status in shared_state.recording.status_snapshot() {
            emit_recording_event(&socket, "recording_session_status", status);
        }

        let shared_state_clone = shared_state.clone();
        let socket_id_clone = socket_id.clone();
        socket.on(
            "recording_session_command",
            move |socket: SocketRef, Data::<Value>(data)| {
                if !shared_state_clone.session_registry.is_valid(&socket_id_clone) {
                    socket.emit("auth_error", serde_json::json!({"reason": "token_expired"})).ok();
                    socket.disconnect().ok();
                    return;
                }
                if !shared_state_clone.command_rate_limiter.check_command(&socket_id_clone) {
                    log_rate_limit_exceeded(&socket_id_clone, "recording_session_command");
                    return;
                }
                touch_activity(&shared_state_clone.video_clients, &socket_id_clone);
                let command = match serde_json::from_value::<RecordingSessionCommand>(data) {
                    Ok(command) => command,
                    Err(error) => {
                        log_validation_error(&socket_id_clone, &format!("recording command: {error}"));
                        return;
                    }
                };
                if let Err(error) = command.validate() {
                    log_validation_error(&socket_id_clone, &format!("recording command: {error}"));
                    emit_recording_event(
                        &socket,
                        "recording_session_command_result",
                        rejected_recording_command(
                            &command.request_id,
                            RecordingReasonCode::InvalidRequest,
                            &error,
                        ),
                    );
                    return;
                }
                let request_id = command.request_id.clone();
                let consumer_id = recording_consumer_id(&request_id);
                match &command.action {
                    RecordingSessionAction::Start { entity_id, .. } => {
                        if !is_target_active(&shared_state_clone, entity_id) {
                            emit_recording_event(
                                &socket,
                                "recording_session_command_result",
                                rejected_recording_command(
                                    &request_id,
                                    RecordingReasonCode::InvalidEntity,
                                    "target rover is not active",
                                ),
                            );
                            return;
                        }
                        if shared_state_clone.recording.active_entities.lock().ok().is_some_and(|active| {
                            active.values().any(|active_entity| active_entity == entity_id)
                        }) {
                            emit_recording_event(
                                &socket,
                                "recording_session_command_result",
                                rejected_recording_command(
                                    &request_id,
                                    RecordingReasonCode::AlreadyRecording,
                                    "rover already has an active recording",
                                ),
                            );
                            return;
                        }
                        if let Err(reason) = shared_state_clone
                            .recording
                            .admit(&request_id, &socket_id_clone, RequestKind::Command)
                        {
                            emit_recording_event(
                                &socket,
                                "recording_session_command_result",
                                rejected_recording_command(&request_id, RecordingReasonCode::ResourceLimit, reason),
                            );
                            return;
                        }
                        if enqueue_recording_demand(&shared_state_clone, entity_id, &consumer_id, true).is_err()
                            || shared_state_clone.recording.commands.lock().map(|mut queue| {
                                queue.push_back(command.clone());
                            }).is_err()
                        {
                            shared_state_clone.recording.take(&request_id);
                            let _ = enqueue_recording_demand(&shared_state_clone, entity_id, &consumer_id, false);
                            emit_recording_event(
                                &socket,
                                "recording_session_command_result",
                                rejected_recording_command(&request_id, RecordingReasonCode::Internal, "recording queue unavailable"),
                            );
                            return;
                        }
                        tracing::info!(action = "recording_start", entity_id, request_id = %request_id, outcome = "queued", "recording request admitted");
                    }
                    RecordingSessionAction::Stop { recording_id } => {
                        if shared_state_clone.recording.active_entity(recording_id).is_none() {
                            emit_recording_event(
                                &socket,
                                "recording_session_command_result",
                                rejected_recording_command(&request_id, RecordingReasonCode::NotFound, "recording session not found"),
                            );
                            return;
                        }
                        if let Err(reason) = shared_state_clone
                            .recording
                            .admit(&request_id, &socket_id_clone, RequestKind::Command)
                        {
                            emit_recording_event(
                                &socket,
                                "recording_session_command_result",
                                rejected_recording_command(&request_id, RecordingReasonCode::ResourceLimit, reason),
                            );
                            return;
                        }
                        if shared_state_clone.recording.commands.lock().map(|mut queue| {
                            queue.push_back(command.clone());
                        }).is_err() {
                            shared_state_clone.recording.take(&request_id);
                            emit_recording_event(
                                &socket,
                                "recording_session_command_result",
                                rejected_recording_command(&request_id, RecordingReasonCode::Internal, "recording queue unavailable"),
                            );
                            return;
                        }
                        tracing::info!(action = "recording_stop", clip_id = %recording_id, request_id = %request_id, outcome = "queued", "recording stop admitted");
                    }
                }
            },
        );

        let shared_state_clone = shared_state.clone();
        let socket_id_clone = socket_id.clone();
        socket.on(
            "recording_clip_list",
            move |socket: SocketRef, Data::<Value>(data)| {
                if !shared_state_clone.session_registry.is_valid(&socket_id_clone) {
                    socket.emit("auth_error", serde_json::json!({"reason": "token_expired"})).ok();
                    socket.disconnect().ok();
                    return;
                }
                if !shared_state_clone.command_rate_limiter.check_command(&socket_id_clone) {
                    log_rate_limit_exceeded(&socket_id_clone, "recording_clip_list");
                    return;
                }
                let query = match serde_json::from_value::<RecordingClipQuery>(data) {
                    Ok(query) => query,
                    Err(error) => {
                        log_validation_error(&socket_id_clone, &format!("recording clip query: {error}"));
                        return;
                    }
                };
                if let Err(error) = query.validate() {
                    emit_recording_event(&socket, "recording_clip_list_result", RecordingClipQueryResult {
                        protocol_version: robo_rover_lib::RECORDING_PROTOCOL_VERSION,
                        request_id: query.request_id,
                        accepted: false,
                        clips: Vec::new(),
                        reason_code: Some(RecordingReasonCode::InvalidRequest),
                        detail: Some(error.chars().take(256).collect()),
                    });
                    return;
                }
                let request_id = query.request_id.clone();
                let admitted = shared_state_clone
                    .recording
                    .admit(&request_id, &socket_id_clone, RequestKind::ClipList);
                if let Err(reason) = admitted {
                    emit_recording_event(&socket, "recording_clip_list_result", RecordingClipQueryResult {
                        protocol_version: robo_rover_lib::RECORDING_PROTOCOL_VERSION,
                        request_id,
                        accepted: false,
                        clips: Vec::new(),
                        reason_code: Some(RecordingReasonCode::ResourceLimit),
                        detail: Some(reason.into()),
                    });
                    return;
                }
                let queued = admitted.is_ok()
                    && shared_state_clone
                        .recording
                        .clip_queries
                        .lock()
                        .map(|mut queue| queue.push_back(query))
                        .is_ok();
                if !queued {
                    shared_state_clone.recording.take(&request_id);
                    emit_recording_event(
                        &socket,
                        "recording_clip_list_result",
                        rejected_recording_clip_query(
                            &request_id,
                            RecordingReasonCode::Internal,
                            "recording queue unavailable",
                        ),
                    );
                    tracing::warn!(action = "recording_clip_list", request_id = %request_id, outcome = "rejected", reason = "queue_full", "recording catalog request rejected");
                    return;
                }
            },
        );

        let shared_state_clone = shared_state.clone();
        let socket_id_clone = socket_id.clone();
        socket.on(
            "recording_playback_ticket",
            move |socket: SocketRef, Data::<Value>(data)| {
                if !shared_state_clone.session_registry.is_valid(&socket_id_clone) {
                    socket.emit("auth_error", serde_json::json!({"reason": "token_expired"})).ok();
                    socket.disconnect().ok();
                    return;
                }
                if !shared_state_clone.command_rate_limiter.check_command(&socket_id_clone) {
                    log_rate_limit_exceeded(&socket_id_clone, "recording_playback_ticket");
                    return;
                }
                let request = match serde_json::from_value::<RecordingPlaybackTicketRequest>(data) {
                    Ok(request) => request,
                    Err(error) => {
                        log_validation_error(&socket_id_clone, &format!("recording playback request: {error}"));
                        return;
                    }
                };
                if let Err(error) = request.validate() {
                    tracing::warn!(action = "playback_ticket", request_id = %request.request_id, outcome = "rejected", reason = %error, "invalid playback ticket request");
                    emit_recording_event(
                        &socket,
                        "recording_playback_ticket_result",
                        rejected_recording_playback_ticket(
                            &request.request_id,
                            RecordingReasonCode::InvalidRequest,
                            &error,
                        ),
                    );
                    return;
                }
                let request_id = request.request_id.clone();
                let admitted = shared_state_clone
                    .recording
                    .admit(&request_id, &socket_id_clone, RequestKind::PlaybackTicket);
                if let Err(reason) = admitted {
                    emit_recording_event(
                        &socket,
                        "recording_playback_ticket_result",
                        serde_json::json!({
                            "protocol_version": robo_rover_lib::RECORDING_PROTOCOL_VERSION,
                            "request_id": request_id,
                            "accepted": false,
                            "reason_code": "resource_limit",
                            "detail": reason,
                        }),
                    );
                    return;
                }
                let queued = admitted.is_ok()
                    && shared_state_clone
                        .recording
                        .playback_queries
                        .lock()
                        .map(|mut queue| queue.push_back(request))
                        .is_ok();
                if !queued {
                    shared_state_clone.recording.take(&request_id);
                    tracing::warn!(action = "playback_ticket", request_id = %request_id, outcome = "rejected", reason = "queue_full", "playback ticket request rejected");
                    emit_recording_event(
                        &socket,
                        "recording_playback_ticket_result",
                        rejected_recording_playback_ticket(
                            &request_id,
                            RecordingReasonCode::Internal,
                            "recording queue unavailable",
                        ),
                    );
                }
            },
        );

        let shared_state_clone = shared_state.clone();
        let socket_id_clone = socket_id.clone();
        socket.on(
            "recording_delete",
            move |socket: SocketRef, Data::<Value>(data)| {
                if !shared_state_clone.session_registry.is_valid(&socket_id_clone) {
                    socket.emit("auth_error", serde_json::json!({"reason": "token_expired"})).ok();
                    socket.disconnect().ok();
                    return;
                }
                if !shared_state_clone.command_rate_limiter.check_command(&socket_id_clone) {
                    log_rate_limit_exceeded(&socket_id_clone, "recording_delete");
                    return;
                }
                touch_activity(&shared_state_clone.video_clients, &socket_id_clone);
                let request = match serde_json::from_value::<RecordingDeleteRequest>(data) {
                    Ok(request) => request,
                    Err(error) => {
                        log_validation_error(&socket_id_clone, &format!("recording delete: {error}"));
                        return;
                    }
                };
                if let Err(error) = request.validate() {
                    emit_recording_event(&socket, "recording_delete_result", rejected_recording_delete(&request.request_id, RecordingReasonCode::InvalidRequest, &error));
                    return;
                }
                let request_id = request.request_id.clone();
                if let Err(reason) = shared_state_clone.recording.admit(&request_id, &socket_id_clone, RequestKind::Delete) {
                    emit_recording_event(&socket, "recording_delete_result", rejected_recording_delete(&request_id, RecordingReasonCode::ResourceLimit, reason));
                    return;
                }
                let queued = shared_state_clone.recording.delete_queries.lock().map(|mut queue| queue.push_back(request)).is_ok();
                if !queued {
                    shared_state_clone.recording.take(&request_id);
                    emit_recording_event(&socket, "recording_delete_result", rejected_recording_delete(&request_id, RecordingReasonCode::Internal, "recording delete queue unavailable"));
                }
            },
        );

        let shared_state_clone = shared_state.clone();
        let socket_id_clone = socket_id.clone();
        socket.on("arm_command", move |socket: SocketRef, Data::<Value>(data)| {
            if !shared_state_clone.session_registry.is_valid(&socket_id_clone) {
                socket.emit("auth_error", serde_json::json!({"reason": "token_expired"})).ok();
                socket.disconnect().ok();
                return;
            }
            if !shared_state_clone.command_rate_limiter.check_command(&socket_id_clone) {
                log_rate_limit_exceeded(&socket_id_clone, "arm_command");
                return;
            }
            touch_activity(&shared_state_clone.video_clients, &socket_id_clone);

            if let Ok(web_cmd) = serde_json::from_value::<WebArmCommand>(data) {
                // Validate joint positions if present
                if let Some(ref positions) = web_cmd.joint_positions {
                    let joint_values = vec![
                        positions.shoulder_pan, positions.shoulder_lift, positions.elbow_flex,
                        positions.wrist_flex, positions.wrist_roll, positions.gripper
                    ];
                    for (i, &angle) in joint_values.iter().enumerate() {
                        if let Err(e) = security::validation::validate_joint_position(angle) {
                            log_validation_error(&socket_id_clone, &format!("Arm joint {}: {}", i, e));
                            tracing::warn!("Arm command validation failed: {}", e);
                            return;
                        }
                    }
                }

                tracing::debug!("Received arm command: {:?}", web_cmd.command_type);
                shared_state_clone
                    .arm_command_queue
                    .lock()
                    .unwrap()
                    .push(web_cmd);
            }
        });

        let shared_state_clone = shared_state.clone();
        let socket_id_clone = socket_id.clone();
        socket.on(
            "rover_command",
            move |socket: SocketRef, Data::<Value>(data)| {
                if !shared_state_clone.session_registry.is_valid(&socket_id_clone) {
                    socket.emit("auth_error", serde_json::json!({"reason": "token_expired"})).ok();
                    socket.disconnect().ok();
                    return;
                }
                if !shared_state_clone.command_rate_limiter.check_command(&socket_id_clone) {
                    log_rate_limit_exceeded(&socket_id_clone, "rover_command");
                    return;
                }
                touch_activity(&shared_state_clone.video_clients, &socket_id_clone);

                if let Ok(web_cmd) = serde_json::from_value::<WebRoverCommand>(data) {
                    // Validate wheel velocities if present
                    let wheels = [web_cmd.wheel1, web_cmd.wheel2, web_cmd.wheel3, web_cmd.wheel4];
                    for (i, wheel_opt) in wheels.iter().enumerate() {
                        if let Some(velocity) = wheel_opt {
                            if let Err(e) = security::validation::validate_wheel_velocity(*velocity) {
                                log_validation_error(&socket_id_clone, &format!("Wheel {}: {}", i+1, e));
                                tracing::warn!("Rover command validation failed: {}", e);
                                return;
                            }
                        }
                    }

                    tracing::debug!("Received rover command: {:?}", web_cmd.command_type);
                    shared_state_clone
                        .rover_command_queue
                        .lock()
                        .unwrap()
                        .push(web_cmd);
                }
            },
        );

        let shared_state_clone = shared_state.clone();
        let socket_id_clone = socket_id.clone();
        socket.on(
            "camera_control",
            move |socket: SocketRef, Data::<Value>(data)| {
                if !shared_state_clone.session_registry.is_valid(&socket_id_clone) {
                    socket.emit("auth_error", serde_json::json!({"reason": "token_expired"})).ok();
                    socket.disconnect().ok();
                    return;
                }
                if !shared_state_clone.command_rate_limiter.check_command(&socket_id_clone) {
                    log_rate_limit_exceeded(&socket_id_clone, "camera_control");
                    return;
                }
                touch_activity(&shared_state_clone.video_clients, &socket_id_clone);
                if let Ok(web_cmd) = serde_json::from_value::<WebCameraCommand>(data) {
                    tracing::debug!("Received camera control: {:?}", web_cmd.command);
                    if let Some(action) = convert_web_command_to_camera_command(&web_cmd) {
                        set_browser_media_demand(
                            &shared_state_clone,
                            &socket_id_clone,
                            "camera",
                            MediaResource::Camera,
                            matches!(action, CameraAction::Start),
                        );
                    }
                }
            },
        );

        let shared_state_clone = shared_state.clone();
        let socket_id_clone = socket_id.clone();
        socket.on(
            "stream_control",
            move |socket: SocketRef, Data::<Value>(data)| {
                if !shared_state_clone.session_registry.is_valid(&socket_id_clone) {
                    socket.emit("auth_error", serde_json::json!({"reason": "token_expired"})).ok();
                    socket.disconnect().ok();
                    return;
                }
                if !shared_state_clone.command_rate_limiter.check_command(&socket_id_clone) {
                    log_rate_limit_exceeded(&socket_id_clone, "stream_control");
                    return;
                }
                touch_activity(&shared_state_clone.video_clients, &socket_id_clone);

                if let Ok(web_cmd) = serde_json::from_value::<WebStreamControlCommand>(data) {
                    let enabled = match web_cmd.command.as_str() {
                        "start" | "resume" => true,
                        "stop" | "pause" => false,
                        _ => web_cmd.video_enabled.unwrap_or(false),
                    };
                    if let Ok(mut clients) = shared_state_clone.video_clients.lock() {
                        if let Some(client) = clients.iter_mut().find(|client| client.socket_id == socket_id_clone) {
                            client.video_enabled = enabled;
                            if let Some(target_fps) = web_cmd.target_fps { client.target_fps = target_fps.clamp(1, 120); }
                            if let Some(audio_enabled) = web_cmd.audio_enabled { client.audio_enabled = audio_enabled; }
                        }
                    }
                    set_browser_media_demand(&shared_state_clone, &socket_id_clone, "stream", MediaResource::Camera, enabled);
                    set_browser_media_demand(&shared_state_clone, &socket_id_clone, "stream", MediaResource::Jpeg, enabled);
                    if let Some(audio_enabled) = web_cmd.audio_enabled {
                        set_browser_media_demand(&shared_state_clone, &socket_id_clone, "stream", MediaResource::Microphone, audio_enabled);
                    }
                }
            },
        );

        let shared_state_clone = shared_state.clone();
        let socket_id_clone = socket_id.clone();
        socket.on(
            "audio_control",
            move |socket: SocketRef, Data::<Value>(data)| {
                if !shared_state_clone.session_registry.is_valid(&socket_id_clone) {
                    socket.emit("auth_error", serde_json::json!({"reason": "token_expired"})).ok();
                    socket.disconnect().ok();
                    return;
                }
                if !shared_state_clone.command_rate_limiter.check_command(&socket_id_clone) {
                    log_rate_limit_exceeded(&socket_id_clone, "audio_control");
                    return;
                }
                touch_activity(&shared_state_clone.video_clients, &socket_id_clone);
                if let Ok(web_cmd) = serde_json::from_value::<WebAudioCommand>(data) {
                    tracing::debug!("Received audio control: {:?}", web_cmd.command);
                    if let Some(action) = convert_web_command_to_audio_command(&web_cmd) {
                        let enabled = matches!(action, AudioAction::Start);
                        if let Ok(mut clients) = shared_state_clone.video_clients.lock() {
                            if let Some(client) = clients.iter_mut().find(|client| client.socket_id == socket_id_clone) { client.audio_enabled = enabled; }
                        }
                        set_browser_media_demand(&shared_state_clone, &socket_id_clone, "audio", MediaResource::Microphone, enabled);
                    }
                }
            },
        );

        let shared_state_clone = shared_state.clone();
        let socket_id_clone = socket_id.clone();
        socket.on(
            "tracking_command",
            move |socket: SocketRef, Data::<Value>(data)| {
                if !shared_state_clone.session_registry.is_valid(&socket_id_clone) {
                    socket.emit("auth_error", serde_json::json!({"reason": "token_expired"})).ok();
                    socket.disconnect().ok();
                    return;
                }
                if !shared_state_clone.command_rate_limiter.check_command(&socket_id_clone) {
                    log_rate_limit_exceeded(&socket_id_clone, "tracking_command");
                    return;
                }
                touch_activity(&shared_state_clone.video_clients, &socket_id_clone);
                if let Ok(web_cmd) = serde_json::from_value::<WebTrackingCommand>(data) {
                    tracing::debug!("Received tracking command: {:?}", web_cmd.command_type);
                    shared_state_clone
                        .tracking_command_queue
                        .lock()
                        .unwrap()
                        .push(web_cmd);
                }
            },
        );

        let shared_state_clone = shared_state.clone();
        let socket_id_clone = socket_id.clone();
        let io_for_voice_clone = io_for_voice.clone();
        socket.on(
            "tts_config_update",
            move |socket: SocketRef, Data::<Value>(data)| {
                if !shared_state_clone.session_registry.is_valid(&socket_id_clone) {
                    socket.emit("auth_error", serde_json::json!({"reason": "token_expired"})).ok();
                    socket.disconnect().ok();
                    return;
                }
                if !shared_state_clone
                    .tts_config_rate_limiter
                    .check_command(&socket_id_clone)
                {
                    log_rate_limit_exceeded(&socket_id_clone, "tts_config_update");
                    return;
                }
                touch_activity(&shared_state_clone.video_clients, &socket_id_clone);

                let update = match serde_json::from_value::<TtsConfigUpdate>(data) {
                    Ok(update) => update,
                    Err(error) => {
                        log_validation_error(&socket_id_clone, &format!("TTS config: {error}"));
                        emit_tts_config_state(&socket, &current_tts_config_state(&shared_state_clone));
                        return;
                    }
                };

                if let Err(error) = update.validate() {
                    log_validation_error(&socket_id_clone, &format!("TTS config: {error}"));
                    emit_tts_config_state(&socket, &current_tts_config_state(&shared_state_clone));
                    return;
                }

                let outcome = {
                    let runtime = shared_state_clone.voice_runtime.lock().unwrap();
                    runtime.preview_config_update(update, current_timestamp_ms())
                };

                match outcome {
                    ConfigUpdateOutcome::Accepted { command, .. } => {
                        let serialized = match serde_json::to_vec(&command) {
                            Ok(serialized) => serialized,
                            Err(error) => {
                                tracing::error!(%error, "failed to serialize TTS config command");
                                emit_tts_config_state(
                                    &socket,
                                    &current_tts_config_state(&shared_state_clone),
                                );
                                return;
                            }
                        };
                        let arrow_data = BinaryArray::from_vec(vec![serialized.as_slice()]);
                        let sent = node_handle
                            .lock()
                            .ok()
                            .and_then(|mut node_guard| {
                                node_guard
                                    .send_output(
                                        tts_config_command_output.clone(),
                                        Default::default(),
                                        arrow_data,
                                    )
                                    .ok()
                            })
                            .is_some();

                        if !sent {
                            tracing::error!("failed to send tts_config_command to Dora");
                            emit_tts_config_state(
                                &socket,
                                &current_tts_config_state(&shared_state_clone),
                            );
                            return;
                        }

                        let state = {
                            let mut runtime = shared_state_clone.voice_runtime.lock().unwrap();
                            match runtime.commit_config_command(command) {
                                ConfigUpdateOutcome::Accepted { state, .. } => state,
                                ConfigUpdateOutcome::Stale { state } => state,
                            }
                        };
                        broadcast_tts_config_state(
                            &io_for_voice_clone,
                            &shared_state_clone.session_registry,
                            &state,
                        );
                    }
                    ConfigUpdateOutcome::Stale { state } => emit_tts_config_state(&socket, &state),
                }
            },
        );

        let shared_state_clone = shared_state.clone();
        let socket_id_clone = socket_id.clone();
        socket.on(
            "tts_command",
            move |socket: SocketRef, Data::<Value>(data)| {
                if !shared_state_clone.session_registry.is_valid(&socket_id_clone) {
                    socket.emit("auth_error", serde_json::json!({"reason": "token_expired"})).ok();
                    socket.disconnect().ok();
                    return;
                }
                if !shared_state_clone.command_rate_limiter.check_command(&socket_id_clone) {
                    log_rate_limit_exceeded(&socket_id_clone, "tts_command");
                    return;
                }
                touch_activity(&shared_state_clone.video_clients, &socket_id_clone);

                match serde_json::from_value::<WebTtsCommand>(data) {
                    Ok(web_cmd) => {
                        let selected_target = selected_target_entity(&shared_state_clone);
                        let tts_command = convert_web_command_to_tts_command(&web_cmd);
                        // Validate TTS text
                        if let Err(e) = security::validation::validate_tts_text(&web_cmd.text) {
                            log_validation_error(&socket_id_clone, &format!("TTS text: {}", e));
                            tracing::warn!("TTS command validation failed: {}", e);
                            emit_tts_ack(
                                &socket,
                                &build_tts_ack(
                                    &tts_command.command_id,
                                    &selected_target,
                                    TtsAckState::Rejected,
                                    Some(VoiceReasonCode::InvalidCommand),
                                    Some("invalid tts text".to_string()),
                                ),
                            );
                            return;
                        }
                        if !is_target_active(&shared_state_clone, &selected_target) {
                            tracing::warn!(
                                target_entity = %selected_target,
                                command_id = %tts_command.command_id,
                                "Rejected TTS command because selected rover is inactive"
                            );
                            emit_tts_ack(
                                &socket,
                                &build_tts_ack(
                                    &tts_command.command_id,
                                    &selected_target,
                                    TtsAckState::Rejected,
                                    Some(VoiceReasonCode::VoiceNotReady),
                                    Some("selected rover is not active".to_string()),
                                ),
                            );
                            return;
                        }
                        if is_walkie_active(&shared_state_clone, &selected_target) {
                            tracing::warn!(
                                target_entity = %selected_target,
                                command_id = %tts_command.command_id,
                                "Rejected TTS command because walkie is active"
                            );
                            emit_tts_ack(
                                &socket,
                                &build_tts_ack(
                                    &tts_command.command_id,
                                    &selected_target,
                                    TtsAckState::Rejected,
                                    Some(VoiceReasonCode::WalkieActive),
                                    Some("walkie stream is active".to_string()),
                                ),
                            );
                            return;
                        }

                        tracing::info!(
                            target_entity = %selected_target,
                            command_id = %tts_command.command_id,
                            text_len = web_cmd.text.len(),
                            "Accepted TTS command from authenticated socket"
                        );
                        emit_tts_ack(
                            &socket,
                            &build_tts_ack(
                                &tts_command.command_id,
                                &selected_target,
                                TtsAckState::Accepted,
                                None,
                                None,
                            ),
                        );
                        shared_state_clone
                            .tts_command_queue
                            .lock()
                            .unwrap()
                            .push(tts_command);
                    }
                    Err(error) => {
                        tracing::warn!(%error, "Rejected malformed tts_command payload");
                    }
                }
            },
        );

        let shared_state_clone = shared_state.clone();
        let socket_id_clone = socket_id.clone();
        socket.on(
            "audio_stream",
            move |socket: SocketRef,
                  TryData::<WalkieAudioFrameMetadata>(metadata),
                  Bin(attachments)| {
                if !shared_state_clone.session_registry.is_valid(&socket_id_clone) {
                    socket.emit("auth_error", serde_json::json!({"reason": "token_expired"})).ok();
                    socket.disconnect().ok();
                    return;
                }
                if !shared_state_clone.command_rate_limiter.check_command(&socket_id_clone) {
                    log_rate_limit_exceeded(&socket_id_clone, "audio_stream");
                    return;
                }
                touch_activity(&shared_state_clone.video_clients, &socket_id_clone);
                let metadata = match metadata {
                    Ok(metadata) => metadata,
                    Err(error) => {
                        if let Ok(mut ingress) = shared_state_clone.walkie_ingress.lock() {
                            ingress.record_malformed_metadata();
                        }
                        log_validation_error(
                            &socket_id_clone,
                            &format!("Walkie metadata: {error}"),
                        );
                        return;
                    }
                };
                let now = Instant::now();
                let admitted = shared_state_clone
                    .walkie_ingress
                    .lock()
                    .map_err(|_| "walkie ingress lock poisoned".to_string())
                    .and_then(|mut ingress| {
                        ingress.admit(&socket_id_clone, metadata, attachments, now)
                    });
                if let Err(error) = admitted {
                    log_validation_error(
                        &socket_id_clone,
                        &format!("Walkie frame: {error}"),
                    );
                    return;
                }
                let selected_target = selected_target_entity(&shared_state_clone);
                if is_target_active(&shared_state_clone, &selected_target) {
                    if let Ok(mut admission) = shared_state_clone.voice_admission.lock() {
                        admission.note_walkie_frame(&selected_target, now);
                    }
                }
            },
        );

        let shared_state_clone = shared_state.clone();
        let socket_id_clone = socket_id.clone();
        socket.on(
            "voice_command_control",
            move |socket: SocketRef, Data::<Value>(data)| {
                if !shared_state_clone.session_registry.is_valid(&socket_id_clone) {
                    socket.emit("auth_error", serde_json::json!({"reason": "token_expired"})).ok();
                    socket.disconnect().ok();
                    return;
                }
                if !shared_state_clone.command_rate_limiter.check_command(&socket_id_clone) {
                    log_rate_limit_exceeded(&socket_id_clone, "voice_command_control");
                    return;
                }
                touch_activity(&shared_state_clone.video_clients, &socket_id_clone);
                let control = match serde_json::from_value::<VoiceCommandControl>(data) {
                    Ok(control) => control,
                    Err(error) => {
                        log_validation_error(
                            &socket_id_clone,
                            &format!("Voice control: {error}"),
                        );
                        return;
                    }
                };
                let selected_target = matches!(&control, VoiceCommandControl::Start { .. })
                    .then(|| {
                        shared_state_clone
                            .fleet_status
                            .lock()
                            .ok()
                            .map(|status| status.selected_entity.clone())
                    })
                    .flatten();
                let target_is_active = selected_target.as_ref().is_some_and(|target| {
                    shared_state_clone
                        .active_rovers_status
                        .lock()
                        .map(|status| status.active_rovers.contains(target))
                        .unwrap_or(false)
                });
                if let Err(error) = shared_state_clone.stt_bridge.handle_control(
                    &socket_id_clone,
                    control,
                    selected_target.as_deref(),
                    target_is_active,
                ) {
                    log_validation_error(&socket_id_clone, &format!("Voice control: {error}"));
                }
            },
        );

        let shared_state_clone = shared_state.clone();
        let socket_id_clone = socket_id.clone();
        socket.on(
            "voice_command_audio",
            move |socket: SocketRef, Data::<Value>(data)| {
                if !shared_state_clone.session_registry.is_valid(&socket_id_clone) {
                    socket.emit("auth_error", serde_json::json!({"reason": "token_expired"})).ok();
                    socket.disconnect().ok();
                    return;
                }
                if !shared_state_clone.command_rate_limiter.check_command(&socket_id_clone) {
                    log_rate_limit_exceeded(&socket_id_clone, "voice_command_audio");
                    return;
                }
                touch_activity(&shared_state_clone.video_clients, &socket_id_clone);
                let frame = match serde_json::from_value::<VoiceCommandAudioFrame>(data) {
                    Ok(frame) => frame,
                    Err(error) => {
                        log_validation_error(&socket_id_clone, &format!("Voice audio: {error}"));
                        return;
                    }
                };
                if let Err(error) = shared_state_clone
                    .stt_bridge
                    .handle_audio(&socket_id_clone, frame)
                {
                    log_validation_error(&socket_id_clone, &format!("Voice audio: {error}"));
                }
            },
        );

        let shared_state_clone = shared_state.clone();
        socket.on(
            "performance_control",
            move |_socket: SocketRef, Data::<Value>(data)| {
                if let Some(enabled) = data.get("enabled").and_then(|v| v.as_bool()) {
                    tracing::info!("Performance monitoring {}", if enabled { "enabled" } else { "disabled" });
                    *shared_state_clone.performance_monitoring_enabled.lock().unwrap() = enabled;
                }
            },
        );

        let shared_state_clone = shared_state.clone();
        let socket_id_clone = socket_id.clone();
        let io_for_fleet_clone = io_for_fleet.clone();
        socket.on(
            "fleet_select",
            move |socket: SocketRef, Data::<Value>(data)| {
                if !shared_state_clone.session_registry.is_valid(&socket_id_clone) {
                    socket.emit("auth_error", serde_json::json!({"reason": "token_expired"})).ok();
                    socket.disconnect().ok();
                    return;
                }
                if !shared_state_clone.command_rate_limiter.check_command(&socket_id_clone) {
                    log_rate_limit_exceeded(&socket_id_clone, "fleet_select");
                    return;
                }
                touch_activity(&shared_state_clone.video_clients, &socket_id_clone);
                if let Ok(select_cmd) = serde_json::from_value::<FleetSelectCommand>(data) {
                    tracing::info!("Fleet select requested: {}", select_cmd.entity_id);

                    let target_is_active = shared_state_clone
                        .active_rovers_status
                        .lock()
                        .ok()
                        .is_some_and(|status| active_rover_status_includes(&status, &select_cmd.entity_id));

                    // Update fleet status with new selection
                    let mut fleet_status = shared_state_clone.fleet_status.lock().unwrap();

                    // A selected rover is also routing authority for browser demand.
                    if fleet_status.fleet_roster.contains(&select_cmd.entity_id) && target_is_active {
                        fleet_status.selected_entity = select_cmd.entity_id.clone();
                        fleet_status.timestamp = select_cmd.timestamp;

                        // Queue command to send to orchestra-bridge
                        if let Ok(mut queue) = shared_state_clone.fleet_select_command_queue.lock() {
                            queue.push(select_cmd.clone());
                        }

                        // Migrate only this browser's owned demand. Recorder demand stays pinned.
                        let status_clone = fleet_status.clone();
                        drop(fleet_status); // Release lock before async operation
                        move_browser_media_demand(&shared_state_clone, &socket_id_clone, &select_cmd.entity_id);

                        io_for_fleet_clone.emit("fleet_status", status_clone).ok();
                        tracing::info!("Fleet selection updated and broadcast to all clients");
                    } else if !target_is_active {
                        tracing::warn!("Cannot select inactive rover: {}", select_cmd.entity_id);
                    } else {
                        tracing::warn!("Invalid entity_id selection: {}", select_cmd.entity_id);
                    }
                }
            },
        );

        let shared_state_clone = shared_state.clone();
        let socket_id_clone = socket_id.clone();
        let io_for_active_rovers = io_for_active_rovers.clone();
        let io_for_voice_clone = io_for_voice.clone();
        socket.on(
            "fleet_subscription",
            move |socket: SocketRef, Data::<Value>(data)| {
                if !shared_state_clone.session_registry.is_valid(&socket_id_clone) {
                    socket.emit("auth_error", serde_json::json!({"reason": "token_expired"})).ok();
                    socket.disconnect().ok();
                    return;
                }
                if !shared_state_clone.command_rate_limiter.check_command(&socket_id_clone) {
                    log_rate_limit_exceeded(&socket_id_clone, "fleet_subscription");
                    return;
                }
                touch_activity(&shared_state_clone.video_clients, &socket_id_clone);
                if let Ok(sub_cmd) = serde_json::from_value::<WebFleetSubscriptionCommand>(data) {
                    tracing::info!("Fleet subscription command: action={}", sub_cmd.action);

                    // Add to queue for processing by Dora task
                    if let Ok(mut queue) = shared_state_clone.fleet_subscription_command_queue.lock() {
                        queue.push(sub_cmd.clone());
                    }

                    // Update active rovers status in memory
                    let mut active_rovers = shared_state_clone.active_rovers_status.lock().unwrap();

                    match sub_cmd.action.as_str() {
                        "activate" => {
                            if let Some(entity_id) = &sub_cmd.entity_id {
                                if !active_rovers.active_rovers.contains(entity_id) {
                                    active_rovers.active_rovers.push(entity_id.clone());
                                    active_rovers.timestamp = SystemTime::now()
                                        .duration_since(UNIX_EPOCH)
                                        .unwrap()
                                        .as_millis() as u64;
                                    tracing::info!("Activated rover: {}", entity_id);
                                }
                            }
                        }
                        "deactivate" => {
                            if let Some(entity_id) = &sub_cmd.entity_id {
                                active_rovers.active_rovers.retain(|id| id != entity_id);
                                active_rovers.timestamp = SystemTime::now()
                                    .duration_since(UNIX_EPOCH)
                                    .unwrap()
                                    .as_millis() as u64;
                                tracing::info!("Deactivated rover: {}", entity_id);
                            }
                        }
                        "set_active" => {
                            if let Some(entity_ids) = &sub_cmd.entity_ids {
                                active_rovers.active_rovers = entity_ids.clone();
                                active_rovers.timestamp = SystemTime::now()
                                    .duration_since(UNIX_EPOCH)
                                    .unwrap()
                                    .as_millis() as u64;
                                tracing::info!("Set active rovers: {:?}", entity_ids);
                            }
                        }
                        _ => {
                            tracing::warn!("Unknown fleet subscription action: {}", sub_cmd.action);
                        }
                    }

                    // Broadcast updated active rovers status to all clients
                    let status_clone = active_rovers.clone();
                    drop(active_rovers); // Release lock before async operation

                    io_for_active_rovers
                        .emit("active_rovers_status", &status_clone)
                        .ok();
                    let voice_state = {
                        let mut runtime = shared_state_clone.voice_runtime.lock().unwrap();
                        runtime.sync_active_rovers(status_clone.active_rovers.clone());
                        runtime.config_state(current_timestamp_ms())
                    };
                    broadcast_tts_config_state(
                        &io_for_voice_clone,
                        &shared_state_clone.session_registry,
                        &voice_state,
                    );
                    tracing::info!("Active rovers status updated and broadcast");
                }
            },
        );

        // auth_refresh: client proactively refreshes its token before expiry
        let jwt_secret_clone = jwt_secret.clone();
        let session_registry_clone = shared_state.session_registry.clone();
        let auth_rate_limiter_clone = shared_state.auth_rate_limiter.clone();
        let socket_id_clone = socket_id.clone();
        socket.on(
            "auth_refresh",
            move |socket: SocketRef, Data::<Value>(data)| {
                if !auth_rate_limiter_clone.check_auth_attempt(&socket_id_clone) {
                    log_rate_limit_exceeded(&socket_id_clone, "auth_refresh");
                    socket.emit("auth_error", serde_json::json!({"reason": "rate_limited"})).ok();
                    socket.disconnect().ok();
                    return;
                }
                let current_token = data.get("token").and_then(|v| v.as_str()).unwrap_or("");
                match security::jwt::validate_token(current_token, &jwt_secret_clone) {
                    Ok(old_claims) => {
                        match security::jwt::generate_token(
                            &old_claims.sub,
                            &old_claims.role,
                            &jwt_secret_clone,
                            session_ttl_secs,
                        ) {
                            Ok((new_token, new_claims)) => {
                                session_registry_clone.register(&socket_id_clone, new_claims);
                                socket.emit("auth_token", new_token).ok();
                                tracing::debug!("Token refreshed for: {}", socket_id_clone);
                            }
                            Err(e) => {
                                tracing::error!("Token refresh generation failed: {}", e);
                                socket.emit("auth_error", serde_json::json!({"reason": "token_expired"})).ok();
                                socket.disconnect().ok();
                            }
                        }
                    }
                    Err(_) => {
                        session_registry_clone.remove(&socket_id_clone);
                        socket.emit("auth_error", serde_json::json!({"reason": "token_expired"})).ok();
                        socket.disconnect().ok();
                    }
                }
            },
        );

        let shared_state_clone = shared_state.clone();
        socket.on_disconnect(move |socket: SocketRef| {
            let socket_id = socket.id.to_string();
            tracing::info!("Client disconnected: {}", socket_id);

            remove_browser_media_demand(&shared_state_clone, &socket_id);
            let stopped_voice_streams = shared_state_clone.stt_bridge.close_owner(&socket_id);
            if stopped_voice_streams > 0 {
                tracing::info!(
                    stopped_voice_streams,
                    "Flushing browser speech streams for disconnected client"
                );
            }
            shared_state_clone.session_registry.remove(&socket_id);
            if let Ok(mut ingress) = shared_state_clone.walkie_ingress.lock() {
                ingress.remove_socket(&socket_id);
            }
            // Process-level cumulative: this client's per-client counters
            // are dropped by `remove_browser_media_demand`, but
            // the cumulative counters in `SharedState` retain all of the
            // work the bridge did for this client during its lifetime.
            shared_state_clone.audio_counters.record_client_disconnect();
        });

        } // end async move
    });

    (io, layer)
}

async fn security_headers(req: axum::http::Request<Body>, next: middleware::Next) -> Response {
    let mut response = next.run(req).await;
    let h = response.headers_mut();
    h.insert("x-frame-options", HeaderValue::from_static("DENY"));
    h.insert(
        "x-content-type-options",
        HeaderValue::from_static("nosniff"),
    );
    h.insert(
        "strict-transport-security",
        HeaderValue::from_static("max-age=31536000; includeSubDomains"),
    );
    h.insert("referrer-policy", HeaderValue::from_static("no-referrer"));
    response
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _guard = init_tracing();
    dotenv::dotenv().ok();

    tracing::info!("Starting Web Bridge...");

    // MongoDB startup
    let mongodb_uri =
        env::var("MONGODB_URI").unwrap_or_else(|_| "mongodb://localhost:27017".to_string());
    let mongodb_database = env::var("MONGODB_DATABASE").unwrap_or_else(|_| "db".to_string());

    tracing::info!(
        "Connecting to MongoDB at {}, database: {}",
        mongodb_uri,
        mongodb_database
    );
    let db = security::connect_db(&mongodb_uri, &mongodb_database)
        .await
        .map_err(|e| {
            tracing::error!("MongoDB connection failed: {}", e);
            e
        })?;
    let user_collection: Arc<Collection<security::User>> =
        Arc::new(db.collection("roboControlUser"));

    security::ensure_indexes(&user_collection).await?;
    security::seed_admin_user(&user_collection).await?;

    // Default credential guard
    let allow_default =
        env::var("ALLOW_DEFAULT_CREDENTIALS").unwrap_or_else(|_| "false".to_string()) == "true";
    if !allow_default {
        if let Ok(Some(admin)) = security::find_user(&user_collection, "admin").await {
            let still_default = tokio::task::spawn_blocking(move || {
                security::verify_password_blocking("password", &admin.password_hash)
            })
            .await
            .unwrap_or(false);

            if still_default {
                tracing::error!(
                    "FATAL: admin account still uses default password. \
                     Change it or set ALLOW_DEFAULT_CREDENTIALS=true"
                );
                std::process::exit(1);
            }
        }
    }

    let jwt_secret = Arc::new(load_or_generate_jwt_secret());
    let session_ttl_secs = env::var("SESSION_TTL_SECONDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3600u64);
    let trust_proxy =
        env::var("TRUST_PROXY_HEADERS").unwrap_or_else(|_| "false".to_string()) == "true";
    let idle_timeout_secs = env::var("IDLE_TIMEOUT_SECONDS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(1800);

    let (node, mut events) = DoraNode::init_from_env()?;
    let arm_command_output = DataId::from("arm_command".to_owned());
    let rover_command_output = DataId::from("rover_command".to_owned());
    let camera_command_output = DataId::from("camera_command".to_owned());
    let audio_command_output = DataId::from("audio_command".to_owned());
    let stream_command_output = DataId::from("stream_command".to_owned());
    let targeted_media_control_output = DataId::from("targeted_media_control".to_owned());
    let tracking_command_output = DataId::from("tracking_command".to_owned());
    let tts_command_output = DataId::from("tts_command".to_owned());
    let tts_config_command_output = DataId::from("tts_config_command".to_owned());
    let audio_stream_output = DataId::from("audio_stream".to_owned());
    let stt_outputs = SttOutputIds {
        audio: DataId::from("voice_command_audio".to_owned()),
        control: DataId::from("voice_command_control".to_owned()),
        status_request: DataId::from("stt_status_request".to_owned()),
    };
    let fleet_subscription_command_output = DataId::from("fleet_subscription_command".to_owned());
    let fleet_select_command_output = DataId::from("fleet_select_command".to_owned());
    let recording_session_command_output = DataId::from("recording_session_command".to_owned());
    let recording_clip_query_output = DataId::from("recording_clip_query".to_owned());
    let recording_playback_ticket_output = DataId::from("recording_playback_ticket".to_owned());
    let recording_delete_output = DataId::from("recording_delete".to_owned());

    let node_handle = Arc::new(Mutex::new(node));
    let shared_state = SharedState::new();
    let (io, layer) = setup_socketio(
        shared_state.clone(),
        node_handle.clone(),
        tts_config_command_output.clone(),
        user_collection,
        jwt_secret.clone(),
        session_ttl_secs,
        trust_proxy,
    );
    let io_handle = Arc::new(Mutex::new(Some(io.clone())));

    // Background sweep: disconnect sessions whose JWT has expired
    let sweep_registry = shared_state.session_registry.clone();
    let sweep_clients = shared_state.video_clients.clone();
    let sweep_media_registry = shared_state.media_demand_registry.clone();
    let sweep_media_queue = shared_state.targeted_media_control_queue.clone();
    let sweep_audio_counters = shared_state.audio_counters.clone();
    let sweep_stt = shared_state.stt_bridge.clone();
    let sweep_io = io.clone();
    tokio::spawn(async move {
        tracing::info!("Session sweep task started (interval: 60s)");
        loop {
            tokio::time::sleep(Duration::from_secs(60)).await;
            let expired = sweep_registry.sweep_expired();
            for socket_id in expired {
                tracing::debug!("Sweep: disconnecting expired session {}", socket_id);
                if let Ok(mut clients) = sweep_clients.lock() {
                    clients.retain(|client| client.socket_id != socket_id);
                }
                let transitions = sweep_media_registry
                    .lock()
                    .ok()
                    .map(|mut registry| {
                        registry.release_consumer_prefix(&browser_consumer_prefix(&socket_id))
                    })
                    .unwrap_or_default();
                enqueue_media_transitions(&sweep_media_queue, transitions);
                sweep_stt.close_owner(&socket_id);
                // Process-level cumulative: sweep-driven disconnects must
                // count toward the lifetime client_disconnects total, and
                // the per-client counters we just dropped are still
                // represented in the cumulative emit totals.
                sweep_audio_counters.record_client_disconnect();
                if let Ok(sid) = socket_id.parse() {
                    if let Some(ns) = sweep_io.of("/") {
                        if let Some(socket) = ns.get_socket(sid) {
                            socket
                                .emit("auth_error", serde_json::json!({"reason": "token_expired"}))
                                .ok();
                            socket.disconnect().ok();
                        }
                    }
                }
            }
        }
    });

    // Background sweep: disconnect idle clients exceeding IDLE_TIMEOUT_SECONDS
    let idle_clients = shared_state.video_clients.clone();
    let idle_media_registry = shared_state.media_demand_registry.clone();
    let idle_media_queue = shared_state.targeted_media_control_queue.clone();
    let idle_audio_counters = shared_state.audio_counters.clone();
    let idle_stt = shared_state.stt_bridge.clone();
    let idle_io = io.clone();
    tokio::spawn(async move {
        tracing::info!(
            "Idle sweep task started (timeout: {}s, interval: 60s)",
            idle_timeout_secs
        );
        let timeout = Duration::from_secs(idle_timeout_secs);
        loop {
            tokio::time::sleep(Duration::from_secs(60)).await;
            let idle_ids: Vec<String> = {
                // Snapshot (socket_id, elapsed) under the outer lock, then release.
                // last_activity.lock() is always acquired after video_clients.lock() so
                // lock ordering is consistent and deadlock-free.
                let clients = idle_clients.lock().unwrap();
                clients
                    .iter()
                    .filter_map(|c| {
                        let elapsed = c.last_activity.lock().unwrap().elapsed();
                        if elapsed >= timeout {
                            Some(c.socket_id.clone())
                        } else {
                            None
                        }
                    })
                    .collect()
            };
            for socket_id in idle_ids {
                tracing::info!("Idle sweep: disconnecting idle client {}", socket_id);
                if let Ok(mut clients) = idle_clients.lock() {
                    clients.retain(|client| client.socket_id != socket_id);
                }
                let transitions = idle_media_registry
                    .lock()
                    .ok()
                    .map(|mut registry| {
                        registry.release_consumer_prefix(&browser_consumer_prefix(&socket_id))
                    })
                    .unwrap_or_default();
                enqueue_media_transitions(&idle_media_queue, transitions);
                idle_stt.close_owner(&socket_id);
                // Process-level cumulative: idle-sweep-driven disconnects
                // must count toward the lifetime client_disconnects total.
                idle_audio_counters.record_client_disconnect();
                if let Ok(sid) = socket_id.parse() {
                    if let Some(ns) = idle_io.of("/") {
                        if let Some(socket) = ns.get_socket(sid) {
                            socket
                                .emit("auth_error", serde_json::json!({"reason": "idle_timeout"}))
                                .ok();
                            socket.disconnect().ok();
                        }
                    }
                }
            }
        }
    });

    let recording_sweep_io = io.clone();
    let recording_sweep_state = shared_state.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(1)).await;
            for (request_id, pending) in recording_sweep_state.recording.expire() {
                if pending.kind == RequestKind::Command {
                    release_recording_demand(&recording_sweep_state, &request_id);
                }
                emit_recording_timeout(
                    &recording_sweep_io,
                    &recording_sweep_state,
                    &request_id,
                    &pending,
                );
                tracing::warn!(
                    action = ?pending.kind,
                    request_id = %request_id,
                    outcome = "timeout",
                    reason = "correlation_deadline",
                    "recording request expired"
                );
            }
        }
    });

    // Start Socket.IO and ticketed playback server
    let recording_access_for_http = shared_state.recording_access.clone();
    let socketio_handle = tokio::spawn(async move {
        // Get allowed origins from environment
        let allowed_origins = parse_allowed_origins();
        warn_http_origins(&allowed_origins);
        tracing::info!("CORS allowed origins: {:?}", allowed_origins);

        let cors_layer = if allowed_origins.iter().any(|o| o == "*") {
            // Wildcard: AllowOrigin::list rejects "*", must use permissive()
            tracing::warn!("CORS wildcard origin configured - credentials disabled");
            CorsLayer::permissive()
        } else {
            // Convert to HeaderValue for CORS layer
            let origins: Vec<HeaderValue> = allowed_origins
                .iter()
                .filter_map(|origin| origin.parse().ok())
                .collect();

            if origins.is_empty() {
                tracing::warn!("No valid CORS origins configured, defaulting to localhost");
                CorsLayer::new()
                    .allow_origin([
                        "http://localhost:3000".parse::<HeaderValue>().unwrap(),
                        "http://localhost:5173".parse::<HeaderValue>().unwrap(),
                    ])
                    .allow_methods([Method::GET, Method::POST, Method::HEAD])
                    .allow_headers([
                        axum::http::header::CONTENT_TYPE,
                        axum::http::header::AUTHORIZATION,
                        axum::http::header::ACCEPT,
                        axum::http::header::RANGE,
                    ])
                    .expose_headers([
                        axum::http::header::CONTENT_RANGE,
                        axum::http::header::ACCEPT_RANGES,
                        axum::http::header::CONTENT_LENGTH,
                    ])
                    .allow_credentials(true)
            } else {
                CorsLayer::new()
                    .allow_origin(origins)
                    .allow_methods([Method::GET, Method::POST, Method::HEAD])
                    .allow_headers([
                        axum::http::header::CONTENT_TYPE,
                        axum::http::header::AUTHORIZATION,
                        axum::http::header::ACCEPT,
                        axum::http::header::RANGE,
                    ])
                    .expose_headers([
                        axum::http::header::CONTENT_RANGE,
                        axum::http::header::ACCEPT_RANGES,
                        axum::http::header::CONTENT_LENGTH,
                    ])
                    .allow_credentials(true)
            }
        };

        let app = axum::Router::new()
            .route(
                "/health",
                axum::routing::get(|| async { axum::Json(serde_json::json!({"status": "ok"})) }),
            )
            .route(
                "/recordings/playback/:ticket",
                axum::routing::get(recording_playback::serve_playback)
                    .head(recording_playback::serve_playback),
            )
            .with_state(recording_access_for_http)
            .layer(middleware::from_fn(security_headers))
            .layer(ServiceBuilder::new().layer(cors_layer).layer(layer));

        let bind_address = env::var("BIND_ADDRESS").unwrap_or_else(|_| "127.0.0.1".to_string());
        let port = env::var("SOCKET_IO_PORT").unwrap_or_else(|_| "3030".to_string());
        let addr = format!("{}:{}", bind_address, port);

        tracing::info!("Binding Socket.IO server to: {}", addr);

        let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();

        info!("Socket.IO server listening on http://{}", addr);
        axum::serve(listener, app).await.unwrap();
    });

    // Process commands
    let node_clone_arm = node_handle.clone();
    let node_clone_rover = node_clone_arm.clone();
    let node_clone_camera = node_clone_arm.clone();
    let node_clone_audio = node_clone_arm.clone();
    let node_clone_stream = node_clone_arm.clone();
    let node_clone_targeted_media = node_clone_arm.clone();
    let node_clone_tracking = node_clone_arm.clone();
    let node_clone_tts = node_clone_arm.clone();
    let node_clone_audio_stream = node_clone_arm.clone();
    let node_clone_stt = node_clone_arm.clone();
    let node_clone_fleet_sub = node_clone_arm.clone();
    let node_clone_recording = node_clone_arm.clone();
    let state_clone_arm = shared_state.clone();

    let state_clone_recording = shared_state.clone();
    let _recording_processor = tokio::spawn(async move {
        loop {
            let next = state_clone_recording
                .recording
                .commands
                .lock()
                .ok()
                .and_then(|mut queue| {
                    queue
                        .pop_front()
                        .and_then(|command| serde_json::to_vec(&command).ok())
                        .map(|bytes| (recording_session_command_output.clone(), bytes))
                })
                .or_else(|| {
                    state_clone_recording
                        .recording
                        .clip_queries
                        .lock()
                        .ok()
                        .and_then(|mut queue| {
                            queue
                                .pop_front()
                                .and_then(|query| serde_json::to_vec(&query).ok())
                                .map(|bytes| (recording_clip_query_output.clone(), bytes))
                        })
                })
                .or_else(|| {
                    state_clone_recording
                        .recording
                        .playback_queries
                        .lock()
                        .ok()
                        .and_then(|mut queue| {
                            queue
                                .pop_front()
                                .and_then(|request| serde_json::to_vec(&request).ok())
                                .map(|bytes| (recording_playback_ticket_output.clone(), bytes))
                        })
                });
            let next = next.or_else(|| {
                state_clone_recording
                    .recording
                    .delete_queries
                    .lock()
                    .ok()
                    .and_then(|mut queue| {
                        queue
                            .pop_front()
                            .and_then(|request| serde_json::to_vec(&request).ok())
                            .map(|bytes| (recording_delete_output.clone(), bytes))
                    })
            });
            if let Some((output, bytes)) = next {
                let arrow_data = BinaryArray::from_vec(vec![bytes.as_slice()]);
                if let Ok(mut node_guard) = node_clone_recording.lock() {
                    let _ = node_guard.send_output(output, Default::default(), arrow_data);
                }
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    });

    let arm_command_processor = tokio::spawn(async move {
        loop {
            if let Ok(mut queue) = state_clone_arm.arm_command_queue.lock() {
                if !queue.is_empty() {
                    let web_cmd = queue.remove(0);
                    if let Some(arm_cmd) = convert_web_command_to_arm_command(&web_cmd) {
                        let metadata = create_metadata();
                        let cmd_with_metadata = ArmCommandWithMetadata {
                            command: Some(arm_cmd),
                            metadata,
                        };

                        if let Ok(serialized) = serde_json::to_vec(&cmd_with_metadata) {
                            let arrow_data = BinaryArray::from_vec(vec![serialized.as_slice()]);
                            if let Ok(mut node_guard) = node_clone_arm.lock() {
                                let _ = node_guard.send_output(
                                    arm_command_output.clone(),
                                    Default::default(),
                                    arrow_data,
                                );
                            }
                        }
                    }
                }
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    });

    // Process rover commands
    let state_clone_rover = shared_state.clone();
    let rover_command_processor = tokio::spawn(async move {
        loop {
            if let Ok(mut queue) = state_clone_rover.rover_command_queue.lock() {
                if !queue.is_empty() {
                    let web_cmd = queue.remove(0);
                    if let Some(rover_cmd) = convert_web_command_to_rover_command(&web_cmd) {
                        let metadata = create_metadata();
                        let cmd_with_metadata = RoverCommandWithMetadata {
                            command: rover_cmd,
                            metadata,
                            target_entity_id: None, // web commands use selected rover
                        };

                        if let Ok(serialized) = serde_json::to_vec(&cmd_with_metadata) {
                            let arrow_data = BinaryArray::from_vec(vec![serialized.as_slice()]);
                            if let Ok(mut node_guard) = node_clone_rover.lock() {
                                let _ = node_guard.send_output(
                                    rover_command_output.clone(),
                                    Default::default(),
                                    arrow_data,
                                );
                            }
                        }
                    }
                }
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    });

    // Process camera control commands
    let state_clone_camera = shared_state.clone();
    let camera_command_processor = tokio::spawn(async move {
        loop {
            if let Ok(mut queue) = state_clone_camera.camera_command_queue.lock() {
                if !queue.is_empty() {
                    let web_cmd = queue.remove(0);
                    if let Some(camera_cmd) = convert_web_command_to_camera_command(&web_cmd) {
                        let timestamp = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_millis() as u64;

                        let camera_control = CameraControl {
                            command: camera_cmd,
                            timestamp,
                        };

                        if let Ok(serialized) = serde_json::to_vec(&camera_control) {
                            let arrow_data = BinaryArray::from_vec(vec![serialized.as_slice()]);
                            if let Ok(mut node_guard) = node_clone_camera.lock() {
                                let _ = node_guard.send_output(
                                    camera_command_output.clone(),
                                    Default::default(),
                                    arrow_data,
                                );
                            }
                        }
                    }
                }
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    });

    // Process audio control commands
    let state_clone_audio = shared_state.clone();
    let _audio_command_processor = tokio::spawn(async move {
        loop {
            if let Ok(mut queue) = state_clone_audio.audio_command_queue.lock() {
                if !queue.is_empty() {
                    let web_cmd = queue.remove(0);
                    if let Some(audio_cmd) = convert_web_command_to_audio_command(&web_cmd) {
                        let timestamp = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_millis() as u64;

                        let audio_control = AudioControl {
                            command: audio_cmd,
                            timestamp,
                        };

                        if let Ok(serialized) = serde_json::to_vec(&audio_control) {
                            let arrow_data = BinaryArray::from_vec(vec![serialized.as_slice()]);
                            if let Ok(mut node_guard) = node_clone_audio.lock() {
                                let _ = node_guard.send_output(
                                    audio_command_output.clone(),
                                    Default::default(),
                                    arrow_data,
                                );
                            }
                        }
                    }
                }
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    });

    // Process aggregate stream demand commands.
    let state_clone_stream = shared_state.clone();
    let _stream_command_processor = tokio::spawn(async move {
        loop {
            let next = state_clone_stream
                .stream_command_queue
                .lock()
                .ok()
                .and_then(|mut queue| {
                    if queue.is_empty() {
                        None
                    } else {
                        Some(queue.remove(0))
                    }
                });

            if let Some(stream_control) = next {
                if let Ok(serialized) = serde_json::to_vec(&stream_control) {
                    let arrow_data = BinaryArray::from_vec(vec![serialized.as_slice()]);
                    if let Ok(mut node_guard) = node_clone_stream.lock() {
                        let _ = node_guard.send_output(
                            stream_command_output.clone(),
                            Default::default(),
                            arrow_data,
                        );
                    }
                }
            }

            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    });

    // Target-aware media demand never relies on mutable fleet selection.
    let state_clone_targeted_media = shared_state.clone();
    let targeted_media_output_for_processor = targeted_media_control_output.clone();
    let targeted_media_control_processor = tokio::spawn(async move {
        loop {
            let next = state_clone_targeted_media
                .targeted_media_control_queue
                .lock()
                .ok()
                .and_then(|mut queue| (!queue.is_empty()).then(|| queue.remove(0)));
            if let Some(control) = next {
                if let Ok(serialized) = serde_json::to_vec(&control) {
                    let arrow_data = BinaryArray::from_vec(vec![serialized.as_slice()]);
                    if let Ok(mut node_guard) = node_clone_targeted_media.lock() {
                        let _ = node_guard.send_output(
                            targeted_media_output_for_processor.clone(),
                            Default::default(),
                            arrow_data,
                        );
                    }
                }
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    });

    // Process tracking commands
    let state_clone_tracking = shared_state.clone();
    let _tracking_command_processor = tokio::spawn(async move {
        loop {
            if let Ok(mut queue) = state_clone_tracking.tracking_command_queue.lock() {
                if !queue.is_empty() {
                    let web_cmd = queue.remove(0);
                    if let Some(tracking_cmd) = convert_web_command_to_tracking_command(&web_cmd) {
                        if let Ok(serialized) = serde_json::to_vec(&tracking_cmd) {
                            let arrow_data = BinaryArray::from_vec(vec![serialized.as_slice()]);
                            if let Ok(mut node_guard) = node_clone_tracking.lock() {
                                let _ = node_guard.send_output(
                                    tracking_command_output.clone(),
                                    Default::default(),
                                    arrow_data,
                                );
                            }
                        }
                    }
                }
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    });

    // Process TTS commands
    let state_clone_tts = shared_state.clone();
    let _tts_command_processor = tokio::spawn(async move {
        loop {
            if let Ok(mut queue) = state_clone_tts.tts_command_queue.lock() {
                if !queue.is_empty() {
                    let tts_cmd = queue.remove(0);
                    if let Ok(serialized) = serde_json::to_vec(&tts_cmd) {
                        let arrow_data = BinaryArray::from_vec(vec![serialized.as_slice()]);
                        if let Ok(mut node_guard) = node_clone_tts.lock() {
                            let _ = node_guard.send_output(
                                tts_command_output.clone(),
                                Default::default(),
                                arrow_data,
                            );
                        }
                    }
                }
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    });

    // Process audio stream from Web UI microphone (walkie-talkie mode)
    let state_clone_audio_stream = shared_state.clone();
    let audio_stream_processor = tokio::spawn(async move {
        let mut metrics_interval = tokio::time::interval(Duration::from_secs(5));
        metrics_interval.tick().await;
        loop {
            tokio::select! {
                _ = metrics_interval.tick() => {
                    if let Ok(mut ingress) = state_clone_audio_stream.walkie_ingress.lock() {
                        ingress.expire_streams(Instant::now());
                        ingress.log_metrics("periodic");
                    }
                }
                _ = tokio::time::sleep(Duration::from_millis(10)) => {
                    let frame = state_clone_audio_stream
                        .walkie_ingress
                        .lock()
                        .ok()
                        .and_then(|mut ingress| ingress.pop_front());
                    if let Some(frame) = frame {
                        let metadata = frame.parameters();
                        let arrow_data = Float32Array::from(frame.samples);
                        let sent = node_clone_audio_stream
                            .lock()
                            .map_err(|_| "Dora node lock poisoned".to_string())
                            .and_then(|mut node| {
                                node.send_output(
                                    audio_stream_output.clone(),
                                    metadata,
                                    arrow_data,
                                )
                                .map_err(|error| error.to_string())
                            });
                        if let Ok(mut ingress) = state_clone_audio_stream.walkie_ingress.lock() {
                            match sent {
                                Ok(()) => ingress.record_forwarded(),
                                Err(error) => {
                                    ingress.record_send_failure();
                                    tracing::warn!(%error, "Failed to forward walkie frame to Dora");
                                }
                            }
                        }
                    }
                }
            }
        }
    });

    // Preserve start/audio/stop ordering while keeping browser STT transport bounded.
    let state_clone_stt = shared_state.clone();
    let _stt_transport_processor = tokio::spawn(async move {
        let mut last_sweep = Instant::now();
        loop {
            if last_sweep.elapsed() >= Duration::from_secs(1) {
                state_clone_stt.stt_bridge.sweep();
                last_sweep = Instant::now();
            }
            for _ in 0..32 {
                let Some(message) = state_clone_stt.stt_bridge.pop_message() else {
                    break;
                };
                let mut delivered = false;
                if let Ok(mut node_guard) = node_clone_stt.lock() {
                    match send_dora_message(&mut node_guard, &stt_outputs, &message) {
                        Ok(()) => delivered = true,
                        Err(error) => {
                            tracing::error!(%error, "Failed to forward browser STT transport message; retrying in order");
                        }
                    }
                } else {
                    tracing::error!("Dora node lock poisoned; retrying browser STT message");
                }
                if delivered {
                    state_clone_stt.stt_bridge.complete_delivery();
                } else {
                    state_clone_stt.stt_bridge.retry_delivery(message);
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    break;
                }
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    });

    // Process fleet subscription commands
    let state_clone_fleet_sub = shared_state.clone();
    let node_for_fleet_sub = node_clone_fleet_sub.clone();
    let _fleet_subscription_processor = tokio::spawn(async move {
        loop {
            if let Ok(mut queue) = state_clone_fleet_sub
                .fleet_subscription_command_queue
                .lock()
            {
                if !queue.is_empty() {
                    let web_cmd = queue.remove(0);
                    tracing::debug!(
                        "Processing fleet subscription command: action={}",
                        web_cmd.action
                    );

                    // Convert Web command to FleetSubscriptionCommand
                    let fleet_cmd = match web_cmd.action.as_str() {
                        "activate" => {
                            if let Some(entity_id) = web_cmd.entity_id {
                                Some(FleetSubscriptionCommand::activate_rover(entity_id))
                            } else {
                                None
                            }
                        }
                        "deactivate" => {
                            if let Some(entity_id) = web_cmd.entity_id {
                                Some(FleetSubscriptionCommand::deactivate_rover(entity_id))
                            } else {
                                None
                            }
                        }
                        "set_active" => {
                            if let Some(entity_ids) = web_cmd.entity_ids {
                                Some(FleetSubscriptionCommand::set_active_rovers(entity_ids))
                            } else {
                                None
                            }
                        }
                        _ => None,
                    };

                    if let Some(cmd) = fleet_cmd {
                        if let Ok(serialized) = serde_json::to_vec(&cmd) {
                            let arrow_data = BinaryArray::from_vec(vec![serialized.as_slice()]);
                            if let Ok(mut node_guard) = node_for_fleet_sub.lock() {
                                let _ = node_guard.send_output(
                                    fleet_subscription_command_output.clone(),
                                    Default::default(),
                                    arrow_data,
                                );
                                tracing::info!("Sent fleet subscription command to zenoh_bridge");
                            }
                        }
                    }
                }
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    });

    // Process fleet select commands
    let state_clone_fleet_select = shared_state.clone();
    let node_clone_fleet_select = node_clone_fleet_sub.clone();
    let fleet_select_command_output_clone = fleet_select_command_output.clone();
    let _fleet_select_processor = tokio::spawn(async move {
        loop {
            if let Ok(mut queue) = state_clone_fleet_select.fleet_select_command_queue.lock() {
                if !queue.is_empty() {
                    let cmd = queue.remove(0);
                    tracing::debug!(
                        "Processing fleet select command: entity_id={}",
                        cmd.entity_id
                    );

                    if let Ok(serialized) = serde_json::to_vec(&cmd) {
                        let arrow_data = BinaryArray::from_vec(vec![serialized.as_slice()]);
                        if let Ok(mut node_guard) = node_clone_fleet_select.lock() {
                            let _ = node_guard.send_output(
                                fleet_select_command_output_clone.clone(),
                                Default::default(),
                                arrow_data,
                            );
                            tracing::info!(
                                "Sent fleet select command to orchestra-bridge: {}",
                                cmd.entity_id
                            );
                        }
                    }
                }
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    });

    tracing::info!("Web Bridge initialized - waiting for events...");

    // Event loop - handle video frames
    let state_for_video = shared_state.clone();
    let io_for_video = io_handle.clone();
    let mut video_emit_metrics = MetricWindow::new(Duration::from_secs(5));
    let mut video_age_metrics = MetricWindow::new(Duration::from_secs(5));
    let mut video_sequence = FrameSequenceTracker::default();
    let mut audio_emit_metrics = MetricWindow::new(Duration::from_secs(5));
    let mut audio_age_metrics = MetricWindow::new(Duration::from_secs(5));
    let mut audio_sequences: HashMap<String, AudioFrameSequenceTracker> = HashMap::new();
    let mut audio_errors = AudioDeliveryErrorCounts::default();
    let mut audio_sequence_drops = 0u64;

    loop {
        if let Some(event) = events.recv() {
            match event {
                Event::Input {
                    id, data, metadata, ..
                } => match id.as_str() {
                    "audio_frame" | "playback_audio_frame" => {
                        let source_kind = if id.as_str() == "playback_audio_frame" {
                            "rover_playback"
                        } else {
                            "rover_microphone"
                        };
                        let started = Instant::now();
                        if let Some(binary_array) = data.as_any().downcast_ref::<BinaryArray>() {
                            if binary_array.len() != 1 {
                                record_audio_error(
                                    &mut audio_emit_metrics,
                                    &mut audio_errors.input,
                                );
                                tracing::warn!("audio frame must contain exactly one payload");
                                continue;
                            }
                            let audio_bytes = binary_array.value(0);
                            match audio_frame_metadata(&metadata.parameters, audio_bytes.len()) {
                                Ok((origin, entity_id)) => {
                                    let sequence_key_base =
                                        entity_id.clone().unwrap_or_else(|| "direct".into());
                                    let sequence_key = format!("{source_kind}:{sequence_key_base}");
                                    match audio_sequences
                                        .entry(sequence_key)
                                        .or_default()
                                        .observe(origin)
                                    {
                                        Ok(observation) => {
                                            audio_emit_metrics
                                                .record_drops(observation.missing_frames);
                                            audio_sequence_drops = audio_sequence_drops
                                                .saturating_add(observation.missing_frames);
                                            // Process-level cumulative: these drops
                                            // would otherwise be lost on client
                                            // disconnect.
                                            state_for_video
                                                .audio_counters
                                                .record_sequence_drops(observation.missing_frames);
                                        }
                                        Err(error) => {
                                            record_audio_error(
                                                &mut audio_emit_metrics,
                                                &mut audio_errors.input,
                                            );
                                            tracing::warn!(%error, "rejected duplicate or regressed browser audio frame");
                                            continue;
                                        }
                                    }
                                    let frame_age_ms = record_capture_age(
                                        &mut audio_age_metrics,
                                        origin.capture_timestamp_ms,
                                    );
                                    if frame_age_ms.is_none() {
                                        record_audio_error(
                                            &mut audio_emit_metrics,
                                            &mut audio_errors.input,
                                        );
                                    }
                                    let audio_frame_data =
                                        browser_audio_frame_payload(origin, entity_id, source_kind);

                                    if let Ok(clients) = state_for_video.video_clients.lock() {
                                        for client in clients
                                            .iter()
                                            .filter(|client| client.should_send_audio())
                                        {
                                            let socket_id = match client.socket_id.parse() {
                                                Ok(socket_id) => socket_id,
                                                Err(error) => {
                                                    record_audio_error(
                                                        &mut audio_emit_metrics,
                                                        &mut audio_errors.socket_id,
                                                    );
                                                    tracing::warn!(%error, "invalid audio client socket ID");
                                                    continue;
                                                }
                                            };
                                            let io_guard = match io_for_video.lock() {
                                                Ok(guard) => guard,
                                                Err(_) => {
                                                    record_audio_error(
                                                        &mut audio_emit_metrics,
                                                        &mut audio_errors.routing,
                                                    );
                                                    continue;
                                                }
                                            };
                                            let Some(io) = io_guard.as_ref() else {
                                                record_audio_error(
                                                    &mut audio_emit_metrics,
                                                    &mut audio_errors.routing,
                                                );
                                                continue;
                                            };
                                            let Some(namespace) = io.of("/") else {
                                                record_audio_error(
                                                    &mut audio_emit_metrics,
                                                    &mut audio_errors.routing,
                                                );
                                                continue;
                                            };
                                            let Some(socket) = namespace.get_socket(socket_id)
                                            else {
                                                record_audio_error(
                                                    &mut audio_emit_metrics,
                                                    &mut audio_errors.socket_missing,
                                                );
                                                audio_emit_metrics.record_drop();
                                                client.mark_audio_dropped();
                                                // Process-level cumulative: this
                                                // drop would otherwise be lost when
                                                // the client disconnects.
                                                state_for_video.audio_counters.record_emit_drop();
                                                continue;
                                            };
                                            match socket
                                                .bin(vec![audio_bytes.to_vec()])
                                                .emit("audio_frame", audio_frame_data.clone())
                                            {
                                                Ok(()) => {
                                                    client.mark_audio_sent();
                                                    // Process-level cumulative: must
                                                    // move in lockstep with the
                                                    // per-client counter so totals
                                                    // survive client disconnect.
                                                    state_for_video
                                                        .audio_counters
                                                        .record_emit_success();
                                                    audio_emit_metrics.record(
                                                        started.elapsed(),
                                                        audio_bytes.len(),
                                                    );
                                                }
                                                Err(error) => {
                                                    record_audio_error(
                                                        &mut audio_emit_metrics,
                                                        &mut audio_errors.emit,
                                                    );
                                                    audio_emit_metrics.record_drop();
                                                    client.mark_audio_dropped();
                                                    // Process-level cumulative: emit
                                                    // errors must persist past
                                                    // disconnect.
                                                    state_for_video
                                                        .audio_counters
                                                        .record_emit_drop();
                                                    tracing::warn!(%error, "failed to emit audio frame");
                                                }
                                            }
                                        }
                                    } else {
                                        record_audio_error(
                                            &mut audio_emit_metrics,
                                            &mut audio_errors.routing,
                                        );
                                    }

                                    if let Some(snapshot) = audio_emit_metrics.snapshot_if_due() {
                                        tracing::info!(metric="audio_pipeline", stage="web_socket_emit",
                                            stream_id=%origin.stream_id, frame_id=origin.frame_id,
                                            frame_age_ms=?frame_age_ms, count=snapshot.count, bytes=snapshot.bytes,
                                            drops=snapshot.drops, errors=snapshot.errors,
                                            p50_us=snapshot.p50_us, p95_us=snapshot.p95_us,
                                            p99_us=snapshot.p99_us, max_us=snapshot.max_us);
                                    }
                                    if let Some(snapshot) = audio_age_metrics.snapshot_if_due() {
                                        tracing::info!(
                                            metric = "audio_pipeline",
                                            stage = "web_socket_emit_age",
                                            count = snapshot.count,
                                            p50_us = snapshot.p50_us,
                                            p95_us = snapshot.p95_us,
                                            p99_us = snapshot.p99_us,
                                            max_us = snapshot.max_us
                                        );
                                    }
                                }
                                Err(error) => {
                                    record_audio_error(
                                        &mut audio_emit_metrics,
                                        &mut audio_errors.input,
                                    );
                                    tracing::warn!(%error, "rejected invalid browser audio frame");
                                    if let Some(snapshot) = audio_emit_metrics.snapshot_if_due() {
                                        tracing::info!(
                                            metric = "audio_pipeline",
                                            stage = "web_socket_emit",
                                            count = snapshot.count,
                                            bytes = snapshot.bytes,
                                            drops = snapshot.drops,
                                            errors = snapshot.errors,
                                            p50_us = snapshot.p50_us,
                                            p95_us = snapshot.p95_us,
                                            p99_us = snapshot.p99_us,
                                            max_us = snapshot.max_us
                                        );
                                    }
                                }
                            }
                        } else {
                            record_audio_error(&mut audio_emit_metrics, &mut audio_errors.input);
                            tracing::error!("Invalid audio frame data type (expected BinaryArray)");
                        }
                    }
                    "video_frame" => {
                        // Extract metadata (added by video-encoder)
                        let width = metadata
                            .parameters
                            .get("width")
                            .and_then(|v| match v {
                                dora_node_api::Parameter::Integer(i) => Some(*i as u32),
                                _ => None,
                            })
                            .unwrap_or(640);

                        let height = metadata
                            .parameters
                            .get("height")
                            .and_then(|v| match v {
                                dora_node_api::Parameter::Integer(i) => Some(*i as u32),
                                _ => None,
                            })
                            .unwrap_or(480);

                        let codec = metadata
                            .parameters
                            .get("codec")
                            .and_then(|v| match v {
                                dora_node_api::Parameter::String(s) => Some(s.clone()),
                                _ => None,
                            })
                            .unwrap_or_else(|| "jpeg".to_string());

                        let capture_frame_id =
                            metadata
                                .parameters
                                .get("frame_id")
                                .and_then(|value| match value {
                                    dora_node_api::Parameter::Integer(value) => {
                                        u64::try_from(*value).ok()
                                    }
                                    _ => None,
                                });
                        let capture_timestamp_ms = metadata
                            .parameters
                            .get("capture_timestamp_ms")
                            .and_then(|value| match value {
                                dora_node_api::Parameter::Integer(value) => {
                                    u64::try_from(*value).ok()
                                }
                                _ => None,
                            });

                        // Get pre-encoded JPEG data from video-encoder (sent as BinaryArray)
                        if let Some(binary_array) = data.as_any().downcast_ref::<BinaryArray>() {
                            if binary_array.len() > 0 {
                                let emit_started = Instant::now();
                                let jpeg_data = binary_array.value(0).to_vec();

                                let (Some(capture_frame_id), Some(capture_timestamp_ms)) =
                                    (capture_frame_id, capture_timestamp_ms)
                                else {
                                    video_emit_metrics.record_error();
                                    tracing::error!("video frame missing capture identity");
                                    continue;
                                };
                                match video_sequence.observe(capture_frame_id) {
                                    Ok(missing) => video_emit_metrics.record_drops(missing),
                                    Err(()) => video_emit_metrics.record_error(),
                                }
                                let frame_age_ms = capture_age_ms(capture_timestamp_ms)
                                    .unwrap_or_else(|| {
                                        video_emit_metrics.record_error();
                                        0
                                    });
                                video_age_metrics.record(Duration::from_millis(frame_age_ms), 0);

                                tracing::debug!(
                                    "Received pre-encoded frame {}: {}x{} {} ({} bytes)",
                                    capture_frame_id,
                                    width,
                                    height,
                                    codec,
                                    jpeg_data.len()
                                );

                                if let Err(reason) = validate_browser_jpeg_payload(&jpeg_data) {
                                    video_emit_metrics.record_error();
                                    tracing::warn!(
                                        reason,
                                        frame_id = capture_frame_id,
                                        bytes = jpeg_data.len(),
                                        head = ?jpeg_data.get(..jpeg_data.len().min(4)),
                                        tail = ?jpeg_data.get(jpeg_data.len().saturating_sub(4)..),
                                        "rejected invalid browser jpeg payload"
                                    );
                                    continue;
                                }

                                let eligible_socket_ids = {
                                    if let Ok(clients) = state_for_video.video_clients.lock() {
                                        clients
                                            .iter()
                                            .filter_map(|client| {
                                                if client.should_send_video() {
                                                    client.mark_video_sent();
                                                    Some(client.socket_id.clone())
                                                } else {
                                                    client.mark_frame_dropped();
                                                    video_emit_metrics.record_drop();
                                                    None
                                                }
                                            })
                                            .collect::<Vec<_>>()
                                    } else {
                                        Vec::new()
                                    }
                                };

                                if !eligible_socket_ids.is_empty() {
                                    let frame_data = browser_video_frame_payload(
                                        capture_timestamp_ms,
                                        capture_frame_id,
                                        width,
                                        height,
                                        &codec,
                                    );

                                    for socket_id in eligible_socket_ids {
                                        if let Some(ref io) = *io_for_video.lock().unwrap() {
                                            if let Ok(parsed_socket_id) = socket_id.parse() {
                                                if let Some(socket) =
                                                    io.of("/").unwrap().get_socket(parsed_socket_id)
                                                {
                                                    match socket
                                                        .bin(vec![jpeg_data.clone()])
                                                        .emit("video_frame", frame_data.clone())
                                                    {
                                                        Ok(_) => {
                                                            video_emit_metrics.record(
                                                                emit_started.elapsed(),
                                                                jpeg_data.len(),
                                                            );
                                                        }
                                                        Err(error) => {
                                                            video_emit_metrics.record_error();
                                                            tracing::warn!(%error, "video frame emit failed");
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                if let Some(snapshot) = video_emit_metrics.snapshot_if_due() {
                                    let frame_age_ms = capture_age_ms(capture_timestamp_ms)
                                        .unwrap_or_else(|| {
                                            video_emit_metrics.record_error();
                                            0
                                        });
                                    tracing::info!(
                                        metric = "video_pipeline",
                                        stage = "web_emit",
                                        frame_id = capture_frame_id,
                                        frame_age_ms,
                                        count = snapshot.count,
                                        bytes = snapshot.bytes,
                                        drops = snapshot.drops,
                                        errors = snapshot.errors,
                                        p50_us = snapshot.p50_us,
                                        p95_us = snapshot.p95_us,
                                        p99_us = snapshot.p99_us,
                                        max_us = snapshot.max_us
                                    );
                                }
                                if let Some(snapshot) = video_age_metrics.snapshot_if_due() {
                                    tracing::info!(
                                        metric = "video_pipeline",
                                        stage = "web_receive_age",
                                        count = snapshot.count,
                                        p50_us = snapshot.p50_us,
                                        p95_us = snapshot.p95_us,
                                        p99_us = snapshot.p99_us,
                                        max_us = snapshot.max_us
                                    );
                                }
                            }
                        } else {
                            tracing::error!("Invalid video frame data type (expected BinaryArray from video-encoder)");
                        }
                    }
                    "detections" => {
                        // Handle detection frames from object_detector
                        if let Some(binary_array) = data.as_any().downcast_ref::<BinaryArray>() {
                            if binary_array.len() > 0 {
                                let detection_data = binary_array.value(0);

                                // Deserialize DetectionFrame
                                match serde_json::from_slice::<DetectionFrame>(detection_data) {
                                    Ok(detection_frame) => {
                                        // Forward detections to all connected clients
                                        if let Ok(_clients) = state_for_video.video_clients.lock() {
                                            if let Some(ref io) = *io_for_video.lock().unwrap() {
                                                // Emit to all clients via Socket.IO
                                                let _ = io.of("/").unwrap().emit(
                                                    "detections",
                                                    serde_json::to_value(&detection_frame).unwrap(),
                                                );
                                                tracing::info!(
                                                    event = "detections_forwarded",
                                                    frame_id = detection_frame.frame_id,
                                                    object_count = detection_frame.detections.len(),
                                                    width = detection_frame.width,
                                                    height = detection_frame.height
                                                );
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        tracing::error!("Failed to deserialize detections: {}", e);
                                    }
                                }
                            } else {
                                tracing::warn!("Received empty detections payload from Dora input");
                            }
                        } else {
                            tracing::error!(
                                "Invalid detections data type (expected BinaryArray from gst-camera)"
                            );
                        }
                    }
                    "tracked_detections" => {
                        // Handle tracked detection frames from object_tracker
                        if let Some(binary_array) = data.as_any().downcast_ref::<BinaryArray>() {
                            if binary_array.len() > 0 {
                                let detection_data = binary_array.value(0);

                                // Deserialize DetectionFrame with tracking IDs
                                match serde_json::from_slice::<DetectionFrame>(detection_data) {
                                    Ok(detection_frame) => {
                                        // Forward tracked detections to all connected clients
                                        if let Some(ref io) = *io_for_video.lock().unwrap() {
                                            let _ = io.of("/").unwrap().emit(
                                                "tracked_detections",
                                                serde_json::to_value(&detection_frame).unwrap(),
                                            );
                                            tracing::info!(
                                                event = "tracked_detections_forwarded",
                                                frame_id = detection_frame.frame_id,
                                                object_count = detection_frame.detections.len(),
                                                width = detection_frame.width,
                                                height = detection_frame.height
                                            );
                                        }
                                    }
                                    Err(e) => {
                                        tracing::error!(
                                            "Failed to deserialize tracked detections: {}",
                                            e
                                        );
                                    }
                                }
                            } else {
                                tracing::warn!(
                                    "Received empty tracked_detections payload from Dora input"
                                );
                            }
                        } else {
                            tracing::error!(
                                "Invalid tracked_detections data type (expected BinaryArray from gst-camera)"
                            );
                        }
                    }
                    "tracking_telemetry" => {
                        // Handle tracking telemetry from object_tracker
                        if let Some(binary_array) = data.as_any().downcast_ref::<BinaryArray>() {
                            if binary_array.len() > 0 {
                                let telemetry_data = binary_array.value(0);

                                // Deserialize TrackingTelemetry
                                match serde_json::from_slice::<TrackingTelemetry>(telemetry_data) {
                                    Ok(telemetry) => {
                                        // Forward telemetry to all connected clients
                                        if let Some(ref io) = *io_for_video.lock().unwrap() {
                                            let _ = io.of("/").unwrap().emit(
                                                "tracking_telemetry",
                                                serde_json::to_value(&telemetry).unwrap(),
                                            );
                                            tracing::info!(
                                                event = "tracking_telemetry_forwarded",
                                                state = ?telemetry.state,
                                                has_target = telemetry.target.is_some(),
                                                control_mode = ?telemetry.control_mode,
                                                timestamp = telemetry.timestamp
                                            );
                                        }
                                    }
                                    Err(e) => {
                                        tracing::error!(
                                            "Failed to deserialize tracking telemetry: {}",
                                            e
                                        );
                                    }
                                }
                            } else {
                                tracing::warn!(
                                    "Received empty tracking_telemetry payload from Dora input"
                                );
                            }
                        } else {
                            tracing::error!(
                                "Invalid tracking_telemetry data type (expected BinaryArray from gst-camera)"
                            );
                        }
                    }
                    "servo_telemetry" => {
                        // Handle servo telemetry from visual-servo-controller
                        // This includes distance estimation and control mode (auto/manual)
                        if let Some(binary_array) = data.as_any().downcast_ref::<BinaryArray>() {
                            if binary_array.len() > 0 {
                                let telemetry_data = binary_array.value(0);

                                // Deserialize TrackingTelemetry (with enhanced distance and mode)
                                match serde_json::from_slice::<TrackingTelemetry>(telemetry_data) {
                                    Ok(telemetry) => {
                                        // Forward enhanced telemetry to all connected clients
                                        if let Some(ref io) = *io_for_video.lock().unwrap() {
                                            let _ = io.of("/").unwrap().emit(
                                                "servo_telemetry",
                                                serde_json::to_value(&telemetry).unwrap(),
                                            );
                                        }
                                    }
                                    Err(e) => {
                                        tracing::error!(
                                            "Failed to deserialize servo telemetry: {}",
                                            e
                                        );
                                    }
                                }
                            }
                        }
                    }
                    "transcription" => {
                        if let Some(binary_array) = data.as_any().downcast_ref::<BinaryArray>() {
                            if binary_array.len() > 0 {
                                let transcription_data = binary_array.value(0);
                                match serde_json::from_slice::<SpeechTranscription>(
                                    transcription_data,
                                ) {
                                    Ok(transcription) => {
                                        tracing::info!(
                                            source = ?transcription.source_kind,
                                            stream_id = %transcription.stream_id,
                                            utterance_id = %transcription.utterance_id,
                                            "Final transcription received"
                                        );
                                        if let Some(ref io) = *io_for_video.lock().unwrap() {
                                            if let Some(namespace) = io.of("/") {
                                                let payload = serde_json::to_value(&transcription)
                                                    .unwrap_or(Value::Null);
                                                match state_for_video
                                                    .stt_bridge
                                                    .route_transcription(&transcription)
                                                {
                                                    TranscriptRoute::Browser { socket_id } => {
                                                        if !state_for_video
                                                            .session_registry
                                                            .is_valid(&socket_id)
                                                        {
                                                            tracing::warn!(
                                                                stream_id = %transcription.stream_id,
                                                                "Dropping browser transcription for unauthenticated owner"
                                                            );
                                                            continue;
                                                        }
                                                        let socket =
                                                            socket_id.parse().ok().and_then(
                                                                |sid| namespace.get_socket(sid),
                                                            );
                                                        if let Some(socket) = socket {
                                                            if let Err(error) = socket.emit(
                                                                "voice_command_transcription",
                                                                payload,
                                                            ) {
                                                                tracing::warn!(
                                                                    %error,
                                                                    stream_id = %transcription.stream_id,
                                                                    "Failed to emit private browser transcription"
                                                                );
                                                            }
                                                        } else {
                                                            tracing::warn!(
                                                                stream_id = %transcription.stream_id,
                                                                "Browser transcription owner disconnected before emit"
                                                            );
                                                        }
                                                    }
                                                    TranscriptRoute::RoverBroadcast => {
                                                        emit_authenticated(
                                                            namespace,
                                                            &state_for_video.session_registry,
                                                            "transcription",
                                                            payload,
                                                        );
                                                    }
                                                    TranscriptRoute::Drop(reason) => {
                                                        tracing::warn!(
                                                            %reason,
                                                            stream_id = %transcription.stream_id,
                                                            "Dropping unroutable transcription"
                                                        );
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        tracing::error!(
                                            "Failed to deserialize transcription: {}",
                                            e
                                        );
                                    }
                                }
                            }
                        }
                    }
                    "stt_status" => {
                        if let Some(binary_array) = data.as_any().downcast_ref::<BinaryArray>() {
                            if binary_array.len() > 0 {
                                match serde_json::from_slice::<SttStatus>(binary_array.value(0)) {
                                    Ok(status) => {
                                        state_for_video.stt_bridge.cache_status(status.clone());
                                        if let Some(ref io) = *io_for_video.lock().unwrap() {
                                            if let Some(namespace) = io.of("/") {
                                                emit_authenticated(
                                                    namespace,
                                                    &state_for_video.session_registry,
                                                    "stt_status",
                                                    serde_json::to_value(status)
                                                        .unwrap_or(Value::Null),
                                                );
                                            }
                                        }
                                    }
                                    Err(error) => {
                                        tracing::error!(%error, "Failed to deserialize STT status");
                                    }
                                }
                            }
                        }
                    }
                    "voice_status" => {
                        if let Some(binary_array) = data.as_any().downcast_ref::<BinaryArray>() {
                            if binary_array.len() > 0 {
                                match serde_json::from_slice::<VoiceStatus>(binary_array.value(0)) {
                                    Ok(status) => {
                                        if let Err(error) = status.validate() {
                                            tracing::warn!(%error, "Rejected invalid voice status");
                                            continue;
                                        }
                                        let accepted = {
                                            let mut runtime =
                                                state_for_video.voice_runtime.lock().unwrap();
                                            runtime.record_voice_status(status.clone())
                                        };
                                        if !accepted {
                                            tracing::debug!(
                                                entity_id = %status.entity_id,
                                                revision = status.applied_revision,
                                                timestamp = status.timestamp,
                                                "Ignored stale voice status"
                                            );
                                            continue;
                                        }
                                        if let Some(ref io) = *io_for_video.lock().unwrap() {
                                            if let Some(namespace) = io.of("/") {
                                                emit_authenticated(
                                                    namespace,
                                                    &state_for_video.session_registry,
                                                    "voice_status",
                                                    serde_json::to_value(&status)
                                                        .unwrap_or(Value::Null),
                                                );
                                            }
                                            if let Some(namespace) = io.of("/") {
                                                let config_state =
                                                    current_tts_config_state(&state_for_video);
                                                emit_authenticated(
                                                    namespace,
                                                    &state_for_video.session_registry,
                                                    "tts_config_state",
                                                    serde_json::to_value(config_state)
                                                        .unwrap_or(Value::Null),
                                                );
                                            }
                                        }
                                    }
                                    Err(error) => {
                                        tracing::error!(
                                            %error,
                                            "Failed to deserialize voice status"
                                        );
                                    }
                                }
                            }
                        }
                    }
                    "tts_command_result" => {
                        if let Some(binary_array) = data.as_any().downcast_ref::<BinaryArray>() {
                            if binary_array.len() > 0 {
                                match serde_json::from_slice::<TtsCommandResult>(
                                    binary_array.value(0),
                                ) {
                                    Ok(result) => {
                                        if let Err(error) = result.validate() {
                                            tracing::warn!(%error, "Rejected invalid TTS result");
                                            continue;
                                        }
                                        if let Some(ref io) = *io_for_video.lock().unwrap() {
                                            if let Some(namespace) = io.of("/") {
                                                emit_authenticated(
                                                    namespace,
                                                    &state_for_video.session_registry,
                                                    "tts_command_result",
                                                    serde_json::to_value(result)
                                                        .unwrap_or(Value::Null),
                                                );
                                            }
                                        }
                                    }
                                    Err(error) => {
                                        tracing::error!(%error, "Failed to deserialize TTS result");
                                    }
                                }
                            }
                        }
                    }
                    "recording_session_command_result" => {
                        if let Some(binary_array) = data.as_any().downcast_ref::<BinaryArray>() {
                            if binary_array.len() > 0 {
                                let result = match serde_json::from_slice::<
                                    RecordingSessionCommandResult,
                                >(
                                    binary_array.value(0)
                                ) {
                                    Ok(result) => result,
                                    Err(error) => {
                                        tracing::warn!(%error, "rejected malformed recording command result");
                                        continue;
                                    }
                                };
                                if let Err(error) = result.validate() {
                                    tracing::warn!(%error, "rejected invalid recording command result");
                                    continue;
                                }
                                if !result.accepted {
                                    release_recording_demand(&state_for_video, &result.request_id);
                                }
                                if let Some(pending) =
                                    state_for_video.recording.take(&result.request_id)
                                {
                                    if let Ok(payload) = serde_json::to_value(&result) {
                                        if let Some(io) =
                                            io_for_video.lock().ok().and_then(|io| io.clone())
                                        {
                                            emit_recording_to_owner(
                                                &io,
                                                &state_for_video,
                                                &pending.socket_id,
                                                "recording_session_command_result",
                                                payload,
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                    "recording_session_status" => {
                        if let Some(binary_array) = data.as_any().downcast_ref::<BinaryArray>() {
                            if binary_array.len() > 0 {
                                let status = match serde_json::from_slice::<RecordingSessionStatus>(
                                    binary_array.value(0),
                                ) {
                                    Ok(status) => status,
                                    Err(error) => {
                                        tracing::warn!(%error, "rejected malformed recording status");
                                        continue;
                                    }
                                };
                                if let Err(error) = status.validate() {
                                    tracing::warn!(%error, "rejected invalid recording status");
                                    continue;
                                }
                                if matches!(
                                    status.state,
                                    RecordingSessionState::Starting
                                        | RecordingSessionState::Recording
                                        | RecordingSessionState::Stopping
                                ) {
                                    if status.state == RecordingSessionState::Starting {
                                        rename_recording_demand(
                                            &state_for_video,
                                            &status.request_id,
                                            &status.recording_id,
                                        );
                                    }
                                    state_for_video
                                        .recording
                                        .remember_active(&status.recording_id, &status.entity_id);
                                } else {
                                    release_recording_demand(
                                        &state_for_video,
                                        &status.recording_id,
                                    );
                                    state_for_video
                                        .recording
                                        .forget_active(&status.recording_id);
                                }
                                let status_changed =
                                    state_for_video.recording.cache_status(status.clone());
                                if status_changed {
                                    if let Some(io) =
                                        io_for_video.lock().ok().and_then(|io| io.clone())
                                    {
                                        if let Some(namespace) = io.of("/") {
                                            emit_authenticated(
                                                namespace,
                                                &state_for_video.session_registry,
                                                "recording_session_status",
                                                serde_json::to_value(status).unwrap_or(Value::Null),
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                    "recording_clip_list_result" => {
                        if let Some(binary_array) = data.as_any().downcast_ref::<BinaryArray>() {
                            if binary_array.len() > 0 {
                                let result = match serde_json::from_slice::<RecordingClipQueryResult>(
                                    binary_array.value(0),
                                ) {
                                    Ok(result) => result,
                                    Err(error) => {
                                        tracing::warn!(%error, "rejected malformed recording clip result");
                                        continue;
                                    }
                                };
                                if let Err(error) = result.validate() {
                                    tracing::warn!(%error, "rejected invalid recording clip result");
                                    continue;
                                }
                                if let Some(pending) =
                                    state_for_video.recording.take(&result.request_id)
                                {
                                    if let Some(io) =
                                        io_for_video.lock().ok().and_then(|io| io.clone())
                                    {
                                        emit_recording_to_owner(
                                            &io,
                                            &state_for_video,
                                            &pending.socket_id,
                                            "recording_clip_list_result",
                                            serde_json::to_value(result).unwrap_or(Value::Null),
                                        );
                                    }
                                }
                            }
                        }
                    }
                    "recording_playback_clip_result" => {
                        if let Some(binary_array) = data.as_any().downcast_ref::<BinaryArray>() {
                            if binary_array.len() > 0 {
                                let result = match serde_json::from_slice::<RecordingClipQueryResult>(
                                    binary_array.value(0),
                                ) {
                                    Ok(result) => result,
                                    Err(error) => {
                                        tracing::warn!(%error, "rejected malformed playback lookup result");
                                        continue;
                                    }
                                };
                                if let Err(error) = result.validate() {
                                    tracing::warn!(%error, "rejected invalid playback lookup result");
                                    continue;
                                }
                                if let Some(pending) =
                                    state_for_video.recording.take(&result.request_id)
                                {
                                    if let Some(io) =
                                        io_for_video.lock().ok().and_then(|io| io.clone())
                                    {
                                        let payload = if result.accepted && result.clips.len() == 1
                                        {
                                            match state_for_video
                                                .recording_access
                                                .issue(&result.request_id, &result.clips[0])
                                            {
                                                Ok(ticket) => serde_json::to_value(ticket)
                                                    .unwrap_or(Value::Null),
                                                Err(error) => serde_json::json!({
                                                    "protocol_version": robo_rover_lib::RECORDING_PROTOCOL_VERSION,
                                                    "request_id": result.request_id,
                                                    "accepted": false,
                                                    "reason_code": "storage_unavailable",
                                                    "detail": error.chars().take(256).collect::<String>(),
                                                }),
                                            }
                                        } else {
                                            serde_json::json!({
                                                "protocol_version": robo_rover_lib::RECORDING_PROTOCOL_VERSION,
                                                "request_id": result.request_id,
                                                "accepted": false,
                                                "reason_code": result.reason_code.map(|code| serde_json::to_value(code).unwrap_or(Value::Null)).unwrap_or(serde_json::json!("not_found")),
                                                "detail": result.detail,
                                            })
                                        };
                                        emit_recording_to_owner(
                                            &io,
                                            &state_for_video,
                                            &pending.socket_id,
                                            "recording_playback_ticket_result",
                                            payload,
                                        );
                                    }
                                }
                            }
                        }
                    }
                    "recording_delete_result" => {
                        if let Some(binary_array) = data.as_any().downcast_ref::<BinaryArray>() {
                            if binary_array.len() > 0 {
                                let result = match serde_json::from_slice::<RecordingDeleteResult>(
                                    binary_array.value(0),
                                ) {
                                    Ok(result) => result,
                                    Err(error) => {
                                        tracing::warn!(%error, "rejected malformed recording delete result");
                                        continue;
                                    }
                                };
                                if let Err(error) = result.validate() {
                                    tracing::warn!(%error, "rejected invalid recording delete result");
                                    continue;
                                }
                                if let Some(pending) =
                                    state_for_video.recording.take(&result.request_id)
                                {
                                    if result.accepted {
                                        if let Some(recording_id) = result.recording_id.as_deref() {
                                            state_for_video.recording_access.revoke(recording_id);
                                        }
                                    }
                                    if let Some(io) =
                                        io_for_video.lock().ok().and_then(|io| io.clone())
                                    {
                                        emit_recording_to_owner(
                                            &io,
                                            &state_for_video,
                                            &pending.socket_id,
                                            "recording_delete_result",
                                            serde_json::to_value(result).unwrap_or(Value::Null),
                                        );
                                    }
                                }
                            }
                        }
                    }
                    "performance_metrics" => {
                        // Handle performance metrics from performance_monitor
                        // Only forward if monitoring is enabled
                        let monitoring_enabled = *state_for_video
                            .performance_monitoring_enabled
                            .lock()
                            .unwrap();

                        if monitoring_enabled {
                            if let Some(binary_array) = data.as_any().downcast_ref::<BinaryArray>()
                            {
                                if binary_array.len() > 0 {
                                    let metrics_data = binary_array.value(0);

                                    // Deserialize SystemMetrics
                                    match serde_json::from_slice::<SystemMetrics>(metrics_data) {
                                        Ok(metrics) => {
                                            tracing::trace!(
                                                "Performance metrics - CPU: {:.1}%, Memory: {:.0}MB, FPS: {:.1}, Latency: {:.1}ms",
                                                metrics.total_cpu_percent,
                                                metrics.total_memory_mb,
                                                metrics.dataflow_fps,
                                                metrics.end_to_end_latency_ms
                                            );

                                            // Forward metrics to all connected clients
                                            if let Some(ref io) = *io_for_video.lock().unwrap() {
                                                let _ = io.of("/").unwrap().emit(
                                                    "performance_metrics",
                                                    serde_json::to_value(&metrics).unwrap(),
                                                );
                                            }
                                        }
                                        Err(e) => {
                                            tracing::error!(
                                                "Failed to deserialize performance metrics: {}",
                                                e
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                    other => {
                        tracing::warn!("Unhandled dora input: '{}'", other);
                    }
                },
                Event::Stop(_) => {
                    let transitions = state_for_video
                        .media_demand_registry
                        .lock()
                        .ok()
                        .map(|mut registry| registry.shutdown())
                        .unwrap_or_default();
                    for control in transitions
                        .into_iter()
                        .map(|transition| transition.targeted_control())
                    {
                        if let Ok(serialized) = serde_json::to_vec(&control) {
                            let arrow_data = BinaryArray::from_vec(vec![serialized.as_slice()]);
                            if let Ok(mut node) = node_handle.lock() {
                                let _ = node.send_output(
                                    targeted_media_control_output.clone(),
                                    Default::default(),
                                    arrow_data,
                                );
                            }
                        }
                    }
                    if let Ok(ingress) = state_for_video.walkie_ingress.lock() {
                        ingress.log_metrics("shutdown");
                    }
                    let stt_metrics = state_for_video.stt_bridge.metrics();
                    tracing::info!(
                        metric = "browser_stt_transport_total",
                        queue_drops = stt_metrics.queue_drops,
                        terminated_streams = stt_metrics.terminated_streams,
                        expired_streams = stt_metrics.expired_streams,
                        late_transcriptions = stt_metrics.late_transcriptions,
                        status_requests = stt_metrics.status_requests,
                        "Browser STT transport shutdown totals"
                    );
                    // Process-level cumulative totals survive client
                    // disconnects, so the shutdown log now reports the
                    // actual lifetime work the bridge performed rather
                    // than just the work for the still-connected subset
                    // of clients. The per-client counters remain in
                    // `ClientState` for live per-client debugging.
                    let cumulative = state_for_video.audio_counters.cumulative_totals();
                    let still_connected = state_for_video
                        .video_clients
                        .lock()
                        .map(|clients| clients.len())
                        .unwrap_or(0);
                    let per_client_sends: u64 = state_for_video
                        .video_clients
                        .lock()
                        .ok()
                        .map(|clients| {
                            clients
                                .iter()
                                .filter_map(|c| c.audio_frames_sent.lock().ok().map(|v| *v))
                                .sum()
                        })
                        .unwrap_or(0);
                    let per_client_drops: u64 = state_for_video
                        .video_clients
                        .lock()
                        .ok()
                        .map(|clients| {
                            clients
                                .iter()
                                .filter_map(|c| c.audio_frames_dropped.lock().ok().map(|v| *v))
                                .sum()
                        })
                        .unwrap_or(0);
                    tracing::info!(
                        metric = "audio_pipeline_total",
                        stage = "web_socket_emit",
                        deliveries_emitted = cumulative.frames_sent,
                        sequence_drops = audio_sequence_drops,
                        client_drops = cumulative.frames_dropped,
                        per_client_sends_still_connected = per_client_sends,
                        per_client_drops_still_connected = per_client_drops,
                        still_connected_clients = still_connected,
                        lifetime_client_disconnects = cumulative.client_disconnects,
                        input_errors = audio_errors.input,
                        socket_id_errors = audio_errors.socket_id,
                        socket_missing_errors = audio_errors.socket_missing,
                        routing_errors = audio_errors.routing,
                        emit_errors = audio_errors.emit,
                        errors = audio_errors.total()
                    );
                    tracing::info!("Stop event received");
                    break;
                }
                _ => {}
            }
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    // Cleanup
    socketio_handle.abort();
    arm_command_processor.abort();
    rover_command_processor.abort();
    camera_command_processor.abort();
    audio_stream_processor.abort();
    targeted_media_control_processor.abort();
    tracing::info!("Web Bridge shutdown complete");

    Ok(())
}

fn convert_web_command_to_arm_command(web_cmd: &WebArmCommand) -> Option<ArmCommand> {
    match web_cmd.command_type.as_str() {
        "joint_position" => {
            if let Some(ref positions) = web_cmd.joint_positions {
                Some(ArmCommand::JointPosition {
                    joint_angles: vec![
                        positions.shoulder_pan,
                        positions.shoulder_lift,
                        positions.elbow_flex,
                        positions.wrist_flex,
                        positions.wrist_roll,
                        positions.gripper,
                    ],
                    max_velocity: None,
                })
            } else {
                None
            }
        }
        "home" => Some(ArmCommand::Home),
        "stop" => Some(ArmCommand::Stop),
        _ => None,
    }
}

fn convert_web_command_to_rover_command(web_cmd: &WebRoverCommand) -> Option<RoverCommand> {
    use std::time::{SystemTime, UNIX_EPOCH};
    use uuid;

    match web_cmd.command_type.as_str() {
        "wheel_positions" => {
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64;
            let command_id = uuid::Uuid::new_v4().to_string();

            Some(RoverCommand::JointPositions {
                wheel1: web_cmd.wheel1.unwrap_or(0.0),
                wheel2: web_cmd.wheel2.unwrap_or(0.0),
                wheel3: web_cmd.wheel3.unwrap_or(0.0),
                timestamp,
                command_id,
            })
        }
        "stop" => {
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64;
            let command_id = uuid::Uuid::new_v4().to_string();

            Some(RoverCommand::Stop {
                timestamp,
                command_id,
            })
        }
        _ => None,
    }
}

fn convert_web_command_to_camera_command(web_cmd: &WebCameraCommand) -> Option<CameraAction> {
    match web_cmd.command.as_str() {
        "start" => Some(CameraAction::Start),
        "stop" => Some(CameraAction::Stop),
        _ => None,
    }
}

fn convert_web_command_to_audio_command(web_cmd: &WebAudioCommand) -> Option<AudioAction> {
    match web_cmd.command.as_str() {
        "start" => Some(AudioAction::Start),
        "stop" => Some(AudioAction::Stop),
        _ => None,
    }
}

fn convert_web_command_to_tracking_command(
    web_cmd: &WebTrackingCommand,
) -> Option<TrackingCommand> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    match web_cmd.command_type.as_str() {
        "enable_detection" => Some(TrackingCommand::EnableDetection { timestamp }),
        "disable_detection" => Some(TrackingCommand::DisableDetection { timestamp }),
        "enable" => Some(TrackingCommand::Enable { timestamp }),
        "disable" => Some(TrackingCommand::Disable { timestamp }),
        "select_target" => {
            if let Some(tracking_id) = web_cmd.tracking_id {
                Some(TrackingCommand::SelectTargetById {
                    tracking_id,
                    timestamp,
                })
            } else if let Some(detection_index) = web_cmd.detection_index {
                Some(TrackingCommand::SelectTarget {
                    detection_index,
                    timestamp,
                })
            } else {
                None
            }
        }
        "clear_target" => Some(TrackingCommand::ClearTarget { timestamp }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use socketioxide::packet::{Packet, PacketData};

    #[test]
    fn video_frame_packet_uses_binary_attachment_not_json_byte_array() {
        let payload = browser_video_frame_payload(1_717_000_000_000, 42, 640, 480, "jpeg");
        let jpeg = vec![0xff, 0xd8, 0xff, 0xd9];
        let packet = Packet::bin_event("/", "video_frame", payload, vec![jpeg.clone()]);

        match packet.inner {
            PacketData::BinaryEvent(event, binary, None) => {
                assert_eq!(event, "video_frame");
                assert_eq!(binary.bin, vec![jpeg]);
                assert_eq!(binary.data[0]["frame_id"], 42);
                assert!(binary.data[0].get("data").is_none());
                assert_eq!(binary.data[1]["_placeholder"], true);
                assert_eq!(binary.data[1]["num"], 0);
            }
            other => panic!("expected binary event, got {other:?}"),
        }
    }

    #[test]
    fn browser_jpeg_payload_validation_accepts_marker_bounded_payload() {
        let jpeg = [0xff, 0xd8, 0x11, 0x22, 0xff, 0xd9];

        assert!(validate_browser_jpeg_payload(&jpeg).is_ok());
    }

    #[test]
    fn browser_jpeg_payload_validation_rejects_empty_or_truncated_payload() {
        assert!(validate_browser_jpeg_payload(&[]).is_err());
        assert!(validate_browser_jpeg_payload(&[0xff, 0xd8, 0x11, 0x22]).is_err());
        assert!(validate_browser_jpeg_payload(&[0x00, 0xd8, 0xff, 0xd9]).is_err());
    }

    #[test]
    fn browser_audio_packet_uses_one_binary_attachment_and_metadata_only() {
        let origin = AudioFrameMetadata {
            stream_id: uuid::Uuid::from_u128(42),
            frame_id: 7,
            capture_timestamp_ms: 1_717_000_000_000,
            sample_rate: 16_000,
            channels: 1,
            sample_count: 800,
            format: PcmSampleFormat::S16Le,
        };
        let payload =
            browser_audio_frame_payload(origin, Some("rover-a".into()), "rover_microphone");
        let pcm = vec![0u8; 1_600];
        let packet = Packet::bin_event("/", "audio_frame", payload, vec![pcm.clone()]);

        match packet.inner {
            PacketData::BinaryEvent(event, binary, None) => {
                assert_eq!(event, "audio_frame");
                assert_eq!(binary.bin, vec![pcm]);
                assert_eq!(binary.data[0]["protocol_version"], 1);
                assert_eq!(binary.data[0]["stream_id"], origin.stream_id.to_string());
                assert_eq!(binary.data[0]["frame_id"], origin.frame_id);
                assert_eq!(
                    binary.data[0]["capture_timestamp_ms"],
                    origin.capture_timestamp_ms
                );
                assert_eq!(binary.data[0]["sample_count"], origin.sample_count);
                assert_eq!(binary.data[0]["duration_ms"], 50.0);
                assert_eq!(binary.data[0]["entity_id"], "rover-a");
                assert!(binary.data[0].get("data").is_none());
                assert_eq!(binary.data[1]["_placeholder"], true);
                assert_eq!(binary.data[1]["num"], 0);
            }
            other => panic!("expected binary event, got {other:?}"),
        }
    }

    #[test]
    fn browser_pcm_validation_accepts_standard_s16le_frame() {
        let metadata = AudioFrameMetadata {
            stream_id: uuid::Uuid::from_u128(42),
            frame_id: 7,
            capture_timestamp_ms: 1_717_000_000_000,
            sample_rate: 16_000,
            channels: 1,
            sample_count: 800,
            format: PcmSampleFormat::S16Le,
        };

        assert!(validate_browser_pcm_payload(metadata, 1_600).is_ok());
    }

    #[test]
    fn browser_pcm_validation_rejects_format_and_length_mismatches() {
        let metadata = AudioFrameMetadata {
            stream_id: uuid::Uuid::from_u128(42),
            frame_id: 7,
            capture_timestamp_ms: 1_717_000_000_000,
            sample_rate: 16_000,
            channels: 1,
            sample_count: 2,
            format: PcmSampleFormat::S16Le,
        };

        assert!(validate_browser_pcm_payload(metadata, 3).is_err());
        assert!(validate_browser_pcm_payload(metadata, 6).is_err());
        assert!(validate_browser_pcm_payload(
            AudioFrameMetadata {
                format: PcmSampleFormat::F32Le,
                ..metadata
            },
            8,
        )
        .is_err());
    }

    #[test]
    fn audio_drop_accounting_is_separate_from_video_drops() {
        let client = ClientState::new("client-a".to_owned());

        client.mark_audio_dropped();

        assert_eq!(*client.audio_frames_dropped.lock().unwrap(), 1);
        assert_eq!(*client.frames_dropped.lock().unwrap(), 0);
    }

    #[test]
    fn inactive_rover_cannot_become_browser_demand_target() {
        let active = ActiveRoversStatus::new(vec!["rover-a".into()]);
        assert!(active_rover_status_includes(&active, "rover-a"));
        assert!(!active_rover_status_includes(&active, "rover-b"));
    }

    #[test]
    fn walkie_activity_expires_after_ttl() {
        let mut admission = VoiceAdmissionState::default();
        let start = Instant::now();

        admission.note_walkie_frame("rover-kiwi", start);
        assert!(admission.is_walkie_active("rover-kiwi", start));
        assert!(admission.is_walkie_active(
            "rover-kiwi",
            start + WALKIE_ACTIVITY_TTL - Duration::from_millis(1),
        ));
        assert!(!admission.is_walkie_active(
            "rover-kiwi",
            start + WALKIE_ACTIVITY_TTL + Duration::from_millis(1),
        ));
    }

    #[test]
    fn rejected_tts_ack_uses_contract_reason_codes() {
        let ack = build_tts_ack(
            "550e8400-e29b-41d4-a716-446655440000",
            "rover-kiwi",
            TtsAckState::Rejected,
            Some(VoiceReasonCode::WalkieActive),
            Some("walkie stream is active".into()),
        );

        assert_eq!(ack.target_entity_id, "rover-kiwi");
        assert_eq!(ack.state, TtsAckState::Rejected);
        assert_eq!(ack.reason_code, Some(VoiceReasonCode::WalkieActive));
        ack.validate().unwrap();
    }
}

fn current_timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

fn current_tts_config_state(shared_state: &SharedState) -> TtsConfigState {
    shared_state
        .voice_runtime
        .lock()
        .map(|runtime| runtime.config_state(current_timestamp_ms()))
        .unwrap_or_else(|_| TtsConfigState {
            desired_revision: 0,
            desired_config: TtsRuntimeConfig::default(),
            applied_rovers: 0,
            active_rovers: 0,
            rovers: Vec::new(),
            timestamp: current_timestamp_ms(),
        })
}

fn current_voice_statuses(shared_state: &SharedState) -> Vec<VoiceStatus> {
    shared_state
        .voice_runtime
        .lock()
        .map(|runtime| runtime.active_voice_statuses(current_timestamp_ms()))
        .unwrap_or_default()
}

fn emit_tts_config_state(socket: &SocketRef, state: &TtsConfigState) {
    if let Err(error) = state.validate() {
        tracing::warn!(%error, "refusing to emit invalid tts_config_state");
        return;
    }
    match serde_json::to_value(state) {
        Ok(payload) => {
            if let Err(error) = socket.emit("tts_config_state", payload) {
                tracing::warn!(%error, "failed to emit tts_config_state");
            }
        }
        Err(error) => {
            tracing::warn!(%error, "failed to serialize tts_config_state");
        }
    }
}

fn emit_voice_status(socket: &SocketRef, status: &VoiceStatus) {
    if let Err(error) = status.validate() {
        tracing::warn!(%error, "refusing to emit invalid voice_status");
        return;
    }
    match serde_json::to_value(status) {
        Ok(payload) => {
            if let Err(error) = socket.emit("voice_status", payload) {
                tracing::warn!(%error, "failed to emit voice_status");
            }
        }
        Err(error) => {
            tracing::warn!(%error, "failed to serialize voice_status");
        }
    }
}

fn broadcast_tts_config_state(
    io: &SocketIo,
    session_registry: &SessionRegistry,
    state: &TtsConfigState,
) {
    if let Err(error) = state.validate() {
        tracing::warn!(%error, "refusing to broadcast invalid tts_config_state");
        return;
    }
    if let Some(namespace) = io.of("/") {
        emit_authenticated(
            namespace,
            session_registry,
            "tts_config_state",
            serde_json::to_value(state).unwrap_or(Value::Null),
        );
    }
}

fn selected_target_entity(shared_state: &SharedState) -> String {
    shared_state
        .fleet_status
        .lock()
        .map(|status| status.selected_entity.clone())
        .unwrap_or_else(|_| "rover-kiwi".to_string())
}

fn is_target_active(shared_state: &SharedState, entity_id: &str) -> bool {
    shared_state
        .active_rovers_status
        .lock()
        .map(|status| status.active_rovers.contains(&entity_id.to_string()))
        .unwrap_or(false)
}

fn recording_consumer_id(recording_id: &str) -> String {
    format!("recording:{recording_id}")
}

fn enqueue_recording_demand(
    shared_state: &SharedState,
    entity_id: &str,
    consumer_id: &str,
    enabled: bool,
) -> Result<(), &'static str> {
    let resources = [
        MediaResource::Camera,
        MediaResource::Jpeg,
        MediaResource::Microphone,
    ];
    let transitions = shared_state
        .media_demand_registry
        .lock()
        .map_err(|_| "media demand unavailable")
        .map(|mut registry| {
            resources
                .into_iter()
                .filter_map(|resource| {
                    if enabled {
                        registry.acquire(entity_id, consumer_id, resource)
                    } else {
                        registry.release_consumer_resource(consumer_id, resource)
                    }
                })
                .collect::<Vec<_>>()
        })?;
    enqueue_media_transitions(&shared_state.targeted_media_control_queue, transitions);
    Ok(())
}

fn rename_recording_demand(shared_state: &SharedState, request_id: &str, recording_id: &str) {
    if let Ok(mut registry) = shared_state.media_demand_registry.lock() {
        registry.rename_consumer(
            &recording_consumer_id(request_id),
            &recording_consumer_id(recording_id),
        );
    }
}

fn release_recording_demand(shared_state: &SharedState, recording_id: &str) {
    let transitions = shared_state
        .media_demand_registry
        .lock()
        .ok()
        .map(|mut registry| registry.release_consumer(&recording_consumer_id(recording_id)))
        .unwrap_or_default();
    enqueue_media_transitions(&shared_state.targeted_media_control_queue, transitions);
}

fn rejected_recording_command(
    request_id: &str,
    reason_code: RecordingReasonCode,
    detail: &str,
) -> RecordingSessionCommandResult {
    RecordingSessionCommandResult {
        protocol_version: robo_rover_lib::RECORDING_PROTOCOL_VERSION,
        request_id: request_id.to_owned(),
        accepted: false,
        recording_id: None,
        reason_code: Some(reason_code),
        detail: Some(detail.chars().take(256).collect()),
    }
}

fn rejected_recording_clip_query(
    request_id: &str,
    reason_code: RecordingReasonCode,
    detail: &str,
) -> RecordingClipQueryResult {
    RecordingClipQueryResult {
        protocol_version: robo_rover_lib::RECORDING_PROTOCOL_VERSION,
        request_id: request_id.to_owned(),
        accepted: false,
        clips: Vec::new(),
        reason_code: Some(reason_code),
        detail: Some(detail.chars().take(256).collect()),
    }
}

fn rejected_recording_playback_ticket(
    request_id: &str,
    reason_code: RecordingReasonCode,
    detail: &str,
) -> Value {
    serde_json::json!({
        "protocol_version": robo_rover_lib::RECORDING_PROTOCOL_VERSION,
        "request_id": request_id,
        "accepted": false,
        "reason_code": reason_code,
        "detail": detail.chars().take(256).collect::<String>(),
    })
}

fn rejected_recording_delete(
    request_id: &str,
    reason_code: RecordingReasonCode,
    detail: &str,
) -> RecordingDeleteResult {
    RecordingDeleteResult {
        protocol_version: robo_rover_lib::RECORDING_PROTOCOL_VERSION,
        request_id: request_id.to_owned(),
        accepted: false,
        recording_id: None,
        reason_code: Some(reason_code),
        detail: Some(detail.chars().take(256).collect()),
    }
}

fn emit_recording_timeout(
    io: &SocketIo,
    state: &SharedState,
    request_id: &str,
    pending: &PendingRequest,
) {
    let detail = "recording request timed out";
    let (event, payload) = match pending.kind {
        RequestKind::Command => (
            "recording_session_command_result",
            serde_json::to_value(rejected_recording_command(
                request_id,
                RecordingReasonCode::Timeout,
                detail,
            ))
            .unwrap_or(Value::Null),
        ),
        RequestKind::ClipList => (
            "recording_clip_list_result",
            serde_json::to_value(rejected_recording_clip_query(
                request_id,
                RecordingReasonCode::Timeout,
                detail,
            ))
            .unwrap_or(Value::Null),
        ),
        RequestKind::PlaybackTicket => (
            "recording_playback_ticket_result",
            rejected_recording_playback_ticket(request_id, RecordingReasonCode::Timeout, detail),
        ),
        RequestKind::Delete => (
            "recording_delete_result",
            serde_json::to_value(rejected_recording_delete(
                request_id,
                RecordingReasonCode::Timeout,
                detail,
            ))
            .unwrap_or(Value::Null),
        ),
    };
    emit_recording_to_owner(io, state, &pending.socket_id, event, payload);
}

fn emit_recording_event(socket: &SocketRef, event: &'static str, payload: impl Serialize) {
    if let Err(error) = socket.emit(event, payload) {
        tracing::debug!(%error, event, "recording event delivery failed");
    }
}

fn emit_recording_to_owner(
    io: &SocketIo,
    state: &SharedState,
    owner: &str,
    event: &'static str,
    payload: Value,
) {
    if !state.session_registry.is_valid(owner) {
        return;
    }
    let Some(namespace) = io.of("/") else {
        return;
    };
    let Ok(socket_id) = owner.parse() else {
        return;
    };
    if let Some(socket) = namespace.get_socket(socket_id) {
        socket.emit(event, payload).ok();
    }
}

fn is_walkie_active(shared_state: &SharedState, entity_id: &str) -> bool {
    shared_state
        .voice_admission
        .lock()
        .map(|mut state| state.is_walkie_active(entity_id, Instant::now()))
        .unwrap_or(false)
}

fn build_tts_ack(
    command_id: &str,
    target_entity_id: &str,
    state: TtsAckState,
    reason_code: Option<VoiceReasonCode>,
    detail: Option<String>,
) -> TtsCommandAck {
    TtsCommandAck {
        command_id: command_id.to_string(),
        target_entity_id: target_entity_id.to_string(),
        state,
        timestamp: current_timestamp_ms(),
        reason_code,
        detail,
    }
}

fn emit_tts_ack(socket: &SocketRef, ack: &TtsCommandAck) {
    if let Err(error) = ack.validate() {
        tracing::warn!(%error, "refusing to emit invalid tts_command_ack");
        return;
    }
    match serde_json::to_value(ack) {
        Ok(payload) => {
            if let Err(error) = socket.emit("tts_command_ack", payload) {
                tracing::warn!(%error, "failed to emit tts_command_ack");
            }
        }
        Err(error) => {
            tracing::warn!(%error, "failed to serialize tts_command_ack");
        }
    }
}

fn convert_web_command_to_tts_command(web_cmd: &WebTtsCommand) -> robo_rover_lib::TtsCommand {
    use robo_rover_lib::TtsPriority;

    robo_rover_lib::TtsCommand {
        command_id: uuid::Uuid::new_v4().to_string(),
        text: web_cmd.text.clone(),
        timestamp: current_timestamp_ms(),
        priority: TtsPriority::Normal,
    }
}

fn create_metadata() -> CommandMetadata {
    CommandMetadata {
        source: InputSource::WebBridge,
        priority: CommandPriority::Normal,
        timestamp: current_timestamp_ms(),
        command_id: uuid::Uuid::new_v4().to_string(),
    }
}
