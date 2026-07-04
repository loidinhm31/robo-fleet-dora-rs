#[derive(Debug, PartialEq, Eq)]
pub enum ArbiterEvent {
    WalkieStarted { interrupted: Option<String> },
    WalkieEnded,
    TtsRejectedWhileWalkie,
    TtsPlaybackCompleted { command_id: String },
    TtsPlaybackFailed { command_id: String },
}
