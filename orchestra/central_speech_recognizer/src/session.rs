use crate::audio_input::{AudioInput, SourceIdentity};
use crate::segmenter::SegmenterFactory;
use crate::session_state::Session;
use eyre::{eyre, Result};
use robo_rover_lib::SttSourceKind;
use std::collections::HashMap;
use uuid::Uuid;

const MAX_SESSIONS: usize = 64;
const MAX_PRESTART_FRAMES: usize = 8;

#[derive(Debug, Clone, PartialEq)]
pub struct DecodeJob {
    pub identity: SourceIdentity,
    pub samples: Vec<f32>,
}

#[derive(Debug, Default, PartialEq)]
pub struct FrameOutcome {
    pub jobs: Vec<DecodeJob>,
    pub sequence_reset: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum SessionKey {
    Browser(Uuid),
    Rover(String, Uuid),
}

struct PendingBrowser {
    identity: SourceIdentity,
    sample_rate: u32,
    frames: Vec<AudioInput>,
}

pub struct SessionManager {
    sessions: HashMap<SessionKey, Session>,
    pending_browsers: HashMap<Uuid, PendingBrowser>,
    factory: SegmenterFactory,
}

impl SessionManager {
    pub fn new(factory: SegmenterFactory) -> Self {
        Self {
            sessions: HashMap::new(),
            pending_browsers: HashMap::new(),
            factory,
        }
    }

    pub fn start_browser(
        &mut self,
        identity: SourceIdentity,
        sample_rate: u32,
    ) -> Result<FrameOutcome> {
        validate_browser_identity(&identity, sample_rate)?;
        let key = SessionKey::Browser(identity.stream_id);
        if self
            .pending_browsers
            .get(&identity.stream_id)
            .is_some_and(|pending| {
                pending.identity != identity || pending.sample_rate != sample_rate
            })
        {
            return Err(eyre!("browser pre-start metadata does not match start"));
        }
        self.ensure_capacity(&key)?;
        let mut session = Session::new(identity, sample_rate, (self.factory)()?)?;
        let mut outcome = FrameOutcome::default();
        if let Some(pending) = self.pending_browsers.remove(&session.identity.stream_id) {
            for frame in pending.frames {
                let frame_outcome = session.accept(frame.frame_id, &frame.samples);
                outcome.jobs.extend(frame_outcome.jobs);
                outcome.sequence_reset |= frame_outcome.sequence_reset;
            }
        }
        self.sessions.insert(key, session);
        Ok(outcome)
    }

    pub fn accept_browser(&mut self, input: AudioInput) -> Result<FrameOutcome> {
        let key = SessionKey::Browser(input.identity.stream_id);
        let Some(session) = self.sessions.get_mut(&key) else {
            self.buffer_prestart(input)?;
            return Ok(FrameOutcome::default());
        };
        if session.identity != input.identity || session.sample_rate != input.sample_rate {
            return Err(eyre!("browser stream metadata changed after start"));
        }
        Ok(session.accept(input.frame_id, &input.samples))
    }

    pub fn accept_rover(&mut self, input: AudioInput) -> Result<FrameOutcome> {
        let entity = input
            .identity
            .entity_id
            .clone()
            .ok_or_else(|| eyre!("rover audio has no entity identity"))?;
        let key = SessionKey::Rover(entity.clone(), input.identity.stream_id);
        let stream_replaced = self.sessions.keys().any(
            |existing| matches!(existing, SessionKey::Rover(id, _) if id == &entity && existing != &key),
        );
        self.sessions.retain(|existing, _| {
            !matches!(existing, SessionKey::Rover(id, _) if id == &entity) || existing == &key
        });
        self.ensure_capacity(&key)?;
        if !self.sessions.contains_key(&key) {
            let session =
                Session::new(input.identity.clone(), input.sample_rate, (self.factory)()?)?;
            self.sessions.insert(key.clone(), session);
        }
        let mut outcome = self
            .sessions
            .get_mut(&key)
            .unwrap()
            .accept(input.frame_id, &input.samples);
        outcome.sequence_reset |= stream_replaced;
        Ok(outcome)
    }

    pub fn stop_browser(&mut self, stream_id: Uuid) -> Result<Vec<DecodeJob>> {
        if let Some(mut session) = self.sessions.remove(&SessionKey::Browser(stream_id)) {
            return Ok(session.flush());
        }
        if self.pending_browsers.remove(&stream_id).is_some() {
            return Ok(Vec::new());
        }
        Err(eyre!("unknown browser stream stop"))
    }

    pub fn flush_all_browsers(&mut self) -> Vec<DecodeJob> {
        self.pending_browsers.clear();
        let keys: Vec<_> = self
            .sessions
            .keys()
            .filter(|key| matches!(key, SessionKey::Browser(_)))
            .cloned()
            .collect();
        keys.into_iter()
            .filter_map(|key| self.sessions.remove(&key))
            .flat_map(|mut session| session.flush())
            .collect()
    }

    /// Drops every active and pre-start stream without flushing a final segment.
    /// Lifecycle quiesce uses this instead of the normal disconnect path: audio
    /// captured before a pause must never become a post-pause transcription.
    pub fn discard_all(&mut self) -> usize {
        self.cancel_all_for_lifecycle().len()
    }

    /// Removes every stream without flushing and returns its identity for the
    /// lifecycle cancellation audit/status. No caller can create a final
    /// decode job after this method returns.
    pub fn cancel_all_for_lifecycle(&mut self) -> Vec<SourceIdentity> {
        let mut cancelled: Vec<_> = self
            .sessions
            .values()
            .map(|session| session.identity.clone())
            .collect();
        cancelled.extend(
            self.pending_browsers
                .values()
                .map(|pending| pending.identity.clone()),
        );
        self.sessions.clear();
        self.pending_browsers.clear();
        cancelled
    }

    fn buffer_prestart(&mut self, input: AudioInput) -> Result<()> {
        validate_browser_identity(&input.identity, input.sample_rate)?;
        let key = SessionKey::Browser(input.identity.stream_id);
        self.ensure_capacity(&key)?;
        let pending = self
            .pending_browsers
            .entry(input.identity.stream_id)
            .or_insert_with(|| PendingBrowser {
                identity: input.identity.clone(),
                sample_rate: input.sample_rate,
                frames: Vec::new(),
            });
        if pending.identity != input.identity || pending.sample_rate != input.sample_rate {
            return Err(eyre!("browser pre-start metadata changed"));
        }
        if pending.frames.len() >= MAX_PRESTART_FRAMES {
            return Err(eyre!("browser pre-start buffer is full"));
        }
        if pending
            .frames
            .last()
            .is_some_and(|last| last.frame_id.checked_add(1) != Some(input.frame_id))
        {
            return Err(eyre!("browser pre-start frame sequence is invalid"));
        }
        pending.frames.push(input);
        Ok(())
    }

    fn ensure_capacity(&self, key: &SessionKey) -> Result<()> {
        let pending_exists =
            matches!(key, SessionKey::Browser(id) if self.pending_browsers.contains_key(id));
        if !self.sessions.contains_key(key)
            && !pending_exists
            && self.sessions.len() + self.pending_browsers.len() >= MAX_SESSIONS
        {
            Err(eyre!("active speech stream limit reached"))
        } else {
            Ok(())
        }
    }
}

fn validate_browser_identity(identity: &SourceIdentity, sample_rate: u32) -> Result<()> {
    if identity.source_kind != SttSourceKind::Browser
        || identity.entity_id.is_some()
        || identity.target_entity_id.trim().is_empty()
        || !(8_000..=192_000).contains(&sample_rate)
    {
        Err(eyre!("invalid browser stream metadata"))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests;
