use crate::audio_input::{AudioInput, SourceIdentity};
use crate::segmenter::SegmenterFactory;
use crate::session_state::Session;
use eyre::{eyre, Result};
use robo_rover_lib::SttSourceKind;
use std::collections::HashMap;
use uuid::Uuid;

const MAX_SESSIONS: usize = 64;

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

pub struct SessionManager {
    sessions: HashMap<SessionKey, Session>,
    factory: SegmenterFactory,
}

impl SessionManager {
    pub fn new(factory: SegmenterFactory) -> Self {
        Self {
            sessions: HashMap::new(),
            factory,
        }
    }

    pub fn start_browser(&mut self, identity: SourceIdentity, sample_rate: u32) -> Result<()> {
        if identity.source_kind != SttSourceKind::Browser
            || identity.entity_id.is_some()
            || identity.target_entity_id.trim().is_empty()
            || !(8_000..=192_000).contains(&sample_rate)
        {
            return Err(eyre!("invalid browser stream start"));
        }
        let key = SessionKey::Browser(identity.stream_id);
        self.ensure_capacity(&key)?;
        self.sessions
            .insert(key, Session::new(identity, sample_rate, (self.factory)()?)?);
        Ok(())
    }

    pub fn accept_browser(&mut self, input: AudioInput) -> Result<FrameOutcome> {
        let key = SessionKey::Browser(input.identity.stream_id);
        let session = self
            .sessions
            .get_mut(&key)
            .ok_or_else(|| eyre!("browser audio received before stream start"))?;
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
        let mut session = self
            .sessions
            .remove(&SessionKey::Browser(stream_id))
            .ok_or_else(|| eyre!("unknown browser stream stop"))?;
        Ok(session.flush())
    }

    pub fn flush_all_browsers(&mut self) -> Vec<DecodeJob> {
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

    fn ensure_capacity(&self, key: &SessionKey) -> Result<()> {
        if !self.sessions.contains_key(key) && self.sessions.len() >= MAX_SESSIONS {
            Err(eyre!("active speech stream limit reached"))
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests;
