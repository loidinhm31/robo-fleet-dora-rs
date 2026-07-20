use serde::{de::DeserializeOwned, Serialize};

pub fn decode_json<T: DeserializeOwned>(payload: &[u8]) -> Result<T, String> {
    serde_json::from_slice(payload).map_err(|_| "invalid scheduler payload".into())
}

pub fn encode_json<T: Serialize>(value: &T) -> Result<Vec<u8>, String> {
    serde_json::to_vec(value).map_err(|_| "scheduler serialization failed".into())
}
