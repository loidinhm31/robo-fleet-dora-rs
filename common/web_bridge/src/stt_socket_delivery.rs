use crate::security::SessionRegistry;
use serde_json::Value;
use socketioxide::operators::BroadcastOperators;

pub const AUTHENTICATED_ROOM: &str = "authenticated";

pub fn emit_authenticated(
    namespace: BroadcastOperators,
    sessions: &SessionRegistry,
    event: &'static str,
    payload: Value,
) -> usize {
    let sockets = namespace.to(AUTHENTICATED_ROOM).sockets().unwrap();
    sockets
        .into_iter()
        .filter(|socket| sessions.is_valid(&socket.id.to_string()))
        .filter(|socket| socket.emit(event, payload.clone()).is_ok())
        .count()
}

#[cfg(test)]
mod tests;
