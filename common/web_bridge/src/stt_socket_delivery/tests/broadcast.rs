use super::*;
use crate::security::{jwt::Claims, SessionRegistry};

#[tokio::test]
async fn authenticated_emit_filters_expired_sessions_at_delivery_time() {
    let (service, io) = SocketIo::builder()
        .ping_interval(Duration::from_secs(60))
        .build_svc();
    let (socket_tx, mut socket_rx) = tokio::sync::mpsc::unbounded_channel();
    io.ns("/", move |socket: SocketRef| {
        let socket_tx = socket_tx.clone();
        async move {
            socket.join(AUTHENTICATED_ROOM).unwrap();
            socket_tx.send(socket.id).unwrap();
        }
    });

    let valid_sid = connect_polling(service.clone(), true).await;
    let valid_socket = socket_rx.recv().await.unwrap();
    let expired_sid = connect_polling(service.clone(), true).await;
    let expired_socket = socket_rx.recv().await.unwrap();
    let sessions = SessionRegistry::new();
    sessions.register(&valid_socket.to_string(), claims(u64::MAX));
    sessions.register(&expired_socket.to_string(), claims(0));

    let delivered = emit_authenticated(
        io.of("/").unwrap(),
        &sessions,
        "transcription",
        serde_json::json!({"text": "halt"}),
    );
    assert_eq!(delivered, 1);
    let valid_event = request(
        service.clone(),
        Method::GET,
        polling_uri(&valid_sid),
        Body::empty(),
    )
    .await;
    assert!(valid_event.contains("transcription"));
    let expired_event = tokio::time::timeout(
        Duration::from_millis(50),
        request(
            service,
            Method::GET,
            polling_uri(&expired_sid),
            Body::empty(),
        ),
    )
    .await;
    assert!(expired_event
        .map(|packet| !packet.contains("transcription"))
        .unwrap_or(true));
}

fn claims(exp: u64) -> Claims {
    Claims {
        sub: "operator".into(),
        role: "admin".into(),
        iat: 0,
        exp,
        jti: uuid::Uuid::new_v4().to_string(),
    }
}
