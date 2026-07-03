use super::*;
use axum::{
    body::{to_bytes, Body, Bytes, HttpBody},
    http::{Method, Request, Response},
};
use serde::Deserialize;
use socketioxide::{
    extract::{SocketRef, TryData},
    SocketIo,
};
use std::{fmt::Debug, time::Duration};
use tower::{Service, ServiceExt};

mod broadcast;

#[derive(Debug, Deserialize)]
struct TestAuth {
    authenticated: bool,
}

#[tokio::test]
async fn authenticated_room_excludes_pending_or_rejected_socket() {
    let (service, io) = SocketIo::builder()
        .ping_interval(Duration::from_secs(60))
        .build_svc();
    io.ns(
        "/",
        |socket: SocketRef, TryData::<TestAuth>(auth)| async move {
            if auth.is_ok_and(|auth| auth.authenticated) {
                socket.join(AUTHENTICATED_ROOM).unwrap();
            }
        },
    );

    connect_polling(service.clone(), true).await;
    connect_polling(service, false).await;

    assert_eq!(io.sockets().unwrap().len(), 2);
    let authenticated = io.to(AUTHENTICATED_ROOM).sockets().unwrap();
    assert_eq!(authenticated.len(), 1);
    assert!(authenticated[0]
        .rooms()
        .unwrap()
        .iter()
        .any(|room| room.as_ref() == AUTHENTICATED_ROOM));
}

#[tokio::test]
async fn targeted_emit_reaches_only_the_selected_socket() {
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

    let first_sid = connect_polling(service.clone(), true).await;
    let first_socket = socket_rx.recv().await.unwrap();
    let second_sid = connect_polling(service.clone(), true).await;
    socket_rx.recv().await.unwrap();

    io.get_socket(first_socket)
        .unwrap()
        .emit(
            "voice_command_transcription",
            serde_json::json!({"text": "stop"}),
        )
        .unwrap();
    let first_event = request(
        service.clone(),
        Method::GET,
        polling_uri(&first_sid),
        Body::empty(),
    )
    .await;
    assert!(first_event.contains("voice_command_transcription"));

    let second_event = tokio::time::timeout(
        Duration::from_millis(50),
        request(
            service,
            Method::GET,
            polling_uri(&second_sid),
            Body::empty(),
        ),
    )
    .await;
    assert!(second_event
        .map(|packet| !packet.contains("voice_command_transcription"))
        .unwrap_or(true));
}

async fn connect_polling<S, ResBody>(service: S, authenticated: bool) -> String
where
    S: Service<Request<Body>, Response = Response<ResBody>> + Clone,
    S::Error: Debug,
    S::Future: Send,
    ResBody: HttpBody<Data = Bytes> + Send + 'static,
    ResBody::Error: Debug + Into<axum::BoxError>,
{
    let open = request(
        service.clone(),
        Method::GET,
        "/socket.io/?EIO=4&transport=polling".into(),
        Body::empty(),
    )
    .await;
    let packet: serde_json::Value = serde_json::from_str(&open[1..]).unwrap();
    let sid = packet["sid"].as_str().unwrap();
    let uri = format!("/socket.io/?EIO=4&transport=polling&sid={sid}");
    let connect = format!(r#"40{{"authenticated":{authenticated}}}"#);
    request(
        service.clone(),
        Method::POST,
        uri.clone(),
        Body::from(connect),
    )
    .await;
    let ack = request(service, Method::GET, uri, Body::empty()).await;
    assert!(ack.starts_with("40"));
    sid.to_owned()
}

fn polling_uri(sid: &str) -> String {
    format!("/socket.io/?EIO=4&transport=polling&sid={sid}")
}

async fn request<S, ResBody>(service: S, method: Method, uri: String, body: Body) -> String
where
    S: Service<Request<Body>, Response = Response<ResBody>>,
    S::Error: Debug,
    S::Future: Send,
    ResBody: HttpBody<Data = Bytes> + Send + 'static,
    ResBody::Error: Debug + Into<axum::BoxError>,
{
    let request = Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "text/plain;charset=UTF-8")
        .body(body)
        .unwrap();
    let response = service.oneshot(request).await.unwrap();
    let bytes = to_bytes(Body::new(response.into_body()), 1_048_576)
        .await
        .unwrap();
    String::from_utf8(bytes.to_vec()).unwrap()
}
