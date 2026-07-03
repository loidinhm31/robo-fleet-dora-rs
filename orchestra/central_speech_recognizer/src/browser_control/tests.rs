use super::*;

#[test]
fn browser_control_requires_server_assigned_target() {
    let missing_target = br#"{
        "command":"start",
        "stream_id":"550e8400-e29b-41d4-a716-446655440000",
        "sample_rate":48000,
        "channels":1
    }"#;
    assert!(serde_json::from_slice::<BrowserControl>(missing_target).is_err());

    let stop = br#"{
        "command":"stop",
        "stream_id":"550e8400-e29b-41d4-a716-446655440000"
    }"#;
    assert!(matches!(
        serde_json::from_slice::<BrowserControl>(stop).unwrap(),
        BrowserControl::Stop { .. }
    ));
}
