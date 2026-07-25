use super::lifecycle_types::*;

fn target() -> LifecycleTarget {
    LifecycleTarget {
        role: LifecycleRole::Rover,
        entity_id: "rover-kiwi".into(),
        node_id: "gst-camera".into(),
    }
}

#[test]
fn lifecycle_command_rejects_expired_or_oversized_ttl() {
    let mut command = LifecycleCommand {
        protocol_version: LIFECYCLE_PROTOCOL_VERSION,
        request_id: "f4f3e2d1-c0b9-48a7-9615-141312111000".into(),
        manager_epoch: 1,
        target: target(),
        desired_state: LifecycleDesiredState::Quiesced,
        expected_revision: 0,
        issued_at_ms: 100,
        expires_at_ms: 60_101,
        origin: Default::default(),
        transition_id: None,
    };
    assert!(command.validate().is_err());
    command.expires_at_ms = 101;
    assert!(command.validate().is_ok());
}

#[test]
fn lifecycle_status_round_trips_as_versioned_contract() {
    let status = LifecycleStatus {
        protocol_version: LIFECYCLE_PROTOCOL_VERSION,
        manager_epoch: 7,
        target: target(),
        revision: 3,
        desired_state: LifecycleDesiredState::Quiesced,
        effective_state: LifecycleEffectiveState::Quiescing,
        transition_id: None,
        components: vec![LifecycleComponentStatus {
            node_id: "gst-camera".into(),
            state: LifecycleComponentState::Quiescing,
            reason_code: None,
        }],
        updated_at_ms: 10,
    };
    let decoded: LifecycleStatus =
        serde_json::from_str(&serde_json::to_string(&status).unwrap()).unwrap();
    assert_eq!(decoded, status);
    assert!(decoded.validate().is_ok());
}

#[test]
fn lifecycle_cross_language_fixture_deserializes_and_validates() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../tests/fixtures/lifecycle-v1.json")).unwrap();
    let capabilities: Vec<LifecycleCapability> =
        serde_json::from_value(fixture["capabilities"].clone()).unwrap();
    let command: LifecycleCommand =
        serde_json::from_value(fixture["accepted_command"].clone()).unwrap();
    let accepted: LifecycleCommandResult =
        serde_json::from_value(fixture["accepted_result"].clone()).unwrap();
    let conflict: LifecycleCommandResult =
        serde_json::from_value(fixture["conflict_result"].clone()).unwrap();
    let status: LifecycleStatus = serde_json::from_value(fixture["status"].clone()).unwrap();

    assert_eq!(capabilities.len(), 1);
    assert!(command.validate().is_ok());
    assert!(accepted.validate().is_ok());
    assert!(conflict.validate().is_ok());
    assert!(status.validate().is_ok());
}
