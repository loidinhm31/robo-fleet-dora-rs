use super::*;

fn demand() -> PowerDemand {
    PowerDemand {
        protocol_version: POWER_PROTOCOL_VERSION,
        demand_id: "11111111-1111-4111-8111-111111111111".into(),
        action: PowerDemandAction::Acquire,
        source: PowerDemandSource::Kws,
        priority: PowerDemandPriority::High,
        role: LifecycleRole::Rover,
        entity_id: "rover-kiwi".into(),
        required_profile: PowerProfile::NormalRover,
        authority: PowerAuthority {
            epoch: 7,
            sequence: 2,
        },
        issued_at_ms: 100,
        not_before_ms: 100,
        expires_at_ms: 10_000,
        renew_sequence: 1,
    }
}

#[test]
fn power_fixture_round_trips_and_validates() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../../test-data/contracts/power-v1.json")).unwrap();
    let command: PowerCommand = serde_json::from_value(fixture["command"].clone()).unwrap();
    let result: PowerCommandResult = serde_json::from_value(fixture["result"].clone()).unwrap();
    let snapshot: PowerAuthoritySnapshot =
        serde_json::from_value(fixture["snapshot"].clone()).unwrap();
    let status: PowerStatus = serde_json::from_value(fixture["status"].clone()).unwrap();
    let transition: PowerTransition =
        serde_json::from_value(fixture["transition"].clone()).unwrap();
    let event: PowerEvent = serde_json::from_value(fixture["event"].clone()).unwrap();
    for valid in [
        &command.validate(),
        &result.validate(),
        &snapshot.validate(),
        &status.validate(),
        &transition.validate(),
        &event.validate(),
    ] {
        assert!(valid.is_ok());
    }
    assert_eq!(serde_json::to_value(command).unwrap(), fixture["command"]);
}

#[test]
fn demand_rejects_cross_entity_bad_renewal_and_profile_source_pairs() {
    let mut value = demand();
    assert!(value
        .validates_for(LifecycleRole::Rover, "rover-kiwi")
        .is_ok());
    assert!(value
        .validates_for(LifecycleRole::Rover, "rover-other")
        .is_err());
    value.renew_sequence = 0;
    assert!(value.validate().is_err());
    value.renew_sequence = 1;
    value.required_profile = PowerProfile::IdleListening;
    assert!(value.validate().is_err());
}

#[test]
fn immutable_demand_identity_excludes_only_renewal_fields() {
    let first = demand();
    let mut renewal = first.clone();
    renewal.action = PowerDemandAction::Renew;
    renewal.renew_sequence = 2;
    renewal.issued_at_ms = 2_000;
    renewal.expires_at_ms = 12_000;
    assert!(first.same_immutable_payload(&renewal));
    renewal.required_profile = PowerProfile::ScheduledCapture;
    assert!(!first.same_immutable_payload(&renewal));
}

#[test]
fn unknown_enum_and_invalid_authority_fail_closed() {
    let mut encoded = serde_json::to_value(demand()).unwrap();
    encoded["source"] = serde_json::json!("browser_supplied");
    assert!(serde_json::from_value::<PowerDemand>(encoded).is_err());
    let mut command = serde_json::to_value(fixture_command()).unwrap();
    command["browser_actor"] = serde_json::json!("spoofed");
    assert!(serde_json::from_value::<PowerCommand>(command).is_err());
    let mut command = fixture_command();
    command.authority.sequence = 0;
    assert!(command.validate().is_err());
}

#[test]
fn snapshot_gate_requires_a_fresh_matching_snapshot_and_newer_authority() {
    let mut gate = PowerSnapshotGate::new(LifecycleRole::Rover, "rover-kiwi".into()).unwrap();
    let next = PowerAuthority {
        epoch: 8,
        sequence: 1,
    };
    assert_eq!(gate.state(100), PowerState::AuthorityUnknown);
    assert!(!gate.consume_profile_authority(next, 100));

    let snapshot: PowerAuthoritySnapshot = serde_json::from_str(
        r#"{"protocol_version":1,"snapshot_id":"33333333-3333-4333-8333-333333333333","role":"rover","entity_id":"rover-kiwi","authority":{"epoch":7,"sequence":2},"state":"active","effective_profile":"normal_rover","captured_at_ms":100,"expires_at_ms":200}"#,
    ).unwrap();
    gate.observe(snapshot, 101).unwrap();
    assert!(!gate.consume_profile_authority(
        PowerAuthority {
            epoch: 7,
            sequence: 2
        },
        101
    ));
    assert!(gate.consume_profile_authority(next, 101));
    let replayed_snapshot: PowerAuthoritySnapshot = serde_json::from_str(
        r#"{"protocol_version":1,"snapshot_id":"33333333-3333-4333-8333-333333333333","role":"rover","entity_id":"rover-kiwi","authority":{"epoch":7,"sequence":2},"state":"active","effective_profile":"normal_rover","captured_at_ms":100,"expires_at_ms":200}"#,
    )
    .unwrap();
    assert!(gate.observe(replayed_snapshot, 102).is_err());
    assert!(!gate.consume_profile_authority(
        PowerAuthority {
            epoch: 8,
            sequence: 2
        },
        101
    ));
    assert_eq!(gate.state(200), PowerState::AuthorityUnknown);
    assert!(!gate.consume_profile_authority(next, 200));
}

fn fixture_command() -> PowerCommand {
    PowerCommand {
        protocol_version: POWER_PROTOCOL_VERSION,
        command_id: "22222222-2222-4222-8222-222222222222".into(),
        role: LifecycleRole::Rover,
        entity_id: "rover-kiwi".into(),
        authority: PowerAuthority {
            epoch: 7,
            sequence: 2,
        },
        action: PowerCommandAction::RegisterDemand { demand: demand() },
        issued_at_ms: 100,
        not_before_ms: 100,
        expires_at_ms: 10_000,
        detail: None,
    }
}
