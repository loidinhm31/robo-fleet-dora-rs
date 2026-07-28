use crate::{power_queues::PowerSocketState, power_socket};
use robo_rover_lib::{
    LifecycleRole, PowerAuthority, PowerAuthoritySnapshot, PowerCommandAction, PowerCommandResult,
    PowerPolicy, PowerProfile, PowerState, POWER_PROTOCOL_VERSION,
};

fn snapshot(authority: PowerAuthority) -> PowerAuthoritySnapshot {
    PowerAuthoritySnapshot {
        protocol_version: POWER_PROTOCOL_VERSION,
        snapshot_id: "00000000-0000-0000-0000-000000000001".into(),
        role: LifecycleRole::Rover,
        entity_id: "rover-a".into(),
        authority,
        state: PowerState::Active,
        effective_profile: PowerProfile::NormalRover,
        captured_at_ms: 1_000,
        expires_at_ms: 20_000,
    }
}

#[test]
fn policy_commands_require_a_fresh_rover_authority_snapshot() {
    let state = PowerSocketState::default();
    assert!(state
        .queue_policy(
            "socket".into(),
            "request".into(),
            "rover-a".into(),
            PowerPolicy::Sleep,
            2_000
        )
        .is_err());
    state.observe_snapshot(
        snapshot(PowerAuthority {
            epoch: 1,
            sequence: 1,
        }),
        2_000,
    );
    state
        .queue_policy(
            "socket".into(),
            "request".into(),
            "rover-a".into(),
            PowerPolicy::Sleep,
            2_000,
        )
        .unwrap();
    let command = state.next_command().unwrap();
    assert_eq!(
        command.authority,
        PowerAuthority {
            epoch: 2,
            sequence: 1
        }
    );
    assert!(matches!(
        command.action,
        PowerCommandAction::SetPolicy {
            policy: PowerPolicy::Sleep
        }
    ));
}

#[test]
fn wake_waits_for_a_newer_snapshot_before_acquiring_ui_demand() {
    let state = PowerSocketState::default();
    state.observe_snapshot(
        snapshot(PowerAuthority {
            epoch: 1,
            sequence: 1,
        }),
        2_000,
    );
    state
        .queue_wake("socket".into(), "request".into(), "rover-a".into(), 2_000)
        .unwrap();
    let policy = state.next_command().unwrap();
    let pending = state.take_pending(&policy.command_id).unwrap();
    state.accept_wake_policy(pending, policy.authority, 2_100);
    assert!(state.next_command().is_none());
    state.observe_snapshot(
        snapshot(PowerAuthority {
            epoch: 3,
            sequence: 1,
        }),
        2_200,
    );
    let demand = state.next_command().unwrap();
    assert!(matches!(
        demand.action,
        PowerCommandAction::RegisterDemand { .. }
    ));
}

#[test]
fn correlated_policy_results_stay_on_the_policy_socket_event() {
    let state = PowerSocketState::default();
    state.observe_snapshot(
        snapshot(PowerAuthority {
            epoch: 1,
            sequence: 1,
        }),
        2_000,
    );
    state
        .queue_policy(
            "socket".into(),
            "request".into(),
            "rover-a".into(),
            PowerPolicy::Sleep,
            2_000,
        )
        .unwrap();
    let command = state.next_command().unwrap();
    let (_, event, payload) = power_socket::handle_result(
        &state,
        &PowerCommandResult {
            protocol_version: POWER_PROTOCOL_VERSION,
            command_id: command.command_id,
            accepted: true,
            authority: PowerAuthority {
                epoch: 2,
                sequence: 1,
            },
            reason_code: None,
            detail: None,
        },
        2_100,
    )
    .unwrap();
    assert_eq!(event, "power_command_result");
    assert_eq!(
        payload.get("request_id").and_then(|value| value.as_str()),
        Some("request")
    );
}

#[test]
fn disconnect_releases_a_completed_wake_with_an_equal_fresh_snapshot() {
    let state = PowerSocketState::default();
    state.observe_snapshot(
        snapshot(PowerAuthority {
            epoch: 1,
            sequence: 1,
        }),
        2_000,
    );
    state
        .queue_wake("socket".into(), "request".into(), "rover-a".into(), 2_000)
        .unwrap();
    let policy = state.next_command().unwrap();
    state.accept_wake_policy(
        state.take_pending(&policy.command_id).unwrap(),
        policy.authority,
        2_050,
    );
    state.observe_snapshot(
        snapshot(PowerAuthority {
            epoch: 3,
            sequence: 1,
        }),
        2_100,
    );
    let demand = state.next_command().unwrap();
    let demand_id = match &demand.action {
        PowerCommandAction::RegisterDemand { demand } => demand.demand_id.clone(),
        _ => unreachable!(),
    };
    state.complete_wake_demand(
        &state.take_pending(&demand.command_id).unwrap(),
        PowerAuthority {
            epoch: 4,
            sequence: 1,
        },
        2_150,
    );
    state.observe_snapshot(
        snapshot(PowerAuthority {
            epoch: 4,
            sequence: 1,
        }),
        2_160,
    );
    state.release_socket("socket", 2_170);
    let release = state.next_command().unwrap();
    assert!(
        matches!(release.action, PowerCommandAction::ReleaseDemand { demand_id: ref released } if released == &demand_id)
    );
}

#[test]
fn repeated_wake_renews_the_existing_ui_demand() {
    let state = PowerSocketState::default();
    state.observe_snapshot(
        snapshot(PowerAuthority {
            epoch: 1,
            sequence: 1,
        }),
        2_000,
    );
    state
        .queue_wake("socket".into(), "first".into(), "rover-a".into(), 2_000)
        .unwrap();
    let first_policy = state.next_command().unwrap();
    state.accept_wake_policy(
        state.take_pending(&first_policy.command_id).unwrap(),
        first_policy.authority,
        2_050,
    );
    state.observe_snapshot(
        snapshot(PowerAuthority {
            epoch: 3,
            sequence: 1,
        }),
        2_100,
    );
    let first_demand = state.next_command().unwrap();
    let first_id = match &first_demand.action {
        PowerCommandAction::RegisterDemand { demand } => demand.demand_id.clone(),
        _ => unreachable!(),
    };
    state.complete_wake_demand(
        &state.take_pending(&first_demand.command_id).unwrap(),
        PowerAuthority {
            epoch: 4,
            sequence: 1,
        },
        2_150,
    );
    state.observe_snapshot(
        snapshot(PowerAuthority {
            epoch: 4,
            sequence: 1,
        }),
        2_160,
    );
    state
        .queue_wake("socket".into(), "second".into(), "rover-a".into(), 2_170)
        .unwrap();
    let second_policy = state.next_command().unwrap();
    state.accept_wake_policy(
        state.take_pending(&second_policy.command_id).unwrap(),
        second_policy.authority,
        2_180,
    );
    state.observe_snapshot(
        snapshot(PowerAuthority {
            epoch: 6,
            sequence: 1,
        }),
        2_190,
    );
    let renewed = state.next_command().unwrap();
    assert!(
        matches!(renewed.action, PowerCommandAction::RegisterDemand { demand } if demand.demand_id == first_id && demand.action == robo_rover_lib::PowerDemandAction::Renew && demand.renew_sequence == 2)
    );
}

#[test]
fn duplicate_browser_request_ids_are_rejected_without_new_commands() {
    let state = PowerSocketState::default();
    state.observe_snapshot(
        snapshot(PowerAuthority {
            epoch: 1,
            sequence: 1,
        }),
        2_000,
    );
    state
        .queue_policy(
            "socket".into(),
            "request".into(),
            "rover-a".into(),
            PowerPolicy::Sleep,
            2_000,
        )
        .unwrap();
    assert_eq!(
        state.queue_policy(
            "socket".into(),
            "request".into(),
            "rover-a".into(),
            PowerPolicy::Sleep,
            2_000
        ),
        Err("duplicate power request".into())
    );
    assert_eq!(
        state.queue_policy(
            "socket".into(),
            "request".into(),
            "rover-a".into(),
            PowerPolicy::Awake,
            2_000
        ),
        Err("duplicate request payload mismatch".into())
    );
    assert!(state.next_command().is_some());
    assert!(state.next_command().is_none());
}
