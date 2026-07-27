use super::{
    ProtectedWorkRelayBody, ProtectedWorkRelayEnvelope, RecordingOccurrence,
    MAX_PROTECTED_WORK_CLOCK_SKEW_MS, PROTECTED_WORK_RELAY_TTL_MS,
};

fn active_occurrence() -> RecordingOccurrence {
    serde_json::from_value(serde_json::json!({
        "occurrence_id": "f4f3e2d1-c0b9-48a7-9615-141312111000",
        "schedule_id": "f4f3e2d1-c0b9-48a7-9615-141312111001",
        "schedule_revision": 1,
        "entity_id": "rover",
        "planned_start_ms": 1,
        "planned_end_ms": 2,
        "dst_resolution": "exact",
        "state": "active",
        "retry_count": 0,
        "next_retry_at_ms": null,
        "group_id": null,
        "start_request_id": "f4f3e2d1-c0b9-48a7-9615-141312111002",
        "attempts": [],
        "last_error": null,
        "suppressed_by_manual": false,
        "created_at_ms": 1,
        "updated_at_ms": 1,
        "terminal_at_ms": null,
        "expires_at_ms": null
    }))
    .unwrap()
}

#[test]
fn protected_work_envelope_rejects_tampering_and_expiry() {
    let key = b"12345678901234567890123456789012";
    let envelope = ProtectedWorkRelayEnvelope::new(
        "rover".into(),
        10,
        PROTECTED_WORK_RELAY_TTL_MS,
        ProtectedWorkRelayBody::Occurrence {
            occurrence: active_occurrence(),
        },
    )
    .sign(key)
    .unwrap();
    assert!(envelope.verify(key, 10).is_ok());
    assert!(envelope
        .verify(key, 10 + PROTECTED_WORK_RELAY_TTL_MS + 1)
        .is_err());

    let mut tampered = envelope;
    tampered.target_entity_id = "other-rover".into();
    assert!(tampered.verify(key, 10).is_err());
}

#[test]
fn protected_work_envelope_rejects_excessive_future_clock_skew() {
    let key = b"12345678901234567890123456789012";
    let tolerated = ProtectedWorkRelayEnvelope::new(
        "rover".into(),
        MAX_PROTECTED_WORK_CLOCK_SKEW_MS,
        PROTECTED_WORK_RELAY_TTL_MS,
        ProtectedWorkRelayBody::Occurrence {
            occurrence: active_occurrence(),
        },
    )
    .sign(key)
    .unwrap();
    assert!(tolerated.verify(key, 0).is_ok());

    let envelope = ProtectedWorkRelayEnvelope::new(
        "rover".into(),
        MAX_PROTECTED_WORK_CLOCK_SKEW_MS + 1,
        PROTECTED_WORK_RELAY_TTL_MS,
        ProtectedWorkRelayBody::Occurrence {
            occurrence: active_occurrence(),
        },
    )
    .sign(key)
    .unwrap();

    assert!(envelope.verify(key, 0).is_err());
}
