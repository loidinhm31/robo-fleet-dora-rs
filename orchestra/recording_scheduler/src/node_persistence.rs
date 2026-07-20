use eyre::Result;
use robo_rover_lib::{
    RecordingAttemptState, RecordingClipAttempt, RecordingOccurrenceState,
    RecordingReconciliationSnapshot, RecordingSessionState,
};

use crate::{
    clock::{Clock, SystemClock},
    mongo_repository::MongoRepository,
    runtime::SchedulerRuntime,
};

pub(crate) fn adopt<C: Clock>(
    scheduler: &mut SchedulerRuntime<C>,
    snapshot: RecordingReconciliationSnapshot,
) {
    let now_ms = scheduler.clock.now_ms();
    for session in snapshot.sessions.into_iter().filter(|session| {
        matches!(
            session.state,
            RecordingSessionState::Starting
                | RecordingSessionState::Recording
                | RecordingSessionState::Stopping
        )
    }) {
        let Some(group_id) = scheduler
            .groups
            .values()
            .find(|group| {
                group.entity_id == session.entity_id
                    && group.start_request_id == session.start_request_id
            })
            .map(|group| group.group_id.clone())
        else {
            continue;
        };
        let group = scheduler.groups.get_mut(&group_id).expect("group exists");
        group.recording_id = Some(session.recording_id.clone());
        group.finish_intent();
        for occurrence in scheduler.occurrences.values_mut().filter(|occurrence| {
            occurrence.group_id.as_deref() == Some(&group_id)
                && !occurrence.state.is_terminal()
                && occurrence.planned_start_ms <= now_ms
                && now_ms < occurrence.planned_end_ms
        }) {
            occurrence.state = RecordingOccurrenceState::Active;
            if !occurrence
                .attempts
                .iter()
                .any(|attempt| attempt.recording_id == session.recording_id)
            {
                occurrence.attempts.push(RecordingClipAttempt {
                    recording_id: session.recording_id.clone(),
                    state: RecordingAttemptState::Recovered,
                    started_at_ms: now_ms,
                    ended_at_ms: None,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::FakeClock;
    use robo_rover_lib::{
        RecordingLocalStart, RecordingReconciliationSession, RecordingSchedule,
        RecordingScheduleDefinition, RecordingScheduleRecurrence, RecordingSessionState,
    };

    const HOUR_MS: i64 = 60 * 60 * 1_000;

    #[test]
    fn reconciliation_adopts_only_live_overlap_members() {
        let now_ms = 1_767_225_600_000; // 2026-01-01T00:00:00Z
        let clock = FakeClock::new(now_ms);
        let mut scheduler = SchedulerRuntime::new(clock.clone());
        scheduler
            .materialize(
                &schedule("00000000-0000-0000-0000-000000000401", "01:00"),
                now_ms + 4 * HOUR_MS,
            )
            .unwrap();
        scheduler
            .materialize(
                &schedule("00000000-0000-0000-0000-000000000402", "01:30"),
                now_ms + 4 * HOUR_MS,
            )
            .unwrap();
        scheduler.complete_reconciliation();

        clock.advance_ms(HOUR_MS);
        let first_id = scheduler.due().pop().unwrap();
        let first_start = scheduler.begin_start(&first_id).unwrap();
        let group = scheduler.groups.values().next().unwrap().clone();

        adopt(
            &mut scheduler,
            RecordingReconciliationSnapshot {
                request_id: "snapshot".into(),
                sessions: vec![RecordingReconciliationSession {
                    entity_id: "kiwi-1".into(),
                    start_request_id: group.start_request_id,
                    recording_id: "recording-1".into(),
                    state: RecordingSessionState::Recording,
                }],
            },
        );
        let second_id = scheduler
            .occurrences
            .keys()
            .find(|id| *id != &first_id)
            .cloned()
            .unwrap();
        assert_eq!(
            scheduler.occurrences[&first_id].state,
            RecordingOccurrenceState::Active
        );
        assert_eq!(
            scheduler.occurrences[&second_id].state,
            RecordingOccurrenceState::Planned
        );

        clock.advance_ms(30 * 60 * 1_000);
        assert_eq!(scheduler.due(), vec![second_id.clone()]);
        assert!(scheduler
            .begin_start(&second_id)
            .unwrap()
            .intent_id
            .is_none());

        clock.advance_ms(30 * 60 * 1_000);
        assert_eq!(scheduler.due_stops(), vec![first_id.clone()]);
        assert!(scheduler.begin_stop(&first_id).unwrap().intent_id.is_none());

        clock.advance_ms(30 * 60 * 1_000);
        assert_eq!(scheduler.due_stops(), vec![second_id.clone()]);
        assert!(scheduler
            .begin_stop(&second_id)
            .unwrap()
            .intent_id
            .is_some());
        assert!(first_start.intent_id.is_some());
    }

    #[test]
    fn reconciliation_ignores_terminal_recorder_sessions() {
        let now_ms = 1_767_225_600_000; // 2026-01-01T00:00:00Z
        let clock = FakeClock::new(now_ms);
        let mut scheduler = SchedulerRuntime::new(clock.clone());
        scheduler
            .materialize(
                &schedule("00000000-0000-0000-0000-000000000403", "01:00"),
                now_ms + 3 * HOUR_MS,
            )
            .unwrap();
        scheduler.complete_reconciliation();
        clock.advance_ms(HOUR_MS);
        let occurrence_id = scheduler.due().pop().unwrap();
        let start = scheduler.begin_start(&occurrence_id).unwrap();
        let group = scheduler.groups.values().next().unwrap().clone();

        adopt(
            &mut scheduler,
            RecordingReconciliationSnapshot {
                request_id: "snapshot".into(),
                sessions: vec![RecordingReconciliationSession {
                    entity_id: "kiwi-1".into(),
                    start_request_id: group.start_request_id,
                    recording_id: "recording-terminal".into(),
                    state: RecordingSessionState::Completed,
                }],
            },
        );

        assert_eq!(
            scheduler.occurrences[&occurrence_id].state,
            RecordingOccurrenceState::StartPending
        );
        assert!(scheduler.groups[&group.group_id].recording_id.is_none());
        assert_eq!(
            scheduler.groups[&group.group_id].pending_intent_id,
            start.intent_id
        );
    }

    fn schedule(id: &str, time: &str) -> RecordingSchedule {
        RecordingSchedule {
            schedule_id: id.into(),
            revision: 1,
            definition: RecordingScheduleDefinition {
                entity_id: "kiwi-1".into(),
                title: "test".into(),
                enabled: true,
                recurrence: RecordingScheduleRecurrence::OneTime {
                    local_start: RecordingLocalStart {
                        date: "2026-01-01".into(),
                        time: time.into(),
                        timezone: "UTC".into(),
                    },
                },
                duration_ms: HOUR_MS,
                relative_directory_template: "scheduled".into(),
            },
            created_at_ms: 0,
            created_by: "test".into(),
            updated_at_ms: 0,
            updated_by: "test".into(),
        }
    }
}

pub(crate) fn persist(
    tokio: &tokio::runtime::Runtime,
    repository: &MongoRepository,
    scheduler: &mut SchedulerRuntime<SystemClock>,
) -> Result<()> {
    for occurrence in scheduler.occurrences.values().cloned().collect::<Vec<_>>() {
        let expected = scheduler.occurrence_revisions[&occurrence.occurrence_id];
        let revision = tokio
            .block_on(repository.save_occurrence(&occurrence, expected))
            .map_err(eyre::Report::msg)?
            .ok_or_else(|| eyre::eyre!("recording occurrence changed concurrently"))?;
        scheduler
            .occurrence_revisions
            .insert(occurrence.occurrence_id, revision);
    }
    for group in scheduler.groups.values().cloned().collect::<Vec<_>>() {
        let expected = scheduler.group_revisions[&group.group_id];
        let revision = tokio
            .block_on(repository.save_group(&group, expected))
            .map_err(eyre::Report::msg)?
            .ok_or_else(|| eyre::eyre!("recording group changed concurrently"))?;
        scheduler.group_revisions.insert(group.group_id, revision);
    }
    for (group_id, revision) in std::mem::take(&mut scheduler.removed_group_revisions) {
        let deleted = tokio
            .block_on(repository.delete_group_cas(&group_id, revision))
            .map_err(eyre::Report::msg)?;
        if !deleted && revision != 0 {
            return Err(eyre::eyre!("recording group changed concurrently"));
        }
    }
    Ok(())
}
