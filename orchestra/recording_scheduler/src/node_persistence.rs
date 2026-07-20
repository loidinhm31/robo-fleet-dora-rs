use eyre::Result;
use robo_rover_lib::{
    RecordingAttemptState, RecordingClipAttempt, RecordingOccurrenceState,
    RecordingReconciliationSnapshot,
};

use crate::{
    clock::{Clock, SystemClock},
    mongo_repository::MongoRepository,
    runtime::SchedulerRuntime,
};

pub(crate) fn adopt(
    scheduler: &mut SchedulerRuntime<SystemClock>,
    snapshot: RecordingReconciliationSnapshot,
) {
    for session in snapshot.sessions {
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
            occurrence.group_id.as_deref() == Some(&group_id) && !occurrence.state.is_terminal()
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
                    started_at_ms: SystemClock.now_ms(),
                    ended_at_ms: None,
                });
            }
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
    Ok(())
}
