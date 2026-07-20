use crate::{clock::Clock, service::ScheduleService};
use robo_rover_lib::{
    AuthenticatedRecordingScheduleCommand, RecordingSchedule, RecordingScheduleAction,
    RecordingScheduleReasonCode,
};
use uuid::Uuid;
type ServiceError = (RecordingScheduleReasonCode, Option<RecordingSchedule>);

impl<C: Clock> ScheduleService<C> {
    pub(crate) async fn execute_valid(
        &self,
        request: AuthenticatedRecordingScheduleCommand,
    ) -> Result<Option<RecordingSchedule>, ServiceError> {
        let now_ms = self.clock.now_ms();
        let actor = request.audit_actor;
        match request.command.action {
            RecordingScheduleAction::Create { schedule } => {
                self.create(schedule, actor, now_ms).await
            }
            RecordingScheduleAction::Update {
                schedule_id,
                expected_revision,
                schedule,
            } => {
                self.update(schedule_id, expected_revision, schedule, actor, now_ms)
                    .await
            }
            RecordingScheduleAction::SetEnabled {
                schedule_id,
                expected_revision,
                enabled,
            } => {
                self.set_enabled(schedule_id, expected_revision, enabled, actor, now_ms)
                    .await
            }
            RecordingScheduleAction::Delete {
                schedule_id,
                expected_revision,
            } => self.delete(schedule_id, expected_revision, now_ms).await,
        }
    }
    async fn create(
        &self,
        definition: robo_rover_lib::RecordingScheduleDefinition,
        actor: String,
        now_ms: i64,
    ) -> Result<Option<RecordingSchedule>, ServiceError> {
        if definition.enabled
            && self
                .repository
                .enabled_count(&definition.entity_id)
                .await
                .map_err(unavailable)?
                >= self.config.max_schedules_per_entity as u64
        {
            return Err((RecordingScheduleReasonCode::InvalidSchedule, None));
        }
        let schedule = RecordingSchedule {
            schedule_id: Uuid::new_v4().to_string(),
            revision: 1,
            definition,
            created_at_ms: now_ms,
            created_by: actor.clone(),
            updated_at_ms: now_ms,
            updated_by: actor,
        };
        self.repository
            .insert_schedule(&schedule)
            .await
            .map_err(unavailable)?;
        Ok(Some(schedule))
    }
    async fn update(
        &self,
        schedule_id: String,
        expected: u64,
        definition: robo_rover_lib::RecordingScheduleDefinition,
        actor: String,
        now_ms: i64,
    ) -> Result<Option<RecordingSchedule>, ServiceError> {
        let current = self.current(&schedule_id, expected).await?;
        if definition.enabled
            && (!current.definition.enabled || definition.entity_id != current.definition.entity_id)
            && self
                .repository
                .enabled_count(&definition.entity_id)
                .await
                .map_err(unavailable)?
                >= self.config.max_schedules_per_entity as u64
        {
            return Err((RecordingScheduleReasonCode::InvalidSchedule, None));
        }
        let schedule = RecordingSchedule {
            schedule_id,
            revision: expected + 1,
            definition,
            created_at_ms: current.created_at_ms,
            created_by: current.created_by,
            updated_at_ms: now_ms,
            updated_by: actor,
        };
        self.replace(schedule, expected).await
    }
    async fn set_enabled(
        &self,
        schedule_id: String,
        expected: u64,
        enabled: bool,
        actor: String,
        now_ms: i64,
    ) -> Result<Option<RecordingSchedule>, ServiceError> {
        let mut schedule = self.current(&schedule_id, expected).await?;
        if enabled
            && !schedule.definition.enabled
            && self
                .repository
                .enabled_count(&schedule.definition.entity_id)
                .await
                .map_err(unavailable)?
                >= self.config.max_schedules_per_entity as u64
        {
            return Err((RecordingScheduleReasonCode::InvalidSchedule, None));
        }
        schedule.revision += 1;
        schedule.definition.enabled = enabled;
        schedule.updated_at_ms = now_ms;
        schedule.updated_by = actor;
        self.replace(schedule, expected).await
    }
    async fn delete(
        &self,
        schedule_id: String,
        expected: u64,
        now_ms: i64,
    ) -> Result<Option<RecordingSchedule>, ServiceError> {
        self.current(&schedule_id, expected).await?;
        self.repository
            .cancel_future(&schedule_id, expected, now_ms)
            .await
            .map_err(unavailable)?;
        if !self
            .repository
            .delete_schedule_cas(&schedule_id, expected)
            .await
            .map_err(unavailable)?
        {
            return Err(self.conflict(&schedule_id).await?);
        }
        Ok(None)
    }
    async fn current(
        &self,
        schedule_id: &str,
        expected: u64,
    ) -> Result<RecordingSchedule, ServiceError> {
        let schedule = self
            .repository
            .find_schedule(schedule_id)
            .await
            .map_err(unavailable)?
            .ok_or((RecordingScheduleReasonCode::NotFound, None))?;
        (schedule.revision == expected)
            .then_some(schedule.clone())
            .ok_or((RecordingScheduleReasonCode::Conflict, Some(schedule)))
    }

    async fn replace(
        &self,
        schedule: RecordingSchedule,
        expected: u64,
    ) -> Result<Option<RecordingSchedule>, ServiceError> {
        if self
            .repository
            .replace_schedule_cas(&schedule, expected)
            .await
            .map_err(unavailable)?
        {
            Ok(Some(schedule))
        } else {
            Err(self.conflict(&schedule.schedule_id).await?)
        }
    }

    async fn conflict(&self, schedule_id: &str) -> Result<ServiceError, ServiceError> {
        Ok((
            RecordingScheduleReasonCode::Conflict,
            self.repository
                .find_schedule(schedule_id)
                .await
                .map_err(unavailable)?,
        ))
    }
}

fn unavailable(_: String) -> ServiceError {
    (RecordingScheduleReasonCode::Unavailable, None)
}
