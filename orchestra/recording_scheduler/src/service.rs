use robo_rover_lib::{
    AuthenticatedRecordingScheduleCommand, RecordingSchedule, RecordingScheduleCommandResult,
    RecordingScheduleReasonCode, RECORDING_SCHEDULE_PROTOCOL_VERSION,
};

use crate::{clock::Clock, config::SchedulerConfig, mongo_repository::MongoRepository};

pub struct ScheduleService<C> {
    pub(crate) clock: C,
    pub(crate) config: SchedulerConfig,
    pub(crate) repository: MongoRepository,
}

impl<C: Clock> ScheduleService<C> {
    pub fn new(clock: C, config: SchedulerConfig, repository: MongoRepository) -> Self {
        Self {
            clock,
            config,
            repository,
        }
    }

    pub async fn execute(
        &self,
        request: AuthenticatedRecordingScheduleCommand,
    ) -> RecordingScheduleCommandResult {
        let request_id = request.command.request_id.clone();
        if request
            .command
            .validate_at(self.clock.now_ms(), self.config.validation_limits())
            .is_err()
        {
            return rejected(
                request_id,
                RecordingScheduleReasonCode::InvalidSchedule,
                None,
            );
        }
        match self.execute_valid(request).await {
            Ok(Some(schedule)) => accepted(request_id, Some(schedule)),
            Ok(None) => accepted(request_id, None),
            Err((reason, current)) => rejected(request_id, reason, current),
        }
    }
}

fn accepted(
    request_id: String,
    schedule: Option<RecordingSchedule>,
) -> RecordingScheduleCommandResult {
    RecordingScheduleCommandResult {
        protocol_version: RECORDING_SCHEDULE_PROTOCOL_VERSION,
        request_id,
        accepted: true,
        schedule,
        current_schedule: None,
        reason_code: None,
        detail: None,
    }
}

fn rejected(
    request_id: String,
    reason_code: RecordingScheduleReasonCode,
    current_schedule: Option<RecordingSchedule>,
) -> RecordingScheduleCommandResult {
    RecordingScheduleCommandResult {
        protocol_version: RECORDING_SCHEDULE_PROTOCOL_VERSION,
        request_id,
        accepted: false,
        schedule: None,
        current_schedule,
        reason_code: Some(reason_code),
        detail: None,
    }
}
