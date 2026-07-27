use crate::mongo_repository::MongoRepository;
use power_coordinator::JournalRecord;
use serde::Serialize;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ProjectionHealth {
    pub healthy: bool,
    pub attempts: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<ProjectionFailureReason>,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProjectionFailureReason {
    MongoUnavailable,
}

pub struct PowerEventProjector {
    deployment_id: String,
    repository: MongoRepository,
}

impl PowerEventProjector {
    pub fn new(deployment_id: String, repository: MongoRepository) -> Self {
        Self {
            deployment_id,
            repository,
        }
    }
    pub async fn initialize(&self) -> Result<(), String> {
        self.repository
            .ensure_indexes()
            .await
            .map_err(|error| error.to_string())
    }

    pub async fn open_with_retry(
        deployment_id: String,
        mongodb_uri: &str,
        mongodb_database: &str,
        max_attempts: u8,
        backoff_ms: u64,
    ) -> Result<Self, ProjectionHealth> {
        let attempts = max_attempts.max(1);
        for attempt in 1..=attempts {
            let opened = async {
                let repository = MongoRepository::connect(mongodb_uri, mongodb_database)
                    .await
                    .map_err(|error| error.to_string())?;
                let projector = Self::new(deployment_id.clone(), repository);
                projector.initialize().await?;
                Ok::<_, String>(projector)
            }
            .await;
            if let Ok(projector) = opened {
                return Ok(projector);
            }
            if attempt < attempts {
                tokio::time::sleep(Duration::from_millis(
                    backoff_ms.saturating_mul(attempt as u64),
                ))
                .await;
            }
        }
        Err(ProjectionHealth {
            healthy: false,
            attempts,
            reason: Some(ProjectionFailureReason::MongoUnavailable),
        })
    }
    pub async fn project(&self, record: &JournalRecord) -> Result<(), String> {
        self.repository.project(&self.deployment_id, record).await
    }

    /// Retry projection only; the caller never acknowledges the journal record
    /// until this method reports a durable Mongo write.
    pub async fn project_with_retry(
        &self,
        record: &JournalRecord,
        max_attempts: u8,
        backoff_ms: u64,
    ) -> ProjectionHealth {
        let attempts = max_attempts.max(1);
        for attempt in 1..=attempts {
            match self.project(record).await {
                Ok(()) => {
                    return ProjectionHealth {
                        healthy: true,
                        attempts: attempt,
                        reason: None,
                    }
                }
                Err(_) => {
                    if attempt < attempts {
                        tokio::time::sleep(Duration::from_millis(
                            backoff_ms.saturating_mul(attempt as u64),
                        ))
                        .await;
                    }
                }
            }
        }
        ProjectionHealth {
            healthy: false,
            attempts,
            reason: Some(ProjectionFailureReason::MongoUnavailable),
        }
    }
}
