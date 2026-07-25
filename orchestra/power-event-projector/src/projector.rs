use crate::mongo_repository::MongoRepository;
use power_coordinator::JournalRecord;

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
    pub async fn project(&self, record: &JournalRecord) -> Result<(), String> {
        self.repository.project(&self.deployment_id, record).await
    }
}
