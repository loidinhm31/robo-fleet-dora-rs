use std::{
    sync::{Arc, Mutex},
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use mongodb::{bson::doc, options::FindOneOptions, Collection, Database, IndexModel};
use robo_rover_lib::RecordingCoordinatorFeedback;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

const COLLECTION: &str = "recording_scheduler_feedback_spool";

#[derive(Clone)]
pub struct RecordingScheduleFeedbackSpool {
    records: Collection<StoredFeedback>,
    persistence_gate: Arc<Mutex<()>>,
}

#[derive(Debug, Clone)]
pub struct SpoolEntry {
    pub id: String,
    pub feedback: RecordingCoordinatorFeedback,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredFeedback {
    #[serde(rename = "_id")]
    id: String,
    feedback: RecordingCoordinatorFeedback,
    created_at_ms: i64,
}

impl RecordingScheduleFeedbackSpool {
    pub fn from_database(database: Database) -> Self {
        Self {
            records: database.collection(COLLECTION),
            persistence_gate: Arc::new(Mutex::new(())),
        }
    }

    pub async fn ensure_indexes(&self) -> Result<(), mongodb::error::Error> {
        self.records
            .create_index(
                IndexModel::builder()
                    .keys(doc! { "created_at_ms": 1_i32, "_id": 1_i32 })
                    .build(),
                None,
            )
            .await
            .map(|_| ())
    }

    /// Each emission is immutable. The scheduler already deduplicates by intent
    /// identity, while a unique spool ID prevents acknowledgement of an older
    /// delivery from deleting newer terminal or manual-suppression evidence.
    pub async fn persist(
        &self,
        feedback: RecordingCoordinatorFeedback,
    ) -> Result<(), mongodb::error::Error> {
        let id = Uuid::new_v4().to_string();
        self.records
            .insert_one(
                StoredFeedback {
                    id,
                    feedback,
                    created_at_ms: now_ms(),
                },
                None,
            )
            .await
            .map(|_| ())
    }

    /// This bridge has synchronous recorder and Socket.IO callbacks. A scoped
    /// runtime makes the Mongo write a durability boundary without introducing
    /// an unbounded in-process retry channel behind those callbacks.
    pub fn persist_blocking(&self, feedback: RecordingCoordinatorFeedback) -> Result<(), String> {
        // A single in-flight durable write bounds the retry work while Mongo is
        // unavailable. Other callbacks wait at this durability boundary rather
        // than spawning independent retry workers or retaining feedback in RAM.
        let _persistence_gate = self
            .persistence_gate
            .lock()
            .map_err(|_| "feedback spool persistence gate unavailable".to_owned())?;
        let mut retry_delay = Duration::from_millis(25);
        loop {
            let spool = self.clone();
            let retry_feedback = feedback.clone();
            let attempt = thread::scope(|scope| {
                scope
                    .spawn(move || {
                        tokio::runtime::Builder::new_current_thread()
                            .enable_all()
                            .build()
                            .map_err(|error| error.to_string())?
                            .block_on(spool.persist(retry_feedback))
                            .map_err(|error| error.to_string())
                    })
                    .join()
                    .map_err(|_| "feedback spool worker panicked".to_owned())?
            });
            match attempt {
                Ok(()) => return Ok(()),
                Err(error) => {
                    // Mongo is a durability dependency for scheduled transitions.
                    // Retrying in this callback deliberately pauses the caller
                    // rather than retaining unbounded recorder evidence in RAM.
                    tracing::error!(%error, "recording scheduler feedback spool unavailable; retrying");
                    thread::sleep(retry_delay);
                    retry_delay = (retry_delay * 2).min(Duration::from_secs(1));
                }
            }
        }
    }

    pub async fn next(&self) -> Result<Option<SpoolEntry>, mongodb::error::Error> {
        self.records
            .find_one(
                None,
                FindOneOptions::builder()
                    .sort(doc! { "created_at_ms": 1_i32, "_id": 1_i32 })
                    .build(),
            )
            .await
            .map(|record| {
                record.map(|record| SpoolEntry {
                    id: record.id,
                    feedback: record.feedback,
                })
            })
    }

    pub async fn acknowledge(&self, id: &str) -> Result<(), mongodb::error::Error> {
        self.records
            .delete_one(doc! { "_id": id }, None)
            .await
            .map(|_| ())
    }
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(i64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use robo_rover_lib::{RecordingScheduleReasonCode, RecordingSessionState};

    fn feedback(intent_id: String, accepted: bool) -> RecordingCoordinatorFeedback {
        RecordingCoordinatorFeedback {
            intent_id,
            occurrence_id: "occurrence".into(),
            generation: 1,
            accepted,
            applied: accepted,
            retryable: !accepted,
            group_id: Some("group".into()),
            recording_id: None,
            recorder_state: Some(RecordingSessionState::Failed),
            manual_suppression: false,
            reason_code: Some(RecordingScheduleReasonCode::Unavailable),
            detail: Some("spool-test".into()),
        }
    }

    #[tokio::test]
    async fn persists_retries_and_acknowledges_against_configured_mongo() {
        let Ok(uri) = std::env::var("RECORDING_FEEDBACK_SPOOL_TEST_URI") else {
            return;
        };
        let database = format!("recording_spool_{}", Uuid::new_v4().simple());
        let options = mongodb::options::ClientOptions::parse(uri).await.unwrap();
        let db = mongodb::Client::with_options(options)
            .unwrap()
            .database(&database);
        let spool = RecordingScheduleFeedbackSpool::from_database(db.clone());
        spool.ensure_indexes().await.unwrap();

        let intent_id = Uuid::new_v4().to_string();
        let blocking_spool = spool.clone();
        let blocking_feedback = feedback(intent_id.clone(), false);
        tokio::task::spawn_blocking(move || blocking_spool.persist_blocking(blocking_feedback))
            .await
            .unwrap()
            .unwrap();
        let first = spool.next().await.unwrap().unwrap();
        assert_eq!(first.feedback.intent_id, intent_id);
        assert_eq!(spool.next().await.unwrap().unwrap().id, first.id);

        spool
            .persist(feedback(intent_id.clone(), true))
            .await
            .unwrap();
        // Acknowledge the fetched older entry after a newer emission arrived.
        // The newer record must remain for delivery rather than being erased by
        // the older acknowledgement.
        spool.acknowledge(&first.id).await.unwrap();
        let second = spool.next().await.unwrap().unwrap();
        assert!(second.feedback.accepted);
        assert_ne!(first.id, second.id);
        spool.acknowledge(&second.id).await.unwrap();
        assert!(spool.next().await.unwrap().is_none());
        db.drop(None).await.unwrap();
    }
}
