use std::time::Duration;

use futures_util::TryStreamExt;
use mongodb::{
    bson::{doc, Bson, Document},
    options::{ClientOptions, IndexOptions},
    Collection, Database, IndexModel,
};
use robo_rover_lib::{RecordingOccurrence, RecordingSchedule, ScheduledRecordingIntent};

use crate::domain::RecordingGroup;
use crate::mongo_documents::{
    document_revision, group_from_document, occurrence_document, occurrence_from_document,
    schedule_document, schedule_from_document,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stored<T> {
    pub value: T,
    pub revision: u64,
}

#[derive(Clone)]
pub struct MongoRepository {
    schedules: Collection<Document>,
    occurrences: Collection<Document>,
    groups: Collection<Document>,
    outbox: Collection<Document>,
}

impl MongoRepository {
    pub async fn connect(uri: &str, database: &str) -> Result<Self, mongodb::error::Error> {
        let options = ClientOptions::parse(uri).await?;
        let database = mongodb::Client::with_options(options)?.database(database);
        Ok(Self::from_database(database))
    }

    pub fn from_database(database: Database) -> Self {
        Self {
            schedules: database.collection("recording_schedules"),
            occurrences: database.collection("recording_occurrences"),
            groups: database.collection("recording_scheduler_groups"),
            outbox: database.collection("recording_scheduler_outbox"),
        }
    }

    pub async fn ensure_indexes(&self) -> Result<(), mongodb::error::Error> {
        self.schedules
            .create_index(index(doc! {"schedule_id": 1}, true), None)
            .await?;
        self.schedules
            .create_index(
                index(
                    doc! {"entity_id": 1, "enabled": 1, "next_occurrence_ms": 1},
                    false,
                ),
                None,
            )
            .await?;
        for model in [
            index(doc! {"occurrence_id": 1}, true),
            index(
                doc! {"schedule_id": 1, "schedule_revision": 1, "planned_start_ms": 1},
                true,
            ),
            index(
                doc! {"entity_id": 1, "state": 1, "planned_end_ms": 1},
                false,
            ),
            ttl_index(),
        ] {
            self.occurrences.create_index(model, None).await?;
        }
        self.groups
            .create_index(index(doc! {"group_id": 1}, true), None)
            .await?;
        self.groups
            .create_index(index(doc! {"entity_id": 1, "start_ms": 1}, true), None)
            .await
            .map(|_| ())?;
        self.outbox
            .create_index(index(doc! {"intent_id": 1}, true), None)
            .await
            .map(|_| ())
    }

    pub async fn insert_schedule(&self, schedule: &RecordingSchedule) -> Result<(), String> {
        self.schedules
            .insert_one(schedule_document(schedule, None)?, None)
            .await
            .map(|_| ())
            .map_err(|error| error.to_string())
    }

    pub async fn replace_schedule_cas(
        &self,
        schedule: &RecordingSchedule,
        expected: u64,
    ) -> Result<bool, String> {
        let result = self
            .schedules
            .replace_one(
                doc! {"schedule_id": &schedule.schedule_id, "revision": expected as i64},
                schedule_document(schedule, None)?,
                None,
            )
            .await
            .map_err(|error| error.to_string())?;
        Ok(result.matched_count == 1)
    }

    pub async fn find_schedule(
        &self,
        schedule_id: &str,
    ) -> Result<Option<RecordingSchedule>, String> {
        self.schedules
            .find_one(doc! {"schedule_id": schedule_id}, None)
            .await
            .map_err(|error| error.to_string())?
            .map(schedule_from_document)
            .transpose()
    }

    pub async fn enabled_count(&self, entity_id: &str) -> Result<u64, String> {
        self.schedules
            .count_documents(doc! {"entity_id": entity_id, "enabled": true}, None)
            .await
            .map_err(|error| error.to_string())
    }

    pub async fn delete_schedule_cas(
        &self,
        schedule_id: &str,
        expected: u64,
    ) -> Result<bool, String> {
        let result = self
            .schedules
            .delete_one(
                doc! {"schedule_id": schedule_id, "revision": expected as i64},
                None,
            )
            .await
            .map_err(|error| error.to_string())?;
        Ok(result.deleted_count == 1)
    }

    pub async fn cancel_future(
        &self,
        schedule_id: &str,
        through_revision: u64,
        now_ms: i64,
    ) -> Result<(), String> {
        let expiry = now_ms.saturating_add(90 * 24 * 60 * 60 * 1_000);
        self.occurrences.update_many(
            doc! {"schedule_id": schedule_id, "schedule_revision": {"$lte": through_revision as i64}, "planned_start_ms": {"$gt": now_ms}, "terminal_at_ms": Bson::Null},
            doc! {"$set": {"state": "cancelled", "terminal_at_ms": now_ms, "expires_at_ms": expiry, "expire_at": mongodb::bson::DateTime::from_millis(expiry)}, "$inc": {"_scheduler_revision": 1_i64}},
            None,
        ).await.map(|_| ()).map_err(|error| error.to_string())
    }

    pub async fn save_occurrence(
        &self,
        occurrence: &RecordingOccurrence,
        expected_revision: u64,
    ) -> Result<Option<u64>, String> {
        let document = occurrence_document(occurrence)?;
        self.save_document(
            &self.occurrences,
            "occurrence_id",
            &occurrence.occurrence_id,
            document,
            expected_revision,
        )
        .await
    }

    pub async fn save_group(
        &self,
        group: &RecordingGroup,
        expected_revision: u64,
    ) -> Result<Option<u64>, String> {
        let mut document = mongodb::bson::to_document(group).map_err(|error| error.to_string())?;
        document.insert("_id", group.group_id.clone());
        self.save_document(
            &self.groups,
            "group_id",
            &group.group_id,
            document,
            expected_revision,
        )
        .await
    }

    pub async fn persist_intent(&self, intent: &ScheduledRecordingIntent) -> Result<(), String> {
        let mut document = mongodb::bson::to_document(intent).map_err(|error| error.to_string())?;
        document.insert("_id", intent.intent_id.clone());
        self.outbox
            .insert_one(document, None)
            .await
            .map(|_| ())
            .or_else(|error| {
                if error.to_string().contains("E11000") {
                    Ok(())
                } else {
                    Err(error)
                }
            })
            .map_err(|error| error.to_string())
    }

    pub async fn pending_intents(&self) -> Result<Vec<ScheduledRecordingIntent>, String> {
        self.outbox
            .find(None, None)
            .await
            .map_err(|error| error.to_string())?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|error| error.to_string())?
            .into_iter()
            .map(|mut document| {
                document.remove("_id");
                mongodb::bson::from_document(document).map_err(|error| error.to_string())
            })
            .collect()
    }

    pub async fn acknowledge_intent(&self, intent_id: &str) -> Result<(), String> {
        self.outbox
            .delete_one(doc! {"intent_id": intent_id}, None)
            .await
            .map(|_| ())
            .map_err(|error| error.to_string())
    }

    pub async fn load_nonterminal(&self) -> Result<Vec<RecordingOccurrence>, String> {
        Ok(self
            .load_nonterminal_stored()
            .await?
            .into_iter()
            .map(|stored| stored.value)
            .collect())
    }

    pub async fn load_nonterminal_stored(
        &self,
    ) -> Result<Vec<Stored<RecordingOccurrence>>, String> {
        self.occurrences
            .find(doc! {"terminal_at_ms": Bson::Null}, None)
            .await
            .map_err(|error| error.to_string())?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|error| error.to_string())?
            .into_iter()
            .map(|document| {
                let revision = document_revision(&document);
                occurrence_from_document(document).map(|value| Stored { value, revision })
            })
            .collect()
    }

    pub async fn load_groups(&self) -> Result<Vec<Stored<RecordingGroup>>, String> {
        self.groups
            .find(None, None)
            .await
            .map_err(|error| error.to_string())?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|error| error.to_string())?
            .into_iter()
            .map(|document| {
                let revision = document_revision(&document);
                group_from_document(document).map(|value| Stored { value, revision })
            })
            .collect()
    }

    pub async fn load_schedules(&self) -> Result<Vec<RecordingSchedule>, String> {
        self.schedules
            .find(None, None)
            .await
            .map_err(|error| error.to_string())?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|error| error.to_string())?
            .into_iter()
            .map(schedule_from_document)
            .collect()
    }

    async fn save_document(
        &self,
        collection: &Collection<Document>,
        id_field: &str,
        id: &str,
        mut document: Document,
        expected_revision: u64,
    ) -> Result<Option<u64>, String> {
        let next_revision = expected_revision.saturating_add(1);
        document.insert("_scheduler_revision", next_revision as i64);
        let mut filter = doc! {"_scheduler_revision": expected_revision as i64};
        filter.insert(id_field, id);
        let replaced = collection
            .replace_one(filter, document.clone(), None)
            .await
            .map_err(|error| error.to_string())?;
        if replaced.matched_count == 1 {
            return Ok(Some(next_revision));
        }
        if expected_revision != 0 {
            return Ok(None);
        }
        if collection
            .find_one(identity_filter(id_field, id), None)
            .await
            .map_err(|error| error.to_string())?
            .is_some()
        {
            return Ok(None);
        }
        collection
            .insert_one(document, None)
            .await
            .map(|_| Some(next_revision))
            .or_else(|error| {
                if error.to_string().contains("E11000") {
                    Ok(None)
                } else {
                    Err(error)
                }
            })
            .map_err(|error| error.to_string())
    }
}

fn identity_filter(id_field: &str, id: &str) -> Document {
    let mut filter = Document::new();
    filter.insert(id_field, id);
    filter
}

fn index(keys: Document, unique: bool) -> IndexModel {
    IndexModel::builder()
        .keys(keys)
        .options(IndexOptions::builder().unique(unique).build())
        .build()
}

fn ttl_index() -> IndexModel {
    IndexModel::builder()
        .keys(doc! {"expire_at": 1})
        .options(IndexOptions::builder().expire_after(Duration::ZERO).build())
        .build()
}
