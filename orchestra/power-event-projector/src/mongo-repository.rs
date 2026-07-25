use crate::mongo_documents::{clamp_window, current_document, cursor_filter, event_document};
use futures_util::TryStreamExt;
use mongodb::{
    bson::{doc, DateTime, Document},
    options::{ClientOptions, FindOptions, IndexOptions, UpdateOptions},
    Client, Collection, Database, IndexModel,
};
use power_coordinator::JournalRecord;

pub struct MongoRepository {
    events: Collection<Document>,
    current: Collection<Document>,
}

impl MongoRepository {
    pub async fn connect(uri: &str, database: &str) -> Result<Self, mongodb::error::Error> {
        let options = ClientOptions::parse(uri).await?;
        Ok(Self::from_database(
            Client::with_options(options)?.database(database),
        ))
    }
    pub fn from_database(database: Database) -> Self {
        Self {
            events: database.collection("power_lifecycle_events"),
            current: database.collection("power_current_state"),
        }
    }

    pub async fn ensure_indexes(&self) -> Result<(), mongodb::error::Error> {
        for index in [
            index(
                doc! {"deployment_id": 1, "entity_id": 1, "event_id": 1},
                true,
                None,
            ),
            index(
                doc! {"deployment_id": 1, "entity_id": 1, "occurred_at": -1},
                false,
                None,
            ),
            index(
                doc! {"deployment_id": 1, "entity_id": 1, "event.event_type": 1, "occurred_at": -1},
                false,
                None,
            ),
            index(
                doc! {"deployment_id": 1, "entity_id": 1, "event.transition_id": 1, "occurred_at": -1},
                false,
                None,
            ),
            index(
                doc! {"deployment_id": 1, "entity_id": 1, "event.reason_code": 1, "occurred_at": -1},
                false,
                None,
            ),
            index(doc! {"expires_at": 1}, false, Some(0)),
        ] {
            self.events.create_index(index, None).await?;
        }
        self.current
            .create_index(
                index(
                    doc! {"deployment_id": 1, "entity_id": 1, "role": 1},
                    true,
                    None,
                ),
                None,
            )
            .await
            .map(|_| ())
    }

    pub async fn project(&self, deployment_id: &str, record: &JournalRecord) -> Result<(), String> {
        let event = event_document(deployment_id, record)?;
        let filter = doc! {"deployment_id": deployment_id, "entity_id": &record.event.entity_id, "event_id": &record.event.event_id};
        self.events
            .update_one(
                filter,
                doc! {"$setOnInsert": event},
                UpdateOptions::builder().upsert(true).build(),
            )
            .await
            .map_err(|error| error.to_string())?;
        if let Some(current) = current_document(deployment_id, record)? {
            self.advance_current(deployment_id, &current).await?;
        }
        Ok(())
    }

    #[allow(dead_code)] // Exposed to the authenticated history API in Phase 07.
    pub async fn history(
        &self,
        deployment_id: &str,
        entity_id: &str,
        from_ms: Option<i64>,
        to_ms: Option<i64>,
        cursor: Option<(DateTime, String)>,
        limit: i64,
    ) -> Result<Vec<Document>, String> {
        let (from, to) = clamp_window(from_ms, to_ms);
        let mut clauses = vec![
            doc! {"deployment_id": deployment_id, "entity_id": entity_id, "occurred_at": {"$gte": DateTime::from_millis(from), "$lte": DateTime::from_millis(to)}},
        ];
        if let Some(filter) = cursor_filter(cursor.as_ref().map(|(at, id)| (at, id.as_str()))) {
            clauses.push(filter);
        }
        self.events
            .find(
                doc! {"$and": clauses},
                FindOptions::builder()
                    .sort(doc! {"occurred_at": -1, "event_id": 1})
                    .limit(limit.clamp(1, 100))
                    .build(),
            )
            .await
            .map_err(|error| error.to_string())?
            .try_collect()
            .await
            .map_err(|error| error.to_string())
    }

    async fn advance_current(&self, deployment_id: &str, current: &Document) -> Result<(), String> {
        let epoch = current
            .get_i64("authority_epoch")
            .map_err(|error| error.to_string())?;
        let sequence = current
            .get_i64("sequence")
            .map_err(|error| error.to_string())?;
        let filter = doc! {"deployment_id": deployment_id, "entity_id": current.get_str("entity_id").map_err(|error| error.to_string())?, "role": current.get("role").cloned().ok_or("missing current role")?, "$or": [{"authority_epoch": {"$lt": epoch}}, {"authority_epoch": epoch, "sequence": {"$lt": sequence}}, {"authority_epoch": {"$exists": false}}]};
        match self
            .current
            .update_one(
                filter,
                doc! {"$set": current},
                UpdateOptions::builder().upsert(true).build(),
            )
            .await
        {
            Ok(_) => Ok(()),
            Err(error) if error.to_string().contains("E11000") => Ok(()),
            Err(error) => Err(error.to_string()),
        }
    }
}

fn index(keys: Document, unique: bool, ttl_seconds: Option<u64>) -> IndexModel {
    IndexModel::builder()
        .keys(keys)
        .options(
            IndexOptions::builder()
                .unique(unique)
                .expire_after(ttl_seconds.map(std::time::Duration::from_secs))
                .build(),
        )
        .build()
}
