use futures_util::TryStreamExt;
use mongodb::{
    bson::{doc, DateTime, Document},
    options::FindOptions,
    Database,
};
use robo_rover_lib::{
    PowerEvent, PowerEventType, PowerReasonCode, PowerStatus, POWER_PROTOCOL_VERSION,
};
use serde::{Deserialize, Serialize};

const HISTORY_WINDOW_MS: i64 = 90 * 24 * 60 * 60 * 1_000;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PowerHistoryQuery {
    pub protocol_version: u8,
    pub request_id: String,
    pub cursor: Option<String>,
    pub limit: Option<i64>,
    pub from_ms: Option<i64>,
    pub to_ms: Option<i64>,
    pub event_type: Option<PowerEventType>,
    pub reason_code: Option<PowerReasonCode>,
}

#[derive(Debug, Serialize)]
pub struct PowerHistoryResult {
    pub protocol_version: u8,
    pub request_id: String,
    pub accepted: bool,
    pub events: Vec<PowerEvent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub historical_status: Option<PowerStatus>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<PowerReasonCode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Clone)]
pub struct PowerHistoryGateway {
    database: Database,
    deployment_id: String,
}

impl PowerHistoryGateway {
    pub fn new(database: Database) -> Self {
        Self {
            database,
            deployment_id: std::env::var("POWER_DEPLOYMENT_ID")
                .unwrap_or_else(|_| "default".into()),
        }
    }

    pub async fn query(
        &self,
        entity_id: &str,
        query: PowerHistoryQuery,
        now_ms: i64,
    ) -> PowerHistoryResult {
        if query.protocol_version != POWER_PROTOCOL_VERSION || !canonical_uuid(&query.request_id) {
            return reject(
                query.request_id,
                PowerReasonCode::InvalidRequest,
                "invalid history request",
            );
        }
        let from = query
            .from_ms
            .unwrap_or(now_ms - HISTORY_WINDOW_MS)
            .clamp(now_ms - HISTORY_WINDOW_MS, now_ms);
        let to = query.to_ms.unwrap_or(now_ms).clamp(from, now_ms);
        let mut clauses = vec![
            doc! { "deployment_id": &self.deployment_id, "entity_id": entity_id, "occurred_at": { "$gte": DateTime::from_millis(from), "$lte": DateTime::from_millis(to) } },
        ];
        match query.cursor.as_deref().map(parse_cursor).transpose() {
            Ok(Some((at, id))) => clauses.push(doc! { "$or": [{ "occurred_at": { "$lt": DateTime::from_millis(at) } }, { "occurred_at": DateTime::from_millis(at), "event_id": { "$gt": id } }] }),
            Ok(None) => {}
            Err(()) => return reject(query.request_id, PowerReasonCode::InvalidRequest, "invalid history cursor"),
        }
        if let Some(value) = query.event_type {
            clauses.push(
                doc! { "event.event_type": mongodb::bson::to_bson(&value).unwrap_or_default() },
            );
        }
        if let Some(value) = query.reason_code {
            clauses.push(
                doc! { "event.reason_code": mongodb::bson::to_bson(&value).unwrap_or_default() },
            );
        }
        let limit = query.limit.unwrap_or(50).clamp(1, 100);
        let fetched = self
            .database
            .collection::<Document>("power_lifecycle_events")
            .find(
                doc! { "$and": clauses },
                FindOptions::builder()
                    .sort(doc! { "occurred_at": -1, "event_id": 1 })
                    .limit(limit + 1)
                    .build(),
            )
            .await;
        let Ok(cursor) = fetched else {
            return reject(
                query.request_id,
                PowerReasonCode::Internal,
                "power history is unavailable",
            );
        };
        let Ok(mut documents) = cursor.try_collect::<Vec<_>>().await else {
            return reject(
                query.request_id,
                PowerReasonCode::Internal,
                "power history is unavailable",
            );
        };
        let more = documents.len() > limit as usize;
        documents.truncate(limit as usize);
        let events = documents
            .iter()
            .filter_map(|document| document.get_document("event").ok())
            .filter_map(|event| mongodb::bson::from_document::<PowerEvent>(event.clone()).ok())
            .collect::<Vec<_>>();
        let next_cursor = more
            .then(|| documents.last().and_then(cursor_from_document))
            .flatten();
        let historical_status = self.database.collection::<Document>("power_current_state").find_one(doc! { "deployment_id": &self.deployment_id, "entity_id": entity_id, "role": "rover" }, None).await.ok().flatten().and_then(|document| document.get_document("status").ok().cloned()).and_then(|status| mongodb::bson::from_document::<PowerStatus>(status).ok()).filter(|status| status.validate().is_ok());
        PowerHistoryResult {
            protocol_version: POWER_PROTOCOL_VERSION,
            request_id: query.request_id,
            accepted: true,
            events,
            historical_status,
            next_cursor,
            reason_code: None,
            detail: None,
        }
    }
}

fn parse_cursor(value: &str) -> Result<(i64, &str), ()> {
    let (at, id) = value.split_once(':').ok_or(())?;
    let at = at.parse::<i64>().map_err(|_| ())?;
    (!id.is_empty() && canonical_uuid(id))
        .then_some((at, id))
        .ok_or(())
}
fn cursor_from_document(document: &Document) -> Option<String> {
    Some(format!(
        "{}:{}",
        document
            .get_datetime("occurred_at")
            .ok()?
            .timestamp_millis(),
        document.get_str("event_id").ok()?
    ))
}
fn canonical_uuid(value: &str) -> bool {
    uuid::Uuid::parse_str(value)
        .map(|id| id.hyphenated().to_string() == value)
        .unwrap_or(false)
}
fn reject(request_id: String, reason_code: PowerReasonCode, detail: &str) -> PowerHistoryResult {
    PowerHistoryResult {
        protocol_version: POWER_PROTOCOL_VERSION,
        request_id,
        accepted: false,
        events: Vec::new(),
        historical_status: None,
        next_cursor: None,
        reason_code: Some(reason_code),
        detail: Some(detail.into()),
    }
}
