use mongodb::bson::{self, doc, Document};
use power_coordinator::JournalRecord;
use std::time::{SystemTime, UNIX_EPOCH};

pub const HISTORY_RETENTION_MS: i64 = 90 * 24 * 60 * 60 * 1_000;

pub fn event_document(deployment_id: &str, record: &JournalRecord) -> Result<Document, String> {
    record.validate()?;
    let event = bson::to_document(&record.event).map_err(|error| error.to_string())?;
    let expiry = record.event.occurred_at_ms as i64 + HISTORY_RETENTION_MS;
    Ok(doc! {
        "deployment_id": deployment_id,
        "entity_id": &record.event.entity_id,
        "role": bson::to_bson(&record.event.role).map_err(|error| error.to_string())?,
        "event_id": &record.event.event_id,
        "authority_epoch": record.event.authority.epoch as i64,
        "sequence": record.event.authority.sequence as i64,
        "journal_sequence": record.sequence as i64,
        "occurred_at": bson::DateTime::from_millis(record.event.occurred_at_ms as i64),
        "expires_at": bson::DateTime::from_millis(expiry),
        "event": event,
        "record": bson::to_document(record).map_err(|error| error.to_string())?,
    })
}

pub fn current_document(
    deployment_id: &str,
    record: &JournalRecord,
) -> Result<Option<Document>, String> {
    let Some(status) = &record.status else {
        return Ok(None);
    };
    status.validate()?;
    Ok(Some(doc! {
        "deployment_id": deployment_id,
        "entity_id": &status.entity_id,
        "role": bson::to_bson(&status.role).map_err(|error| error.to_string())?,
        "authority_epoch": record.event.authority.epoch as i64,
        "sequence": record.event.authority.sequence as i64,
        "updated_at": bson::DateTime::from_millis(status.updated_at_ms as i64),
        "status": bson::to_document(status).map_err(|error| error.to_string())?,
    }))
}

pub fn clamp_window(from_ms: Option<i64>, to_ms: Option<i64>) -> (i64, i64) {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64;
    (
        from_ms
            .unwrap_or(now - HISTORY_RETENTION_MS)
            .max(now - HISTORY_RETENTION_MS),
        to_ms.unwrap_or(now).min(now),
    )
}

pub fn cursor_filter(cursor: Option<(&bson::DateTime, &str)>) -> Option<Document> {
    cursor.map(|(at, event_id)| doc! { "$or": [ { "occurred_at": { "$lt": at } }, { "occurred_at": at, "event_id": { "$gt": event_id } } ] })
}
