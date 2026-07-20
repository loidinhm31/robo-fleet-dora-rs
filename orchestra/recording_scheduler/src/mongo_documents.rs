use bson::{self, Bson, DateTime, Document};
use robo_rover_lib::{RecordingOccurrence, RecordingSchedule};

use crate::domain::RecordingGroup;

pub fn schedule_document(
    schedule: &RecordingSchedule,
    next_occurrence_ms: Option<i64>,
) -> Result<Document, String> {
    let mut document = bson::to_document(schedule).map_err(|error| error.to_string())?;
    document.insert("_id", schedule.schedule_id.clone());
    document.insert("next_occurrence_ms", next_occurrence_ms);
    Ok(document)
}

pub fn occurrence_document(occurrence: &RecordingOccurrence) -> Result<Document, String> {
    let mut document = bson::to_document(occurrence).map_err(|error| error.to_string())?;
    document.insert("_id", occurrence.occurrence_id.clone());
    document.insert(
        "expire_at",
        occurrence.expires_at_ms.map(DateTime::from_millis),
    );
    Ok(document)
}

pub fn schedule_from_document(mut document: Document) -> Result<RecordingSchedule, String> {
    document.remove("_id");
    document.remove("next_occurrence_ms");
    bson::from_document(document).map_err(|error| error.to_string())
}

pub fn occurrence_from_document(mut document: Document) -> Result<RecordingOccurrence, String> {
    document.remove("_id");
    document.remove("expire_at");
    bson::from_document(document).map_err(|error| error.to_string())
}

pub fn group_from_document(mut document: Document) -> Result<RecordingGroup, String> {
    document.remove("_id");
    document.remove("_scheduler_revision");
    bson::from_document(document).map_err(|error| error.to_string())
}

pub fn document_revision(document: &Document) -> u64 {
    document
        .get_i64("_scheduler_revision")
        .ok()
        .and_then(|value| u64::try_from(value).ok())
        .unwrap_or_default()
}

pub fn is_signed_millisecond(value: &Bson) -> bool {
    matches!(value, Bson::Int64(ms) if *ms >= 0)
}
