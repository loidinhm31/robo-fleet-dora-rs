use chrono::{DateTime, Datelike, Days, LocalResult, NaiveDateTime, TimeZone, Utc};
use chrono_tz::Tz;
use robo_rover_lib::{DstResolution, IsoWeekday, RecordingSchedule, RecordingScheduleRecurrence};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedStart {
    pub start_ms: i64,
    pub resolution: DstResolution,
}

pub fn candidates(
    schedule: &RecordingSchedule,
    from_ms: i64,
    through_ms: i64,
) -> Result<Vec<ResolvedStart>, String> {
    let local = schedule.definition.recurrence.local_start();
    let timezone: Tz = local.timezone.parse().map_err(|_| "invalid timezone")?;
    let anchor = parse_local(local.date.as_str(), local.time.as_str())?;
    let mut result = Vec::new();
    let start_date = DateTime::from_timestamp_millis(from_ms)
        .ok_or("invalid lower bound")?
        .with_timezone(&timezone)
        .date_naive();
    let mut date = anchor.date().max(start_date);
    let end_date = DateTime::from_timestamp_millis(through_ms)
        .ok_or("invalid horizon")?
        .with_timezone(&timezone)
        .date_naive();
    while date <= end_date {
        if matches_date(&schedule.definition.recurrence, date, anchor.date()) {
            let resolved = resolve_local(timezone, date.and_time(anchor.time()))?;
            if (from_ms..=through_ms).contains(&resolved.start_ms) {
                result.push(resolved);
            }
        }
        date = date.checked_add_days(Days::new(1)).ok_or("date overflow")?;
    }
    Ok(result)
}

pub fn resolve_local(timezone: Tz, local: NaiveDateTime) -> Result<ResolvedStart, String> {
    match timezone.from_local_datetime(&local) {
        LocalResult::Single(value) => Ok(resolved(value, DstResolution::Exact)),
        LocalResult::Ambiguous(first, _) => Ok(resolved(first, DstResolution::FoldEarlier)),
        LocalResult::None => {
            for minutes in 1..=(24 * 60) {
                let shifted = local + chrono::Duration::minutes(minutes);
                if let LocalResult::Single(value) | LocalResult::Ambiguous(value, _) =
                    timezone.from_local_datetime(&shifted)
                {
                    return Ok(resolved(value, DstResolution::GapShifted));
                }
            }
            Err("unable to resolve DST gap".into())
        }
    }
}

fn matches_date(
    recurrence: &RecordingScheduleRecurrence,
    date: chrono::NaiveDate,
    anchor: chrono::NaiveDate,
) -> bool {
    match recurrence {
        RecordingScheduleRecurrence::OneTime { .. } => date == anchor,
        RecordingScheduleRecurrence::Daily { .. } => date >= anchor,
        RecordingScheduleRecurrence::Weekly { weekdays, .. } => {
            date >= anchor
                && weekdays
                    .iter()
                    .any(|weekday| weekday_number(*weekday) == date.weekday().number_from_monday())
        }
    }
}

fn parse_local(date: &str, time: &str) -> Result<NaiveDateTime, String> {
    let value = format!("{date} {time}");
    ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"]
        .iter()
        .find_map(|format| NaiveDateTime::parse_from_str(&value, format).ok())
        .ok_or_else(|| "invalid local start".into())
}

fn resolved(value: DateTime<Tz>, resolution: DstResolution) -> ResolvedStart {
    ResolvedStart {
        start_ms: value.with_timezone(&Utc).timestamp_millis(),
        resolution,
    }
}

fn weekday_number(weekday: IsoWeekday) -> u32 {
    match weekday {
        IsoWeekday::Monday => 1,
        IsoWeekday::Tuesday => 2,
        IsoWeekday::Wednesday => 3,
        IsoWeekday::Thursday => 4,
        IsoWeekday::Friday => 5,
        IsoWeekday::Saturday => 6,
        IsoWeekday::Sunday => 7,
    }
}
