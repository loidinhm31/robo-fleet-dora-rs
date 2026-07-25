use uuid::Uuid;

pub(crate) const MAX_DETAIL_BYTES: usize = 256;
const MAX_ID_BYTES: usize = 128;

pub(crate) fn validate_id(label: &str, value: &str) -> Result<(), String> {
    if value.is_empty()
        || value.len() > MAX_ID_BYTES
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(format!("invalid power {label}"));
    }
    Ok(())
}

pub(crate) fn validate_uuid(label: &str, value: &str) -> Result<(), String> {
    let uuid = Uuid::parse_str(value).map_err(|_| format!("invalid power {label}"))?;
    if uuid.hyphenated().to_string() != value {
        return Err(format!("invalid power {label}"));
    }
    Ok(())
}

pub(crate) fn validate_detail(detail: Option<&String>) -> Result<(), String> {
    let Some(detail) = detail else {
        return Ok(());
    };
    if detail.len() > MAX_DETAIL_BYTES
        || detail
            .bytes()
            .any(|byte| !byte.is_ascii_graphic() && byte != b' ')
    {
        return Err("invalid power detail".into());
    }
    Ok(())
}
