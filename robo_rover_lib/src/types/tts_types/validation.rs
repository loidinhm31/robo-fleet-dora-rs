const MAX_EXTERNAL_DETAIL_CHARS: usize = 256;
const MAX_WIRE_INTEGER: u64 = 9_007_199_254_740_991;

pub(super) fn validate_timestamp(timestamp: u64) -> Result<(), String> {
    validate_wire_integer(timestamp, "voice timestamp")
}

pub(super) fn validate_wire_integer(value: u64, name: &str) -> Result<(), String> {
    if value > MAX_WIRE_INTEGER {
        return Err(format!("{name} exceeds the exact JSON integer range"));
    }
    Ok(())
}

pub(super) fn validate_external_detail(detail: &Option<String>) -> Result<(), String> {
    let Some(detail) = detail else {
        return Ok(());
    };
    if detail.chars().count() > MAX_EXTERNAL_DETAIL_CHARS {
        return Err(format!(
            "voice error detail exceeds {MAX_EXTERNAL_DETAIL_CHARS} characters"
        ));
    }
    if detail.chars().any(char::is_control) || detail.split_whitespace().any(looks_like_path) {
        return Err("voice error detail is not sanitized".into());
    }
    Ok(())
}

/// Normalize external error detail, redact absolute paths, and enforce a wire bound.
pub fn sanitize_external_detail(detail: &str) -> Option<String> {
    let normalized = detail
        .split_whitespace()
        .map(|token| {
            if looks_like_path(token) {
                "<redacted-path>"
            } else {
                token
            }
        })
        .collect::<Vec<_>>()
        .join(" ");
    let bounded = normalized
        .chars()
        .take(MAX_EXTERNAL_DETAIL_CHARS)
        .collect::<String>();
    (!bounded.is_empty()).then_some(bounded)
}

fn looks_like_path(token: &str) -> bool {
    let value = token.trim_matches(|character: char| "'\"()[]{}.,;".contains(character));
    let lower = value.to_ascii_lowercase();
    value.starts_with('/')
        || value.starts_with("\\\\")
        || lower.starts_with("file://")
        || value.contains("=/")
        || value.contains("=\\")
        || value.split_once(':').is_some_and(|(prefix, remainder)| {
            !lower.contains("://")
                && !prefix.is_empty()
                && prefix.chars().all(|character| {
                    character.is_ascii_alphanumeric() || matches!(character, '_' | '-')
                })
                && matches!(remainder.chars().next(), Some('/' | '\\'))
        })
        || value
            .as_bytes()
            .get(1..3)
            .is_some_and(|pair| pair[0] == b':' && matches!(pair[1], b'/' | b'\\'))
}
