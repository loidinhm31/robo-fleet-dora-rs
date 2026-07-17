use crate::recording_access::RecordingAccess;
use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::{header, HeaderMap, Method, Response, StatusCode};
use std::ops::Range;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncSeekExt, SeekFrom};
use tokio_util::io::ReaderStream;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteRange {
    Full,
    Partial { start: u64, end: u64 },
}

impl ByteRange {
    fn bounds(self, length: u64) -> Range<u64> {
        match self {
            Self::Full => 0..length,
            Self::Partial { start, end } => start..end.saturating_add(1),
        }
    }
}

pub fn parse_range(value: Option<&str>, length: u64) -> Result<ByteRange, ()> {
    let Some(value) = value else {
        return Ok(ByteRange::Full);
    };
    let value = value.strip_prefix("bytes=").ok_or(())?;
    if value.contains(',') || length == 0 {
        return Err(());
    }
    let (start, end) = value.split_once('-').ok_or(())?;
    if start.is_empty() {
        let suffix = end.parse::<u64>().map_err(|_| ())?;
        if suffix == 0 {
            return Err(());
        }
        let start = length.saturating_sub(suffix);
        return Ok(ByteRange::Partial {
            start,
            end: length - 1,
        });
    }
    let start = start.parse::<u64>().map_err(|_| ())?;
    if start >= length {
        return Err(());
    }
    let end = if end.is_empty() {
        length - 1
    } else {
        end.parse::<u64>().map_err(|_| ())?.min(length - 1)
    };
    (start <= end)
        .then_some(ByteRange::Partial { start, end })
        .ok_or(())
}

pub async fn serve_playback(
    State(access): State<Arc<RecordingAccess>>,
    Path(ticket): Path<String>,
    headers: HeaderMap,
    method: Method,
) -> Response<Body> {
    let authorized = match access.authorize(&ticket) {
        Ok(file) => file,
        Err(_) => return simple_error(StatusCode::NOT_FOUND),
    };
    let range = match parse_range(
        headers
            .get(header::RANGE)
            .and_then(|value| value.to_str().ok()),
        authorized.length,
    ) {
        Ok(range) => range,
        Err(()) => {
            return Response::builder()
                .status(StatusCode::RANGE_NOT_SATISFIABLE)
                .header(
                    header::CONTENT_RANGE,
                    format!("bytes */{}", authorized.length),
                )
                .header(header::ACCEPT_RANGES, "bytes")
                .body(Body::empty())
                .unwrap();
        }
    };
    let bounds = range.bounds(authorized.length);
    let content_length = bounds.end.saturating_sub(bounds.start);
    let status = if matches!(range, ByteRange::Partial { .. }) {
        StatusCode::PARTIAL_CONTENT
    } else {
        StatusCode::OK
    };
    let mut builder = Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "video/mp4")
        .header(header::ACCEPT_RANGES, "bytes")
        .header(header::CONTENT_LENGTH, content_length.to_string());
    if let ByteRange::Partial { start, end } = range {
        builder = builder.header(
            header::CONTENT_RANGE,
            format!("bytes {start}-{end}/{}", authorized.length),
        );
    }
    if method == Method::HEAD || content_length == 0 {
        return builder.body(Body::empty()).unwrap();
    }
    let mut file = authorized.file;
    if file.seek(SeekFrom::Start(bounds.start)).await.is_err() {
        return simple_error(StatusCode::NOT_FOUND);
    }
    let stream = ReaderStream::new(file.take(content_length));
    builder.body(Body::from_stream(stream)).unwrap()
}

fn simple_error(status: StatusCode) -> Response<Body> {
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "text/plain; charset=utf-8")
        .body(Body::from(
            status.canonical_reason().unwrap_or("request failed"),
        ))
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_parser_supports_single_normal_and_suffix_ranges() {
        assert_eq!(parse_range(None, 100), Ok(ByteRange::Full));
        assert_eq!(
            parse_range(Some("bytes=10-19"), 100),
            Ok(ByteRange::Partial { start: 10, end: 19 })
        );
        assert_eq!(
            parse_range(Some("bytes=90-"), 100),
            Ok(ByteRange::Partial { start: 90, end: 99 })
        );
        assert_eq!(
            parse_range(Some("bytes=-10"), 100),
            Ok(ByteRange::Partial { start: 90, end: 99 })
        );
    }

    #[test]
    fn range_parser_rejects_multi_range_overflow_and_unsatisfiable_ranges() {
        for value in [
            "bytes=1-2,4-5",
            "bytes=100-101",
            "bytes=9-2",
            "bytes=-0",
            "bytes=18446744073709551616-",
        ] {
            assert!(parse_range(Some(value), 100).is_err(), "accepted {value}");
        }
    }
}
