use super::*;
use dora_node_api::arrow::array::{BinaryArray, Float32Array};
use std::collections::BTreeMap;

fn metadata(format: &str, count: i64) -> MetadataParameters {
    BTreeMap::from([
        (
            "stream_id".into(),
            Parameter::String(Uuid::new_v4().to_string()),
        ),
        ("frame_id".into(), Parameter::Integer(4)),
        ("capture_timestamp_ms".into(), Parameter::Integer(100)),
        ("sample_rate".into(), Parameter::Integer(16_000)),
        ("channels".into(), Parameter::Integer(1)),
        ("sample_count".into(), Parameter::Integer(count)),
        ("format".into(), Parameter::String(format.into())),
    ])
}

#[test]
fn rover_s16le_conversion_preserves_boundaries() {
    let mut params = metadata("s16le", 3);
    params.insert("entity_id".into(), Parameter::String("rover-a".into()));
    let bytes = [
        i16::MIN.to_le_bytes(),
        0i16.to_le_bytes(),
        i16::MAX.to_le_bytes(),
    ]
    .concat();
    let array = BinaryArray::from_vec(vec![bytes.as_slice()]);
    let frame = parse_rover(&params, &array).unwrap();
    assert_eq!(frame.samples[0], -1.0);
    assert_eq!(frame.samples[1], 0.0);
    assert!(frame.samples[2] < 1.0);
    assert_eq!(frame.identity.target_entity_id, "rover-a");
}

#[test]
fn rover_rejects_wrong_shape_and_payload() {
    let mut params = metadata("f32le", 2);
    params.insert("entity_id".into(), Parameter::String("rover-a".into()));
    let array = BinaryArray::from_vec(vec![&[0u8; 4][..]]);
    assert!(parse_rover(&params, &array).is_err());
    params.insert("format".into(), Parameter::String("s16le".into()));
    params.insert("sample_count".into(), Parameter::Integer(3));
    assert!(parse_rover(&params, &array).is_err());
}

#[test]
fn browser_rejects_non_finite_or_mismatched_samples() {
    let mut params = metadata("f32le", 2);
    params.insert(
        "target_entity_id".into(),
        Parameter::String("rover-a".into()),
    );
    assert!(parse_browser(&params, &Float32Array::from(vec![0.0, f32::NAN])).is_err());
    params.insert("sample_count".into(), Parameter::Integer(3));
    assert!(parse_browser(&params, &Float32Array::from(vec![0.0, 0.5])).is_err());
}
