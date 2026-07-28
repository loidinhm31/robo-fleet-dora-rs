#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="$root_dir/test-data/power/noise-corpus-manifest.json"
output=""

usage() {
  cat <<'EOF'
Usage: KWS_TRIAL_COMMAND='...' scripts/benchmark-rover-kws.sh --output FILE [--manifest FILE]

The manifest must contain target-local sample paths and SHA-256 values. The
trial command receives sample path and sample ID, then writes one JSON object
to stdout: {"accepted":bool,"latency_ms":number}. Negative samples must use
expected_accepted:false. Empty template manifests are intentionally rejected.
EOF
}

while (($#)); do
  case "$1" in
    --output) output="${2:?--output needs a path}"; shift 2 ;;
    --manifest) manifest="${2:?--manifest needs a path}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$output" ]] || { usage >&2; exit 2; }
[[ "$(uname -m)" =~ ^(aarch64|armv7l|armv8l)$ ]] || { echo "physical KWS evidence requires an ARM Rover; this host is $(uname -m)" >&2; exit 2; }
: "${KWS_TRIAL_COMMAND:?set a target-local KWS trial command}"
: "${POWER_HARDWARE_ID:?set a stable Rover hardware identifier}"
: "${POWER_ROVER_ID:?set the exact Rover entity ID}"
: "${POWER_TOPOLOGY:?set direct or split topology}"
: "${POWER_CONFIG_FILE:?set the effective target configuration file}"
[[ "$POWER_TOPOLOGY" == direct || "$POWER_TOPOLOGY" == split ]] || { echo "POWER_TOPOLOGY must be direct or split" >&2; exit 2; }
[[ -r "$POWER_CONFIG_FILE" ]] || { echo "POWER_CONFIG_FILE is unreadable" >&2; exit 2; }

mkdir -p "$(dirname "$output")"

results="$(mktemp)"
sample_definitions="${results}.samples"
trap 'rm -f "$results" "$sample_definitions"' EXIT
python3 - "$manifest" <<'PY' > "$sample_definitions"
import hashlib, json, pathlib, sys
with open(sys.argv[1], encoding="utf-8") as source: document = json.load(source)
if document.get("schema_version") != 1 or document.get("keyword") != "Hey Kiwi": raise SystemExit("invalid KWS manifest")
if not isinstance(document.get("model_checksum"), str) or len(document["model_checksum"]) != 64 or any(value not in "0123456789abcdef" for value in document["model_checksum"].lower()): raise SystemExit("model_checksum must be SHA-256")
samples = document.get("samples")
if not samples: raise SystemExit("KWS manifest has no target-local samples")
classes = set(document.get("required_classes", []))
if not classes: raise SystemExit("KWS manifest has no required_classes")
minimum = document.get("minimum_trials_per_class")
if not isinstance(minimum, int) or minimum < 1: raise SystemExit("invalid minimum_trials_per_class")
ids, paths = set(), set()
for sample in samples:
    for key in ("id", "path", "sha256", "expected_accepted", "noise_class"):
        if key not in sample: raise SystemExit(f"sample missing {key}")
    if not isinstance(sample["id"], str) or not sample["id"].replace("-", "").isalnum() or sample["id"] in ids: raise SystemExit("sample ID is invalid or duplicate")
    if not isinstance(sample["expected_accepted"], bool) or not isinstance(sample["noise_class"], str) or sample["noise_class"] not in classes: raise SystemExit("sample class or expected_accepted is invalid")
    if not isinstance(sample["sha256"], str) or len(sample["sha256"]) != 64: raise SystemExit("sample SHA-256 is invalid")
    path = pathlib.Path(sample["path"])
    if str(path) in paths: raise SystemExit("sample path is duplicate")
    if not path.is_file(): raise SystemExit(f"missing sample: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sample["sha256"]: raise SystemExit(f"checksum mismatch: {path}")
    ids.add(sample["id"]); paths.add(str(path))
    print(json.dumps(sample, separators=(",", ":")))
PY

while IFS= read -r sample; do
  path="$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["path"])' <<< "$sample")"
  sample_id="$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["id"])' <<< "$sample")"
  result="$(bash -c "$KWS_TRIAL_COMMAND" -- "$path" "$sample_id")"
  python3 - "$sample" "$result" >> "$results" <<'PY'
import json, sys
sample, result = map(json.loads, sys.argv[1:])
if not isinstance(result.get("accepted"), bool) or not isinstance(result.get("latency_ms"), (int, float)) or result["latency_ms"] < 0:
    raise SystemExit("trial result must contain accepted bool and non-negative latency_ms")
result["sample"] = sample
print(json.dumps(result, separators=(",", ":")))
PY
done < "$sample_definitions"

python3 - "$results" "$output" "$(git -C "$root_dir" rev-parse HEAD)" "$manifest" "$(sha256sum "$POWER_CONFIG_FILE" | awk '{print $1}')" <<'PY'
import json, os, sys
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
with open(sys.argv[4], encoding="utf-8") as source: manifest = json.load(source)
classes = set(manifest["required_classes"])
counts = {noise_class: 0 for noise_class in classes}
for row in rows:
    noise_class = row["sample"]["noise_class"]
    if noise_class not in counts: raise SystemExit(f"unexpected noise class: {noise_class}")
    counts[noise_class] += 1
if any(count < manifest["minimum_trials_per_class"] for count in counts.values()):
    raise SystemExit("insufficient trials for one or more required noise classes")
latencies = sorted(row["latency_ms"] for row in rows if row["accepted"] and row["sample"]["expected_accepted"])
if not latencies: raise SystemExit("no accepted KWS trials")
def percentile(values, value): return values[max(0, min(len(values) - 1, round((len(values) - 1) * value)))]
false_accepts = sum(row["accepted"] and not row["sample"]["expected_accepted"] for row in rows)
false_rejects = sum(not row["accepted"] and row["sample"]["expected_accepted"] for row in rows)
with open(sys.argv[2], "w", encoding="utf-8") as destination:
    context = {key.lower(): os.environ[key] for key in ("POWER_HARDWARE_ID", "POWER_ROVER_ID", "POWER_TOPOLOGY")}
    json.dump({"schema_version": 1, "kind": "physical-rover-kws", "outcome": "measured", "git_sha": sys.argv[3], "host_arch": os.uname().machine, "config_sha256": sys.argv[5], "model_checksum": manifest["model_checksum"], "manifest": sys.argv[4], "context": context, "wake_ack_p95_ms": percentile(latencies, .95), "false_accepts": false_accepts, "false_rejects": false_rejects, "trial_counts": counts, "trials": rows}, destination, indent=2)
    destination.write("\n")
PY
