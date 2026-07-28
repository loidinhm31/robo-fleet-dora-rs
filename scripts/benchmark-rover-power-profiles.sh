#!/usr/bin/env bash
set -euo pipefail

profiles=(Awake NormalRover IdleListening ScheduledCapture Dormant)
iterations=30
output=""

usage() {
  cat <<'EOF'
Usage: POWER_PROFILE_COMMAND='...' scripts/benchmark-rover-power-profiles.sh --output FILE [options]

Runs the supplied target-local profile transition command once for every
profile/iteration. The command receives profile and iteration as its final two
arguments. It must not be a synthetic sleep: it should wait for the measured
authoritative Ready/Idle status before returning.

Set POWER_PROFILE_PIDS to the comma-separated workload PIDs whose CPU/RSS are
measured. Optionally set POWER_POWER_PROXY_COMMAND to print one numeric power
proxy sample after each transition.

Options: --output FILE  --iterations N  --profiles csv
EOF
}

while (($#)); do
  case "$1" in
    --output) output="${2:?--output needs a path}"; shift 2 ;;
    --iterations) iterations="${2:?--iterations needs a value}"; shift 2 ;;
    --profiles) IFS=, read -r -a profiles <<< "${2:?--profiles needs values}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$output" ]] || { usage >&2; exit 2; }
[[ "$iterations" =~ ^[1-9][0-9]*$ ]] || { echo "iterations must be positive" >&2; exit 2; }
[[ "$(uname -m)" =~ ^(aarch64|armv7l|armv8l)$ ]] || { echo "physical power evidence requires an ARM Rover; this host is $(uname -m)" >&2; exit 2; }
: "${POWER_PROFILE_COMMAND:?set a target-local command that waits for authoritative profile status}"
: "${POWER_PROFILE_PIDS:?set comma-separated target workload PIDs for CPU/RSS evidence}"
: "${POWER_HARDWARE_ID:?set a stable Rover hardware identifier}"
: "${POWER_ROVER_ID:?set the exact Rover entity ID}"
: "${POWER_TOPOLOGY:?set direct or split topology}"
: "${POWER_MODEL_CHECKSUM:?set the SHA-256 of the active KWS model bundle}"
: "${POWER_CONFIG_FILE:?set the effective target configuration file}"
[[ "$POWER_TOPOLOGY" == direct || "$POWER_TOPOLOGY" == split ]] || { echo "POWER_TOPOLOGY must be direct or split" >&2; exit 2; }
[[ "$POWER_MODEL_CHECKSUM" =~ ^[[:xdigit:]]{64}$ && -r "$POWER_CONFIG_FILE" ]] || { echo "model checksum or effective config is invalid" >&2; exit 2; }

mkdir -p "$(dirname "$output")"
samples="$(mktemp)"
trap 'rm -f "$samples"' EXIT
process_metrics() {
  local pid stat rss_pages page_kib executable start_time
  page_kib="$(( $(getconf PAGESIZE) / 1024 ))"
  IFS=, read -r -a pids <<< "$POWER_PROFILE_PIDS"
  for pid in "${pids[@]}"; do
    [[ "$pid" =~ ^[1-9][0-9]*$ && -r "/proc/$pid/stat" && -r "/proc/$pid/statm" ]] || { echo "missing configured PID $pid" >&2; return 1; }
    stat="$(awk '{print $14 + $15}' "/proc/$pid/stat")"
    rss_pages="$(awk '{print $2}' "/proc/$pid/statm")"
    start_time="$(awk '{print $22}' "/proc/$pid/stat")"
    executable="$(readlink "/proc/$pid/exe")"
    printf '%s:%s:%s:%s:%s;' "$pid" "$stat" "$((rss_pages * page_kib))" "$start_time" "$executable"
  done
}

for profile in "${profiles[@]}"; do
  for ((iteration = 1; iteration <= iterations; iteration++)); do
    before_metrics="$(process_metrics)"
    started_ns="$(date +%s%N)"
    transition="$(bash -c "$POWER_PROFILE_COMMAND" -- "$profile" "$iteration")"
    python3 - "$transition" "$profile" <<'PY'
import json, sys
result = json.loads(sys.argv[1])
if result.get("profile") != sys.argv[2] or result.get("terminal_state") not in {"Ready", "IdleListening", "Dormant"} or not isinstance(result.get("transition_id"), str):
    raise SystemExit("profile adapter must return matching profile, terminal_state, and transition_id")
PY
    elapsed_ms=$(( ($(date +%s%N) - started_ns) / 1000000 ))
    after_metrics="$(process_metrics)"
    temperature=""
    for sensor in /sys/class/thermal/thermal_zone*/temp; do
      [[ -r "$sensor" ]] || continue
      temperature="$(awk '{printf "%.3f", $1 / 1000}' "$sensor")"
      break
    done
    power_proxy=""
    if [[ -n "${POWER_POWER_PROXY_COMMAND:-}" ]]; then power_proxy="$(bash -lc "$POWER_POWER_PROXY_COMMAND")"; fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$profile" "$iteration" "$elapsed_ms" "$temperature" "$before_metrics" "$after_metrics" "$power_proxy" "$transition" >> "$samples"
  done
done

python3 - "$samples" "$output" "$(git -C "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" rev-parse HEAD)" "${POWER_PROFILE_PIDS:-}" "$(sha256sum "$POWER_CONFIG_FILE" | awk '{print $1}')" <<'PY'
import csv, json, os, statistics, sys
samples, output, revision, pids, config_sha = sys.argv[1:]
rows = []
def processes(value):
    result = {}
    for item in value.rstrip(";").split(";"):
        if item:
            pid, ticks, rss_kib, start_time, executable = item.split(":", 4)
            result[pid] = {"cpu_ticks": int(ticks), "rss_kib": int(rss_kib), "start_time": int(start_time), "executable": executable}
    return result
with open(samples, newline="", encoding="utf-8") as source:
    for profile, iteration, elapsed, thermal, before, after, power_proxy, transition in csv.reader(source, delimiter="\t"):
        before, after = processes(before), processes(after)
        elapsed_seconds = max(int(elapsed) / 1000, .001)
        delta_ticks = sum(max(0, after[pid]["cpu_ticks"] - item["cpu_ticks"]) for pid, item in before.items() if pid in after)
        rows.append({"profile": profile, "iteration": int(iteration), "latency_ms": int(elapsed), "thermal_c": float(thermal) if thermal else None, "cpu_percent": delta_ticks / os.sysconf("SC_CLK_TCK") / elapsed_seconds * 100, "rss_kib": sum(item["rss_kib"] for item in after.values()), "power_proxy": float(power_proxy) if power_proxy else None, "transition": json.loads(transition), "processes": after})
groups = {}
for row in rows: groups.setdefault(row["profile"], []).append(row["latency_ms"])
def percentile(values, value):
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, round((len(ordered) - 1) * value)))]
metrics = {profile: {"samples": len(values), "p50_ms": percentile(values, .50), "p95_ms": percentile(values, .95), "p99_ms": percentile(values, .99)} for profile, values in groups.items()}
with open(output, "w", encoding="utf-8") as destination:
    context = {key.lower(): os.environ[key] for key in ("POWER_HARDWARE_ID", "POWER_ROVER_ID", "POWER_TOPOLOGY", "POWER_MODEL_CHECKSUM")}
    json.dump({"schema_version": 1, "kind": "physical-rover-power-profile", "outcome": "measured", "git_sha": revision, "host_arch": os.uname().machine, "config_sha256": config_sha, "context": context, "profile_metrics": metrics, "samples": rows, "observed_pids": [value for value in pids.split(",") if value]}, destination, indent=2)
    destination.write("\n")
PY
