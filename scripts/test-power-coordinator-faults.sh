#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="$root_dir/test-data/power/fault-scenarios.json"
only=""
with_mongo=false
docker_smoke=false
workstation_health=false
stack_smoke=false
operator_topology=false

usage() {
  cat <<'EOF'
Usage: scripts/test-power-coordinator-faults.sh [options]

Run the declarative automated fault matrix. Operator topology cases are never
reported as passed unless POWER_TOPOLOGY_FAULT_COMMAND is explicitly supplied.

Options:
  --only ID          Run one scenario from fault-scenarios.json
  --with-mongo       Run the opt-in Mongo projection integration gate
  --docker-smoke     Run Docker/Podman info, hello-world, and compose preflight
  --workstation-health Check a running workstation stack's health and processes
  --stack-smoke      Build, start, check, and clean up an exclusive workstation stack
  --operator-topology Run operator scenarios through POWER_TOPOLOGY_FAULT_COMMAND
  --validate         Validate the manifest only
  --list             List scenarios
EOF
}

validate_manifest() {
  python3 - "$manifest" <<'PY'
import json, sys
path = sys.argv[1]
with open(path, encoding="utf-8") as source:
    document = json.load(source)
if document.get("schema_version") != 1 or not isinstance(document.get("scenarios"), list):
    raise SystemExit("invalid fault scenario schema")
seen = set()
for scenario in document["scenarios"]:
    scenario_id = scenario.get("id")
    if not isinstance(scenario_id, str) or not scenario_id or scenario_id in seen:
        raise SystemExit("scenario IDs must be unique non-empty strings")
    if scenario.get("mode") not in {"automated", "operator"}:
        raise SystemExit(f"{scenario_id}: unsupported mode")
    if not isinstance(scenario.get("covers"), list) or not scenario["covers"]:
        raise SystemExit(f"{scenario_id}: covers must be a non-empty list")
    seen.add(scenario_id)
PY
}

scenario_mode() {
  python3 - "$manifest" "$1" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as source:
    scenarios = json.load(source)["scenarios"]
for scenario in scenarios:
    if scenario["id"] == sys.argv[2]:
        print(scenario["mode"])
        break
else:
    raise SystemExit(f"unknown scenario: {sys.argv[2]}")
PY
}

list_scenarios() {
  python3 - "$manifest" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as source:
    for scenario in json.load(source)["scenarios"]:
        print(f"{scenario['id']} ({scenario['mode']}): {', '.join(scenario['covers'])}")
PY
}

run_scenario() {
  local scenario="$1"
  case "$scenario" in
    contracts-and-authority) cargo test -p robo_rover_lib --lib power_contract_tests ;;
    coordinator-lifecycle)
      cargo test -p power_coordinator --lib
      cargo test -p power_coordinator --bin power_coordinator
      ;;
    journal-recovery) cargo test -p power_coordinator --test journal-recovery ;;
    projector-outage)
      cargo test -p power_event_projector --test mongo-integration projector_reports_a_bounded_failure_when_mongo_is_unavailable
      if "$with_mongo"; then
        make test-power-projector-mongo
        SCHEDULER_TEST_MONGODB_URI="${SCHEDULER_TEST_MONGODB_URI:-mongodb://127.0.0.1:${MONGODB_PORT:-27017}}" \
          cargo test -p recording_scheduler --test mongo-integration
      fi
      ;;
    reservation-mutation)
      cargo test -p recording_scheduler --test reservation_faults
      cargo test -p recording_scheduler --test runtime-reconciliation
      ;;
    wake-safety) cargo test -p voice_wake ;;
    split-topology-restart|direct-topology-restart)
      if ! "$operator_topology"; then
        echo "$scenario requires --operator-topology and POWER_TOPOLOGY_FAULT_COMMAND" >&2
        return 3
      fi
      : "${POWER_TOPOLOGY_FAULT_COMMAND:?set POWER_TOPOLOGY_FAULT_COMMAND for operator topology scenarios}"
      : "${POWER_TOPOLOGY_EVIDENCE_FILE:?set POWER_TOPOLOGY_EVIDENCE_FILE for operator topology scenarios}"
      bash -c "$POWER_TOPOLOGY_FAULT_COMMAND" -- "$scenario"
      python3 - "$POWER_TOPOLOGY_EVIDENCE_FILE" "$scenario" "$(git -C "$root_dir" rev-parse HEAD)" <<'PY'
import json, time, sys
with open(sys.argv[1], encoding="utf-8") as source: evidence = json.load(source)
expected_topology = "split" if sys.argv[2].startswith("split-") else "direct"
if evidence.get("schema_version") != 1 or evidence.get("scenario") != sys.argv[2] or evidence.get("outcome") != "pass" or evidence.get("topology") != expected_topology or evidence.get("git_sha") != sys.argv[3]:
    raise SystemExit("topology evidence must bind the scenario and a pass outcome")
for key in ("git_sha", "topology", "hardware_id", "started_at_ms", "finished_at_ms", "faults"):
    if not evidence.get(key): raise SystemExit(f"topology evidence missing {key}")
if not all(isinstance(evidence[key], int) for key in ("started_at_ms", "finished_at_ms")) or not (evidence["started_at_ms"] <= evidence["finished_at_ms"] <= int(time.time() * 1000)):
    raise SystemExit("topology timestamps are invalid")
if not isinstance(evidence["faults"], list) or not evidence["faults"]:
    raise SystemExit("topology evidence needs one or more fault assertions")
for fault in evidence["faults"]:
    if not isinstance(fault, dict) or not all(isinstance(fault.get(key), str) and fault[key] for key in ("action", "assertion", "log")):
        raise SystemExit("each topology fault needs action, assertion, and log")
PY
      ;;
    *) echo "no test command for $scenario" >&2; return 2 ;;
  esac
}

workstation_health_ready() {
  local container health
  for container in robo-orchestra robo-rover-kiwi; do
    docker ps --filter "name=^/${container}$" --format '{{.Names}}' | grep -qx "$container"
    health="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$container")"
    [[ "$health" == healthy ]] || return 1
    docker top "$container" >/dev/null
  done
}

run_workstation_health() {
  export XDG_RUNTIME_DIR="/run/user/$(id -u)"
  if workstation_health_ready; then return 0; fi
  docker logs --tail 200 robo-orchestra >&2 || true
  docker logs --tail 200 robo-rover-kiwi >&2 || true
  echo "workstation containers are not healthy" >&2
  return 1
}

run_stack_smoke() {
  local compose=(docker compose -f "$root_dir/docker/docker-compose.yml" -f "$root_dir/docker/docker-compose.workstation.yml" --profile mongodb --profile orchestra --profile rover-kiwi)
  export XDG_RUNTIME_DIR="/run/user/$(id -u)"
  for container in robo-mongodb robo-orchestra robo-rover-kiwi; do
    if docker ps -a --filter "name=^/${container}$" --format '{{.Names}}' | grep -qx "$container"; then
      echo "stack smoke requires no existing $container container" >&2
      return 2
    fi
  done
  : "${HOST_RECORDING_PATH:?set a dedicated existing directory below /home}"
  : "${POWER_PROTECTED_WORK_HMAC_KEY:?set a test-only 32-byte key}"
  : "${POWER_PROTECTED_WORK_HMAC_KEYS:?set a test-only entity-to-key map}"
  : "${POWER_COMMAND_HMAC_KEY:?set a test-only 32-byte key}"
  : "${POWER_COMMAND_HMAC_KEYS:?set a test-only entity-to-key map}"
  "${compose[@]}" up -d --build
  trap '"${compose[@]}" down --remove-orphans' RETURN
  for _ in $(seq 1 36); do
    if workstation_health_ready; then break; fi
    sleep 5
  done
  run_workstation_health
  "${compose[@]}" logs --tail 200 orchestra rover-kiwi
  "${compose[@]}" down --remove-orphans
  trap - RETURN
}

run_docker_smoke() {
  local recording_dir
  export XDG_RUNTIME_DIR="/run/user/$(id -u)"
  docker info
  docker run --rm hello-world
  recording_dir="$(mktemp -d)"
  trap 'rmdir "$recording_dir"' RETURN
  HOST_RECORDING_PATH="$recording_dir" \
  POWER_PROTECTED_WORK_HMAC_KEY="01234567890123456789012345678901" \
  POWER_PROTECTED_WORK_HMAC_KEYS='{"rover-kiwi":"01234567890123456789012345678901"}' \
  POWER_COMMAND_HMAC_KEY="01234567890123456789012345678901" \
  POWER_COMMAND_HMAC_KEYS='{"rover-kiwi":"01234567890123456789012345678901"}' \
  docker compose -f "$root_dir/docker/docker-compose.yml" -f "$root_dir/docker/docker-compose.workstation.yml" \
    --profile mongodb --profile orchestra --profile rover-kiwi config >/dev/null
  rmdir "$recording_dir"
  trap - RETURN
}

while (($#)); do
  case "$1" in
    --only) only="${2:?--only needs a scenario ID}"; shift 2 ;;
    --with-mongo) with_mongo=true; shift ;;
    --docker-smoke) docker_smoke=true; shift ;;
    --workstation-health) workstation_health=true; shift ;;
    --stack-smoke) stack_smoke=true; shift ;;
    --operator-topology) operator_topology=true; shift ;;
    --validate) validate_manifest; exit 0 ;;
    --list) validate_manifest; list_scenarios; exit 0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

validate_manifest
if "$docker_smoke"; then run_docker_smoke; fi
if "$workstation_health"; then run_workstation_health; fi
if "$stack_smoke"; then run_stack_smoke; fi

if [[ -n "$only" ]]; then
  scenario_mode "$only" >/dev/null
  run_scenario "$only"
  exit 0
fi

while IFS= read -r scenario; do
  if [[ "$(scenario_mode "$scenario")" == automated || "$operator_topology" == true ]]; then
    run_scenario "$scenario"
  else
    echo "SKIP $scenario: requires target/operator topology evidence" >&2
  fi
done < <(python3 - "$manifest" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as source:
    print(*[item["id"] for item in json.load(source)["scenarios"]], sep="\n")
PY
)
