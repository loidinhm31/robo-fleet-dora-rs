#!/usr/bin/env bash

set -euo pipefail

usage() {
  printf 'Usage: %s ATTESTATION_JSON EVIDENCE_REGISTRY_JSON\n' "$0" >&2
}

fail() {
  printf 'FAIL: %s\n' "$1" >&2
  exit 1
}

if [[ $# -ne 2 ]]; then
  usage
  exit 2
fi

if ! command -v jq >/dev/null 2>&1; then
  fail 'jq is required'
fi

attestation_file=$1
evidence_registry_file=$2
expected_approver='loidinhm31'

[[ -r "$attestation_file" ]] || fail "attestation is not readable: $attestation_file"
[[ -r "$evidence_registry_file" ]] || fail "evidence registry is not readable: $evidence_registry_file"

jq -e 'type == "object"' "$attestation_file" >/dev/null \
  || fail 'attestation must be a JSON object'
jq -e 'type == "object"' "$evidence_registry_file" >/dev/null \
  || fail 'evidence registry must be a JSON object'

jq -e '.decision == "Promote"' "$attestation_file" >/dev/null \
  || fail 'decision must be exactly Promote'

jq -e --arg expected "$expected_approver" \
  '.approver_identity == $expected' "$attestation_file" >/dev/null \
  || fail "approver_identity must be exactly $expected_approver"

if jq -e '
  has("approver_role") or
  has("approval_role") or
  has("approver_subject_ref") or
  has("subject_ref") or
  has("role")
' "$attestation_file" >/dev/null; then
  fail 'role and subject-reference fields are not allowed'
fi

jq -e '
  (.profile_id | type == "string") and
  (.profile_id | test("^[a-z0-9][a-z0-9._-]{0,127}$"))
' "$attestation_file" >/dev/null \
  || fail 'profile_id must match the approved lowercase ASCII format'

jq -e '.release_id | type == "string" and length > 0' "$attestation_file" >/dev/null \
  || fail 'release_id must be a non-empty string'

jq -e '.composite_digest | type == "string" and test("^[0-9a-f]{64}$")' \
  "$attestation_file" >/dev/null \
  || fail 'composite_digest must be a lowercase 64-character SHA-256 hex digest'

jq -e '
  (.evidence_refs | type == "array") and
  (.evidence_refs | length > 0) and
  (.evidence_refs as $refs |
    (all($refs[];
      if type != "string" or length == 0 then
        false
      elif contains("/") or contains("\\") or contains("://") then
        false
      else
        true
      end
    )) and
    (($refs | unique | length) == ($refs | length))
  )
' "$attestation_file" >/dev/null \
  || fail 'evidence_refs must be a non-empty, unique array of opaque references'

if jq -e '
  any(.. | objects | keys_unsorted[]?;
    test("private[_-]?key|public[_-]?key|secret|api[_-]?key|access[_-]?token|raw[_-]?audio|participant([_-](id|identity|name|ref))?|private[_-]?path|storage[_-]?path"; "i")
  )
' "$attestation_file" >/dev/null; then
  fail 'attestation contains a forbidden key-material, participant, or private-path field'
fi

if jq -e '
  any(.. | strings;
    test("BEGIN [A-Z ]*PRIVATE KEY"; "i") or
    contains("file://") or
    contains("/home/") or
    contains("/mnt/") or
    contains("/tmp/") or
    contains("/var/")
  )
' "$attestation_file" >/dev/null; then
  fail 'attestation contains key material or an obvious private path'
fi

mapfile -t evidence_refs < <(jq -r '.evidence_refs[]' "$attestation_file")

for ref in "${evidence_refs[@]}"; do
  jq -e --arg ref "$ref" \
    '(.[$ref] | type == "object" and .available == true)' \
    "$evidence_registry_file" >/dev/null \
    || fail "evidence reference is missing or unavailable: $ref"
done

profile_id=$(jq -r '.profile_id' "$attestation_file")
printf 'PASS: approver=%s profile_id=%s evidence_refs=%d\n' \
  "$expected_approver" "$profile_id" "${#evidence_refs[@]}"
printf 'Advisory only: this checks the named approver and registry availability; it does not authorize runtime activation or verify the canonical digest.\n'
