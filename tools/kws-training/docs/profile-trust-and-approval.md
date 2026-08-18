# Wake-Word Profile Trust and Approval Contract

Status: **BLOCKED — trust design draft; security/release approval absent**
Draft date: 2026-08-18 (Asia/Ho_Chi_Minh)
Scope: exact-ID startup selection and immutable profile release.

The proposed minimum trust model separates promotion authority from payload
integrity:

```text
human Promote decision
  -> digest-bound approval attestation
  -> exact-ID deployment catalog
  -> immutable installed bundle
  -> KWS_PROFILE_ID at process startup
  -> pre-Dora validation
  -> exactly one engine
```

This is a blocked design proposal, not an approval. A manifest boolean or a
matching hash alone never proves that a profile was promoted by an authorized
owner.

## Authorities and identity

The deployment catalog is the authority for which exact profile IDs are
selectable. A bundle manifest is declarative integrity and compatibility data;
it cannot grant itself approval. The catalog publisher, approver, installer,
runtime operator, archive authority, revoker, and emergency-recovery authority
must be separate named roles with least privilege. Exact people, group names,
trust root, and archive location are **TBD [BLOCKING]**.

Every approval attestation must bind, at minimum:

```text
profile_id
release_id
composite_digest_algorithm = SHA-256 (proposal)
composite_digest
display_phrase
spoken_phrase/canonical pronunciation
threshold and cooldown
engine_contract
ORT/provider compatibility
evidence and license references
approver role/reference and approval date
decision = Promote | Reject | Revoke
```

Canonical field ordering and digest construction are owner-approved below.
Allowed characters, maximum sizes, evidence reference format, and approval
identity format remain **TBD [BLOCKING]**. Do not put keys, private identity,
raw audio, or private paths in the attestation committed to this repository.

Owner canonical-digest decision recorded for `TRUST-APP-01`:

```text
owner: loidinhm31
recorded_at: 2026-08-18 (Asia/Ho_Chi_Minh)
serialization: UTF-8 canonical JSON
keys: lexicographic
whitespace: none
hash: SHA-256
binding: profile ID; release ID; display/spoken/canonical phrase; threshold; cooldown; engine/provider; complete file inventory
file_entry: normalized relative POSIX path; byte size; file SHA-256
reject: aliases; duplicate paths; absolute paths; undeclared runtime files
decision_hash: 1451768af14484c282032115d4999c4f4bbbea483996133ffaaec722c3b74635
state: OWNER APPROVED PARTIAL — schema implementation, allowed characters/sizes, evidence references, approval identity, and test vectors remain blocking
```

Owner key-material decision recorded for `TRUST-APP-01`:

```text
owner: loidinhm31
recorded_at: 2026-08-19 (Asia/Ho_Chi_Minh)
decision: No key material
private_keys: never stored in repository attestations or evidence
public_keys: not required for the first-release digest-attestation model
pki: deferred
attestation_trust: controlled digest-approval record outside repository secrets
decision_hash: 08adbadac51117d98471bf30de3cc6717e1378c34da5e4a56e9c12111ad7fbf6
state: OWNER APPROVED PARTIAL — evidence references, approval identity, authorities, implementation, and threat-model review remain blocking
```

## Catalog, bundle, and startup validation

The catalog must map an exact `profile_id` to one immutable release, digest,
manifest, approval attestation, and installed bundle root. The runtime accepts
only the deployment-provided exact `KWS_PROFILE_ID`; production must fail for a
missing ID. Development/test defaults, if later allowed, require a separate
explicit policy and cannot influence production.

Before constructing the engine or processing Dora events, the selector must:

1. Parse the catalog and locate the exact ID; reject unknown, duplicate, or
   revoked identities.
2. Verify the attestation decision is exactly `Promote`, approver authority,
   digest binding, and evidence/license references; reject missing, `Reject`,
   and `Revoke` decisions.
3. Parse a supported manifest schema; reject unknown schema/engine/provider,
   phrase/token/threshold mismatch, missing hashes, extra undeclared required
   files, absolute paths, traversal, and incompatible ORT/provider values.
4. Resolve only regular files within the approved profile root; reject symlink
   escapes, mutable paths, missing files, and hash mismatches.
5. Construct exactly one engine and run the cheap open/contract check plus any
   approved public golden-vector smoke check.
6. Emit a structured identity/result record and only then allow Dora startup.

Any failure is non-zero and fail-closed. There is no search, guessed profile,
compiled phrase fallback, or silent rollback. The prior last-known-good profile
is selected only by an explicit catalog/ID change and restart.

Owner manifest-validation contract recorded for `TRUST-APP-03`:

```text
owner: loidinhm31
recorded_at: 2026-08-18 (Asia/Ho_Chi_Minh)
required_checks: supported schema; exact and unique profile ID; Promote decision and approver authority; composite digest; evidence/license references; phrase/token/threshold match; declared file hashes; regular files within approved root; no absolute paths; no traversal; no symlink escapes; known engine/provider; ORT compatibility; exactly one engine; approved public golden-vector smoke check
failure: non-zero before Dora startup
fallback: no search; no guessed profile; no compiled phrase fallback; no silent rollback
decision_hash: ff2d2cb336e8cd11c43f62f00b865cd218adc0a4fbd2f0b34918a7be5a046e64
state: OWNER APPROVED PARTIAL — schema/catalog implementation, provider allowlist, smoke fixture, evidence, and release approval remain blocking
```

## Publication, revocation, and rollback

Publish a new bundle to a new immutable location, compute the canonical
composite digest, obtain the bound Promote attestation, update the catalog
atomically, and restart the process. Never edit a mounted bundle in place.
Installation must stage and validate before an atomic activation; an
interrupted install must leave the previous approved catalog and bundle usable.

Revocation records must identify profile/release/digest, reason, effective
time, authority, replacement/disable action, and propagation/audit evidence.
Offline recovery must be least-privilege and must not bypass validation. The
revocation store, publisher, archive retention, recovery time objective,
disable procedure, and emergency authorization are **TBD [BLOCKING]**.

Owner operator-recovery principles recorded for `TRUST-APP-04`:

```text
owner: loidinhm31
recorded_at: 2026-08-18 (Asia/Ho_Chi_Minh)
install: stage and validate before atomic activation
last_known_good: retain verified rollback bundle
disable: auditable catalog action
rollback: explicit profile-ID change and process restart
recovery_drill: required; validation bypass prohibited
recovery_time_objective: TBD [BLOCKING]
runbook_and_emergency_authority: TBD [BLOCKING]
decision_hash: 07768023a19f802676934d18e888d4a639a092ea5d2a73209f9b5cea5239d409
state: OWNER APPROVED PARTIAL — RTO, runbook, emergency authority, implementation, evidence, and release approval remain blocking
```

Owner release-authority principles recorded for `TRUST-APP-02`:

```text
owner: loidinhm31
recorded_at: 2026-08-18 (Asia/Ho_Chi_Minh)
roles: separate named least-privilege catalog publisher; approver; installer; runtime operator; archive authority; revoker; emergency-recovery authority
release: immutable bundle; atomic activation; no in-place edit
rollback: explicit profile-ID change and process restart
revocation_record: profile; release; digest; reason; effective time; authority; replacement/disable action; propagation/audit evidence
decision_hash: 7b897e1e3899339bddadad34a86e19ecd4c49810c5fde005a203a6524123ad7f
state: OWNER APPROVED PARTIAL — named assignments, stores, retention, recovery objectives, disable procedure, implementation, and release approval remain blocking
```

The first-release trust boundary uses a digest-bound attestation without new
PKI. A security threat model must still explicitly accept this boundary; PKI
key generation, custody, rotation, verification, revocation, recovery, and
test fixtures are deferred unless a later threat-model decision requires them.

Owner trust-mechanism decision recorded for `TRUST-APP-01`:

```text
owner: loidinhm31
recorded_at: 2026-08-18 (Asia/Ho_Chi_Minh)
decision: Approve digest-bound attestation without new PKI for first release
accepted_decision: Promote only
rejected_decisions: missing; Reject; Revoke
binding: profile_id; release_id; composite SHA-256 digest; display/spoken/canonical phrase; threshold; cooldown; engine contract; ORT/provider compatibility; evidence; license references; approver role/reference; approval date
manifest_boolean_alone: insufficient
pki: deferred unless a later threat-model decision requires it
decision_hash: 796556a3badc20c05ee07b1f0060e4c1085927849026fc85a7ff1e26ffe341d6
state: OWNER APPROVED PARTIAL — named authorities, canonical digest construction, catalog implementation, threat model, and release approval remain blocking
```

## Observability and privacy

Startup, heartbeat, detection, and evidence envelopes must include
`profile_id`, `release_id`, canonical composite digest, engine contract,
provider, validation result, and rollback reason. Logs must not include raw
audio, secrets, approval keys, participant identity, or private storage paths.
The retention and access policy is governed by
`privacy-consent-retention.md` and remains **TBD [BLOCKING]**.

## Approval register

| Owner | Approval role | Decision ID | Required decision | Evidence ref | Date | Decision hash | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| loidinhm31 | Security | TRUST-APP-01 | Threat model, root/path rules, digest and optional PKI decision | Owner decision blocks above | 2026-08-18 / 2026-08-19 | `796556a3badc20c05ee07b1f0060e4c1085927849026fc85a7ff1e26ffe341d6`; `1451768af14484c282032115d4999c4f4bbbea483996133ffaaec722c3b74635`; `08adbadac51117d98471bf30de3cc6717e1378c34da5e4a56e9c12111ad7fbf6` | **PARTIAL — attestation, digest, and key-material boundary approved; evidence references, approval identity, authorities, implementation, and threat-model review blocking** |
| loidinhm31 | Release | TRUST-APP-02 | Catalog publisher, attestation workflow, revocation and archive | Owner decision block above | 2026-08-18 | `7b897e1e3899339bddadad34a86e19ecd4c49810c5fde005a203a6524123ad7f` | **PARTIAL — governance principles approved; named authorities and implementation blocking** |
| loidinhm31 | Technical | TRUST-APP-03 | Schema, engine/ORT allowlist, canonicalization and startup failure | Owner decision block above | 2026-08-18 | `ff2d2cb336e8cd11c43f62f00b865cd218adc0a4fbd2f0b34918a7be5a046e64` | **PARTIAL — validation contract approved; implementation and fixtures blocking** |
| loidinhm31 | Operator | TRUST-APP-04 | Install, disable, last-known-good retention and recovery drill | Owner decision block above | 2026-08-18 | `07768023a19f802676934d18e888d4a639a092ea5d2a73209f9b5cea5239d409` | **PARTIAL — principles approved; RTO and runbook implementation blocking** |

Phase 1 cannot freeze trust or authorize Phase 2 until every row has an
accepted/revised/rejected decision, owner, date, and canonical decision hash.
