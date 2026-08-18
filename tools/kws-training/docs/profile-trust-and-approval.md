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

Canonical field ordering, digest construction, allowed characters, maximum
sizes, evidence reference format, and approval identity format are **TBD
[BLOCKING]**. Do not put keys, private identity, raw audio, or private paths in
the attestation committed to this repository.

## Catalog, bundle, and startup validation

The catalog must map an exact `profile_id` to one immutable release, digest,
manifest, approval attestation, and installed bundle root. The runtime accepts
only the deployment-provided exact `KWS_PROFILE_ID`; production must fail for a
missing ID. Development/test defaults, if later allowed, require a separate
explicit policy and cannot influence production.

Before constructing the engine or processing Dora events, the selector must:

1. Parse the catalog and locate the exact ID; reject unknown, duplicate, or
   revoked identities.
2. Verify the attestation decision, approver authority, digest binding, and
   evidence/license references.
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

The preferred first-release proposal is a digest-bound attestation without new
PKI. A security threat model must explicitly accept this boundary; otherwise
PKI key generation, custody, rotation, verification, revocation, recovery, and
test fixtures become mandatory. The PKI decision is **TBD [BLOCKING]**.

## Observability and privacy

Startup, heartbeat, detection, and evidence envelopes must include
`profile_id`, `release_id`, digest (or approved digest alias), engine contract,
provider, validation result, and rollback reason. Logs must not include raw
audio, secrets, approval keys, participant identity, or private storage paths.
The retention and access policy is governed by
`privacy-consent-retention.md` and remains **TBD [BLOCKING]**.

## Approval register

| Owner | Approval role | Decision ID | Required decision | Evidence ref | Date | Decision hash | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| loidinhm31 | Security | TRUST-APP-01 | Threat model, root/path rules, digest and optional PKI decision | TBD | TBD | Not assigned | **BLOCKING** |
| loidinhm31 | Release | TRUST-APP-02 | Catalog publisher, attestation workflow, revocation and archive | TBD | TBD | Not assigned | **BLOCKING** |
| loidinhm31 | Technical | TRUST-APP-03 | Schema, engine/ORT allowlist, canonicalization and startup failure | TBD | TBD | Not assigned | **BLOCKING** |
| loidinhm31 | Operator | TRUST-APP-04 | Install, disable, last-known-good retention and recovery drill | TBD | TBD | Not assigned | **BLOCKING** |

Phase 1 cannot freeze trust or authorize Phase 2 until every row has an
accepted/revised/rejected decision, owner, date, and canonical decision hash.
