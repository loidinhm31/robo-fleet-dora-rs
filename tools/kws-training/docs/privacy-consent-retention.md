# Wake-Word Privacy, Consent, Retention, and License Policy

Status: **BLOCKED — policy draft; privacy/legal approval absent**
Draft date: 2026-08-18 (Asia/Ho_Chi_Minh)
Scope: Phase 1 contract only; no participant data is present here.

This document defines the minimum policy to approve before recording, corpus
collection, training, or redistribution. It is not consent language and does
not authorize collection. Every proposed control remains blocking until the
privacy, legal, security, and release owners sign the decision record.

## Collection boundary

No recording, corpus freeze, hosted trainer, or production model download is
authorized by this draft. The first release proposes a universal detector from
consented speakers, not personalization, voice identification, biometric
verification, or arbitrary user phrase generation.

Permitted data classes, only after approval:

- Consent metadata with a pseudonymous participant ID and collection/session
  metadata needed for split integrity and acoustic condition analysis.
- Audio clips and derived features required for the approved training/evaluation
  purpose, kept outside Git, Docker build contexts, logs, and MongoDB.
- Sanitized aggregate metrics, hashes, provenance, and adjudication decisions
  without raw audio or direct participant identity.

Owner data-boundary decision recorded for `PRIV-APP-01`:

```text
owner: loidinhm31
recorded_at: 2026-08-18 (Asia/Ho_Chi_Minh)
decision: Approve minimum data-minimization boundary before audio collection
allowed_data: pseudonymous consent metadata; audio and derived features only for the approved training/evaluation purpose
storage_exclusions: Git; Docker build contexts; logs; MongoDB; developer-local copies by default
repository_evidence: sanitized aggregates; hashes; provenance; adjudication decisions only
prohibited_purpose: personalization; voice identification; biometric verification; arbitrary user phrase generation
hosted_or_external_processor: written privacy/legal and security approval required
decision_hash: 43247047d39d8bd9fd7047e2e708cbf0bb6a6554bf8098a37fddf8c8530d3294
state: OWNER APPROVED PARTIAL — consent language, lawful basis, jurisdictions, storage controls, retention, withdrawal, licensing, and release approval remain blocking
```

The exact data fields, lawful basis, jurisdictions, age/ability safeguards,
cross-border transfer rule, and whether any derived feature is biometric data
are **TBD [BLOCKING]**.

## Consent and withdrawal

Consent must be informed, specific to wake-word training/evaluation, voluntary,
recorded before capture, revocable, and separate from employment or product
access. The consent flow must state purpose, data classes, retention, access,
processors, jurisdictions, publication/redistribution, risks, and contact for
withdrawal. Participants must be able to withdraw without penalty.

Withdrawal must stop future use, identify all approved copies and derivatives,
remove or quarantine raw audio and directly identifying metadata within the
approved objective, and record the resulting dataset/model impact. The exact
withdrawal SLA, re-training trigger, exception handling, and audit evidence are
**TBD [BLOCKING]**.

Owner consent-and-withdrawal principles recorded for `PRIV-APP-01`:

```text
owner: loidinhm31
recorded_at: 2026-08-18 (Asia/Ho_Chi_Minh)
consent: specific to wake-word training/evaluation; recorded before capture; voluntary; revocable; separate from employment and product access
consent_notice: purpose; data classes; retention; access; processors; jurisdictions; publication/redistribution; risks; withdrawal contact
withdrawal: stop future use; cover approved copies and derivatives; remove or quarantine raw audio and directly identifying metadata; record dataset/model impact
legal_wording_and_sla: TBD [BLOCKING]
decision_hash: 30afa716344e057ddd015c3693c4663989986e396a3c185bef057a18c930e9e3
state: OWNER APPROVED PARTIAL — exact consent language, withdrawal SLA, retraining trigger, exception handling, audit evidence, and legal approval remain blocking
```

## Storage, access, and blind custody

Raw audio and blind labels must use an approved encrypted store with least
privilege, separate identity mapping, access logging, and no developer-local
copy by default. Encryption at rest/in transit, key custody, backup treatment,
region, access review cadence, and deletion verification are **TBD [BLOCKING]**.

Owner storage-and-blind-custody controls recorded for `PRIV-APP-03`:

```text
owner: loidinhm31
recorded_at: 2026-08-18 (Asia/Ho_Chi_Minh)
storage: encrypted at rest and in transit
access: least privilege with access logging
identity_mapping: separate from audio and labels
blind_labels: independent custodian or equivalent access boundary
developer_local_copy: prohibited by default
repository: raw audio and blind labels prohibited
region: TBD [BLOCKING]
key_custody: TBD [BLOCKING]
backup_treatment: TBD [BLOCKING]
access_review_cadence: TBD [BLOCKING]
deletion_verification: TBD [BLOCKING]
decision_hash: 1aba8f1e4a2ce5b5fd7b544228fd92511574f463cf2c63f19a2b1fe0a3bec7e6
state: OWNER APPROVED PARTIAL — region, key custody, backups, review cadence, deletion verification, and security approval remain blocking
```

The test partition and blind labels must be held by an independent custodian or
equivalent access boundary. Training owners must not access blind labels before
the one-use evaluation. Evidence committed to this repository may contain only
sanitized aggregates, stable pseudonymous IDs where strictly necessary, and
content hashes; never names, contact data, private storage paths, raw samples,
credentials, or keys.

Retention is **TBD [BLOCKING]** for raw audio, consent metadata, derived
features, labels, model weights, evaluation reports, access logs, backups, and
withdrawal records. Each class needs a duration, deletion owner, legal hold
rule, and verifiable deletion method.

Owner retention decision recorded for `PRIV-APP-01`:

```text
owner: loidinhm31
recorded_at: 2026-08-18 (Asia/Ho_Chi_Minh)
decision: Keep retention pending
classes: raw audio; consent metadata; derived features; labels; model weights; evaluation reports; access logs; backups; withdrawal records
required_before_collection: duration; deletion owner; legal-hold rule; verifiable deletion method for every class
collection_status: not authorized by this decision
decision_hash: 7b0a2cf9c614423b339b2656f0702b5ff05287f5759da089aae92d4c6507b0d8
state: BLOCKING — no retention schedule or deletion policy approved
```

## Synthetic sources, training code, and redistribution

Synthetic clips may be used only when their generator, voice/style licensing,
prompt or seed policy, generated-data license, and provenance are approved.
Synthetic data must not be represented as human consent or substitute for the
target-rover human gate.

Training code, dependencies, pretrained components, model weights, tokenizer,
corpus, augmentations, golden vectors, and output bundles each require a
license inventory with SPDX identifier, version/commit, source, obligations,
notice requirements, redistribution scope, and commercial-use status. A hosted
trainer or external processor is prohibited without written privacy/legal and
security approval that covers data transfer and retention.

Owner processing-boundary decision recorded for `PRIV-APP-02`:

```text
owner: loidinhm31
recorded_at: 2026-08-18 (Asia/Ho_Chi_Minh)
synthetic_data: generator, voice/style licensing, prompt/seed policy, generated-data license, and provenance approval required
human_consent: synthetic clips never represent human consent or replace the target-rover human gate
hosted_or_external_processor: prohibited unless written privacy, legal, and security approval covers transfer and retention
decision_hash: 4e93088f6fb8fdc66004f043ae8adc6056928851da7f9f57d6a9dfe35e0e7503
state: OWNER APPROVED PARTIAL — exact generator, terms, jurisdictions, transfer, retention, and processor evidence remain blocking
```

The exact jurisdictions, commercial release rights, notices, and whether any
component has copyleft or attribution obligations are **TBD [BLOCKING]**.

## Approval register

| Owner | Approval role | Decision ID | Required decision | Evidence ref | Date | Decision hash | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| loidinhm31 | Privacy/data owner | PRIV-APP-01 | Data classes, lawful basis, minimization and participant rights | Owner decision blocks above | 2026-08-18 | `43247047d39d8bd9fd7047e2e708cbf0bb6a6554bf8098a37fddf8c8530d3294`; `30afa716344e057ddd015c3693c4663989986e396a3c185bef057a18c930e9e3`; `7b0a2cf9c614423b339b2656f0702b5ff05287f5759da089aae92d4c6507b0d8` | **PARTIAL — boundary and consent principles approved; retention, rights, and legal details blocking** |
| loidinhm31 | Legal/licensing owner | PRIV-APP-02 | Jurisdictions, processor terms, licenses and redistribution | Owner decision block above | 2026-08-18 | `4e93088f6fb8fdc66004f043ae8adc6056928851da7f9f57d6a9dfe35e0e7503` | **PARTIAL — processing boundary approved; legal and processor details blocking** |
| loidinhm31 | Security owner | PRIV-APP-03 | Encryption, key custody, access, blind custody and audit | Owner decision block above | 2026-08-18 | `1aba8f1e4a2ce5b5fd7b544228fd92511574f463cf2c63f19a2b1fe0a3bec7e6` | **PARTIAL — controls approved; deployment details and security approval blocking** |
| loidinhm31 | Release owner | PRIV-APP-04 | Artifact publication, notices, deletion impact and archive | TBD | TBD | Not assigned | **BLOCKING** |

Approval references must be opaque, least-privilege, and free of private keys
or participant identity. This repository records policy decisions and hashes,
not the underlying sensitive evidence.

## Blocking checklist

- Consent text and withdrawal workflow: **TBD [BLOCKING]**.
- Retention/deletion windows and legal holds: **TBD [BLOCKING]**.
- Storage region, encryption, key/access control, and blind custodian: **TBD [BLOCKING]**.
- Synthetic, pretrained, code, weights, corpus, and output licenses: **TBD [BLOCKING]**.
- Hosted training and redistribution decision: **TBD [BLOCKING]**.

No collection or later phase is authorized until all rows have an owner,
accepted/revised/rejected decision, date, and canonical decision hash.
