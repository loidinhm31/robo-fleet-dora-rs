# Wake-Word Training and Release License Inventory

Status: **BLOCKED — inventory draft; legal/release approval absent**
Draft date: 2026-08-18 (Asia/Ho_Chi_Minh)
Scope: proposed KWS training, evaluation, and immutable runtime bundle.

This is an inventory schema and initial repository evidence, not a license
grant or redistribution approval. No participant data, model payload, private
key, hosted-training output, or production bundle is included. Every component
must be reviewed before it enters a corpus, trainer, image, or release bundle.

## Required evidence per component

Each row must identify the exact name, version/commit, source URL or repository
reference, SPDX identifier and license text, copyright/notice obligations,
modifications, model/data terms, commercial use, redistribution, export or
processor restrictions, package inclusion, reviewer, date, and decision hash.
Unknown, transitive, generated, or downloaded content is a release blocker.

Owner license-inventory gate recorded for `LIC-APP-01`:

```text
owner: loidinhm31
recorded_at: 2026-08-18 (Asia/Ho_Chi_Minh)
required_evidence: exact name; version/commit; source; SPDX identifier; license text; notices; modifications; model/data terms; commercial use; redistribution; export/processor restrictions; package inclusion
unknown_transitive_generated_downloaded_content: release blocker
aggregate_third_party_row: insufficient
decision_hash: 7c9d837587c90bbc82fc7eaf3234eda8ae9951d8db0618e561651a7efd8ded93
state: OWNER APPROVED GATE — every component remains blocked until its evidence is complete and legally reviewed
```

## Initial inventory

| Component | Version/source evidence | License/rights | Intended use | Status |
| --- | --- | --- | --- | --- |
| Sherpa-ONNX Rust/runtime baseline | Cargo lock package `sherpa-onnx` 1.13.3; source and native build provenance required | **TBD — legal review** | Explicit named last-known-good detector | **BLOCKING** |
| ONNX Runtime shared library | Existing local runtime directory identifies 1.16.3 and contains `LICENSE`, `Privacy.md`, and `ThirdPartyNotices.txt` | **TBD — verify exact binary provenance and redistribution terms** | Approved engine/provider runtime if compatibility is accepted | **BLOCKING** |
| Baseline KWS model payload | Exact profile/release/digest not selected; no new payload added | **TBD — model card, source, training and redistribution terms** | Rollback profile only after approval | **BLOCKING** |
| Candidate training engine/export | Engine and pinned commit not selected; no download permitted in Phase 1 | **TBD — trainer, dependencies, export/runtime terms** | Offline feasibility and later training | **BLOCKING** |
| Tokenizer/BPE/phoneme resources | Phrase and tokenizer contract not frozen | **TBD — source and model/data license** | Phrase-specific bundle | **BLOCKING** |
| Pretrained weights/backbone | No candidate or source approved | **TBD — weights, model card, commercial and redistribution terms** | Candidate training only if approved | **BLOCKING** |
| Human corpus | No collection authorized; no participant data in repository | **TBD — consent, jurisdiction, ownership and withdrawal impact** | Train/dev/test and blind evaluation | **BLOCKING** |
| Synthetic generator and clips | Generator/style/license not selected; no clips created | **TBD — generator and generated-output rights** | Augmentation and lexical bake-off | **BLOCKING** |
| Negative/noise sources | Exact sources and terms not frozen | **TBD — source, recording, broadcast and redistribution terms** | False-accept exposure and hard negatives | **BLOCKING** |
| Golden vectors/reference scorer | Not generated; format and provenance not frozen | **TBD — derived-artifact and source terms** | Python/Rust parity and smoke checks | **BLOCKING** |
| Rust transitive dependencies | Cargo.lock exists; full license report not generated | **TBD — exact package/version inventory** | Build/runtime support | **BLOCKING** |
| Container/base image | Release image and digest not selected | **TBD — base image and bundled notices** | Target packaging | **BLOCKING** |

The local ORT notices are evidence to inspect, not approval. The existing
repository model notices do not establish rights for a future KWS payload.

## Review and publication controls

Legal/release must approve the complete transitive dependency and model/data
inventory before any profile is marked promotable. The release bundle must
carry or reference the required notices, exact versions/commits, provenance,
composite digest, and allowed redistribution scope. If a term is incompatible
with commercial distribution, the component is excluded or a written exception
is required; never hide it in an aggregate “third-party” row.

Training and evaluation assets remain offline and outside Git/Docker contexts
unless their owners explicitly approve sanitized, redistributable fixtures.
Hosted training or an external data processor is prohibited without written
privacy, legal, and security approval.

Owner processing-boundary decision recorded for `LIC-APP-02`:

```text
owner: loidinhm31
recorded_at: 2026-08-18 (Asia/Ho_Chi_Minh)
synthetic_data: generator, voice/style licensing, prompt/seed policy, generated-data license, and provenance approval required
human_consent: synthetic clips never represent human consent or replace the target-rover human gate
hosted_or_external_processor: prohibited unless written privacy, legal, and security approval covers transfer and retention
decision_hash: 4f1ea90a9bf196ea7f8bc7f24bb8ae686565ada96155583ca1cb62b11d7ae18f
state: OWNER APPROVED PARTIAL — exact generator, terms, jurisdictions, transfer, retention, and processor evidence remain blocking
```

## Approval register

| Owner | Approval role | Decision ID | Required decision | Evidence ref | Date | Decision hash | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| loidinhm31 | Legal | LIC-APP-01 | Component/data/model terms and commercial redistribution | Owner decision block above | 2026-08-18 | `7c9d837587c90bbc82fc7eaf3234eda8ae9951d8db0618e561651a7efd8ded93` | **PARTIAL — gate approved; component rights and legal review blocking** |
| loidinhm31 | Privacy | LIC-APP-02 | Human/synthetic data processing and withdrawal impact | Owner decision block above | 2026-08-18 | `4f1ea90a9bf196ea7f8bc7f24bb8ae686565ada96155583ca1cb62b11d7ae18f` | **PARTIAL — processing boundary approved; data rights and terms blocking** |
| loidinhm31 | Security | LIC-APP-03 | Provenance, downloads, processor and supply-chain controls | TBD | TBD | Not assigned | **BLOCKING** |
| loidinhm31 | Release | LIC-APP-04 | Notices, bundle contents, archive and publication scope | TBD | TBD | Not assigned | **BLOCKING** |

## Blocking checklist

- Exact engine/export and pinned source: **TBD [BLOCKING]**.
- Complete transitive dependency license report: **TBD [BLOCKING]**.
- Model/pretrained/tokenizer/noise provenance and rights: **TBD [BLOCKING]**.
- Human/synthetic corpus terms and withdrawal consequences: **TBD [BLOCKING]**.
- Container, ORT binary, notices, and commercial redistribution: **TBD [BLOCKING]**.

No license inventory is signed and no later phase is authorized until each row
has exact evidence, owner, accepted/revised/rejected decision, date, and
canonical decision hash.
