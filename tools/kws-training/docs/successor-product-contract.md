# Dynamic Rover Wake-Word Successor Product Contract

Status: **BLOCKED — working draft; no human approval recorded**
Draft date: 2026-08-18 (Asia/Ho_Chi_Minh)
Plan: `260813-0906-dynamic-wake-word-profiles`, Phase 1

This record freezes the questions that must be answered before collection,
training, provisioning, or runtime selector work. It contains no participant
data, credentials, private paths, audio, model payload, or approval secret.
Research proposals are labeled as proposals and are not release authority.

## Product and runtime boundary

The first release proposes exactly one approved wake-word profile per
`voice-wake` process. `KWS_PROFILE_ID` is the deployment-owned exact identity;
restart is the switch boundary. Phrase, tokenizer/pronunciation, threshold,
engine contract, and payload digest are immutable profile fields. A “fallback
phrase” is only a separately approved candidate for a later explicit profile
selection; it is never an automatic runtime fallback. There is no live reload,
simultaneous multi-word inference, browser selection, arbitrary path/URL,
personalization, or fleet-wide rollout in this phase.

Startup must validate the catalog entry, attestation, manifest, compatibility,
and every declared file before Dora event processing. Missing, empty, unknown,
unapproved, malformed, path-escaping, symlinked, corrupt, hash-mismatched, or
incompatible inputs fail non-zero before Dora starts. Production has no unset-ID
default and no silent fallback. The current Sherpa detector remains an explicit
named last-known-good rollback profile.

## Decision register

| ID | Decision | Working proposal | Status |
| --- | --- | --- | --- |
| PROD-01 | Startup semantics | One profile; restart to switch; no hot reload | **USER MUST APPROVE — blocking** |
| PROD-02 | Primary/fallback phrase | Primary display `Hey E.C`; spoken/canonical pronunciation `Hey Ee Cee`; fallback remains a separately approved later candidate and never an automatic runtime fallback | **PARTIAL — phrase recorded; scope/evidence approval blocking** |
| PROD-03 | Candidate eligibility | Prefer a short 2–4 word phrase with distinctive phonemes, stable pronunciation, low ordinary-use collision, and no safety/control vocabulary overlap | **USER MUST APPROVE — blocking** |
| PROD-04 | Baseline profile | Exact named Sherpa profile ID, release ID, and digest | **TBD — operator/release approval blocking** |
| PROD-05 | Production identity | Explicit `KWS_PROFILE_ID`; no unset-ID fallback | **USER MUST APPROVE — blocking** |
| PROD-06 | Engine contract | Sherpa Zipformer transducer/BPE baseline and one approved candidate engine contract; ORT/provider compatibility is allowlisted | **TBD — technical feasibility and release approval blocking** |
| PROD-07 | Population | Universal/multi-speaker target; language, accent, hearing/speech population, and exclusions | **TBD — product/privacy/data approval blocking** |
| PROD-08 | Out-of-scope behavior | Defer live reload, multi-profile inference, personalization, arbitrary phrase generation, browser selection, ARM acceptance, and fleet rollout | **USER MUST APPROVE — blocking** |

## Phrase and baseline record

The following fields are intentionally empty until the product/operator/data
owners approve them. The record must contain values, not just a display label,
before Phase 2 begins:

```text
primary.display_phrase: Hey E.C
primary.spoken_phrase: Hey Ee Cee
primary.canonical_pronunciation: Hey Ee Cee
primary.language_and_accent_scope: English; exact accent coverage and exclusions TBD [BLOCKING]
primary.population: Universal/multi-speaker
fallback.display_phrase: TBD [BLOCKING]
fallback.spoken_phrase: TBD [BLOCKING]
baseline.profile_id: TBD [BLOCKING]
baseline.release_id: TBD [BLOCKING]
baseline.composite_digest: TBD [BLOCKING]
```

Owner input recorded for `PROD-APP-01`:

```text
owner: loidinhm31
recorded_at: 2026-08-18 (Asia/Ho_Chi_Minh)
decision_hash: e0f2805543d2957d52a7d9e87868d32888d1910ac4a8158dcbca0509dce77920
state: PARTIAL — phrase fields only; language/accent scope, fallback, feasibility, and product approval remain blocking
audio_evidence: not required for this label; required later for acoustic feasibility and target evaluation
```

Owner scope input recorded for `PROD-APP-01`:

```text
owner: loidinhm31
recorded_at: 2026-08-18 (Asia/Ho_Chi_Minh)
language: English
population: Universal/multi-speaker
accent_coverage: TBD [BLOCKING]
exclusions: TBD [BLOCKING]
decision_hash: 14f2ffd4d81d78ada235dfce51d8c36551e26ee6cf9f7b7e761e9634329eb88a
state: PARTIAL — language/population recorded; accent coverage and exclusions remain blocking
```

The fallback fields describe a later explicit selection candidate only. Missing,
unknown, rejected, or corrupt `KWS_PROFILE_ID` still fails closed; it never
causes the fallback phrase to load.

Eligibility screening must reject common command words, easily truncated or
ambiguous spellings, unsafe collisions, and phrases that occur in Rover TTS or
normal command speech. Synthetic screening is a ranking gate only; target
hardware human evidence is required for promotion. A phrase or threshold change
creates a new immutable profile and repeats the relevant evidence gates.

## Proposed SLOs and preregistration

All numbers below are **PROPOSED — USER MUST APPROVE**. A revised value must
replace the proposal and retain owner/date/decision-hash evidence.

| Measure | Proposed target | Formula/clock that must be frozen |
| --- | --- | --- |
| Speakers and positives | At least 30 consented speakers; 10–20 positives per speaker across sessions | Speaker-disjoint train/dev/test partitions; exact counts and exclusions |
| Overall recall | At least 95% | Accepted eligible positives / eligible positives; confidence method and CI required |
| Required subgroup recall | At least 90% for every required subgroup | Same denominator rule per subgroup; minimum subgroup sample size required |
| False accepts | One-sided 95% upper bound ≤0.05 FA/h | Count one event after the preregistered cooldown; `H` clean-negative hours; Poisson upper rate `χ²(0.95, 2(F+1)) / (2H)` |
| Planning exposure | Initial clean-negative planning floor ≥100 h | The approved statistical bound, not this floor, is authoritative; exclusions and source mix required |
| Self-trigger | Zero WakeAck/TTS self-triggers | Playback start, tail, and suppression-window event semantics required |
| Continuous soak | At least 24 h on the exact target | Valid only with complete identity, drops/gaps, exposure, health, and thermal records |
| Warm decision | p99 ≤100 ms per inference window | Clock starts at accepted frame and ends at decision; warm-up policy required |
| Phrase-end to detection | p95 ≤600 ms | Phrase-end anchor and accepted-detection anchor must be observable |
| Phrase-end to `NormalRover` Ready | p95 <5 s | Readiness transition and timeout/error semantics required |
| Ready to WakeAck start | p95 <1.5 s | Readiness and audio-start clocks from the same monotonic source |
| Phrase-end to WakeAck start | p95 <6.5 s | Composite target; retries, cancellation, and invalid samples excluded by rule |

Threshold, cooldown, confidence-bound method, subgroup definitions, clean-hour
exclusions, soak validity, canary exposure, disable bound, and rollback recovery
time are **TBD [BLOCKING]** until the statistician/technical, operator, product,
and release owners sign them.

## Target and measurement contract

The release target is exact `linux/amd64`/x86_64 hardware. ARM/Raspberry Pi is
outside this release. The following identity is required in evidence and is
currently **TBD [BLOCKING]**: host/model, CPU, RAM, OS/kernel, container/image,
microphone/interface, gain, AGC/noise processing, ORT build/provider, and power
or thermal envelope.

The proposed capture contract is 16-kHz mono F32. Any resampling, framing,
window/hop, VAD/noise suppression, queue/drop behavior, and clock source must be
recorded before collection. Resource limits for CPU, RSS, peak RSS, temperature,
throttling, and power are **TBD [BLOCKING]**. `phrase_end`, `detection`,
`NormalRover Ready`, and `WakeAck start` must use documented monotonic anchors;
wall-clock timestamps are evidence metadata only.

## Safety, rollback, and release gates

The selector must preserve sole microphone ownership, `IdleListening` gating,
playback plus tail suppression, bounded wake demand, readiness-gated WakeAck,
and zero command/actuator/media edges. A candidate is never accepted merely
because a manifest says `approved: true` or because hashes match.

Publish a new immutable bundle, retain the verified last-known-good bundle, and
switch only by changing the exact deployment ID and restarting. The retention
window, offline archive authority, disable procedure, canary exposure, rollback
recovery objective, and emergency recovery authority are **TBD [BLOCKING]**.

## Approval register

| Owner | Approval role | Decision ID | Required decision | Evidence ref | Date | Decision hash | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| loidinhm31 | Product | PROD-APP-01 | Phrase, population, deferrals, numeric SLOs | TBD | TBD | Not assigned | **BLOCKING** |
| loidinhm31 | Operator | PROD-APP-02 | Target identity, baseline, measurement and rollback procedure | TBD | TBD | Not assigned | **BLOCKING** |
| loidinhm31 | Statistician/technical | PROD-APP-03 | Formulas, cooldown, exclusions, subgroup and soak rules | TBD | TBD | Not assigned | **BLOCKING** |
| loidinhm31 | Privacy/legal | PROD-APP-04 | Consent, retention, jurisdictions, licenses and redistribution | TBD | TBD | Not assigned | **BLOCKING** |
| loidinhm31 | Security/release | PROD-APP-05 | Catalog authority, attestation, revocation and recovery | TBD | TBD | Not assigned | **BLOCKING** |

Hashes are for canonical decision records after approval; do not store private
keys or participant identities in this repository.

## Phase 1 feasibility status

Only public fixtures may be used. No approved public KWS fixture/export harness
was found in the existing tracked training tools during this draft; no corpus,
model download, or production artifact was created. Engine/ORT feasibility is
**TBD [BLOCKING]** and must record the exact public fixture, engine/export,
ORT version/provider, host, command, result, and report hash.

Phase 1 remains open until every blocking row has an owner, accepted/revised/
rejected decision, date, and canonical decision hash. Do not update the
superseded plan relationship or begin Phase 2 while any blocker remains.
