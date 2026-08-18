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
it must match lowercase ASCII `[a-z0-9][a-z0-9._-]{0,127}` with no whitespace
or normalization, and catalog lookup is byte-exact. Restart is the switch
boundary. Phrase, tokenizer/pronunciation, threshold,
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
| PROD-01 | Startup semantics | Exactly one `KWS_PROFILE_ID`; restart to switch; no hot reload; invalid or unapproved identity fails before Dora | **OWNER APPROVED — implementation and release gates remain blocking** |
| PROD-02 | Primary/fallback phrase | Primary display `Hey E.C`; spoken/canonical pronunciation `Hey Ee Cee`; fallback remains intentionally pending as a separately approved later candidate and never an automatic runtime fallback | **PARTIAL — primary recorded; fallback intentionally pending** |
| PROD-03 | Candidate eligibility | Require 2–4 spoken words, stable pronunciation, distinctive phonemes, low ordinary-speech collision, no safety/control vocabulary overlap, and no common Rover TTS/command collision | **OWNER APPROVED — acoustic and evidence gates remain blocking** |
| PROD-04 | Baseline profile | Logical ID `sherpa-hey-kv-v1` mapped to the current Sherpa last-known-good bundle; release ID and digest remain required | **PARTIAL — logical ID recorded; release ID/digest and approval blocking** |
| PROD-05 | Production identity | Explicit exact `KWS_PROFILE_ID`; no unset-ID or phrase fallback | **OWNER APPROVED — catalog, implementation, evidence, and release gates remain blocking** |
| PROD-06 | Engine contract | Sherpa Zipformer transducer/BPE rollback baseline is approved; successor engine and ORT/provider allowlist remain required | **PARTIAL — successor engine, provider evidence, and release approval blocking** |
| PROD-07 | Population | Universal/multi-speaker target; English; broad intended accent coverage without a claim until the exact matrix and exclusions are approved | **PARTIAL — matrix and exclusions remain product/privacy/data blockers** |
| PROD-08 | Out-of-scope behavior | Defer live reload, multi-profile inference, personalization, arbitrary phrase generation, browser selection, ARM acceptance, and fleet rollout | **OWNER APPROVED — release implementation and approval gates remain blocking** |

## Phrase and baseline record

The following fields are intentionally empty until the product/operator/data
owners approve them. The record must contain values, not just a display label,
before Phase 2 begins:

```text
primary.display_phrase: Hey E.C
primary.spoken_phrase: Hey Ee Cee
primary.canonical_pronunciation: Hey Ee Cee
primary.language_and_accent_scope: English; broad target-user accents; exact matrix and exclusions TBD [BLOCKING]
primary.population: Universal/multi-speaker
fallback.display_phrase: TBD [BLOCKING]
fallback.spoken_phrase: TBD [BLOCKING]
baseline.profile_id: sherpa-hey-kv-v1
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

Owner accent input recorded for `PROD-APP-01`:

```text
owner: loidinhm31
recorded_at: 2026-08-18 (Asia/Ho_Chi_Minh)
accent_coverage: Broad target-user English accents
minimum_accent_matrix: TBD [BLOCKING]
exclusions: TBD [BLOCKING]
decision: Keep accent matrix pending
decision_hash: d994953475a14bf2bad216bbf38eea6652ed794b543a9214275b2c24f3f325b0
state: PARTIAL — broad coverage recorded; matrix and exclusions remain blocking
```

Owner accent-matrix decision recorded for `PROD-07`:

```text
owner: loidinhm31
recorded_at: 2026-08-18 (Asia/Ho_Chi_Minh)
decision: Keep accent matrix pending
coverage_claim: broad intended coverage only; no tested accent coverage claimed
decision_hash: 7f919a399d19fc9c3abb59f935d756b7d507ca8f0bdd1a6d5d17125fa4a0919f
state: BLOCKING — exact included accent groups and exclusions are not approved
```

Owner fallback decision recorded for `PROD-02`:

```text
owner: loidinhm31
recorded_at: 2026-08-18 (Asia/Ho_Chi_Minh)
decision: Keep fallback pending
decision_hash: 370b490640774e79be658eeabf8236aada8dfcc58724a70a52b663669775b44e
state: BLOCKING — no fallback candidate selected; no runtime fallback permitted
```

Owner candidate-eligibility input recorded for `PROD-03`:

```text
owner: loidinhm31
recorded_at: 2026-08-18 (Asia/Ho_Chi_Minh)
spoken_word_count: 2-4
pronunciation: stable canonical pronunciation required
phonemes: distinctive sequence required
ordinary_speech_collision: low collision required
safety_and_control_vocabulary: no overlap permitted
rover_tts_and_command_collision: no common collision permitted
synthetic_screening: ranking only
target_speaker_evidence: required for promotion
decision_hash: b470e8da7b9f545c21e6143368bc177287e26f613f84454eae41aa0ba64d0cb3
state: OWNER APPROVED — acoustic feasibility, target-speaker evaluation, and release approval remain blocking
```

Owner release-scope input recorded for `PROD-08`:

```text
owner: loidinhm31
recorded_at: 2026-08-18 (Asia/Ho_Chi_Minh)
in_scope: one profile per voice-wake process; restart-only switching; linux/amd64 and x86_64 target
deferred: live reload; multi-profile inference; personalization; arbitrary phrase generation; browser selection; ARM acceptance; fleet rollout
decision_hash: d55bbdaad736211358ac77d04216527214505f18fb87a7084f64527c32c77db1
state: OWNER APPROVED — implementation, target evidence, and release approval remain blocking
```

Owner rollback-engine input recorded for `PROD-06`:

```text
owner: loidinhm31
recorded_at: 2026-08-18 (Asia/Ho_Chi_Minh)
engine: sherpa-onnx 1.13.3 with static feature
model: Zipformer transducer with BPE tokens
input: 16-kHz mono F32
inference_threads: 1
bundle: sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01
bundle_archive_sha256: f170013b4716e41b62b9bfd809687c207cef798ef9bc6534d524e17af9b6561a
repository_ort_version: 1.16.3
successor_engine: TBD [BLOCKING]
provider_allowlist: TBD [BLOCKING]
decision_hash: 6857b7dac18bbefd6329381e31fcf4688e7c694a3af146606528f98969ad22a5
state: OWNER APPROVED — successor engine, provider/ORT feasibility evidence, and release approval remain blocking
```

Owner baseline identity input recorded for `PROD-APP-02`:

```text
owner: loidinhm31
recorded_at: 2026-08-18 (Asia/Ho_Chi_Minh)
profile_id: sherpa-hey-kv-v1
engine: Sherpa-ONNX Zipformer GigaSpeech 3.3M
bundle_path: models/.cache/sherpa-onnx/kws/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01
display_phrase: Hey K.V
spoken_phrase: Hey Kay Vee
required_files: encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx; decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx; joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx; tokens.txt; bpe.model
sample_rate_hz: 16000
release_id: TBD [BLOCKING]
composite_digest: TBD [BLOCKING]
decision_hash: 0591809d47077f7844981915a020c3c2a82c08415cbc52634c742139b0526029
state: PARTIAL — logical ID and current last-known-good mapping recorded; release ID, payload digest, target identity, and operator approval remain blocking
```

Owner startup policy input recorded for `PROD-01` and `PROD-05`:

```text
owner: loidinhm31
recorded_at: 2026-08-18 (Asia/Ho_Chi_Minh)
decision: Approve fail-closed exact-profile startup policy
profile_selector: KWS_PROFILE_ID (exactly one value per voice-wake process)
switch_boundary: process restart only
pre_dora_failure_cases: unset; empty; unknown; unapproved; rejected; revoked; malformed; corrupt; hash-mismatched; path-escaping; symlinked; incompatible
automatic_fallback: prohibited, including the pending fallback phrase
decision_hash: 4effe4fd34b7c1c16d771bc6af1aa41b5e6e49a070f6756a63f7467be514ba6e
state: OWNER APPROVED — catalog state, implementation, evidence, and release approval remain blocking
```

Owner profile-ID canonicalization input recorded for `PROD-05`:

```text
owner: loidinhm31
recorded_at: 2026-08-18 (Asia/Ho_Chi_Minh)
grammar: [a-z0-9][a-z0-9._-]{0,127}
normalization: none
whitespace: reject
unicode_and_uppercase: reject
catalog_lookup: byte-exact
decision_hash: abbee064f84b55793f780702c157d39ad08a8417fe95c0901925ebf9439aa5a0
state: OWNER APPROVED — catalog implementation, evidence, and release approval remain blocking
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

All numbers below are **OWNER-APPROVED WORKING ACCEPTANCE TARGETS**, not
results. A revised value must replace the target and retain owner/date/
decision-hash evidence.

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

Owner SLO-target input recorded for `PROD-APP-01`:

```text
owner: loidinhm31
recorded_at: 2026-08-18 (Asia/Ho_Chi_Minh)
speakers: at least 30 consented speakers
positives: 10-20 per speaker across sessions
partitions: speaker-disjoint train/dev/test
overall_recall: at least 95%
required_subgroup_recall: at least 90% for every required subgroup
false_accepts: one-sided 95% upper bound <= 0.05 FA/h
planning_exposure: initial clean-negative floor >= 100 h
self_trigger: zero WakeAck/TTS self-triggers
continuous_soak: at least 24 h on exact target
latency: warm p99 <= 100 ms; phrase-end-to-detection p95 <= 600 ms; phrase-end-to-NormalRover-Ready p95 < 5 s; Ready-to-WakeAck-start p95 < 1.5 s; phrase-end-to-WakeAck-start p95 < 6.5 s
decision_hash: a98f145fa623c5b9c49746ee9caa47d2707434c86d8c4b2f04c8dff5c3c1dcac
state: OWNER APPROVED WORKING GATE — formulas, subgroup definitions, exclusions, evidence, and release approval remain blocking
```

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

Owner feasibility-host input recorded for `PROD-APP-02`:

```text
owner: loidinhm31
recorded_at: 2026-08-18 (Asia/Ho_Chi_Minh)
scope: feasibility-only; not production target approval
host: linux
architecture: x86_64
kernel: 7.1.7-200.fc44.x86_64
os: Fedora Linux 44 Server Edition
cpu: AMD Ryzen 7 8840U with Radeon 780M Graphics; 16 logical CPUs
ram: 23 GiB visible system memory
enumerated_capture_devices: HD-Audio Generic / ALC245 Analog; PC-LM1E Camera / USB Audio
ort_version: 1.16.3
selected_capture_device: TBD [BLOCKING]
gain_agc_noise_processing: TBD [BLOCKING]
container_or_image: TBD [BLOCKING]
ort_provider: TBD [BLOCKING]
power_and_thermal_envelope: TBD [BLOCKING]
decision_hash: 0ae79d9dfd3ea8a6f9f15fa6156599fff9e2a3c5e6e3d087cdfeaa7148950a9c
state: PARTIAL — host metadata recorded for feasibility; exact rover target and capture/runtime controls remain blocking
```

Owner detector-capture boundary input recorded for `PROD-APP-02`:

```text
owner: loidinhm31
recorded_at: 2026-08-18 (Asia/Ho_Chi_Minh)
detector_input: 16-kHz mono F32
resampling: TBD [BLOCKING]
framing: TBD [BLOCKING]
window_and_hop: TBD [BLOCKING]
vad_and_noise_suppression: TBD [BLOCKING]
queue_and_drop_behavior: TBD [BLOCKING]
clock_source: TBD [BLOCKING]
decision_hash: 46577ce643db2e1eeb2d9db4018ec27ab2778178d8bd5c0b42d00cf7e4a437fa
state: OWNER APPROVED PARTIAL — only the detector boundary is frozen; capture pipeline details and target evidence remain blocking
```

The owner-approved detector boundary is 16-kHz mono F32. Any resampling, framing,
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
| loidinhm31 | Operator | PROD-APP-02 | Target identity, baseline, measurement and rollback procedure | Feasibility-host, baseline, and capture-boundary decision blocks above | 2026-08-18 | `0591809d47077f7844981915a020c3c2a82c08415cbc52634c742139b0526029`; `0ae79d9dfd3ea8a6f9f15fa6156599fff9e2a3c5e6e3d087cdfeaa7148950a9c`; `46577ce643db2e1eeb2d9db4018ec27ab2778178d8bd5c0b42d00cf7e4a437fa` | **PARTIAL — baseline, feasibility host, and detector boundary recorded; exact target and capture details blocking** |
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
