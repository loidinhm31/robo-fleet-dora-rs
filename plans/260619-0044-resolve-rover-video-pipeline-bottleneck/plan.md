---
title: "Resolve Rover Video Pipeline Bottleneck"
description: "Replace raw cross-machine video with bounded rover-side JPEG and validate the pipeline on the constrained x86_64 workstation before later rover hardware testing."
status: complete
priority: P1
effort: 44h
branch: main
tags: [bugfix, backend, frontend, critical, performance]
created: 2026-06-19
---

# Resolve Rover Video Pipeline Bottleneck

## Overview

Remove the deterministic 221 Mbps raw RGB transport bottleneck without coupling operator video to local ML or visual servo. Develop the refactor on the current Fedora x86_64 workstation using the USB UVC camera, direct Dora smoke runs, and split rover/orchestra containers. Deliver one rover/one viewer at 640x480 JPEG quality 80 and 15 FPS target.

Closure note 2026-06-25: this plan is closed on a headless release-validation gate. Browser-render, long camera stability, and constrained field-soak certification are explicitly deferred because the available execution environment cannot provide a stable browser session or stable camera setup.

## Fixed Decisions

- JPEG only. H.264, HLS, WebRTC deferred.
- Coordinated rover/orchestra/UI cutover. No permanent raw fallback.
- Direct Dora mode is the fast functional loop; split local dataflows are the transport acceptance topology.
- Blocking workstation resource envelope: rover container limited to 3 CPU equivalents and 4 GiB RAM.
- Results certify this host and resource envelope only. Raspberry Pi and other edge-device performance remain future validation.
- Primary integration input is the KYE PC-LM1E UVC camera. Repeatable ML comparisons use a real captured frame corpus with a checksum manifest.
- Servo contract: >=10 Hz, input age p95 <=150 ms.
- Viewer contract: average >=14.5 FPS, capture-to-display p95 <=500 ms. Deferred to field/browser certification after headless closure.
- Resource contract: no OOM, peak RSS <=3.5 GiB, and no sustained CPU saturation above 90% of the 3-CPU quota. Deferred to constrained field certification after headless closure.
- Failed headless milestone stops release closure. Deferred field milestones require separate certification before production claims.

## Phases

| # | Phase | Status | Effort | Gate |
|---|---|---|---:|---|
| 1 | [Measurement contract and baseline](./phase-01-measurement-contract-and-baseline.md) | Complete | 6h | Trustworthy correlated metrics |
| 1.1 | [Repository formatting baseline](./phase-01-1-repository-formatting-baseline.md) | Complete | 2h | Isolated format-only diff; CI guard enabled |
| 2 | [Rover JPEG and Zenoh cutover](./phase-02-rover-jpeg-and-zenoh-cutover.md) | Complete 2026-06-24 | 10h | Raw transport removed; bandwidth/CPU pass |
| 3 | [Binary browser delivery and demand control](./phase-03-binary-browser-delivery-and-demand-control.md) | Complete 2026-06-24 | 8h | Binary payload; zero work without viewer |
| 4 | [Latest-frame ML isolation](./phase-04-latest-frame-ml-isolation.md) | Complete 2026-06-24 | 12h | Servo freshness and bounded queue pass |
| 5 | [Final validation and release](./phase-05-final-validation-and-release.md) | Complete headless 2026-06-25 | 6h | Headless release gate pass; field certification deferred |

## Dependencies

- Fedora workstation with Ryzen 7 8840U, 23 GiB RAM, Podman Docker compatibility, and the KYE PC-LM1E UVC camera exposed through V4L2.
- Camera device access from the execution environment; prefer a stable `/dev/v4l/by-id/` path and resolve `/dev/video0` only as fallback.
- UI repository: `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app`.
- Existing YOLO and ReID models under `models/.cache/` and a resolvable repository-local or container ONNX Runtime library.
- Preserve existing dirty changes in both repositories.

## Out of Scope

- Multi-rover/multi-viewer capacity.
- Audio transport changes.
- General performance-dashboard redesign.
- Raspberry Pi 5, ARM64 runtime, thermal, and production-network certification.
- H.264 feasibility or implementation; JPEG is fixed for this refactor.
- ML algorithm/cadence changes without a failed, diagnosed Phase 4 gate.

## Validation Summary

**Validated:** 2026-06-19
**Questions asked:** 8 decision topics
### Confirmed Decisions

- Run native direct Dora smoke tests and split local rover/orchestra acceptance tests.
- Limit the rover acceptance container to 3 CPU equivalents and 4 GiB RAM.
- Make workstation results final only for this host; defer edge-device claims.
- Use the physical UVC camera plus a repeatable real-frame corpus.
- Include the adjacent UI repository in the coordinated cutover.
- Keep JPEG quality 80 at 15 FPS; abandon H.264 for this plan.

### Required Plan Revisions

- [x] Replace Raspberry Pi-only dependencies and acceptance language.
- [x] Replace invalid `/home/raspb4` paths with current workspace paths or environment variables.
- [x] Add native and constrained-container execution profiles.
- [x] Add deterministic corpus and host-resource evidence requirements.

## Unresolved Questions

- None before implementation. Gate failures create evidence-based replanning; they do not authorize weakening acceptance criteria.
