---
title: "Web Bridge Authentication Improvements"
description: "Harden web_bridge auth: MongoDB users, JWT sessions, bcrypt hashing, security headers, rate limiting"
status: in_progress
priority: P1
effort: 8h
branch: main
tags: [security, auth, web-bridge, jwt, bcrypt, mongodb]
created: 2026-03-26
---

# Web Bridge Authentication Improvements

## Current State

The web_bridge has fundamental auth weaknesses:
- Plain-text password comparison at line 267 of `main.rs` (timing-attack vulnerable)
- Default credentials `admin/password` only warn, never block
- `bcrypt = "0.15"` and `jsonwebtoken = "9.2"` declared in `Cargo.toml` but **never used**
- No session tokens — a connected socket stays authenticated forever
- No session expiry or idle timeout
- `tracking_command`, `camera_control`, `audio_control` exempt from command rate limiting
- No security headers (X-Frame-Options, HSTS, etc.)
- Failed auth logs the username (leaks valid usernames to log aggregators)
- Client stores credentials in `localStorage`, has no token handling, no auth error feedback

## Goals

1. Replace env-var credentials with MongoDB-backed user accounts (`qm_hub.robo_control_user`)
2. Use bcrypt for password hashing, JWT for session tokens (both deps already in Cargo.toml)
3. Issue 1-hour JWT tokens; validate on command events; proactive client refresh
4. Extend rate limiting to all events; add idle-timeout and per-IP limiting
5. Update the React client to store/refresh the JWT and surface auth errors

## Phases

| # | Phase | Effort | Status |
|---|-------|--------|--------|
| 01 | [MongoDB + JWT + Bcrypt Core (Server-Side)](phase-01-jwt-bcrypt-core.md) | ~4h | done |
| 02 | [Rate Limiting & Hardening (Server-Side)](phase-02-rate-limit-hardening.md) | ~2h | done |
| 03 | [Client-Side JWT Integration](phase-03-client-jwt-integration.md) | ~1h | pending |

Phases 01 and 02 are server-only and can run sequentially. Phase 03 depends on Phase 01 events (`auth_token`, `auth_error`) being defined.

## Backlog Coverage

| Backlog | Status | Notes |
|---------|--------|-------|
| **SEC-001** Plaintext password | Covered (Phase 01) | bcrypt hash + verify via MongoDB user lookup |
| **SEC-002** Hardcoded defaults | Covered (Phase 01) | Credentials moved to MongoDB; env file for MONGODB_URI in .gitignore; exit on default creds |
| **SEC-005** No session mgmt | Covered (Phase 01+03) | JWT with 1h TTL, per-command validation, in-memory storage |

## Validated Decisions

- **User store**: MongoDB (`qm_hub.robo_control_user`) replaces env-var credentials. MONGODB_URI in `.env` file, gitignored.
- **User schema**: `{ username, password_hash, role, enabled, created_at, updated_at }`. Role stored but not enforced yet (RBAC deferred).
- **TTL**: 1 hour. Balances security with operational continuity for rover control. Proactive refresh at ~55 min.
- **JWT validation scope**: Command events only (rover_command, arm_command, tracking_command, tts_command, fleet_select, audio_stream). Server→client events (video_frame) and background sweep handle the rest.
- **Default credentials**: `process::exit(1)` if admin/password used without `ALLOW_DEFAULT_CREDENTIALS=true`.
- **JWT storage (client)**: In-memory primary, sessionStorage fallback for page-refresh UX.
- **Sensitive command re-auth**: Deferred to future RBAC plan. Role field stored now for forward compatibility.
- **Dataflow YAML**: Remove hardcoded `AUTH_USERNAME`/`AUTH_PASSWORD` from `orchestra-dataflow.yml`.

## Validation Summary

**Validated:** 2026-03-26
**Questions asked:** 6

### Confirmed Decisions
- JWT TTL: 1 hour (not 15 min) — operational continuity for rover control
- Per-event validation: commands only, not every event — background sweep covers expiry
- Default credentials: hard exit with ALLOW_DEFAULT_CREDENTIALS escape hatch
- User schema: minimal with role field for future RBAC
- Re-auth for sensitive commands: deferred entirely
- User store: MongoDB (qm_hub.robo_control_user) with env file for URI

### Action Items
- [x] Update Phase 01: MongoDB dependency, user collection, seed script, TTL change
- [x] Update Phase 01: Per-command validation (not per-event)
- [x] Update Phase 01: Remove AUTH_USERNAME/AUTH_PASSWORD env vars from dataflow YAML
- [ ] Update Phase 02: No changes needed (rate limiting is independent of user store)
- [ ] Update Phase 03: TTL change affects refresh timer scheduling
