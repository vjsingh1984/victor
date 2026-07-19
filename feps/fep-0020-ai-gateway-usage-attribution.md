---
fep: "0020"
title: "AI usage gateway: per-user/team attribution + shared-key metering"
type: Standards Track
status: Draft
created: 2026-07-18
modified: 2026-07-18
authors:
  - name: Vijaykumar Singh
    email: singhvjd@gmail.com
    github: vjsingh1984
reviewers: []
discussion: https://github.com/anvai-labs/victor/discussions/0020
---

# FEP-0020: AI usage gateway — per-user/team attribution + shared-key metering

## Table of Contents

1. [Summary](#summary)
2. [Motivation](#motivation)
3. [Proposed Change](#proposed-change)
4. [Benefits](#benefits)
5. [Drawbacks and Alternatives](#drawbacks-and-alternatives)
6. [Unresolved Questions](#unresolved-questions)
7. [Implementation Plan](#implementation-plan)
8. [Migration Path](#migration-path)
9. [Compatibility](#compatibility)
10. [References](#references)
11. [Review Process](#review-process)

## Summary

Victor tracks token usage and USD cost per request, but only **per session**
(`victor/agent/session_cost_tracker.py::SessionCostTracker`,
`victor/agent/usage_analytics.py::UsageAnalytics`). It has **no per-user / per-team
attribution and no gateway** in front of a shared provider key. So a team on an
internal network that shares one `ANTHROPIC_API_KEY` cannot answer "who spent
what," cannot budget per person, and cannot rate-limit a runaway user.

This FEP adopts a small OSS gateway — **Sandhi** (`anvai-labs/sandhi`, Apache-2.0,
its own repo; Sanskrit *संधि* "junction", *the metering layer for AI agents*) — that
Victor consumes two ways: (1) an **in-process metering
middleware** wrapping `BaseProvider`, and (2) an optional **reverse-proxy serve
mode** that issues **virtual keys** (one shared upstream key → many per-user keys)
and attributes every call to a `subject_id` (user) and `group_id` (team). It emits
one **neutral usage event** (tokens + cache split, no dollars) that Victor's
existing cost surfaces consume for local display, and that AnvaiOps prices
downstream. `sandhi` lives in its own repo (not folded into Victor) so ProximaDB
(Rust) and third parties can consume the language-agnostic proxy on equal footing.
As part of this (ADR-0047 D10), Sandhi also becomes the home of the **unified provider
transport** (`sandhi-providers`, Rust): Victor migrates its ~28 hand-rolled Python adapters
onto the Sandhi binding — phased, with a Python escape hatch — becoming a thin wrapper/SDK
consumer while keeping its framework-specific concerns (see § Provider transport migration).

Victor is the **primary adopter and owner-of-record for this public feature spec**;
the authoritative open-core split is AnvaiOps ADR-0047.

## Motivation

### Problem Statement

- **No attribution across a shared key.** `SessionCostTracker` is keyed by
  `session_id`; `UsageAnalytics` is a process singleton. Neither carries a user or
  team identity. The API server (`victor/integrations/api/fastapi_server.py`)
  already maps `Authorization: Bearer <key>` → a `client_id` in `_verify_api_key`,
  but that `client_id` **is never joined to the cost records** — the identity
  exists and is thrown away at exactly the seam where attribution should happen.
- **Shared-key teams are the common internal-network deployment.** Cost pressure
  and "who is burning the budget" are real operator questions Victor cannot answer.
- **No budgets / rate limits per user.** A single user can exhaust a shared key;
  there is no per-subject cap.
- **Each sibling repo re-implements provider plumbing.** Victor (Python),
  ProximaDB (Rust `LLMClient`), AnvaiOps (Python `LLMProvider`) each have a
  provider layer. Metering that must span all three cannot be one library — it
  must be a proxy + a shared wire schema.

### Goals

1. Join a **subject (user)** and **group (team)** identity to every usage record.
2. **Virtual keys**: front a shared upstream key with per-user keys, so
   attribution and revocation are per person, not per shared secret.
3. **Budgets + rate limits** per virtual key / team (enforcement mechanism).
4. **Local cost display** from a community price table (visibility, not billing).
5. Emit the **neutral usage event** (AnvaiOps ADR-0047 D3 wire contract) so the
   same signal feeds Victor's dashboard and AnvaiOps's commercial billing.
6. Keep it **OSS and in its own repo** — Victor depends on `sandhi`; it is not a
   `victor/` internal module.

### Non-Goals

- **Pricing/billing authority.** Victor displays local cost from a public table;
  authoritative multi-tenant billing, invoicing, and tier→$ policy are AnvaiOps
  (the commercial side of the open-core line below).
- **Governance (SSO/SAML/SCIM/RBAC, compliance/audit).** Commercial (AnvaiOps).
- Replacing the provider layer — the middleware **wraps** `BaseProvider`, reusing
  the `usage` dict already on `CompletionResponse` / `StreamChunk`.

## Proposed Change

### High-Level Design

One Rust core (`sandhi-core`), two deployment shapes, one neutral event (AnvaiOps
ADR-0047 D2/D3). `sandhi-core` is a single Rust library — usage/token accounting
(full cache split), virtual-key resolution, budgets/rate-limits, provider-usage
parsers, the neutral-event emitter — exposed to Victor via a **PyO3 Python binding**
(and napi/wasm for the TS UIs). This is the "one fast implementation, not three
drifting ones" answer to "why not a Rust core with Python/TS wrappers" — that *is*
the core.

- **In-process (via the PyO3 binding)** — Victor keeps its 28-provider layer and
  calls `sandhi-core` for accounting + virtual keys + budgets, wrapping `BaseProvider`
  and reading its existing `usage` dict. Zero network hop; transport stays Python.
- **Reverse-proxy serve mode (the same core + an HTTP listener)** — `sandhi` runs as a
  small server; agents point their provider `base_url` at it with a **virtual key**.
  The proxy resolves `vk_…` → `subject_id`/`group_id`, **holds the real upstream key
  server-side** (the client never sees it), forwards, streams the response back
  byte-for-byte (O(1) memory), and emits the event after the stream closes. This is
  the **in-path / inline** shape (AnvaiOps ADR-0047 D8) — the only one that serves an
  internal-network team sharing one upstream key, and the only one a non-Python
  caller (ProximaDB) can use. It is not a redirect: metering a client that self-reports
  would be bypassable.

### Neutral usage event (the boundary object — AnvaiOps ADR-0047 D3)

```json
{
  "request_id": "…", "occurred_at": "…",
  "provider": "anthropic", "model": "claude-…", "backend": "external",
  "virtual_key_id": "vk_…", "subject_id": "alice", "group_id": "platform-team",
  "route": "…",
  "tokens_in": 0, "tokens_out": 0,
  "cache_creation_tokens": 0, "cache_read_tokens": 0,
  "gpu_seconds": null
}
```

**No dollars, no tier/SKU names** (neutral-naming rule). Victor's local display
multiplies tokens by a community price table client-side; AnvaiOps prices the same
event authoritatively downstream. This event supersets the AnvaiOps `llm_tokens`
event and the ADR-0024 `anvai-plane-core` metering envelope.

### The open-core feature line

Identical to AnvaiOps ADR-0047 D4 (single source of truth for the split):

The two columns are independent lists (not row-paired):

| OSS core (`sandhi`, Apache-2.0) — what Victor ships/uses | Commercial (AnvaiOps, private) |
|---|---|
| **Unified provider transport** — Anthropic/OpenAI-compat/Gemini/Bedrock/local, streaming, pooling, retry, circuit-breaker (§ Provider migration) | Authoritative multi-tenant billing → invoices |
| Reverse-proxy + in-proc middleware (Rust core + bindings) | Tier → **$ rate** policy / cost-multiplier resolution |
| **Virtual keys** (one shared upstream key → many user keys) | Managed dashboards **at scale** (multi-account) |
| **Per-user / per-team usage attribution** | **SSO / SAML / SCIM / RBAC** governance |
| **Budgets + rate limits** (enforcement mechanism) | Compliance / audit logs, retention |
| **Local cost display** (community price table, visibility only) | Cross-cloud control plane; quota **authority** |
| Self-hosted single-node dashboard + Prometheus | AnvaiOps taxonomy status + capability grants |
| Neutral usage-event schema + local sink (SQLite/JSONL) | |

### Detailed Specification (seams in `victor/`)

- **Attribution join** — extend `victor/integrations/api/fastapi_server.py` so the
  `client_id` resolved by `_verify_api_key` (and the WebSocket `auth` path) is
  carried into the cost record, and generalize it to a `subject_id` + `group_id`.
  This is the load-bearing change: the identity already exists at the auth seam;
  it must reach `SessionCostTracker.record_request(...)`.
- **Provider middleware** — a metering wrapper around `BaseProvider` that reads the
  existing `usage` dict (`prompt_tokens`, `completion_tokens`,
  `cache_creation_input_tokens`, `cache_read_input_tokens`) and emits the event.
  Streaming: finalize counts from the terminal usage chunk (never estimate).
- **Virtual-key store + budgets** — provided by `sandhi`; Victor configures it
  (which upstream key, which subjects, which caps). Reuse
  `victor/config/api_keys.py` / `provider_settings.py` for the upstream secret and
  `victor/core/identity/` for enterprise-identity subjects where present.
- **Display** — extend `victor/ui/dashboard/` + `victor/observability/` to render
  per-subject/per-team roll-ups from the local sink.
- **Session / prompt-cache / KV affinity (must-preserve — AnvaiOps ADR-0047 D9).**
  The gate must **not** flatten many users into one "single user" session or mangle
  the cacheable prefix — that destroys prompt-cache economics and thrashes KV cache.
  Concretely, for Victor: (a) forward the cacheable prefix **byte-exact** and put
  `subject_id`/`group_id`/`virtual_key_id` in headers/metadata **outside** the cached
  prompt (never a per-request nonce inside the system block — that turns a ~0.1× cache
  *read* into a ~1.25× *write* every turn); (b) derive and propagate a stable
  **per-conversation affinity key** (from the virtual key + conversation id) so hosted
  APIs keep hitting their prefix cache and a self-hosted vLLM fleet **consistent-hash
  routes** the same conversation to the warm instance (avoiding KV recompute per turn);
  (c) record the D3 `cache_creation_tokens` vs `cache_read_tokens` **per subject/session**
  so a shared-prefix read is attributed cheaply to whoever read it. This is the metering
  counterpart to FEP-0011's `CacheCostModel` — FEP-0011 optimizes *what to cache*; this
  ensures the gate *preserves* the cache it earns. Cache-namespace default: shared within
  a `group_id` (team), with per-`subject_id` isolation as a stricter opt-in.

### Provider transport migration (Victor → Sandhi) — AnvaiOps ADR-0047 D10

Today Victor **hand-rolls ~28 provider adapters** in Python (`victor/providers/*.py`,
httpx-based) — the same wire layer ProximaDB (Rust `LLMClient`) and AnvaiOps (Python
`LLMProvider`) *also* re-implement. That triplication sits exactly where metering trust is
decided (usage / cache-token parsing is provider-specific). This FEP adopts the ADR-0047
D10 decision: **Sandhi owns the unified provider transport** (`sandhi-providers`, Rust), and
Victor becomes a **thin wrapper / SDK consumer** of it for in-process use — the direction
the operator asked for ("Victor becomes a wrapper-compatible SDK user for Sandhi… instead of
hand-rolling most of the APIs").

- **What moves to `sandhi-providers` (Rust):** the wire adapters (built on the official
  Anthropic / OpenAI-compatible / Gemini / Bedrock / Cohere specs — OpenAI-compat alone
  covers ~20 of Victor's 28), streaming/SSE parsing, **usage + cache-split extraction at the
  source**, connection pooling, retry, circuit-breaker, rate-limit handling. Design patterns:
  **adapter** (per-provider), **strategy** (routing/fallback), **factory** (from config),
  **decorator** (metering + resilience wrappers). Exposed both **in-process** (PyO3 binding)
  and **inter-process** (the proxy).
- **What stays in Victor (Python):** prompt assembly + the **FEP-0011 `CacheCostModel`**
  hints to the assembler, tool-format translation, agent-aware provider *selection*, plugin
  ergonomics, and capability surfacing to the agent. Victor keeps `BaseProvider` as its
  *framework* seam; its concrete adapters delegate transport to the binding.
- **Python escape hatch (mandatory).** Sandhi's provider registry accepts a **host-language
  adapter** (a Python callback), so Victor's custom / air-gapped / community providers
  register **without** a Rust contribution — Victor's extensibility principle ("extensibility
  must not compromise reliability") is preserved.
- **Phased, opt-in, never big-bang.** Migrate provider-by-provider behind a flag; keep the
  existing Python adapters as fallback until Sandhi's transport is proven. Victor's mature
  resilience path (circuit breaker, OAuth FEP-0004, health, capability detection) is not
  hard-cut in one step.

### API Changes

- **New (in `sandhi`, consumed by Victor):** the metering middleware, the virtual-key
  proxy, the neutral usage-event type, **and the `sandhi-providers` transport layer** (the
  concrete provider adapters Victor delegates to).
- **Modified (in `victor/`):** the API-server auth path carries `subject_id` /
  `group_id`; `SessionCostTracker` / `UsageAnalytics` gain optional subject/group
  fields (default `None` → today's behavior).

### Configuration Changes

```yaml
usage_gateway:
  enabled: false            # opt-in; default off = today's behavior
  mode: middleware          # middleware (PyO3 in-proc) | proxy (inline serve mode)
  sink: sqlite              # local sink for the neutral events
  price_table: community    # local cost display only (not billing)
  cache_namespace: group    # group (team-shared cache, max hit rate) | subject (isolation)
  session_affinity: true    # preserve per-conversation cache-affinity key; false only for stateless
```

### Dependencies

- New optional extra depending on Sandhi (`anvai-labs/sandhi`), mirroring the
  `proximadb` optional extra + the `victor-codegraph` git-dep pattern. The dep is the
  **PyO3 binding of the Rust core** — published to **PyPI as `sandhi-gateway`** (the bare
  `sandhi` name is taken by an unrelated Sanskrit-linguistics lib; the crate + GitHub repo
  stay `sandhi`), a prebuilt abi3 wheel like `proximadb-embedded`, so Victor gets the fast
  shared core (transport + accounting) without a Rust toolchain at install time; the
  reverse-proxy binary is a separate artifact for serve mode.

## Benefits

### For the Framework

- Answers "who spent what" across a shared key — the top operator question Victor
  cannot answer today — and matches LiteLLM/Helicone on the exact feature that
  drives adoption, without a paywall.
- One neutral event feeds both Victor's local display and AnvaiOps's commercial
  billing, so metering stops drifting across the three repos' provider layers.

### For Vertical Developers

- Verticals get per-user attribution + budgets for free via the middleware; no
  vertical code change unless they ship a custom provider (then they wrap it once).

## Drawbacks and Alternatives

### Drawbacks

- A new external dependency (`sandhi`) and a cross-repo wire contract to keep stable.
  Mitigation: default-off; `sandhi` is its own semver'd repo (same discipline as
  `victor-codegraph`).
- The proxy adds a network hop. Mitigation: O(1)-memory stream pass-through,
  off-critical-path emission; middleware mode has no hop at all.

### Alternatives Considered

1. **Fold the gateway into `victor/` as an internal module.** Rejected: a
   Python-internal home cannot serve ProximaDB (Rust) or third parties over the
   proxy; a neutral repo keeps them first-class (AnvaiOps ADR-0047 D1).
2. **Extend `SessionCostTracker` with attribution, no proxy.** Rejected as
   insufficient: it covers same-process Victor use but not shared-key teams or the
   internal-network deployment, which inherently need a proxy + virtual keys.
3. **Fold into AnvaiOps (commercial).** Rejected: it would not be OSS and could not
   drive Victor adoption; AnvaiOps holds only the commercial policy layer.

## Unresolved Questions

- Name + packaging decided: **Sandhi** (Sanskrit "junction"; AnvaiOps ADR-0047 § D1a).
  Availability verified 2026-07-19 — crate + GitHub `anvai-labs/sandhi` free; PyPI/npm bare
  `sandhi` taken by a Sanskrit-linguistics lib, so publish **PyPI `sandhi-gateway`** + **npm
  `@anvai-labs/sandhi`**.
- Migration granularity: which provider ports first? Proposed: an OpenAI-compat adapter
  (unlocks ~20 providers at once), then Anthropic (validates the cache-split parsing that
  metering depends on), then the rest behind the flag.
- Is `group_id` a single team or a path (org/team/project)? Proposed: an opaque
  string the operator assigns; hierarchy is a display concern, not a schema one.
- Does the local sink default to SQLite (queryable) or JSONL (append-only)?
  Proposed: SQLite for the dashboard, JSONL export for portability.
- Where does the per-conversation **affinity key** come from when the caller is a raw
  proxy client with no conversation id? Proposed: fall back to a hash of the stable
  prompt prefix (still gives prefix-cache affinity) and let SDK callers pass an explicit
  `conversation_id`. Getting this wrong is the KV-thrash / lost-discount failure (D9).
- Cross-user cache visibility within a `group_id`: acceptable for internal teams, but is
  it ever a data-isolation concern? Proposed: `cache_namespace: subject` opt-in covers it.

## Implementation Plan

Code is **out of scope for this FEP** (decision doc only, per AnvaiOps ADR-0047).
The phased build, when it starts:

### Phase 1: Attribution join (Victor-local, non-breaking)
- Carry `subject_id`/`group_id` from the API-server auth seam into the cost
  records; add optional fields to `SessionCostTracker`/`UsageAnalytics`.

### Phase 2: `sandhi` middleware adoption
- Depend on `sandhi`; wrap `BaseProvider` with the metering middleware; emit the
  neutral event to a local sink; render per-subject display.

### Phase 3: Reverse-proxy serve mode
- Virtual keys + budgets + rate limits for shared-key teams; the internal-network
  deployment story.

### Phase 4: Provider transport migration (ADR-0047 D10)
- Migrate Victor's Python adapters onto `sandhi-providers` provider-by-provider behind a
  flag (OpenAI-compat first → Anthropic → rest), Python adapters as fallback, the Python
  escape hatch for custom providers. Victor retains prompt assembly / FEP-0011 hints /
  tool translation / agent-aware selection. Runs in parallel with / after Phase 2.

## Migration Path

1. Ships default-off (`usage_gateway.enabled: false`) — zero behavior change.
2. Operators opt into middleware mode (in-process, no hop) first.
3. Teams sharing a key opt into proxy mode when they want per-user virtual keys.

## Compatibility

- **Breaking change:** No. Default-off; attribution fields default to `None`
  (today's session-scoped behavior).
- **Minimum Python:** 3.11 (unchanged).

## References

- AnvaiOps **ADR-0047** — AI usage gateway open-core split + cross-repo homing
  (authoritative for the OSS/commercial line and the wire contract).
- Victor **ADR-018** — decision to adopt/depend on `sandhi`.
- ProximaDB **ADR-067** — the sibling egress-metering consumer.
- FEP-0011 (provider cache cost model — the provider-seam precedent; explicitly
  non-metering, which this FEP complements).
- `victor/agent/session_cost_tracker.py`, `victor/agent/usage_analytics.py`,
  `victor/integrations/api/fastapi_server.py` (`_verify_api_key`),
  `victor/providers/base.py` (`usage` on `CompletionResponse`/`StreamChunk`).

## Review Process

- **Submitted by:** Vijaykumar Singh
- **Initial review period:** 14 days minimum.
- **PR:** TBD.

---

## Copyright

This FEP is licensed under the Apache License 2.0, same as the Victor project.
