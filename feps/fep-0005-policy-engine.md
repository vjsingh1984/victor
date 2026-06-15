---
fep: "0005"
title: "Governance Policy Engine (ALLOW/DENY/ASK over tool execution)"
type: Standards Track
status: Draft
created: 2026-06-14
modified: 2026-06-14
authors:
  - name: Vijaykumar Singh
    email: singhvjd@gmail.com
    github: vjsingh1984
reviewers: []
discussion: https://github.com/vjsingh1984/victor/discussions/TBD
---

# FEP-0005: Governance Policy Engine

## Summary

Add a declarative **governance policy engine** to Victor that intercepts agent
tool execution at defined lifecycle phases and returns one of three verdicts —
**ALLOW / DENY / ASK** — composing an ordered list of policies (session →
agent → server, strictest-first). The engine is assembled entirely from
existing framework primitives (the `MiddlewareChain` tool seam, the
`HITLController` approval flow, and live token/cost accounting), exposed to the
tool pipeline as a single `CRITICAL`-priority middleware. Ships with three
builtins: cost budget, tool-approval gate, and per-session tool-call cap.
Disabled by default behind the `USE_POLICY_ENGINE` feature flag.

This is a cross-learning from **Omnigent** (a meta-harness over Claude Code,
Codex, and Pi), whose standout, directly-transplantable feature is exactly such
a three-level policy engine.

## Motivation

### Problem Statement

Victor has no unified, user-vetoable governance surface. Safety today is
scattered across per-vertical middleware (`GitSafetyMiddleware`,
`SecretMaskingMiddleware`, …), and there is:

- no spend cap or cost-aware gating,
- no per-session tool-call cap as a runaway/loop guard,
- no "ask me before this risky action" approval flow tied to tool execution,
- no single place to declare governance that stacks across trust levels.

Omnigent demonstrates that this control plane is high-value and can be modeled
cleanly. Victor already owns every primitive needed; they were simply never
assembled into one declarative layer.

### Goals

- A small, composable policy abstraction with ALLOW/DENY/ASK verdicts.
- Phase-based interception at `TOOL_CALL` (gate/modify args) and `TOOL_RESULT`
  (redact/transform output).
- Three-level ordering (session → agent → server) with strict-first
  short-circuit on DENY.
- ASK resolved through the existing `HITLController`, with a fail-safe fallback
  when no approval handler is wired.
- Builtins: `CostBudgetPolicy`, `AskOnToolsPolicy`, `MaxToolCallsPolicy`.
- Zero behavior change when disabled; opt-in via feature flag + setting.
- No new parallel abstraction layer; reuse `MiddlewareChain`, `HITLController`,
  and `ConversationStore` cost accounting.

### Non-Goals

- REQUEST / LLM_REQUEST / RESPONSE phases (the `Phase` enum is left extensible;
  only tool phases are wired now).
- Streaming-path integration (`StreamingChatPipeline` has its own loop).
- Server-wide admin policy distribution/persistence and an interactive CLI
  approval UI (the approval handler is pluggable; default is fail-safe deny).
- Sandboxing and the Executor-seam refactor (separate future FEPs).

## Proposed Change

### High-Level Design

```python
from victor.framework.policies import (
    PolicyEngine, PolicyEngineMiddleware,
    CostBudgetPolicy, AskOnToolsPolicy, MaxToolCallsPolicy,
)

engine = PolicyEngine([
    CostBudgetPolicy(max_cost_usd=5.0, ask_thresholds_usd=[3.0]),
    AskOnToolsPolicy(["run_command"]),
    MaxToolCallsPolicy(limit=50),
])
chain.add(PolicyEngineMiddleware(engine, context_provider, approval_handler=handler))
```

A `Policy` evaluates a `PolicyEvent` (phase + tool name + arguments + a
`PolicyContext` snapshot of cost/model/labels) and returns a `PolicyVerdict`.
The `PolicyEngine` composes verdicts: first DENY short-circuits; any ASK (no
DENY) collapses to a single approval; otherwise ALLOW. Argument and result
modifications thread between policies.

### Detailed Specification

#### Verdict composition (`engine.py`)
- Policies run in list order (caller pre-orders strictest/most-local first).
- First `DENY` returns immediately (later policies do not run).
- `ASK` verdicts accumulate; if any ASK and no DENY, return a single ASK with
  combined reason and the first asking policy's name.
- A policy that raises is skipped (fail-open for that one policy) and logged —
  a misbehaving policy never crashes the tool pipeline.
- DENY/ASK are emitted to an optional `(topic, payload)` observability sink.

#### Middleware bridge (`middleware.py`)
- Implements the existing `MiddlewareProtocol`; priority `CRITICAL`, applies to
  all tools (per-policy scoping happens inside the engine).
- `before_tool_call` → `TOOL_CALL` phase. DENY → `MiddlewareResult(proceed=False)`;
  ASK → await `HITLController.process_approval` and collapse to proceed True/False;
  ALLOW → `proceed=True` with any modified arguments.
- `after_tool_call` → `TOOL_RESULT` phase; returns a transformed (redacted)
  result or `None` (unchanged) per the chain contract.
- Because `MiddlewareResult` is boolean-only, ASK is resolved *synchronously
  inside* the async `before_tool_call`. If no approval handler is configured,
  the ASK resolves via `ask_fallback` (`"deny"` default = fail safe).

### API Changes

New public package `victor.framework.policies` exporting `PolicyEngine`,
`PolicyEngineMiddleware`, `Policy`, `Phase`, `PolicyAction`, `PolicyContext`,
`PolicyEvent`, `PolicyVerdict`, and the three builtins. New public method
`ConversationController.get_session_cost_usd()` (read-only, returns live
session cost; 0.0 when unavailable).

### Configuration Changes

New nested settings group `settings.governance` (`GovernanceSettings`):

```yaml
governance:
  enabled: false                       # master switch (also needs the flag)
  cost_budget_usd: 0.0                  # hard cap (0 = off)
  cost_ask_thresholds_usd: []           # soft checkpoints -> one-time ASK
  expensive_models: ["opus", "fable", "gpt-5.5"]
  ask_on_tools: []                      # tools requiring approval
  max_tool_calls_per_session: 0         # 0 = off
  ask_fallback: "deny"                  # "deny" (fail safe) | "allow"
```

New feature flag `USE_POLICY_ENGINE` (`VICTOR_USE_POLICY_ENGINE=true`), opt-in,
off by default. Wiring in `CoordinationBuilders.create_middleware_chain()` adds
the middleware only when the flag and `governance.enabled` are both set.

### Dependencies

None new. Reuses `victor-contracts` middleware protocol, `victor.framework.hitl`,
and `victor.agent.conversation` cost accounting.

## Benefits

### For Framework Users
Spend caps, "ask before risky tools," and runaway-call guards — declaratively,
without writing middleware.

### For Vertical Developers
A first-class extension point: ship domain policies as `Policy` subclasses
instead of bespoke middleware.

### For the Ecosystem
Closes a documented competitive gap (governance/production maturity) and gives
a foundation for future REQUEST/RESPONSE-phase policies and admin distribution.

## Drawbacks and Alternatives

### Drawbacks
- ASK blocks the tool call synchronously; a slow/absent approver delays the turn
  (bounded by `ask_timeout_seconds`, mitigated by fail-safe fallback).
- Per-session cost relies on the `ConversationController` being resolvable; when
  absent, the cost gate fails open (other policies are unaffected).

### Alternatives Considered
- **A new parallel interception layer** — rejected; violates the "smallest layer
  that fits / no new parallel abstraction" mandate. The existing `MiddlewareChain`
  already provides the exact hook points.
- **Extending `MiddlewareResult` with a native ASK action** — rejected for the
  MVP to avoid a contracts (SDK) change; ASK is collapsed in the middleware.
  Can be revisited as a follow-up.

## Unresolved Questions

- Should `ask_on_tools` support glob/category matching in addition to exact
  names?
- Where should the interactive CLI/TUI approval handler live, and how does it
  surface elicitation to the user?

## Implementation Plan

### Phase 1: Foundation (done)
`victor/framework/policies/` (types, base, engine, middleware, builtins) + unit
tests.

### Phase 2: Integration (done)
`GovernanceSettings`, `USE_POLICY_ENGINE` flag, wiring in `CoordinationBuilders`,
`ConversationController.get_session_cost_usd()`.

### Phase 3: Rollout (in progress)
Done: observability emit of DENY/ASK to the event bus; reusable console
approval handler (`console_approval_handler` / `make_console_approval_handler`)
wired via the opt-in `governance.interactive_approval` setting (TTY-gated).
Remaining: docs; optional REQUEST/RESPONSE phases; streaming-path integration;
container-registered approval handler for non-TTY surfaces (API/TUI).

### Testing Strategy
36 unit tests covering engine composition, middleware DENY/ASK/fallback, result
redaction, and the three builtins. Integration smoke test through the real
`MiddlewareChain` confirms flag/setting gating and tool-call blocking.

### Rollout Plan
Opt-in flag, off by default; no behavior change until explicitly enabled.

## Compatibility

### Backward Compatibility
Fully backward compatible. New optional settings group (auto-defaults), new opt-in
flag, new package. No existing API changed; one additive method on
`ConversationController`.

### Version Compatibility
No `victor-contracts` change required.

### Vertical Compatibility
Verticals may ship `Policy` subclasses; existing middleware is unaffected.

## References

- Omnigent policy engine: `../omnigent/omnigent/policies/`, `docs/POLICIES.md`.
- Victor middleware seam: `victor/agent/middleware_chain.py`,
  `victor-contracts/.../promoted.py`.
- HITL: `victor/framework/hitl.py`.
- Token/cost accounting: `victor/agent/conversation/store.py`.

## Review Process

### Submission
Submitted alongside the reference implementation under
`victor/framework/policies/`.

### Review Timeline
Standard 14-day review per FEP-0001.

### Review Checklist

#### Technical Review
- [x] Reuses existing primitives (no new parallel layer)
- [x] Off by default; opt-in flag + setting
- [x] Boundary-guard clean (no new container/global-state/singleton usage)
- [x] Unit + integration tests passing

#### Community Review
- [ ] API naming feedback
- [ ] Builtin coverage feedback

### Decisions
Pending.

### Revision History
- 2026-06-14: Initial draft + reference implementation.

## Acceptance Criteria

- [x] `victor.framework.policies` package with engine, middleware, builtins.
- [x] `GovernanceSettings` + `USE_POLICY_ENGINE` flag (off by default).
- [x] Wiring in `CoordinationBuilders`; no-op when disabled.
- [x] Tests green; regression and guard suites unaffected.
