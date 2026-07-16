---
fep: "0019"
title: "LSP-integrated verification + real-time code feedback"
type: Standards Track
status: Draft
created: 2026-07-15
modified: 2026-07-15
authors:
  - name: Vijaykumar Singh
    email: singhvjd@gmail.com
    github: vjsingh1984
reviewers: []
discussion: https://github.com/vjsingh1984/victor/discussions/0019
---

# FEP-0019: LSP-integrated verification + real-time code feedback

## Summary

Wire Victor's **complete but dormant** LSP infrastructure into the agentic loop
in two complementary ways, so the agent stops editing blind and gets real,
multi-language type/syntax feedback with no subprocess cost.

First, an **`LSPVerifier`** — a `Verifier` (FEP-0018) that, after the agent
claims COMPLETE, checks LSP diagnostics on every edited file via
`orchestrator.lsp.get_diagnostics(...)`. It is instant (no command execution)
and multi-language (Python via pyright, Rust via rust-analyzer, Go via gopls,
TypeScript, …). It plugs into the existing `--verify` surface as
`--verify lsp`.

Second, an **`LSPDiagnosticMiddleware`** — a tool `MiddlewareProtocol` that
hooks `after_tool_call` and injects LSP diagnostics into the edit tool result
**in the same turn** the edit happens. The agent sees `Line 42: undefined name
'foo'` immediately and can fix it without first claiming done and re-running
the whole suite. This is the generate→diagnose→fix loop, far cheaper than the
current generate→claim done→test→retry loop.

Both paths degrade to no-ops when LSP is unavailable (victor-coding not
installed): the root repo defines the framework integration; the language
servers live in the victor-coding sibling package. Activation is automatic —
`AgentOrchestrator.set_lsp(capability)` registers the middleware, so a vertical
that provides LSP gets same-turn diagnostics with no further wiring.

## Motivation

Victor has had a full LSP capability layer since the cross-vertical intelligence
work:

- `LSPServiceProtocol` (`get_diagnostics`, `get_hover`, `get_completions`,
  `get_definition`) — `victor/framework/lsp_protocols.py`
- `LSPCapability` wrapper on the orchestrator — `victor/framework/capabilities/lsp.py`
- **Real server implementations** for 15+ languages in victor-coding (pyright,
  rust-analyzer, gopls, typescript-language-server)
- Push-based diagnostics (`textDocument/publishDiagnostics` → stored per-URI →
  pulled via `get_diagnostics`)
- A tool `MiddlewareChain` with `after_tool_call` hooks

**None of it is wired in production.** `orchestrator.set_lsp()` is never called;
the agent edits blind — no diagnostics feedback during or after editing. The
agent's only feedback signal today is "I claimed done" or a post-hoc test run.

The paradigm shift LSP enables: **real-time diagnostics during editing** — the
agent sees a type error in the same turn as the edit, not after claiming done
and re-running the whole suite.

FEP-0018 already added the verify-and-retry gate; FEP-0019 adds an instant,
multi-language Verifier for it (no subprocess) and a same-turn feedback path.

## Proposed Change

### Phase 1 — `LSPVerifier` (DECIDE gate, post-completion)

**File:** `victor/framework/verifiers.py`

An `LSPVerifier` implementing the FEP-0018 `Verifier` protocol. After the agent
claims COMPLETE:

1. Get edited files from `workspace_files_modified()` (`victor/framework/workspace.py`).
2. For each file, call `orchestrator.lsp.get_diagnostics(file_path)` — no subprocess.
3. Count severity-1 errors. Return `VerificationResult(passed, total, raw, feedback)`.

Degradation: `lsp_capability is None` → `VerificationResult(0, 0, "", "LSP not available")`
(vacuous pass — the gate is a no-op, no regression).

Wired via `_build_verifier("lsp", orchestrator)` in `agent.py`, reading
`orchestrator.lsp`. Exposed as `--verify lsp`.

### Phase 2 — `LSPDiagnosticMiddleware` (ACT phase, same-turn feedback)

**File:** `victor/framework/lsp_middleware.py` (new)

A `MiddlewareProtocol` (`before_tool_call` / `after_tool_call` / `get_priority`
/ `get_applicable_tools`). After any file-modifying tool (edit, write,
file_editor, lsp_write_enhancer):

1. Push the new content into the LSP server (`update_document`).
2. Wait briefly (~100 ms) for the server to publish updated diagnostics.
3. Pull `get_diagnostics(file_path)`.
4. Append errors to the tool result (`"⚠ LSP errors (2): Line 42: undefined name 'foo'"`).

Design decisions:
- **Non-blocking**: any LSP failure is swallowed; the edit result passes through
  unchanged. Never delays the agent.
- **Scoped**: `get_applicable_tools()` limits it to file-modifying tools; the
  middleware self-guards as defense-in-depth.
- **Low priority**: runs after most other middleware in the after-phase.
- **Configurable**: `mode="errors"` (severity=1, default) or `"all"` (warnings too).

Wiring is automatic: `AgentOrchestrator.set_lsp()` registers this middleware on
the chain whenever a vertical provides an LSP capability. The middleware is
created lazily and bound to the capability, so it works whether `set_lsp` runs
before or after the chain is built (the `set_middleware_chain` path handles the
reverse ordering). `set_lsp_feedback_mode(mode)` propagates
`SessionConfig.lsp_feedback` to the middleware.

### Phase 3 — LSP-guided generation (exploration, deferred)

Deeper integration: LSP completions / hover / go-to-definition injected into the
perception/planning phase. Requires changes to perception layers, not just
tool/middleware layers — a follow-up design.

## Benefits

- **Same-turn feedback**: the agent learns about a type error the instant it
  writes it, not after claiming done — the generate→diagnose→fix loop.
- **Multi-language, zero-subprocess**: pyright/rust-analyzer/gopls already
  running; `LSPVerifier` checks diagnostics instantly with no command exec.
- **Reuses dormant infra**: connects `LSPServiceProtocol`, `LSPCapability`, the
  middleware chain, and `workspace_files_modified()` — no new subsystem.
- **Backward-compatible**: when LSP is unavailable the middleware and verifier
  are no-ops; every existing test and session behaves exactly as before.
- **Framework-first**: any vertical that calls `set_lsp()` gets the loop; no
  eval-only or surface-local branching.

## Drawbacks and Alternatives

- **Hardcoded tool set**: the middleware's `get_applicable_tools()` lists known
  file-modifying tools. Custom edit tools must be added to the set (or rely on a
  `path`/`file_path` arg, which the middleware also requires). Acceptable: the
  set is small and stable.
- **~100 ms debounce**: the post-edit diagnostic fetch waits briefly for the
  server to publish. Non-blocking and configurable; only paid on file edits.
- **Alternative considered**: route diagnostics through a separate "verification
  middleware" outside the tool chain. Rejected: the tool `after_tool_call` hook
  is the natural place — it already sees the edited file and result.
- **Alternative considered**: keep LSP as DECIDE-gate-only (no same-turn
  feedback). Rejected: the same-turn loop is the higher-value path — it catches
  errors before the agent rationalizes a COMPLETE.

## Unresolved Questions

- **End-to-end activation from victor-coding**: the root repo defines the
  framework integration; the victor-coding vertical must call
  `orchestrator.set_lsp(LSPCapability(impl=...))` (or register it via the
  capability system) to activate. Follow-up in victor-coding.
- **Diagnostic staleness**: push-based diagnostics depend on the server having
  analyzed the document. The `update_document` + debounce mitigates this; the
  Phase-1 `LSPVerifier` re-checks at COMPLETE as a backstop.
- **Phase 3 scope**: where exactly in perception/planning to inject hover /
  completion / definition — deferred to a follow-up design.

## Implementation Plan

- **Phase 1 (this PR)**: `LSPVerifier` + `_build_verifier("lsp")` + `--verify lsp`
  + `SessionConfig.lsp_feedback` + `from_cli_flags` plumbing.
- **Phase 2 (this PR)**: `LSPDiagnosticMiddleware` + auto-registration in
  `AgentOrchestrator.set_lsp()` / `set_middleware_chain()` +
  `set_lsp_feedback_mode()` + `Agent.create` propagation.
- **Phase 3 (follow-up)**: LSP-guided generation in the perception phase.

## Migration Path

Additive and non-breaking. Both new paths are dormant by default: the middleware
is only registered when `set_lsp(capability)` is called (nothing calls it today),
and `LSPVerifier` returns a vacuous pass when `orchestrator.lsp` is None. The new
`SessionConfig.lsp_feedback` field defaults to `"errors"`. No existing session,
test, or surface changes behavior unless a vertical activates LSP.

## Compatibility

- No public API removal (new optional params, new module, one new SessionConfig
  field with a safe default).
- Graceful degradation: root repo without victor-coding is unchanged.
- Verified: ruff/black/mypy clean; 21 new LSP-integration unit tests pass; 150
  agentic-loop + streaming-parity + integration tests pass; repo-hygiene and
  import-boundary tests pass.

## References

- FEP-0018 (framework verification hook) — `LSPVerifier` implements its
  `Verifier` protocol and plugs into the DECIDE verify gate.
- `victor/framework/lsp_protocols.py` — `LSPServiceProtocol.get_diagnostics()`.
- `victor/framework/capabilities/lsp.py` — `LSPCapability`.
- `victor/core/verticals/protocols/middleware.py` — `MiddlewareProtocol`.
- victor-coding sibling package — the real language-server implementations.
