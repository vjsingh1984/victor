# Tiered Execution Backlog (2026-03-26)

**Purpose**: Durable issue-sized backlog derived from the refreshed codebase assessment. This document is the local source of truth for GitHub issue creation when remote issue updates are not performed in the same session.

## Foundational

| Backlog ID | Proposed Issue Title | Priority | Suggested Labels | Primary Touchpoints |
|---|---|---|---|---|
| `FND-26-01` | Refresh canonical roadmap dates and link it to the new assessment/backlog | P0 | `E2-governance`, `priority-P0`, `M1-foundation` | `roadmap.md` |
| `FND-26-02` | Supersede stale assessment artifacts and keep one current codebase baseline | P1 | `E2-governance`, `priority-P1` | `docs/tech-debt/codebase-assessment-2026-03-15.md`, `docs/tech-debt/codebase-assessment-2026-03-26.md` |
| `FND-26-03` | Convert the 19 actionable TODO/FIXME markers into tracked work items | P1 | `E2-governance`, `priority-P1` | `docs/tech-debt/todo-triage-2026-03-15.md` |
| `FND-26-04` | Keep a concise product vision alongside the roadmap | P1 | `E2-governance`, `priority-P1` | `VISION.md`, `README.md` |

## Security

| Backlog ID | Proposed Issue Title | Priority | Suggested Labels | Primary Touchpoints |
|---|---|---|---|---|
| `SEC-26-01` | Migrate generic provider and server secrets to `SecretStr` | P0 | `priority-P0`, `security` | `victor/config/settings.py`, `victor/config/unified_settings.py`, `victor/providers/base.py` |
| `SEC-26-02` | Document SBOM consumption and scanner exception handling for operators | P1 | `priority-P1`, `security`, `docs` | `SECURITY.md`, release docs |
| `SEC-26-03` | Add explicit review guardrails for sandboxed `eval()` call sites | P1 | `priority-P1`, `security` | `victor/workflows/handlers.py`, `victor/ui/slash/commands/debug.py` |

## Design / Architecture

| Backlog ID | Proposed Issue Title | Priority | Suggested Labels | Primary Touchpoints |
|---|---|---|---|---|
| `ARC-26-01` | Remove deprecated sync provider switching paths from orchestrator | P0 | `E1-orchestration`, `priority-P0`, `M2-midpoint` | `victor/agent/orchestrator.py` |
| `ARC-26-02` | Split conversation memory into storage, retrieval, and policy layers | P0 | `E1-orchestration`, `priority-P0`, `M2-midpoint` | `victor/agent/conversation_memory.py` |
| `ARC-26-03` | Replace the workflow compiler migration stub with a real compiler boundary | P0 | `priority-P0`, `workflows` | `victor/workflows/compiler/unified_compiler.py`, `victor/workflows/unified_compiler.py` |
| `ARC-26-04` | Decompose `VictorAPIServer` into smaller API composition and service modules | P1 | `priority-P1`, `api`, `M2-midpoint` | `victor/integrations/api/server.py`, `victor/integrations/api/routes/` |
| `ARC-26-05` | Decompose framework vertical integration into manifest/runtime/config seams | P1 | `priority-P1`, `framework`, `verticals` | `victor/framework/vertical_integration.py` |
| `ARC-26-06` | Split ProximaDB multi-model provider into extraction, indexing, and query services | P1 | `priority-P1`, `storage`, `search` | `victor/storage/vector_stores/proximadb_multi.py` |
| `ARC-26-07` | Decide whether observability dashboard is a production surface or an archived prototype | P1 | `priority-P1`, `observability` | `victor/observability/dashboard/app.py` |

## Product / Vision / DX

| Backlog ID | Proposed Issue Title | Priority | Suggested Labels | Primary Touchpoints |
|---|---|---|---|---|
| `PRD-26-01` | Narrow the next-quarter product story to runtime trust, SDK clarity, and benchmark credibility | P1 | `priority-P1`, `product`, `roadmap` | `roadmap.md`, `VISION.md`, `README.md` |
| `PRD-26-02` | Publish benchmark execution results instead of code-complete-only status | P1 | `E6-benchmark`, `priority-P1` | `docs/benchmarking/`, benchmark scripts |
| `PRD-26-03` | Create a true 5-minute quickstart path for first-time users and contributors | P2 | `docs`, `priority-P2` | `README.md`, getting-started docs |

## Execution Order

1. `FND-26-01` to `FND-26-04`
2. `SEC-26-01` to `SEC-26-03`
3. `ARC-26-01` to `ARC-26-07`
4. `PRD-26-01` to `PRD-26-03`

## Notes

- If GitHub issue creation is available in-session, use this document as the source text for issue titles, labels, and touchpoints.
- If GitHub access is unavailable, keep this file current and reference it from the canonical roadmap.
