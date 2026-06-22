# Observability & Learning Data Axes

<!-- markdownlint-disable-file MD013 -->
<!-- Wide inventory/axis tables are intentional and unwrappable at 80 cols. -->

Status: active design rule (R0). Companion to the RL/trace unification plan.

This document is the single catalogue of every trace / telemetry / learning sink in Victor,
classifies each by **axis**, and states the **rule** that prevents the trace and RL proliferation
we already have from growing further. Read this before adding any new event, table, or counter that
records what tools/decisions/outcomes happened.

## The three axes

Telemetry and learning data fall on exactly three axes. Conflating them is the root cause of the
current duplication (the same fact recorded in several places, computed several ways, joinable by
nothing).

| Axis | Question it answers | Canonical home | Scope |
|---|---|---|---|
| **Capture** | *What was offered / decided / invoked / resulted?* | per-session event log + observability bus | project (`./.victor/project.db`) |
| **Learn (policy)** | *What did we conclude?* (Q-values, thresholds, params) | `RL_Q_VALUE` / `RL_PARAM` | global (`~/.victor/victor.db`) |
| **Decide** | *What do we do this turn?* | reads the **policy** (and current context), not raw traces | runtime |

Flow: **capture once → RL learns from the capture → gates read the learned policy.** RL must not
re-record what the trace already captured; gates must not re-tally what the policy already learned.

The two-DB split is intentional and preserved: **per-session capture is project-scoped** (joins the
ConversationStore), **learned policy is cross-project** (global). They join on the globally-unique
`session_id` (+ `turn_id`). This is why RL can *reuse* the capture without copying it into a second
store.

## Correlation spine

Every capture record must carry, where available:

- `session_id` — conversation/session (already globally unique, e.g. `codingagent-<hash>`)
- `turn_id` — one PERCEIVE→ACT→EVALUATE turn (LLM call + its tool batch)
- `request_id` — one operation (a single tool call / decision)

Propagated via `victor/runtime/trace_context.py` (`TraceContext`, contextvars). This spine is what
makes *offered → invoked → resulted → outcome* a single joinable chain. Its absence today
(`rl_outcome` has no `session_id`) is the central gap.

## Sink inventory

Axis legend: **O**=offered, **D**=decided, **I**=invoked, **R**=resulted, **P**=policy/learned,
**U**=usage/metrics, **M**=trace metadata.

| # | Sink | File | Axis | Persistence | Correlation | Consumers |
|---|---|---|---|---|---|---|
| 1 | JSONL usage logger | `victor/observability/analytics/enhanced_logger.py` | U (all) | `~/.victor/logs/usage.jsonl` (rotated/gz) | `session_id` | prompt optimizer (`_collect_traces`) |
| 2 | Observability bus (~47 topics) | `victor/core/events/backends.py` | O,D,I,R,M | in-mem (opt. SQLite) | request-corr id (often absent) | dashboard, subscribers |
| 3 | ConversationStore | `victor/agent/conversation/store.py` | I,R,U | project.db | `session_id` | prompt optimizer, analytics |
| 4 | RL outcome fact | `victor/core/schema.py` `rl_outcome` | R | global.db | **none** (no session_id) ⚠ | RL learners, GEPA |
| 5 | Edge-model decisions | `victor/agent/decisions/` | D | ephemeral (opt JSONL) | optional | decision flow |
| 6 | Task execution report | `victor/agent/services/metrics_service.py` | R,U | in-mem (opt export) | `task_id` | dashboard, benchmark |
| 7 | TraceContext spans | `victor/runtime/trace_context.py` | M | in-mem (contextvars) | `trace_id`/`span_id` | debug, error prop |
| 8 | MetricsCollector | `victor/agent/metrics_collector.py` | O,I,U | in-mem | session (implicit) | dashboard |
| 9 | tool.intent event | `victor/agent/tool_pipeline.py::_emit_tool_intent` | I (pre-exec) | bus topic `tool.intent` | none ⚠ | future voting/gating |
| 10 | RL metrics exporter | `victor/observability/rl_metrics.py` | P,U | in-mem snap / Prometheus | learner_id | dashboard, alerts |
| 11 | Event sourcing (opt) | `victor/core/event_sourcing.py` | full | opt SQLite | `aggregate_id` | audit/replay |
| 12 | Topology telemetry | `victor/agent/topology_telemetry.py` | D | bus | implicit | dashboard, evaluator |
| 13 | **tool.supply (ToolSupplyTrace)** | `victor/tools/tool_supply_trace.py` (PR #220) | **O** | bus topic `tool.supply` | (spine added in R1) | over-restriction analysis |
| 14 | RL Q-values / params (policy) | `victor/core/schema.py` `rl_q_value`/`rl_param` | **P** | global.db | learner/state/action | decision gates |
| 15 | ToolExperienceStore | `victor/tools/experience_store.py` | R (dup of #4) ⚠ | in-mem (max 5000) | none | E3-TIR exploration |

## Known duplications (being removed)

- **Outcome captured twice** (#4 `rl_outcome` + #15 `ToolExperienceStore`) from the same
  `TOOL_EXECUTED` hook, no sync → drift. *Fix (R3): one `record_outcome` writes the durable fact and
  updates the in-mem projection; the experience store stops being independently fed.*
- **Reward computed ≥3 ways** (per-learner `_compute_reward`, `implicit_feedback.compute_reward`, and
  an inline `quality_score or 1.0/0.3` in `tool_selection_runtime`). *Fix (R2): one canonical reward;
  shaping overrides delegate to it; the inline path is deleted.*
- **Gate-1 double-counts tool success** (#14 RL Q-value boost **and** #15 experience `success_rate`
  both feed the same ranking). *Fix (R4): the RL Q-value is the single tool-success signal; the
  experience store is used only for exploration scheduling (underused/stale tools).*
- **No correlation spine** (#4, #9, #13 lack joinable ids). *Fix (R1).*

## The rule (apply before adding any telemetry/learning data)

1. **Selection-side** telemetry (what was offered/considered/dropped) → the `tool.supply`
   `ToolSupplyTrace`. Do not add a parallel selection log.
2. **Execution outcome** (what a tool/decision produced) → the single `record_outcome(...)` path →
   `rl_outcome` (durable) + the in-mem projection. Do not record outcomes from a second site.
3. **Learned signal** (anything aggregated/updated over time to influence future turns) → policy
   (`rl_q_value`/`rl_param`) via an RL learner. Do not keep a second running tally next to it.
4. **Decision gates** read the **policy** (and current context). A gate must not re-aggregate raw
   traces inline (that re-implements the learner and risks double-counting).
5. **Every capture record carries the correlation spine** (`session_id`/`turn_id`/`request_id`) so
   the lifecycle is joinable. A new sink without it must justify why.
6. **Pick the right DB:** per-session capture → project.db; cross-project learned policy → global.db.
   Never store project capture in the global DB or learned policy in the project DB.

A new event topic, table, column, or counter that records tools/decisions/outcomes should map to
exactly one axis and one canonical home above. If it seems to need a new home, that is a signal to
extend an existing axis, not to add a sink.
