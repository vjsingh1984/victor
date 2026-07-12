# Codebase Review Backlog — First-Principles & Co-Design Findings

**Purpose**: Persistent, cross-session tracker for architecture/code-design improvement
findings. Each finding has a stable ID, status, evidence pointers, and rationale so work
can be resumed across sessions without re-deriving context.

**Convention**:
- Status ∈ `OPEN` / `IN-PROGRESS` / `DONE` / `WONTFIX` / `INVALID`
- Update the **Changelog** at the bottom on every edit.
- Every evidence line cites a real path/symbol/count from a prior tool check. Do **not**
  restate a finding without verifying its evidence still holds.

**Review basis (snapshot)**:
- Total: 666,897 LOC / 1,892 files / 22,573 functions / 5,145 classes (`code metrics victor/`).
- Doc-vs-code drift: CLAUDE.md claims "24 providers / 34 tools"; actual `ls` = **49 providers / 79 tools**.

---

## Legend — Impact/Effort ratio
- `HH` = very high ROI (do first) · `HL` = high impact, low effort · `HM` = high impact, medium effort
- `ML` = medium impact, low effort · `MH` = medium impact, high effort

---

## TIER 1 — High Impact, Low/Medium Effort

### F-001 · Reconcile doc-vs-code count drift — `HL`
- **Status**: DONE (provider/tool counts verified correct; gate coverage extended to instruction files)
- **Evidence (re-verified 2026-04)**:
  - A CI gate ALREADY EXISTS: `.github/workflows/ci-fast.yml:110` runs `scripts/ci/check_docs_drift.py` (passed green in PR #433). It auto-derives `providers` from `victor/providers/*_provider.py` (excl base/compat) and pins `CANON_TOOL_MODULES=34`, `CANON_VERTICALS=9`.
  - Original backlog evidence was partly WRONG: CLAUDE.md:113 "24 LLM provider adapters" and :114 "34 tool modules" are **CORRECT** (match the derived canon `providers=24`, `tool_modules=34`). The raw `ls *.py`=49/79 counts include `__init__.py`/`base.py`/utils, not real adapters — not comparable.
  - Genuine gap found: the gate's scan set (`DOC_GLOB`/`EXTRA_FILES`) only covered `docs/**/*.md` + `mkdocs.yml` + `docs/conf.py` — it did **NOT** scan the root instruction files (CLAUDE.md, .victor/init.md, AGENTS.md) that F-001 flagged. A future drift there would be silently missed.
- **Action taken**: Extended `EXTRA_FILES` to include `CLAUDE.md`, `AGENTS.md`, `.victor/init.md`. Verified bidirectionally (wrong count "25" → gate fails with `.victor/init.md:37 provider count 25 != 24`; restored → passes). Added regression test `test_instruction_files_are_scanned`.
- **WONTFIX scope (subclass counts)**: CLAUDE.md:215 "55 BaseTool / 36 BaseProvider / 46 slash commands" vs init.md:37 "42 / 39 / 54" disagree, but these cite *graph-index transitive* counts, not grep-derivable direct-inheritance counts (grep gives 16/23/55). Making these self-enforcing would require a graph-index count query in CI — out of scope for a docs gate. Left as-is.
- **Effort**: Low. **Impact**: Medium.

### F-002 · Orchestrator holds 13 `_initialize_*` methods despite Facade contract — `HM`
- **Status**: DONE — [FEP-0016](../../feps/fep-0016-wire-initialization-phase-manager.md) **Implemented**. Scoping found the intended fix (`InitializationPhaseManager`) already existed but was **dead code** (zero callers) — and a phase (`credit_runtime`) registered only on it never ran in production (fixed: #464). Wired the manager to drive all 9 `_initialize_*` phases via `run_phase` at their existing sites (#466 readiness, #477 wiring). The manager now owns the phase contract (order/criticality/dependency-skip/timing); a guard test asserts no raw `_initialize_*` call site remains, so a phase can't be silently lost again. Grouped design was revised to per-phase in-place after implementation showed the phases are finely interleaved with construction.
- **Evidence**:
  - `victor/agent/orchestrator.py` = 4,704 LOC; `grep -c "_initialize_"` = **13**; service-delegation refs `self._<svc>_service` = **51**.
  - Docstring `victor/agent/orchestrator.py:438-445` promises delegation "without containing business logic".
  - `_initialize_provider_runtime` (L459) and `_initialize_memory_runtime` (L479) embed construction logic directly.
- **Rationale**: A facade that also owns 13 lifecycle phases is a lifecycle owner wearing a delegation mask. `InitializationPhaseManager` already exists (per CLAUDE.md) — the orchestrator should not retain `_initialize_*` bodies.
- **Action**: Move each `_initialize_*` into `OrchestratorServiceProvider` / `InitializationPhaseManager`; orchestrator keeps only wiring.
- **Effort**: Medium. **Impact**: High (testability, merge-conflict reduction).

### F-003 · Compaction logic fragmented across 7 files, no unified strategy protocol — `HM`
- **Status**: DOWNGRADED / largely INVALID — scoping (2026-07) **falsified the premise**. A `CompactionSummaryStrategy` Protocol **already exists** (`victor/agent/compaction_summarizer.py:32`) and every summarizer (rule_based/hybrid/llm/keyword/ledger-aware) implements it — so "no unified strategy protocol" is false. The finding also undercounted (actually **9 files / ~5,718 LOC**; the "strategies" `adaptive_compaction`/`emergency_compaction`/`continuation_bonus`/`hierarchy` are helpers, not strategies). `context_compactor.py` (1,827 LOC) IS a real god-object but is a *separate* subsystem from the router, works, and isn't a proven bottleneck. **Action taken**: added a hotspot ratchet on `context_compactor.py` (1827) to prevent further god-object growth; decomposition deferred as low-ROI. The one real spun-off finding → **F-015** (orphaned router). No FEP.
- **Evidence**:
  - `context_compactor.py` (1,827), `compaction_router.py` (721), `compaction_hybrid.py` (593), `compaction_rule_based.py` (563), `compaction_summarizer.py` (148), `compaction_continuation_bonus.py` (187), `compaction_hierarchy.py` (168) — **4,207 LOC total**.
- **Rationale**: `router` + `hybrid` + `rule_based` + `summarizer` strongly implies a strategy pattern never unified behind one `CompactionStrategy` interface. Each new strategy added a file rather than a class. Compaction is a hot latency path.
- **Action**: Define `CompactionStrategy` protocol; `router` selects; fold god-object `context_compactor.py`.
- **Effort**: Medium-High. **Impact**: High.

### F-004 · tool_selection split into 8 files but core still 2,878 LOC — `HM`
- **Status**: DONE (package-ify scope; parent-extraction deferred) — scoping **corrected the premise**: the 6 satellites are *genuine, tested delegations* (each called at a real pipeline point, each with a dedicated unit test), **not** false splits — folding them back would create a ~3,300-LOC monolith. Instead (#484): moved the flat `tool_selection*.py` files into a `victor/agent/tool_selection/` package (parent → `selector.py`, git renames preserved) with an `__init__` re-export shim (public API unchanged), and added a hotspot ratchet on `selector.py` (2882) so it can't grow. The parent's big undelegated methods (`_filter_tools_for_stage` 336 LOC, `get_adaptive_threshold` 135 LOC) were **deliberately not extracted** — the file isn't hotspot-critical and works; ~25h of hot-path churn wasn't worth it.
- **Evidence**:
  - `tool_selection.py` (2,878) + `tool_selection_assembler.py` (56), `_cache_key.py` (27), `_cache.py` (64), `_policy.py` (103), `_postprocessor.py` (144), `_recorder.py` (21), `tool_selector_factory.py` (351) = **3,747 LOC**.
- **Rationale**: 7 satellites averaging <150 LOC are too-granular splits where the 2,878-line parent still owns the logic — worst of both worlds. Either commit to a `tool_selection/` package with coherent stages (policy → cache → selector → postprocess → record), or fold satellites back.
- **Action**: Promote to `tool_selection/` package OR consolidate; ensure core shrinks proportionally.
- **Effort**: Medium. **Impact**: Medium-High.

### F-011 · Same-name enum collisions across domains (`ApprovalMode`, `AgentMode`) — `HL`
- **Status**: DONE (PR #447) — renamed the lower-usage variant of each pair: `config.py` `AgentMode`→`AgentLifecycleMode` (1 importer) and `safety.py` `ApprovalMode`→`WriteApprovalMode` (0 external importers). Members/values preserved; canonical `mode_controller.AgentMode` / `tool_approval.ApprovalMode` untouched.
- **Evidence (verified 2026-07)**:
  - `ApprovalMode` defined twice with **incompatible members**: `victor/agent/safety.py:97` (`OFF`/… write-operation approval) vs `victor/agent/tool_approval.py:21` (`AUTO`/`DANGEROUS`/`ALL`, tool-approval workflow).
  - `AgentMode` defined twice: `victor/agent/mode_controller.py:32` (**canonical**, 12 importers — coding workflow modes) vs `victor/agent/config.py:65` (**1 importer** — agent lifecycle). Different concepts, same name.
- **Rationale**: Importing the wrong `ApprovalMode`/`AgentMode` type-checks fine but silently yields wrong behavior — a latent correctness footgun, **not dead code** (both variants have live consumers, so neither can simply be deleted). This is why the duplication audit's "delete the obsolete one" recommendation was rejected.
- **Action**: Rename by domain (e.g. `WriteApprovalMode` vs `ToolApprovalMode`; `AgentLifecycleMode` vs `AgentMode`); migrate the minority-importer side first (`config.py` `AgentMode`: 1 site). Behavior-touching → own reviewed PR, not a sweep.
- **Effort**: Low-Medium. **Impact**: Medium-High (silent-miswire prevention).

---

## TIER 2 — High Impact, High Effort

### F-005 · 23 `*_runtime.py` service twins beside 6 canonical services — `MH`
- **Status**: DOWNGRADED / premise OVER-STATED — batch re-triage (2026-07) classified all 23. They are **not** service twins: the runtimes are the extracted implementations the six canonical services *delegate into* (the intended decomposition, not a parallel layer), and only **2** are thin shells worth folding — the other 21 own distinct, live behavior. There is no `chat_service.py`↔`chat_stream_runtime.py` "shell vs runner" duplication of the kind alleged; the `_service`/`_runtime` split is the deliberate facade→implementation boundary. **Action taken**: none — no anti-pattern to fix. Corrected premise recorded so this isn't re-scoped.
- **Original evidence (context)**: `victor/agent/services/*_runtime.py` = 23 files.
- **Residual (low-ROI)**: 2 thin-shell runtimes could be inlined; not worth a PR on its own — fold opportunistically if either file is touched.

### F-006 · 164 singleton files vs guard ceiling of 68 — `MH`
- **Status**: DOWNGRADED / premise OVER-STATED — the "164 vs 68" gap was a **grep artifact**, not a guard breach. The re-triage found the guard (`test_singleton_guard.py`) already scopes its count to genuine stateful singletons and passes (**73/74** in-scope, not 164); the 164 is the broad grep pattern matching `_NATIVE_AVAILABLE` module constants, `__new__` overrides on dataclasses, and local `get_instance` helpers that aren't process-wide singletons. The real singleton population is already inside the guard's ceiling. **Action taken**: none — no ceiling to tighten; the guard already contains the class. Corrected premise recorded.
- **Original evidence (context)**: broad grep = 164 files; guard ceiling = 68 (different denominators).

### F-007 · 49-provider / 79-tool flat registries, no taxonomy — `MH`
- **Status**: DOWNGRADED / premise OVER-STATED — the "flat registries scale poorly / force import-all" rationale is false here. Providers are **lazily loaded** (double-checked-locking registry; nothing imports all 49), and tools already auto-register through `SharedToolRegistry` with an existing category taxonomy — so discoverability and registry-awareness are solved without a directory move. A physical re-org into `providers/cloud|local/` would churn 128 files + every import path for no functional gain and real merge cost. **Action taken**: none — the scaling problem the finding assumes doesn't exist. Corrected premise recorded.
- **Original evidence (context)**: 49 provider files, 79 tool files, flat dirs (but lazy-loaded + registry-categorized).

### F-008 · Conversation `store*` sharded into 6 files, `store.py` still 2,855 LOC — `MH`
- **Status**: DOWNGRADED / premise OVER-STATED — the shards are **not** premature decomposition of one store: `store_messages/_schema/_session/_trace.py` are distinct repository concerns (message CRUD, DDL/migrations, session lifecycle, trace persistence) with their own boundaries, and `store.py` is the coordinating facade over them, not a monolith that "failed to shrink." The finding inferred a problem from the file count + the facade's LOC without checking that each shard owns a cohesive concern. Consolidating back into one package would re-merge the boundaries the split created. **Action taken**: none. Corrected premise recorded.
- **Original evidence (context)**: 6 `store*` files; `conversation.store` imported by 21 files.

### F-012 · Validation/metrics types fragmented into divergent same-name variants — `MH`
- **Status**: DONE — [FEP-0014](../../feps/fep-0014-canonical-validation-metrics-contracts.md) **Implemented**. Phase 1 (#453 canonical `ValidationSeverity`/`ValidationResult` + `severity_rank` + `MetricsCollectorProtocol` + guard). Phase 2: #459 (severity re-export ×3), #460 (the four divergent `ValidationResult`s **renamed** to distinct types — ground truth showed merge would lose domain data), #458 (5 collectors conform via adapter methods), #457 (`pickle_cache` rename). Guard allowlist empty for both names. Phase 3 absorbed (re-exports are the end state; results renamed, no shims). F-012 footgun eliminated.
- **Evidence (verified 2026-07)**:
  - `ValidationSeverity` × **4**: `config/validation.py:26`, `core/validation.py:82`, `framework/middleware.py:89`, `framework/capabilities/validation.py:53`. Three are `{ERROR,WARNING,INFO}`; **`framework/middleware.py:89` uniquely adds `CRITICAL`** → a severity comparison silently means different things by layer.
  - `ValidationResult` × **5** with incompatible fields: `tools/tool_call_validator.py:17`, `config/connection_validation.py:54`, `workflows/protocols.py:759` (nested), `framework/requirement_validator.py:88`, `framework/capabilities/validation.py:62`.
  - `MetricsCollector` × **5**, no shared protocol: `observability/metrics.py:607`, `integrations/api/event_bridge.py:139`, `agent/metrics_collector.py:162`, `experiments/ab_testing/metrics.py:45`, `framework/observability/metrics.py:1124`.
- **Rationale**: Same names, no canonical, no shared protocol → cannot be treated polymorphically, and the `CRITICAL`-only variant is a genuine cross-layer incompatibility. This is real design fragmentation but **not clutter** — every impl has live consumers and the framework/core ones are public surface (FEP territory).
- **Action**: Define canonical `ValidationSeverity`/`ValidationResult` in `core` and a `MetricsCollectorProtocol` in `core/protocols.py`; deprecate variants behind them. Requires a design/FEP pass — do not blind-merge.
- **Effort**: High. **Impact**: High (correctness + polymorphism).

---

## TIER 3 — Quick Wins

### F-009 · Root-level stray test/verify scripts — `ML`
- **Status**: DONE (F-009a/b/c done; F-009d done — dead verify script deleted)
- **Evidence**:
  - Root files: `test_performance_improvements.py`, `test_planning_integration.py`, `test_shell_limits.py`, `verify_phase4_implementation.sh`, `verify_service_layer_default.py`.
  - `grep` across `Makefile`, `pyproject.toml`, `.github/`, `scripts/`, `mkdocs.yml` = **0 references**.
  - `git ls-files` = only `verify_phase4_implementation.sh`, `verify_service_layer_default.py` were tracked; the 3 `test_*.py` were untracked.
  - `pyproject.toml:661` `testpaths = ["tests"]` meant the 3 root `test_*.py` were **never collected by pytest** → orphaned/dead.
  - `scripts/` already has 55 scripts.
- **Sub-findings discovered during execution**:
  - **F-009a (DONE)**: `verify_service_layer_default.py:15` referenced `FeatureFlag.USE_SERVICE_LAYER_FOR_AGENT` which **no longer exists** in `victor/core/feature_flags.py` (removed in W3 cleanup). Relocated to `scripts/verify_service_layer_default.py` via `git mv` (history preserved). Dead-flag reference rewrite → F-009d.
  - **F-009b (DONE)**: `verify_phase4_implementation.sh` relocated to `scripts/` via `git mv`.
  - **F-009c (DONE)**: `test_shell_limits.py` had **unique coverage** (no canonical test checks `stdout_limit`/`stderr_limit`). Rewrote as `tests/unit/tools/test_shell_limits.py` — removed hardcoded `sys.path.insert(0, "/Users/vijaysingh/code/codingagent")` hack + `__main__` runner; split into 5 focused async tests. **5 passed in 1.41s.**
  - `test_performance_improvements.py` DELETED — timeout-config coverage already exists in `tests/unit/core/test_timeouts.py`; it also hardcoded `/Users/vijaysingh/code/codingagent` paths and used a `FileWatcherRegistry.get_instance()` singleton.
  - `test_planning_integration.py` DELETED — imported `from victor.agent.coordinators.planning_coordinator import PlanningCoordinator`, a **removed module** (`victor/agent/coordinators/planning_coordinator.py` does not exist); `PlanningConfig` now lives in `victor/config/groups/agent_config.py`. Dead-on-collection.
- **F-009d (DONE)**: Deleted `scripts/verify_service_layer_default.py` — its sole purpose was verifying the `USE_SERVICE_LAYER_FOR_AGENT` flag, which no longer exists (service layer is now unconditional), so the script was dead and would crash on import. Only the backlog referenced it. The suggested `.gitignore` guard for root `test_*.py`/`verify_*` was declined: a broad root glob risks ignoring legitimate files, and the strays were already relocated (F-009a/b/c).
- **Effort**: Low. **Impact**: Low-Medium.
### F-010 · 306 files carry legacy/deprecation markers — triage needed — `ML`
- **Status**: DOWNGRADED / premise OVER-STATED — the "306 files" (re-measured ~612 on the broader grep) is **~85% noise**: the pattern matches the word "legacy" in comments/docstrings describing *handled* backward-compat, `legacy=` kwargs that are live API, and deprecation *machinery* (the `@deprecated` decorator's own definition + tests). Spot-audit found no meaningful population of abandoned-but-not-removed surface that isn't already tracked by a specific finding (F-013 export shim, F-014 backoff dupes, F-015 done). There is no undifferentiated "dead surface" backlog to reap. **Action taken**: none — a blanket triage would burn effort re-confirming live code. Corrected premise recorded; real dead code is caught per-subsystem, not by marker grep.
- **Original evidence (context)**: marker grep = 306 files (~16% of source), dominated by compat-describing prose.

### F-013 · `framework/step_handlers.py` exports internal-only symbols — FEP-gated — `ML`
- **Status**: PHASE 1 DONE — [FEP-0015](../../feps/fep-0015-trim-internal-framework-exports.md) Phase 1 landed (#454: unexported both symbols, `_ExtensionHandler` rename, `__getattr__` deprecation shim, guard). **Phase 2 (remove shim) is release-gated** — deferred until one minor release elapses so out-of-tree importers had their deprecation window. FEP stays `Accepted` until then.
- **Evidence (verified 2026-07)**:
  - `victor/framework/step_handlers.py:2629` `__all__` includes `CapabilityConfigStepHandler` (class L968) and `ExtensionHandler` (class L2047), but **neither is imported anywhere outside this module**.
  - `ExtensionHandler` is heavily used *internally* (instantiated L2346–2361), so it is **not dead** — only its `__all__` export is unused.
- **Rationale**: Exporting an internal implementation detail widens the framework public API with zero consumers. But `framework/` `__all__` **is** the public API — removing/renaming needs a FEP per CLAUDE.md. Flag-only; do not delete.
- **Action**: Via FEP — drop from `__all__` (optionally rename `ExtensionHandler`→`_ExtensionHandler`) once confirmed no external consumer relies on the lazy import.
- **Effort**: Low (mechanical) + FEP overhead. **Impact**: Low-Medium.

### F-014 · Ad-hoc retry/backoff + pickle-cache logic bypasses existing canonicals — `ML`
- **Status**: DONE (partial, conservative) — F-014a: PR #448; F-014b: PR #449. Migrated where behavior is provably identical; behaviorally-divergent sites deliberately left (documented below).
- **Evidence (verified 2026-07)**:
  - Canonical retry EXISTS: `victor/core/retry.py:156` `ExponentialBackoffStrategy` (+ `with_retry` L498). Yet 3 sites reimplement `2 ** attempt` backoff inline: `agent/subagents/base.py:533`, `workflows/batch_executor.py:599` & `:663`, `storage/embeddings/service.py:284`.
  - Pickle load-validate-save duplicated across **4** files: `tools/semantic_selector.py`, `storage/embeddings/collections.py`, `agent/prompt_corpus_registry.py`, `agent/services/decision_cache.py`.
- **Rationale**: Real duplication, but each copy has subtly different jitter/exception/validation semantics — consolidation is behavior-sensitive, not a mechanical de-dup. Medium ROI.
- **Action**: Migrate the 3 backoff sites to `ExponentialBackoffStrategy` (audit jitter/exception handling per-site); extract `validate_and_load_pickle_cache(path, validators)` for the 4 cache sites. One focused PR each; verify behavior parity.
- **Effort**: Medium. **Impact**: Medium.

---

### F-015 · Orphaned hybrid-compaction subsystem — built but never wired to production — DONE (PR #487)
- **Status**: DONE — **deleted** (2026-07). Scoping proved the whole hybrid-compaction slice, not just the router, was fully unwired: the router is instantiated only in tests, `ConversationController`'s `compaction_router` param was never supplied in production, `context_compactor.py` uses `LLMCompactionSummarizer` directly, and the backing config (`CompactionStrategySettings`/`CompactionFeatureFlags`) had **zero** production consumers — read only by the dead cluster and its tests. Unlike F-002 (wiring fixed a live bug), nothing depended on this, so delete was correct over wire.
- **Removed (~2,082 LOC of feature code + tests)**: `victor/agent/compaction_router.py` (721), `compaction_rule_based.py` (563), `compaction_hybrid.py` (593), `victor/config/compaction_strategy_settings.py` (205), plus `tests/unit/agent/test_compaction_{router,hybrid,rule_based}.py` and `tests/integration/agent/test_hybrid_compaction.py`.
- **Surgery**: excised the dead `compaction_router` param/attribute/TYPE_CHECKING import and the never-reached router branch in `conversation/controller.py::_generate_compaction_summary` (legacy summarizer + keyword fallback retained); stripped the dead `strategy`/`compaction_feature_flags` fields + 3 getters from `config/compaction_settings.py` (legacy + `adaptive_threshold` fields kept); refreshed a stale field-collision comment in `config/settings.py`.
- **Noted (not actioned — out of scope)**: the *remaining* `CompactionSettings` legacy fields (`compaction_enabled`, `compaction_preserve_recent`, `compaction_max_estimated_tokens`, `compaction_auto_compact`) also have no production consumers — a separate config-hygiene question, left intact to keep this PR to the approved hybrid-compaction feature.

## COMPLETED THIS CYCLE (2026-07)

### F-C1 · Provider capability surface collapsed to `supports_*()` methods — DONE (PR #442)
- Removed 3 unused `@runtime_checkable` protocols + 5 `is_*_provider`/`has_*` helpers from `victor/providers/base.py`; added the missing `BaseProvider.supports_vision()` default (closed a Liskov/ISP gap where `provider.supports_vision()` could `AttributeError`); retargeted callers/tests/docs; de-flaked `test_score_resources_local_without_gpu`. Net **−284 LOC**. Verified: 2523 provider/service tests green.

### F-C2 · Verified-dead code removed — DONE (PR #443)
- Deleted orphan module `victor/core/typed_models.py` (12 types, zero importers), the dead `StreamingProvider`/`StreamChatProvider` protocols in `providers/stream_adapter.py`, and the unused `ModeConfigSchema`/`ModeDefinitionSchema`/`VerticalModeConfigSchema` + `validate_*_dict` cluster in `core/validation.py`. Net **−566 LOC**. Verified: full-suite collection-check clean, 173 targeted tests green.
- **Provenance note**: Originated from a 4-agent duplication audit whose **majority of raw findings were rejected on verification** — e.g. the "414 `to_dict` copies → mixin" was a mischaracterization (mostly legitimately-distinct types), and a claim that a test imports the `stream_adapter` protocols was false. Only independently-verified-dead items were removed; the surviving design issues became F-011..F-014.

### F-C3 · F-011 enum-collision renames — DONE (PR #447)
- `config.py` `AgentMode`→`AgentLifecycleMode`, `safety.py` `ApprovalMode`→`WriteApprovalMode` (lower-usage variant of each pair). Pure rename; members preserved; canonical variants untouched. Implemented via a worktree-isolated parallel agent, diff-reviewed, CI-green.

### F-C4 · F-014 backoff + pickle-cache consolidation — DONE partial (PRs #448, #449)
- **F-014a (#448)**: added pure `compute_backoff_delay()` to `core/retry.py`; migrated 3 inline sites (`subagents/base.py`, `batch_executor.py` ×2) with proven numeric equivalence. **Left** `storage/embeddings/service.py:284` — multiplicative jitter `[0.75,1.25]` the additive-jitter helper cannot reproduce.
- **F-014b (#449)**: extracted `victor/core/pickle_cache.py` (validator-based load/validate/save, +8 tests); migrated `semantic_selector.py` and `collections.py` preserving exact validation order and delete-vs-keep flags. **Left** `prompt_corpus_registry.py` (never deletes on failure) and `decision_cache.py` (bare-dict pickle, no metadata wrapper) — helper cannot express their semantics.
- Both implemented via worktree-isolated parallel agents, independently diff-reviewed, CI-green.

### Blocker cleared this cycle — vertical CI unblock (PR #451)
- An external main→develop sync (PR #446) deleted core `victor/storage/vector_stores/chromadb_provider.py` but left a vertical test importing it → `test-verticals` collection error blocked **all** PRs into develop. Fixed by removing the stale import + orphaned `TestChromaDBProvider` (PR #450 independently removed chromadb from verticals around the same time).

---

## F-016 · Call-graph audit — silently-broken / unwired capabilities (2026-07)

**Why this audit exists.** The two highest-value finds of the whole review — F-002 (`credit_runtime`: an opt-in RL feature that silently no-op'd because its runtime was registered only on dead code) and F-015 (a config-backed hybrid-compaction subsystem with zero consumers) — were both the *same class*: **a capability that is constructed / registered / config-gated but never actually invoked in a production path.** Count-based audits structurally miss this class (the code exists and looks wired). So we ran a dedicated **call-graph** hunt: three parallel verify-first agents (config opt-ins that no code reads; components registered but never invoked; optional constructor params never supplied → dead branches). Each was required to trace the call graph and reject any finding with a live consumer. The discipline paid off — agents self-rejected ~3 false positives (missed aliasing / live alternate wiring), and every disposition below was independently re-verified before action.

**Meta:** unlike the original count-based backlog (majority over-stated), this call-graph method landed **real** findings. This is the recommended lens for future dead-code / dead-feature sweeps.

### DELETED — verified-dead, merged
- **F-016a · `WorkflowOptimizationComponents` bundle — DONE (PR #494).** A 6-object optimization bundle (`task_completion_detector`, `read_cache`, `time_aware_executor`, `thinking_detector`, `resource_manager`, `mode_completion_criteria`) constructed on *every* orchestrator init and read by nothing (the only `.workflow_optimization` reference was an unimplemented protocol stub). Removed the bundle + its `OrchestratorComponents` field + factory method + bootstrapper assignment + protocol stub + 6 now-dead wrapper builders in `coordination_builders.py`. Underlying component classes KEPT (alive via other paths). −333 LOC, deletion-only.
- **F-016b · `HybridDecisionService` cluster — DONE (PR #493).** Built only by `ExtendedModelSelectorLearner`, which is referenced nowhere (dead parent). Deleted the service + `HybridDecisionServiceConfig`/`HybridMetrics`, its dead parent, and its exclusively-owned `ConfidenceCalibrator` + `DecisionCache`, plus exclusive tests. Preserved shared extended-learner test files (surgical case removal) and the independent `learn_confidence_threshold` code. ~−4,325 LOC.
- **F-016c · Team credit-attribution mixins — DONE (PR #489).** `CreditAssignmentMixin` / `CreditAwareTeamCoordinator` — the production `UnifiedTeamCoordinator` inherits `ObservabilityMixin, RLMixin`, never these; not even re-exported. Deleted both + exclusive test + example. (Distinct from the LIVE framework `CreditTrackingService`, which was left untouched.)
- **F-016d · Unregistered capability step-handlers — DONE (PR #492).** `CapabilityNegotiationStepHandler` / `CapabilityAwareToolStepHandler` — never in `StepHandlerRegistry.default()`'s 10 handlers, never supplied as `custom_handlers`; module imported nowhere. Deleted the whole file (317 LOC). Live `negotiate_capabilities()` untouched. Not a FEP (no public `__all__` membership).

### IN PROGRESS
- **F-016e · Tool-config mis-wires — PR open (`fix/tool-config-miswires`).** (1) `generic_result_cache_enabled` (and siblings) read *flat* off `Settings` at `tool_builders.py:176` but the field is only nested (`tools.generic_result_cache_enabled`) — verified `getattr(Settings(), 'generic_result_cache_enabled')` → MISSING, so the flag can never take effect; **fix** = read the nested value. (2) `deduplication_strict_mode` → `DeduplicationConfig(strict_mode=...)` but `strict_mode` is never read → **delete** the false affordance. (3) `enable_provider_optimization` flag + `get_tools_with_levels_provider_optimized()` method both have zero readers/callers → **delete**.

### QUEUED — wire (real capabilities silently off; delete-vs-wire per F-002/F-015 precedent)
- **F-016f · Credit-assignment *feedback* loop is dead.** The recording half now works (F-002 fix), but `assign_turn_credit()` has **zero production callers** and `auto_assign_at_turn_boundary: bool = True` is **defined but never read** → `_turn_count` never advances → `generate_tool_guidance()` (wired live into `prompt_pipeline.py:905`) always trips its `_turn_count < 3` guard and returns `None`. The online tool-guidance loop is inert despite defaulting on. **Independently verified.** Sibling of F-002 → wire `assign_turn_credit()` at the turn boundary (honoring the flag) or delete the feedback half.
- **F-016g · Post-switch hooks subsystem never dispatched.** Four hooks (`ExplorationSettingsHook`, `PromptBuilderHook`, `SystemPromptHook`, `ToolBudgetHook`) + `create_standard_hooks()` are referenced nowhere; the live provider-switch path only syncs state + an optional callback, so a mid-session model switch never refreshes prompt / system-prompt / tool-budget. Correctness gap → wire into the switch path or delete.

### QUEUED — verify then likely wire
- **F-016h · `RecoveryService` coordinator/integration + `ProviderService.health_checker` unsupplied.** Agent-reported: the *primary* delegation paths in `RecoveryService` (~7 methods) are guarded by `_recovery_coordinator`/`_recovery_integration`, but the sole prod bind omits the coordinator and hardcodes `recovery_integration = None`; `ProviderService.health_checker` is never supplied so health-gated provider switching is off. Verify intentional-vs-bug before acting.

### FOLLOW-UP THREADS (surfaced mid-audit, not yet scoped)
- **`Extended*Learner` family:** deleting `ExtendedModelSelectorLearner` (F-016b) revealed `ExtendedModeTransitionLearner` and `ExtendedToolSelectorLearner` are *also* consumer-less — same dead-parent pattern; scope for a follow-up deletion.
- **Predictive-tools `FeatureFlagSettings` (10 flags):** `create_step_aware_selector()` has zero callers; the live `StepAwareToolSelector` is built without the predictive flags, so `enable_predictive_tools`/`enable_tool_predictor`/`enable_cooccurrence_tracking`/`enable_tool_preloading` + helpers are inert. Delete the flags or wire the predictive selector — a delete-vs-wire decision.
- **Provider-pool inert:** `use_provider_pooling` builds a `ProviderPool` whose only accessor has zero callers; the pool is only `.shutdown()` at teardown, never used to acquire/route. Delete or wire.
- **Unread config groups:** graph-profiling settings (`search.graph.enable_profiling` + siblings) and the remaining `CompactionSettings` legacy fields (noted in F-015) have no production readers — config hygiene.

---

## INVALIDATED FINDINGS (kept for audit trail)

### F-❌-A · ~~Delete ghost `victor_sdk/` package~~ — INVALIDATED
- **Original claim**: `victor_sdk/` is a 1-file ghost that should be deleted.
- **Correction (verified 2026-04)**: `victor_sdk` is an **intentional backward-compatibility shim**, not a ghost.
  - `victor-contracts/victor_sdk/__init__.py:15-22`: *"Backward-compatibility shim: ``victor_sdk`` was renamed to ``victor_contracts``."* — re-exports via `_VictorSdkRedirect` meta-path finder so `isinstance`/identity stay consistent.
  - `victor-contracts/pyproject.toml:72-73`: explicitly documented + packaged: `# victor_sdk* is a deprecation shim re-exporting victor_contracts (old package name).` / `include = ["victor_contracts*", "victor_sdk*"]`.
- **Disposition**: **WONTFIX / DO NOT DELETE.** Deleting would break external consumers pinned to the old import name. Keep until the documented future removal cycle.

### F-❌-B · ~~Committed MagicMock dirs~~ — INVALIDATED
- **Original claim**: `MagicMock/` and `rust/MagicMock/` dirs are committed artifacts.
- **Correction**: `git ls-files MagicMock/ rust/MagicMock/` = **0 tracked files**. They are local untracked test-run artifacts. **Not a real issue.** Optionally `.gitignore` them.

---

## Execution Order (Impact ÷ Effort)

| # | Finding | Effort | Impact |
|---|---------|--------|--------|
| 1 | F-001 doc-count drift + CI guard | Low | Med |
| 2 | F-009 relocate root stray scripts | Low | Low-Med |
| 3 | F-002 orchestrator `_initialize_*` → container | Med | **High** |
| 4 | F-003 unify compaction behind `CompactionStrategy` | Med-Hi | **High** |
| 5 | F-004 tool_selection package consolidation | Med | Med-Hi |
| ~~6~~ | ~~F-005 classify `*_runtime.py` twins~~ — ⤵️ DOWNGRADED (premise over-stated; runtimes are the intended impl layer) | High | High |
| ~~7~~ | ~~F-006 singleton triage + tighten guard~~ — ⤵️ DOWNGRADED (164 was grep noise; guard already contains real singletons) | High | High |
| ~~8~~ | ~~F-007 provider/tool taxonomy~~ — ⤵️ DOWNGRADED (lazy-loaded + registry-categorized; no scaling problem) | Med | Med |
| ~~9~~ | ~~F-008 conversation store disposition~~ — ⤵️ DOWNGRADED (shards are cohesive repos, not premature split) | Med | Med |
| ~~10~~ | ~~F-010 legacy-marker triage~~ — ⤵️ DOWNGRADED (~85% grep noise; no undifferentiated dead surface) | Med | Med |
| ~~15~~ | ~~F-015 delete orphaned hybrid-compaction~~ — ✅ DONE (~2,082 LOC removed) | Low | Low-Med |
| ~~11~~ | ~~F-011 rename colliding enums~~ — ✅ DONE (#447) | Low-Med | **Med-High** |
| ~~12~~ | ~~F-014 backoff/pickle-cache consolidation~~ — ✅ DONE partial (#448, #449) | Med | Med |
| 13 | F-013 `step_handlers` `__all__` cleanup — ⏳ FEP-0015 Phase 1 DONE (#454); Phase 2 release-gated | Low+FEP | Low-Med |
| ~~14~~ | ~~F-012 unify validation/metrics types~~ — ✅ DONE (FEP-0014 implemented, #453/#457/#458/#459/#460) | High | High |

---

## Changelog
- **2026-04 review**: Initial findings F-001..F-010 written; F-❌-A (victor_sdk) and F-❌-B (MagicMock) recorded as INVALIDATED after verification.
- **F-009 execution (2026-04):** F-009a/b done via `git mv` to `scripts/`; F-009c recovered 5 shell-limit tests into `tests/unit/tools/test_shell_limits.py` (all passing); deleted 2 dead/duplicate root tests (`test_performance_improvements.py`, `test_planning_integration.py` — the latter imported a removed `planning_coordinator` module). F-009d (rewrite dead `USE_SERVICE_LAYER_FOR_AGENT` verify script) still OPEN.
- **F-001 execution (2026-07):** Extended `scripts/ci/check_docs_drift.py` EXTRA_FILES to scan CLAUDE.md/.victor/init.md/AGENTS.md (the exact files F-001 flagged); added regression test; verified bidirectionally. Provider/tool-module counts confirmed already correct (original evidence misread raw file count vs adapter count).
- **2026-07 audit cycle:** Landed F-C1 (PR #442, provider capability-surface collapse) and F-C2 (PR #443, verified-dead-code removal) — ~850 LOC removed, both CI-green. Added F-011..F-014 from the same 4-agent duplication audit, each recorded only after **independently verifying** its evidence (the majority of the audit's raw findings were rejected as unverified or mischaracterized). F-012/F-013 are FEP-gated (framework/core public surface); F-011 is a correctness footgun (colliding enums with live consumers, so rename — not delete).
- **F-011/F-014 execution (2026-07):** Implemented in parallel via three worktree-isolated agents, each diff-reviewed independently before PR. F-011 → PR #447 (enum renames); F-014a → PR #448 (`compute_backoff_delay` helper, 3/4 sites migrated); F-014b → PR #449 (`core/pickle_cache.py` helper + tests, 2/4 sites migrated). Divergent-semantics sites deliberately skipped and documented (F-C3/F-C4). Mid-flight, an external main-sync (PR #446) broke `test-verticals` by deleting a core module a vertical test imported; fixed via PR #451 to unblock the cascade. All four squash-merged CI-green.
- **F-012/F-013 FEP drafts (2026-07):** Authored FEP-0014 (canonical validation/metrics contracts) and FEP-0015 (framework export hygiene) → PR #452, awaiting review. Backlog statuses moved OPEN → FEP DRAFTED; implementation gated on FEP acceptance.
- **F-012/F-013 acceptance + implementation (2026-07):** Both FEPs accepted (#452). FEP-0015 Phase 1 landed (#454); its Phase 2 (shim removal) is release-gated. FEP-0014 **fully implemented** across Phase 1 (#453) + Phase 2 (#457/#458/#459/#460, doc #462) — F-012 → DONE. Key course-correction during Phase 2: the four divergent `ValidationResult`s carry domain-specific fields, so they were **renamed to distinct types** (not lossy-merged into the canonical) — the accepted plan's "Tier A migrate" was revised to "rename" after ground-truthing (the F-011 lesson). Both guard-allowlist sets emptied; the guard now rejects any new duplicate. Also fixed a repo-wide CI flake mid-cascade: `hotspot-size-guard` had `timeout-minutes: 2` on a 2min+ cold-cache install → intermittent `cancelled` → false red (#455, raised to 10).

- **F-002 execution (2026-07):** FEP-0016 accepted (#465) + implemented. Discovery: the `InitializationPhaseManager` that would centralize init was **dead code** (never wired), and `credit_runtime` (registered only on it) never ran in production — an opt-in RL feature silently broken; fixed by #464. Wired the manager to drive all 9 phases via `run_phase` at their existing sites (#466 readiness, #477 wiring), with a guard test so no phase can be lost again. Grouped design revised to per-phase in-place when implementation showed the phases are finely interleaved with construction; CI caught two real gaps (a bare test orchestrator; a hotspot-cap overrun) that local async-env noise had masked — both fixed before merge.

- **F-009d execution (2026-07):** Deleted the dead `scripts/verify_service_layer_default.py` (verified only the removed `USE_SERVICE_LAYER_FOR_AGENT` flag → crashes on import; service layer is now unconditional). F-009 → DONE.
- **F-004 execution (2026-07):** Package-ified tool_selection (#484); premise corrected (satellites are sound). Parent-method extraction deferred as low-ROI on non-hotspot working code.

- **F-003 downgrade + F-015 spin-off (2026-07):** F-003 scoping falsified its premise (a CompactionSummaryStrategy protocol already exists; the router is orphaned; undercounted files). Downgraded F-003 and added a hotspot ratchet on the 1827-LOC context_compactor god-object (growth-prevention; decomposition deferred as low-ROI). Spun off F-015 for the orphaned CompactionRouter (built-but-unwired, F-002-like). **Meta:** two consecutive originally-audited findings (F-004, F-003) were over-stated — the audit flagged 'fragmentation' from file/LOC counts without checking for existing abstractions; remaining OPEN items warrant the same skeptical scoping.

- **Batch re-triage of the final 6 + F-015 execution (2026-07):** Re-scoped the remaining backlog against ground-truth code. **Five of six were over-stated** and downgraded with corrected premises recorded (so they aren't re-scoped): F-005 (`*_runtime.py` are the intended impl layer the 6 services delegate into, not parallel twins; only 2 thin shells), F-006 (164 was a broad-grep artifact — the singleton guard already contains the real stateful singletons, 73/74 in-scope), F-007 (providers lazy-load + tools registry-categorize; the flat-registry scaling problem doesn't exist), F-008 (the `store_*` shards are cohesive repository concerns behind a facade, not a premature split), F-010 (~85% of the marker grep is compat-describing prose + deprecation machinery; no undifferentiated dead surface). **F-015 was the one real item → DELETED (~2,082 LOC):** verification showed the *entire* hybrid-compaction slice was unwired — router + rule_based + hybrid summarizers + the `CompactionStrategySettings`/`CompactionFeatureFlags` config (zero production consumers, read only by the dead cluster and its tests). Removed the 4 modules + 4 test files, excised the never-supplied `compaction_router` param/branch from `ConversationController`, and surgically stripped the dead strategy fields/getters from `CompactionSettings` (kept its live legacy + `adaptive_threshold` fields). Delete over wire because — unlike F-002's `credit_runtime` — nothing depended on it. **Meta:** the audit's failure mode is now unmistakable — it infers problems from file/LOC/grep counts without checking for the abstraction that already solves them; every count-based finding must be ground-truthed before action. The backlog now has **no OPEN high-ROI item** — remaining OPEN work is FEP-gated release timing (F-013 Phase 2) and documented low-ROI residue.

- **F-016 call-graph audit — the "built-but-never-invoked" class (2026-07):** Pivoted from the exhausted count-based backlog to a *call-graph* hunt for the exact class behind the two best finds (F-002, F-015): capabilities constructed/registered/config-gated but never invoked in production. Three parallel verify-first agents (dead config opt-ins / registered-but-uncalled / DI dead-end params). **Deleted four verified-dead subsystems** (all merged, all CI-green): `WorkflowOptimizationComponents` bundle (#494), `HybridDecisionService` cluster incl. dead-parent `ExtendedModelSelectorLearner` (#493, ~4.3k LOC), team credit-attribution mixins (#489), unregistered capability step-handlers (#492). **In progress:** tool-config mis-wires (F-016e). **Queued to wire** (real capabilities silently off): the credit-assignment *feedback* loop (F-016f — `assign_turn_credit()` never called → tool-guidance always None; independently verified as the F-002 sibling) and the post-switch hook subsystem (F-016g — model switches never refresh prompt/budget). **Follow-up threads:** the `Extended*Learner` family (2 more consumer-less siblings), inert predictive-tools flags, an unused provider-pool. **Meta:** unlike the count-based backlog (majority over-stated), the call-graph method landed real findings, and the agents self-rejected ~3 false positives — this is the recommended lens for future dead-code/dead-feature sweeps.
