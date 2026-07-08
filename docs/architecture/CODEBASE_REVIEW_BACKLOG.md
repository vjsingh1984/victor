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
- **Status**: OPEN
- **Evidence**:
  - `victor/agent/orchestrator.py` = 4,704 LOC; `grep -c "_initialize_"` = **13**; service-delegation refs `self._<svc>_service` = **51**.
  - Docstring `victor/agent/orchestrator.py:438-445` promises delegation "without containing business logic".
  - `_initialize_provider_runtime` (L459) and `_initialize_memory_runtime` (L479) embed construction logic directly.
- **Rationale**: A facade that also owns 13 lifecycle phases is a lifecycle owner wearing a delegation mask. `InitializationPhaseManager` already exists (per CLAUDE.md) — the orchestrator should not retain `_initialize_*` bodies.
- **Action**: Move each `_initialize_*` into `OrchestratorServiceProvider` / `InitializationPhaseManager`; orchestrator keeps only wiring.
- **Effort**: Medium. **Impact**: High (testability, merge-conflict reduction).

### F-003 · Compaction logic fragmented across 7 files, no unified strategy protocol — `HM`
- **Status**: OPEN
- **Evidence**:
  - `context_compactor.py` (1,827), `compaction_router.py` (721), `compaction_hybrid.py` (593), `compaction_rule_based.py` (563), `compaction_summarizer.py` (148), `compaction_continuation_bonus.py` (187), `compaction_hierarchy.py` (168) — **4,207 LOC total**.
- **Rationale**: `router` + `hybrid` + `rule_based` + `summarizer` strongly implies a strategy pattern never unified behind one `CompactionStrategy` interface. Each new strategy added a file rather than a class. Compaction is a hot latency path.
- **Action**: Define `CompactionStrategy` protocol; `router` selects; fold god-object `context_compactor.py`.
- **Effort**: Medium-High. **Impact**: High.

### F-004 · tool_selection split into 8 files but core still 2,878 LOC — `HM`
- **Status**: OPEN
- **Evidence**:
  - `tool_selection.py` (2,878) + `tool_selection_assembler.py` (56), `_cache_key.py` (27), `_cache.py` (64), `_policy.py` (103), `_postprocessor.py` (144), `_recorder.py` (21), `tool_selector_factory.py` (351) = **3,747 LOC**.
- **Rationale**: 7 satellites averaging <150 LOC are too-granular splits where the 2,878-line parent still owns the logic — worst of both worlds. Either commit to a `tool_selection/` package with coherent stages (policy → cache → selector → postprocess → record), or fold satellites back.
- **Action**: Promote to `tool_selection/` package OR consolidate; ensure core shrinks proportionally.
- **Effort**: Medium. **Impact**: Medium-High.

---

## TIER 2 — High Impact, High Effort

### F-005 · 23 `*_runtime.py` service twins beside 6 canonical services — `MH`
- **Status**: OPEN
- **Evidence**:
  - `victor/agent/services/*_runtime.py` = **23 files** (e.g. `chat_stream_runtime.py` beside `chat_service.py`, `planning_runtime.py`, `tool_execution_runtime.py`).
- **Rationale**: A service-first runtime should have 6 canonical services. The `_runtime` suffix implies "the part that actually runs" → the `*_service.py` may be a thin shell. This is the exact anti-pattern CLAUDE.md warns against: *"Do not create a new parallel abstraction layer inside `victor/agent`."*
- **Action**: Read 2–3 pairs to classify each twin (fold-into-service vs. promote-to-service); produce a per-file disposition table.
- **Effort**: High. **Impact**: High (ownership ambiguity).
- **Needs follow-up**: per-pair classification not yet done.

### F-006 · 164 singleton files vs guard ceiling of 68 — `MH`
- **Status**: OPEN
- **Evidence**:
  - `grep -rln "_instance.*Optional|__new__|get_instance|_get_instance" victor/ --include="*.py"` = **164 files**.
  - CLAUDE.md guard test `test_singleton_guard.py` caps singleton file count at **68**.
- **Rationale**: Singletons defeat the state-passed coordinator pattern (`ExecutionContext`, `ContextSnapshot`) the architecture promotes. 164 = 2.4× the stated ceiling.
- **Action**: Classify legitimate (`_NATIVE_AVAILABLE` module constants) vs. accidental (stateful `_instance`/`get_instance()`); tighten guard.
- **Effort**: High. **Impact**: High.
- **Needs follow-up**: classification of singleton kinds not yet done.

### F-007 · 49-provider / 79-tool flat registries, no taxonomy — `MH`
- **Status**: OPEN
- **Evidence**:
  - `ls victor/providers/*.py` = **49**; `ls victor/tools/*.py` = **79**.
  - `tools/verification/` subdir already proves namespace taxonomy works.
- **Rationale**: Flat registries scale poorly past ~30 entries; force import-all or import-by-string. Namespaced taxonomy (`providers/cloud/`, `providers/local/`, `tools/verification/`) improves discoverability + registry awareness.
- **Action**: Move into taxonomic dirs behind re-exports; update registry loader.
- **Effort**: Medium (mechanical + re-exports). **Impact**: Medium.

### F-008 · Conversation `store*` sharded into 6 files, `store.py` still 2,855 LOC — `MH`
- **Status**: OPEN
- **Evidence**:
  - `conversation/` has `store.py`, `store_messages.py`, `store_schema.py`, `store_session.py`, `store_trace.py`, `migrations.py` + `message_store.py`, `embedding_manager.py`, `session_manager.py`.
  - `from victor.agent.conversation.store` imported by **21 files**.
- **Rationale**: Splitting one SQLite store into 5 `store_*.py` shards may be premature decomposition — shards didn't reduce the core's size. Audit: 5 shards vs. 1 cohesive `store/` package with clear repository boundaries.
- **Action**: Decide per-shard disposition; either consolidate into `store/` package or justify each shard's boundary.
- **Effort**: Medium. **Impact**: Medium.

---

## TIER 3 — Quick Wins

### F-009 · Root-level stray test/verify scripts — `ML`
- **Status**: IN-PROGRESS (F-009a/b/c DONE; F-009d OPEN)
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
- **Action remaining (F-009d, OPEN)**: Rewrite `scripts/verify_service_layer_default.py` to drop the removed `USE_SERVICE_LAYER_FOR_AGENT` flag reference (or delete it — services are now mandatory, no flag). Add `.gitignore` guard for root `test_*.py`/`verify_*`.
- **Effort**: Low. **Impact**: Low-Medium.
### F-010 · 306 files carry legacy/deprecation markers — triage needed — `ML`
- **Status**: OPEN
- **Evidence**:
  - `grep -rln "DeprecationWarning|deprecated|@deprecated|legacy|DEPRECATED" victor/ --include="*.py"` = **306 files** (~16% of source).
- **Rationale**: 5 contrib verticals deprecate-by-design, but 306 is far beyond those 5. Triage: deprecated-but-kept-for-compat vs. abandoned-but-not-removed.
- **Action**: Produce a categorized report; remove dead surface area.
- **Effort**: Medium (triage). **Impact**: Medium.

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
| 6 | F-005 classify `*_runtime.py` twins | High | High |
| 7 | F-006 singleton triage + tighten guard | High | High |
| 8 | F-007 provider/tool taxonomy | Med | Med |
| 9 | F-008 conversation store disposition | Med | Med |
| 10 | F-010 legacy-marker triage | Med | Med |

---

## Changelog
- **2026-04 review**: Initial findings F-001..F-010 written; F-❌-A (victor_sdk) and F-❌-B (MagicMock) recorded as INVALIDATED after verification.
- **F-009 execution (2026-04):** F-009a/b done via `git mv` to `scripts/`; F-009c recovered 5 shell-limit tests into `tests/unit/tools/test_shell_limits.py` (all passing); deleted 2 dead/duplicate root tests (`test_performance_improvements.py`, `test_planning_integration.py` — the latter imported a removed `planning_coordinator` module). F-009d (rewrite dead `USE_SERVICE_LAYER_FOR_AGENT` verify script) still OPEN.
- **F-001 execution (2026-07):** Extended `scripts/ci/check_docs_drift.py` EXTRA_FILES to scan CLAUDE.md/.victor/init.md/AGENTS.md (the exact files F-001 flagged); added regression test; verified bidirectionally. Provider/tool-module counts confirmed already correct (original evidence misread raw file count vs adapter count).
