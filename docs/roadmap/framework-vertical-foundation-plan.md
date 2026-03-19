# Framework + Vertical Foundation Plan

Last updated: 2026-03-15
Owner: Architecture/Foundation
Status: Active

## Goal
Stabilize framework and vertical integration foundations before feature expansion.

This plan tracks implementation state across sessions and should be updated at the
end of each coding tranche.

## Priority Order
1. Contract correctness and architectural drift removal
2. Registry/adapter consistency on runtime paths
3. Reliability/performance hardening on hot paths
4. Extensibility cleanup and deprecation removals

## Work Items
| ID | Priority | Item | Status | Acceptance Criteria |
|---|---|---|---|---|
| FND-001 | P0 | Step-handler contract drift cleanup | Implemented + Verified | Default handler contract, dependencies, and docs are consistent and test-covered. |
| FND-002 | P0 | Registry-driven stream chunk conversion | Implemented + Verified | `stream_with_events()` converts chunk events through `EventRegistry` instead of manual mapping. |
| FND-007 | P0 | Step-handler dependency contract validation | Implemented + Verified | Sync/async integration paths surface missing `depends_on` handlers as warning/error and record contract status. |
| FND-008 | P0 | Validation/governance path repair | Implemented + Verified | Validation workflows trigger correctly, modified verticals are discovered from any changed file, repo metadata points to the canonical repo, and local `make lint` matches the clean mypy gate. |
| FND-003 | P1 | Legacy extension-registry path retirement | Implemented + Verified | Inactive legacy registry path removed or fully fenced with migration test coverage. |
| FND-004 | P1 | Entry-point/env-hash startup optimization | Implemented + Verified | Startup scanning avoids full package sweep on warm paths. |
| FND-005 | P1 | Observability drop-policy hardening | Not Started | Critical topics have durable/blocking policy and pressure alerts. |
| FND-009 | P1 | Audit-backed roadmap hygiene automation | Implemented + Verified | Workflow syntax, canonical-link drift, wrong-repo metadata, and stale roadmap claims are checked automatically in CI. |
| FND-006 | P2 | Generic capability promotion from vertical loaders | Not Started | Shared defaults moved to framework providers with compatibility adapters. |
| FND-010 | P0 | Add tests for untested critical modules | In Progress | `tool_pipeline`, `cqrs`, `workflows/executor`, `sqlite_lancedb` each have ≥5 passing tests. |
| FND-011 | P1 | Security CI baseline hardening | Implemented + Verified | At least one blocking critical-path security scan exists, `SECURITY.md` documents thresholds/exception handling, and remaining advisory scanners have explicit rationale. |
| FND-012 | P1 | Move heavyweight deps behind extras | Not Started | `sentence-transformers`, `lancedb`, `pyarrow` moved to `[embeddings]` extra; base install <100MB. |
| FND-013 | P2 | Triage TODO/FIXME markers to issues | Not Started | 81 markers triaged: converted to GitHub issues, resolved, or documented as intentional. |
| FND-014 | P1 | Security scanner escalation beyond baseline | Not Started | `dependency-audit`, `bandit`, and `semgrep` move from advisory to tuned, merge-blocking thresholds with documented exceptions. |

## Current Tranche (2026-03-15)
### Scope
- FND-008: Repair validation workflow truth paths and repo metadata drift
- Capture a tiered codebase/roadmap assessment for follow-on work
- Define FND-009 so future sessions can automate the drift checks instead of re-auditing manually

### Planned file touchpoints
- `.github/workflows/validation.yml`
- `.github/workflows/vertical-validation.yml`
- `.github/workflows/fep-validation.yml`
- `.github/workflows/pr-comment.yml`
- `scripts/ci/find_modified_vertical_dirs.py`
- `Makefile`
- `roadmap.md`
- `docs/tech-debt/codebase-assessment-2026-03-15.md`
- `docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md`
- `scripts/ci/repo_hygiene_check.py`
- `tests/unit/scripts/test_repo_hygiene_check.py`

## Session Log
### 2026-03-04 (Session A)
- Created persistent execution plan.
- Selected first critical tranche: FND-001 + FND-002.
- Next: implement code + tests, then update status and residual risks.

### 2026-03-04 (Session B)
- Implemented FND-001:
  - aligned default step-handler documentation with runtime behavior
  - removed unresolved default dependencies (`safety`, `middleware`) from top-level dependency graph
  - added regression tests for default order and dependency resolution
- Implemented FND-002:
  - `stream_with_events()` now converts chunk-derived events through `EventRegistry.from_external(...)`
  - added `STREAM_CHUNK` aliases in event converters for content/thinking/tool call/tool result
  - added unit test asserting registry conversion path is used
- Verification:
  - `python -m py_compile` passed for all touched source/test files
  - full targeted pytest run blocked by pre-existing syntax error in `victor/tools/registry.py:92`

### 2026-03-04 (Session C)
- Implemented FND-007:
  - added `_validate_step_handler_dependency_contract(...)` in `VerticalIntegrationPipeline`
  - added `_collect_dependency_contract_violations(...)` to detect duplicate handler names and missing dependencies
  - wired validation into both `apply(...)` and `apply_async(...)`
  - violations now write `IntegrationResult.step_status["dependency_contract"]` with warning/error severity
- Added tests:
  - `test_apply_records_dependency_contract_warning_for_missing_handlers`
  - `test_apply_async_records_dependency_contract_error_in_strict_mode`
- Verification:
  - `python -m py_compile victor/framework/vertical_integration.py tests/unit/framework/test_vertical_integration.py`
  - `pytest -q tests/unit/framework/test_vertical_integration.py -k "dependency_contract or classify_handlers or build_execution_levels"` (6 passed)
  - `pytest -q tests/unit/framework/test_framework_internal.py tests/unit/framework/test_framework_step_handler.py -k "stream_with_events or StepHandlerRegistryContract"` (2 passed)
  - note: pytest-cov emitted warnings due local `.coverage` DB inconsistency (`no such table: line_bits`), but tests completed successfully

### 2026-03-04 (Session D)
- Implemented FND-003:
  - added explicit deprecation fence helper for legacy extension-registry APIs in `vertical_integration.py`
  - `VerticalIntegrationPipeline(extension_registry=...)` now emits a deprecation warning and remains intentionally inactive
  - `get_extension_handler_registry()` and `register_extension_handler()` now emit deprecation warnings with migration guidance
  - updated architecture doc extension example to use active `ExtensionsStepHandler.extension_registry` path
- Added migration tests:
  - existing legacy-constructor test now asserts deprecation warning emission
  - new test verifies `register_extension_handler(...)` warns and remains inactive for default pipeline execution
- Verification:
  - `python -m py_compile victor/framework/vertical_integration.py tests/unit/framework/test_vertical_integration.py`
  - `pytest -q tests/unit/framework/test_vertical_integration.py -k "legacy_extension_registry or register_extension_handler or dependency_contract"` (4 passed)
  - note: pytest-cov emitted warnings due local `.coverage` DB inconsistency (`no such table: tracer`), but tests completed successfully

### 2026-03-04 (Session E)
- Implemented FND-004:
  - added warm-start metadata fast path to `EntryPointCache._load_from_disk(...)`
  - introduced lightweight installation fingerprint to reuse persisted `env_hash` without full package distribution sweep when environment is unchanged
  - added `_meta` persistence payload in `_save_to_disk(...)` for env hash + fingerprint reuse
  - preserved strict fallback behavior: fingerprint mismatch or missing metadata still computes full env hash
- Added tests:
  - `test_load_from_disk_reuses_meta_env_hash_without_full_scan`
  - `test_load_from_disk_meta_mismatch_falls_back_to_full_scan`
  - `test_save_to_disk_persists_meta_env_hash`
- Verification:
  - `python -m py_compile victor/framework/module_loader.py tests/unit/framework/test_module_loader.py`
  - `pytest -q tests/unit/framework/test_module_loader.py -k "reuses_meta_env_hash_without_full_scan or meta_mismatch_falls_back_to_full_scan or persists_meta_env_hash or load_from_disk_ignores_wrong_env_hash"` (4 passed)
  - note: pytest-cov emitted warnings due local `.coverage` DB inconsistency (`no such table: tracer`), but tests completed successfully

### 2026-03-15 (Session F)
- Implemented FND-008:
  - fixed `.github/workflows/validation.yml` trigger structure so the workflow is executable again
  - replaced broken modified-vertical detection with reusable `scripts/ci/find_modified_vertical_dirs.py`
  - aligned validation workflows with the local `victor-sdk` install pattern already used elsewhere in CI
  - corrected wrong GitHub repo links in PR helper output and bundled vertical metadata
  - restored `make lint` to fail on mypy after confirming the current tree is clean
- Added planning artifacts:
  - `docs/tech-debt/codebase-assessment-2026-03-15.md`
  - roadmap/foundation-plan refresh entries capturing the corrected baseline
- Verification:
  - `mypy victor` (0 findings across 1508 files)
  - `python scripts/ci/find_modified_vertical_dirs.py` exercised with changed vertical file paths
  - workflow YAML parsed successfully after edits

### 2026-03-15 (Session G)
- Implemented FND-009:
  - added `scripts/ci/repo_hygiene_check.py` to enforce foundational drift checks
  - covered workflow trigger presence, banned first-party repo URLs, uppercase `ROADMAP.md` markdown links, archived-roadmap banner presence, and non-advisory `Makefile` lint behavior
  - wired the check into both `make lint` and `CI - Fast Checks`
  - added focused unit tests in `tests/unit/scripts/test_repo_hygiene_check.py`
- Verification:
  - `pytest -q tests/unit/scripts/test_repo_hygiene_check.py`
  - `python scripts/ci/repo_hygiene_check.py`

## Resume Protocol
1. Open this file and continue from the first `In Progress` P0/P1 item.
2. Run targeted tests for touched modules before closing tranche.
3. Append a dated session log entry with:
   - what changed
   - tests executed
   - unresolved risks
4. Update `Status` and acceptance criteria completion notes in the table.
