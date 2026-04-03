# TODO/FIXME/HACK Marker Triage (2026-03-15)

**Total markers**: 81 across `victor/`
**Triaged**: All categorized below

## Category 1: Intentional Stubs (No Action — by design)

These are placeholder implementations in features that aren't fully built yet:

| File | Marker | Status |
|------|--------|--------|
| `benchmark/safety.py:123` | TODO: Add benchmark-specific safety rules | Intentional — benchmark vertical is utility |
| `benchmark/mode_config.py:117` | TODO: Define benchmark-specific task budgets | Intentional |
| `workflows/orchestrator_pool.py:63` | TODO: Implement proper orchestrator pool | Future feature |
| `workflows/deployment.py:1372` | TODO: HTTP/gRPC to remote | Future distributed execution |
| `workflows/executors/factory.py:129` | TODO: Phase 2 - Use new executor implementations | Phased rollout |
| `workflows/compiler/unified_compiler.py:118` | TODO: Implement proper compilation | WIP compiler |
| `workflows/services/credentials.py:271,286` | TODO: Add Windows Hello support | Platform-specific |
| `workflows/services/credentials.py:333,338` | TODO: Implement FIDO2 support | Hardware auth |
| `workflows/services/providers/aws.py:161` | TODO: Implement CloudWatch log retrieval | AWS integration |
| `workflows/services/providers/base.py:436` | TODO: Implement proper gRPC health check | Future protocol |
| `processing/native/__init__.py:2692,2718` | TODO: Return Rust implementation when available | Rust FFI pending |
| `native/python/symbol_extractor.py:266,274,282` | TODO: Add support for other languages | Language expansion |

**Count**: ~16 markers. **Action**: None needed — these are tracked feature gaps.

## Category 2: Detection Patterns (Not real TODOs)

These are string literals in code that detects/matches TODO patterns:

| File | Context |
|------|---------|
| `benchmark/escape_hatches.py:381-382` | Detection: checks if solution code contains TODO |
| `tools/batch_processor_tool.py:271` | Example: batch search for TODO pattern |
| `tools/scaffold_tool.py:364` | Template: scaffold generates TODO placeholder |
| `ui/commands/fep.py:*` | Template: FEP uses XXXX placeholder (not TODO) |
| `feps/schema.py:539` | Validation: allows XXXX placeholder values |

**Count**: ~10 markers. **Action**: None — these are correct code behavior.

## Category 3: Dashboard App (Concentrated)

`victor/observability/dashboard/app.py` has 28 TODO markers — it's a prototype dashboard.

**Action**: This file is a development prototype. When productionized, TODOs will be resolved. No action now.

## Category 4: Migration/Cleanup Debt (Actionable)

| File | Marker | Action |
|------|--------|--------|
| `core/events/adapter.py:79` | TODO: Remove function once migration complete | Check if migration is complete; remove if so |
| `agent/response_quality.py` | 3 markers | Review: response quality scoring improvements |
| `agent/grounding_verifier.py` | 2 markers | Review: grounding verification gaps |
| `agent/coordinators/planning_coordinator.py` | 2 markers | Review: planning coordinator improvements |
| `evaluation/correction/validators/generic_validator.py` | 2 markers | Review: validation gaps |
| `protocols/quality.py` | 2 markers | Review: quality protocol gaps |
| `storage/embeddings/task_classifier.py` | 2 markers | Review: classifier improvements |
| `verticals/contrib/coding/testgen/generator.py` | 4 markers | Review: test generation gaps |

**Count**: ~19 markers. **Action**: Convert to GitHub issues for tracking.

## Summary

| Category | Count | Action |
|----------|-------|--------|
| Intentional stubs | ~16 | None — tracked features |
| Detection patterns | ~10 | None — correct code |
| Dashboard prototype | 28 | None — prototype |
| Migration/cleanup | ~19 | Convert to issues |
| FEP/template XXXX | 8 | None — template syntax |
| **Total** | **81** | **19 actionable** |
