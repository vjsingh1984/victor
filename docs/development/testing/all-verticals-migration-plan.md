# All Verticals Test Migration Plan

**Date:** 2026-03-04
**Scope:** Migrate vertical-specific tests from victor framework to respective vertical repositories

## Executive Summary

This plan addresses the migration of vertical-specific integration tests for all external verticals:
- victor-devops
- victor-research
- victor-rag
- victor-dataanalysis
- victor-benchmark (partial)
- victor-iac (none found)
- victor-classification (none found)
- victor-security (none found)

## Migration Categories

### Category A: Migrate to Vertical Repositories

These tests test vertical-specific safety rule implementations and should be migrated:

#### From `tests/unit/framework/test_config.py` - TestVerticalSafetyIntegration class

| Test Method | Target Vertical | Lines | Reason |
|-------------|----------------|-------|--------|
| `test_devops_safety_deployment_rules` | victor-devops | 473-486 | Tests DevOps deployment safety rules |
| `test_devops_safety_container_rules` | victor-devops | 488-501 | Tests DevOps container safety rules |
| `test_devops_safety_infrastructure_rules` | victor-devops | 503-521 | Tests DevOps infrastructure safety rules |
| `test_create_all_devops_safety_rules` | victor-devops | 540-555 | Tests all DevOps safety rules |
| `test_rag_safety_deletion_rules` | victor-rag | 557-575 | Tests RAG deletion safety rules |
| `test_rag_safety_ingestion_rules` | victor-rag | 577-605 | Tests RAG ingestion safety rules |
| `test_create_all_rag_safety_rules` | victor-rag | 607-627 | Tests all RAG safety rules |
| `test_research_safety_source_rules` | victor-research | 629-661 | Tests Research source safety rules |
| `test_research_safety_content_rules` | victor-research | 663-686 | Tests Research content safety rules |
| `test_create_all_research_safety_rules` | victor-research | 688-707 | Tests all Research safety rules |
| `test_dataanalysis_safety_pii_rules` | victor-dataanalysis | 709-731 | Tests DataAnalysis PII safety rules |
| `test_dataanalysis_safety_export_rules` | victor-dataanalysis | 733-755 | Tests DataAnalysis export safety rules |
| `test_create_all_dataanalysis_safety_rules` | victor-dataanalysis | 757-776 | Tests all DataAnalysis safety rules |

**Total: 12 tests to migrate** (4 each to devops, rag, research, dataanalysis)

### Category B: Keep in Framework (Internal Framework Tests)

These tests use framework-internal modules and should remain in victor:

| Test Method | Lines | Reason |
|-------------|-------|--------|
| `test_coding_safety_git_rules` | 444-459 | Uses `victor.framework.safety.create_git_safety_rules` |
| `test_coding_safety_file_rules` | 461-471 | Uses `victor.framework.safety.create_file_safety_rules` |
| `test_benchmark_safety_repository_rules` | 778-802 | Uses `victor.benchmark.safety` (framework internal) |
| `test_benchmark_safety_resource_rules` | 804-819 | Uses `victor.benchmark.safety` (framework internal) |
| `test_benchmark_safety_test_rules` | 821-845 | Uses `victor.benchmark.safety` (framework internal) |
| `test_benchmark_safety_data_rules` | 847-880 | Uses `victor.benchmark.safety` (framework internal) |

**Total: 6 tests to keep** (2 coding, 4 benchmark - all framework internal)

### Category C: Integration Tests (Acceptable as-is)

These tests are marked with `@pytest.mark.integration` and test cross-vertical integration:

| Test Method | Location | Reason |
|-------------|----------|--------|
| `test_devops_middleware_configuration` | test_framework_middleware.py:861 | Integration test marked as such |

### Category D: Framework Tests with Graceful Fallback (Acceptable as-is)

| Test Method | Location | Reason |
|-------------|----------|--------|
| `test_capability_config_defaults_with_custom_capability` | test_framework_step_handler.py:619 | Has proper try/except with pytest.skip |
| `test_list_verticals_includes_builtins` | test_framework_shim.py:597 | Has proper try/except with pytest.skip |

## Migration Plan

### Phase 1: victor-devops (4 tests)

**Target File:** `victor-devops/tests/safety/test_safety_integration.py`

**Tests to migrate:**
- test_devops_safety_deployment_rules
- test_devops_safety_container_rules
- test_devops_safety_infrastructure_rules
- test_create_all_devops_safety_rules

### Phase 2: victor-rag (3 tests)

**Target File:** `victor-rag/tests/safety/test_safety_integration.py`

**Tests to migrate:**
- test_rag_safety_deletion_rules
- test_rag_safety_ingestion_rules
- test_create_all_rag_safety_rules

### Phase 3: victor-research (3 tests)

**Target File:** `victor-research/tests/safety/test_safety_integration.py`

**Tests to migrate:**
- test_research_safety_source_rules
- test_research_safety_content_rules
- test_create_all_research_safety_rules

### Phase 4: victor-dataanalysis (3 tests)

**Target File:** `victor-dataanalysis/tests/safety/test_safety_integration.py`

**Tests to migrate:**
- test_dataanalysis_safety_pii_rules
- test_dataanalysis_safety_export_rules
- test_create_all_dataanalysis_safety_rules

## Files to Modify in victor

### `tests/unit/framework/test_config.py`

**Remove lines 473-776** (all vertical-specific safety integration tests except framework internal ones)

**Keep:**
- Lines 1-440: Framework-level config tests
- Lines 444-471: Coding safety rules (framework internal)
- Lines 778-880: Benchmark safety rules (framework internal)

## Verticals Without Tests to Migrate

The following verticals have no framework-level tests that need migration:

- **victor-iac**: No test imports found in victor framework
- **victor-classification**: No test imports found in victor framework
- **victor-security**: No test imports found in victor framework (infrastructure tests have graceful fallback)

## Success Criteria

After migration:
1. Framework tests run without external vertical dependencies
2. Each vertical has its own safety integration tests
3. No circular import dependencies
4. All tests pass with `pytest tests/unit/` in victor

## Risk Assessment

### Low Risk
- Additive changes to vertical repositories
- Clear separation of concerns
- Framework tests become more independent

### Medium Risk
- Need to create test directories in vertical repos that may not exist
- Need to ensure test dependencies are installed in vertical repos

## Rollback Plan

Each phase is independently reversible:
- Revert removal of tests from victor
- Remove migrated tests from vertical repos
