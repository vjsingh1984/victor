# Test Migration Plan: Framework to Vertical Repositories

**Status**: In Progress  
**Created**: 2026-03-04  
**Author**: Vijaykumar Singh <singhvjd@gmail.com>

## Executive Summary

This plan migrates tests from the victor (framework) repository that depend on `victor_coding` to the `victor-coding` (vertical) repository. This eliminates test-time circular dependencies and ensures the framework can be tested completely without external packages.

---

## Problem Statement

### Current State

**18+ test files** in `tests/unit/` require `victor_coding` to run:

```python
pytest.importorskip("victor_coding.languages.base")  # Creates dependency!
```

This violates our architecture principles:
- **Framework tests depend on external vertical**
- **Incomplete test coverage** when victor_coding is not installed
- **Circular dependency at test time**
- **Cannot verify framework works independently**

### Target State

```
victor/ (framework)                    victor-coding/ (vertical)
├── tests/                              ├── tests/
│   ├── unit/                           │   ├── integration/
│   │   ├── contrib/                    │   │   ├── test_documentation_tool.py
│   │   ├── framework/                  │   │   └── test_lsp_tools.py
│   │   │   └── (no victor_coding deps) │   ├── protocols/
│   │   └── (no victor_coding deps)     │   │   └── test_docgen_protocol_impl.py
│   └── integration/                    │   └── indexers/
│                                        │       └── test_tree_sitter_extractor.py
```

---

## Test Categorization

### Category 1: Protocol Definition Tests (Keep in victor, Use Mocks)

**Purpose**: Verify protocol structure exists, no implementation testing

| Test File | Action | New Approach |
|-----------|--------|--------------|
| `tests/unit/protocols/test_vertical_protocols.py` | **KEEP** | Use mocks, test protocol exists |
| `tests/unit/framework/test_vertical_integration.py` | **KEEP** | Test with victor.contrib stubs |

### Category 2: Tool Integration Tests (Move to victor-coding)

| Test File | Action | Target Location |
|-----------|--------|-----------------|
| `tests/unit/tools/test_documentation_tool_unit.py` | **MOVE** | `victor-coding/tests/integration/` |
| `tests/unit/tools/test_lsp_tool.py` | **MOVE** | `victor-coding/tests/integration/` |
| `tests/unit/tools/test_lsp.py` | **MOVE** | `victor-coding/tests/integration/` |
| `tests/unit/tools/test_query_expander.py` | **MOVE** | `victor-coding/tests/integration/` |
| `tests/unit/tools/test_code_search_tool.py` | **MOVE** | `victor-coding/tests/integration/` |

### Category 3: Indexer Tests (Move to victor-coding)

| Test File | Action | Target Location |
|-----------|--------|-----------------|
| `tests/unit/indexers/` (all files) | **MOVE** | `victor-coding/tests/indexers/` |

---

## Migration Phases

### Phase 1: Prepare victor Repository (Current)

**Goal**: Ensure victor tests can run without victor_coding

**Tasks**:
1. Update protocol tests to use mocks instead of requiring victor_coding
2. Remove `pytest.importorskip("victor_coding")` from protocol tests
3. Verify `pytest tests/unit/` runs completely without victor_coding installed

### Phase 2: Create Test Inventory

**Goal**: Document which tests will be removed from victor

**Tasks**:
1. Create `MIGRATED_TESTS.md` listing all moved tests
2. Add deprecation notices to test files

### Phase 3: Migrate to victor-coding

**Goal**: Move victor_coding-dependent tests

**Tasks**:
1. Create directory structure in `../victor-coding/tests/`
2. Move tests to appropriate categories
3. Update imports in moved tests

### Phase 4: Delete from victor

**Goal**: Clean up victor repository

**Tasks**:
1. Delete migrated test files from victor
2. Update documentation
3. Verify test suite

---

## Next Steps

1. ✅ Create migration plan
2. ⏳ Start Phase 1: Update protocol tests
3. ⏳ Execute migration

---

**Last Updated**: 2026-03-04  
**Status**: In Progress
