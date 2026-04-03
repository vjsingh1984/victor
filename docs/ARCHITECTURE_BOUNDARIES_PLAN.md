# Victor Architecture Refactoring Plan

**Date**: April 2, 2026
**Status**: IN PROGRESS
**Objective**: Fix circular imports, establish clear architectural boundaries, prevent future coupling issues

---

## Executive Summary

This document outlines a comprehensive plan to fix architectural issues in the Victor AI framework, specifically:
1. Circular import between `victor.protocols.team` and `victor.teams.protocols`
2. Unclear dependency boundaries between victor-ai, victor-sdk, and verticals
3. Missing enforcement of architectural rules

---

## Current State Analysis

### Problems Identified

#### 1. Circular Import Chain
```
victor.teams.protocols → victor.protocols.team → victor.teams.types → victor.teams.protocols
```

**Status**: ✅ PARTIALLY FIXED
- Fixed `victor/teams/__init__.py` to import from canonical location
- Still need to verify and fix all other import chains

#### 2. victor-sdk Confusion
- victor-sdk is a subdirectory of victor-ai monorepo
- Published as separate package on PyPI
- Tests unclear if victor-sdk is installed
- Import paths: `victor_sdk.*` vs `victor.framework.*`

**Status**: ⚠️ NEEDS CLARIFICATION

#### 3. Import Boundary Violations
- Verticals importing from internal modules: `victor.agent.*`, `victor.core.*`
- Tests using `victor_sdk.constants` without proper installation
- No enforcement of allowed imports

**Status**: ❌ NOT ENFORCED

---

## Target Architecture

### Dependency Rules

```
┌─────────────────────────────────────────────────────────────┐
│                    victor-sdk                                │
│  (Zero runtime dependencies, pure protocols/types)          │
│  victor_sdk.protocols.* - Protocol definitions              │
│  victor_sdk.types.* - Shared data types                     │
│  victor_sdk.constants.* - Constants/enums                   │
└─────────────────────────────────────────────────────────────┘
                            ▲
                            │
                            │ imports from
                            │
┌─────────────────────────────────────────────────────────────┐
│                    victor-ai                                 │
│  (Core framework, depends on victor-sdk)                    │
│  victor.framework.* - Public API (Agent, StateGraph, etc.)  │
│  victor.protocols.* - Protocol implementations               │
│  victor.agent.* - Internal orchestrator/coordinators        │
│  victor.core.* - Internal core (events, state, middleware)   │
│  victor.tools.* - Tool implementations                       │
│  victor.teams.* - Multi-agent coordination                   │
│  victor.verticals.contrib.* - Built-in verticals             │
└─────────────────────────────────────────────────────────────┘
                            ▲
                            │
                            │ imports from (public API only)
                            │
┌─────────────────────────────────────────────────────────────┐
│              External Verticals                              │
│  victor-coding, victor-devops, victor-rag, etc.             │
│                                                             │
│  Allowed imports:                                           │
│  ✅ victor_sdk.*                                            │
│  ✅ victor.framework.*                                      │
│  ✅ victor.protocols.team.* (canonical team protocols)      │
│  ✅ victor.config.*                                         │
│  ✅ victor.storage.*                                        │
│                                                             │
│  Forbidden imports:                                         │
│  ❌ victor.agent.* (internal)                               │
│  ❌ victor.core.verticals.* (internal)                      │
│  ❌ victor.teams.protocols.* (use canonical location)       │
└─────────────────────────────────────────────────────────────┘
```

### Canonical Import Locations

| Concept | Canonical Import | Deprecated Import |
|---------|-----------------|-------------------|
| Agent Protocol | `victor.protocols.team.IAgent` | `victor.teams.protocols.IAgent` |
| Team Protocols | `victor.protocols.team` | `victor.teams.protocols` |
| Team Types | `victor.teams.types` | (none, this is correct) |
| Vertical Registration | `victor_sdk.registration.register_vertical` | `victor.core.verticals.registration.register_vertical` |
| Capability Provider | `victor.framework.extensions.CapabilityProvider` | `victor.core.verticals.capabilities.CapabilityProvider` |

---

## Implementation Plan

### Phase 1: Fix Immediate Circular Imports ✅

**Status**: COMPLETED

- [x] Fixed `victor/teams/__init__.py` to import from `victor.protocols.team`
- [x] Committed fix (commit 22573610e)

---

### Phase 2: Establish Canonical Import Locations

#### 2.1 Audit All Protocol Imports

**Goal**: Ensure all code imports protocols from canonical locations

**Actions**:
1. Find all imports from `victor.teams.protocols`
2. Replace with `victor.protocols.team`
3. Find all imports from internal modules in verticals
4. Replace with public API equivalents

**Commands**:
```bash
# Find all non-canonical protocol imports
grep -r "from victor.teams.protocols" victor/
grep -r "from victor.framework.protocols" victor/
grep -r "import victor.teams.protocols" victor/
```

**Files to Update**:
- victor/teams/protocols.py (already a re-export shim, keep for backward compat)
- victor/teams/unified_coordinator.py
- victor/framework/vertical_integration.py
- victor-verticals/contrib/*/assistant.py

---

#### 2.2 Create Import Boundary Tests

**Goal**: Prevent future import boundary violations

**Actions**:
1. Create `tests/unit/architecture/test_import_boundaries.py`
2. Add test to verify verticals don't import from internal modules
3. Add test to verify victor-sdk has zero dependencies on victor-ai
4. Add mypy plugin to enforce import rules

**Test Template**:
```python
# tests/unit/architecture/test_import_boundaries.py
"""Tests to enforce architectural boundaries."""

import importlib
import subprocess
import sys
from pathlib import Path

import pytest


class TestVerticalImportBoundaries:
    """Verify verticals only import from public API."""

    VERTICALS = ["coding", "devops", "rag", "dataanalysis", "research"]

    FORBIDDEN_IMPORTS = [
        "victor.agent.",
        "victor.core.verticals.",
        "victor.teams.protocols",  # Use victor.protocols.team
    ]

    @pytest.mark.parametrize("vertical", VERTICALS)
    def test_vertical_no_internal_imports(self, vertical):
        """Verify vertical doesn't import internal modules."""
        module_path = f"victor.verticals.contrib.{vertical}"
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            pytest.skip(f"Vertical {vertical} not installed")

        # Get all Python files in vertical
        vertical_path = Path(module.__file__).parent
        py_files = list(vertical_path.rglob("*.py"))

        for py_file in py_files:
            content = py_file.read_text()
            for forbidden in self.FORBIDDEN_IMPORTS:
                assert forbidden not in content, (
                    f"{py_file.relative_to(vertical_path)} "
                    f"imports forbidden module {forbidden}"
                )


class TestVictorSDKNoDependencies:
    """Verify victor-sdk has zero dependencies on victor-ai."""

    def test_victor_sdk_no_victor_ai_imports(self):
        """Verify victor-sdk doesn't import from victor-ai."""
        sdk_path = Path("victor-sdk/victor_sdk")
        py_files = list(sdk_path.rglob("*.py"))

        for py_file in py_files:
            if "__pycache__" in str(py_file):
                continue
            content = py_file.read_text()
            # victor-sdk should not import from victor package
            assert "from victor." not in content or "from victor_sdk." in content, (
                f"{py_file} imports from victor-ai, "
                "but victor-sdk must have zero dependencies"
            )


class TestCanonicalProtocolImports:
    """Verify protocols are imported from canonical locations."""

    def test_teams_protocols_from_canonical_location(self):
        """victor.teams.protocols should re-export from victor.protocols.team."""
        import victor.teams.protocols as teams_protocols
        import victor.protocols.team as canonical

        # Verify all exports come from canonical location
        for name in teams_protocols.__all__:
            canonical_attr = getattr(canonical, name, None)
            teams_attr = getattr(teams_protocols, name, None)
            assert canonical_attr is teams_attr or teams_attr is None, (
                f"{name} in victor.teams.protocols should come from "
                f"victor.protocols.team (canonical location)"
            )
```

---

### Phase 3: Fix victor-sdk Installation

#### 3.1 Clarify victor-sdk Installation

**Problem**: victor-sdk is part of victor-ai monorepo but needs to be installable separately

**Solution**: Ensure both work correctly:
1. Install from monorepo: `pip install -e /path/to/victor-ai/victor-sdk`
2. Install from PyPI: `pip install victor-sdk`

**Actions**:
- [x] Update CI workflows to install victor-sdk from monorepo
- [ ] Verify victor-sdk/pyproject.toml is standalone
- [ ] Add tests to verify victor-sdk imports work

---

#### 3.2 Fix victor_sdk.constants Imports

**Problem**: Tests import `victor_sdk.constants` but it's unclear what's available

**Actions**:
1. Document what's in victor_sdk.constants
2. Update imports to use specific modules:
   - `victor_sdk.constants.capability_ids`
   - `victor_sdk.constants.tool_names`
3. Add `__all__` to victor_sdk/constants/__init__.py

---

### Phase 4: Update Documentation

#### 4.1 Create Architecture Documentation

**Files to Create**:
1. `docs/verticals/ARCHITECTURE_BOUNDARIES.md` - This file
2. `docs/verticals/IMPORT_GUIDE.md` - How to import correctly
3. `docs/verticals/MIGRATION_GUIDE.md` - Update with import rules

#### 4.2 Update Developer Guide

Add section on architectural boundaries:
- Allowed imports for verticals
- Public vs internal API
- How to add new protocols
- How to avoid circular imports

---

### Phase 5: Enforcement Mechanisms

#### 5.1 Add Pre-commit Hook

Create `.git/hooks/pre-commit` check:
```bash
#!/bin/bash
# Check for forbidden imports in verticals
python scripts/check_import_boundaries.py || exit 1
```

#### 5.2 Add CI Check

Update `.github/workflows/test.yml`:
```yaml
- name: Check import boundaries
  run: |
    python tests/unit/architecture/test_import_boundaries.py
```

#### 5.3 Add Mypy Plugin

Create `mypy_import_plugin.py` to enforce import rules at type-check time.

---

## Success Criteria

### Must Have
- [x] No circular imports in victor-ai codebase
- [ ] All verticals only import from public API
- [ ] victor-sdk has zero dependencies on victor-ai
- [ ] CI/CD passes for all verticals

### Should Have
- [ ] Import boundary tests pass
- [ ] Pre-commit hook enforces rules
- [ ] Documentation updated

### Nice to Have
- [ ] Mypy plugin for import enforcement
- [ ] Automated import fixer tool
- [ ] Architecture decision records (ADRs)

---

## Implementation Order

1. ✅ **Phase 1**: Fix immediate circular imports (DONE)
2. **Phase 2**: Establish canonical import locations (IN PROGRESS)
   - 2.1 Audit all protocol imports
   - 2.2 Create import boundary tests
3. **Phase 3**: Fix victor-sdk installation
   - 3.1 Clarify installation process
   - 3.2 Fix victor_sdk.constants imports
4. **Phase 4**: Update documentation
5. **Phase 5**: Add enforcement mechanisms

---

## Risks and Mitigations

### Risk 1: Breaking Changes in Verticals
**Mitigation**: Maintain backward compatibility shims, deprecate old imports

### Risk 2: External Verticals Break
**Mitigation**: Provide migration guide, support old imports for one release cycle

### Risk 3: CI/CD Failures During Migration
**Mitigation**: Implement in phases, test each phase before proceeding

---

## Next Steps

1. Complete Phase 2.1: Audit and fix protocol imports
2. Complete Phase 2.2: Create import boundary tests
3. Address victor_sdk.constants import issues
4. Update all documentation
5. Add enforcement mechanisms

---

**Last Updated**: April 2, 2026
**Owner**: Vijaykumar Singh
**Status**: Ready for implementation
