# Provider Coordinator Cleanup Proposal

**Date:** 2026-05-04
**Status:** BREAKING CHANGE PROPOSAL
**Related:** Agent Facade Service Migration Audit (Item #6)

## Current State

### Provider Coordinator Classes

Two coordinator classes remain in the codebase:

1. **ProviderCoordinator** (`victor/agent/provider/coordinator.py`)
   - 20 KB (653 lines)
   - Wraps ProviderManager with rate limiting, retry logic
   - NOT exported from `victor/agent/provider/__init__.py`
   - NOT used in production code (removed from ProviderRuntimeComponents)

2. **ProviderSwitchCoordinator** (`victor/agent/provider/switch_coordinator.py`)
   - 18 KB (531 lines)
   - Manages provider switching with hooks and coordination
   - Exported from `victor/agent/provider/__init__.py`
   - NOT used in production code (removed from ProviderRuntimeComponents)

### Current Usage

**Internal code:**
- ✅ Removed from `ProviderRuntimeComponents` (2026-05-01)
- ✅ Guard tests prevent new internal usage
- ✅ ProviderService is canonical authority for provider operations

**Compatibility shims:**
- `ProviderFacade.provider_coordinator` - Creates lazy compatibility shim
- `ProviderFacade.provider_switch_coordinator` - Creates lazy compatibility shim
- `AgentOrchestrator._provider_coordinator` - Lazy property with deprecation warning
- `AgentOrchestrator._provider_switch_coordinator` - Lazy property with deprecation warning

**External packages:**
- ❓ Unknown if external packages import these classes
- Need to check victor-coding, victor-devops, victor-rag, victor-dataanalysis, victor-research

## Breaking Change Analysis

### Impact Assessment

**Internal impact:** NONE
- No production code uses these coordinators
- All internal usage goes through ProviderService
- Compatibility shims can be removed

**External impact:** UNKNOWN
- ProviderCoordinator is NOT in `__all__` (not officially exported)
- ProviderSwitchCoordinator IS in `__all__` (officially exported)
- External packages may import ProviderSwitchCoordinator

### Risk Evaluation

**Low risk:**
- ProviderCoordinator not in `__all__` (never officially exported)
- Only exported from __init__.py is ProviderSwitchCoordinator
- Guard tests prevent new internal code from using them
- Already removed from runtime components

**Medium risk:**
- ProviderSwitchCoordinator is exported from __init__.py
- External packages may depend on it
- Need to validate external package compatibility

## Proposal: Two-Phase Removal

### Phase 1: Validate External Package Compatibility (BACKWARD COMPATIBLE)

**Goal:** Determine if external packages use ProviderSwitchCoordinator

**Actions:**
1. Check if victor-coding, victor-devops, victor-rag, victor-dataanalysis, victor-research import ProviderSwitchCoordinator
2. If imports found, add deprecation warnings to ProviderSwitchCoordinator class
3. Document migration path for external packages
4. Allow one release cycle for external packages to migrate

**Timeline:** One release (e.g., v0.8.0)

### Phase 2: Complete Removal (BREAKING CHANGE)

**Goal:** Remove both coordinator classes entirely

**Actions:**
1. Delete `victor/agent/provider/coordinator.py`
2. Delete `victor/agent/provider/switch_coordinator.py`
3. Remove from `victor/agent/provider/__init__.py`
4. Update `ProviderFacade` to remove coordinator properties
5. Update `AgentOrchestratorProperties` to remove coordinator properties
6. Update compatibility shims to not reference coordinators
7. Remove import guard tests (no longer needed)
8. Update documentation to remove coordinator references

**Timeline:** Breaking release (e.g., v1.0.0)

## Alternative: Immediate Removal (Recommended)

**Recommendation:** Proceed directly to Phase 2 in next breaking release

**Rationale:**

1. **Internal code doesn't use them:**
   - Already removed from ProviderRuntimeComponents
   - Guard tests prevent new usage
   - ProviderService is canonical authority

2. **External impact minimal:**
   - ProviderCoordinator not in `__all__` (never official API)
   - ProviderSwitchCoordinator only used for provider switching
   - External packages should use ProviderService directly
   - Provider switching is internal concern

3. **Cleaner architecture:**
   - Removes 1,184 lines of dead code
   - Eliminates confusion about canonical ownership
   - Simplifies provider package
   - No need to maintain compatibility shims

4. **Migration path clear:**
   - External packages should use ProviderService
   - Provider switching is an internal implementation detail
   - External packages don't need provider switching logic

## Implementation Plan

### If Immediate Removal (Recommended)

**Single breaking release:**

1. **Delete coordinator files**
   - Delete `victor/agent/provider/coordinator.py`
   - Delete `victor/agent/provider/switch_coordinator.py`

2. **Update package exports**
   - Remove ProviderSwitchCoordinator from `__all__`
   - Remove imports from `__init__.py`

3. **Update compatibility shims**
   - Remove provider_coordinator property from ProviderFacade
   - Remove provider_switch_coordinator property from ProviderFacade
   - Remove _provider_coordinator property from AgentOrchestratorProperties
   - Remove _provider_switch_coordinator property from AgentOrchestratorProperties

4. **Update tests**
   - Remove `test_provider_coordinator_import_guard.py`
   - Update `test_provider_runtime.py` to remove coordinator tests
   - Update `test_runtime_lazy_init.py` to remove coordinator tests

5. **Update documentation**
   - Remove coordinator references from CLAUDE.md
   - Update migration audit to mark as COMPLETED
   - Add breaking change notes to RELEASE_NOTES

6. **Version bump:** Major version bump (v1.0.0)

## Benefits

**Code quality:**
- Remove 1,184 lines of dead code
- Cleaner provider package
- No confusion about canonical ownership

**Architecture:**
- ProviderService is unambiguous single authority
- No compatibility shim layer
- Simpler mental model

**Maintenance:**
- Less code to maintain
- Fewer tests to run
- Clearer documentation

## Risks and Mitigations

### Risk: External packages break

**Mitigation:** 
- ProviderCoordinator not in `__all__` (no official API commitment)
- ProviderSwitchCoordinator used for internal provider switching
- External packages should use ProviderService API
- Check external packages before removal

### Risk: Breaking change too large

**Mitigation:**
- This is targeted removal of already-unused code
- Internal code already migrated to ProviderService
- Clear migration path (use ProviderService)
- Can add deprecation period if needed

## Recommendation

**Proceed with immediate removal in next breaking release (v1.0.0)**

**Justification:**
1. Internal code already migrated to ProviderService
2. Coordinator classes are dead code (not used in production)
3. ProviderCoordinator not officially exported (not in `__all__`)
4. ProviderSwitchCoordinator is internal implementation detail
5. External packages should use ProviderService API anyway
6. Removing dead code improves architecture
7. Simpler to remove now than to maintain compatibility shims indefinitely

## External Package Validation (2026-05-04)

**Result:** ✅ NO EXTERNAL USAGE FOUND

Validated that external verticals do NOT import ProviderCoordinator or ProviderSwitchCoordinator:
- victor-coding: NO imports
- victor-devops: NO imports
- victor-rag: NO imports
- victor-dataanalysis: NO imports
- victor-research: NO imports

**Conclusion:** Safe to remove without breaking external packages.

## Updated Recommendation

**DECISION:** Proceed with immediate removal in next breaking release (v1.0.0)

**Rationale (UPDATED):**
1. ✅ External packages do NOT use ProviderCoordinator or ProviderSwitchCoordinator
2. ✅ Internal code already migrated to ProviderService
3. ✅ Coordinator classes are dead code (not used in production)
4. ✅ ProviderCoordinator not in `__all__` (no official API commitment)
5. ✅ ProviderSwitchCoordinator only used for provider switching (internal)
6. ✅ External packages use ProviderService API instead
7. ✅ Removing dead code improves architecture significantly

**Implementation:** Immediate removal in breaking release v1.0.0

## Open Questions

1. ~~Should we check external packages before removal?~~ ✅ COMPLETED: No external usage found
2. ~~Should we add a deprecation period?~~ DECIDED: No need, external packages don't use them
3. ~~Is ProviderSwitchCoordinator used by any external packages?~~ ✅ COMPLETED: NO

## Related Documentation

- Agent Facade Service Migration Audit (provider coordinator sections)
- Provider runtime documentation
- CLAUDE.md (provider architecture sections)
