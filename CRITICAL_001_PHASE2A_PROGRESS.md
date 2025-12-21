# CRITICAL-001 Phase 2A: RecoveryCoordinator Extraction - Progress Report

**Date**: 2025-01-XX
**Phase**: Phase 2A - Extract RecoveryCoordinator (Steps 1-3 Complete)
**Status**: IN PROGRESS - 50% Complete
**Estimated Completion**: 4 more hours remaining

## Progress Summary

### ✅ Completed Steps (3/6)

#### Step 1: Create RecoveryCoordinator Component ✅ **COMPLETE** (3 hours)
**File Created**: `victor/agent/recovery_coordinator.py` (711 lines)

**Components Implemented**:
1. **RecoveryContext dataclass** - Encapsulates all state needed for recovery decisions
   - 14 fields (iteration, elapsed_time, tool_calls_used, tool_budget, max_iterations, etc.)
   - Eliminates need for mutable state in RecoveryCoordinator

2. **RecoveryCoordinator class** - Main coordination component
   - 6 condition checking methods (time limit, iteration limit, natural completion, tool budget, progress, blocked threshold)
   - 6 action handling methods (empty response, blocked tool, force execution, force completion, loop warning, recovery integration)
   - 4 filtering/truncation methods
   - 4 prompt/message generation methods
   - 3 metrics/formatting methods
   - **Total**: 23 methods (consolidated from 29 scattered orchestrator methods)

**Architecture Patterns**:
- **Facade**: Simplifies complex recovery subsystem
- **Strategy**: RecoveryHandler provides pluggable strategies
- **Delegation**: Most logic delegated to specialized handlers (StreamingChatHandler, RecoveryHandler, OrchestratorRecoveryIntegration)

**Dependencies**:
- RecoveryHandler (optional)
- OrchestratorRecoveryIntegration (optional)
- StreamingChatHandler (required)
- ContextCompactor (optional)
- UnifiedTaskTracker (required)
- Settings (required)

---

#### Step 2: Add Protocol and DI Registration ✅ **COMPLETE** (1 hour)
**Files Modified**:
1. **victor/agent/protocols.py** (+211 lines)
   - Added `RecoveryCoordinatorProtocol` with 23 method signatures
   - Added to `__all__` exports
   - Line 945-1154

2. **victor/agent/service_provider.py** (+52 lines)
   - Added `_create_recovery_coordinator()` factory method (Line 1110-1151)
   - Added `RecoveryCoordinatorProtocol` import
   - Registered in `register_singleton_services` (Line 466-471)
   - **Lifetime**: SINGLETON (stateless coordinator)

**DI Resolution**:
```python
# Factory method creates RecoveryCoordinator with all dependencies
recovery_handler = container.get_optional(RecoveryHandlerProtocol)
recovery_integration = None  # Not in DI yet
streaming_handler = container.get(StreamingHandlerProtocol)
context_compactor = container.get_optional(ContextCompactorProtocol)
unified_tracker = container.get(TaskTrackerProtocol)

return RecoveryCoordinator(...)
```

---

#### Step 3: Update OrchestratorFactory ✅ **COMPLETE** (1 hour)
**File Modified**: `victor/agent/orchestrator_factory.py` (+14 lines)

**Added Method** (Line 795-809):
```python
def create_recovery_coordinator(self) -> Any:
    """Create RecoveryCoordinator via DI container.

    The RecoveryCoordinator centralizes all recovery and error handling logic
    for streaming chat sessions. It's resolved from the DI container to enable
    proper dependency management and testability.

    Returns:
        RecoveryCoordinator instance for recovery coordination
    """
    from victor.agent.protocols import RecoveryCoordinatorProtocol

    recovery_coordinator = self.container.get(RecoveryCoordinatorProtocol)
    logger.debug("RecoveryCoordinator created via DI")
    return recovery_coordinator
```

**Integration Point**: Added right after `create_recovery_integration` method, maintaining logical grouping of recovery-related factory methods.

---

### 🔄 Remaining Steps (3/6)

#### Step 4: Update AgentOrchestrator (2 hours) ⏳ PENDING
**Tasks**:
1. Add RecoveryCoordinator initialization in `__init__`
2. Add `recovery_coordinator` property accessor
3. Update all 29 `*_with_handler` methods to delegate to RecoveryCoordinator
4. Mark old methods as deprecated with `@deprecated` decorator
5. Update streaming loop to use RecoveryCoordinator

**Expected Changes**:
- Add RecoveryCoordinator initialization (~5 lines)
- Add property accessor (~8 lines)
- Deprecate 29 methods with delegation wrappers (~145 lines)
- Update streaming loop calls (~15 locations)

---

#### Step 5: Create Tests (1.5 hours) ⏳ PENDING
**Tasks**:
1. Create `tests/unit/test_recovery_coordinator.py`:
   - Unit tests for all 23 methods
   - Mock all dependencies
   - Test error cases and edge conditions

2. Create `tests/unit/test_recovery_coordinator_di.py`:
   - Test DI resolution
   - Test service lifetime (SINGLETON)
   - Test dependency injection

**Expected Test Count**: ~40-50 tests

---

#### Step 6: Integration Testing (0.5 hours) ⏳ PENDING
**Tasks**:
1. Run full test suite
2. Verify no regressions
3. Check test coverage
4. Performance benchmarking

**Commands**:
```bash
pytest tests/unit/test_recovery_coordinator.py -v
pytest tests/unit/test_orchestrator*.py -v
pytest tests/integration/ -v
pytest --cov=victor.agent.recovery_coordinator --cov-report=html
```

---

## Files Modified Summary

| File | Lines Added | Lines Removed | Net Change | Status |
|------|-------------|---------------|------------|--------|
| `victor/agent/recovery_coordinator.py` | +711 | 0 | +711 | ✅ Complete |
| `victor/agent/protocols.py` | +211 | 0 | +211 | ✅ Complete |
| `victor/agent/service_provider.py` | +52 | 0 | +52 | ✅ Complete |
| `victor/agent/orchestrator_factory.py` | +14 | 0 | +14 | ✅ Complete |
| `victor/agent/orchestrator.py` | TBD | TBD | ~+170/-0 | ⏳ Pending |
| `tests/unit/test_recovery_coordinator.py` | TBD | 0 | ~+300 | ⏳ Pending |
| `tests/unit/test_recovery_coordinator_di.py` | TBD | 0 | ~+50 | ⏳ Pending |
| **Total** | **+988** | **0** | **+988** | **50% Complete** |

---

## Architecture Impact

### Code Organization
- ✅ **RecoveryCoordinator**: 711 lines of well-structured recovery logic
- ✅ **Protocol**: Clear interface with 23 methods
- ✅ **DI Integration**: Singleton service with proper dependency resolution
- ✅ **Factory Method**: Clean integration with OrchestratorFactory

### Dependency Graph
```
RecoveryCoordinator
  ├── RecoveryHandler (optional)
  ├── OrchestratorRecoveryIntegration (optional, not in DI yet)
  ├── StreamingChatHandler (required, from DI)
  ├── ContextCompactor (optional, from DI)
  ├── UnifiedTaskTracker (required, from DI)
  └── Settings (required)
```

### Design Patterns Applied
1. **Facade Pattern**: RecoveryCoordinator provides simple interface to complex recovery subsystem
2. **Strategy Pattern**: RecoveryHandler provides pluggable recovery strategies
3. **Delegation Pattern**: Most logic delegated to specialized handlers
4. **Protocol Pattern**: Type-safe dependency injection via protocols
5. **Factory Pattern**: OrchestratorFactory creates RecoveryCoordinator with all dependencies

---

## Key Decisions

### 1. **Singleton vs Scoped Lifetime**
**Decision**: SINGLETON
**Rationale**: RecoveryCoordinator is stateless - all state passed via RecoveryContext. No need for per-session instances.

### 2. **RecoveryContext Dataclass**
**Decision**: Create dedicated context object
**Rationale**:
- Eliminates mutable state in RecoveryCoordinator
- Makes method signatures cleaner
- Easier to extend without breaking existing methods
- Clearer separation of state vs behavior

### 3. **Optional Dependencies**
**Decision**: Use `get_optional()` for RecoveryHandler, ContextCompactor, OrchestratorRecoveryIntegration
**Rationale**: These services may not be configured/enabled, and RecoveryCoordinator should gracefully handle their absence

### 4. **Delegation to StreamingChatHandler**
**Decision**: Keep existing delegation pattern
**Rationale**: Many recovery methods already delegate to StreamingChatHandler. RecoveryCoordinator wraps these delegations with additional logic (RL outcomes, logging, etc.)

---

## Testing Strategy

### Unit Tests
- Mock all dependencies (RecoveryHandler, StreamingChatHandler, etc.)
- Test each method in isolation
- Test error handling and edge cases
- Verify RecoveryContext is properly used

### Integration Tests
- Test DI resolution from container
- Test interaction between RecoveryCoordinator and StreamingChatHandler
- Test recovery integration flow end-to-end

### Regression Tests
- Run full orchestrator test suite
- Verify backward compatibility with deprecated methods
- Performance benchmarking to ensure no degradation

---

## Next Session Plan

### Immediate Tasks (Step 4: Update AgentOrchestrator)
1. Add RecoveryCoordinator initialization to `__init__`
2. Create `recovery_coordinator` property accessor
3. Update `_handle_stream_chunk` to use RecoveryCoordinator
4. Deprecate 29 `*_with_handler` methods with delegation wrappers
5. Update all call sites in streaming loop

### Test Creation (Step 5)
1. Create unit test file with 40-50 tests
2. Create DI resolution test file
3. Run tests and achieve >95% coverage

### Integration Testing (Step 6)
1. Run full test suite
2. Verify zero regressions
3. Performance benchmarking

---

## Success Metrics

### Quantitative (Current Progress)
- ✅ **Files Created**: 1 (recovery_coordinator.py)
- ✅ **Files Modified**: 3 (protocols.py, service_provider.py, orchestrator_factory.py)
- ✅ **Lines Added**: 988
- ✅ **Methods Extracted**: 23 (from 29 scattered methods)
- ⏳ **Tests Created**: 0 (pending)
- ⏳ **Test Coverage**: TBD

### Qualitative (Current Progress)
- ✅ **Single Responsibility**: RecoveryCoordinator has one clear purpose
- ✅ **Protocol-Based**: Type-safe dependency injection
- ✅ **DI Integration**: Properly registered in container
- ✅ **Factory Pattern**: Clean integration with OrchestratorFactory
- ⏳ **Testability**: Can be tested in isolation (tests pending)
- ⏳ **Backward Compatibility**: Deprecated methods maintain compatibility (pending)

---

## Estimated Completion

**Total Effort**: 8 hours (per plan)
**Completed**: 4 hours (50%)
**Remaining**: 4 hours (50%)

**Next Session**:
- Step 4: Update AgentOrchestrator (2 hours)
- Step 5: Create Tests (1.5 hours)
- Step 6: Integration Testing (0.5 hours)

**Expected Total Impact**:
- **Orchestrator Reduction**: ~1200 lines (from ~6000 to ~4800)
- **Method Reduction**: 29 methods extracted (from 154 to ~125)
- **Test Coverage**: 100% for RecoveryCoordinator
- **Zero Regressions**: All existing tests pass

---

## Conclusion

Phase 2A is **50% complete** with solid foundation laid:
- ✅ RecoveryCoordinator component created with 23 methods
- ✅ Protocol defined with clear interface
- ✅ DI registration complete
- ✅ Factory method integrated

The remaining work focuses on:
- Updating orchestrator to use RecoveryCoordinator
- Creating comprehensive test coverage
- Verifying zero regressions

This follows the proven pattern from CRITICAL-004 (DI migration), using protocols, DI container registration, and gradual deprecation to ensure zero breaking changes.

**Status**: ⏳ IN PROGRESS - On track for completion
**Next Checkpoint**: After Step 4 (AgentOrchestrator update)
