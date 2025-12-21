# Error Hierarchy Consolidation Audit

**Date**: 2025-12-20
**Phase**: CRITICAL-003 - Multiple Error Hierarchies
**Status**: Phase 1 Complete - Audit Finished

## Executive Summary

Found **4 error classes duplicated** across 3 modules:
- `ProviderError` (2 locations)
- `ToolError` (2 locations)
- `ConfigurationError` (2 locations)
- `ValidationError` (2 locations)

**Total**: 8 error class definitions that should be consolidated to 4.

## Detailed Findings

### 1. ProviderError (2 locations)

#### Location A: victor/core/errors.py:142
```python
class ProviderError(VictorError):
    """Errors related to LLM providers."""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        status_code: Optional[int] = None,
        raw_error: Optional[Any] = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.raw_error = raw_error
```

**Features**:
- Comprehensive error context (provider, model, status_code, raw_error)
- Inherits from VictorError (structured error with category, severity, context)
- 6 specialized subclasses: AuthError, TimeoutError, RateLimitError, ConnectionError, InvalidResponseError, NotFoundError

#### Location B: victor/framework/errors.py:40
```python
class ProviderError(AgentError):
    """Error from LLM provider communication."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        status_code: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.provider = provider
        self.status_code = status_code
```

**Features**:
- Simplified API (provider required, no model/raw_error fields)
- Inherits from AgentError (framework layer base)
- User-friendly error messages
- Custom `__str__` with `[provider]` prefix

**Usage**:
- Imported by: `victor/framework/agent.py`, `victor/framework/__init__.py`
- Used in: Framework's simplified Agent API

**Consolidation Impact**: LOW - Framework is a facade, can import and re-export

---

### 2. ToolError (2 locations)

#### Location A: victor/core/errors.py:300
```python
class ToolError(VictorError):
    """Base error for tool execution failures."""

    def __init__(
        self,
        message: str,
        tool_name: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        self.tool_name = tool_name
```

**Features**:
- 4 specialized subclasses: ExecutionError, ValidationError (for tools), NotFoundError, TimeoutError
- Structured error context from VictorError
- Optional tool_name attribute

#### Location B: victor/framework/errors.py:69
```python
class ToolError(AgentError):
    """Error from tool execution."""

    def __init__(
        self,
        message: str,
        *,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.tool_name = tool_name
        self.arguments = arguments or {}
```

**Features**:
- Includes arguments dict for debugging
- tool_name is required (not optional)
- Custom `__str__` with `[tool_name]` prefix

**Usage**:
- Imported by: `victor/framework/agent_components.py`, `victor/framework/__init__.py`

**Consolidation Impact**: LOW - Framework is a facade, can import and re-export

---

### 3. ConfigurationError (2 locations)

#### Location A: victor/core/errors.py:390
```python
class ConfigurationError(VictorError):
    """Configuration errors."""

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        self.config_key = config_key
```

**Features**:
- Tracks specific config_key that failed
- Structured context from VictorError

#### Location B: victor/framework/errors.py:96
```python
class ConfigurationError(AgentError):
    """Error in agent configuration."""

    def __init__(
        self,
        message: str,
        *,
        invalid_fields: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, recoverable=False, **kwargs)
        self.invalid_fields = invalid_fields or []
```

**Features**:
- Tracks multiple invalid_fields (list, not single key)
- Explicitly marked as non-recoverable
- Custom `__str__` with field listing

**Usage**:
- Imported by: `victor/framework/agent_components.py`, `victor/framework/__init__.py`

**Consolidation Impact**: MEDIUM - Different field tracking (config_key vs invalid_fields)

---

### 4. ValidationError (2 locations)

#### Location A: victor/core/errors.py:408
```python
class ValidationError(VictorError):
    """Validation errors."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        self.field = field
```

**Features**:
- Tracks specific field that failed validation
- General-purpose validation error

#### Location B: victor/core/cqrs.py:830
```python
class ValidationError(CQRSError):
    """Command validation error."""

    pass
```

**Features**:
- Minimal definition (no additional fields)
- CQRS-specific (command validation context)
- Inherits from CQRSError (line 806)

**Usage**:
- Only used in tests: `tests/unit/test_cqrs.py:453`
- CQRS module has its own error hierarchy (CQRSError, CommandError, QueryError, HandlerNotFoundError)

**Consolidation Impact**: HIGH - Name collision risk, CQRS needs namespacing

---

## Module Architecture Analysis

### victor/core/errors.py
**Role**: Comprehensive, production-grade error hierarchy for all Victor internals

**Error Hierarchy**:
```
VictorError (base)
├── ProviderError
│   ├── ProviderAuthError
│   ├── ProviderTimeoutError
│   ├── ProviderRateLimitError
│   ├── ProviderConnectionError
│   ├── ProviderInvalidResponseError
│   └── ProviderNotFoundError
├── ToolError
│   ├── ToolExecutionError
│   ├── ToolValidationError
│   ├── ToolNotFoundError
│   └── ToolTimeoutError
├── ConfigurationError
├── ValidationError
├── FileError
└── NetworkError
```

**Features**:
- ErrorCategory enum (PROVIDER, TOOL, CONFIG, VALIDATION, FILE, NETWORK)
- ErrorSeverity enum (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Structured context dict
- ErrorHandler utility class
- Backward compatibility aliases

**LOC**: ~450 lines

---

### victor/framework/errors.py
**Role**: Simplified, user-friendly error API for framework layer (90% use cases)

**Error Hierarchy**:
```
AgentError (base)
├── ProviderError ← DUPLICATE
├── ToolError ← DUPLICATE
├── ConfigurationError ← DUPLICATE
├── BudgetExhaustedError (framework-specific)
├── CancellationError (framework-specific)
└── StateTransitionError (framework-specific)
```

**Features**:
- User-friendly messages
- `recoverable` flag for retry logic
- Details dict for context
- Custom `__str__` formatting

**Used By**:
- `victor/framework/agent.py`
- `victor/framework/agent_components.py`
- `victor/framework/__init__.py` (public exports)
- `tests/unit/test_agent_components.py`

**LOC**: 190 lines

**Unique Errors** (not in core):
1. `BudgetExhaustedError` - Tool call budget tracking
2. `CancellationError` - User/system cancellation
3. `StateTransitionError` - State machine violations

---

### victor/core/cqrs.py
**Role**: CQRS pattern implementation (Command Query Responsibility Segregation)

**Error Hierarchy**:
```
CQRSError (base)
├── CommandError
├── QueryError
├── HandlerNotFoundError
└── ValidationError ← DUPLICATE (name collision)
```

**Features**:
- Self-contained error hierarchy
- Minimal definitions (pass-only)
- CQRS-specific context

**Usage**: Internal to CQRS module + tests

**LOC**: ~40 lines (errors section)

---

## Import Analysis

### Files Importing Framework Errors
```python
victor/framework/agent.py:26:
    from victor.framework.errors import AgentError, CancellationError, ProviderError

victor/framework/agent_components.py:41:
    from victor.framework.errors import AgentError, ConfigurationError

victor/framework/__init__.py:63:
    from victor.framework.errors import (
        AgentError,
        BudgetExhaustedError,
        CancellationError,
        ConfigurationError,
        ProviderError,
        StateTransitionError,
        ToolError,
    )

tests/unit/test_agent_components.py:26:
    from victor.framework.errors import AgentError, ConfigurationError
```

**Total**: 4 files importing framework errors

### Files Importing Core Errors
```python
victor/providers/base.py:84:
    from victor.core.errors import (
        ProviderError,
        ProviderNotFoundError,
        ProviderTimeoutError,
        ProviderRateLimitError,
        ProviderAuthError as ProviderAuthenticationError,
        ProviderConnectionError,
        ProviderInvalidResponseError,
    )

tests/unit/test_ollama_provider_full.py:10:
    from victor.providers.base import Message, ProviderError, ProviderTimeoutError, ToolDefinition
```

**Note**: test imports from providers/base.py which re-exports from core/errors.py

---

## Consolidation Recommendations

### Priority 1: Framework Errors (ProviderError, ToolError, ConfigurationError)

**Strategy**: Import and re-export from core

**Rationale**:
1. Framework is a facade layer - should delegate to core
2. Maintains backward compatibility for framework API users
3. Zero breaking changes for existing code
4. Reduces maintenance burden (single source of truth)

**Implementation**:
```python
# victor/framework/errors.py (refactored)

from victor.core.errors import (
    ProviderError as CoreProviderError,
    ToolError as CoreToolError,
    ConfigurationError as CoreConfigurationError,
)

class AgentError(Exception):
    """Base exception for agent errors."""
    # Keep existing implementation

# Re-export core errors with framework-friendly wrappers
class ProviderError(CoreProviderError, AgentError):
    """Error from LLM provider communication.

    This is a wrapper around victor.core.errors.ProviderError
    for backward compatibility with the framework API.
    """
    pass

# Similar for ToolError and ConfigurationError

# Keep framework-specific errors
class BudgetExhaustedError(AgentError):
    # Existing implementation

class CancellationError(AgentError):
    # Existing implementation

class StateTransitionError(AgentError):
    # Existing implementation
```

**Impact**:
- Reduces victor/framework/errors.py from 190 → ~120 LOC
- Zero breaking changes (same import paths work)
- Framework errors inherit from both core error and AgentError (multiple inheritance)

---

### Priority 2: CQRS ValidationError

**Strategy**: Rename to avoid name collision

**Rationale**:
1. CQRS ValidationError is minimal (pass-only)
2. Only used in tests (low impact)
3. Name collision with core.ValidationError is confusing
4. CQRS has its own self-contained error hierarchy

**Implementation**:
```python
# victor/core/cqrs.py (refactored)

class CQRSError(Exception):
    """Base exception for CQRS errors."""
    pass

class CommandValidationError(CQRSError):  # Renamed from ValidationError
    """Command validation error."""
    pass

# Update test usage
# tests/unit/test_cqrs.py:453
raise CommandValidationError("Invalid email")  # Was: raise ValidationError(...)
```

**Impact**:
- Removes name collision
- 1 file to update (test file)
- Clear naming (CommandValidationError vs generic ValidationError)

---

## Testing Impact Analysis

### Current Test Status
- **Total**: 9,424 tests
- **Passing**: 9,354 (99.3%)
- **Failing**: 70

### Files Requiring Test Updates

1. **Framework error tests**:
   - `tests/unit/test_agent_components.py` (already imports framework errors)
   - No changes needed (backward compatible re-exports)

2. **CQRS error tests**:
   - `tests/unit/test_cqrs.py:453` - Update ValidationError → CommandValidationError
   - Search for other CQRS ValidationError usages

3. **Provider error tests**:
   - `tests/unit/test_ollama_provider_full.py` (imports from providers/base.py)
   - No changes needed (providers/base.py already imports from core/errors.py)

**Expected Impact**:
- CQRS tests: 1-5 test updates (low risk)
- Framework tests: 0 updates (backward compatible)
- Provider tests: 0 updates (already using core errors)

---

## Migration Plan

### Phase 2: Consolidate Framework Errors (3-4 hours)

**Step 1**: Refactor victor/framework/errors.py
- Import ProviderError, ToolError, ConfigurationError from core
- Create wrapper classes with multiple inheritance
- Keep framework-specific errors (BudgetExhaustedError, etc.)
- Run tests: `pytest tests/unit/test_agent_components.py -v`

**Step 2**: Verify backward compatibility
- Check all 4 import sites still work
- Ensure error instances pass isinstance() checks for both base classes
- Run framework tests: `pytest tests/unit/test_framework*.py -v`

**Step 3**: Update documentation
- Add docstrings explaining wrapper pattern
- Document error hierarchy in ERROR_CONSOLIDATION_AUDIT.md

**Verification**:
```bash
pytest tests/unit/test_framework*.py tests/unit/test_agent_components.py -v
```

---

### Phase 3: Consolidate CQRS ValidationError (1-2 hours)

**Step 1**: Rename in victor/core/cqrs.py
- ValidationError → CommandValidationError
- Update docstring

**Step 2**: Update test usage
- Search: `grep -r "ValidationError" tests/unit/test_cqrs.py`
- Replace with CommandValidationError

**Step 3**: Run tests
```bash
pytest tests/unit/test_cqrs.py -v
```

**Verification**:
```bash
# Ensure no more name collisions
grep -r "class ValidationError" victor/
# Should only find victor/core/errors.py:408
```

---

### Phase 4: Full Test Suite (30 minutes)

**Step 1**: Run complete test suite
```bash
pytest --cov --cov-report=html
```

**Step 2**: Verify no regressions
- Check test pass rate remains ≥99%
- Review any new failures
- Ensure error handling paths still work

**Step 3**: Update ARCHITECTURE_IMPROVEMENT_PLAN.md
- Mark CRITICAL-003 as COMPLETED
- Document LOC reduction
- Update status to green

---

## Expected Outcomes

### LOC Reduction
- **victor/framework/errors.py**: 190 → 120 LOC (-70 LOC)
- **victor/core/cqrs.py**: 0 net change (rename only)
- **Total**: -70 LOC

### Maintenance Benefits
- Single source of truth for ProviderError, ToolError, ConfigurationError
- No more duplicate definitions to keep in sync
- Clear separation: core errors vs framework wrappers vs CQRS errors

### Code Quality
- Zero breaking changes (backward compatible)
- Removes name collision (ValidationError)
- Clearer error hierarchy

### Test Coverage
- All existing tests continue to pass
- No new test gaps introduced
- Error handling paths preserved

---

## Risk Assessment

### Low Risk
- Framework error consolidation (backward compatible re-exports)
- Provider error imports (already using core)

### Medium Risk
- CQRS ValidationError rename (requires test updates)
- Multiple inheritance for framework wrappers (ensure isinstance() works)

### Mitigation
- Run tests after each phase
- Keep git commits atomic (easy rollback)
- Verify all import sites before and after
- Test isinstance() checks explicitly

---

## Appendix: Error Definitions

### Core Errors (victor/core/errors.py)
```python
VictorError(Exception)                    # Line 23
├── ProviderError(VictorError)            # Line 142
│   ├── ProviderAuthError                 # Line 194
│   ├── ProviderTimeoutError              # Line 206
│   ├── ProviderRateLimitError            # Line 218
│   ├── ProviderConnectionError           # Line 230
│   ├── ProviderInvalidResponseError      # Line 242
│   └── ProviderNotFoundError             # Line 254
├── ToolError(VictorError)                # Line 300
│   ├── ToolExecutionError                # Line 330
│   ├── ToolValidationError               # Line 342
│   ├── ToolNotFoundError                 # Line 354
│   └── ToolTimeoutError                  # Line 366
├── ConfigurationError(VictorError)       # Line 390
├── ValidationError(VictorError)          # Line 408
├── FileError(VictorError)                # Line 426
└── NetworkError(VictorError)             # Line 444
```

### Framework Errors (victor/framework/errors.py)
```python
AgentError(Exception)                     # Line 12
├── ProviderError(AgentError)             # Line 40 ← DUPLICATE
├── ToolError(AgentError)                 # Line 69 ← DUPLICATE
├── ConfigurationError(AgentError)        # Line 96 ← DUPLICATE
├── BudgetExhaustedError(AgentError)      # Line 123
├── CancellationError(AgentError)         # Line 150
└── StateTransitionError(AgentError)      # Line 165
```

### CQRS Errors (victor/core/cqrs.py)
```python
CQRSError(Exception)                      # Line 806
├── CommandError(CQRSError)               # Line 812
├── QueryError(CQRSError)                 # Line 818
├── HandlerNotFoundError(CQRSError)       # Line 824
└── ValidationError(CQRSError)            # Line 830 ← DUPLICATE
```

---

## Conclusion

**Audit Status**: ✅ COMPLETE

**Duplicates Found**: 4 error classes across 8 definitions

**Recommended Approach**:
1. Framework errors → Import and re-export from core (backward compatible)
2. CQRS ValidationError → Rename to CommandValidationError (low impact)

**Next Phase**: Proceed to Phase 2 - Consolidation Implementation

**Estimated Time**: 4-6 hours total (3-4 hours framework + 1-2 hours CQRS)

**Risk Level**: LOW (backward compatible strategy, atomic commits)
