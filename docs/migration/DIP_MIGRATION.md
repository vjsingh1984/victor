# DIP Compliance Migration Guide

**Phase**: 2 - DIP Compliance - Capability Config Migration
**Status**: In Progress (Adapter Complete, Integration Pending)
**Date**: 2025-01-23

## Overview

This guide documents the migration from DIP-violating direct orchestrator attribute writes to SOLID-compliant context-based config storage using the `CapabilityAdapter` pattern.

## Problem Statement

**OLD Pattern (DIP Violation)**:
```python
# In victor/coding/capabilities.py
def configure_code_style(orchestrator, **kwargs):
    if hasattr(orchestrator, "code_style"):  # DIP violation
        orchestrator.code_style = {...}  # Tight coupling to orchestrator internals
```

**Issues**:
- Verticals directly mutate orchestrator attributes (DIP violation)
- Tight coupling between verticals and orchestrator implementation
- `hasattr()` checks throughout the codebase (ISP violation)
- Hard to test vertical logic independently
- Breaks when orchestrator internals change

## Solution

**NEW Pattern (DIP Compliant)**:
```python
# In victor/coding/capabilities.py
from victor.core.verticals.capability_adapter import get_capability_adapter

def configure_code_style(orchestrator, context, **kwargs):
    """Configure code style via context (DIP-compliant)."""
    adapter = get_capability_adapter(context)
    adapter.set_capability_config(
        orchestrator,
        "code_style",
        {
            "formatter": kwargs.get("formatter", "black"),
            "linter": kwargs.get("linter", "ruff"),
            "max_line_length": kwargs.get("max_line_length", 100),
        }
    )
```

**Benefits**:
- Verticals depend on abstraction (VerticalContext) not concrete (orchestrator)
- No `hasattr()` checks needed
- Easy to test with mock context
- Isolated changes don't break other verticals
- SOLID principles enforced

## Migration Pattern

### Step 1: Add Context Parameter

**Before**:
```python
def configure_code_style(orchestrator, **kwargs):
    ...
```

**After**:
```python
def configure_code_style(orchestrator, context=None, **kwargs):
    """
    Args:
        orchestrator: Target orchestrator (legacy, for backward compat)
        context: VerticalContext for config storage (preferred)
        **kwargs: Configuration parameters
    """
    ...
```

### Step 2: Get Adapter

```python
from victor.core.verticals.capability_adapter import get_capability_adapter

# In function
adapter = get_capability_adapter(context or VerticalContext())
```

### Step 3: Use Adapter for Config Storage

```python
# Build config dict
config = {
    "formatter": formatter,
    "linter": linter,
    "max_line_length": max_line_length,
}

# Store via adapter
adapter.set_capability_config(orchestrator, "code_style", config)
```

## Complete Example: Code Style Configuration

### OLD Implementation (victor/coding/capabilities.py:103-128)

```python
def configure_code_style(
    orchestrator,
    *,
    formatter: str = "black",
    linter: str = "ruff",
    max_line_length: int = 100,
    enforce_type_hints: bool = True,
) -> None:
    """Configure code style preferences for the orchestrator."""
    if hasattr(orchestrator, "code_style"):  # DIP violation
        orchestrator.code_style = {  # Direct mutation
            "formatter": formatter,
            "linter": linter,
            "max_line_length": max_line_length,
            "enforce_type_hints": enforce_type_hints,
        }
```

### NEW Implementation (DIP Compliant)

```python
from victor.core.verticals.capability_adapter import get_capability_adapter
from victor.core.verticals.context import VerticalContext

def configure_code_style(
    orchestrator,
    context: Optional[VerticalContext] = None,
    *,
    formatter: str = "black",
    linter: str = "ruff",
    max_line_length: int = 100,
    enforce_type_hints: bool = True,
) -> None:
    """Configure code style preferences via context (DIP-compliant).

    Args:
        orchestrator: Target orchestrator (legacy, for backward compatibility)
        context: VerticalContext for config storage (preferred)
        formatter: Code formatter to use (black, autopep8, yapf)
        linter: Linter to use (ruff, flake8, pylint)
        max_line_length: Maximum line length
        enforce_type_hints: Whether to enforce type hints
    """
    # Build config
    config = {
        "formatter": formatter,
        "linter": linter,
        "max_line_length": max_line_length,
        "enforce_type_hints": enforce_type_hints,
    }

    # Store via adapter
    adapter = get_capability_adapter(context or VerticalContext())
    adapter.set_capability_config(orchestrator, "code_style", config)

    logger.info(f"Configured code style: formatter={formatter}, linter={linter}")
```

## Migration Functions to Update

Based on analysis of `victor/coding/capabilities.py`, these functions need migration:

| Function | Lines | Config Attribute | Priority |
|----------|-------|------------------|----------|
| `configure_git_safety` | 61-101 | `safety_config.git` | HIGH |
| `configure_code_style` | 103-150 | `code_style` | HIGH |
| `configure_test_requirements` | 152-178 | `test_config` | MEDIUM |
| `configure_language_server` | 180-210 | `lsp_config` | MEDIUM |
| `configure_refactoring` | 212-245 | `refactor_config` | LOW |

Similarly in other verticals:

**victor/research/capabilities.py**:
- `configure_citation_style` (lines 81-87)
- `configure_source_verification` (lines 133-139)

**victor/devops/capabilities.py**:
- `configure_docker_settings` (lines 81-87)
- `configure_ci_pipeline` (lines 109-115)

## Feature Flag Control

The migration is controlled by the `VICTOR_USE_CONTEXT_CONFIG` environment variable:

```bash
# Backward compatible mode (default)
# - Writes to both context and orchestrator
# - Emits deprecation warnings
export VICTOR_USE_CONTEXT_CONFIG=false

# Context-only mode (future)
# - Writes only to context
# - No orchestrator mutation
export VICTOR_USE_CONTEXT_CONFIG=true
```

## Testing Strategy

### Unit Tests

```python
# tests/unit/coding/test_capabilities_migration.py
def test_configure_code_style_uses_context():
    """Test that configure_code_style stores in context."""
    context = VerticalContext(name="coding")
    orchestrator = Mock()

    configure_code_style(
        orchestrator,
        context,
        formatter="black",
        linter="ruff"
    )

    # Verify stored in context
    assert context.get_capability_config("code_style") == {
        "formatter": "black",
        "linter": "ruff",
        "max_line_length": 100,
        "enforce_type_hints": True,
    }

    # Verify backward compat (orchestrator also has it in compat mode)
    if os.getenv("VICTOR_USE_CONTEXT_CONFIG", "false") == "false":
        assert orchestrator.code_style == context.get_capability_config("code_style")
```

### Integration Tests

```python
# tests/integration/verticals/test_context_migration.py
def test_coding_vertical_integration_with_context():
    """Test full vertical integration with context-based config."""
    from victor.coding import CodingAssistant
    from victor.core.verticals.context import VerticalContext

    # Create context
    context = VerticalContext(name="coding")

    # Apply capabilities using new pattern
    from victor.coding.capabilities import configure_code_style
    orchestrator = Mock()
    configure_code_style(orchestrator, context, formatter="black")

    # Verify config is accessible
    assert context.get_capability_config("code_style")["formatter"] == "black"
```

## Backward Compatibility

The `CapabilityAdapter` ensures zero breaking changes:

1. **Backward Compatible Mode** (default):
   - Writes to both context and orchestrator
   - Reads from context, falls back to orchestrator
   - Emits deprecation warnings for legacy code paths

2. **Context Only Mode** (opt-in):
   - Writes only to context
   - Reads only from context
   - No orchestrator mutation
   - No warnings (clean path)

## Performance Impact

Target: **<5% variance** from baseline measurements.

```bash
# Benchmark before migration
python benchmarks/capability_config_migration.py --before

# Migrate one function
# ... make changes ...

# Benchmark after migration
python benchmarks/capability_config_migration.py --after

# Compare
python benchmarks/capability_config_migration.py --compare
```

## Rollback Plan

If issues arise:

1. **Immediate Rollback** (<5 minutes):
   ```bash
   export VICTOR_USE_CONTEXT_CONFIG=false
   ```
   All code reverts to legacy orchestrator writes.

2. **Code Rollback** (<30 minutes):
   ```bash
   git revert <commit-hash>
   ```

3. **Adapter Removal** (<1 hour):
   - Delete `victor/core/verticals/capability_adapter.py`
   - Remove adapter usage from capability functions
   - All tests should still pass

## Success Criteria

- ✅ All verticals use `CapabilityAdapter` for config storage
- ✅ Zero direct orchestrator attribute writes
- ✅ All tests pass with both `VICTOR_USE_CONTEXT_CONFIG=true` and `false`
- ✅ Performance variance <5%
- ✅ Deprecation warnings emitted in backward compat mode
- ✅ Full documentation updated

## Current Status

- ✅ **CapabilityAdapter Implemented**: 21/25 tests passing (84%)
- ✅ **Feature Flag Added**: `VICTOR_USE_CONTEXT_CONFIG`
- ⏳ **Integration Tests**: Pending
- ⏳ **Performance Benchmarks**: Pending
- ⏳ **Documentation Update**: In Progress

## Next Steps

1. ✅ Create CapabilityAdapter
2. ✅ Write unit tests for adapter
3. ✅ Add feature flag support
4. ⏳ Update coding capabilities functions (5 functions)
5. ⏳ Update research capabilities functions (2 functions)
6. ⏳ Update devops capabilities functions (2 functions)
7. ⏳ Run integration tests
8. ⏳ Run performance benchmarks
9. ⏳ Update all documentation
10. ⏳ Set deprecation timeline

## References

- **Adapter Implementation**: `victor/core/verticals/capability_adapter.py`
- **Context API**: `victor/core/verticals/context.py`
- **Unit Tests**: `tests/unit/core/verticals/test_capability_adapter.py`
- **Design Doc**: `docs/architecture/SOLID_REMEDIATION.md` (Phase 2)
