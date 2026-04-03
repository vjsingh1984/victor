# Known Issues - Victor v0.6.0

**Date**: 2026-03-31
**Version**: 0.6.0
**Status**: Pre-release

---

## Issue #1: Class Name Construction in Extension Loader

**Severity**: Medium
**Impact**: Contrib verticals (built-in) only
**External Verticals**: ✅ Not affected

### Description

The extension loader has a bug where it uses `canonical_name` (lowercase) to construct class names for runtime extensions, resulting in incorrect camelCase class names instead of proper PascalCase.

### Affected Components

**Contrib verticals only** (victor.verticals.contrib):
- victor-coding
- victor-dataanalysis
- victor-devops
- victor-rag
- victor-research

**External verticals** are NOT affected because they define their classes with proper names and don't rely on auto-generated class names.

### Example of Bug

```python
# In extension_loader.py, line 1086-1090
metadata = VerticalMetadata.from_class(cls)
vertical_name = metadata.canonical_name  # "rag" (lowercase)
return cls._resolve_class_or_factory_extension(
    "rl_config_provider",
    "rl",
    class_name=f"{vertical_name}RLConfig",  # "ragRLConfig" (WRONG)
)
```

Should construct `RAGRLConfig` but instead constructs `ragRLConfig`.

### Root Cause

The `canonical_name` field is intentionally lowercase for consistency (e.g., "rag", "coding", "devops"), but class names require proper PascalCase (e.g., "RAG", "Coding", "DevOps").

### Test Failures

14 tests in `tests/unit/core/verticals/test_runtime_helper_defaults.py` fail due to this issue:

```
FAILED test_capability_provider_autoloads_for_verticals_using_default_loader
  AttributeError: module '...devops.capabilities' has no attribute 'devopsCapabilityProvider'
  Did you mean: 'DevOpsCapabilityProvider'?

FAILED test_rag_runtime_extensions_resolve_via_shared_loader_defaults
  AttributeError: module '...rag.rl' has no attribute 'ragRLConfig'
  Did you mean: 'RAGRLConfig'?
```

### Workaround

No workaround needed for external verticals. Contrib verticals should use explicit class names in their extension factories rather than relying on auto-generation.

### Fix Required

Update `victor/core/verticals/extension_loader.py` to extract class name prefix from `qualname` instead of using `canonical_name`:

```python
# Current (incorrect):
metadata = VerticalMetadata.from_class(cls)
vertical_name = metadata.canonical_name  # lowercase
class_name = f"{vertical_name}{extension_type}"  # camelCase

# Fixed:
metadata = VerticalMetadata.from_class(cls)
# Extract prefix from class name (preserves case)
prefix = metadata.qualname.replace("Assistant", "").replace("Vertical", "")
class_name = f"{prefix}{extension_type}"  # PascalCase
```

### Timeline

- **v0.6.0**: Issue exists, documented in release notes
- **v0.6.1**: Fix scheduled for next minor release

### Related Files

- `victor/core/verticals/extension_loader.py`: Lines 1086-1090, 1102-1106
- `victor/core/verticals/vertical_metadata.py`: Add `class_prefix` property
- `tests/unit/core/verticals/test_runtime_helper_defaults.py`: 14 failing tests

---

## Issue #2: Conversation Coordinator Internal API Usage

**Severity**: Low
**Impact**: victor-dataanalysis, victor-rag

### Description

The `conversation_enhanced.py` files in victor-dataanalysis and victor-rag use internal `ConversationCoordinator` from `victor.agent.coordinators`, which is not part of the public SDK.

### Status

✅ **Documented**: Files include comments marking this as internal API
⏳ **TODO**: Refactor to use framework conversation protocols when available

### Files Affected

- `victor-dataanalysis/victor_dataanalysis/conversation_enhanced.py`
- `victor-rag/victor_rag/conversation_enhanced.py`

### Fix Plan

Refactor these files to use framework-level conversation protocols when they become available. For now, documented as accepted internal dependency.

---

## Summary

- **2 known issues** documented
- **External verticals**: Fully aligned, no issues
- **Contrib verticals**: 1 class naming bug (needs fix in v0.6.1)
- **Conversation coordinators**: Documented for future refactoring

All issues are non-blocking for v0.6.0 release. External verticals are production-ready.
