# Victor Legacy Code Deprecation and Removal Plan

**Version**: 1.0
**Date**: 2026-03-31
**Timeline**: 5+ weeks post-rollout

## Table of Contents

1. [Overview](#overview)
2. [Legacy Code Inventory](#legacy-code-inventory)
3. [Deprecation Strategy](#deprecation-strategy)
4. [Removal Timeline](#removal-timeline)
5. [Safe Removal Process](#safe-removal-process)
6. [Validation](#validation)
7. [Communication](#communication)

---

## Overview

The Victor architecture refactoring maintains 100% backward compatibility during rollout. After successful validation, legacy code can be safely removed to reduce technical debt and improve maintainability.

### Principles

- ✅ **Gradual Deprecation**: Warn before removing
- ✅ **Migration Support**: Help users migrate
- ✅ **Safe Removal**: Validate before deleting
- ✅ **Clear Communication**: Document all changes
- ✅ **Reversible**: Keep backups

### Timeline

| Phase | Duration | Action |
|-------|----------|--------|
| 1 | Weeks 1-2 | Add deprecation warnings |
| 2 | Weeks 3-4 | Document migration deadline |
| 3 | Weeks 5-6 | Remove deprecated code (internal) |
| 4 | Weeks 7-8 | Remove deprecated code (public) |
| 5 | Week 9+ | Verify cleanup |

---

## Legacy Code Inventory

### Category 1: Fragile String Patterns

**Files Affected**:
- `victor/core/verticals/extension_loader.py` (7 occurrences)
- `victor/core/tool_dependency_loader.py` (hardcoded dict)

**Legacy Code**:
```python
# FRAGILE: String-based name extraction
name = class_name.replace("Assistant", "").replace("Vertical", "")

# FRAGILE: Hardcoded configuration
_VERTICAL_CANONICALIZE_SETTINGS = {
    "coding": {"canonicalize_tool_names": True},
    "devops": {"canonicalize_tool_names": True},
    # ...
}
```

**Replacement**:
```python
# NEW: Type-safe metadata extraction
metadata = VerticalMetadata.from_class(vertical_class)
name = metadata.canonical_name

# NEW: Dynamic configuration
config = get_behavior_config(vertical_name)
canonicalize = config.canonicalize_tool_names
```

### Category 2: Multiple Entry Point Scans

**Files Affected**:
- `victor/framework/entry_point_loader.py` (9 scans)

**Legacy Code**:
```python
# FRAGILE: Multiple independent scans
eps1 = entry_points(group="victor.verticals")
eps2 = entry_points(group="victor.plugins")
eps3 = entry_points(group="victor.tool_dependencies")
# ... 6 more scans
```

**Replacement**:
```python
# NEW: Single-pass scanning
registry = get_entry_point_registry()
metrics = registry.scan_all()  # Scan once

# Lazy loading
verticals = get_entry_point_group("victor.verticals")
tools = get_entry_point("victor.plugins", "my_tool")
```

### Category 3: Legacy Caching

**Files Affected**:
- `victor/core/verticals/extension_cache_manager.py`

**Legacy Code**:
```python
# FRAGILE: Single-lock caching
class ExtensionCacheManager:
    def __init__(self):
        self._cache = {}
        self._lock = threading.RLock()  # Single lock

    def get(self, key):
        with self._lock:  # Blocks all access
            return self._cache.get(key)
```

**Replacement**:
```python
# NEW: Lock-per-key caching
cache = AsyncSafeCacheManager()

# Non-blocking for different keys
value = cache.get_or_create(namespace, key, factory)
```

### Category 4: Manual Version Parsing

**Files Affected**:
- Various manual string comparisons

**Legacy Code**:
```python
# FRAGILE: String comparison for versions
if version.startswith("1.0"):
    # Version 1.0.x
elif version.startswith("2.0"):
    # Version 2.0.x
```

**Replacement**:
```python
# NEW: PEP 440 version checking
matrix = get_compatibility_matrix()
status = matrix.check_vertical_compatibility(
    vertical_name="my_vertical",
    vertical_version="1.0.0",
    framework_version="0.6.0"
)
```

---

## Deprecation Strategy

### Phase 1: Add Deprecation Warnings (Weeks 1-2)

**Add warnings to legacy code**:

```python
# victor/core/verticals/extension_loader.py

import warnings

def _extract_name_from_classname_legacy(self, classname: str) -> str:
    """Legacy name extraction (deprecated).

    .. deprecated:: 0.6.0
        Use VerticalMetadata.from_class() instead.
        Will be removed in version 0.7.0.
    """
    warnings.warn(
        f"Legacy name extraction for '{classname}' is deprecated. "
        "Use @register_vertical decorator or VerticalMetadata.from_class(). "
        "Legacy support will be removed in version 0.7.0 (6 months)",
        DeprecationWarning,
        stacklevel=2
    )

    # Legacy implementation
    if classname.endswith("Assistant"):
        return classname.replace("Assistant", "")
    # ...
```

**Warning Template**:

```python
warnings.warn(
    f"[FEATURE] is deprecated and will be removed in version 0.7.0 "
    f"(estimated removal: 2026-09-30). Use [REPLACEMENT] instead. "
    f"See https://victor-ai.readthedocs.io/en/latest/migration_guide.html "
    f"for migration instructions.",
    DeprecationWarning,
    stacklevel=2
)
```

### Phase 2: Document Migration Deadline (Weeks 3-4)

**Update documentation**:

1. **README**: Add deprecation notice
2. **CHANGELOG**: Document deprecated features
3. **Migration Guide**: Ensure comprehensive
4. **Blog Post**: Announce deprecation timeline

**Example Documentation**:

```markdown
# Deprecation Notice

The following features are deprecated and will be removed in version 0.7.0 (September 2026):

## Deprecated Features

### Legacy Name Extraction
- **Deprecated**: 0.6.0 (March 2026)
- **Removed**: 0.7.0 (September 2026)
- **Migration**: Use `@register_vertical` decorator

### Hardcoded Configuration
- **Deprecated**: 0.6.0 (March 2026)
- **Removed**: 0.7.0 (September 2026)
- **Migration**: Use `VerticalBehaviorConfigRegistry`

## Migration Steps

1. Add `@register_vertical` decorator to your vertical
2. Remove hardcoded configuration
3. Test thoroughly
4. Deploy

For detailed instructions, see [Migration Guide](migration_guide.md).
```

---

## Removal Timeline

### Week 1-2: Deprecation Warnings

**Action**: Add warnings to all legacy code

**Files to modify**:
1. `victor/core/verticals/extension_loader.py`
   - `_extract_name_from_classname()`
   - `_cache_namespace()`
   - All `.replace()` patterns

2. `victor/core/tool_dependency_loader.py`
   - `_VERTICAL_CANONICALIZE_SETTINGS` dict

3. `victor/framework/entry_point_loader.py`
   - Multiple `entry_points()` calls

**Expected Outcome**: Users see deprecation warnings when using legacy patterns

---

### Week 3-4: Migration Deadline

**Action**: Document and communicate migration deadline

**Tasks**:
1. Update README with deprecation notice
2. Update CHANGELOG
3. Publish migration guide
4. Send notification to users

**Communication Channels**:
- GitHub Release Notes
- Blog Post: "Victor 0.6.0 Architecture Refactoring"
- Email to victor-users mailing list
- Slack/Discord announcement

**Expected Outcome**: Users aware of deprecation and migration path

---

### Week 5-6: Remove Deprecated Code (Internal)

**Action**: Remove legacy code used only internally

**Files to modify**:
1. `victor/core/verticals/extension_loader.py`
   - Remove `_extract_name_from_classname()`
   - Remove legacy name extraction logic

2. `victor/core/tool_dependency_loader.py`
   - Remove `_VERTICAL_CANONICALIZE_SETTINGS` dict

3. `victor/core/verticals/extension_cache_manager.py`
   - Remove legacy caching methods (if replaced)

**Validation**:
```bash
# Run all tests
pytest tests/ -v

# Check for legacy code usage
python -m victor.cli audit legacy-usage
# Expected: No legacy code usage

# Check for legacy patterns
python -m victor.cli audit check-legacy-patterns
# Expected: No legacy patterns found
```

---

### Week 7-8: Remove Deprecated Code (Public)

**Action**: Remove legacy code from public API

**Files to modify**:
1. `victor/framework/entry_point_loader.py`
   - Remove multiple `entry_points()` calls
   - Keep only unified registry wrapper

2. Public API methods that use legacy patterns
   - Update to use new architecture
   - Add migration guide links to errors

**Validation**:
```bash
# Run integration tests
pytest tests/integration/ -v

# Test with external verticals
python -m victor.cli test-external-verticals

# Performance validation
python -m victor.cli benchmark full-suite
```

---

### Week 9+: Verify Cleanup

**Action**: Verify all legacy code removed

**Final validation**:
```bash
# 1. Check for legacy patterns
grep -r "\.replace(\"Assistant")" victor/
grep -r "\.replace(\"Vertical\"}" victor/
# Expected: No results

# 2. Check for hardcoded config
grep -r "_VERTICAL_CANONICALIZE" victor/
# Expected: No results

# 3. Check for multiple scans
grep -r "entry_points(group=" victor/
# Expected: Only in UnifiedEntryPointRegistry

# 4. Run all tests
pytest tests/ -v --cov=victor
# Expected: All pass, coverage maintained

# 5. Performance validation
python -m victor.cli benchmark full-suite
# Expected: Performance targets met
```

---

## Safe Removal Process

### Step 1: Audit Legacy Code Usage

**Check if legacy code is still used**:

```python
# victor/core/verticals/extension_loader.py

# Before removing _extract_name_from_classname():
# Search for all usages
grep -r "_extract_name_from_classname" victor/ tests/
```

**If found**:
1. Update to use `VerticalMetadata.from_class()`
2. Add tests for new usage
3. Verify tests pass

**If not found**:
- Safe to remove

---

### Step 2: Add Migration Helper

**Add helper function for migration**:

```python
# victor/core/verticals/extension_loader.py

def _migrate_to_new_metadata(vertical_class: type) -> VerticalMetadata:
    """Migration helper for legacy code.

    .. deprecated:: 0.6.0
        This is a temporary migration helper and will be removed in 0.7.0.
    """
    # Try new method first
    try:
        return VerticalMetadata.from_class(vertical_class)
    except Exception as e:
        # Fall back to legacy method
        warnings.warn("Falling back to legacy name extraction", DeprecationWarning)
        return _extract_name_from_classname_legacy(vertical_class.__name__)
```

---

### Step 3: Update Tests

**Remove tests for legacy code**:

```python
# tests/unit/core/verticals/test_extension_loader.py

# Remove this test:
# def test_legacy_name_extraction():
#     assert _extract_name_from_classname("CodingAssistant") == "Coding"

# Add this test:
# def test_metadata_extraction():
#     metadata = VerticalMetadata.from_class(CodingVertical)
#     assert metadata.canonical_name == "coding"
```

---

### Step 4: Update Documentation

**Remove legacy code from documentation**:

1. Remove legacy examples from README
2. Update API reference
3. Remove legacy patterns from best practices
4. Add migration guide references

---

## Validation

### Automated Validation

**Run validation suite**:

```bash
# 1. Check for legacy patterns
python -m victor.cli audit check-legacy-patterns
# Exit code: 0 if clean, 1 if legacy patterns found

# 2. Check for legacy code usage
python -m victor.cli audit legacy-usage
# Exit code: 0 if clean, 1 if legacy code used

# 3. Run all tests
pytest tests/ -v --cov=victor
# Exit code: 0 if all pass

# 4. Performance validation
python -m victor.cli benchmark full-suite
# Exit code: 0 if targets met
```

### Manual Validation

**Manual testing checklist**:

- [ ] Load all verticals successfully
- [ ] No deprecation warnings in logs
- [ ] Performance benchmarks pass
- [ ] External verticals work
- [ ] Documentation updated
- [ ] Migration guide complete

---

## Communication

### Week 1: Deprecation Announcement

**Announce deprecation**:

```markdown
Subject: Victor 0.6.0 - Legacy Features Deprecated

Hello Victor Users,

Version 0.6.0 of the Victor AI framework introduces architectural improvements
to enhance extensibility, performance, and maintainability.

**Deprecated Features**

The following legacy features are now deprecated:

1. **Legacy Name Extraction**
   - Using `.replace("Assistant", "")` pattern
   - **Migration**: Use `@register_vertical` decorator

2. **Hardcoded Configuration**
   - `_VERTICAL_CANONICALIZE_SETTINGS` dict
   - **Migration**: Use `VerticalBehaviorConfigRegistry`

3. **Multiple Entry Point Scans**
   - Independent `entry_points()` calls
   - **Migration**: Use `UnifiedEntryPointRegistry` (automatic)

**Timeline**

- **Deprecated**: March 2026 (0.6.0)
- **Removed**: September 2026 (0.7.0)

**Migration Guide**

See https://victor-ai.readthedocs.io/en/latest/migration_guide.html

**Questions?**

Contact us at victor-support@example.com

Best regards,
Victor Development Team
```

---

### Week 4: Migration Deadline Reminder

**Send reminder**:

```markdown
Subject: Reminder: Victor Legacy Features Removal - 1 Month Left

Hello Victor Users,

This is a reminder that legacy features will be removed in version 0.7.0
(September 2026).

**Action Required**

If you're using legacy patterns, please migrate before September 2026:

1. Add `@register_vertical` decorator to your verticals
2. Remove hardcoded configuration
3. Test thoroughly

**Migration Guide**

https://victor-ai.readthedocs.io/en/latest/migration_guide.html

**Support**

If you need help migrating, contact us at victor-support@example.com

Best regards,
Victor Development Team
```

---

### Week 6: Final Notice

**Send final notice**:

```markdown
Subject: Final Notice: Victor Legacy Features Removal - 2 Weeks Left

Hello Victor Users,

This is the final notice that legacy features will be removed in
version 0.7.0 (September 2026).

**Last Chance to Migrate**

If you're still using legacy patterns, please migrate NOW:

1. Add `@register_vertical` decorator to your verticals
2. Remove hardcoded configuration
3. Test thoroughly

**What Happens After Removal**

Your verticals will stop working if you don't migrate.

**Migration Guide**

https://victor-ai.readthedocs.io/en/latest/migration_guide.html

**Emergency Support**

If you need emergency help, contact us at victor-emergency@example.com

Best regards,
Victor Development Team
```

---

## Summary

The legacy code deprecation and removal plan provides:

- ✅ **Inventory of legacy code** across 4 categories
- ✅ **Deprecation strategy** with clear warnings
- ✅ **Removal timeline** over 9+ weeks
- ✅ **Safe removal process** with validation
- ✅ **Communication plan** with notifications
- ✅ **Migration support** with documentation

**Key Dates**:
- **Deprecation**: March 2026 (0.6.0)
- **Removal**: September 2026 (0.7.0)
- **Migration Deadline**: September 2026

**Zero Breaking Changes**: All changes are backward compatible until removal.

For migration instructions, see [Migration Guide](migration_guide.md).
For deployment procedures, see [Deployment Playbook](deployment_playbook.md).
For rollout timeline, see [Rollout Plan](rollout_plan.md).
