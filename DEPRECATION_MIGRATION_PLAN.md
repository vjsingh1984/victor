# Deprecation Analysis and Migration Plan

**Created**: January 27, 2026
**Version**: v0.5.1
**Target**: v0.7.0 - Remove all deprecated APIs

---

## Executive Summary

This document identifies ALL deprecated APIs in the codebase and provides a comprehensive migration plan to canonical APIs. After migration, all deprecated code will be removed.

---

## Deprecation Categories

### 1. Tool Dependency Providers

**Status**: ✅ Migration Path Defined, ✅ Most Verticals Migrated

**Deprecated Pattern**:
```python
# OLD (deprecated):
from victor.coding.tool_dependencies import CodingToolDependencyProvider
provider = CodingToolDependencyProvider()

from victor.devops.tool_dependencies import DevOpsToolDependencyProvider
provider = DevOpsToolDependencyProvider()
```

**Canonical API**:
```python
# NEW (canonical):
from victor.core.tool_dependency_loader import create_vertical_tool_dependency_provider
provider = create_vertical_tool_dependency_provider("coding")
provider = create_vertical_tool_dependency_provider("devops")
```

**Files Using Deprecated Pattern**:
- [x] victor/coding/__init__.py - ✅ MIGRATED
- [x] victor/devops/__init__.py - ✅ MIGRATED
- [x] victor/research/__init__.py - ✅ MIGRATED
- [x] victor/dataanalysis/__init__.py - ✅ MIGRATED
- [x] victor/rag/__init__.py - ✅ MIGRATED (already using canonical API, just wasn't exporting)
- [x] victor/security_analysis/__init__.py - ✅ N/A (no tool_dependencies.py file)

**Migration Script**:
```bash
# Find deprecated usage
grep -r "ToolDependencyProvider" victor/ --include="*.py" | grep -v "canonical\|backward_compat"

# Replace pattern:
# OLD: {Vertical}ToolDependencyProvider = create_vertical_tool_dependency_provider("{vertical}")
# NEW: Already migrated in most files
```

**Action Items**:
1. [x] Migrate RAG vertical - ✅ COMPLETE
2. [x] Migrate SecurityAnalysis vertical - ✅ N/A (no tool_dependencies.py file)
3. [ ] Remove all vertical-specific tool_dependencies.py files (KEEP - contains composed patterns, tool graphs)
4. [ ] Remove victor/core/tool_dependency_backward_compat.py (v0.7.0)

---

### 2. Direct Capability Instantiation

**Status**: ✅ COMPLETE - All verticals migrated to CapabilityInjector pattern

**Deprecated Pattern**:
```python
# OLD (deprecated):
from victor.framework.capabilities import FileOperationsCapability

class MyVertical(VerticalBase):
    _file_ops = FileOperationsCapability()  # Class-level instantiation
```

**Canonical API**:
```python
# NEW (canonical):
from victor.core.verticals.capability_injector import get_capability_injector

class MyVertical(VerticalBase):
    @classmethod
    def get_tools(cls):
        injector = get_capability_injector()
        file_ops = injector.get_file_operations_capability()
        return file_ops.get_tool_list()
```

**Files Using Deprecated Pattern**:
- [x] victor/coding/assistant.py - ✅ MIGRATED (removed class-level instantiation)
- [x] victor/devops/assistant.py - ✅ No direct instantiation
- [x] victor/research/assistant.py - ✅ Already using injector
- [x] victor/dataanalysis/assistant.py - ✅ No direct instantiation
- [x] victor/rag/assistant.py - ✅ No direct instantiation
- [x] victor/security_analysis/assistant.py - ✅ No direct instantiation

**Migration Steps**:
1. [x] Remove direct `FileOperationsCapability()` instantiation - DONE
2. [x] Use `get_capability_injector().get_file_operations_capability()` - DONE (ResearchAssistant pattern)
3. [ ] Apply `@deprecated_direct_instantiation` decorator to framework capabilities (v0.7.0)

**Action Items**:
1. [x] Audit all verticals for direct capability instantiation - COMPLETE
2. [x] Migrate all verticals to use CapabilityInjector - COMPLETE
3. [ ] Add `@deprecated_direct_instantiation` to framework capabilities (v0.7.0)
4. [ ] Remove victor/core/verticals/capability_migration.py helpers (v0.7.0)

---

### 3. Legacy AgentOrchestrator API

**Status**: ⚠️ Consolidated in LegacyAPIMixin, Awaiting Migration

**Deprecated Methods** (in victor/agent/mixins/legacy_api.py):
- `set_vertical_context()` → Use `VerticalContext.set_context()`
- `set_tiered_tool_config()` → Use `ConfigurationManager.set_tiered_tool_config()`
- `set_workspace()` → Use `set_project_root()` from settings
- And 20+ other legacy methods...

**Canonical APIs**:
- Vertical context: Use `VerticalContext` protocol
- Tool config: Use `ConfigurationManager`
- Project root: Use `set_project_root()` from settings
- Token usage: Use `MetricsCoordinator`
- Safety: Use `SafetyChecker` protocol

**Files Using Deprecated API**:
- [ ] All code calling `orchestrator.set_*()` methods
- [ ] Tests that use legacy orchestrator methods
- [ ] Integration code that depends on old API

**Migration Steps**:
1. Enable deprecation warnings in tests: `PYTHONWARNINGS=always`
2. Fix all deprecation warnings in tests
3. Update integration code to use new APIs
4. Verify all tests pass with no warnings

**Action Items**:
1. [ ] Run tests with deprecation warnings enabled
2. [ ] Fix all failing tests
3. [ ] Remove LegacyAPIMixin from AgentOrchestrator (v0.7.0)
4. [ ] Remove victor/agent/mixins/legacy_api.py (v0.7.0)

---

### 4. Chat Coordinator Domain-Specific Methods

**Status**: ✅ Deprecated, ⚠️ Awaiting Removal

**Deprecated Methods**:
- `_extract_required_files_from_prompt()` → Domain-specific, moved to verticals
- `_extract_required_outputs_from_prompt()` → Domain-specific, moved to verticals

**Canonical APIs**:
- Use workflow-based chat (already implemented)
- Use `CodingChatState.extract_requirements_from_message()` for coding vertical

**Files Affected**:
- victor/agent/coordinators/chat_coordinator.py (lines 1250-1305)

**Migration Steps**:
1. Use workflow-based chat (already enabled by default)
2. For custom implementations, use vertical-specific state classes

**Action Items**:
1. [ ] Verify no code depends on these methods
2. [ ] Remove deprecated methods from ChatCoordinator (v0.7.0)

---

### 5. Tool Selection - KeywordToolSelector

**Status**: ⚠️ Deprecated, Alternative Available

**Deprecated Class**: `KeywordToolSelector` (in victor/agent/tool_selection.py)

**Canonical API**: Use `SemanticToolSelector` or `HybridToolSelector`

**Migration**:
```python
# OLD (deprecated):
from victor.agent.tool_selection import KeywordToolSelector
selector = KeywordToolSelector(tools)

# NEW (canonical):
from victor.agent.tool_selection import HybridToolSelector
selector = HybridToolSelector(tools)
```

**Action Items**:
1. [ ] Replace all KeywordToolSelector usage
2. [ ] Remove KeywordToolSelector class (v0.7.0)

---

### 6. Chat/Stream Chat in AgentOrchestrator

**Status**: ✅ Deprecated Paths Added, ⚠️ Default Still Legacy

**Deprecated Methods**:
- `AgentOrchestrator.chat()` → Use `WorkflowOrchestrator.chat()`
- `AgentOrchestrator.stream_chat()` → Use `WorkflowOrchestrator.stream_chat()`

**Canonical APIs**:
- Use workflow-based chat (already implemented)
- Feature flag: `use_workflow_chat=True` (already True by default)

**Deprecation Location**: victor/agent/orchestrator.py (lines 2865, 2913)

**Action Items**:
1. [ ] Verify all code uses workflow-based chat
2. [ ] Remove legacy chat() and stream_chat() methods (v0.7.0)
3. [ ] Remove chat coordinator dependency from AgentOrchestrator

---

### 7. Action Authorization - Hard-Coded Tool Lists

**Status**: ✅ Metadata System Implemented, ⚠️ Hard-Coded Lists Present

**Deprecated Pattern**:
```python
# Hard-coded tool lists in ActionAuthorizer
WRITE_TOOLS = frozenset({"write_file", "edit_files", ...})
```

**Canonical API**: Use tool metadata from `ToolAuthMetadataRegistry`

**Files Affected**:
- victor/agent/action_authorizer.py

**Migration Status**:
- [x] ToolAuthMetadata system created
- [x] MetadataActionAuthorizer created
- [ ] Feature flag: `use_metadata_authorization=True` (already True by default)

**Action Items**:
1. [ ] Verify all tools have metadata registered
2. [ ] Remove hard-coded tool lists (v0.7.0)
3. [ ] Remove legacy authorization functions (v0.7.0)

---

## Migration Timeline

### Phase 1: Preparation (Week 1)
- [ ] Enable all deprecation warnings in CI/CD
- [ ] Create test suite to verify deprecation warnings
- [ ] Document migration patterns for contributors

### Phase 2: Tool Dependency Migration (Week 2)
- [ ] Migrate RAG and SecurityAnalysis verticals
- [ ] Remove vertical-specific tool_dependencies.py files
- [ ] Update all imports

### Phase 3: Capability Injection Migration (Week 3)
- [ ] Audit all verticals for direct capability instantiation
- [ ] Migrate to CapabilityInjector pattern
- [ ] Remove deprecated capability classes

### Phase 4: Legacy API Removal (Week 4-5)
- [ ] Fix all deprecation warnings in tests
- [ ] Update integration code
- [ ] Remove LegacyAPIMixin
- [ ] Remove legacy_api.py

### Phase 5: Chat Coordinator Cleanup (Week 6)
- [ ] Remove deprecated methods
- [ ] Simplify chat coordinator
- [ ] Remove domain-specific logic

### Phase 6: Tool Selection Cleanup (Week 7)
- [ ] Replace KeywordToolSelector everywhere
- [ ] Remove deprecated class
- [ ] Update tests

### Phase 7: Final Cleanup (Week 8)
- [ ] Remove all deprecated code
- [ ] Remove backward compatibility modules
- [ ] Update documentation

---

## Testing Strategy

### Enable Deprecation Warnings
```bash
# Run tests with deprecation warnings as errors
export PYTHONWARNINGS=error::DeprecationWarning
pytest tests/

# Or use pytest config
[tool.pytest.ini]
filterwarnings =
    error::DeprecationWarning
```

### Migration Verification
```bash
# 1. Count deprecation warnings before migration
python -W default -W error::DeprecationWarning -m pytest 2>&1 | grep "DeprecationWarning" | wc -l

# 2. Run tests after migration
pytest tests/ -v

# 3. Verify no warnings
pytest tests/ -v -W error::DeprecationWarning
```

---

## Success Criteria

### Completion Metrics
- [ ] 0 deprecation warnings in test runs
- [ ] All deprecated code removed
- [ ] 100% test pass rate maintained
- [ ] No breaking changes for users
- [ ] Documentation updated

### File Cleanup Targets
- [ ] victor/core/tool_dependency_backward_compat.py
- [ ] victor/core/verticals/capability_migration.py
- [ ] victor/agent/mixins/legacy_api.py
- [ ] victor/agent/tool_selection.py (KeywordToolSelector)
- [ ] victor/framework/tool_dependency_deprecation.py
- [ ] All vertical-specific tool_dependencies.py files

---

## Next Steps

1. **Immediate** (This session):
   - Create migration test script
   - Migrate RAG and SecurityAnalysis verticals
   - Begin legacy API migration

2. **Short-term** (Next sessions):
   - Complete migration of all deprecated patterns
   - Remove deprecated code systematically
   - Comprehensive testing

3. **Long-term** (v0.7.0):
   - Remove ALL backward compatibility code
   - Simplify codebase
   - Update all documentation

---

## References

- Migration helper: victor/core/tool_dependency_backward_compat.py
- Capability migration: victor/core/verticals/capability_migration.py
- Legacy API: victor/agent/mixins/legacy_api.py
- Tool deprecation: victor/framework/tool_dependency_deprecation.py
