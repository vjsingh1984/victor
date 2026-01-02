# Technical Debt Registry

**Last Updated**: January 2026
**Purpose**: Track and prioritize technical debt for systematic resolution

---

## Priority Matrix

| Priority | Definition | SLA |
|----------|------------|-----|
| P0 | Blocks releases or causes data loss | Fix immediately |
| P1 | Significant code quality issue | Fix within sprint |
| P2 | Moderate issue affecting maintainability | Fix within quarter |
| P3 | Minor improvement opportunity | Backlog |

---

## P0 - Critical

### TD-001: LSP Violation in LMStudio Provider

**File**: `victor/providers/lmstudio_provider.py:190`
**Issue**: Method signature differs from base class
**Impact**: Breaks Liskov Substitution Principle, can cause runtime errors

```python
# Current (WRONG)
def supports_tools(self, tools: List[str]) -> bool:

# Expected (BaseProvider)
def supports_tools(self) -> bool:
```

**Fix**: Remove the `tools` parameter to match base class signature.

---

## P1 - High Priority

### TD-002: God Class - AgentOrchestrator

**File**: `victor/agent/orchestrator.py`
**Metrics**: 7,948 lines, 197 methods
**Issue**: Single Responsibility Principle violation - handles 8+ distinct responsibilities

**Responsibilities to Extract**:
1. `ConversationManager` - Conversation state, history, memory
2. `ToolExecutionController` - Tool selection, execution, result handling
3. `ProviderCoordinator` - Provider selection, failover, rate limiting
4. `StreamingManager` - Stream handling, chunk processing
5. `SessionStateManager` - State machine, session lifecycle

**Effort**: High (2-3 sprints)
**Risk**: High - central component, needs careful refactoring

### TD-003: Factory Bloat - OrchestratorFactory

**File**: `victor/agent/orchestrator_factory.py`
**Metrics**: 2,211 lines, 75+ methods
**Issue**: Factory contains business logic that belongs elsewhere

**Recommended Split**:
- `OrchestratorFactory` - Pure instantiation
- `OrchestratorConfigurator` - Configuration logic
- `OrchestratorBuilder` - Complex construction patterns

**Effort**: Medium (1 sprint)

### TD-004: Generic Exception Catches

**Files**: Multiple (56 instances total)
**Top Offenders**:
- `orchestrator.py`: 18 instances
- `tool_pipeline.py`: 8 instances
- `conversation_controller.py`: 6 instances

**Issue**: Generic `except Exception` masks specific errors, makes debugging difficult

**Fix Pattern**:
```python
# Before
try:
    result = await some_operation()
except Exception as e:
    logger.error(f"Failed: {e}")

# After
try:
    result = await some_operation()
except SpecificError as e:
    logger.error(f"Known failure: {e}")
except Exception as e:
    logger.exception(f"Unexpected error in some_operation")
    raise
```

**Effort**: Medium (scattered across codebase)

---

## P2 - Medium Priority

### TD-005: Missing Tool Calling Adapters

**Issue**: Only 6 of 21 providers have dedicated tool calling adapters
**Impact**: 15 providers rely on generic adapter, may miss provider-specific optimizations

**Providers with Adapters**:
- Anthropic
- OpenAI
- Google
- Mistral
- Groq
- Together

**Providers Needing Adapters**:
- DeepSeek (high priority - popular)
- xAI
- Fireworks
- Azure OpenAI
- Bedrock
- Others (11 total)

**Effort**: Low per adapter, High total (1 day each)

### TD-006: Stubbed Features ✅ RESOLVED

**Issue**: Features claimed in documentation but not implemented
**Resolution**: Marked as "Experimental (Not Yet Implemented)" in `docs/guides/GRAPH_BACKENDS.md`

| Feature | Location | Status |
|---------|----------|--------|
| LanceDB graph backend | `victor/coding/codebase/graph/lancedb_store.py` | Documented as experimental |
| Neo4j graph backend | `victor/coding/codebase/graph/neo4j_store.py` | Documented as experimental |

Note: "Parallel workflow execution" was incorrectly flagged - the executor supports parallel node execution via `parallel` node type in YAML workflows.

### TD-007: Tool Catalog Incomplete ✅ RESOLVED

**Issue**: `TOOL_CATALOG.md` documents 45 tools, but registry has 55
**Resolution**: Regenerated tool catalog with all 55 tools

**Fix Applied**: `python scripts/generate_tool_catalog.py`

---

## P3 - Low Priority

### TD-008: BaseProvider Protocol Too Fat

**Issue**: All providers must implement methods they may not support
**Impact**: Unnecessary method stubs, ISP violation

**Current Fat Methods**:
- `supports_function_calling()` - Not all support this
- `get_token_limit()` - Varies significantly
- `supports_streaming()` - Always true for modern providers

**Fix**: Split into smaller protocols
```python
class StreamingProvider(Protocol):
    def stream_chat(self, ...) -> AsyncIterator: ...

class ToolCallingProvider(Protocol):
    def supports_tools(self) -> bool: ...
```

### TD-009: Test Coverage Gaps

**Areas with Insufficient Coverage**:
- Provider failover scenarios
- Tool calling adapter edge cases
- Workflow parallel execution (stubbed)
- Graph backend operations (stubbed)

### TD-010: Embedding Cache Not Project-Isolated

**Issue**: Single global embedding cache, not project-scoped
**Impact**: Cache pollution between projects
**Fix**: Add project hash to cache key

---

## Debt Tracking Process

1. **Discovery**: Add new debt items with TD-XXX ID
2. **Triage**: Weekly review, assign priority
3. **Sprint Planning**: Include P1 items in each sprint
4. **Resolution**: Link to PR when fixed
5. **Removal**: Delete item after PR merged

---

## Resolution Log

| ID | Description | Resolved | PR |
|----|-------------|----------|-----|
| TD-001 | LSP Violation in LMStudio Provider | 2026-01-02 | 2ce4e4d |
| TD-004 | Generic Exception Catches (partial: 6 critical fixes) | 2026-01-02 | ee6cd5e, f82f1be |
| TD-005 | Missing Tool Calling Adapters (DeepSeek) | 2026-01-02 | a32d031 |
| TD-006 | Stubbed Features (documented as experimental) | 2026-01-02 | - |
| TD-007 | Tool Catalog Incomplete | 2026-01-02 | 59fa014 |

---

## Metrics Dashboard

```
Total Debt Items: 7
├── P0 (Critical): 0
├── P1 (High): 3 (TD-004 partially resolved)
├── P2 (Medium): 1 (TD-005 partially resolved)
└── P3 (Low): 3

Code Quality Score: 7.5/10
├── SOLID Compliance: 8/10 (LSP violation fixed)
├── Error Handling: 6/10 (critical catches fixed)
├── Test Coverage: 7/10
└── Documentation Accuracy: 9/10 (catalog + stubs documented)
```

---

*This document should be updated when technical debt is discovered or resolved.*
