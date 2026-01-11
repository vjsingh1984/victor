# Victor Migration Guide

This guide helps you upgrade Victor between versions by documenting breaking changes, deprecated features, and migration steps.

## Table of Contents

- [Version Migration Paths](#version-migration-paths)
- [Breaking Changes by Version](#breaking-changes-by-version)
- [Configuration Changes](#configuration-changes)
- [API Changes](#api-changes)
- [Workflow Changes](#workflow-changes)
- [Tool Changes](#tool-changes)
- [Team Coordination Changes](#team-coordination-changes)

---

## Version Migration Paths

### Upgrading from 0.4.x to 0.5.x

**Impact Level**: Medium

**Summary**: Major architectural refactoring including workflow executor cleanup, cache serialization changes, and canonical import consolidation.

**Migration Steps**:

1. **Update workflow executor calls** (Critical - Breaking)
   ```python
   # Old (removed in 0.5.x)
   from victor.research.workflows import ResearchWorkflowProvider
   provider = ResearchWorkflowProvider()
   executor = provider.create_executor(orchestrator)
   result = await executor.execute(workflow, context)

   # New (0.5.x)
   from victor.research.workflows import ResearchWorkflowProvider
   provider = ResearchWorkflowProvider()
   compiled = provider.compile_workflow("workflow_name")
   result = await compiled.invoke(context)
   ```

2. **Update streaming workflows** (Critical - Breaking)
   ```python
   # Old (removed in 0.5.x)
   executor = provider.create_streaming_executor(orchestrator)
   async for node_id, state in executor.astream(workflow, context):
       print(node_id, state)

   # New (0.5.x)
   async for node_id, state in provider.stream_compiled_workflow("workflow_name", context):
       print(node_id, state)
   ```

3. **Clear IntegrationResult cache** (High Impact)
   - The cache format changed from pickle to JSON
   - Old cache files are automatically invalidated
   - No manual action required

4. **Update imports for canonical locations** (Medium Impact)
   ```python
   # Old (deprecated)
   from victor.framework.team_coordinator import FrameworkTeamCoordinator
   from victor.framework.complexity_classifier import ComplexityClassifier

   # New (0.5.x)
   from victor.teams import create_coordinator
   from victor.classification import ComplexityClassifier
   ```

**Testing**:
```bash
# Run tests to verify migration
pytest tests/unit/framework/test_vertical_integration.py -v
pytest tests/integration/workflows/test_yaml_workflow_e2e.py -v
```

---

### Upgrading from 0.3.x to 0.5.x

**Impact Level**: High

**Summary**: Includes all changes from 0.4.x plus tool selection API refactor.

**Additional Migration Steps** (on top of 0.4.x steps):

1. **Update tool selection configuration** (Critical - Breaking)
   ```python
   # Old (0.3.x)
   from victor.config.settings import Settings
   settings = Settings(use_semantic_tool_selection=True)

   tools = await selector.select_tools(message, use_semantic=True)

   # New (0.5.x)
   from victor.config.settings import Settings
   settings = Settings(tool_selection_strategy="semantic")  # or "keyword", "hybrid", "auto"

   tools = await selector.select_tools(message)
   ```

2. **Update tool selector protocol imports** (High Impact)
   ```python
   # Old (0.3.x - removed)
   from victor.tools.semantic_selector import ToolSelectorProtocol, SemanticToolSelectorProtocol

   # New (0.5.x)
   from victor.protocols import IToolSelector
   from victor.tools.semantic_selector import SemanticToolSelector
   from victor.tools.keyword_selector import KeywordToolSelector
   from victor.tools.hybrid_selector import HybridToolSelector
   ```

3. **Update environment variables** (Medium Impact)
   ```bash
   # Old (0.3.x)
   VICTOR_USE_SEMANTIC_TOOL_SELECTION=true

   # New (0.5.x)
   VICTOR_TOOL_SELECTION_STRATEGY=semantic
   ```

**Configuration Migration**:

```yaml
# Old config (0.3.x)
use_semantic_tool_selection: true

# New config (0.5.x)
tool_selection_strategy: semantic  # auto|keyword|semantic|hybrid
```

---

### Upgrading from 0.2.x to 0.5.x

**Impact Level**: High

**Summary**: Major architectural changes including workflow system introduction and team coordination consolidation.

**Key Breaking Changes**:

1. **Workflow System Introduction** (0.4.0)
2. **Tool Selection Refactor** (0.3.0)
3. **Team Coordination Consolidation** (0.4.1)
4. **RL Framework Migration** (0.3.0)

**Migration Strategy**:

Recommended approach: **Incremental migration**

1. **First upgrade to 0.3.0** and fix tool selection
2. **Then upgrade to 0.4.0** and adopt workflow system
3. **Finally upgrade to 0.5.x** for latest features

**Alternative**: Direct upgrade if codebase is small or test coverage is good.

---

## Breaking Changes by Version

### Version 0.5.x (Current)

#### 1. Deprecated Workflow Executor APIs Removed

**Impact**: Critical

**What Changed**:
- `BaseYAMLWorkflowProvider.create_executor()` - Removed
- `BaseYAMLWorkflowProvider.create_streaming_executor()` - Removed
- `BaseYAMLWorkflowProvider.astream()` - Removed
- `BaseYAMLWorkflowProvider.run_workflow()` - Removed
- `StateGraph` alias in `graph_dsl.py` - Removed

**Why Changed**:
The old executor pattern was inconsistent with the rest of the codebase. The new `compile_workflow()` + `invoke()` pattern provides better caching, observability, and consistency.

**Migration**:

```python
# Before (0.4.x)
from victor.research.workflows import ResearchWorkflowProvider

provider = ResearchWorkflowProvider()
executor = provider.create_executor(orchestrator)
result = await executor.execute(workflow_definition, context)

# After (0.5.x)
from victor.research.workflows import ResearchWorkflowProvider

provider = ResearchWorkflowProvider()
compiled = provider.compile_workflow("workflow_name")
result = await compiled.invoke(context)

# Streaming equivalent
# Before
async for node_id, state in executor.astream(workflow_definition, context):
    pass

# After
async for node_id, state in provider.stream_compiled_workflow("workflow_name", context):
    pass
```

**Tests to Update**:
- `tests/integration/workflows/test_vertical_streaming_workflows.py`
- Any tests using `create_executor()` or `create_streaming_executor()`

---

#### 2. IntegrationResult Cache Serialization Changed

**Impact**: High

**What Changed**:
- IntegrationResult cache changed from pickle to JSON serialization
- Cache files in `~/.victor/cache/` are automatically invalidated
- Old cache entries are ignored and recreated

**Why Changed**:
Pickle cannot serialize middleware objects and has security implications. JSON is safer and more portable.

**Migration**:
- **No code changes required**
- Cache automatically rebuilds on first run
- Temporary performance impact on first run

**Verification**:
```bash
# Check cache directory
ls -la ~/.victor/cache/

# Clear cache if needed (optional)
rm -rf ~/.victor/cache/*
```

---

#### 3. Canonical Import Consolidation

**Impact**: Medium

**What Changed**:
- All imports moved to canonical locations
- Backward compatibility aliases removed

**Affected Modules**:
- `ComplexityClassifier`: Now at `victor.classification.complexity_classifier`
- Team protocols: Now at `victor.protocols.team`
- Event types: Now at `victor.core.events.types`

**Migration**:

```python
# Before
from victor.framework.complexity_classifier import ComplexityClassifier
from victor.framework.team_coordinator import FrameworkTeamCoordinator
from victor.teams.protocols import ITeamMember, ITeamCoordinator

# After
from victor.classification import ComplexityClassifier
from victor.teams import create_coordinator, ITeamMember
from victor.protocols.team import ITeamCoordinator, ITeamMember
```

**See Also**: [Canonical Import Locations](#canonical-import-locations)

---

### Version 0.4.0

#### 1. YAML-First Workflow Architecture

**Impact**: Critical

**What Changed**:
- Workflows now defined in YAML instead of Python
- `BaseYAMLWorkflowProvider` introduced as base class
- Escape hatch pattern for complex logic

**Why Changed**:
YAML workflows are more maintainable, easier to version control, and accessible to non-developers.

**Migration**:

```python
# Before (0.3.x - Python workflows)
class MyWorkflow:
    def definition(self):
        return WorkflowDefinition(
            nodes=[...]
        )

# After (0.4.x - YAML workflows)
# victor/myvertical/workflows/my_workflow.yaml
workflows:
  my_workflow:
    nodes:
      - id: start
        type: agent
        role: researcher
        goal: "Do research"
        next: [end]

# Python code
from victor.framework.workflows import BaseYAMLWorkflowProvider

class MyWorkflowProvider(BaseYAMLWorkflowProvider):
    def _get_escape_hatches_module(self) -> str:
        return "victor.myvertical.escape_hatches"
```

**Documentation**:
- [Workflow Development Guide](/docs/guides/workflow-development/)
- [YAML Workflow Syntax](/docs/user-guide/workflows.md)

---

#### 2. External Vertical Plugin System

**Impact**: Medium

**What Changed**:
- Verticals can be loaded via entry points in `pyproject.toml`
- `VerticalBase` protocol for external verticals

**Migration** (for external vertical authors):

```toml
# pyproject.toml
[project.entry-points."victor.verticals"]
my_vertical = "my_victor_extension:MyVertical"
```

```python
# my_victor_extension/__init__.py
from victor.core.verticals.base import VerticalBase

class MyVertical(VerticalBase):
    name = "my_vertical"

    def get_tools(self):
        return [...]

    def get_system_prompt(self):
        return "..."
```

---

### Version 0.3.0

#### 1. Tool Selection API Refactor

**Impact**: Critical

**What Changed**:
- `use_semantic_tool_selection` setting replaced by `tool_selection_strategy`
- `ToolSelector.select_tools(use_semantic=...)` parameter removed
- Protocols renamed: `ToolSelectorProtocol` and `SemanticToolSelectorProtocol` → `IToolSelector`

**Why Changed**:
New architecture supports multiple selection strategies (keyword, semantic, hybrid, auto) with clean dependency injection.

**Migration**:

```python
# Before (0.2.x)
from victor.config.settings import Settings
from victor.tools.semantic_selector import ToolSelector, ToolSelectorProtocol

settings = Settings(use_semantic_tool_selection=True)
selector = ToolSelector(settings)
tools = await selector.select_tools(message, use_semantic=True)

# After (0.3.0+)
from victor.config.settings import Settings
from victor.tools.semantic_selector import SemanticToolSelector
from victor.protocols import IToolSelector

settings = Settings(tool_selection_strategy="semantic")
# or: Settings(tool_selection_strategy="keyword")
# or: Settings(tool_selection_strategy="hybrid")
# or: Settings(tool_selection_strategy="auto")

selector = SemanticToolSelector(settings)
tools = await selector.select_tools(message)  # No use_semantic parameter
```

**Environment Variables**:

```bash
# Before
export VICTOR_USE_SEMANTIC_TOOL_SELECTION=true

# After
export VICTOR_TOOL_SELECTION_STRATEGY=semantic
```

**New Strategies**:
- `"auto"` - Automatically selects based on environment (air-gapped → keyword, embeddings → semantic)
- `"keyword"` - Fast registry-based selection (<1ms)
- `"semantic"` - Embedding-based selection
- `"hybrid"` - Blends semantic (70%) and keyword (30%) with RRF

**See Also**: [Tool Selection Architecture](/docs/development/architecture/tool-selection.md)

---

#### 2. RL Framework Migration

**Impact**: High

**What Changed**:
- Three separate RL learners unified into centralized framework
- Storage moved to `~/.victor/graph/graph.db` (SQLite)
- Old implementations moved to `archive/deprecated_rl_modules/`

**Affected Learners**:
- Continuation prompt learner
- Semantic threshold learner
- Model selector learner

**Migration**:

```python
# Before (0.2.x - separate learners)
from victor.agent.rl_continuation import ContinuationLearner
from victor.semantic_search.rl_threshold import ThresholdLearner

continuation = ContinuationLearner(...)
threshold = ThresholdLearner(...)

# After (0.3.0+ - unified coordinator)
from victor.framework.rl import get_rl_coordinator

rl = get_rl_coordinator()
# All RL operations through unified interface
```

**Data Migration**:
- Old RL data is NOT automatically migrated
- Learners start fresh with new storage
- Old data preserved in `archive/` if needed

---

#### 3. Semantic Similarity Threshold Lowered

**Impact**: Low

**What Changed**:
- Default threshold lowered from 0.7 to 0.5
- Further lowered to 0.25 in 0.5.x

**Why Changed**:
Reduce false negatives in semantic search for technical queries.

**Migration**:

```yaml
# settings.yaml
semantic_similarity_threshold: 0.25  # Was 0.7 in 0.2.x
```

**Tuning Guide**:
- 0.1-0.2: Very permissive (high recall, low precision)
- 0.25-0.3: Balanced (default in 0.5.x)
- 0.4-0.5: Strict (low recall, high precision)
- 0.6+: Very strict (many false negatives)

---

### Version 0.2.0

#### 1. Initial Public Release

**Impact**: Foundation Release

**Notable Changes**:
- PyPI publishing via Trusted Publishing
- Docker images on Docker Hub
- Multi-platform binaries (macOS, Windows)

**No Breaking Changes** from pre-0.2.0 versions.

---

## Configuration Changes

### Deprecated Settings

| Setting | Deprecated In | Removed In | Replacement |
|---------|---------------|------------|-------------|
| `use_semantic_tool_selection` | 0.3.0 | 0.5.0 | `tool_selection_strategy` |
| `semantic_similarity_threshold: 0.7` | 0.3.0 | 0.5.0 | `0.25` (new default) |
| `victor.serverUrl` (VS Code) | 0.4.0 | 1.0.0 | `victor.serverPort` |

### New Settings (0.5.x)

#### Observability Logging
```yaml
# Enable JSONL event logging for dashboard
enable_observability_logging: false
observability_log_path: ~/.victor/metrics/victor.jsonl
```

#### Workflow Cache
```yaml
# Workflow definition cache (P1 Scalability)
workflow_definition_cache_enabled: true
workflow_definition_cache_ttl: 3600
workflow_definition_cache_max_entries: 100

# StateGraph copy-on-write (P2 Scalability)
stategraph_copy_on_write_enabled: true
```

#### Event System
```yaml
# Canonical event system configuration
event_backend_type: in_memory  # in_memory|sqlite|redis|kafka|sqs|rabbitmq
event_delivery_guarantee: at_least_once
```

### Settings Value Changes

#### Semantic Similarity Threshold
```yaml
# 0.2.x
semantic_similarity_threshold: 0.7

# 0.3.0-0.4.x
semantic_similarity_threshold: 0.5

# 0.5.x
semantic_similarity_threshold: 0.25  # Better for technical queries
```

#### Tool Selection Strategy
```yaml
# 0.2.x
use_semantic_tool_selection: true

# 0.3.0+
tool_selection_strategy: auto  # auto|keyword|semantic|hybrid
```

---

## API Changes

### Removed Methods/Classes

#### 0.5.x Removals

1. **BaseYAMLWorkflowProvider.create_executor()**
   - Replacement: `compile_workflow().invoke()`

2. **BaseYAMLWorkflowProvider.create_streaming_executor()**
   - Replacement: `stream_compiled_workflow()`

3. **BaseYAMLWorkflowProvider.astream()**
   - Replacement: `stream_compiled_workflow()`

4. **BaseYAMLWorkflowProvider.run_workflow()**
   - Replacement: `compile_workflow().invoke()`

5. **FrameworkTeamCoordinator** (deprecated wrapper)
   - Replacement: `create_coordinator(lightweight=True)`

6. **StateGraph alias in graph_dsl.py**
   - Replacement: `WorkflowGraph` or `victor.framework.graph.StateGraph`

#### 0.3.0 Removals

1. **Settings.use_semantic_tool_selection**
   - Replacement: `Settings.tool_selection_strategy`

2. **ToolSelector.select_tools(use_semantic=...)**
   - Replacement: Configure strategy globally

3. **ToolSelectorProtocol** and **SemanticToolSelectorProtocol**
   - Replacement: `IToolSelector`

### Renamed Methods/Classes

| Old Name | New Name | Version |
|----------|----------|---------|
| `ToolSelectorProtocol` | `IToolSelector` | 0.3.0 |
| `SemanticToolSelectorProtocol` | `IToolSelector` | 0.3.0 |
| `FrameworkTeamCoordinator` | `create_coordinator(lightweight=True)` | 0.4.1 |
| `victor.framework.complexity_classifier` | `victor.classification.complexity_classifier` | 0.5.0 |

### Signature Changes

#### ToolSelector.select_tools()

```python
# Before (0.2.x)
async def select_tools(
    self,
    message: str,
    use_semantic: bool = False,
    max_tools: int = 10,
) -> List[ToolDefinition]:

# After (0.3.0+)
async def select_tools(
    self,
    message: str,
    max_tools: int = 10,
    context: Optional[ToolSelectionContext] = None,
) -> List[ToolDefinition]:
```

**Migration**:
```python
# Before
tools = await selector.select_tools(message, use_semantic=True)

# After
# Set strategy during selector initialization
selector = SemanticToolSelector(settings)  # or Keyword, Hybrid
tools = await selector.select_tools(message)
```

---

## Workflow Changes

### YAML Syntax Changes

#### 0.4.0+ YAML Structure

```yaml
workflows:
  my_workflow:
    description: "Workflow description"
    nodes:
      - id: start_node
        type: agent
        role: researcher
        goal: "Task description"
        tool_budget: 10
        next: [next_node]

      - id: compute_step
        type: compute
        handler: stats_compute
        inputs:
          data: $ctx.result_key
        next: [check_condition]

      - id: check_condition
        type: condition
        condition: quality_threshold
        branches:
          high_quality: proceed
          needs_cleanup: cleanup
```

### Node Type Changes

#### 0.4.0+ Node Types

| Node Type | Description | Example |
|-----------|-------------|---------|
| `agent` | LLM-powered agent | See above |
| `compute` | Handler execution | See above |
| `condition` | Branching logic | See above |
| `parallel` | Parallel execution | Multiple nodes simultaneously |
| `transform` | State transformation | Modify context |
| `hitl` | Human-in-the-loop | Wait for user input |

### Workflow Execution API Changes

```python
# Before (0.3.x - deprecated)
executor = provider.create_executor(orchestrator)
result = await executor.execute(workflow, context)

# After (0.5.x - current)
provider = ResearchWorkflowProvider()
compiled = provider.compile_workflow("deep_research")
result = await compiled.invoke({"query": "AI trends"})
```

---

## Tool Changes

### Removed Tools

No tools have been removed in versions 0.2.x through 0.5.x.

### Renamed Tools

No tools have been renamed in versions 0.2.x through 0.5.x.

### Tool Parameter Changes

#### code_search Tool

**New Parameters (0.3.0+)**:
```yaml
enable_hybrid_search: false  # Enable semantic + keyword RRF
enable_semantic_threshold_rl_learning: false  # Learn optimal thresholds
semantic_threshold_overrides: {}  # Override thresholds per context
```

---

## Team Coordination Changes

### Multi-Agent System Consolidation (0.4.1+)

**Overview**:
Three separate coordinator implementations consolidated into one unified system.

**Before (0.4.0)**:
```python
from victor.framework.team_coordinator import FrameworkTeamCoordinator

coordinator = FrameworkTeamCoordinator()
coordinator.add_member(researcher).add_member(executor)
result = await coordinator.execute_task("Implement feature", {})
```

**After (0.4.1+)**:
```python
from victor.teams import create_coordinator, TeamFormation

coordinator = create_coordinator(lightweight=True)
coordinator.set_formation(TeamFormation.PIPELINE)
coordinator.add_member(researcher).add_member(executor)
result = await coordinator.execute_task("Implement feature", {})
```

**Benefits**:
- ~500 lines of duplicate code eliminated
- SOLID principles compliance
- Pluggable formation strategies
- 100% backward compatible

**Deprecated APIs**:

1. **FrameworkTeamCoordinator**
   - Status: Deprecated (wrapper provided)
   - Removal: v1.0.0
   - Replacement: `create_coordinator(lightweight=True)`

2. **Old Import Paths**
   ```python
   # Old (deprecated)
   from victor.framework.team_coordinator import FrameworkTeamCoordinator
   from victor.framework.agent_protocols import ITeamMember

   # New (recommended)
   from victor.teams import create_coordinator
   from victor.protocols.team import ITeamMember, ITeamCoordinator
   ```

**New Features**:

1. **Consensus Formation** (0.4.1+)
   ```python
   coordinator.set_formation(TeamFormation.CONSENSUS)
   result = await coordinator.execute_task("Review this PR", {})
   if result.get("consensus_achieved"):
       print(f"Consensus in {result['consensus_rounds']} rounds")
   ```

2. **Formation Strategies**
   - `SEQUENTIAL`: Chain agents one after another
   - `PARALLEL`: Execute all agents simultaneously
   - `HIERARCHICAL`: Manager-worker pattern
   - `PIPELINE`: Processing pipeline with output passing
   - `CONSENSUS`: Multi-round agreement building

**Documentation**:
- [Team Consolidation Release Notes](/victor/teams/RELEASE_NOTES.md)
- [Team Consolidation Details](/victor/teams/CONSOLIDATION.md)

---

## Canonical Import Locations

Import types from their canonical sources (not from integrations or duplicate modules):

| Type | Canonical Import |
|------|------------------|
| `ToolCall` | `from victor.agent.tool_calling.base import ToolCall` |
| `CircuitState` | `from victor.providers.circuit_breaker import CircuitState` |
| `StreamChunk` | `from victor.providers.base import StreamChunk` |
| `UnifiedWorkflowCompiler` | `from victor.workflows.unified_compiler import UnifiedWorkflowCompiler` |
| `BaseYAMLWorkflowProvider` | `from victor.framework.workflows import BaseYAMLWorkflowProvider` |
| `WorkflowEngine` | `from victor.framework.workflow_engine import WorkflowEngine` |
| Protocol types | `from victor.protocols import ...` |
| Team types | `from victor.teams import TeamFormation, AgentMessage, create_coordinator` |
| `EventBus` | `from victor.observability.event_bus import EventBus, EventCategory` |
| Search protocols | `from victor.protocols import ISemanticSearch, IIndexable` |
| Team protocols | `from victor.protocols import ITeamCoordinator, ITeamMember, IAgent` |
| LSP types | `from victor.protocols import Position, Range, Diagnostic, DocumentSymbol` |
| RL types | `from victor.framework.rl import RLManager, LearnerType, RLOutcome, get_rl_coordinator` |

---

## Testing Your Migration

### Unit Tests

```bash
# Run all unit tests
pytest tests/unit -v

# Run specific test files
pytest tests/unit/framework/test_vertical_integration.py -v
pytest tests/unit/protocols/test_team_protocols.py -v
```

### Integration Tests

```bash
# Run all integration tests
pytest tests/integration -v

# Run workflow tests
pytest tests/integration/workflows/test_yaml_workflow_e2e.py -v
pytest tests/integration/workflows/test_compiler_executor_integration.py -v
```

### Specific Migration Tests

```bash
# Test tool selection migration
pytest tests/unit/tools/test_semantic_selector.py -v

# Test workflow migration
pytest tests/integration/workflows/test_vertical_streaming_workflows.py -v

# Test team coordination migration
pytest tests/unit/teams/test_unified_coordinator.py -v
```

### Manual Testing

```bash
# Start Victor and verify basic functionality
victor chat --provider openai --model gpt-4

# Test workflow execution
victor workflow validate path/to/workflow.yaml

# Test tool selection
victor tools list --lightweight
```

---

## Common Migration Patterns

### Pattern 1: Workflow Provider Migration

**Scenario**: Migrating from Python-based workflows to YAML workflows.

```python
# Before (0.3.x)
from victor.workflows.definition import WorkflowDefinition, AgentNode

class ResearchWorkflowProvider:
    def get_workflows(self):
        return {
            "deep_research": WorkflowDefinition(
                nodes=[
                    AgentNode(id="research", role="researcher", goal="Research topic"),
                ]
            )
        }

# After (0.4.x+)
# victor/research/workflows/deep_research.yaml
workflows:
  deep_research:
    nodes:
      - id: research
        type: agent
        role: researcher
        goal: "Research topic"

# victor/research/workflows/__init__.py
from victor.framework.workflows import BaseYAMLWorkflowProvider

class ResearchWorkflowProvider(BaseYAMLWorkflowProvider):
    def _get_escape_hatches_module(self) -> str:
        return "victor.research.escape_hatches"
```

### Pattern 2: Team Coordination Migration

**Scenario**: Migrating from FrameworkTeamCoordinator to unified system.

```python
# Before (0.4.0)
from victor.framework.team_coordinator import FrameworkTeamCoordinator
from victor.framework.agent_protocols import ITeamMember

class MyAgent(ITeamMember):
    async def execute_task(self, task, context):
        return {"result": "done"}

coordinator = FrameworkTeamCoordinator()
coordinator.add_member(MyAgent("agent1"))
result = await coordinator.execute_task("Do work", {})

# After (0.4.1+)
from victor.teams import create_coordinator, TeamFormation
from victor.protocols import ITeamMember

class MyAgent(ITeamMember):
    async def execute_task(self, task, context):
        return {"result": "done"}

coordinator = create_coordinator(lightweight=True)
coordinator.set_formation(TeamFormation.SEQUENTIAL)
coordinator.add_member(MyAgent("agent1"))
result = await coordinator.execute_task("Do work", {})
```

### Pattern 3: Tool Selection Migration

**Scenario**: Migrating from use_semantic to strategy-based selection.

```python
# Before (0.2.x)
from victor.config.settings import Settings
from victor.tools.semantic_selector import ToolSelector

settings = Settings(use_semantic_tool_selection=True)
selector = ToolSelector(settings)

# In async function
if use_semantic:
    tools = await selector.select_tools(message, use_semantic=True)
else:
    tools = await selector.select_tools(message, use_semantic=False)

# After (0.3.0+)
from victor.config.settings import Settings
from victor.tools.semantic_selector import SemanticToolSelector
from victor.tools.keyword_selector import KeywordToolSelector

settings = Settings(tool_selection_strategy="semantic")  # or "keyword", "hybrid", "auto"

if settings.tool_selection_strategy == "semantic":
    selector = SemanticToolSelector(settings)
elif settings.tool_selection_strategy == "keyword":
    selector = KeywordToolSelector(settings)
# ... or use factory: create_tool_selector_strategy(settings)

tools = await selector.select_tools(message)
```

---

## Rollback Plan

If you encounter issues after upgrading:

### Rollback to Previous Version

```bash
# Uninstall current version
pip uninstall victor-ai

# Install specific previous version
pip install victor-ai==0.4.0

# Or from git
pip install git+https://github.com/vjsingh1984/victor@v0.4.0
```

### Clear Caches

```bash
# Clear all Victor caches
rm -rf ~/.victor/cache/*
rm -rf ~/.victor/graph/*

# Clear project-specific caches
rm -rf .victor/cache/*
rm -rf .victor/embeddings/*
```

### Verify Rollback

```bash
# Verify version
victor --version

# Run smoke tests
victor chat --provider openai --model gpt-4 --help
pytest tests/unit -k "test_" --co -q
```

---

## Getting Help

### Documentation

- [Architecture Documentation](/docs/architecture/)
- [User Guide](/docs/user-guide/)
- [Development Guide](/docs/development/)
- [API Reference](/docs/api-reference/)

### Community

- [GitHub Issues](https://github.com/vjsingh1984/victor/issues)
- [GitHub Discussions](https://github.com/vjsingh1984/victor/discussions)

### Debug Mode

Enable debug logging for detailed migration information:

```bash
# Enable debug logging
export VICTOR_LOG_LEVEL=DEBUG

# Run with debug output
victor chat --log-level DEBUG
```

---

## Checklist

Use this checklist when upgrading between major versions:

### Pre-Upgrade

- [ ] Read breaking changes for target version
- [ ] Review configuration changes
- [ ] Backup configuration files
- [ ] Note current Victor version: `victor --version`
- [ ] Run full test suite to establish baseline

### During Upgrade

- [ ] Update dependencies: `pip install --upgrade victor-ai`
- [ ] Update configuration settings
- [ ] Update imports in code
- [ ] Update API calls
- [ ] Clear caches if required

### Post-Upgrade

- [ ] Verify version: `victor --version`
- [ ] Run full test suite
- [ ] Run integration tests
- [ ] Test workflows
- [ ] Test tool selection
- [ ] Test team coordination
- [ ] Monitor logs for deprecation warnings
- [ ] Update documentation as needed

### Verification

- [ ] All tests passing
- [ ] No deprecation warnings
- [ ] Workflows execute correctly
- [ ] Tools function as expected
- [ ] Performance acceptable
- [ ] No errors in logs

---

**Last Updated**: 2026-01-10

**For Version**: 0.5.x

**Maintainer**: Vijaykumar Singh <singhvjd@gmail.com>
