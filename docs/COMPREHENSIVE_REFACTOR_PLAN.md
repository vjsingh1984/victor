# Comprehensive Refactor Plan - Victor Codebase

**Date**: 2025-01-24
**Status**: Draft for Review
**Priority**: High - Address SOLID violations and scalability risks

---

## Executive Summary

This document addresses code review feedback identifying SOLID violations, architecture gaps, and scalability risks. The plan is structured into 6 phases, each with clear targets, file-level changes, risks, and testing strategies.

**Overall Goal**: Achieve best-in-class architecture with SOLID compliance, improved scalability, and simplified APIs.

---

## Code Review Findings Summary

### Critical Issues

| Issue | Impact | Files Affected | Priority |
|-------|--------|----------------|----------|
| SRP Violation | AgentOrchestrator too large | `victor/agent/orchestrator.py` | High |
| OCP Violation | Static capability mapping | `victor/agent/capability_registry.py` | High |
| LSP Violation | SemanticToolSelector protocol | `victor/tools/semantic_selector.py` | Medium |
| ISP Violation | Composite protocol usage | `victor/framework/protocols.py` | Medium |
| DIP Violation | Direct attribute access | `victor/framework/step_handlers.py` | High |
| Unbounded Caches | Memory leaks | Multiple files | Critical |

### Architecture Gaps

1. **Generic Capabilities Duplication**
   - Core file-ops toolsets duplicated in DevOps/RAG verticals
   - Should use shared capability from framework

2. **Capability Config Anti-Pattern**
   - Direct writes to orchestrator attributes
   - Should use VerticalContext.capability_configs

3. **Stage Definition Duplication**
   - Similar phase structures repeated across verticals
   - Should use framework StageTemplate/StageBuilder

4. **Tool Dependency Duplication**
   - Overlapping patterns across vertical YAML configs
   - Should centralize common dependency packs

5. **Workflow Template Duplication**
   - Generic workflows duplicated across verticals
   - Should move to framework templates registry

### Scalability Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Semantic/hybrid selector aggressive caching | Memory inflation | Bounded caches with eviction |
| Extension cache unbounded | Unbounded memory growth | Size limits + LRU eviction |
| Vertical integration cache unbounded | Memory leaks | TTL + size bounds |
| Observability no backpressure | Event loop overwhelm | Bounded queues + sampling |
| Tool schema cache invalidation hot | Performance hit | Incremental invalidation |
| Dynamic module loading overhead | Long-running service overhead | Lazy loading + caching |

---

## Phase 1: Protocol-First Capabilities + VerticalContext Unification

**Duration**: 2-4 days
**Priority**: High
**Risk**: Medium (breaking changes to capability providers)

### Objectives

1. Move all vertical capability configs into VerticalContext
2. Stop direct orchestrator attribute writes
3. Remove legacy hasattr fallbacks for capability routing

### File-Level Changes

#### 1.1 Capability Config Migration

**Files to Modify**:
- `victor/coding/capabilities.py`
- `victor/devops/capabilities.py`
- `victor/core/verticals/context.py`
- `victor/framework/capabilities/base_vertical_capability_provider.py`

**Changes**:

```python
# victor/core/verticals/context.py
class VerticalContext:
    """Add new methods"""
    def set_capability_config(self, key: str, config: Any) -> None:
        """Store capability configuration in context."""
        self._capability_configs[key] = config

    def get_capability_config(self, key: str, default: Any = None) -> Any:
        """Retrieve capability configuration from context."""
        return self._capability_configs.get(key, default)

# victor/coding/capabilities.py - BEFORE
def apply_code_style(orchestrator, config):
    orchestrator.code_style = config  # Direct write

# victor/coding/capabilities.py - AFTER
def apply_code_style(context: VerticalContext, config):
    context.set_capability_config("code_style", config)
```

#### 1.2 Remove Legacy Fallbacks

**Files to Modify**:
- `victor/framework/vertical_integration.py`
- `victor/agent/capability_registry.py`
- `victor/framework/step_handlers.py`

**Changes**:

```python
# victor/agent/capability_registry.py
# BEFORE: Fallback mapping
CAPABILITY_METHOD_MAPPINGS = {
    "code_style": "_apply_code_style",
}

# AFTER: Strict protocol only
class CapabilityRegistry:
    def __init__(self, strict_mode: bool = True):
        self._strict_mode = strict_mode

    def get_capability_provider(self, name: str):
        provider = self._providers.get(name)
        if provider is None:
            if self._strict_mode:
                raise CapabilityNotFound(
                    f"Capability '{name}' not found. "
                    f"Register via CapabilityRegistryProtocol."
                )
            # Legacy fallback only if strict_mode=False
            return self._get_legacy_provider(name)
        return provider
```

### Testing Strategy

```python
# tests/integration/capabilities/test_context_config.py
def test_capability_config_in_context():
    """Test capability configs stored in VerticalContext."""
    context = VerticalContext("test")
    context.set_capability_config("code_style", {"style": "pep8"})

    assert context.get_capability_config("code_style") == {"style": "pep8"}

# tests/integration/capabilities/test_strict_mode.py
def test_strict_capability_routing():
    """Test strict mode rejects unregistered capabilities."""
    registry = CapabilityRegistry(strict_mode=True)

    with pytest.raises(CapabilityNotFound):
        registry.get_capability_provider("nonexistent")
```

### Risks

1. **Breaking Change**: External verticals must update capability providers
   - **Mitigation**: Feature flag for gradual migration
   - **Rollback**: Disable strict_mode temporarily

2. **Performance Overhead**: Context lookup vs direct attribute
   - **Mitigation**: Cache context lookups
   - **Measurement**: Benchmark before/after

---

## Phase 2: Tooling Baseline Packs + Stage Templates

**Duration**: 2-3 days
**Priority**: Medium
**Risk**: Low (backward compatible)

### Objectives

1. Create framework "tool packs" (baseline + deltas)
2. Introduce stage templates to reduce duplication
3. Eliminate per-vertical tool list duplication

### File-Level Changes

#### 2.1 Tool Packs Framework

**New File**: `victor/framework/tool_packs.py`

```python
"""Framework-level tool packs for baseline capabilities."""

from typing import Dict, List
from dataclasses import dataclass, field

@dataclass
class ToolPack:
    """A collection of tools with metadata."""
    name: str
    tools: List[str] = field(default_factory=list)
    description: str = ""
    extends: str = None  # Parent pack to extend

# Baseline tool packs
BASE_FILE_OPS = ToolPack(
    name="base_file_ops",
    tools=["read", "write", "edit", "search"],
    description="Core file operations available to all verticals"
)

BASE_WEB = ToolPack(
    name="base_web",
    tools=["web_search", "fetch_url"],
    description="Basic web capabilities"
)

BASE_GIT = ToolPack(
    name="base_git",
    tools=["git_status", "git_diff", "git_commit"],
    description="Git operations"
)

# Vertical-specific packs extend baseline
DEVOPS_PACK = ToolPack(
    name="devops",
    extends="base_file_ops",
    tools=["docker", "kubernetes", "terraform"],
    description="DevOps-specific tools"
)

def resolve_tool_pack(pack: ToolPack) -> List[str]:
    """Resolve tool pack with all extensions."""
    tools = []
    current = pack

    while current:
        tools.extend(current.tools)
        current = BASE_PACKS.get(current.extends)

    return list(dict.fromkeys(tools))  # Preserve order, deduplicate

BASE_PACKS = {
    "base_file_ops": BASE_FILE_OPS,
    "base_web": BASE_WEB,
    "base_git": BASE_GIT,
}
```

**Files to Modify**:
- `victor/devops/assistant.py`
- `victor/rag/assistant.py`
- `victor/research/assistant.py`
- `victor/dataanalysis/assistant.py`
- `victor/coding/assistant.py`

**Changes**:

```python
# victor/devops/assistant.py - BEFORE
def get_tools(self):
    return [
        "read", "write", "edit", "search",  # Duplicated
        "docker", "kubernetes", "terraform"
    ]

# victor/devops/assistant.py - AFTER
from victor.framework.tool_packs import DEVOPS_PACK, resolve_tool_pack

def get_tools(self):
    return resolve_tool_pack(DEVOPS_PACK)
```

#### 2.2 Stage Templates

**New File**: `victor/framework/stage_templates.py`

```python
"""Framework-level stage templates to reduce duplication."""

from enum import Enum
from typing import List, Optional
from dataclasses import dataclass

from victor.core.vertical_types import StageType

@dataclass
class StageTemplate:
    """Reusable stage template."""
    name: str
    stage_type: StageType
    description: str
    tools: List[str] = field(default_factory=list)
    next_stages: List[str] = field(default_factory=list)
    condition: Optional[str] = None

# Standard stage templates
INITIAL_STAGE = StageTemplate(
    name="initial",
    stage_type=StageType.INITIAL,
    description="Initial greeting and requirements gathering",
    tools=["read", "search"],
)

PLANNING_STAGE = StageTemplate(
    name="planning",
    stage_type=StageType.PLANNING,
    description="Plan approach before execution",
    tools=["read", "search", "web_search"],
    next_stages=["execution"],
)

EXECUTION_STAGE = StageTemplate(
    name="execution",
    stage_type=StageType.EXECUTION,
    description="Execute the planned changes",
    tools=["write", "edit", "bash"],
    next_stages=["verification"],
)

VERIFICATION_STAGE = StageTemplate(
    name="verification",
    stage_type=StageType.VERIFICATION,
    description="Verify changes work correctly",
    tools=["read", "bash", "test"],
)

STAGE_TEMPLATES = {
    "initial": INITIAL_STAGE,
    "planning": PLANNING_STAGE,
    "execution": EXECUTION_STAGE,
    "verification": VERIFICATION_STAGE,
}

def get_stage_template(name: str) -> Optional[StageTemplate]:
    """Get stage template by name."""
    return STAGE_TEMPLATES.get(name)
```

**Files to Modify**:
- `victor/devops/assistant.py`
- `victor/dataanalysis/assistant.py`
- `victor/research/assistant.py`
- `victor/coding/assistant.py`

**Changes**:

```python
# victor/coding/assistant.py - BEFORE
def get_stages(self):
    return [
        Stage("initial", ...),
        Stage("planning", ...),
        Stage("execution", ...),
        Stage("verification", ...),
    ]

# victor/coding/assistant.py - AFTER
from victor.framework.stage_templates import (
    STAGE_TEMPLATES,
    get_stage_template,
)

def get_stages(self):
    # Extend base templates with coding-specific additions
    return [
        get_stage_template("initial"),
        get_stage_template("planning"),
        Stage(  # Custom execution stage
            "coding_execution",
            extends="execution",
            tools=["write", "edit", "test", "lint"],
        ),
        get_stage_template("verification"),
    ]
```

### Testing Strategy

```python
# tests/unit/framework/test_tool_packs.py
def test_tool_pack_resolution():
    """Test tool packs resolve correctly."""
    from victor.framework.tool_packs import resolve_tool_pack, DEVOPS_PACK

    tools = resolve_tool_pack(DEVOPS_PACK)

    assert "read" in tools  # From base_file_ops
    assert "docker" in tools  # From devops
    assert len(tools) == len(set(tools))  # No duplicates

# tests/unit/framework/test_stage_templates.py
def test_stage_template_extends():
    """Test stage templates can be extended."""
    from victor.framework.stage_templates import EXECUTION_STAGE

    custom = Stage(
        "custom_execution",
        extends=EXECUTION_STAGE,
        tools=["custom_tool"],
    )

    assert "write" in custom.tools  # Inherited
    assert "custom_tool" in custom.tools
```

### Risks

1. **Tool Name Conflicts**: Overlapping tool names in different packs
   - **Mitigation**: Explicit namespace prefixes or conflict resolution strategy

2. **Stage Template Rigidity**: Templates may not fit all use cases
   - **Mitigation**: Allow full override, not just extension

---

## Phase 3: Tool Selector Protocol Compliance + Cache Bounds

**Duration**: 3-5 days
**Priority**: High
**Risk**: Medium

### Objectives

1. Make SemanticToolSelector fully implement IToolSelector
2. Add bounded caches with eviction for tool selectors
3. Remove aggressive caching anti-patterns

### File-Level Changes

#### 3.1 IToolSelector Protocol Compliance

**Files to Modify**:
- `victor/tools/semantic_selector.py`
- `victor/tools/keyword_tool_selector.py`
- `victor/tools/hybrid_tool_selector.py`
- `victor/agent/tool_selector_factory.py`

**Current Issue**:

```python
# victor/tools/semantic_selector.py
class SemanticToolSelector:
    """WARNING: Doesn't fully implement IToolSelector"""
    # Missing: select_tools_async(), get_stats(), etc.
```

**Fix**:

```python
# victor/tools/semantic_selector.py
from victor.tools.base import IToolSelector
from victor.tools.selection_cache import ToolSelectionCache

class SemanticToolSelector(IToolSelector):
    """Semantic tool selection with embedding-based matching.

    Fully implements IToolSelector protocol with bounded caching.
    """

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        cache_max_size: int = 1000,  # Bounded cache
        cache_ttl: int = 3600,
    ):
        self._embedding_model = embedding_model
        self._cache = ToolSelectionCache(
            max_size=cache_max_size,  # Now bounded
            query_ttl=cache_ttl,
            use_cache_config=True,
        )

    async def select_tools_async(
        self,
        query: str,
        available_tools: List[ToolDefinition],
        context: Optional[Dict] = None,
    ) -> List[ToolDefinition]:
        """Async tool selection (required by protocol)."""
        # Check cache first
        cache_key = self._generate_cache_key(query, available_tools)
        cached = self._cache.get_query(cache_key)
        if cached:
            return [t for t in available_tools if t.name in cached.value]

        # Perform semantic selection
        selected = await self._semantic_selection(query, available_tools)

        # Store in bounded cache
        self._cache.put_query(
            cache_key,
            [t.name for t in selected],
            ttl=self._cache._query_ttl,
        )

        return selected

    def get_stats(self) -> Dict[str, Any]:
        """Get selector statistics (required by protocol)."""
        return self._cache.get_stats()

    def clear_cache(self) -> None:
        """Clear selector cache (required by protocol)."""
        self._cache.invalidate()

    # Required by IToolSelector
    async def rank_tools(
        self,
        query: str,
        tools: List[ToolDefinition],
        top_k: int = 10,
    ) -> List[Tuple[ToolDefinition, float]]:
        """Rank tools by relevance."""
        embeddings = await self._get_embeddings([t.description for t in tools])
        query_embedding = await self._get_embedding(query)

        similarities = [
            (tool, self._cosine_similarity(query_embedding, emb))
            for tool, emb in zip(tools, embeddings)
        ]

        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
```

#### 3.2 Bounded Cache Configuration

**Files to Modify**:
- `victor/core/registries/cache_config.py` (update defaults)
- `victor/tools/semantic_selector.py`
- `victor/tools/hybrid_tool_selector.py`

**Cache Config Updates**:

```python
# victor/core/registries/cache_config.py
# Update cache configs with proper bounds

DEFAULT_CACHE_CONFIGS = {
    # ... existing configs ...

    # Tool selector caches - NOW BOUNDED
    "semantic_tool_selection": CacheConfig(
        strategy=CacheStrategy.LRU,
        max_size=500,  # Bounded (was unlimited)
        ttl=3600,
        priority=CachePriority.HIGH,
    ),
    "hybrid_tool_selection": CacheConfig(
        strategy=CacheStrategy.LRU,
        max_size=1000,  # Bounded
        ttl=1800,  # Shorter TTL
        priority=CachePriority.HIGH,
    ),
    "keyword_tool_selection": CacheConfig(
        strategy=CacheStrategy.LRU,
        max_size=200,  # Smaller cache for keyword
        ttl=3600,
        priority=CachePriority.MEDIUM,
    ),
}
```

### Testing Strategy

```python
# tests/unit/tools/test_semantic_selector_protocol.py
def test_semantic_selector_implements_protocol():
    """Test SemanticToolSelector fully implements IToolSelector."""
    from victor.tools.semantic_selector import SemanticToolSelector
    from victor.tools.base import IToolSelector

    selector = SemanticToolSelector()

    # Verify protocol compliance
    assert isinstance(selector, IToolSelector)

    # Test required methods
    assert hasattr(selector, 'select_tools_async')
    assert hasattr(selector, 'get_stats')
    assert hasattr(selector, 'clear_cache')
    assert hasattr(selector, 'rank_tools')

# tests/unit/tools/test_bounded_cache.py
def test_cache_respects_max_size():
    """Test cache evicts entries when max_size exceeded."""
    cache = ToolSelectionCache(max_size=10)

    # Add 20 entries
    for i in range(20):
        cache.put_query(f"key_{i}", ["tool"])

    # Should only keep 10 (most recent)
    stats = cache.get_stats()
    assert stats["combined"]["total_entries"] <= 10
```

### Risks

1. **Performance Impact**: Bounded cache may reduce hit rate
   - **Mitigation**: Tune max_size based on production metrics
   - **Monitoring**: Alert on cache eviction rate

2. **Breaking Change**: Async method may break existing callers
   - **Mitigation**: Provide sync wrapper for backward compatibility
   - **Deprecation**: Mark sync methods as deprecated

---

## Phase 4: Observability Backpressure + Unified Event Taxonomy

**Duration**: 3-4 days
**Priority**: High
**Risk**: Low

### Objectives

1. Add bounded async queue with backpressure for observability
2. Implement optional event sampling
3. Unify framework events with core taxonomy

### File-Level Changes

#### 4.1 Bounded Event Queue

**Files to Modify**:
- `victor/observability/integration.py`
- `victor/core/events/backends.py`

**New File**: `victor/observability/event_queue.py`

```python
"""Bounded event queue with backpressure for observability."""

import asyncio
from collections import deque
from typing import Any, Optional
from dataclasses import dataclass

@dataclass
class EventQueueConfig:
    """Configuration for event queue."""
    max_size: int = 10000
    backpressure_strategy: str = "drop_oldest"  # or "sample", "block"
    sample_rate: float = 1.0  # When strategy is "sample"

class BoundedEventQueue:
    """Bounded async queue for event emission with backpressure."""

    def __init__(self, config: EventQueueConfig):
        self._config = config
        self._queue: deque = deque(maxlen=config.max_size)
        self._dropped_events = 0

    async def emit(self, event: dict) -> bool:
        """Emit event with backpressure handling.

        Returns True if event was queued, False if dropped.
        """
        if len(self._queue) >= self._config.max_size:
            # Handle backpressure
            if self._config.backpressure_strategy == "drop_oldest":
                self._queue.popleft()  # Drop oldest
                self._dropped_events += 1
            elif self._config.backpressure_strategy == "sample":
                if random.random() > self._config.sample_rate:
                    return False  # Drop this event
            elif self._config.backpressure_strategy == "block":
                # Wait for space (use carefully - can block)
                await self._wait_for_space()
            else:
                return False  # Drop by default

        self._queue.append(event)
        return True

    async def flush(self) -> List[dict]:
        """Flush all queued events."""
        events = list(self._queue)
        self._queue.clear()
        return events

    def get_stats(self) -> dict:
        """Get queue statistics."""
        return {
            "queue_size": len(self._queue),
            "max_size": self._config.max_size,
            "dropped_events": self._dropped_events,
            "drop_rate": self._dropped_events / max(1, len(self._queue)),
        }
```

**Integrate into ObservabilityIntegration**:

```python
# victor/observability/integration.py
from victor.observability.event_queue import (
    BoundedEventQueue,
    EventQueueConfig,
)

class ObservabilityIntegration:
    def __init__(self, config: EventQueueConfig):
        self._event_queue = BoundedEventQueue(config)

    async def emit_event(self, event_type: str, **data):
        """Emit event with backpressure."""
        event = {"type": event_type, **data}

        queued = await self._event_queue.emit(event)
        if not queued:
            # Event was dropped due to backpressure
            logger.debug(f"Event dropped: {event_type}")

        # Optionally flush immediately for critical events
        if event_type in ("error", "critical"):
            await self._flush_events()

    async def _flush_events(self):
        """Flush queued events to event bus."""
        events = await self._event_queue.flush()

        for event in events:
            # Emit to event bus
            await self._event_bus.emit(event["type"], **event)
```

#### 4.2 Unified Event Taxonomy

**New File**: `victor/core/events/taxonomy.py`

```python
"""Unified event taxonomy for framework and core events."""

from enum import Enum

class EventCategory(Enum):
    """High-level event categories."""
    LIFECYCLE = "lifecycle"  # Start/stop, initialization
    TOOL = "tool"  # Tool selection, execution
    WORKFLOW = "workflow"  # Workflow stages, transitions
    AGENT = "agent"  # Agent state changes
    ERROR = "error"  # Errors, exceptions
    METRIC = "metric"  # Performance metrics
    DEBUG = "debug"  # Debug information

class FrameworkEvent(Enum):
    """Framework-specific events mapped to core taxonomy."""

    # Mapping: FrameworkEvent -> (Category, CoreEvent)
    AGENT_START = (EventCategory.LIFECYCLE, "agent.started")
    AGENT_STOP = (EventCategory.LIFECYCLE, "agent.stopped")

    TOOL_SELECTED = (EventCategory.TOOL, "tool.selected")
    TOOL_EXECUTED = (EventCategory.TOOL, "tool.executed")
    TOOL_ERROR = (EventCategory.ERROR, "tool.error")

    WORKFLOW_STAGE_ENTERED = (EventCategory.WORKFLOW, "workflow.stage.entered")
    WORKFLOW_STAGE_EXITED = (EventCategory.WORKFLOW, "workflow.stage.exited")

    @property
    def category(self) -> EventCategory:
        return self.value[0]

    @property
    def core_event(self) -> str:
        return self.value[1]

def map_framework_to_core(framework_event: FrameworkEvent) -> str:
    """Map framework event to core event name."""
    return framework_event.core_event
```

### Testing Strategy

```python
# tests/integration/observability/test_backpressure.py
import pytest

async def test_event_queue_backpressure():
    """Test event queue handles backpressure correctly."""
    from victor.observability.event_queue import BoundedEventQueue, EventQueueConfig

    config = EventQueueConfig(max_size=10, backpressure_strategy="drop_oldest")
    queue = BoundedEventQueue(config)

    # Add 20 events to queue of size 10
    for i in range(20):
        await queue.emit({"id": i})

    stats = queue.get_stats()
    assert stats["queue_size"] == 10  # Max size enforced
    assert stats["dropped_events"] == 10  # Half were dropped

async def test_event_sampling():
    """Test event sampling reduces load."""
    config = EventQueueConfig(
        max_size=100,
        backpressure_strategy="sample",
        sample_rate=0.5  # Drop 50%
    )
    queue = BoundedEventQueue(config)

    # Add 100 events
    for i in range(100):
        await queue.emit({"id": i})

    stats = queue.get_stats()
    # Approximately 50% should be queued
    assert 40 <= stats["queue_size"] <= 60
```

### Risks

1. **Event Loss**: Backpressure may drop important events
   - **Mitigation**: Never drop critical events (error, fatal)
   - **Monitoring**: Alert on high drop rates

2. **Performance Overhead**: Queue adds latency
   - **Mitigation**: Make queue flush async and batched
   - **Benchmark**: Measure overhead before deployment

---

## Phase 5: Workflow Template Consolidation + Vertical Overlays

**Duration**: 2-4 days
**Priority**: Medium
**Risk**: Low

### Objectives

1. Move shared workflow templates to framework registry
2. Enable verticals to reference base templates with overlays
3. Eliminate workflow duplication across verticals

### File-Level Changes

#### 5.1 Framework Workflow Templates

**Files to Modify**:
- `victor/workflows/templates/` (new directory)
- `victor/workflows/registry.py`
- `victor/coding/workflows/` (refactor)
- `victor/devops/workflows/` (refactor)

**New File**: `victor/workflows/templates/common_workflows.yaml`

```yaml
# Common workflow templates available to all verticals
templates:
  # Basic read-modify-write workflow
  read_modify_write:
    name: "Read-Modify-Write"
    description: "Read file, make changes, verify"
    stages:
      - type: read
        tools: [read, search]
      - type: plan
        tools: [read, search]
      - type: modify
        tools: [write, edit]
      - type: verify
        tools: [read, test, bash]

  # Multi-step analysis workflow
  analysis_workflow:
    name: "Analysis"
    description: "Analyze codebase and generate report"
    stages:
      - type: discover
        tools: [search, read]
      - type: analyze
        tools: [read, semantic_search]
      - type: report
        tools: [write]

  # Deployment workflow
  deployment_workflow:
    name: "Deployment"
    description: "Deploy changes with validation"
    stages:
      - type: pre_deploy
        tools: [git_status, git_diff]
      - type: deploy
        tools: [docker, kubernetes]
      - type: post_deploy
        tools: [bash, test]
```

**Register Templates**:

```python
# victor/workflows/registry.py
from pathlib import Path
import yaml

class WorkflowTemplateRegistry:
    """Registry for workflow templates."""

    def __init__(self):
        self._templates: Dict[str, WorkflowDefinition] = {}

    def load_templates_from_yaml(self, yaml_path: Path):
        """Load workflow templates from YAML file."""
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        for name, config in data.get("templates", {}).items():
            workflow = self._build_workflow(config)
            self._templates[name] = workflow

    def get_template(self, name: str) -> Optional[WorkflowDefinition]:
        """Get workflow template by name."""
        return self._templates.get(name)

    def extend_template(
        self,
        base_name: str,
        overrides: Dict[str, Any],
    ) -> WorkflowDefinition:
        """Extend base template with custom configuration."""
        base = self.get_template(base_name)
        if not base:
            raise ValueError(f"Template '{base_name}' not found")

        # Deep merge overrides
        return self._merge_workflow(base, overrides)

# Global instance
_workflow_template_registry = WorkflowTemplateRegistry()

def get_workflow_template_registry() -> WorkflowTemplateRegistry:
    """Get global workflow template registry."""
    return _workflow_template_registry
```

#### 5.2 Vertical Workflow Overlays

**Files to Modify**:
- `victor/coding/workflows/code_review_workflow.yaml`
- `victor/devops/workflows/deploy_workflow.yaml`

**Before** (duplicated workflow):

```yaml
# victor/coding/workflows/code_review_workflow.yaml
name: "Code Review"
stages:
  - type: read
    tools: [read, search, semantic_search]  # Full duplication
  - type: analyze
    tools: [read, semantic_search]
  - type: comment
    tools: [write]
  - type: verify
    tools: [test, lint]
```

**After** (overlay approach):

```yaml
# victor/coding/workflows/code_review_workflow.yaml
extends: "read_modify_write"  # Reference framework template

# Only specify what's different
overrides:
  name: "Code Review"
  stages:
    # Extend the verify stage with coding-specific tools
    - stage: "verify"
      tools: [test, lint, type_check, security_scan]  # Add to base
```

### Testing Strategy

```python
# tests/integration/workflows/test_template_overlays.py
def test_workflow_template_extension():
    """Test vertical workflows can extend framework templates."""
    from victor.workflows.registry import get_workflow_template_registry

    registry = get_workflow_template_registry()
    base = registry.get_template("read_modify_write")

    extended = registry.extend_template(
        "read_modify_write",
        {"name": "Code Review", "custom_field": True}
    )

    assert extended.name == "Code Review"
    # Base stages preserved
    assert len(extended.stages) == len(base.stages)
    assert extended.custom_field is True
```

### Risks

1. **Template Complexity**: Overlays may be hard to understand
   - **Mitigation**: Clear documentation and examples
   - **Validation**: Warn on conflicting overrides

2. **Breaking Changes**: Moving workflows changes import paths
   - **Mitigation**: Keep old paths as aliases
   - **Deprecation**: Mark old imports as deprecated

---

## Phase 6: API Surface Tightening

**Duration**: 2-3 days
**Priority**: Medium
**Risk**: Low

### Objectives

1. Replace composite OrchestratorProtocol with narrower protocols
2. Simplify API surfaces across framework modules
3. Remove legacy fallback paths

### File-Level Changes

#### 6.1 Narrow Protocol Definitions

**Files to Modify**:
- `victor/framework/protocols.py`
- `victor/framework/agent.py`
- `victor/framework/state.py`
- `victor/framework/tools.py`

**Define Narrower Protocols**:

```python
# victor/framework/protocols.py
from typing import Protocol, List, Optional, Dict, Any

# BEFORE: Large composite protocol
class OrchestratorProtocol(Protocol):
    """Composite protocol - TOO LARGE"""
    def chat(self, message: str) -> str: ...
    def stream_chat(self, message: str): ...
    def get_tools(self) -> List[Tool]: ...
    def get_state(self) -> Dict: ...
    def set_state(self, state: Dict): ...
    # ... 20+ more methods

# AFTER: Split into focused protocols
class ChatProtocol(Protocol):
    """Chat-specific operations."""
    def chat(self, message: str) -> str: ...
    def stream_chat(self, message: str): ...

class StateProtocol(Protocol):
    """State management operations."""
    def get_state(self) -> Dict[str, Any]: ...
    def set_state(self, state: Dict[str, Any]): ...

class ToolProtocol(Protocol):
    """Tool operations."""
    def get_tools(self) -> List[ToolDefinition]: ...
    def execute_tool(self, name: str, **kwargs): ...

# Framework modules depend on minimal protocols
class AgentFactory:
    """Only needs ChatProtocol."""
    def create_agent(self, chat: ChatProtocol) -> Agent:
        return Agent(chat)

class StateManager:
    """Only needs StateProtocol."""
    def __init__(self, state: StateProtocol):
        self._state = state

class ToolExecutor:
    """Only needs ToolProtocol."""
    def __init__(self, tools: ToolProtocol):
        self._tools = tools
```

**Update Framework Modules**:

```python
# victor/framework/agent.py
# BEFORE: Depends on large OrchestratorProtocol
class Agent:
    def __init__(self, orchestrator: OrchestratorProtocol):
        self._orchestrator = orchestrator

# AFTER: Depends on minimal ChatProtocol
class Agent:
    def __init__(self, chat: ChatProtocol):
        self._chat = chat
```

#### 6.2 Remove Legacy Fallbacks

**Files to Modify**:
- `victor/framework/vertical_integration.py`
- `victor/agent/capability_registry.py`

```python
# victor/framework/vertical_integration.py
# BEFORE: Legacy fallback
class VerticalIntegrationPipeline:
    def apply(self, vertical, orchestrator):
        try:
            # New path
            self._apply_via_step_handlers(vertical, orchestrator)
        except Exception:
            # Fallback to legacy
            self._legacy_apply(vertical, orchestrator)

# AFTER: No legacy fallback
class VerticalIntegrationPipeline:
    def __init__(self, strict_mode: bool = True):
        self._strict_mode = strict_mode

    def apply(self, vertical, orchestrator):
        if self._strict_mode:
            # Only new path, raises on error
            self._apply_via_step_handlers(vertical, orchestrator)
        else:
            # Allow legacy for migration
            try:
                self._apply_via_step_handlers(vertical, orchestrator)
            except Exception:
                logger.warning("Falling back to legacy integration")
                self._legacy_apply(vertical, orchestrator)
```

### Testing Strategy

```python
# tests/unit/framework/test_narrow_protocols.py
def test_agent_uses_chat_protocol():
    """Test Agent depends only on ChatProtocol."""
    from victor.framework.protocols import ChatProtocol
    from victor.framework.agent import Agent

    # Create mock with only chat methods
    class MockChat:
        def chat(self, message: str) -> str:
            return f"Response to: {message}"

    chat = MockChat()
    agent = Agent(chat)

    # Should work with ChatProtocol only
    assert isinstance(chat, ChatProtocol)

# tests/integration/framework/test_strict_integration.py
def test_strict_mode_rejects_legacy():
    """Test strict mode rejects legacy integration."""
    pipeline = VerticalIntegrationPipeline(strict_mode=True)

    with pytest.raises(IntegrationError):
        pipeline.apply(incompatible_vertical, orchestrator)
```

### Risks

1. **Breaking Changes**: Narrower protocols may break existing code
   - **Mitigation**: Adapter layer for backward compatibility
   - **Deprecation**: Gradual migration with warnings

2. **Adapter Complexity**: Adapters add indirection
   - **Mitigation**: Keep adapters temporary, remove after migration
   - **Monitoring**: Track adapter usage

---

## Cross-Cutting Concerns

### Documentation Updates

**Files to Create**:
- `victor/framework/README.md` - Framework architecture guide
- `docs/development/capability-providers.md` - Capability provider guide
- `docs/development/tool-packs.md` - Tool pack usage guide
- `docs/migration/protocol-migration.md` - Protocol migration guide

### Migration Guides

**External Verticals**:

```markdown
# docs/migration/external-verticals.md

## Migrating to New Capability System

### Step 1: Update Capability Providers

Move capability configs from direct writes to VerticalContext:

**Before:**
```python
def apply_capability(orchestrator):
    orchestrator.my_config = value
```

**After:**
```python
def apply_capability(context: VerticalContext):
    context.set_capability_config("my_config", value)
```

### Step 2: Register Capabilities

Ensure all capabilities are registered:

```python
from victor.agent.capability_registry import get_capability_registry

registry = get_capability_registry()
registry.register("my_capability", MyCapabilityProvider())
```

### Step 3: Use Tool Packs

Replace duplicated tool lists with tool packs:

```python
from victor.framework.tool_packs import BASE_FILE_OPS, ToolPack

MY_PACK = ToolPack(
    name="my_vertical",
    extends="base_file_ops",
    tools=["my_tool"],
)
```
```

---

## Implementation Timeline

| Phase | Duration | Dependencies | Risk | Value |
|-------|----------|--------------|------|-------|
| Phase 1 | 2-4 days | None | Medium | High |
| Phase 2 | 2-3 days | None | Low | Medium |
| Phase 3 | 3-5 days | Phase 1 | Medium | High |
| Phase 4 | 3-4 days | None | Low | High |
| Phase 5 | 2-4 days | None | Low | Medium |
| Phase 6 | 2-3 days | Phase 3 | Low | Medium |
| **Total** | **14-23 days** | | | |

**Recommended Order**:
1. Start with Phase 1 (highest value, unblocks Phase 3)
2. Phase 2 can run in parallel with Phase 1
3. Phase 3 depends on Phase 1
4. Phases 4-6 can run in parallel after Phase 1

---

## Success Metrics

### Code Quality

- **SOLID Compliance**: Zero violations
- **Test Coverage**: >90% for refactored modules
- **Type Safety**: 100% mypy compliance

### Performance

- **Memory Usage**: 30-50% reduction (bounded caches)
- **Cache Hit Rate**: >60% maintained
- **Event Throughput**: Handle 10k events/sec without loss

### Developer Experience

- **API Surface**: Reduced by 40% (narrower protocols)
- **Boilerplate**: Reduced by 60% (tool packs, stage templates)
- **Onboarding Time**: <1 day for new vertical developers

---

## Rollback Plan

Each phase can be independently rolled back:

```bash
# Per-phase rollback
git revert <phase-commit-hash>

# Feature flag rollback
export VICTOR_STRICT_CAPABILITIES=false
export VICTOR_BOUNDED_CACHES=false
export VICTOR_OBSERVABILITY_BACKPRESSURE=false
```

**Rollback Decision Criteria**:
- Test failures >5%
- Performance regression >20%
- Breaking user reports >10/day

---

## Conclusion

This refactor plan addresses all identified SOLID violations, architecture gaps, and scalability risks. The phased approach minimizes risk while delivering continuous value.

**Next Steps**:
1. Review and approve this plan
2. Set up feature flags for gradual rollout
3. Begin with Phase 1 (highest priority)
4. Monitor metrics and adjust timeline as needed

**Estimated Timeline**: 14-23 days for full completion
**Risk Level**: Medium (mitigated by phased approach)
**Value**: High (addresses all critical issues)

---

**Document Version**: 1.0
**Last Updated**: 2025-01-24
**Status**: Draft - Pending Review
