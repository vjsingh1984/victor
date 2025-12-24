# Tool Selection Architecture

**HIGH-002: Unified Tool Selection Architecture**

This document describes the tool selection architecture in Victor, implemented across 3 releases as part of HIGH-002.

## Overview

Victor's tool selection system dynamically chooses which tools to provide to the LLM based on the user's message, conversation context, and system configuration. The system uses a strategy pattern with three pluggable strategies:

- **Keyword**: Fast registry-based matching (<1ms)
- **Semantic**: ML-based embedding similarity (10-50ms)
- **Hybrid**: Blends both approaches (best of both worlds)

## Architecture

### Core Components

```
┌──────────────────────────────────────────────────┐
│           AgentOrchestrator                      │
│   (calls ToolSelector.select_tools())            │
└────────────────┬─────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────┐
│           ToolSelector                            │
│   (Orchestrator with strategy injection)         │
│                                                   │
│   - conversation_state                            │
│   - task_tracker                                  │
│   - _strategy: IToolSelector                     │
└────────────────┬─────────────────────────────────┘
                 │
                 │ delegates to
                 ▼
┌──────────────────────────────────────────────────┐
│         IToolSelector (Protocol)                  │
│                                                   │
│  + select_tools(prompt, context) → tools         │
│  + get_supported_features() → features           │
│  + record_tool_execution(name, success)          │
│  + close()                                        │
└──────────────┬───────────────────────────────────┘
               │
     ┌─────────┼─────────┐
     │         │         │
     ▼         ▼         ▼
┌────────┐ ┌────────┐ ┌────────┐
│Keyword │ │Semantic│ │ Hybrid │
│Selector│ │Selector│ │Selector│
└────────┘ └────────┘ └────────┘
```

### Strategy Implementations

#### KeywordToolSelector

Fast keyword-based selection using tool metadata registry:

- **Performance**: <1ms selection time
- **Requirements**: None (no embeddings needed)
- **Features**:
  - Registry-based keyword matching from @tool decorators
  - Vertical mode support (enabled_tools filtering)
  - Stage-aware filtering (read-only for analysis)
  - Small model tool capping (max 10 tools)

```python
from victor.tools.keyword_tool_selector import KeywordToolSelector

selector = KeywordToolSelector(
    tools=tool_registry,
    conversation_state=conversation_state,
    model="gpt-4",
    provider_name="openai",
    enabled_tools=None,
)
```

#### SemanticToolSelector

ML-based selection using embedding similarity:

- **Performance**: 10-50ms selection time
- **Requirements**: Embedding service (sentence-transformers, Ollama, etc.)
- **Features**:
  - Semantic similarity matching
  - Context-aware selection
  - Cost optimization
  - Usage learning (tracks success/failure)
  - Workflow pattern detection

```python
from victor.tools.semantic_selector import SemanticToolSelector

selector = SemanticToolSelector(
    embedding_service=embedding_service,
    cache_embeddings=True,
)
```

#### HybridToolSelector

Blends semantic and keyword strategies:

- **Performance**: 10-50ms (semantic dominated)
- **Requirements**: Embedding service (for semantic component)
- **Features**: All features (union of semantic + keyword)
- **Weights**: Configurable (default: 70% semantic, 30% keyword)

```python
from victor.tools.hybrid_tool_selector import (
    HybridToolSelector,
    HybridSelectorConfig,
)

config = HybridSelectorConfig(
    semantic_weight=0.7,
    keyword_weight=0.3,
    min_semantic_tools=3,
    min_keyword_tools=2,
    max_total_tools=15,
)

selector = HybridToolSelector(
    semantic_selector=semantic_selector,
    keyword_selector=keyword_selector,
    config=config,
)
```

## Configuration

### Tool Selection Strategy

Configure via `tool_selection_strategy` setting:

```python
# In .env or profiles.yaml
tool_selection_strategy = "auto"  # auto (recommended), keyword, semantic, hybrid
```

### Auto-Selection Logic

When `strategy = "auto"`, Victor automatically selects the best strategy:

1. If `airgapped_mode` → **keyword** (no embeddings available)
2. If embedding service available → **semantic** (best quality)
3. Fallback → **keyword** (always works)

```python
# Example auto-selection
from victor.agent.tool_selector_factory import create_tool_selector_strategy

selector = create_tool_selector_strategy(
    strategy="auto",
    tools=tool_registry,
    settings=settings,
    embedding_service=embedding_service,  # None in air-gapped mode
)
```

## IToolSelector Protocol

All strategies implement the `IToolSelector` protocol:

```python
from typing import Protocol, List
from victor.providers.base import ToolDefinition
from victor.agent.protocols import ToolSelectionContext, ToolSelectorFeatures

class IToolSelector(Protocol):
    """Unified protocol for all tool selection strategies."""

    async def select_tools(
        self,
        prompt: str,
        context: ToolSelectionContext,
    ) -> List[ToolDefinition]:
        """Select relevant tools for the given prompt and context."""
        ...

    def get_supported_features(self) -> ToolSelectorFeatures:
        """Return features supported by this selector."""
        ...

    def record_tool_execution(
        self,
        tool_name: str,
        success: bool,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record tool execution for learning (optional)."""
        ...

    async def close(self) -> None:
        """Cleanup resources (e.g., embedding service)."""
        ...
```

### ToolSelectionContext

Context passed to all selectors:

```python
from dataclasses import dataclass
from victor.agent.conversation_state import ConversationStage
from victor.agent.task_classifier import ClassificationResult

@dataclass
class ToolSelectionContext:
    """Context for tool selection."""
    conversation_history: List[Dict[str, Any]]
    conversation_depth: int
    conversation_stage: ConversationStage
    classification_result: Optional[ClassificationResult]
    planned_tools: Optional[Set[str]]
    max_tools: int
    use_cost_aware: bool
    vertical: Optional[str]  # Vertical mode (e.g., "research", "devops")
```

### ToolSelectorFeatures

Feature flags for selector capabilities:

```python
from dataclasses import dataclass

@dataclass
class ToolSelectorFeatures:
    """Features supported by a tool selector."""
    supports_semantic_matching: bool
    supports_context_awareness: bool
    supports_cost_optimization: bool
    supports_usage_learning: bool
    supports_workflow_patterns: bool
    requires_embeddings: bool
```

## Factory Pattern

Create selectors using the factory function:

```python
from victor.agent.tool_selector_factory import create_tool_selector_strategy

selector = create_tool_selector_strategy(
    strategy="hybrid",  # keyword, semantic, hybrid, auto
    tools=tool_registry,
    conversation_state=conversation_state,
    model="gpt-4",
    provider_name="openai",
    enabled_tools=None,  # Optional vertical filter
    embedding_service=embedding_service,  # Required for semantic/hybrid
    settings=settings,  # For auto-selection logic
)
```

## Migration Guide

### Breaking Changes (v2.0.0)

1. **Removed `use_semantic_tool_selection` setting**
   - Old: `use_semantic_tool_selection = True`
   - New: `tool_selection_strategy = "semantic"`

2. **Removed `use_semantic` parameter from `ToolSelector.select_tools()`**
   - Old: `selector.select_tools(message, use_semantic=True)`
   - New: `selector.select_tools(message)`
   - Strategy is now configured globally via settings

3. **Removed deprecated protocols**
   - `ToolSelectorProtocol` → Use `IToolSelector`
   - `SemanticToolSelectorProtocol` → Use `IToolSelector`

### Migration Steps

**Step 1**: Update settings

```yaml
# Old (profiles.yaml)
use_semantic_tool_selection: true

# New (profiles.yaml)
tool_selection_strategy: semantic  # or keyword, hybrid, auto
```

**Step 2**: Update code (if calling ToolSelector directly)

```python
# Old
tools = await selector.select_tools(message, use_semantic=True)

# New
tools = await selector.select_tools(message)
```

**Step 3**: Update custom selectors (if implementing IToolSelector)

```python
# Old (custom protocol)
class MyToolSelector(ToolSelectorProtocol):
    def select_tools(self, prompt: str, max_tools: int, stage: Optional[ConversationStage]) -> List[str]:
        ...

# New (IToolSelector protocol)
class MyToolSelector(IToolSelector):
    async def select_tools(self, prompt: str, context: ToolSelectionContext) -> List[ToolDefinition]:
        ...

    def get_supported_features(self) -> ToolSelectorFeatures:
        ...

    def record_tool_execution(self, tool_name: str, success: bool, context=None):
        ...

    async def close(self):
        ...
```

## Performance Characteristics

| Strategy | Selection Time | Embeddings Required | Quality | Use Case |
|----------|----------------|---------------------|---------|----------|
| Keyword  | <1ms           | No                  | Good    | Air-gapped, fast responses |
| Semantic | 10-50ms        | Yes                 | Best    | Best quality, context-aware |
| Hybrid   | 10-50ms        | Yes                 | Best    | Best of both worlds |

## Shared Utilities

Extracted utilities for all selectors:

### selection_filters.py

Pure filtering functions:

- `is_small_model(model_name, provider_name) → bool`
- `needs_web_tools(message) → bool`
- `deduplicate_tools(tools) → List[ToolDefinition]`
- `blend_tool_results(semantic_tools, keyword_tools, weights) → List[ToolDefinition]`
- `cap_tools_to_max(tools, max_tools) → List[ToolDefinition]`

### selection_common.py

Stateful utilities:

- `get_critical_tools(tools: ToolRegistry) → Set[str]`
- `get_tools_by_category(category: str) → Set[str]`
- `detect_categories_from_message(message: str) → Set[str]`
- `get_web_tools() → Set[str]`

## Testing

Run tool selection tests:

```bash
# All tool selection tests
pytest tests/unit/test_tool_selection*.py -v

# Specific strategy tests
pytest tests/unit/test_keyword_selector.py -v
pytest tests/unit/test_hybrid_selector.py -v

# Performance benchmarks
pytest tests/unit/test_tool_selection.py::TestPerformance -v
```

## Implementation Timeline

- **Release 1 (Week 1)**: Protocol + Utils
  - IToolSelector protocol defined
  - Shared utilities extracted
  - ToolSelector and SemanticToolSelector implement IToolSelector

- **Release 2 (Weeks 2-3)**: Strategy Pattern
  - KeywordToolSelector created
  - Strategy factory implemented
  - ToolSelector uses strategy injection
  - Settings migration with backward compatibility

- **Release 3 (Week 4)**: Cleanup + Hybrid
  - HybridToolSelector implemented
  - Deprecated code removed
  - Comprehensive testing
  - Documentation complete

## References

- [HIGH-002 Implementation Plan](/Users/vijaysingh/.claude/plans/stateless-brewing-crane.md)
- [CLAUDE.md](../CLAUDE.md) - Architecture overview
- [Tool Catalog](TOOL_CATALOG.md) - All available tools
- [Tool Metadata Registry](../victor/tools/metadata_registry.py)
