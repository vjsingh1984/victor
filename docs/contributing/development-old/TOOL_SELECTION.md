# Tool Selection Architecture

> **Archived**: This document is kept for historical context and may be outdated. See `docs/contributing/index.md` for current guidance.


**Status**: Release 2 (Strategy Pattern) - ✅ Complete
**Plan Reference**: `~/.claude/plans/stateless-brewing-crane.md`
**Priority**: HIGH (Priority 7)

## Overview

Victor's unified tool selection architecture provides a flexible, pluggable strategy pattern for selecting relevant tools based on user queries. The system supports multiple selection strategies with automatic fallback and intelligent adaptation based on model capabilities, conversation context, and performance requirements.

### Key Features

- **Strategy Pattern**: Pluggable selection algorithms (keyword, semantic, hybrid, auto)
- **Performance**: Keyword <1ms, Semantic ~50ms, Hybrid ~30ms
- **Adaptive**: Auto-selects optimal strategy based on environment
- **Backward Compatible**: Migrates from legacy `use_semantic_tool_selection` setting
- **RL-Ready**: Supports reinforcement learning for adaptive thresholds

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    OrchestratorFactory                      │
│  create_tool_selector() → uses new strategy factory         │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│          create_tool_selector_strategy()                     │
│  victor/agent/tool_selector_factory.py                       │
│                                                               │
│  strategy: "auto" | "keyword" | "semantic" | "hybrid"        │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
   ┌──────────┐     ┌──────────┐     ┌──────────┐
   │ Keyword  │     │ Semantic │     │ Hybrid   │
   │ Selector │     │ Selector │     │ Selector │
   │  <1ms    │     │  ~50ms   │     │  ~30ms   │
   └──────────┘     └──────────┘     └──────────┘
         │                 │                 │
         └─────────────────┴─────────────────┘
                           ▼
                  ┌────────────────┐
                  │  IToolSelector │
                  │    Protocol    │
                  └────────────────┘
```

## Strategies

### 1. Keyword Strategy (`keyword`)

**File**: `victor/tools/keyword_tool_selector.py`

Fast metadata-based selection using tool keywords and categories. No embeddings required.

- **Performance**: <1ms per selection
- **Requirements**: None (works offline, air-gapped)
- **Best For**: Small models, air-gapped mode, fast response requirements
- **Method**: Pattern matching on tool keywords, categories, descriptions

```python
from victor.agent.tool_selector_factory import create_tool_selector_strategy

selector = create_tool_selector_strategy(
    strategy="keyword",
    tools=tool_registry,
    model="qwen2.5-coder:7b",
    provider_name="ollama",
)

result = selector.select_tools("Search for Python files", limit=10)
# Returns: ["search", "read", "ls", ...]
```

### 2. Semantic Strategy (`semantic`)

**File**: `victor/tools/semantic_selector.py`

ML-based embedding similarity search using vector embeddings.

- **Performance**: ~50ms per selection (with caching)
- **Requirements**: Embedding service (sentence-transformers, Ollama, etc.)
- **Best For**: Complex queries, ambiguous intent, large models
- **Method**: Embedding similarity search with threshold filtering

```python
from victor.storage.embeddings.service import get_embedding_service

selector = create_tool_selector_strategy(
    strategy="semantic",
    tools=tool_registry,
    embedding_service=get_embedding_service(),
)

result = selector.select_tools("Analyze code performance", limit=10)
# Returns: ["profile", "search", "read", "shell", ...]
```

### 3. Hybrid Strategy (`hybrid`)

**File**: `victor/tools/hybrid_tool_selector.py`

Combines semantic quality with keyword reliability using Reciprocal Rank Fusion (RRF).

- **Performance**: ~30ms per selection
- **Requirements**: Embedding service
- **Best For**: Production use, balanced quality/speed
- **Method**: RRF blend (70% semantic, 30% keyword)

```python
selector = create_tool_selector_strategy(
    strategy="hybrid",
    tools=tool_registry,
    embedding_service=get_embedding_service(),
    settings=settings,
)

result = selector.select_tools("Debug failing test", limit=10)
# Returns semantic + keyword blend with RL boost applied
```

### 4. Auto Strategy (`auto`)

**File**: `victor/agent/tool_selector_factory.py:_auto_select_strategy()`

Automatically selects optimal strategy based on environment.

**Selection Logic**:
1. If `airgapped_mode=True` → `keyword` (no embeddings available)
2. If embedding service available → `semantic` (best quality)
3. Fallback → `keyword` (always works, no dependencies)

```python
selector = create_tool_selector_strategy(
    strategy="auto",
    tools=tool_registry,
    embedding_service=get_embedding_service(),
    settings=settings,
)

# Automatically picks semantic or keyword based on environment
```

## Configuration

### Settings

Configure tool selection strategy via `tool_selection_strategy` setting:

```yaml
# ~/.victor/profiles.yaml or .victor/config.yaml
tool_selection_strategy: auto  # auto | keyword | semantic | hybrid

# Legacy setting (DEPRECATED - auto-migrates to new setting)
use_semantic_tool_selection: true  # Will be removed in v2.0
```

### Environment Variables

```bash
# Set strategy via environment
export VICTOR_TOOL_SELECTION_STRATEGY=hybrid

# Legacy setting (DEPRECATED)
export VICTOR_USE_SEMANTIC_TOOL_SELECTION=true
```

### Python Configuration

```python
from victor.config.settings import Settings

# New setting (recommended)
settings = Settings(tool_selection_strategy="semantic")

# Legacy setting (DEPRECATED - shows warning)
settings = Settings(use_semantic_tool_selection=True)
# Auto-migrates to tool_selection_strategy="semantic"
```

## Migration Guide

### From Legacy to New Setting

**Old (Deprecated)**:
```python
settings = Settings(use_semantic_tool_selection=True)
```

**New (Recommended)**:
```python
settings = Settings(tool_selection_strategy="semantic")
```

### Mapping Table

| Old Setting | New Setting | Strategy |
|------------|-------------|----------|
| `use_semantic_tool_selection=True` | `tool_selection_strategy="semantic"` | Semantic selection |
| `use_semantic_tool_selection=False` | `tool_selection_strategy="keyword"` | Keyword selection |
| (not set / default) | `tool_selection_strategy="auto"` | Auto selection |

### Factory Method Migration

**Old (Deprecated)**:
```python
from victor.agent.tool_selection import ToolSelector

selector = ToolSelector(
    tools=tool_registry,
    semantic_selector=semantic_selector,
    model=model,
    provider_name=provider_name,
)
```

**New (Recommended)**:
```python
from victor.agent.tool_selector_factory import create_tool_selector_strategy

selector = create_tool_selector_strategy(
    strategy="auto",
    tools=tool_registry,
    settings=settings,
    embedding_service=get_embedding_service(),
)
```

## API Reference

### `create_tool_selector_strategy()`

Factory function for creating tool selectors.

```python
def create_tool_selector_strategy(
    strategy: str,  # "auto" | "keyword" | "semantic" | "hybrid"
    tools: ToolRegistry,
    conversation_state: Optional[ConversationStateMachine] = None,
    model: str = "",
    provider_name: str = "",
    enabled_tools: Optional[Set[str]] = None,
    embedding_service: Optional[EmbeddingService] = None,
    settings: Optional[Settings] = None,
) -> IToolSelector:
    ...
```

**Parameters**:
- `strategy`: Selection strategy name
- `tools`: Tool registry with available tools
- `conversation_state`: Optional state machine for stage-aware selection
- `model`: Model name for capability detection
- `provider_name`: Provider name for small model detection
- `enabled_tools`: Optional vertical-specific tool filter
- `embedding_service`: Required for semantic/hybrid strategies
- `settings`: Optional settings for auto-selection

**Returns**: `IToolSelector` implementation

### `IToolSelector` Protocol

```python
@runtime_checkable
class IToolSelector(Protocol):
    def select_tools(
        self,
        task: str,
        *,
        limit: int = 10,
        min_score: float = 0.0,
        context: Optional[ToolSelectionContext] = None,
    ) -> ToolSelectionResult:
        ...

    def get_tool_score(
        self,
        tool_name: str,
        task: str,
        *,
        context: Optional[ToolSelectionContext] = None,
    ) -> float:
        ...

    @property
    def strategy(self) -> ToolSelectionStrategy:
        ...
```

### `ToolSelectionResult`

```python
@dataclass
class ToolSelectionResult:
    tool_names: List[str]  # Ordered tool names (most relevant first)
    scores: Dict[str, float]  # Relevance scores (0.0-1.0)
    strategy_used: ToolSelectionStrategy  # Strategy that produced result
    metadata: Dict[str, Any]  # Additional metadata
```

## Performance Benchmarks

| Strategy | Latency | Quality | Requirements |
|----------|---------|---------|--------------|
| Keyword | <1ms | Good | None |
| Semantic | ~50ms | Excellent | Embedding service |
| Hybrid | ~30ms | Very Good | Embedding service |
| Auto | Variable | Excellent | Auto-detected |

**Benchmarks**: `tests/unit/tools/test_*_tool_selector.py`

## Testing

### Unit Tests

```bash
# Test all selectors
pytest tests/unit/tools/test_keyword_tool_selector.py -v
pytest tests/unit/tools/test_hybrid_tool_selector.py -v
pytest tests/unit/tools/test_selection.py -v

# Test factory
pytest tests/unit/agent/test_tool_selector_factory.py -v

# Test settings migration
pytest tests/unit/config/test_settings.py::test_tool_selection_migration -v
```

### Integration Tests

```bash
# Test with real embedding service
pytest tests/integration/tools/test_semantic_selector_integration.py -v
```

### Manual Testing

```python
from victor.config.settings import Settings
from victor.agent.tool_selector_factory import create_tool_selector_strategy
from victor.tools.base import ToolRegistry
from victor.storage.embeddings.service import get_embedding_service

# Setup
settings = Settings(tool_selection_strategy="auto")
tools = ToolRegistry()
# ... register tools ...

# Create selector
selector = create_tool_selector_strategy(
    strategy="auto",
    tools=tools,
    settings=settings,
    embedding_service=get_embedding_service(),
)

# Test selection
result = selector.select_tools("Search for Python files", limit=10)
print(f"Selected tools: {result.tool_names}")
print(f"Scores: {result.scores}")
print(f"Strategy: {result.strategy_used}")
```

## Troubleshooting

### Issue: "Embedding service not available"

**Cause**: Semantic/hybrid strategy requires embedding service.

**Solution**:
1. Ensure embedding service is initialized
2. Use `keyword` strategy for air-gapped mode
3. Check `embedding_provider` and `embedding_model` settings

```python
from victor.storage.embeddings.service import get_embedding_service

try:
    embedding_service = get_embedding_service()
except Exception as e:
    # Fallback to keyword strategy
    selector = create_tool_selector_strategy(strategy="keyword", ...)
```

### Issue: "Unknown tool selection strategy"

**Cause**: Invalid strategy name.

**Solution**: Use valid strategy: `"auto"`, `"keyword"`, `"semantic"`, or `"hybrid"`

```python
# Invalid
selector = create_tool_selector_strategy(strategy="invalid", ...)

# Valid
selector = create_tool_selector_strategy(strategy="semantic", ...)
```

### Issue: Deprecation warning for `use_semantic_tool_selection`

**Cause**: Using legacy setting.

**Solution**: Migrate to `tool_selection_strategy` setting.

```python
# Old (shows deprecation warning)
settings = Settings(use_semantic_tool_selection=True)

# New (recommended)
settings = Settings(tool_selection_strategy="semantic")
```

## Future Enhancements (Release 3)

- [ ] Remove deprecated `use_semantic_tool_selection` setting
- [ ] Remove old `ToolSelector` class
- [ ] Add RL-based adaptive thresholds
- [ ] Add tool selection analytics dashboard
- [ ] Support custom strategy plugins

## References

- **Plan**: `~/.claude/plans/stateless-brewing-crane.md`
- **Factory**: `victor/agent/tool_selector_factory.py`
- **Protocol**: `victor/protocols/tool_selector.py`
- **Keyword Selector**: `victor/tools/keyword_tool_selector.py`
- **Semantic Selector**: `victor/tools/semantic_selector.py`
- **Hybrid Selector**: `victor/tools/hybrid_tool_selector.py`
- **Orchestrator Factory**: `victor/agent/orchestrator_factory.py:create_tool_selector()`
- **Settings**: `victor/config/settings.py:tool_selection_strategy`

## Changelog

### Release 2 (Current)
- ✅ Added `tool_selection_strategy` setting
- ✅ Created `create_tool_selector_strategy()` factory
- ✅ Implemented Keyword, Semantic, Hybrid strategies
- ✅ Added deprecation warnings for legacy setting
- ✅ Updated `OrchestratorFactory.create_tool_selector()`
- ✅ Auto-migration from `use_semantic_tool_selection`

### Release 3 (Planned)
- [ ] Remove deprecated code (`use_semantic_tool_selection`, old `ToolSelector`)
- [ ] RL-based adaptive thresholds
- [ ] Performance optimizations
- [ ] Custom strategy plugins
