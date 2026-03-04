# UnifiedToolRegistry Design

## Overview

The UnifiedToolRegistry consolidates tool management functionality currently scattered across multiple components:

1. **SharedToolRegistry** - Tool discovery and class caching
2. **ToolRegistry** - Tool registration and lifecycle management
3. **SemanticToolSelector** - Intelligent tool selection
4. **ToolMetadataRegistry** - Tool metadata and categorization

## Goals

1. **Single Source of Truth**: One registry for all tool operations
2. **Backward Compatibility**: Existing code continues to work
3. **Performance**: No performance regression (maintain singleton pattern)
4. **Extensibility**: Plugin tools, dynamic registration, deprecation
5. **Observability**: Metrics, logging, and debugging support

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    UnifiedToolRegistry                        │
│  (Singleton - thread-safe, memory-efficient)                   │
├─────────────────────────────────────────────────────────────┤
│ ┌───────────────────────────────────────────────────────────┐ │
│ │         Tool Discovery & Registration                      │ │
│ │  - discover() - Scan for tool modules                      │ │
│ │  - register() - Add tool to registry                       │ │
│ │  - unregister() - Remove tool                              │ │
│ │  - register_plugin() - Load external plugins               │ │
│ └───────────────────────────────────────────────────────────┘ │
│ ┌───────────────────────────────────────────────────────────┐ │
│ │              Tool Selection Engine                         │ │
│ │  - select_tools() - Primary selection API                  │ │
│ │  - SemanticToolSelector - Embedding-based                 │ │
│ │  - KeywordToolSelector - Fast keyword match               │ │
│ │  - HybridToolSelector - Combined approach                 │ │
│ └───────────────────────────────────────────────────────────┘ │
│ ┌───────────────────────────────────────────────────────────┐ │
│ │              Metadata & Categorization                     │ │
│ │  - get_metadata() - Tool information                       │ │
│ │  - get_by_category() - Filter by category                  │ │
│ │  - get_by_tier() - Filter by cost tier                     │ │
│ │  - search() - Semantic search                              │ │
│ └───────────────────────────────────────────────────────────┘ │
│ ┌───────────────────────────────────────────────────────────┐ │
│ │              Lifecycle Management                          │ │
│ │  - enable/disable - Tool state management                 │ │
│ │  - deprecate - Mark tools as deprecated                  │ │
│ │  - alias - Tool name aliases                              │ │
│ └───────────────────────────────────────────────────────────┘ │
│ ┌───────────────────────────────────────────────────────────┐ │
│ │              Observability & Hooks                         │ │
│ │  - Hooks - pre/post execution callbacks                   │ │
│ │  - Metrics - Usage statistics                             │ │
│ │  - Logging - Audit trail                                  │ │
│ └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Core Interface

```python
class UnifiedToolRegistry:
    """Unified registry for tool discovery, registration, and selection.

    Thread-safe singleton pattern for memory efficiency across
    multiple concurrent sessions.
    """

    # Singleton
    @classmethod
    def get_instance(cls) -> "UnifiedToolRegistry": ...
    @classmethod
    def reset_instance(cls) -> None: ...

    # Registration
    async def register(
        self,
        tool: Union[BaseTool, Callable, ToolDefinition],
        *,
        enabled: bool = True,
        category: Optional[str] = None,
        tier: CostTier = CostTier.MEDIUM,
    ) -> None: ...

    async def unregister(self, name: str) -> bool: ...
    async def register_plugin(self, plugin_path: str) -> None: ...

    # Discovery
    async def discover(
        self,
        paths: Optional[List[str]] = None,
        airgapped: bool = False,
    ) -> List[str]: ...

    # Selection (main API)
    async def select_tools(
        self,
        query: str,
        *,
        context: Optional[ToolSelectionContext] = None,
        max_tools: int = 10,
        strategy: SelectionStrategy = SelectionStrategy.AUTO,
    ) -> List[str]: ...

    # Access
    def get(self, name: str) -> Optional[BaseTool]: ...
    def list_tools(
        self,
        *,
        enabled_only: bool = True,
        category: Optional[str] = None,
        tier: Optional[CostTier] = None,
    ) -> List[str]: ...

    # Metadata
    def get_metadata(self, name: str) -> ToolMetadata: ...
    def get_categories(self) -> Dict[str, List[str]]: ...
    def search(self, query: str) -> List[str]: ...

    # Lifecycle
    async def enable(self, name: str) -> bool: ...
    async def disable(self, name: str) -> bool: ...
    async def deprecate(
        self,
        name: str,
        replacement: Optional[str] = None,
        message: str = "",
    ) -> None: ...

    # Aliases
    def add_alias(self, name: str, alias: str) -> None: ...
    def resolve_alias(self, alias: str) -> str: ...

    # Schemas
    def get_schemas(
        self,
        *,
        enabled_only: bool = True,
    ) -> List[Dict[str, Any]]: ...

    # Hooks
    def register_hook(
        self,
        hook: ToolHook,
        phase: HookPhase = HookPhase.BEFORE,
    ) -> None: ...

    # Observability
    def get_metrics(self) -> ToolMetrics: ...
```

## Migration Strategy

### Phase 1: Adapter Layer (Week 1-2)

Create adapters for existing registries:

```python
# victor/tools/unified_adapter.py

class SharedToolRegistryAdapter:
    """Adapter to make SharedToolRegistry use UnifiedToolRegistry."""

    def __init__(self, unified: UnifiedToolRegistry):
        self._unified = unified

    # Delegate to unified registry
    def get_tool_classes(self, airgapped: bool = False):
        return self._unified.get_tool_classes(airgapped)

    def create_tool_instance(self, tool_name: str):
        return self._unified.create(tool_name)

class ToolRegistryAdapter:
    """Adapter to make ToolRegistry use UnifiedToolRegistry."""

    def __init__(self, unified: UnifiedToolRegistry):
        self._unified = unified

    # Delegate to unified registry
    def register(self, tool, enabled=True):
        return self._unified.register(tool, enabled=enabled)

    def get_tool_schemas(self, only_enabled=True):
        return self._unified.get_schemas(enabled_only=only_enabled)
```

### Phase 2: Gradual Migration (Week 3-8)

1. Update AgentOrchestrator to use UnifiedToolRegistry
2. Update tool coordinators
3. Update verticals
4. Update tests

### Phase 3: Deprecation (Week 9-10)

1. Mark old registries as deprecated
2. Add migration warnings
3. Update documentation

### Phase 4: Cleanup (Week 11-13)

1. Remove deprecated code
2. Finalize migration
3. Update all references

## Implementation Plan

### File Structure

```
victor/tools/
├── unified/
│   ├── __init__.py
│   ├── registry.py          # UnifiedToolRegistry
│   ├── discovery.py          # Tool discovery logic
│   ├── selection.py          # Selection strategies
│   ├── metadata.py           # Metadata management
│   ├── lifecycle.py          # Enable/disable/deprecate
│   └── adapters.py           # Backward compatibility
```

### Dependencies

- EmbeddingService (for semantic selection)
- ToolMetadataRegistry (for tool categories)
- BaseRegistry (base class)

### Backward Compatibility

Keep existing APIs working via adapters:

```python
# victor/agent/shared_tool_registry.py

from victor.tools.unified import UnifiedToolRegistry

class SharedToolRegistry:
    """Backward compatibility wrapper."""

    @classmethod
    def get_instance(cls):
        return UnifiedToolRegistryAdapter(
            UnifiedToolRegistry.get_instance()
        )
```

## Success Criteria

1. No performance regression
2. All existing tests pass
3. Single registry for all tool operations
4. Semantic selection available by default
5. Plugin tools supported
6. Deprecation handling for legacy tools
7. Migration path documented
