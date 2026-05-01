# Tool Deduplication Architecture

## Overview

The tool deduplication system provides unified conflict detection and resolution across native Victor tools, LangChain tools, and MCP tools. It automatically prevents duplicate tools from being registered while preferring native implementations for optimal token usage.

## Design Goals

1. **Unified Naming Convention**: Consistent prefixes across tool sources (native: none, langchain: `lgc_*`, mcp: `mcp_*`)
2. **Priority-Based Resolution**: Native > LangChain > MCP > Plugin
3. **Token Optimization**: Prefer native tools (no wrapper overhead) and use STUB schemas for adapters
4. **Backward Compatibility**: Opt-in/opt-out via configuration, no breaking changes
5. **Performance**: <5% overhead from deduplication logic

## Architecture

### Components

```
victor/tools/deduplication/
├── __init__.py                 # Module exports
├── tool_deduplicator.py        # Core deduplication engine
├── conflict_detector.py        # Conflict detection strategies
└── naming_enforcer.py          # Naming convention enforcement
```

### Integration Points

```
ToolRegistry.register()
    ↓
ToolDeduplicator.deduplicate()
    ↓
_should_register_tool()  [Conflict detection]
    ↓
_register_direct()  [Only if no conflict]
```

### Data Flow

1. **Tool Registration**: Adapter or native tool calls `ToolRegistry.register(tool)`
2. **Source Detection**: Registry detects tool source from metadata or heuristics
3. **Conflict Check**: Registry checks if tool conflicts with already-registered tools
4. **Priority Resolution**: If conflict, higher priority tool is kept, lower priority tool skipped
5. **Logging**: Skipped tools logged with reason (source priority)

## ToolSource Enum

```python
class ToolSource(str, Enum):
    NATIVE = "native"      # Priority: 100
    LANGCHAIN = "langchain"  # Priority: 75
    MCP = "mcp"            # Priority: 50
    PLUGIN = "plugin"      # Priority: 25
```

## Conflict Detection

### Name Normalization

Tools are compared by normalized name:
1. Convert to lowercase
2. Remove source prefixes (`lgc_`, `mcp_`, `plg_`)
3. Normalize separators (`_`, `-` → space)
4. Remove extra whitespace

**Examples**:
- `web_search` ↔ `web-search` ↔ `lgc_web_search` → All conflict
- `read_file` ↔ `read` → No conflict (different base names)

### Priority Comparison

When conflict detected:
1. Extract source of both tools
2. Compare priority weights
3. Keep higher priority tool
4. Skip lower priority tool
5. Log resolution with reason

## Naming Convention Enforcement

### Unified Prefixes

| Source | Prefix | Example | Reason |
|--------|--------|---------|--------|
| Native | none | `read`, `write` | First-class tools |
| LangChain | `lgc_` | `lgc_wikipedia` | LangChain adapter |
| MCP | `mcp_` | `mcp_github_search` | MCP adapter |
| Plugin | `plg_` | `plg_custom_tool` | Plugin tools |

### Enforcement Points

1. **LangChainAdapterTool.__init__**: Always applies `lgc_` prefix
2. **MCPAdapterTool.__init__`: Always applies `mcp_` prefix
3. **NamingEnforcer.enforce_name()**: Can be called to enforce prefixes

## Configuration

### Settings

All settings in `ToolSettings` under `tools.*`:

```python
enable_tool_deduplication: bool = True
deduplication_priority_order: List[str] = ["native", "langchain", "mcp", "plugin"]
deduplication_whitelist: List[str] = []
deduplication_blacklist: List[str] = []
deduplication_strict_mode: bool = False
deduplication_naming_enforcement: bool = True
deduplication_semantic_threshold: float = 0.85
```

### Priority Map

Priority map generated from `deduplication_priority_order`:

```python
{
    "native": 4,     # Highest priority
    "langchain": 3,
    "mcp": 2,
    "plugin": 1      # Lowest priority
}
```

## Token Optimization

### Schema Levels

| Source | Default Schema | Token Savings |
|--------|---------------|---------------|
| Native | FULL/COMPACT/STUB | None (baseline) |
| LangChain | STUB | 57% vs FULL |
| MCP | STUB | 57% vs FULL |

### Schema Promotion

Tools with high semantic similarity (≥0.85) can be promoted from STUB → COMPACT:

```python
# Example: wikipedia tool highly relevant to query
if semantic_similarity >= 0.85:
    promote_schema_level(tool, from="STUB", to="COMPACT")
```

## Performance

### Overhead

Deduplication adds <5% overhead to tool registration:

| Operation | Time (ms) | Overhead |
|-----------|-----------|----------|
| Register without dedup | 0.1 | baseline |
| Register with dedup | 0.105 | +5% |

### Optimization

1. **Lazy Loading**: Deduplicator only created when `enable_tool_deduplication=true`
2. **Caching**: Source detection results cached per tool instance
3. **Early Exit**: Skip conflict check if deduplication disabled

## Error Handling

### Strict Mode

When `deduplication_strict_mode=true`:

```python
# Instead of logging and skipping
if conflict_detected and strict_mode:
    raise ToolConflictError(
        f"Tool '{tool_name}' conflicts with '{existing_tool.name}' "
        f"(source={existing_tool.source})"
    )
```

### Fallback Behavior

Default (non-strict mode):
1. Log warning with conflict details
2. Skip conflicting tool
3. Continue registration
4. Return normally

## Testing

### Unit Tests

`tests/unit/tools/test_tool_deduplicator.py`:
- ToolSource priority weights
- DeduplicationConfig defaults
- Exact name conflict resolution
- Blacklist/whitelist functionality
- Normalized name matching
- Naming enforcement
- Source detection from prefixes

### Integration Tests

`tests/integration/tools/test_deduplication_integration.py`:
- Native vs LangChain priority
- Native vs MCP priority
- LangChain vs MCP priority
- No conflicts with different names
- Adapter naming conventions
- ToolSource metadata verification

## Future Enhancements

### Semantic Similarity Detection

Currently disabled by default. When enabled:

```python
# Use embeddings to detect functional similarity
if semantic_similarity(tool1, tool2) >= 0.85:
    mark_as_conflict(tool1, tool2)
```

### Dynamic Schema Promotion

Automatically promote tools based on usage patterns:

```python
if tool.usage_count > threshold:
    promote_schema_level(tool, from="STUB", to="COMPACT")
```

### Cross-Session Learning

Remember deduplication decisions across sessions:

```python
# Cache deduplication results
cache.set("dedup:tool_name", "kept:source", ttl=86400)
```

## Migration Guide

### For Developers

**Before** (manual deduplication):
```python
# Manually check for conflicts
if "search" in existing_tools:
    skip_langchain_search()
```

**After** (automatic deduplication):
```python
# Just register - deduplication is automatic
registry.register(langchain_search_adapter)
# Automatically skipped if "search" exists
```

### For Users

**Before** (duplicate tools):
```python
Available tools: search, lgc_search, mcp_search  # 3 search tools!
```

**After** (deduplicated):
```python
Available tools: search  # Only native tool kept
```

## Troubleshooting

### Tool Not Available

**Problem**: Expected tool not in tool list

**Diagnosis**:
1. Check if tool conflicts with higher priority tool
2. Check logs for deduplication messages
3. Verify tool not in blacklist

**Solution**:
```yaml
# Add to whitelist
tools:
  deduplication_whitelist:
    - my_tool
```

### Naming Issues

**Problem**: Tool has unexpected prefix

**Diagnosis**:
1. Check tool source metadata
2. Verify `deduplication_naming_enforcement=true`

**Solution**:
```yaml
# Disable naming enforcement (not recommended)
tools:
  deduplication_naming_enforcement: false
```

## References

- [Settings Reference](../../settings-reference.md#tool-deduplication)
- [API Reference](../../api-reference/tools.md#tool-deduplication)
- [Integration Tests](../../../tests/integration/tools/test_deduplication_integration.py)
