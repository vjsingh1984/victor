# Lazy Loading for Verticals

## Overview

Victor implements lazy loading for vertical imports to significantly improve startup time. Instead of eagerly importing all vertical assistant classes at startup, Victor defers the import until the vertical is first accessed.

## Performance Impact

Lazy loading provides **90.3% faster startup time**:

| Metric | Eager Loading | Lazy Loading | Improvement |
|--------|--------------|--------------|-------------|
| Core Import | 1805ms | 176ms | **90.3% faster** |
| First Access (cached) | 0ms | 30ms | One-time cost |
| Total Startup | 1805ms | 205ms | **88.6% faster** |

### Real-World Scenarios

**Scenario 1: Use only one vertical (e.g., coding)**
- Lazy Loading: 163ms
- Eager Loading: 333ms
- **Improvement: 51.1% faster**

**Scenario 2: List verticals only (no access)**
- Lazy Loading: 133ms
- Eager Loading: 343ms
- **Improvement: 61.2% faster**

## How It Works

### Registration

Verticals are registered as lazy imports using the `VerticalRegistry.register_lazy_import()` method:

```python
# victor/core/verticals/__init__.py

def _register_builtin_verticals() -> None:
    """Register all built-in verticals with lazy loading."""
    lazy_loading_enabled = os.getenv("VICTOR_LAZY_LOADING", "true").lower() == "true"

    if lazy_loading_enabled:
        # Register lazy imports - loaded on first access
        VerticalRegistry.register_lazy_import(
            "coding",
            "victor.coding:CodingAssistant"
        )
        VerticalRegistry.register_lazy_import(
            "research",
            "victor.research:ResearchAssistant"
        )
        # ... more verticals
    else:
        # Eager loading - legacy behavior
        from victor.coding import CodingAssistant
        VerticalRegistry.register(CodingAssistant)
```

### Access Pattern

When you access a vertical, it's automatically loaded:

```python
from victor.core.verticals import VerticalRegistry

# First access triggers lazy import
coding = VerticalRegistry.get("coding")  # Loads victor.coding here

# Subsequent accesses use cached class
coding2 = VerticalRegistry.get("coding")  # Returns cached instance

# Use the vertical normally
config = coding.get_config()
tools = coding.get_tools()
```

## Configuration

### Enable/Disable Lazy Loading

Lazy loading is **enabled by default**. You can control it via environment variable:

```bash
# Enable lazy loading (default)
export VICTOR_LAZY_LOADING=true

# Disable lazy loading (eager loading)
export VICTOR_LAZY_LOADING=false
```

### When to Use Eager Loading

Consider disabling lazy loading (eager loading) when:
- You need to access all verticals immediately (e.g., batch processing)
- You want to avoid the one-time lazy load cost during critical operations
- You're debugging vertical import issues

## API Reference

### VerticalRegistry.register_lazy_import()

Register a vertical for lazy loading.

```python
VerticalRegistry.register_lazy_import(
    name: str,           # Vertical name (e.g., "coding")
    import_path: str     # Import path "module:ClassName"
)
```

**Example:**
```python
VerticalRegistry.register_lazy_import(
    "my_vertical",
    "my_package:MyVerticalAssistant"
)
```

### VerticalRegistry.get()

Get a vertical by name, triggering lazy load if needed.

```python
vertical = VerticalRegistry.get(name: str) -> Optional[Type[VerticalBase]]
```

**Returns:**
- `VerticalBase` subclass if found
- `None` if not found or import fails

### VerticalRegistry.list_names()

List all registered vertical names (including lazy imports).

```python
names = VerticalRegistry.list_names() -> List[str]
```

**Note:** This does **not** trigger lazy loading. It returns both registered and lazy-imported vertical names.

## Implementation Details

### Thread Safety

Lazy loading is thread-safe with double-checked locking:

1. Check if already loaded (fast path)
2. Acquire lock
3. Check again (another thread might have loaded it)
4. Load if still not loaded

Multiple threads can safely access the same vertical simultaneously.

### Error Handling

Failed lazy imports are automatically removed from the registry:

```python
# Invalid import path
VerticalRegistry.register_lazy_import(
    "invalid",
    "nonexistent.module:InvalidClass"
)

# First access fails and removes the entry
result = VerticalRegistry.get("invalid")  # Returns None
assert "invalid" not in VerticalRegistry._lazy_imports  # Removed
```

### Name Normalization

Vertical names are normalized using `normalize_vertical_name()`:

- "dataanalysis" → "data_analysis"
- "myVertical" → "my_vertical"
- "My_Vertical" → "my_vertical"

This ensures consistent naming regardless of how the vertical is registered or accessed.

## Best Practices

### 1. Use Lazy Loading by Default

```python
# GOOD - Lazy loading enabled
from victor.core.verticals import VerticalRegistry

coding = VerticalRegistry.get("coding")
config = coding.get_config()
```

### 2. Access Verticals Once and Reuse

```python
# GOOD - Access once, reuse
coding = VerticalRegistry.get("coding")
tools1 = coding.get_tools()
tools2 = coding.get_tools()  # Uses same class

# AVOID - Repeated access (slight overhead)
tools1 = VerticalRegistry.get("coding").get_tools()
tools2 = VerticalRegistry.get("coding").get_tools()
```

### 3. Preload Critical Verticals (if needed)

```python
# If you need eager loading for specific verticals
from victor.coding import CodingAssistant
from victor.research import ResearchAssistant

# These are now cached and won't be lazy-loaded later
```

## Migration Guide

### For Victor Users

**No changes required!** Lazy loading is transparent and backward compatible.

Existing code continues to work:

```python
# This works exactly the same with lazy loading
from victor.coding import CodingAssistant

config = CodingAssistant.get_config()
tools = CodingAssistant.get_tools()
```

### For Vertical Developers

If you're developing a custom vertical, register it for lazy loading:

```python
# In your vertical's __init__.py or registration module
from victor.core.verticals import VerticalRegistry

VerticalRegistry.register_lazy_import(
    "my_vertical",
    "my_package:MyVerticalAssistant"
)
```

Or use entry points in `pyproject.toml`:

```toml
[project.entry-points."victor.verticals"]
my_vertical = "my_package:MyVerticalAssistant"
```

## Benchmarks

Run the benchmark suite to measure performance on your system:

```bash
# Compare eager vs lazy loading
python benchmarks/benchmark_lazy_loading.py

# Detailed startup time analysis
python benchmarks/benchmark_startup.py
```

### Benchmark Results (Reference System)

**System:** macOS, Python 3.12, M-series CPU

| Benchmark | Eager | Lazy | Improvement |
|-----------|-------|------|-------------|
| Core Import | 1805ms | 176ms | 90.3% |
| Use 1 Vertical | 333ms | 163ms | 51.1% |
| List Verticals | 343ms | 133ms | 61.2% |

## Troubleshooting

### Issue: Vertical Not Found

**Symptom:** `VerticalRegistry.get("my_vertical")` returns `None`

**Solution:** Check that the vertical is registered:
```python
from victor.core.verticals import VerticalRegistry

print(VerticalRegistry.list_names())
print(VerticalRegistry._lazy_imports)
```

### Issue: Import Error on Access

**Symptom:** First access to a vertical raises `ImportError`

**Solution:** Verify the import path format:
```python
# Correct format: "module.path:ClassName"
VerticalRegistry.register_lazy_import(
    "coding",
    "victor.coding:CodingAssistant"  # ✅ Correct
)

# Wrong formats:
"victor.coding.CodingAssistant"  # ❌ Missing colon
"victor.coding:CodingAssistant:Extra"  # ❌ Too many parts
```

### Issue: Performance Degradation

**Symptom:** First access to vertical is slow

**Explanation:** This is expected! The first access triggers the import, which has a one-time cost. Subsequent accesses are fast (cached).

**Solution:** If you need consistent performance, preload critical verticals:
```python
# Preload at startup
from victor.coding import CodingAssistant
from victor.research import ResearchAssistant
```

## Lazy Loading for Vertical Extensions (New Implementation)

### Overview

In addition to lazy import loading, Victor now supports lazy loading for **vertical extensions** to further improve startup time. This feature defers the loading of heavy extension modules (prompts, safety, workflows) until they are actually needed.

### Configuration

The `vertical_loading_mode` setting controls when vertical extensions are loaded:

```bash
# In settings file or environment variable
export VICTOR_VERTICAL_LOADING_MODE=lazy

# Or in .env file
VICTOR_VERTICAL_LOADING_MODE=lazy

# Or in profiles.yaml
vertical_loading_mode: lazy
```

### Loading Modes

1. **eager** (default): Load all extensions immediately at startup
2. **lazy**: Load metadata only, defer heavy modules until first access
3. **auto**: Automatically choose based on environment (production=lazy, dev=eager)

### Usage

```python
from victor.core.verticals.vertical_loader import VerticalLoader, load_vertical

# Configure lazy mode
loader = VerticalLoader()
loader.configure_lazy_mode(settings)

# Load vertical (respects configured mode)
vertical = loader.load("coding")

# Force lazy/eager loading
vertical_lazy = loader.load("coding", lazy=True)
vertical_eager = loader.load("coding", lazy=False)
```

### LazyVerticalProxy API

```python
from victor.core.verticals.lazy_loader import LazyVerticalProxy, LoadTrigger

# Create lazy loader
loader = LazyVerticalLoader(load_trigger=LoadTrigger.ON_DEMAND)
loader.register_vertical("coding", lambda: CodingAssistant)

# Get vertical (loads on first access)
coding = loader.get_vertical("coding")

# Check if loaded
if coding.is_loaded():
    print("Already loaded")

# Force immediate loading
coding.force_load()

# Unload to free memory
loader.unload_vertical("coding")
```

### Thread Safety

All lazy loading operations are thread-safe using double-checked locking. Multiple threads can safely access the same vertical simultaneously - it will only be loaded once.

### Performance

Expected improvements:
- **Startup Time**: 20% reduction (2.5s → 2.0s)
- **First Access Overhead**: ~50ms (acceptable, one-time)
- **Memory**: Similar (lazy loader overhead negligible)

### Best Practices

**Use lazy mode when:**
- Running CLI tools that may not use all verticals
- Running in production with limited startup time budget
- Memory-constrained environments
- Using multiple verticals but not accessing all of them

**Use eager mode when:**
- Debugging vertical loading issues
- Running in development with fast startup not critical
- You need predictable error handling at startup
- All verticals will be accessed immediately anyway

### Implementation

See `victor/core/verticals/lazy_loader.py` for the thread-safe LazyVerticalProxy implementation.

## Future Improvements

Potential enhancements to lazy loading:

1. **Async Loading:** Load verticals asynchronously in the background
2. **Dependency Tracking:** Auto-load dependent verticals
3. **Selective Loading:** Load only required extensions
4. **Hot Reload:** Reload verticals without restart
5. **Metrics:** Track lazy loading statistics

## Related Documentation

- [Vertical Architecture](../architecture/verticals.md)
- [Vertical Registry](../api/vertical_registry.md)
- [Performance Best Practices](../performance/best_practices.md)
- [Extension Loading](../architecture/extensions.md)

## See Also

- `victor/core/verticals/base.py` - VerticalRegistry implementation
- `victor/core/verticals/lazy_loader.py` - Lazy loading utilities
- `tests/unit/verticals/test_lazy_loading.py` - Test suite
