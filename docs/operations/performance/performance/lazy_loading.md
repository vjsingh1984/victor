# Lazy Loading Implementation for Vertical Extensions

## Overview

Victor implements lazy loading for vertical extensions to improve startup time and reduce initial memory footprint. This
  document describes the lazy loading architecture,
  configuration options, and performance impact.

## Architecture

### Key Components

1. **LazyVerticalExtensions** (`victor/core/verticals/lazy_extensions.py`)
   - Thread-safe wrapper for `VerticalExtensions`
   - Defers loading until first access
   - Uses double-checked locking pattern

2. **LazyVerticalProxy** (`victor/core/verticals/lazy_loader.py`)
   - Proxy for lazy-loaded vertical classes
   - Transparent attribute forwarding
   - Thread-safe lazy initialization

3. **ExtensionLoader Integration** (`victor/core/verticals/extension_loader.py`)
   - `get_extensions()` method supports lazy loading
   - Returns `LazyVerticalExtensions` wrapper when enabled
   - Falls back to eager loading for backward compatibility

4. **VerticalLoader Integration** (`victor/core/verticals/vertical_loader.py`)
   - `load()` method supports lazy parameter
   - Returns `LazyVerticalProxy` when lazy=True
   - Configurable via `vertical_loading_mode` setting

## Configuration

### Environment Variables

#### VICTOR_LAZY_EXTENSIONS

Controls when vertical extensions are loaded:

- **true** (default): Load extensions lazily (on first access)
- **false**: Load extensions eagerly (at startup)
- **auto**: Automatically choose based on profile (production=lazy, dev=eager)

```bash
# Enable lazy loading (default)
export VICTOR_LAZY_EXTENSIONS=true

# Disable lazy loading (eager mode)
export VICTOR_LAZY_EXTENSIONS=false

# Auto mode (profile-based)
export VICTOR_LAZY_EXTENSIONS=auto
```

#### VICTOR_VERTICAL_LOADING_MODE

Controls when vertical classes are loaded:

- **eager** (default): Load verticals immediately at startup
- **lazy**: Load vertical metadata only, defer heavy imports
- **auto**: Automatically choose based on environment

```bash
# Enable lazy vertical loading
export VICTOR_VERTICAL_LOADING_MODE=lazy

# Disable (eager mode)
export VICTOR_VERTICAL_LOADING_MODE=eager
```

### Python Settings

```python
from victor.config.settings import Settings

settings = Settings()

# Check current mode
print(settings.vertical_loading_mode)  # "eager", "lazy", or "auto"
```

## Performance Impact

### Benchmark Results

Based on benchmark_lazy_loading.py (2026-01-20):

| Metric | Eager Mode | Lazy Mode | Improvement |
|--------|-----------|-----------|-------------|
| Startup Time | 678.44ms | 658.33ms | 3.0% (20ms saved) |
| Memory (startup) | ~150MB | ~145MB | 3.3% reduction |
| First Access | <1ms | ~265ms avg | One-time overhead |

**Note**: The current implementation shows modest improvement (3%) because:
1. Most extensions are already lightweight
2. YAML configuration loading dominates startup time
3. Only 5 verticals tested (coding, research, devops, dataanalysis, benchmark)

Expected improvement scales with:
- Number of verticals (more verticals = more benefit)
- Extension complexity (heavy extensions = more benefit)
- Use case (CLI tools benefit more than long-running servers)

### When to Use Lazy Loading

**Enable lazy loading when:**
- Building CLI tools or scripts with short runtime
- Running in resource-constrained environments
- Many verticals are installed but few are used
- Startup time is critical

**Disable lazy loading (use eager) when:**
- Running long-running server processes
- All verticals will be used immediately
- First-access latency is unacceptable
- Debugging extension loading issues

## Implementation Details

### Thread Safety

Lazy loading uses the double-checked locking pattern:

```python
def load(self):
    # Fast path: no lock
    if self._loaded:
        return self._instance
    
    # Slow path: with lock
    with self._load_lock:
        # Double-check
        if self._loaded:
            return self._instance
        
        # Load
        self._instance = self.loader()
        self._loaded = True
        return self._instance
```

This ensures:
1. Thread-safe lazy initialization
2. Minimal lock contention (fast path is lock-free)
3. No recursive loading

### LazyVerticalExtensions Protocol

The lazy wrapper implements the full `VerticalExtensions` protocol by proxying:

```python
class LazyVerticalExtensions:
    @property
    def middleware(self):
        extensions = self._load_extensions()  # Triggers loading
        return extensions.middleware
    
    @property
    def safety_extensions(self):
        extensions = self._load_extensions()  # Uses cached value
        return extensions.safety_extensions
```

All extension properties trigger loading on first access and cache the result.

### Integration Points

**1. VerticalExtensionLoader.get_extensions()**

```python
@classmethod
def get_extensions(cls, *, use_cache: bool = True, use_lazy: Optional[bool] = None):
    # Determine if lazy loading should be used
    if use_lazy is None:
        trigger = get_extension_load_trigger()
        use_lazy = trigger != "eager"
    
    # Return lazy wrapper if enabled
    if use_lazy:
        return create_lazy_extensions(
            vertical_name=cls.name,
            loader=lambda: cls._load_extensions_eager(use_cache=use_cache),
            trigger=trigger,
        )
    
    # Eager loading path
    return cls._load_extensions_eager(use_cache=use_cache)
```

**2. VerticalLoader.load()**

```python
def load(self, name: str, lazy: Optional[bool] = None):
    # Determine if lazy loading should be used
    use_lazy = lazy if lazy is not None else self._lazy_mode
    
    vertical = VerticalRegistry.get(name)
    
    # Return lazy proxy if enabled
    if use_lazy:
        def _load_vertical():
            self._activate(vertical)
            return vertical
        
        proxy = LazyVerticalProxy(vertical_name=name, loader=_load_vertical)
        return proxy
    
    # Eager mode: activate immediately
    self._activate(vertical)
    return vertical
```

## Testing

### Running Benchmarks

```bash
# Run full benchmark suite
python scripts/benchmark_lazy_loading.py

# Test specific verticals
python scripts/benchmark_lazy_loading.py --verticals coding,research

# Generate markdown report
python scripts/benchmark_lazy_loading.py --output markdown

# Multiple iterations for accuracy
python scripts/benchmark_lazy_loading.py --iterations 5
```

### Unit Tests

```bash
# Test lazy loading functionality
pytest tests/unit/test_lazy_loading.py -v

# Test extension loading
pytest tests/unit/verticals/test_extension_loader.py -v

# Test vertical loading
pytest tests/unit/verticals/test_vertical_loader.py -v
```

## Troubleshooting

### Issue: Extensions not loading lazily

**Symptoms**: Startup time doesn't improve with `VICTOR_LAZY_EXTENSIONS=true`

**Solutions**:
1. Verify environment variable is set: `echo $VICTOR_LAZY_EXTENSIONS`
2. Check if calling code bypasses lazy loading: `get_extensions(use_lazy=False)`
3. Ensure extensions are actually being loaded (check logs for "Lazy loading extensions")

### Issue: First access too slow

**Symptoms**: First extension access takes >500ms

**Solutions**:
1. Check if heavy imports in extension modules
2. Consider preloading critical extensions: `get_extensions(use_lazy=False)`
3. Profile to identify bottlenecks: `python -m cProfile -o profile.stats script.py`

### Issue: Recursive loading detected

**Symptoms**: `RuntimeError: Recursive loading detected`

**Solutions**:
1. Check for circular dependencies in extension imports
2. Ensure `__init_subclass__` doesn't call `get_extensions()`
3. Use lazy imports within extension modules: `import module` inside methods

### Issue: Memory not reduced

**Symptoms**: Memory usage similar between eager and lazy modes

**Solutions**:
1. Verify lazy loading is actually enabled (check logs)
2. Measure memory after first access (lazy loading defers, doesn't eliminate)
3. Check for other memory-intensive operations during startup

## Best Practices

### 1. For Vertical Developers

- **Keep extensions lightweight**: Avoid heavy imports at module level
- **Use lazy imports**: Import heavy modules inside methods, not at top level
- **Document dependencies**: Clearly document which extensions require which resources

Example:
```python
# BAD: Heavy import at module level
from transformers import pipeline  # 500MB+

class MyExtension:
    def __init__(self):
        self.model = pipeline("translation")

# GOOD: Lazy import
class MyExtension:
    def __init__(self):
        pass
    
    def get_model(self):
        if not hasattr(self, '_model'):
            from transformers import pipeline
            self._model = pipeline("translation")
        return self._model
```

### 2. For Application Developers

- **Profile before optimizing**: Measure to find actual bottlenecks
- **Consider use case**: CLI tools benefit more than servers
- **Test both modes**: Verify functionality works in both eager and lazy modes

### 3. For DevOps Engineers

- **Use eager mode in production servers**: All verticals used anyway
- **Use lazy mode in CLI tools**: Faster startup for short-running tasks
- **Monitor first-access latency**: Ensure acceptable user experience

## Future Improvements

### Planned Enhancements

1. **Selective lazy loading**: Load only specific extensions lazily
   ```python
   # Future API
   get_extensions(lazy=["safety", "workflow"], eager=["middleware"])
   ```

2. **Async preloading**: Start loading in background, return when ready
   ```python
   # Future API
   extensions = await get_extensions_async()
   ```

3. **Dependency tracking**: Automatically load dependent extensions
   ```python
   # Future: Automatically load workflow if provider loaded
   ```

4. **Metrics integration**: Track lazy loading effectiveness
   ```python
   # Future API
   stats = get_lazy_loading_stats()
   print(f"Extensions loaded: {stats.loaded_count}/{stats.total_count}")
   ```

## References

- **Implementation**: `victor/core/verticals/lazy_extensions.py`
- **Vertical Loader**: `victor/core/verticals/vertical_loader.py`
- **Extension Loader**: `victor/core/verticals/extension_loader.py`
- **Benchmarks**: `scripts/benchmark_lazy_loading.py`
- **Related**: Track 6 - Lazy Loading Implementation for Extensions

---

**Last Updated:** February 01, 2026
**Reading Time:** 4 minutes
