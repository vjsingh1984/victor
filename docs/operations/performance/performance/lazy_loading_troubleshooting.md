# Lazy Loading Troubleshooting Guide

This guide helps diagnose and resolve common issues with lazy loading in Victor.

## Quick Diagnosis

### Verify Lazy Loading is Enabled

```bash
# Check environment variables
echo "VICTOR_LAZY_EXTENSIONS: $VICTOR_LAZY_EXTENSIONS"
echo "VICTOR_VERTICAL_LOADING_MODE: $VICTOR_VERTICAL_LOADING_MODE"

# Run with debug logging
VICTOR_LOG_LEVEL=DEBUG victor chat --no-tui 2>&1 | grep -i "lazy"
```

Expected output when enabled:
```
DEBUG:victor.core.verticals.lazy_extensions:Lazy loading extensions for vertical: coding
DEBUG:victor.core.verticals.lazy_extensions:Successfully loaded extensions for vertical: coding
```

### Check Current Mode

```python
from victor.config.settings import get_settings

settings = get_settings()
print(f"Vertical loading mode: {settings.vertical_loading_mode}")
```

## Common Issues

### Issue 1: No Startup Time Improvement

**Symptoms**:
- Startup time is the same with and without lazy loading
- Benchmark shows <5% improvement

**Diagnosis**:
```bash
# Run benchmark
python scripts/benchmark_lazy_loading.py --output text

# Check if extensions are actually loaded lazily
VICTOR_LOG_LEVEL=DEBUG python -c "
from victor.coding import CodingAssistant
ext = CodingAssistant.get_extensions()
print(f'Type: {type(ext).__name__}')
print(f'Loaded: {ext.is_loaded() if hasattr(ext, \"is_loaded\") else \"N/A\"}')
"
```

**Possible Causes**:

1. **Extensions are already lightweight**
   - Most extensions are simple classes with fast initialization
   - Solution: Lazy loading has minimal impact, this is expected

2. **Configuration loading dominates startup**
   - YAML parsing, registry initialization take time
   - Solution: Profile to identify actual bottleneck

3. **Lazy loading not actually enabled**
   - Environment variable not set
   - Code bypasses lazy loading with `use_lazy=False`
   - Solution: Verify configuration (see above)

**Solution**:
```bash
# Ensure lazy loading is enabled
export VICTOR_LAZY_EXTENSIONS=true
export VICTOR_VERTICAL_LOADING_MODE=lazy

# Verify in code
python -c "
import os
print(f'VICTOR_LAZY_EXTENSIONS: {os.getenv(\"VICTOR_LAZY_EXTENSIONS\")}')
print(f'VICTOR_VERTICAL_LOADING_MODE: {os.getenv(\"VICTOR_VERTICAL_LOADING_MODE\")}')
"
```

### Issue 2: First Access Too Slow

**Symptoms**:
- First call to `extensions.middleware` takes >500ms
- User-visible latency when first using a vertical

**Diagnosis**:
```python
import time
from victor.coding import CodingAssistant

# Time first access
start = time.time()
extensions = CodingAssistant.get_extensions()
first_access = time.time() - start

# Time middleware access
start = time.time()
middleware = extensions.middleware
middleware_access = time.time() - start

print(f"Get extensions: {first_access * 1000:.2f}ms")
print(f"Access middleware: {middleware_access * 1000:.2f}ms")
```

**Possible Causes**:

1. **Heavy imports in extension modules**
   ```python
   # BAD: Heavy import at module level
   from transformers import pipeline  # 500MB+
   
   # GOOD: Lazy import
   def get_model(self):
       if not hasattr(self, '_model'):
           from transformers import pipeline
           self._model = pipeline("translation")
       return self._model
   ```

2. **Complex initialization logic**
   - Extension `__init__` does heavy computation
   - Solution: Move to lazy initialization

3. **Circular dependencies**
   - Extensions import each other
   - Solution: Use lazy imports or dependency injection

**Solution**:
```python
# Profile to find bottleneck
import cProfile
import pstats

def test_access():
    from victor.coding import CodingAssistant
    extensions = CodingAssistant.get_extensions()
    _ = extensions.middleware

cProfile.run('test_access()', 'profile_stats')
pstats.Stats('profile_stats').sort_stats('cumulative').print_stats(20)
```

### Issue 3: Recursive Loading Detected

**Symptoms**:
```
RuntimeError: Recursive loading detected for vertical 'coding'
RuntimeError: Recursive loading detected for extensions of vertical 'coding'
```

**Diagnosis**:
```python
# Check for circular dependencies
import sys

def check_circular_imports(module_name):
    if module_name in sys.modules:
        module = sys.modules[module_name]
        print(f"Module: {module_name}")
        print(f"File: {module.__file__}")
        # Print imports
        for attr in dir(module):
            obj = getattr(module, attr)
            if hasattr(obj, '__module__'):
                print(f"  {attr}: {obj.__module__}")

check_circular_imports('victor.coding.safety')
check_circular_imports('victor.coding.prompts')
```

**Possible Causes**:

1. **`__init_subclass__` calls `get_extensions()`**
   ```python
   # BAD
   class MyVertical(VerticalBase):
       def __init_subclass__(cls, **kwargs):
           super().__init_subclass__(**kwargs)
           extensions = cls.get_extensions()  # Recursive!
   
   # GOOD
   class MyVertical(VerticalBase):
       def __init_subclass__(cls, **kwargs):
           super().__init_subclass__(**kwargs)
           # Don't call get_extensions() here
   ```

2. **Extension imports its own vertical**
   ```python
   # BAD: Circular import
   # victor/coding/safety.py
   from victor.coding import CodingAssistant  # Imports safety.py
   
   # GOOD: Use lazy import
   def get_vertical():
       from victor.coding import CodingAssistant
       return CodingAssistant
   ```

**Solution**:
- Use lazy imports within methods
- Avoid circular dependencies between vertical and extensions
- Use dependency injection instead of direct imports

### Issue 4: Memory Usage Not Reduced

**Symptoms**:
- Memory usage is the same in eager and lazy modes
- Lazy loading doesn't reduce memory footprint

**Diagnosis**:
```bash
# Measure memory after startup
python -c "
import psutil
import os

# Eager mode
os.environ['VICTOR_LAZY_EXTENSIONS'] = 'false'
from victor.coding import CodingAssistant
_ = CodingAssistant.get_extensions()
eager_mem = psutil.Process().memory_info().rss / 1024 / 1024

# Lazy mode (in separate process)
lazy_mem = float(os.popen('''
    python -c "
import os
os.environ['VICTOR_LAZY_EXTENSIONS'] = 'true'
import psutil
from victor.coding import CodingAssistant
_ = CodingAssistant.get_extensions()
print(psutil.Process().memory_info().rss / 1024 / 1024)
''').read())

print(f'Eager: {eager_mem:.2f}MB')
print(f'Lazy: {lazy_mem:.2f}MB')
print(f'Difference: {eager_mem - lazy_mem:.2f}MB')
"
```

**Possible Causes**:

1. **Measuring memory after first access**
   - Lazy loading defers, doesn't eliminate loading
   - Memory usage should be measured before first access
   - Solution: Measure immediately after startup, before using extensions

2. **Other memory-intensive operations**
   - Embedding models, registries, caches dominate memory
   - Solution: Profile memory usage to find actual consumers

3. **Lazy loading not actually enabled**
   - Environment variable not set
   - Solution: Verify configuration (see Issue 1)

**Solution**:
```bash
# Measure memory before any extension access
python -c "
import psutil
import os
import sys

# Force lazy mode
os.environ['VICTOR_LAZY_EXTENSIONS'] = 'true'
os.environ['VICTOR_VERTICAL_LOADING_MODE'] = 'lazy'

# Import but don't use extensions
from victor.coding import CodingAssistant
extensions = CodingAssistant.get_extensions()  # Returns LazyVerticalExtensions

# Measure before first access
mem_before = psutil.Process().memory_info().rss / 1024 / 1024

# Now access extensions (triggers loading)
_ = extensions.middleware
mem_after = psutil.Process().memory_info().rss / 1024 / 1024

print(f'Memory before first access: {mem_before:.2f}MB')
print(f'Memory after first access: {mem_after:.2f}MB')
print(f'Difference: {mem_after - mem_before:.2f}MB')
"
```

### Issue 5: Extensions Not Loading At All

**Symptoms**:
- `extensions.middleware` returns empty list
- `extensions.safety_extensions` returns empty list
- Extensions are `None` or missing

**Diagnosis**:
```python
from victor.coding import CodingAssistant

extensions = CodingAssistant.get_extensions()

# Check type
print(f"Type: {type(extensions)}")
print(f"Module: {type(extensions).__module__}")

# Check if loaded
if hasattr(extensions, 'is_loaded'):
    print(f"Loaded: {extensions.is_loaded()}")

# Check attributes
print(f"Middleware: {extensions.middleware}")
print(f"Safety: {extensions.safety_extensions}")

# Check logs
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Possible Causes**:

1. **Extension modules don't exist**
   - Vertical doesn't have `safety.py`, `prompts.py`, etc.
   - Solution: Create missing modules or handle gracefully

2. **Import errors in extension modules**
   - Syntax errors, missing dependencies
   - Solution: Check logs for ImportError

3. **Class naming mismatch**
   - Extension class doesn't match expected name
   - Solution: Use exact naming or override getter

**Solution**:
```python
# Test extension loading directly
try:
    from victor.coding.safety import CodingSafetyExtension
    print("✓ Safety extension imports successfully")
    ext = CodingSafetyExtension()
    print(f"✓ Safety extension instantiates: {ext}")
except ImportError as e:
    print(f"✗ Import error: {e}")
except Exception as e:
    print(f"✗ Instantiation error: {e}")

# Check expected class name
from victor.core.verticals.extension_loader import VerticalExtensionLoader

# Auto-generated class name format:
# {VerticalName}{ExtensionType}
# Example: CodingSafetyExtension, ResearchPromptContributor
```

## Debugging Tools

### Enable Debug Logging

```bash
# Enable all debug logs
export VICTOR_LOG_LEVEL=DEBUG

# Filter for lazy loading logs
victor chat --no-tui 2>&1 | grep -i "lazy"
```

### Profile Extension Loading

```python
import cProfile
import pstats

def profile_extensions():
    from victor.coding import CodingAssistant
    
    # Profile get_extensions
    extensions = CodingAssistant.get_extensions()
    
    # Profile first access
    _ = extensions.middleware
    _ = extensions.safety_extensions

cProfile.run('profile_extensions()', 'profile_stats')
p = pstats.Stats('profile_stats')
p.sort_stats('cumulative').print_stats(20)
```

### Memory Profiling

```bash
# Install memory_profiler
pip install memory_profiler

# Profile memory usage
python -m memory_profiler -m victor.ui.cli chat --no-tui
```

### Check Extension Cache

```python
from victor.core.verticals.extension_loader import VerticalExtensionLoader

# Get cache stats
stats = CodingAssistant.get_extension_cache_stats(detailed=True)
print(f"Total entries: {stats['total_entries']}")
print(f"Expired: {stats['expired_entries']}")
print(f"Hit rate: {stats['cache_hit_rate']:.2%}")

# Print detailed breakdown
for entry in stats.get('entries', []):
    print(f"  {entry['key']}: age={entry['age']:.1f}s, accesses={entry['access_count']}")
```

## Performance Tuning

### Optimize Startup Time

1. **Use lazy loading for CLI tools**
   ```bash
   export VICTOR_LAZY_EXTENSIONS=true
   victor chat --no-tui
   ```

2. **Preload critical extensions**
   ```python
   # Preload middleware (used by everyone)
   from victor.coding import CodingAssistant
   extensions = CodingAssistant.get_extensions(use_lazy=False)
   ```

3. **Disable unused verticals**
   ```bash
   # Only load coding vertical
   victor chat --vertical coding
   ```

### Optimize First Access

1. **Use lazy imports in extensions**
   ```python
   class MyExtension:
       def __init__(self):
           pass  # Don't import here
       
       def get_heavy_dependency(self):
           if not hasattr(self, '_heavy'):
               import heavy_module
               self._heavy = heavy_module.HeavyClass()
           return self._heavy
   ```

2. **Cache expensive operations**
   ```python
   class MyExtension:
       def __init__(self):
           self._cache = {}
       
       def expensive_operation(self, arg):
           if arg not in self._cache:
               result = self._do_expensive_work(arg)
               self._cache[arg] = result
           return self._cache[arg]
   ```

3. **Use lightweight initialization**
   ```python
   # BAD: Load model in __init__
   class MyExtension:
       def __init__(self):
           self.model = load_big_model()  # Slow!
   
   # GOOD: Load on first use
   class MyExtension:
       def __init__(self):
           self._model = None
   
       @property
       def model(self):
           if self._model is None:
               self._model = load_big_model()
           return self._model
   ```

## Getting Help

If you're still experiencing issues:

1. **Check logs**: Enable debug logging and look for error messages
2. **Run benchmarks**: `python scripts/benchmark_lazy_loading.py`
3. **Search issues**: Check GitHub for similar problems
4. **Create minimal reproduction**: Simplify your code to isolate the issue
5. **File a bug**: Include debug logs, benchmark results, and minimal example

## Additional Resources

- **Architecture**: `docs/performance/lazy_loading.md`
- **Implementation**: `victor/core/verticals/lazy_extensions.py`
- **Benchmarks**: `scripts/benchmark_lazy_loading.py`
- **Tests**: `tests/unit/verticals/test_extension_loader.py`
