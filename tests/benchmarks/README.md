# Victor Performance Benchmarks

This directory contains performance benchmarks for critical Victor components, following TDD (Test-Driven Development) principles.

## Overview

Benchmarks are created **BEFORE** optimization to establish baseline metrics. This ensures we have concrete data to measure improvement against.

## Tool Selection Benchmarks

`test_tool_selection_performance.py` provides comprehensive benchmarks for tool selection optimization.

### Benchmark Categories

1. **Baseline Performance**: Tool selection with different tool set sizes (10, 47, 100 tools)
2. **Caching Impact**: Cached vs uncached query embedding performance
3. **Batch Embeddings**: Batch embedding generation for multiple items
4. **Category Filtering**: Impact of category pre-filtering on selection time
5. **Cache Warming**: Cold cache vs warm cache performance
6. **Memory Usage**: Memory footprint per tool embedding
7. **Bottleneck Identification**: Identify performance bottlenecks
8. **Regression Tests**: Detect performance degradation over time

## Running Benchmarks

### Run All Benchmarks

```bash
pytest tests/benchmarks/ -v --benchmark-only
```

### Run Specific Benchmark Group

```bash
# Baseline performance
pytest tests/benchmarks/test_tool_selection_performance.py -v --benchmark-only -k "baseline"

# Caching performance
pytest tests/benchmarks/test_tool_selection_performance.py -v --benchmark-only -k "cache"

# Batch embeddings
pytest tests/benchmarks/test_tool_selection_performance.py -v --benchmark-only -k "batch"

# Memory usage
pytest tests/benchmarks/test_tool_selection_performance.py -v --benchmark-only -k "memory"
```

### Generate Comparison Report

```bash
# Run benchmarks and save to file
pytest tests/benchmarks/test_tool_selection_performance.py -v --benchmark-only --benchmark-autosave

# Compare with previous run
pytest-benchmark compare <run1> <run2>
```

### Generate Histogram

```bash
pytest tests/benchmarks/test_tool_selection_performance.py -v --benchmark-only --benchmark-histogram
```

## Performance Targets

### Baseline Performance
- Tool selection (10 tools): <50ms
- Tool selection (47 tools): <100ms
- Tool selection (100 tools): <200ms
- Keyword-only selection: <5ms
- Hybrid selection: <70ms

### Caching Performance
- Cached query lookup: <5ms
- Uncached query embedding: >50ms (baseline)
- Cache hit rate (70%): ~30-40ms

### Batch Embeddings
- Batch (10 items): <100ms
- Batch (47 items): <500ms
- Tool initialization (47): <2s (one-time)

### Memory Usage
- Per tool embedding: ~1.5KB
- 47 tool embeddings: ~70KB
- Cache overhead: <10KB

## Interpreting Results

### pytest-benchmark Output

```
---------------------------------------------------------
Benchmark (time in ms)          Min        Max        Mean
---------------------------------------------------------
test_baseline_10_tools         35.2       48.1       40.3
test_baseline_47_tools         72.4       95.6       82.1
test_cached_query               0.8        2.3        1.2
test_uncached_query            38.5       52.4       44.7
---------------------------------------------------------
```

### Regression Detection

```bash
# Run benchmarks multiple times to collect history
pytest tests/benchmarks/test_tool_selection_performance.py -v --benchmark-only

# Compare with previous run (auto-detects regression)
pytest-benchmark compare --columns=min,max,mean std
```

A test fails if performance degrades by >20% from baseline.

## Benchmark Structure

### Fixtures

- `mock_tool_registry`: Creates mock tools with realistic metadata
- `semantic_selector`: SemanticToolSelector with temp cache
- `keyword_selector`: KeywordToolSelector instance
- `hybrid_selector`: HybridToolSelector combining both approaches

### Benchmark Queries

Predefined queries representing different patterns:
- Simple: "read the file"
- Complex: "find all classes that inherit from BaseController"
- Multi-step: "read config, update url, restart server"
- Vague: "fix it"

### Performance Measurements

Each benchmark uses `benchmark.pedantic()` for accurate measurement:

```python
async def select_tools():
    return await semantic_selector.select_relevant_tools(...)

result = benchmark.pedantic(select_tools, iterations=10, rounds=20)
```

- `iterations`: Number of times to run per round
- `rounds`: Number of rounds to measure
- Result is returned as `(result, stats)` tuple

## Creating New Benchmarks

1. **Add to existing group**: Use `@pytest.mark.benchmark(group="group_name")`
2. **Use fixtures**: Leverage existing fixtures for consistency
3. **Measure**: Use `benchmark.pedantic()` for async functions
4. **Document**: Add docstring explaining what's measured
5. **Target**: Add performance target to this README

Example:

```python
@pytest.mark.benchmark(group="new_feature")
@pytest.mark.asyncio
async def test_new_feature_performance(benchmark, semantic_selector):
    """Benchmark new feature X.

    Target: <100ms for 10 items
    """

    async def new_feature():
        return await semantic_selector.new_feature(...)

    result = benchmark.pedantic(new_feature, iterations=10, rounds=20)
    assert result is not None
```

## Continuous Integration

Benchmarks run in CI to detect performance regressions:

```yaml
# .github/workflows/benchmarks.yml
- name: Run benchmarks
  run: |
    pytest tests/benchmarks/test_tool_selection_performance.py \
      -v --benchmark-only --benchmark-autosave

- name: Compare with baseline
  run: |
    pytest-benchmark compare --fail=regression baseline.json
```

## Resources

- [pytest-benchmark documentation](https://pytest-benchmark.readthedocs.io/)
- [Victor tool selection architecture](../../docs/development/TOOL_SELECTION.md)
- [Performance optimization guide](../../docs/development/PERFORMANCE.md)
