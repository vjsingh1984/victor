# Tool Selection Performance Benchmarks - Quick Start

This guide helps you get started with tool selection performance benchmarks.

## Prerequisites

Install dev dependencies (including pytest-benchmark):

```bash
pip install -e ".[dev]"
```

## Quick Start

### 1. Run All Benchmarks

```bash
pytest tests/benchmarks/test_tool_selection_performance.py -v --benchmark-only
```

### 2. Run Specific Benchmark Group

```bash
# Baseline performance only
pytest tests/benchmarks/test_tool_selection_performance.py -v --benchmark-only -k "baseline"

# Caching performance only
pytest tests/benchmarks/test_tool_selection_performance.py -v --benchmark-only -k "cache"

# Memory usage only
pytest tests/benchmarks/test_tool_selection_performance.py -v --benchmark-only -k "memory"
```

### 3. Using the Helper Script

```bash
# Run all benchmarks
python scripts/run_tool_selection_benchmarks.py run

# Run specific group
python scripts/run_tool_selection_benchmarks.py run --group baseline

# Run with histogram
python scripts/run_tool_selection_benchmarks.py run --histogram

# List saved runs
python scripts/run_tool_selection_benchmarks.py list

# Compare two runs
python scripts/run_tool_selection_benchmarks.py compare 0001.json 0002.json

# Show summary
python scripts/run_tool_selection_benchmarks.py summary
```

## Understanding Results

### Sample Output

```
------------------------------------------------------------------------------------------------
Benchmark (time in ms)            Min       Max       Mean     StdDev    Rounds    Iterations
------------------------------------------------------------------------------------------------
test_baseline_tool_selection_10   35.2341   48.1523   40.3124  3.2145    20        10
test_cached_query_embedding        0.8234    2.3456    1.2345   0.4321    20        100
test_uncached_query_embedding     38.4567   52.3456   44.7234  4.5678    10        50
------------------------------------------------------------------------------------------------
```

**Columns:**
- `Min`: Fastest execution time
- `Max`: Slowest execution time
- `Mean`: Average execution time
- `StdDev`: Standard deviation (consistency)
- `Rounds`: Number of benchmark rounds
- `Iterations`: Iterations per round

### Performance Targets

| Benchmark | Target | Status |
|-----------|--------|--------|
| Tool selection (10 tools) | <50ms | 🟢 40ms |
| Tool selection (47 tools) | <100ms | 🟢 82ms |
| Cached query lookup | <5ms | 🟢 1.2ms |
| Batch embedding (10 items) | <100ms | 🟡 TBD |

## Regression Detection

### 1. Establish Baseline

Run benchmarks to establish baseline:

```bash
pytest tests/benchmarks/test_tool_selection_performance.py \
  -v --benchmark-only --benchmark-autosave
```

This saves results to `.benchmarks/` directory.

### 2. Make Changes

Implement your optimizations.

### 3. Compare with Baseline

```bash
# List saved runs
python scripts/run_tool_selection_benchmarks.py list

# Compare latest with baseline
python scripts/run_tool_selection_benchmarks.py compare <baseline.json> <latest.json>
```

### 4. Regression Threshold

Tests automatically fail if performance degrades by >20%:

```bash
# This will fail if any benchmark is >20% slower than baseline
pytest tests/benchmarks/test_tool_selection_performance.py \
  -v --benchmark-only --benchmark-autosave --benchmark-compare-fail=min:20%
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Performance Benchmarks

on: [pull_request, push]

jobs:
  benchmark:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"

      - name: Run benchmarks
        run: |
          pytest tests/benchmarks/test_tool_selection_performance.py \
            -v --benchmark-only --benchmark-autosave

      - name: Store benchmark results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: .benchmarks/

      - name: Compare with baseline
        if: github.event_name == 'pull_request'
        run: |
          pytest-benchmark compare baseline.json \
            --fail=regression --columns=min,max,mean
```

## Profiling Deep Dives

### Identify Bottlenecks

Run specific bottleneck tests:

```bash
pytest tests/benchmarks/test_tool_selection_performance.py \
  -v --benchmark-only -k "bottleneck"
```

### Memory Profiling

Use memory-specific benchmarks:

```bash
pytest tests/benchmarks/test_tool_selection_performance.py \
  -v --benchmark-only -k "memory" -s
```

## Tips for Accurate Benchmarks

1. **Close other applications**: Reduce system noise
2. **Use consistent hardware**: Run on same machine
3. **Multiple rounds**: More rounds = more accurate (default: 20)
4. **Warm up**: First run is slower (cache warmup)
5. **Disable debuggers**: Don't run under debugger

## Common Issues

### Issue: "No module named 'sentence_transformers'"

**Solution:**
```bash
pip install -e ".[dev]"
```

### Issue: Benchmarks are slow

**Solution:** Reduce iterations/rounds:
```bash
pytest tests/benchmarks/test_tool_selection_performance.py \
  -v --benchmark-only --benchmark-min-rounds=3
```

### Issue: Inconsistent results

**Solution:**
- Close other applications
- Increase `--benchmark-min-rounds`
- Use `--benchmark-warmup` for JIT compilers

## Next Steps

1. ✅ Run baseline benchmarks
2. 📊 Record baseline metrics
3. 🔧 Implement optimizations
4. 📈 Compare with baseline
5. ✅ Verify improvements

## Resources

- [Full benchmark documentation](../../tests/benchmarks/README.md)
- [Tool selection architecture](../../docs/development/TOOL_SELECTION.md)
- [pytest-benchmark docs](https://pytest-benchmark.readthedocs.io/)
