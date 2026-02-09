# Performance Benchmarking Guide

This guide provides comprehensive instructions for running, interpreting, and extending performance benchmarks for
  Victor AI.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Running Benchmarks](#running-benchmarks)
- [Interpreting Results](#interpreting-results)
- [Adding New Benchmarks](#adding-new-benchmarks)
- [Continuous Monitoring](#continuous-monitoring)

## Overview

Victor AI includes a comprehensive benchmark suite to validate Phase 4 performance improvements:

### Performance Targets

| Metric | Target | Improvement |
|--------|--------|-------------|
| **Initialization Time** | < 500ms | 95% reduction (from ~10s) |
| **Throughput** | 15-25% faster | Parallel execution optimization |
| **Memory Usage** | < 100MB | 50% reduction (from ~200MB) |

### Benchmark Categories

1. **Initialization** (`test_initialization.py`)
   - Cold start time
   - Lazy loading benefits
   - Component initialization

2. **Throughput** (`test_throughput.py`)
   - Parallel tool execution
   - Concurrent request handling
   - StateGraph workflow performance

3. **Memory** (`test_memory.py`)
   - Memory usage patterns
   - Memory leak detection
   - Cache effectiveness

## Quick Start

```bash
# Quick benchmark suite (~30 seconds)
python scripts/benchmark_suite.py --profile quick

# Full benchmark suite (~5 minutes)
python scripts/benchmark_suite.py --profile full

# Export results
python scripts/benchmark_suite.py --profile quick --export results.json
```

## Running Benchmarks

### Benchmark Suite Script

```bash
# Profiles
python scripts/benchmark_suite.py --profile quick    # ~30 seconds
python scripts/benchmark_suite.py --profile full     # ~5 minutes
python scripts/benchmark_suite.py --profile stress   # ~30 minutes

# Categories
python scripts/benchmark_suite.py --category initialization
python scripts/benchmark_suite.py --category throughput
python scripts/benchmark_suite.py --category memory
```

### Using pytest Directly

```bash
# Run all benchmarks
pytest tests/performance/benchmarks/ -v --benchmark-only

# Run specific category
pytest tests/performance/benchmarks/test_initialization.py -v --benchmark-only

# Save to JSON
pytest tests/performance/benchmarks/ --benchmark-only --benchmark-json=results.json
```

## Interpreting Results

### Understanding Output

```
------------------------------------------------------------------------------------------
Name (time in ms)                       Min       Max      Mean    StdDev    Median     Rounds
------------------------------------------------------------------------------------------
test_cold_start_orchestrator_factory  45.2341  67.8912  52.3456   8.2345   51.2345        10
------------------------------------------------------------------------------------------
```

- **Min**: Fastest execution time
- **Max**: Slowest execution time
- **Mean**: Average execution time
- **StdDev**: Standard deviation (lower = more consistent)
- **Median**: Median execution time

### Performance Assertions

Benchmarks include automatic assertions to validate Phase 4 targets:

```python
assert elapsed < 0.5, f"Factory initialization too slow: {elapsed:.3f}s (target: < 500ms)"
```

### Comparing Results

```bash
# Generate comparison report
python scripts/benchmark_suite.py --compare baseline.json current.json --report comparison.html
```

## Adding New Benchmarks

### Template

```python
class Test<Component>Performance:
    """Performance benchmarks for <component>."""

    def test_<operation>_performance(self, benchmark):
        """Benchmark <operation>.

        Expected: <target_time>
        """
        def operation():
            return result

        result = benchmark(operation)
        assert result is not None
```

### Best Practices

1. Use pytest-benchmark decorator
2. Include both benchmark and assertion tests
3. Mock external dependencies
4. Clean up resources

## Continuous Monitoring

### CI/CD Integration

Add to CI pipeline:

```yaml
- name: Run benchmarks
  run: python scripts/benchmark_suite.py --profile quick --export results.json
```

### Pre-commit Hook

```bash
# Add alias to .bashrc/.zshrc
alias bench='python scripts/benchmark_suite.py --profile quick'
```

## Resources

- [pytest-benchmark Documentation](https://pytest-benchmark.readthedocs.io/)
- [Phase 4 Performance Baselines](../performance/benchmark_results.md)
- [Architecture Documentation](../architecture/overview.md)

---

## See Also

- [Documentation Home](../../README.md)


**Last Updated:** February 01, 2026
**Reading Time:** 1 min
