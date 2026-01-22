# Performance Benchmarks

Comprehensive benchmark suite for validating Phase 4 performance improvements in Victor AI.

## Overview

This benchmark suite validates the following Phase 4 improvements:

- **95% initialization time reduction** through lazy loading
- **15-25% throughput improvement** through parallel execution
- **15-25% memory reduction** through component optimization

## Benchmark Files

### test_initialization.py

Performance benchmarks for initialization and startup operations.

**Benchmarks (8 total)**:
- Cold start time (< 500ms target)
- Lazy loading benefits (> 90% time saved)
- Component initialization (< 10ms per component)
- Memory usage during startup (< 100MB target)

### test_throughput.py

Performance benchmarks for throughput and concurrent operations.

**Benchmarks (7 total)**:
- Parallel tool execution (15-25% faster)
- Concurrent request handling (linear scaling)
- StateGraph workflow throughput (20% improvement)
- Multi-agent coordination (15% improvement)

### test_memory.py

Performance benchmarks for memory usage and optimization.

**Benchmarks (8 total)**:
- Memory usage patterns (< 100MB target)
- Memory leak detection (no unbounded growth)
- Cache effectiveness (> 80% hit rate)
- Memory consolidation (> 50% released after GC)

## Quick Start

```bash
# Run all benchmarks with pytest
pytest tests/performance/benchmarks/ -v --benchmark-only

# Run specific category
pytest tests/performance/benchmarks/test_initialization.py -v --benchmark-only

# Run specific benchmark
pytest tests/performance/benchmarks/test_initialization.py::TestColdStartPerformance::test_cold_start_orchestrator_factory -v --benchmark-only
```

## Using Benchmark Suite Script

The `scripts/benchmark_suite.py` provides a convenient CLI:

```bash
# Quick benchmarks (~30 seconds)
python scripts/benchmark_suite.py --profile quick

# Full benchmarks (~5 minutes)
python scripts/benchmark_suite.py --profile full

# Stress tests (~30 minutes)
python scripts/benchmark_suite.py --profile stress

# Export results
python scripts/benchmark_suite.py --profile full --export results.json

# Compare runs
python scripts/benchmark_suite.py --compare baseline.json current.json --report comparison.html
```

## Performance Targets

| Metric | Target | Previous | Improvement |
|--------|--------|----------|-------------|
| Initialization Time | < 500ms | ~10,000ms | 95% |
| Memory Usage | < 100MB | ~200MB | 50% |
| Throughput | +15-25% | baseline | 20% avg |

## Benchmark Structure

Each benchmark file follows this structure:

```python
class Test<Feature>Performance:
    """Performance benchmarks for <feature>."""

    def test_<operation>_performance(self, benchmark):
        """Benchmark <operation>.

        Expected: <target_time>
        """
        result = benchmark(operation)
        assert result is not None

    def test_<operation>_meets_target(self):
        """Assert <operation> meets performance target.

        Target: <target_value>
        """
        # Measure absolute time
        assert elapsed < target_time
```

## Writing New Benchmarks

### Template

```python
def test_my_benchmark(self, benchmark):
    """Benchmark my operation.

    Expected: < 100ms
    """
    def operation():
        # Code to benchmark
        return result

    result = benchmark(operation)
    assert result is not None
```

### Best Practices

1. **Use pytest-benchmark decorator**: Always use the `benchmark` fixture
2. **Include assertions**: Validate results meet targets
3. **Mock dependencies**: Use mocks for external services
4. **Clean up resources**: Use `gc.collect()` between iterations
5. **Document targets**: Clearly state expected performance

## pytest-benchmark Options

```bash
# Minimum rounds
--benchmark-min-rounds=10

# Maximum time per benchmark (seconds)
--benchmark-max-time=5

# Generate histogram
--benchmark-histogram

# Sort results
--benchmark-sort={name,mean,stddev,min,max}

# Save to JSON
--benchmark-json=results.json

# Only run benchmarks
--benchmark-only
```

## Interpreting Results

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
- **Median**: Median execution time (less affected by outliers)
- **Rounds**: Number of benchmark iterations

## Performance Assertions

Benchmarks include automatic assertions:

```python
def test_initialization_time_meets_target(self):
    """Assert initialization time meets Phase 4 target.

    Target: < 500ms (95% reduction from ~10s)
    """
    start = time.perf_counter()
    factory = create_orchestrator_factory(...)
    elapsed = time.perf_counter() - start

    assert elapsed < 0.5, f"Too slow: {elapsed:.3f}s (target: < 500ms)"
```

## CI/CD Integration

Add to CI pipeline:

```yaml
- name: Performance Benchmarks
  run: |
    python scripts/benchmark_suite.py --profile quick --export benchmark_results.json

- name: Check for regressions
  run: |
    python scripts/compare_benchmarks.py \
      --baseline results/baseline.json \
      --current benchmark_results.json \
      --threshold 20
```

## Troubleshooting

### Benchmarks Too Slow

```bash
# Use quick profile
python scripts/benchmark_suite.py --profile quick

# Reduce iterations
python scripts/benchmark_suite.py --iterations 1
```

### Inconsistent Results

```bash
# Increase minimum rounds
pytest tests/performance/benchmarks/ --benchmark-min-rounds=20

# Use pedantic mode
benchmark.pedantic(operation, rounds=20, iterations=5)
```

## Documentation

- [Performance Baselines](../../../docs/PERFORMANCE_BASELINES.md) - Detailed results
- [Benchmarking Guide](../../../docs/guides/PERFORMANCE_BENCHMARKING.md) - Usage guide
- [pytest-benchmark Docs](https://pytest-benchmark.readthedocs.io/)

## Contributing

When adding performance improvements:

1. Add benchmarks for your changes
2. Update performance baselines
3. Document improvements
4. Add assertions for validation
5. Run full suite before PR

## License

Apache License 2.0 - See LICENSE file for details
