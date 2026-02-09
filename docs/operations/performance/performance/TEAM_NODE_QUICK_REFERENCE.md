# Team Node Performance Benchmarks - Quick Reference

## Quick Start

```bash
# Run all benchmarks
pytest tests/performance/test_team_node_performance_benchmark.py -v

# Run summary report only
pytest tests/performance/test_team_node_performance_benchmark.py::test_team_node_performance_summary -v -s

# Using the convenience script
python scripts/run_team_node_benchmarks.py --all
python scripts/run_team_node_benchmarks.py --report
```text

## Benchmark Categories

### 1. Single Level Team Execution
Tests team execution with varying member counts (2, 4, 8 members).

**Key Results:**
- 2 members: ~17ms
- 4 members: ~17ms
- 8 members: ~17ms
- **Constant time complexity** for parallel formation

### 2. Nested Team Execution
Tests nested team execution with recursion tracking (depths 1-3).

**Key Results:**
- Depth 1: ~2ms per level
- Depth 2: ~1.9ms per level
- Depth 3: ~1.8ms per level
- **Overhead decreases with depth** (excellent)

### 3. Formation Performance
Compares different team formations (sequential, parallel, pipeline, hierarchical).

**Key Results:**
- Parallel: ~17ms (3x speedup)
- Sequential: ~50ms (baseline)
- Pipeline: ~50ms (similar to sequential)
- Hierarchical: ~34ms (1.5x speedup)

### 4. Recursion Tracking Overhead
Measures overhead of RecursionContext tracking.

**Key Results:**
- With tracking: ~9ms
- Without tracking: ~8.8ms
- **Overhead: 2.6-3.0%** (excellent)

### 5. Memory Usage
Measures memory footprint for different team sizes.

**Key Results:**
- 2 members: ~25KB
- 4 members: ~30KB
- 8 members: ~42KB
- **No memory leaks detected**

### 6. RecursionContext Operations
Benchmarks individual RecursionContext operations.

**Key Results:**
- Enter/Exit: ~0.26μs per operation
- Thread-safe locking: minimal overhead
- **Linear scaling: O(n)**

## Performance Targets

| Metric                            | Target              | Achieved          | Status |
|-----------------------------------|---------------------|-------------------|--------|
| Team execution (parallel)         | < 100ms             | ~17ms             | ✅     |
| Recursion tracking overhead       | < 10%               | 2.6-3.0%          | ✅     |
| Memory usage (8 members)          | < 1MB               | 42KB              | ✅     |
| Nested execution overhead         | < 10ms per level    | ~2ms per level    | ✅     |
| Parallel speedup vs sequential    | > 2x                | 3.0x              | ✅     |
| Memory leak growth                | < 50%               | 5.5%              | ✅     |
| RecursionContext overhead         | < 2KB per level     | 1.85KB per level  | ✅     |

**All targets met or exceeded.**

## Running Specific Benchmarks

```bash
# Single level execution
pytest tests/performance/test_team_node_performance_benchmark.py -k "single_level" -v

# Nested execution
pytest tests/performance/test_team_node_performance_benchmark.py -k "nested" -v

# Formation comparison
pytest tests/performance/test_team_node_performance_benchmark.py -k "formation" -v

# Recursion context
pytest tests/performance/test_team_node_performance_benchmark.py -k "recursion_context" -v

# Memory usage
pytest tests/performance/test_team_node_performance_benchmark.py -k "memory" -v
```

## Files

- **Benchmark File:** `tests/performance/test_team_node_performance_benchmark.py`
- **Report:** `docs/performance/TEAM_NODE_BENCHMARK_REPORT.md`
- **Results JSON:** `/tmp/benchmark_results/team_node_performance_benchmark.json`
- **Convenience Script:** `scripts/run_team_node_benchmarks.py`

## Key Findings

### Performance
- ✅ **Excellent:** All operations complete in <50ms
- ✅ **Scalable:** Parallel formation provides constant execution time
- ✅ **Efficient:** Recursion tracking adds only 2.6% overhead

### Memory
- ✅ **Minimal:** <50KB for 8-member teams
- ✅ **Stable:** No memory leaks detected
- ✅ **Linear:** Predictable memory growth

### Production Readiness
- ✅ All performance targets met
- ✅ Comprehensive test coverage
- ✅ Thread-safe implementation
- ✅ Robust error handling

## Recommendations

### Team Formation Selection
1. **Default to Parallel** for independent tasks (3x faster)
2. **Use Sequential** for dependent tasks
3. **Use Pipeline** for staged processing
4. **Use Hierarchical** for manager-worker patterns

### Team Sizing
- **2-3 members:** Simple tasks, minimal overhead
- **4-6 members:** Balanced for medium complexity
- **7-10 members:** Complex tasks, monitor overhead

### Recursion Depth
- **Default max_depth: 3** optimal for most cases
- **Increase to 5** for complex nested workflows
- **Decrease to 2** for performance-critical paths

## Generating Reports

### Summary Report
```bash
python scripts/run_team_node_benchmarks.py --report
```text

### Full Benchmark Results
```bash
pytest tests/performance/test_team_node_performance_benchmark.py \
    --benchmark-only \
    --benchmark-json=team_node_results.json
```

### Performance Comparison
Compare two benchmark runs:
```python
import json

# Load baseline
with open("baseline.json") as f:
    baseline = json.load(f)

# Load current
with open("current.json") as f:
    current = json.load(f)

# Compare metrics (manual comparison or use tool)
```bash

## Interpretation Guide

### Execution Time
- <20ms: Excellent (parallel teams)
- 20-50ms: Good (sequential/pipeline)
- >50ms: Investigate (complex formations)

### Memory Usage
- <50KB: Excellent (all team sizes)
- 50-100KB: Good
- >100KB: Monitor (large teams)

### Recursion Overhead
- <5%: Excellent (well within target)
- 5-10%: Good (acceptable)
- >10%: Investigate optimization opportunities

## Troubleshooting

### Benchmarks Fail to Run
- Ensure pytest-benchmark is installed: `pip install pytest-benchmark`
- Check Python version >= 3.10
- Verify file paths are correct

### Unexpected Performance Degradation
- Check system load (CPU, memory)
- Verify no background processes
- Re-run benchmarks multiple times
- Check for asyncio event loop issues

### Memory Leaks Detected
- Run leak detection benchmark: `-k "memory_leak"`
- Check if growth exceeds 50%
- Review team member cleanup code
- Verify context manager usage

## Contributing

When adding new benchmarks:

1. Follow existing naming pattern: `test_[category]_[detail]`
2. Use `@pytest.mark.benchmark` decorator
3. Add clear performance targets in docstring
4. Print results for visibility
5. Update this quick reference

## Support

For issues or questions:
- Check main report: `docs/performance/TEAM_NODE_BENCHMARK_REPORT.md`
- Review benchmark code: `tests/performance/test_team_node_performance_benchmark.py`
- Run with verbose output: `-v -s` flags

---

## See Also

- [Documentation Home](../../README.md)


**Last Updated:** February 01, 2026
**Reading Time:** 3 minutes
