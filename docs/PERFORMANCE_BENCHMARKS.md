# Victor Workflow System - Performance Benchmarks

This document provides comprehensive documentation for the performance benchmarking system for the Victor workflow system, including team nodes, visual workflow editor, and workflow execution.

## Table of Contents

1. [Overview](#overview)
2. [Benchmark Suites](#benchmark-suites)
3. [Performance Targets](#performance-targets)
4. [Running Benchmarks](#running-benchmarks)
5. [Generating Reports](#generating-reports)
6. [CI/CD Integration](#cicd-integration)
7. [Performance Regression Testing](#performance-regression-testing)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

## Overview

The Victor workflow system includes a comprehensive performance benchmarking framework designed to:

- **Measure performance** across team nodes, visual editor, and workflow execution
- **Detect regressions** by comparing against baseline measurements
- **Generate reports** in multiple formats (Markdown, JSON, HTML)
- **Support CI/CD integration** with automated regression detection
- **Provide actionable insights** for optimization

### Key Features

- **8 Team Formations**: Sequential, Parallel, Pipeline, Hierarchical, Consensus, Dynamic, Adaptive, Hybrid
- **3 Benchmark Suites**: Team nodes, Visual editor, Workflow execution
- **Multiple Metrics**: Latency, throughput, memory usage, scalability
- **Automated Testing**: pytest-based with `pytest-benchmark` integration
- **Regression Detection**: Automatic comparison with baselines

## Benchmark Suites

### 1. Team Node Performance

**Location**: `tests/performance/team_node_benchmarks.py`

Tests the performance of team node execution with different formations and team sizes.

#### Key Benchmarks

| Benchmark | Description | Target |
|-----------|-------------|--------|
| `test_formation_performance` | Execution time per formation type | <30-80ms (varies by formation) |
| `test_scalability_sequential` | Sequential scaling (1-10 members) | O(n) linear |
| `test_scalability_parallel` | Parallel scaling (1-10 members) | O(1) constant |
| `test_recursion_depth_overhead` | Recursion tracking overhead | <1ms per level |
| `test_memory_per_member` | Memory usage per member | <1MB per member |
| `test_consensus_formation` | Consensus formation performance | <80ms for 3 members |
| `test_tool_budget_impact` | Tool budget effect on performance | Linear relationship |

#### Formation Performance Targets

| Formation | Target (3 members) | Description |
|-----------|-------------------|-------------|
| Sequential | <50ms | Members execute one after another |
| Parallel | <30ms | All members execute simultaneously |
| Pipeline | <40ms | Output flows through stages |
| Hierarchical | <35ms | Manager delegates to workers |
| Consensus | <80ms | Members must agree (multiple rounds) |
| Dynamic | <35ms | Adaptive formation selection |
| Adaptive | <40ms | Performance-tuned execution |
| Hybrid | <45ms | Pipeline + parallel stages |

### 2. Visual Workflow Editor

**Location**: `tests/performance/editor_benchmarks.py`

Tests the performance of the visual workflow editor UI with large workflows.

#### Key Benchmarks

| Benchmark | Description | Target |
|-----------|-------------|--------|
| `test_editor_load_time` | Page load time (10-200 nodes) | <500ms for 100 nodes |
| `test_node_rendering_performance` | Node rendering (50-200 nodes) | <16ms per node (60fps) |
| `test_edge_rendering_performance` | Edge/connection rendering | <5ms per edge |
| `test_zoom_performance` | Zoom operation time | <16ms (60fps) |
| `test_pan_performance` | Pan operation time | <16ms (60fps) |
| `test_search_performance` | Node search time | <10ms for 100 nodes |
| `test_auto_layout_performance` | Auto-layout calculation | <500ms for hierarchical |
| `test_yaml_export_performance` | YAML export time | <300ms for 100 nodes |
| `test_yaml_import_performance` | YAML import time | <500ms for 100 nodes |
| `test_memory_usage` | Memory footprint | <100MB for 100 nodes |

#### Editor Performance Targets

| Operation | Target (100 nodes) | Description |
|-----------|-------------------|-------------|
| Load time | <500ms | Initialize and render editor |
| Node render | <16ms/node | Render nodes at 60fps |
| Zoom/Pan | <16ms/operation | Smooth interactions |
| Auto-layout | <500ms | Hierarchical layout |
| YAML export | <300ms | Serialize to YAML |
| YAML import | <500ms | Parse from YAML |
| Memory | <100MB | Total memory usage |

### 3. Workflow Execution

**Location**: `tests/performance/workflow_execution_benchmarks.py`

Tests the performance of workflow execution across different scenarios.

#### Key Benchmarks

| Benchmark | Description | Target |
|-----------|-------------|--------|
| `test_linear_workflow_execution` | Linear workflow (5-50 nodes) | <1000ms for 50 nodes |
| `test_parallel_workflow_execution` | Parallel workflow (2-5 branches) | >70% efficiency |
| `test_node_throughput` | Nodes per second | >100 nodes/s |
| `test_recursion_depth_impact` | Nested workflow overhead | <5% per level |
| `test_tool_execution_overhead` | Tool call cost | <5ms per call |
| `test_conditional_branching` | Conditional edge performance | Minimal overhead |
| `test_state_management_overhead` | State update cost | <10ms per update |
| `test_caching_effectiveness` | Cache speedup | >1.5x improvement |
| `test_memory_workflow_execution` | Memory usage | <50MB for 100 nodes |

#### Workflow Execution Targets

| Workflow Type | Target | Description |
|---------------|--------|-------------|
| Simple (5 nodes) | <100ms | Basic linear workflow |
| Medium (20 nodes) | <400ms | Moderate complexity |
| Complex (50 nodes) | <1000ms | Large workflow |
| Throughput | >100 nodes/s | Nodes per second |
| Recursion | <5% per level | Nested workflow overhead |
| Tool calls | <5ms/call | Tool execution overhead |

## Performance Targets

### Summary Table

| Category | Metric | Target |
|----------|--------|--------|
| **Team Nodes** | Execution (3 members, parallel) | <30ms |
| | Recursion overhead | <1ms per level |
| | Memory (10 members) | <10MB |
| **Editor** | Load time (100 nodes) | <500ms |
| | Node rendering | <16ms/node (60fps) |
| | Auto-layout | <500ms |
| | YAML import/export | <500ms/<300ms |
| | Memory (100 nodes) | <100MB |
| **Workflow** | Simple (5 nodes) | <100ms |
| | Medium (20 nodes) | <400ms |
| | Complex (50 nodes) | <1000ms |
| | Throughput | >100 nodes/s |
| | Tool calls | <5ms/call |

## Running Benchmarks

### Quick Start

```bash
# Run all benchmarks
python scripts/benchmark_runner.py --all

# Run specific suite
python scripts/benchmark_runner.py --suite team_nodes
python scripts/benchmark_runner.py --suite editor
python scripts/benchmark_runner.py --suite workflow_execution

# Quick smoke test (key benchmarks only)
python scripts/benchmark_runner.py --quick
```

### Advanced Usage

#### Run with Filters

```bash
# Run only formation benchmarks
python scripts/benchmark_runner.py --suite team_nodes --filter "formation"

# Run only rendering benchmarks
python scripts/benchmark_runner.py --suite editor --filter "rendering"

# Run only recursion benchmarks
python scripts/benchmark_runner.py --suite workflow_execution --filter "recursion"
```

#### Run with Custom Iterations

```bash
# Run 5 iterations for more accurate results
python scripts/benchmark_runner.py --all --iterations 5
```

#### Run with Verbose Output

```bash
# Enable verbose pytest output
python scripts/benchmark_runner.py --all --verbose
```

#### Using pytest Directly

```bash
# Run all benchmarks
pytest tests/performance/ -v --benchmark-only

# Run specific file
pytest tests/performance/team_node_benchmarks.py -v --benchmark-only

# Run with filter
pytest tests/performance/ -k "formation" -v --benchmark-only

# Generate JSON output
pytest tests/performance/team_node_benchmarks.py --benchmark-only --benchmark-json=results.json

# Run summary tests
pytest tests/performance/team_node_benchmarks.py -k "summary" -v -s
```

## Generating Reports

### Generate Reports After Running

```bash
# Run benchmarks and generate markdown report
python scripts/benchmark_runner.py --all --report --format markdown

# Generate HTML report with visualizations
python scripts/benchmark_runner.py --all --report --format html

# Generate JSON report for CI/CD
python scripts/benchmark_runner.py --all --report --format json
```

### Generate Reports from Existing Results

```bash
# Generate report from existing benchmark JSON
python scripts/performance_report.py \
    --input /tmp/benchmark_results/team_nodes_20250115_120000.json \
    --output-format markdown

# Generate report with baseline comparison
python scripts/performance_report.py \
    --input current_results.json \
    --baseline baseline_results.json \
    --output-format html

# Generate reports for all suites
python scripts/performance_report.py --all --output-dir ./reports
```

### Report Formats

#### Markdown

```bash
python scripts/benchmark_runner.py --all --report --format markdown
```

Output: Human-readable Markdown report with tables and summaries.

#### HTML

```bash
python scripts/benchmark_runner.py --all --report --format html
```

Output: Styled HTML report with interactive tables and color-coded results.

#### JSON

```bash
python scripts/benchmark_runner.py --all --report --format json
```

Output: Machine-readable JSON for programmatic analysis.

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Performance Benchmarks

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  benchmark:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
          pip install pytest-benchmark

      - name: Run benchmarks
        run: |
          python scripts/benchmark_runner.py --all --report --format json

      - name: Check for regressions
        run: |
          python scripts/benchmark_runner.py \
            --all \
            --compare baseline.json \
            --regression-check

      - name: Upload benchmark results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: /tmp/benchmark_results/*.json
```

### GitLab CI Example

```yaml
performance:
  stage: test
  script:
    - pip install -e ".[dev]"
    - pip install pytest-benchmark
    - python scripts/benchmark_runner.py --all --report --format json
    - python scripts/benchmark_runner.py --all --compare baseline.json --regression-check
  artifacts:
    paths:
      - /tmp/benchmark_results/*.json
    reports:
      benchmark: benchmark_results.json
```

### Performance Regression Detection

```bash
# Run benchmarks and compare with baseline
python scripts/benchmark_runner.py \
    --all \
    --compare baseline.json \
    --regression-check

# Exit code 1 if regression detected
echo $?
```

Regression threshold: 10% slower than baseline (configurable).

## Performance Regression Testing

### Setting Up Baselines

```bash
# Run benchmarks to establish baseline
python scripts/benchmark_runner.py --all

# Save as baseline
cp /tmp/benchmark_results/team_nodes_*.json baseline/team_nodes_baseline.json
cp /tmp/benchmark_results/editor_*.json baseline/editor_baseline.json
cp /tmp/benchmark_results/workflow_execution_*.json baseline/workflow_execution_baseline.json
```

### Comparing with Baseline

```bash
# Compare current results with baseline
python scripts/benchmark_runner.py \
    --all \
    --compare baseline/team_nodes_baseline.json \
    --report \
    --format markdown
```

### Regression Thresholds

| Severity | Threshold | Action |
|----------|-----------|--------|
| Low | <5% change | Informational |
| Medium | 5-10% change | Warning |
| High | >10% change | Fail CI/CD |

### Example Regression Output

```
REGRESSION: team_nodes/test_formation_performance[PARALLEL]: 15.3% slower (25.2ms -> 29.1ms)
REGRESSION: editor/test_editor_load_time[100]: 12.8% slower (450.0ms -> 507.6ms)
```

## Best Practices

### Running Benchmarks

1. **Use a consistent environment**:
   - Same machine, OS, and Python version
   - Minimal background processes
   - Consistent CPU throttling settings

2. **Run multiple iterations**:
   ```bash
   python scripts/benchmark_runner.py --all --iterations 5
   ```

3. **Use quick tests for development**:
   ```bash
   python scripts/benchmark_runner.py --quick
   ```

4. **Run full suite before releases**:
   ```bash
   python scripts/benchmark_runner.py --all --report
   ```

### Analyzing Results

1. **Look for trends** rather than single measurements
2. **Focus on p95/p99** values, not just median
3. **Check variance** - high variance indicates instability
4. **Compare with baseline** to detect regressions
5. **Monitor memory** for leaks with large workflows

### Optimizing Performance

1. **Profile before optimizing**:
   ```bash
   python -m cProfile -o profile.stats tests/performance/team_node_benchmarks.py
   ```

2. **Focus on hot paths** identified by benchmarks
3. **Measure impact** of optimizations with benchmarks
4. **Document tradeoffs** in code comments

## Troubleshooting

### Benchmarks Failing

**Issue**: Benchmarks timeout or fail intermittently

**Solutions**:
- Increase timeout: `pytest --timeout=300`
- Reduce iterations: `--iterations 1`
- Check system resources: CPU, memory, disk I/O
- Disable power saving features

### High Variance

**Issue**: Benchmark results vary significantly between runs

**Solutions**:
- Close unnecessary applications
- Increase iterations: `--iterations 5`
- Use consistent environment
- Check for background processes
- Disable Turbo Boost (Intel) or Precision Boost (AMD)

### Regression Detection

**Issue**: False positive regressions

**Solutions**:
- Adjust regression threshold: `--regression-threshold 15.0`
- Update baseline if improvement is intentional
- Run multiple iterations to confirm
- Check for environmental factors

### Memory Issues

**Issue**: Out of memory errors with large workflows

**Solutions**:
- Reduce node count in benchmarks
- Increase system swap space
- Check for memory leaks with `tracemalloc`
- Profile memory usage

## Benchmark Data Storage

### Directory Structure

```
/tmp/benchmark_results/
├── team_nodes_20250115_120000.json
├── editor_20250115_120100.json
├── workflow_execution_20250115_120200.json
└── benchmark_report_20250115_120300.md
```

### Baseline Storage

```
baseline/
├── team_nodes_baseline.json
├── editor_baseline.json
└── workflow_execution_baseline.json
```

### Historical Data

```bash
# Store historical data for trend analysis
mkdir -p benchmark_history/$(date +%Y/%m/%d)
cp /tmp/benchmark_results/*.json benchmark_history/$(date +%Y/%m/%d)/
```

## Additional Resources

- [pytest-benchmark Documentation](https://pytest-benchmark.readthedocs.io/)
- [Python Profiling](https://docs.python.org/3/library/profile.html)
- [Performance Best Practices](https://docs.python.org/3/performance.html)
- [Victor Architecture Documentation](./architecture/)

## Contributing

When adding new benchmarks:

1. Follow existing naming conventions
2. Include performance targets
3. Add documentation to this file
4. Update `scripts/benchmark_runner.py` if new suite
5. Run `--quick` tests first
6. Ensure CI/CD passes

## Changelog

### Version 1.0.0 (2025-01-15)

- Initial comprehensive benchmark suite
- Team node formation benchmarks (8 formations)
- Visual editor benchmarks (rendering, layout, YAML)
- Workflow execution benchmarks (linear, parallel, recursion)
- Automated report generation (Markdown, HTML, JSON)
- CI/CD integration with regression detection
- Performance targets and baselines
