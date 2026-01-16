# Performance Testing and Benchmarking Suite

Comprehensive performance testing and benchmarking for the Victor workflow system, focusing on team nodes and the visual workflow editor.

## Quick Start

```bash
# Run all benchmarks
python scripts/benchmark_runner.py --all

# Run quick smoke test
python scripts/benchmark_runner.py --quick

# Generate report
python scripts/benchmark_runner.py --all --report --format markdown
```

## What's Included

### 1. Team Node Benchmarks (`tests/performance/team_node_benchmarks.py`)

Comprehensive benchmarks for team node execution:

- **8 Formation Types**: Sequential, Parallel, Pipeline, Hierarchical, Consensus, Dynamic, Adaptive, Hybrid
- **Scalability Tests**: 1-10 team members
- **Recursion Depth**: Nested team execution overhead
- **Memory Profiling**: Per-member memory footprint
- **Tool Budget Impact**: Performance with different budgets

**Key Performance Targets:**
- Team execution (3 members, parallel): <30ms
- Recursion overhead: <1ms per level
- Memory usage: <10MB for 10 members

### 2. Visual Editor Benchmarks (`tests/performance/editor_benchmarks.py`)

Comprehensive benchmarks for the visual workflow editor UI:

- **Page Load Time**: 10-200 nodes
- **Node Rendering**: Per-node rendering performance
- **Connection Rendering**: Edge/Bezier curve performance
- **Zoom/Pan Responsiveness**: 60fps target
- **Search Performance**: Node search in large workflows
- **Auto-Layout**: Grid, hierarchical, force-directed
- **YAML Import/Export**: Format conversion performance
- **Memory Profiling**: Memory footprint with large workflows

**Key Performance Targets:**
- Editor load (100 nodes): <500ms
- Node rendering: <16ms per node (60fps)
- Auto-layout: <500ms for hierarchical
- YAML import: <500ms, export: <300ms
- Memory: <100MB for 100 nodes

### 3. Workflow Execution Benchmarks (`tests/performance/workflow_execution_benchmarks.py`)

Comprehensive benchmarks for workflow execution:

- **End-to-End Execution**: Linear, parallel, conditional workflows
- **Node Throughput**: Nodes per second
- **Parallel Efficiency**: Speedup from parallelization
- **Recursion Depth**: Nested workflow overhead
- **Tool Execution**: Tool call overhead
- **Conditional Branching**: Performance of conditional edges
- **State Management**: Overhead of state updates
- **Caching Effectiveness**: Performance improvement from caching

**Key Performance Targets:**
- Simple workflow (5 nodes): <100ms
- Medium workflow (20 nodes): <400ms
- Complex workflow (50 nodes): <1000ms
- Throughput: >100 nodes/second
- Recursion overhead: <5% per level
- Tool execution: <5ms per call

## Usage Examples

### Run Specific Benchmark Suite

```bash
# Team node benchmarks
python scripts/benchmark_runner.py --suite team_nodes

# Visual editor benchmarks
python scripts/benchmark_runner.py --suite editor

# Workflow execution benchmarks
python scripts/benchmark_runner.py --suite workflow_execution
```

### Run with Filters

```bash
# Run only formation benchmarks
python scripts/benchmark_runner.py --suite team_nodes --filter "formation"

# Run only rendering benchmarks
python scripts/benchmark_runner.py --suite editor --filter "rendering"

# Run only recursion benchmarks
python scripts/benchmark_runner.py --suite workflow_execution --filter "recursion"
```

### Generate Reports

```bash
# Generate markdown report
python scripts/benchmark_runner.py --all --report --format markdown

# Generate HTML report
python scripts/benchmark_runner.py --all --report --format html

# Generate JSON report for CI/CD
python scripts/benchmark_runner.py --all --report --format json
```

### Compare with Baseline

```bash
# Run benchmarks and compare with baseline
python scripts/benchmark_runner.py --all --compare baseline.json

# Check for regressions (exits with error if regression found)
python scripts/benchmark_runner.py --all --compare baseline.json --regression-check
```

### Using pytest Directly

```bash
# Run all benchmarks
pytest tests/performance/ -v --benchmark-only

# Run specific file
pytest tests/performance/team_node_benchmarks.py -v --benchmark-only

# Run with filter
pytest tests/performance/ -k "formation" -v --benchmark-only

# Run summary tests
pytest tests/performance/team_node_benchmarks.py -k "summary" -v -s
```

## Benchmark Scripts

### `scripts/benchmark_runner.py`

Unified benchmark runner for all suites.

**Features:**
- Run all or specific benchmark suites
- Filter benchmarks by name
- Compare with baseline
- Generate reports in multiple formats
- Detect performance regressions
- CI/CD integration support

**Examples:**
```bash
python scripts/benchmark_runner.py --all --report
python scripts/benchmark_runner.py --suite team_nodes --filter "parallel"
python scripts/benchmark_runner.py --all --compare baseline.json --regression-check
```

### `scripts/performance_report.py`

Generate performance reports from benchmark results.

**Features:**
- Generate reports from existing results
- Compare with baseline
- Multiple output formats (Markdown, HTML, JSON)
- Regression detection
- Trend analysis

**Examples:**
```bash
python scripts/performance_report.py --input results.json --output-format markdown
python scripts/performance_report.py --all --baseline baseline.json --output-format html
```

## Documentation

- **[PERFORMANCE_BENCHMARKS.md](PERFORMANCE_BENCHMARKS.md)**: Complete benchmark documentation
  - Overview and architecture
  - Performance targets
  - Running benchmarks
  - CI/CD integration
  - Best practices
  - Troubleshooting

- **[PERFORMANCE_RESULTS.md](PERFORMANCE_RESULTS.md)**: Baseline performance results
  - Executive summary
  - Detailed benchmark results
  - Performance trends
  - Regression detection
  - Recommendations

## Performance Targets Summary

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

## File Structure

```
victor/
├── tests/
│   └── performance/
│       ├── team_node_benchmarks.py         # Team node benchmarks
│       ├── editor_benchmarks.py            # Visual editor benchmarks
│       ├── workflow_execution_benchmarks.py # Workflow execution benchmarks
│       ├── test_registry_performance.py    # Registry benchmarks
│       └── test_team_node_performance.py   # Legacy team node tests
├── scripts/
│   ├── benchmark_runner.py                 # Unified benchmark runner
│   └── performance_report.py              # Report generator
└── docs/
    ├── PERFORMANCE_BENCHMARKS.md           # Complete documentation
    └── PERFORMANCE_RESULTS.md             # Baseline results
```

## Key Features

1. **Comprehensive Coverage**: 45+ benchmarks across 3 suites
2. **Performance Targets**: Defined targets for all operations
3. **Regression Detection**: Automatic comparison with baselines
4. **Multiple Formats**: Markdown, HTML, JSON reports
5. **CI/CD Ready**: Easy integration with GitHub Actions, GitLab CI
6. **Memory Profiling**: Track memory usage patterns
7. **Scalability Testing**: Test with 1-200+ nodes
8. **Detailed Metrics**: Median, min, max, std dev, iterations

## Next Steps

1. **Run the benchmarks**:
   ```bash
   python scripts/benchmark_runner.py --quick
   ```

2. **Review the documentation**:
   - [PERFORMANCE_BENCHMARKS.md](docs/PERFORMANCE_BENCHMARKS.md)
   - [PERFORMANCE_RESULTS.md](docs/PERFORMANCE_RESULTS.md)

3. **Integrate with CI/CD**:
   - Add benchmark step to your CI pipeline
   - Set up regression detection
   - Track performance over time

4. **Establish baselines**:
   ```bash
   python scripts/benchmark_runner.py --all
   # Save results as baseline
   ```

5. **Monitor performance**:
   - Run benchmarks regularly
   - Track trends over time
   - Detect regressions early

## Troubleshooting

**Benchmarks failing?**
- Increase timeout: `pytest --timeout=300`
- Reduce iterations: `--iterations 1`
- Check system resources

**High variance?**
- Close unnecessary applications
- Increase iterations: `--iterations 5`
- Use consistent environment

**Regressions detected?**
- Verify with multiple runs
- Check for environmental changes
- Update baseline if improvement is intentional

## Contributing

When adding new benchmarks:

1. Follow existing naming conventions
2. Include performance targets
3. Add documentation to PERFORMANCE_BENCHMARKS.md
4. Update scripts/benchmark_runner.py if new suite
5. Run `--quick` tests first
6. Ensure CI/CD passes

## License

Apache License 2.0 - See LICENSE file for details

## Support

For issues or questions:
- GitHub Issues: [victor-ai/issues](https://github.com/your-org/victor-ai/issues)
- Documentation: [docs/](docs/)
- Performance Team: performance@victor-ai.org
