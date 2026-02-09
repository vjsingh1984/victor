# Victor AI Benchmark Suite - Quick Reference

## Quick Commands

```bash
# Run all benchmarks
python scripts/benchmark_comprehensive.py run --all

# Run specific scenario
python scripts/benchmark_comprehensive.py run --scenario startup
python scripts/benchmark_comprehensive.py run --scenario memory
python scripts/benchmark_comprehensive.py run --scenario throughput
python scripts/benchmark_comprehensive.py run --scenario latency

# Generate reports
python scripts/benchmark_comprehensive.py report --format markdown
python scripts/benchmark_comprehensive.py report --format html
python scripts/benchmark_comprehensive.py report --format json

# Compare runs
python scripts/compare_benchmarks.py baseline.json current.json
python scripts/compare_benchmarks.py baseline.json  # Uses latest
python scripts/compare_benchmarks.py baseline.json current.json --format html --output comparison.html

# Run unit tests
pytest tests/unit/benchmark/test_comprehensive_benchmark.py -v
```

## File Locations

```
scripts/
├── benchmark_comprehensive.py     # Main benchmark runner (32KB, ~900 lines)
├── compare_benchmarks.py           # Comparison tool (24KB, ~600 lines)
└── benchmark_tool_selection.py     # Tool selection benchmarks (existing)

configs/
└── benchmark_config.yaml           # Configuration (7.3KB, ~250 lines)

docs/performance/
├── README.md                       # User guide (7KB)
├── BENCHMARK_CI_INTEGRATION.md     # CI/CD guide (15KB)
├── BENCHMARK_SUMMARY.md            # Summary (8.4KB)
└── QUICK_REFERENCE.md              # This file

tests/unit/benchmark/
└── test_comprehensive_benchmark.py # Unit tests (17KB, ~500 lines)

examples/
└── benchmark_example.py            # Usage examples (5.9KB)
```

## Benchmark Categories

### 1. Startup Performance
- **Cold Start**: Time to first import
- **Warm Start**: Time for subsequent imports
- **Bootstrap**: Time to initialize DI container
- **Target**: <200ms cold start, <50ms warm start

### 2. Memory Usage
- **Baseline**: Peak memory after import
- **Leak Detection**: Memory growth over iterations
- **Target**: <50MB baseline, <5MB growth

### 3. Throughput
- **Tool Selection**: Selections per second
- **Cache Operations**: Ops per second
- **Target**: >1000 ops/sec

### 4. Latency
- **p50**: 50th percentile latency
- **p95**: 95th percentile latency
- **p99**: 99th percentile latency
- **Target**: p50<1ms, p95<2ms, p99<5ms

## Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| Cold Start | <200ms | 98.7% improvement |
| Warm Start | <50ms | ✅ |
| Bootstrap | <100ms | ✅ |
| Memory | <50MB | ✅ |
| Throughput | >1000 ops/sec | ✅ |
| p50 Latency | <1ms | ✅ |
| p95 Latency | <2ms | ✅ |
| p99 Latency | <5ms | ✅ |

## CI/CD Integration

### Basic Workflow

```yaml
name: Performance Benchmarks

on:
  pull_request:
    branches: [main]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Run benchmarks
        run: python scripts/benchmark_comprehensive.py run --all

      - name: Check regressions
        run: |
          LATEST=$(ls -t .benchmark_results/*.json | head -1)
          python scripts/compare_benchmarks.py baseline.json "$LATEST"
```

## Configuration

Edit `configs/benchmark_config.yaml`:

```yaml
regression_limits:
  startup_time_pct: 10   # 10% slower = regression
  memory_usage_pct: 20   # 20% more = regression
  throughput_pct: 15     # 15% slower = regression
  latency_pct: 15        # 15% higher = regression

scenarios:
  startup:
    benchmarks:
      cold_start:
        threshold_ms: 200
```

## Output

Results saved to `.benchmark_results/`:

```
.benchmark_results/
├── comprehensive_benchmark_20250118_120000.json
├── comprehensive_benchmark_20250118_130000.json
└── baseline.json
```

## Exit Codes

- **0**: All benchmarks passed
- **1**: One or more benchmarks failed or regression detected

## Environment Variables

```bash
VICTOR_PROFILE=ci                    # CI environment
VICTOR_BENCHMARK_ITERATIONS=100      # Reduce iterations
VICTOR_BENCHMARK_TIMEOUT_SECONDS=300 # Increase timeout
VICTOR_BENCHMARK_ALLOWED_FLAKINESS_PCT=5  # Allow flakiness
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Timeout | `VICTOR_BENCHMARK_TIMEOUT_SECONDS=600` |
| False positive | Adjust thresholds in config |
| Flaky tests | `VICTOR_BENCHMARK_ALLOWED_FLAKINESS_PCT=10` |
| Memory leak not detected | Run longer iterations |

## Examples

See `examples/benchmark_example.py` for:
- Running benchmarks
- Custom benchmarks
- Report generation
- Comparison
- CI/CD integration

## Documentation

- **User Guide**: `docs/performance/README.md`
- **CI/CD Guide**: `docs/performance/BENCHMARK_CI_INTEGRATION.md`
- **Summary**: `docs/performance/BENCHMARK_SUMMARY.md`
- **Examples**: `examples/benchmark_example.py`

## Support

For issues or questions:
1. Check troubleshooting section in README
2. Review CI/CD integration guide
3. See examples directory
4. Check unit tests for usage patterns

## Stats

- **Total Lines**: ~7,890
- **Code Files**: 3 scripts
- **Documentation**: 4 markdown files
- **Configuration**: 1 YAML file
- **Tests**: 1 test file (500 lines)
- **Examples**: 1 example file

---

## See Also

- [Documentation Home](../../README.md)


**Last Updated:** February 01, 2026
**Reading Time:** 2 min
