# Victor AI Benchmark CI/CD Integration Guide

This guide explains how to integrate comprehensive performance benchmarks into your CI/CD pipeline for Victor AI 0.5.0.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [GitHub Actions Integration](#github-actions-integration)
- [Performance Regression Detection](#performance-regression-detection)
- [Automated Performance Monitoring](#automated-performance-monitoring)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

The Victor AI benchmark suite provides:

1. **Startup Performance**: Cold/warm start, bootstrap time
2. **Memory Usage**: Footprint tracking, leak detection
3. **Throughput**: Operations per second under load
4. **Latency**: p50, p95, p99 percentiles
5. **Cache Performance**: Hit rates, effectiveness
6. **Provider Pool**: Connection pooling, load balancing
7. **Tool Selection**: Semantic vs keyword vs hybrid
8. **Response Caching**: Exact match vs semantic similarity

### Key Performance Targets

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Startup Time | ~8000ms | <200ms | 98.7% |
| Runtime | Baseline | +30-50% | Caching |
| Memory | N/A | <50MB | Optimized |
| Throughput | Baseline | >1000 ops/s | Tool Selection |

## Quick Start

### 1. Install Dependencies

```bash
# Install benchmark dependencies
pip install pytest pytest-benchmark locust memory-profiler

# Or with dev dependencies
pip install -e ".[dev]"
```text

### 2. Run Benchmarks Locally

```bash
# Run all benchmarks
python scripts/benchmark_comprehensive.py run --all

# Run specific scenario
python scripts/benchmark_comprehensive.py run --scenario startup

# Generate report
python scripts/benchmark_comprehensive.py report --format html
```

### 3. Compare Results

```bash
# Compare with baseline
python scripts/compare_benchmarks.py baseline.json current.json

# Compare with latest run
python scripts/compare_benchmarks.py baseline.json
```text

## GitHub Actions Integration

### Basic Workflow

Create `.github/workflows/benchmarks.yml`:

```yaml
name: Performance Benchmarks

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]
  schedule:
    # Run daily at 2 AM UTC
    - cron: '0 2 * * *'

jobs:
  benchmark:
    runs-on: ubuntu-latest
    timeout-minutes: 30

    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for comparison

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'
          cache: 'pip'

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
          pip install pytest pytest-benchmark locust memory-profiler

      - name: Run benchmarks
        run: |
          python scripts/benchmark_comprehensive.py run --all
        env:
          VICTOR_PROFILE: ci

      - name: Upload benchmark results
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: .benchmark_results/*.json
          retention-days: 30

      - name: Generate report
        run: |
          python scripts/benchmark_comprehensive.py report --format markdown > BENCHMARK_REPORT.md

      - name: Upload report
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-report
          path: BENCHMARK_REPORT.md
          retention-days: 90

      - name: Compare with baseline
        run: |
          # Download baseline from main branch
          git checkout main -- .benchmark_results/baseline.json || echo "No baseline found"

          # Compare if baseline exists
          if [ -f .benchmark_results/baseline.json ]; then
            LATEST=$(ls -t .benchmark_results/comprehensive_benchmark_*.json | head -1)
            python scripts/compare_benchmarks.py .benchmark_results/baseline.json "$LATEST" --output COMPARISON.md
          fi

      - name: Upload comparison
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-comparison
          path: COMPARISON.md
          retention-days: 90

      - name: Comment PR with results
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const report = fs.readFileSync('BENCHMARK_REPORT.md', 'utf8');

            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `## 🚀 Performance Benchmark Results\n\n${report}`
            });

      - name: Check for regressions
        run: |
          LATEST=$(ls -t .benchmark_results/comprehensive_benchmark_*.json | head -1)

          # Parse results and check for failures
          FAILED=$(python -c "
          import json
          import sys
          with open('$LATEST') as f:
              data = json.load(f)
          failed = data['summary']['failed']
          print(failed)
          ")

          if [ "$FAILED" -gt 0 ]; then
            echo "❌ $FAILED benchmark(s) failed!"
            exit 1
          fi

          echo "✅ All benchmarks passed!"
```

### Advanced Workflow with Performance Budget

Create `.github/workflows/performance-budget.yml`:

```yaml
name: Performance Budget

on:
  pull_request:
    branches: [main]

jobs:
  performance-budget:
    runs-on: ubuntu-latest
    timeout-minutes: 15

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'
          cache: 'pip'

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Get baseline
        run: |
          # Download baseline from main branch
          git checkout main -- .benchmark_results/baseline.json || echo "No baseline"

      - name: Run benchmarks
        run: python scripts/benchmark_comprehensive.py run --all

      - name: Check performance budget
        run: |
          LATEST=$(ls -t .benchmark_results/comprehensive_benchmark_*.json | head -1)

          python - <<'EOF'
          import json
          import sys

          # Performance budget
          BUDGET = {
              "startup_time_ms": 200,
              "memory_mb": 50,
              "throughput_ops_per_sec": 1000,
              "p99_latency_ms": 10,
          }

          with open('$LATEST') as f:
              data = json.load(f)

          violations = []

          for result in data.get('results', []):
              for metric in result.get('metrics', []):
                  name = metric['name']
                  value = metric['value']

                  if name in BUDGET:
                      budget = BUDGET[name]

                      # Check if exceeds budget (for latency/memory)
                      if any(word in name for word in ['time', 'latency', 'memory']):
                          if value > budget:
                              violations.append(f"❌ {name}: {value:.2f} exceeds budget {budget}")
                      else:
                          # Check if below budget (for throughput)
                          if value < budget:
                              violations.append(f"❌ {name}: {value:.2f} below budget {budget}")

          if violations:
              print("Performance Budget Violations:")
              for v in violations:
                  print(v)
              sys.exit(1)
          else:
              print("✅ All metrics within performance budget")
          EOF

      - name: Update baseline
        if: github.ref == 'refs/heads/main'
        run: |
          LATEST=$(ls -t .benchmark_results/comprehensive_benchmark_*.json | head -1)
          cp "$LATEST" .benchmark_results/baseline.json

          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add .benchmark_results/baseline.json
          git commit -m "Update performance baseline [skip ci]" || echo "No changes"
          git push
```text

## Performance Regression Detection

### Regression Thresholds

Configure regression thresholds in `configs/benchmark_config.yaml`:

```yaml
regression_limits:
  # Percentage degradation allowed before flagging as regression
  startup_time_pct: 10   # 10% slower
  memory_usage_pct: 20   # 20% more memory
  throughput_pct: 15     # 15% lower throughput
  latency_pct: 15        # 15% higher latency

  # Absolute limits
  max_startup_time_ms: 500
  max_memory_mb: 100
  min_throughput_ops_per_sec: 500
  max_p99_latency_ms: 10
```

### Automated Regression Detection

The comparison tool automatically detects regressions:

```bash
# Will exit with code 1 if regressions found
python scripts/compare_benchmarks.py baseline.json current.json

# Check exit code
if [ $? -ne 0 ]; then
    echo "Performance regression detected!"
    # Handle regression (notify, fail CI, etc.)
fi
```text

### CI/CD Regression Notification

Add to your workflow:

```yaml
- name: Check for regressions
  run: |
    if [ -f .benchmark_results/baseline.json ]; then
      LATEST=$(ls -t .benchmark_results/comprehensive_benchmark_*.json | head -1)

      python scripts/compare_benchmarks.py \
        .benchmark_results/baseline.json \
        "$LATEST" \
        --output comparison.md

      # Exit code indicates regressions
      if [ $? -ne 0 ]; then
        echo "::error::Performance regression detected!"
        cat comparison.md
        exit 1
      fi
    fi
```

## Automated Performance Monitoring

### Daily Performance Tracking

```yaml
# .github/workflows/daily-benchmarks.yml
name: Daily Performance Tracking

on:
  schedule:
    - cron: '0 2 * * *'  # 2 AM UTC daily
  workflow_dispatch:

jobs:
  benchmark:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Run benchmarks
        run: python scripts/benchmark_comprehensive.py run --all

      - name: Store results
        uses: actions/upload-artifact@v4
        with:
          name: daily-benchmark-${{ github.run_number }}
          path: .benchmark_results/*.json
          retention-days: 90

      - name: Update performance database
        # Custom action to store in database/monitoring system
        run: |
          # TODO: Implement storage in your monitoring system
          # Examples: InfluxDB, Prometheus, Graphite, etc.
```text

### Performance Dashboard Integration

Export metrics to monitoring systems:

```python
# scripts/export_metrics.py
import json
from datetime import datetime

def export_to_prometheus(results_file: str):
    """Export benchmark results to Prometheus."""
    with open(results_file) as f:
        data = json.load(f)

    timestamp = datetime.fromisoformat(data['start_time']).timestamp()

    for result in data.get('results', []):
        for metric in result.get('metrics', []):
            # Create Prometheus metric
            prom_metric = f"""
            victor_benchmark_{metric['name']}{{benchmark="{result['name']}"}} {metric['value']} {int(timestamp)}
            """
            # Send to Prometheus Pushgateway
            # TODO: Implement Prometheus push
            print(prom_metric)

def export_to_influxdb(results_file: str):
    """Export benchmark results to InfluxDB."""
    # TODO: Implement InfluxDB export
    pass

if __name__ == "__main__":
    import sys
    export_to_prometheus(sys.argv[1])
```

## Best Practices

### 1. Baseline Management

- **Update baselines regularly**: Update on releases, not on every commit
- **Tag baselines**: Use git tags for version-specific baselines
- **Store baselines in repo**: Commit `.benchmark_results/baseline.json` to main branch

```bash
# Update baseline after release
cp .benchmark_results/comprehensive_benchmark_latest.json .benchmark_results/baseline.json
git add .benchmark_results/baseline.json
git commit -m "Update performance baseline for v0.5.0"
```text

### 2. Benchmark Frequency

- **Every commit**: Quick smoke tests (startup, basic operations)
- **Every PR**: Full benchmark suite
- **Daily**: Comprehensive benchmarks with load testing
- **Weekly**: Cross-platform benchmarks (macOS, Windows, Linux)

### 3. Performance Budgets

Define and enforce performance budgets:

```yaml
# .github/workflows/performance-budget.yml
performance_budget:
  startup_time_ms: 200
  memory_mb: 50
  throughput_ops_per_sec: 1000
  p95_latency_ms: 2
  p99_latency_ms: 5
```

### 4. Regression Handling

When regressions are detected:

1. **Verify**: Re-run benchmarks to rule out flakiness
2. **Investigate**: Profile to identify bottleneck
3. **Fix**: Optimize code or adjust budget if acceptable
4. **Document**: Record reason for budget change

### 5. Benchmark Stability

- Use fixed Python versions
- Pin dependencies in CI
- Run on consistent hardware
- Use CI-specific timeouts
- Allow some flakiness margin (5%)

```yaml
env:
  VICTOR_PROFILE: ci
  VICTOR_BENCHMARK_ALLOWED_FLAKINESS_PCT: 5
```text

## Troubleshooting

### Benchmarks Fail in CI

**Issue**: Benchmarks timeout or fail intermittently

**Solutions**:
```yaml
# Increase timeout
- name: Run benchmarks
  timeout-minutes: 30
  run: python scripts/benchmark_comprehensive.py run --all
  env:
    VICTOR_BENCHMARK_TIMEOUT_SECONDS: 300

# Run fewer iterations
env:
  VICTOR_BENCHMARK_ITERATIONS: 100  # Instead of 1000
```

### False Positive Regressions

**Issue**: Regression detected but code is actually faster

**Solutions**:
```yaml
# Increase significance threshold
regression_limits:
  startup_time_pct: 15  # Instead of 10%

# Allow more flakiness
env:
  VICTOR_BENCHMARK_ALLOWED_FLAKINESS_PCT: 10
```text

### Memory Leaks Not Detected

**Issue**: Memory benchmarks pass but leaks exist

**Solutions**:
```bash
# Run with longer iterations
python scripts/benchmark_comprehensive.py run --scenario memory

# Use memory profiler
python -m memory_profiler scripts/benchmark_comprehensive.py run --all
```

### Performance Degradation Over Time

**Issue**: Gradual performance decline across multiple commits

**Solutions**:
1. **Plot trends**: Store results in time-series database
2. **Detect drift**: Alert on cumulative degradation >5%
3. **Baseline refresh**: Update baseline quarterly or on major releases

## Additional Resources

- Benchmark Configuration: `configs/benchmark_config.yaml`
- Comprehensive Benchmark Script: `scripts/benchmark_comprehensive.py`
- Comparison Tool: `scripts/compare_benchmarks.py`
- [pytest-benchmark Documentation](https://pytest-benchmark.readthedocs.io/)

## Contributing

When adding new benchmarks:

1. Add to appropriate category in `benchmark_comprehensive.py`
2. Set performance targets based on measurements
3. Update configuration file with thresholds
4. Document expected performance characteristics
5. Add CI integration if critical

## License

Apache License 2.0 - See LICENSE file for details

---

## See Also

- [Documentation Home](../../README.md)


**Last Updated:** February 01, 2026
**Reading Time:** 3 minutes
