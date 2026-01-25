# Team Node Performance Benchmarks - Quick Start

This guide helps you quickly run and interpret team node performance benchmarks.

## Quick Start

### 1. Install Dependencies

```bash
pip install pytest-benchmark
```

### 2. Run All Benchmarks

```bash
# Run all team node benchmarks
python scripts/benchmark_team_nodes.py run --all

# Or use pytest directly
pytest tests/performance/test_team_node_performance.py -v
```

### 3. Generate Report

```bash
# Generate markdown report
python scripts/benchmark_team_nodes.py report --format markdown

# Generate JSON for analysis
python scripts/benchmark_team_nodes.py report --format json > results.json
```

## Running Specific Benchmarks

### Formation Comparison

Compare performance of all 5 team formations:

```bash
python scripts/benchmark_team_nodes.py run --group formations
```

**Output example:**
```
test_formation_performance[sequential] ✅ 45.23ms
test_formation_performance[parallel] ✅ 18.12ms
test_formation_performance[pipeline] ✅ 38.45ms
test_formation_performance[hierarchical] ✅ 42.67ms
test_formation_performance[consensus] ✅ 98.34ms
```

### Team Size Scaling

Test how performance scales with team size:

```bash
python scripts/benchmark_team_nodes.py run --group size
```

### Tool Budget Impact

Measure performance with different tool budgets:

```bash
python scripts/benchmark_team_nodes.py run --group budget
```

## Understanding Results

### Key Metrics

- **Avg Latency:** Average execution time
- **Min/Max:** Best/worst case performance
- **Throughput:** Teams executed per second
- **Memory:** Memory used during execution

### Performance Categories

**Excellent:** < 50ms for 3-5 members
**Good:** 50-200ms for 3-5 members
**Acceptable:** 200-500ms
**Needs Optimization:** > 500ms

## Example Usage in CI/CD

```yaml
# .github/workflows/benchmarks.yml
name: Performance Benchmarks

on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
          pip install pytest-benchmark

      - name: Run benchmarks
        run: |
          python scripts/benchmark_team_nodes.py run --all

      - name: Generate report
        run: |
          python scripts/benchmark_team_nodes.py report --format markdown > benchmark_results.md

      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: benchmark-results
          files: benchmark_results.md
```

## Troubleshooting

### Benchmarks fail to run

**Issue:** Missing dependencies
```bash
# Install pytest-benchmark
pip install pytest-benchmark
```

### Results seem slow

**Issue:** Running on slower hardware or debug mode
```bash
# Ensure you're not in debug mode
export VICTOR_PROFILE=production

# Run with fewer iterations for faster testing
pytest tests/performance/test_team_node_performance.py -k formations --benchmark-min-rounds=1
```

### Out of memory errors

**Issue:** Too many concurrent benchmarks
```bash
# Run one benchmark group at a time
python scripts/benchmark_team_nodes.py run --group formations
```

## Advanced Usage

### Custom Benchmark

```python
# tests/performance/test_my_benchmark.py
import pytest
from victor.teams import UnifiedTeamCoordinator
from victor.teams.types import TeamFormation

@pytest.mark.benchmark(group="custom")
def test_my_scenario(benchmark):
    """Benchmark my custom team scenario."""

    async def run_team():
        coordinator = UnifiedTeamCoordinator(lightweight_mode=True)

        # Add members
        for i in range(3):
            from tests.performance.test_team_node_performance import MockTeamMember
            member = MockTeamMember(f"member_{i}", "assistant")
            coordinator.add_member(member)

        coordinator.set_formation(TeamFormation.PARALLEL)

        return await coordinator.execute_task(
            task="My custom task",
            context={"team_name": "custom_test"}
        )

    result = benchmark(asyncio.run, run_team())
    assert result["success"]
```

### Comparing Performance Over Time

```bash
# Run baseline benchmarks
python scripts/benchmark_team_nodes.py run --all

# Make code changes

# Run new benchmarks
python scripts/benchmark_team_nodes.py run --all

# Compare results
python scripts/benchmark_team_nodes.py list
python scripts/benchmark_team_nodes.py compare \
  .benchmark_results/team_nodes_BASELINE.json \
  .benchmark_results/team_nodes_LATEST.json
```

## Best Practices

1. **Run on consistent hardware** - Results vary by machine
2. **Use same Python version** - Performance changes between versions
3. **Disable debug mode** - Use `VICTOR_PROFILE=production`
4. **Run multiple iterations** - pytest-benchmark handles this automatically
5. **Save baseline results** - Compare against to detect regressions

## Further Reading

- [Full Performance Guide](./team_node_performance.md)
- [Team Node Usage](./team_nodes.md)
- [Workflow Documentation](../user-guide/workflows.md)
