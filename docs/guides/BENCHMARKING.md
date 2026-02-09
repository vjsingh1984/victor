# Benchmarking Guide

Run industry-standard AI coding benchmarks with Victor.

## Available Benchmarks

| Benchmark | Tasks | Description |
|-----------|-------|-------------|
| swe-bench | ~2300 | Real-world GitHub issue resolution (Princeton NLP) |
| swe-bench-lite | ~300 | Curated subset of SWE-bench |
| humaneval | 164 | Code generation from docstrings (OpenAI) |
| mbpp | 974 | Mostly Basic Python Problems (Google Research) |

## Quick Start

```bash
# List available benchmarks
victor benchmark list

# Run a small test
victor benchmark run humaneval --max-tasks 5 --profile default

# Run SWE-bench with setup
victor benchmark setup swe-bench --max-tasks 10
victor benchmark run swe-bench --max-tasks 10 --profile deepseek
```text

## SWE-bench Setup

SWE-bench evaluates agents on real GitHub issues. Each task requires cloning and indexing repositories.

### Two-Phase Approach

1. **Setup phase** (optional but recommended):
   ```bash
   victor benchmark setup swe-bench --max-tasks 50
   ```
   - Clones repos to `~/.victor/swe_bench_cache/`
   - Builds code indexes for semantic search
   - Can be run once and reused

2. **Execution phase**:
   ```bash
   victor benchmark run swe-bench --max-tasks 50 --profile anthropic
```text
   - Uses cached repos if available
   - Falls back to on-the-fly setup if not

### Profile Configuration

Create a benchmark-optimized profile in `~/.victor/config.yaml`:

```yaml
profiles:
  benchmark:
    provider: anthropic
    model: claude-sonnet-4-20250514
    timeout: 600  # 10 minutes per task
    max_turns: 15

  deepseek-bench:
    provider: deepseek
    model: deepseek-coder
    timeout: 300
```

## Running Benchmarks

### Basic Usage

```bash
# Run with default profile
victor benchmark run swe-bench --max-tasks 10

# Run with specific profile and model
victor benchmark run humaneval --profile benchmark --model claude-3-opus

# Run with parallel execution
victor benchmark run mbpp --parallel 4 --max-tasks 100

# Save results to file
victor benchmark run humaneval --output results.json
```text

### Resuming Interrupted Runs

If a benchmark run is interrupted (Ctrl+C, timeout, crash), use `--resume`:

```bash
# Resume from checkpoint
victor benchmark run swe-bench --max-tasks 50 --resume

# Checkpoints are stored in ~/.victor/checkpoints/
```

### Command Options

| Option | Description |
|--------|-------------|
| `--max-tasks, -n` | Maximum number of tasks to run |
| `--model, -m` | Model to use (overrides profile) |
| `--profile, -p` | Victor profile to use |
| `--timeout, -t` | Timeout per task in seconds (default: 300) |
| `--max-turns` | Maximum conversation turns per task (default: 10) |
| `--parallel` | Number of parallel tasks (default: 1) |
| `--resume, -r` | Resume from checkpoint if interrupted |
| `--output, -o` | Output file for results (JSON) |
| `--log-level` | Logging level (DEBUG, INFO, WARNING, ERROR) |

## Interpreting Results

### Metrics

| Metric | Description |
|--------|-------------|
| Pass Rate | Percentage of tasks that passed all tests |
| Total Tasks | Number of tasks evaluated |
| Passed | Tasks with all tests passing |
| Failed | Tasks with test failures |
| Errors | Tasks that errored during execution |
| Timeouts | Tasks that exceeded timeout |
| Total Tokens | Total input + output tokens used |
| Duration | Total wall-clock time |

### Result Files

Results are automatically saved to `~/.victor/evaluations/`:

```json
{
  "benchmark": "swe-bench",
  "model": "claude-sonnet-4-20250514",
  "timestamp": "2025-01-03T12:00:00",
  "metrics": {
    "total_tasks": 50,
    "passed": 12,
    "failed": 35,
    "errors": 2,
    "timeouts": 1,
    "pass_rate": 0.24,
    "duration_seconds": 3600,
    "total_tokens": 500000
  }
}
```text

## Framework Comparison

Compare Victor against other AI coding frameworks:

```bash
# Show published results
victor benchmark compare --benchmark swe-bench

# Show capability comparison
victor benchmark capabilities

# Show leaderboard
victor benchmark leaderboard --benchmark swe-bench
```

## Common Issues

### Slow SWE-bench Setup

**Problem**: Initial setup takes too long.

**Solution**: Run setup separately before benchmarking:
```bash
victor benchmark setup swe-bench --max-tasks 100
```text

### Rate Limiting

**Problem**: API rate limits cause failures.

**Solution**: Use `--parallel 1` and increase timeout:
```bash
victor benchmark run swe-bench --parallel 1 --timeout 600
```

### Memory Issues

**Problem**: Large codebases cause memory pressure.

**Solution**: Limit parallel tasks:
```bash
victor benchmark run swe-bench --parallel 2 --max-tasks 20
```text

### Checkpoint Recovery

**Problem**: Need to restart an interrupted run.

**Solution**: Use `--resume` flag:
```bash
victor benchmark run swe-bench --resume
```

Checkpoints are stored in `~/.victor/checkpoints/` and are automatically cleared on successful completion.

## Best Practices

1. **Start small**: Run with `--max-tasks 5` first to verify setup
2. **Use setup phase**: Pre-clone repos for faster execution
3. **Monitor costs**: SWE-bench tasks can consume many tokens
4. **Save results**: Use `--output` to preserve detailed results
5. **Resume on failure**: Use `--resume` to continue interrupted runs

## Performance Tips

- Use local models (Ollama, LMStudio) to avoid API costs during testing
- Pre-cache repositories with `victor benchmark setup`
- Adjust `--timeout` based on model speed
- Limit `--max-turns` to control token usage

---

## See Also

- [Documentation Home](../../README.md)


**Last Updated:** February 01, 2026
**Reading Time:** 2 min
