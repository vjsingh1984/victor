# Scoring Methodology for Competitive Benchmarks

**Version**: 1.0
**Last Updated**: 2026-03-10

## Overview

This document defines the detailed scoring methodology for comparing agentic AI frameworks across benchmark tasks. The scoring is designed to be objective, reproducible, and fair to all frameworks.

## Scoring Dimensions

### 1. Task Success Rate (40% weight)

**Definition**: Percentage of tasks completed without critical errors.

**Scoring**:
```
Success Rate = (Tasks Completed Without Errors) / (Total Tasks Attempted)
Score = Success Rate * 40
```

**Critical Errors** (that count as failure):
- Code generation: Syntax errors, import errors, runtime crashes
- Tool usage: File not found, permission denied, timeout
- Analysis: Missing required output, hallucinated information
- Workflow: deadlock, crash, data loss

**Non-Critical Errors** (that don't count as failure):
- Minor style issues
- Suboptimal but functional approaches
- Warnings that don't affect execution

### 2. Output Quality (20% weight)

**Definition**: Human-rated quality of generated output on a 1-5 scale.

**Scoring**:
```
Quality Score = (Sum of Human Ratings / 5) / Number of Tasks
Score = Quality Score * 20
```

**Quality Criteria by Category**:

| Category | 5 Points | 3 Points | 1 Point |
|----------|----------|----------|----------|
| Code | Production-ready, idiomatic, well-documented | Functional but messy | Barely working, major issues |
| Analysis | Comprehensive, accurate, well-structured | Partial but accurate | Superficial or hallucinated |
| Tool Use | All operations correct, good error handling | Works but edge cases fail | Fails basic operations |
| Architecture | Complete, realistic, well-justified | Basic design, gaps | Incomplete or impractical |

### 3. Execution Speed (10% weight)

**Definition**: Relative performance compared to competitors.

**Scoring**:
```
Speed Score = max(0, 1 - (Median Latency / Slowest Median Latency))
Score = Speed Score * 10
```

**Measurement**:
- Record wall-clock time for each task
- Calculate median latency per category
- Normalize against slowest framework

**Example**:
- Victor: 15s median → Speed Score = 1 - (15/30) = 0.5
- LangGraph: 20s median → Speed Score = 1 - (20/30) = 0.33
- CrewAI: 30s median → Speed Score = 1 - (30/30) = 0.0

### 4. Resource Efficiency (15% weight)

**Definition**: Memory and CPU usage during task execution.

**Scoring**:
```
Resource Score = (1 - Normalized Resource Usage) * 15
```

**Measurement**:
- Peak memory usage (MB)
- Average CPU percentage
- Normalized to worst performer

**Formula**:
```
Memory Score = max(0, 1 - (Peak Memory / Highest Peak Memory))
CPU Score = max(0, 1 - (Avg CPU / Highest Avg CPU))
Resource Score = (Memory Score + CPU Score) / 2 * 15
```

### 5. Reliability (10% weight)

**Definition**: Error rate and stability across multiple runs.

**Scoring**:
```
Reliability Score = (1 - Error Rate) * 10
Error Rate = (Errors / (Errors + Successes))
```

**Error Types**:
- Critical: Task failure (counts as error)
- Warning: Non-fatal issues (0.5 error)
- Info: Informational (0 errors)

**Requirement**: Minimum 3 runs per task for reliability assessment.

### 6. Developer Experience (5% weight)

**Definition**: Subjective assessment of framework usability.

**Scoring**:
```
DX Score = (Setup Score + API Score + Debug Score + Docs Score) / 4 * 5
```

**Criteria**:

| Dimension | Questions |
|-----------|-----------|
| **Setup** | How many steps to install? How clear are errors? |
| **API** | Is the API intuitive? Is it consistent? |
| **Debug** | Can I see what's happening? Are logs useful? |
| **Docs** | Are examples clear? Is reference complete? |

## Overall Score Calculation

```
Overall Score = Success Score + Quality Score + Speed Score + Resource Score + Reliability Score + DX Score
Overall Score = (Success Rate * 40) + (Quality Avg / 5 * 20) + (Speed Score * 10) + (Resource Score * 15) + (Reliability Score * 10) + (DX Score * 5)
```

**Maximum**: 100 points
**Passing Threshold**: 70 points

## Statistical Significance

### Multiple Runs

Each task is run 3 times to account for variability:
- Use median score for final comparison
- Report standard deviation
- Flag high-variance results (std dev > 20% of mean)

### Confidence Intervals

For overall scores:
```
95% CI = Mean ± 1.96 * (Std Dev / √n)
where n = number of tasks
```

Frameworks with overlapping confidence intervals are considered statistically tied.

## Benchmark Execution Protocol

### 1. Environment Setup

```bash
# Create clean environment
python -m venv .venv/bench_$(date +%s)
source .venv/bench_$(date +%s)/bin/activate

# Install framework
pip install -e /path/to/framework

# Verify installation
python -c "import <framework>; print('<framework> ready')"
```

### 2. Task Execution

```python
import asyncio
import time
from pathlib import Path

async def run_task(task_id: str, framework: str):
    start = time.time()
    try:
        result = await execute_benchmark(task_id, framework)
        duration = time.time() - start
        return {
            "task_id": task_id,
            "framework": framework,
            "success": result.completed,
            "duration": duration,
            "quality": result.quality_score,
            "memory_mb": result.peak_memory,
            "error": None
        }
    except Exception as e:
        return {
            "task_id": task_id,
            "framework": framework,
            "success": False,
            "duration": time.time() - start,
            "quality": 0,
            "memory_mb": 0,
            "error": str(e)
        }

# Run 3 times
results = []
for i in range(3):
    result = await run_task(task_id, framework)
    results.append(result)
    time.sleep(5)  # Cooldown between runs
```

### 3. Result Recording

Save to `docs/benchmarking/results/{framework}/{date}/{task_id}.json`:

```json
{
  "task_id": "C1",
  "framework": "victor",
  "timestamp": "2026-03-10T12:00:00Z",
  "runs": [
    {"run": 1, "success": true, "duration_ms": 15234, "quality": 4, "memory_mb": 256, "error": null},
    {"run": 2, "success": true, "duration_ms": 16102, "quality": 4, "memory_mb": 261, "error": null},
    {"run": 3, "success": true, "duration_ms": 14987, "quality": 5, "memory_mb": 251, "error": null}
  ],
  "summary": {
    "median_duration_ms": 15234,
    "median_quality": 4,
    "median_memory_mb": 256,
    "success_rate": 1.0,
    "quality_stddev": 0.5
  }
}
```

## Fairness Considerations

### LLM Equality

- All frameworks use same LLM backend and model
- Same temperature (0.7 unless task specifies otherwise)
- Same max_tokens limits
- No caching of responses between frameworks

### Tool Equality

- Same tool versions (e.g., git 2.44, Python 3.11)
- Same file system permissions
- Same network conditions

### Timeout Equality

- Fair timeout for all frameworks (task-dependent)
- No framework gets preferential treatment
- Timeouts measured from task start, not framework overhead

## Reporting Format

### Summary Table

| Framework | Overall | Success | Quality | Speed | Resource | Reliability | DX |
|-----------|---------|---------|---------|-------|----------|-------------|-----|
| Victor | 82 | 90% | 4.2 | 8 | 12 | 9 | 8 |
| LangGraph | 75 | 85% | 4.0 | 6 | 14 | 7 | 7 |
| CrewAI | 68 | 80% | 3.8 | 5 | 16 | 6 | 6 |

### Category Breakdown

| Category | Victor | LangGraph | CrewAI | Winner |
|----------|--------|----------|--------|-------|
| Code Generation | 88 | 82 | 75 | Victor |
| Multi-Step Reasoning | 78 | 85 | 70 | LangGraph |
| Tool Usage | 85 | 70 | 68 | Victor |
| Analysis | 75 | 80 | 65 | LangGraph |
| Workflow | 82 | 75 | 72 | Victor |

### Task-by-Task Detail

Appendix with per-task scores available in `docs/benchmarking/results/detailed/`.

## Re-Benchmarking Policy

**When to Re-Benchmark**:
- Framework version update (major or minor)
- New task categories added
- Significant LLM backend changes
- Quarterly (at minimum)

**Version Tracking**:
All benchmarks must include:
- Framework version
- LLM model and version
- Benchmark suite version
- Date and time of execution
- Executor information (for reproducibility)
