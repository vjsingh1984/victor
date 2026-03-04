# Benchmark Spec And Scoring Rubric (M1)

This document defines the M1 benchmark task set, scoring rubric, and
reproducibility checklist for 90-day execution tracking.

## M1 Task Set (24 Tasks)

| Benchmark | Tasks | Purpose |
|---|---:|---|
| HumanEval | 8 | Function-level code generation quality |
| MBPP | 8 | Practical Python problem solving |
| SWE-bench Lite | 4 | Multi-file issue resolution under constraints |
| SWE-bench (full) | 4 | Real-world repo task handling |

Selection rules:
- Freeze exact task IDs per run and store in run metadata.
- Keep the same task IDs across model/framework comparisons.
- Any task-set change requires a new benchmark version tag.

## Scoring Rubric (100 Points)

| Dimension | Weight | Measure |
|---|---:|---|
| Correctness | 45 | Pass rate across selected tasks |
| Reliability | 20 | Error + timeout penalty |
| Cost Efficiency | 15 | Tokens per passed task |
| Latency | 10 | Median task completion time |
| Reproducibility Hygiene | 10 | Checklist completeness |

Scoring details:
- Correctness: linear score from pass rate.
- Reliability: starts at 20, subtracts for failures/timeouts.
- Cost efficiency: normalized against baseline model run.
- Latency: normalized against baseline median runtime.
- Reproducibility hygiene: binary per checklist item.

## Reproducibility Checklist

For every reported run, capture all of the following:

- Git commit SHA
- Benchmark command and full flags
- Provider/model/profile
- Task IDs and benchmark version tag
- Start/end timestamp (UTC)
- Machine/runtime metadata (CPU, RAM, OS, Python)
- Timeout/max-turn settings
- Raw result JSON path and generated report path
- Any retries/resume events
- Notes on environment anomalies

## Reporting Format

Minimum report fields:
- `benchmark_name`
- `task_set_version`
- `model/provider/profile`
- `pass_rate`, `errors`, `timeouts`
- `tokens_total`, `tokens_per_pass`
- `duration_seconds`, `median_task_seconds`
- `rubric_score_total`
- `checklist_complete` (true/false)

## Owner

Owner role: Platform Lead  
Tracker: `[90D][E6][M1] Finalize benchmark spec and scoring rubric` (#51)
