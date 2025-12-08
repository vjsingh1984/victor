# Victor Evaluation Harness

This module provides benchmark infrastructure for evaluating LLM coding capabilities at two distinct levels:

## Benchmark Architecture

Victor separates benchmarks by evaluation type:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         EVALUATION TYPES                                 │
├──────────────────────────────────┬──────────────────────────────────────┤
│     CODE GENERATION              │         AGENTIC TASKS                │
│  (Provider-Only Mode)            │      (With Tools & Actions)          │
├──────────────────────────────────┼──────────────────────────────────────┤
│  Benchmarks:                     │  Benchmarks:                         │
│  - HumanEval (164 Python tasks)  │  - SWE-bench (GitHub issues)         │
│  - MBPP (Basic problems)         │  - Aider Polyglot (multi-file edits) │
│  - BigCodeBench                  │  - Custom agentic scenarios          │
│                                  │                                      │
│  What it tests:                  │  What it tests:                      │
│  - Raw LLM code generation       │  - Tool selection & usage            │
│  - Function completion           │  - File editing accuracy             │
│  - Algorithm implementation      │  - Multi-step problem solving        │
│  - Code quality                  │  - Test-driven development           │
│                                  │                                      │
│  Harness:                        │  Harness:                            │
│  code_generation_harness.py      │  agentic_harness.py                  │
└──────────────────────────────────┴──────────────────────────────────────┘
```

## Why Two Separate Harnesses?

### Code Generation (HumanEval, MBPP)

These benchmarks test **pure LLM capability**. Adding tools causes:

1. **Tool hallucination**: Models try to read/write files that don't exist
2. **Unnecessary latency**: Tool overhead without accuracy improvement
3. **Measurement distortion**: Tests agent layer, not LLM capability
4. **Reduced pass rates**: Observed 20-40% with tools vs 90%+ provider-only

**Use provider-only mode for code generation benchmarks.**

### Agentic Tasks (SWE-bench, Aider Polyglot)

These benchmarks test **agent capability** with realistic tasks requiring:

1. **File navigation**: Finding relevant code across a codebase
2. **Multi-file edits**: Coordinated changes with consistency
3. **Tool selection**: Choosing appropriate tools for each step
4. **Test validation**: Ensuring changes pass existing tests

**Use the agentic harness for realistic development scenarios.**

## Usage

### Code Generation Benchmarks

```python
from victor.evaluation import (
    CodeGenerationBenchmark,
    create_code_gen_runner,
    HumanEvalRunner,
)

# Create provider runner (no tools)
runner = create_code_gen_runner(
    profile="default",
    model_override="qwen2.5-coder:14b",
)

# Run HumanEval
benchmark = CodeGenerationBenchmark(runner)
harness = get_harness()
harness.register_runner(HumanEvalRunner())

# Execute with test callback
metrics = await benchmark.run_benchmark(
    tasks=tasks,
    test_callback=test_code,
    config=config,
)

print(benchmark.generate_report())
```

### Agentic Benchmarks

```python
from victor.evaluation import (
    AgenticBenchmarkRunner,
    AgenticMetrics,
    PatchApplicationValidator,
    TestPassingValidator,
)

# Configure validators for SWE-bench
validators = [
    PatchApplicationValidator(),      # Patch applies cleanly
    TestPassingValidator(),           # Tests pass after edit
]

# Create agentic runner (with tools)
runner = AgenticBenchmarkRunner(
    orchestrator=orchestrator,
    validators=validators,
    timeout=300,
)

# Run with execution trace
result = await runner.run_task(swe_bench_task)

# Check validation results
for v_result in result.validation_results:
    print(f"{v_result.validator}: {'PASS' if v_result.passed else 'FAIL'}")
```

## Metrics

### Code Generation Metrics (CodeGenMetrics)

| Metric | Description |
|--------|-------------|
| `pass_rate` | Tasks where generated code passes all tests |
| `avg_tokens` | Average tokens per task (efficiency) |
| `avg_time` | Average time per task |
| `timeouts` | Tasks exceeding time limit |
| `errors` | Tasks with generation failures |

### Agentic Metrics (AgenticMetrics)

| Metric | Description |
|--------|-------------|
| `file_edit_accuracy` | Correctness of file modifications |
| `tool_usage_accuracy` | Appropriate tool selection |
| `patch_application_rate` | Patches that apply cleanly |
| `test_pass_rate` | Tasks where tests pass after changes |
| `avg_tool_calls` | Average tool invocations per task |

## CLI Usage

```bash
# Run HumanEval (provider mode only)
python scripts/run_full_benchmark.py \
    --profile default \
    --model qwen2.5-coder:14b \
    --output results.json

# View results
python scripts/run_full_benchmark.py --output results.json --view
```

## Backward Compatibility

The following imports are deprecated but still work:

| Old Import | New Import |
|------------|------------|
| `MultiLevelBenchmark` | `CodeGenerationBenchmark` |
| `ProviderLevelRunner` | `CodeGenerationRunner` |
| `create_provider_runner` | `create_code_gen_runner` |
| `BenchmarkLevel.PROVIDER` | (use `CodeGenerationBenchmark` directly) |

Removed modes (will raise `NotImplementedError`):
- `create_orchestrator_runner` - Use `AgenticBenchmarkRunner` instead
- `create_cli_runner` - Use `AgenticBenchmarkRunner` instead

## Module Structure

```
victor/evaluation/
├── __init__.py              # Public API exports
├── protocol.py              # Data types (BenchmarkTask, TaskResult, etc.)
├── harness.py               # Base harness infrastructure
├── benchmarks/              # Benchmark implementations
│   ├── humaneval.py         # HumanEval runner
│   ├── mbpp.py              # MBPP runner
│   └── swebench.py          # SWE-bench runner
├── code_generation_harness.py  # Provider-only benchmark runner
├── agentic_harness.py       # Tool-enabled benchmark runner
├── code_quality.py          # Lint and quality analysis
├── pass_at_k.py             # Pass@k calculation
└── analyzers.py             # Analyzer registry
```

## Key Learnings from Benchmarking

Our empirical testing revealed:

1. **Provider mode achieves 90%+ on HumanEval** with capable models (qwen2.5-coder:32b)
2. **Orchestrator/CLI modes dropped to 20-40%** due to tool hallucination
3. **Smaller models (8B)** plateau around 60% even with extended prompts
4. **Code-specialized models** (qwen2.5-coder, deepseek-coder) outperform general models
5. **Pass@k sampling** provides more accurate capability estimates than single shots

This led to the architectural decision: **separate code generation from agentic benchmarks**.
