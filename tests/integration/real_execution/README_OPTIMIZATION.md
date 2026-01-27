# Integration Test Optimization Guide

This guide explains how to optimize integration tests for performance using appropriate model selection and timing instrumentation.

## Quick Start

### 1. Use Timing Instrumentation

```python
from tests.integration.real_execution.test_utils import (
    TimingContext,
    assert_time_context,
    log_performance_info,
    TaskComplexity,
)

# Option 1: Use TimingContext for detailed tracking
@pytest.mark.asyncio
async def test_something(ollama_provider, ollama_model_name, temp_workspace):
    with TimingContext("test_something") as timing:
        # Setup
        timing.checkpoint("setup_complete")

        # Do work
        result = await some_operation()

        timing.checkpoint("operation_complete")

        # Log performance
        log_performance_info(
            "test_something",
            ollama_model_name,
            timing.elapsed(),
            TaskComplexity.MEDIUM
        )

# Option 2: Use assert_time_context for automatic assertions
@pytest.mark.asyncio
async def test_something(ollama_provider, ollama_model_name, temp_workspace):
    with assert_time_context(ollama_model_name, TaskComplexity.LOW, "test_something"):
        result = await some_operation()
        # Automatically asserts time is reasonable
```

### 2. Select Appropriate Models

Models are automatically selected in order of preference (fastest first). The `ollama_provider` fixture will try models in this order:

**Ultra-fast (0.5B-3B) - 3-10s:**
- `qwen2.5-coder:1.5b` (986 MB) - FASTEST
- `llama3.2:latest` (2.0 GB)
- `mistral:latest` (4.1 GB)

**Fast (7B-8B) - 5-20s:**
- `qwen2.5-coder:7b` (4.7 GB)
- `llama3.1:8b` (4.9 GB)
- `mistral:7b-instruct` (4.4 GB)

**Balanced (14B-20B) - 10-45s:**
- `deepseek-coder-v2:16b` (8.9 GB)
- `qwen25-coder-tools:14b-64K` (9.0 GB)
- `deepseek-r1:14b` (9.0 GB)

**Capable (30B+) - 20-90s:**
- `qwen3-coder-tools:30b-64K` (18 GB) - EXCELLENT for coding
- `deepseek-coder:33b` (18 GB)
- `deepseek-r1:32b` (19 GB)

## Task Complexity Guidelines

Use `TaskComplexity` to classify your tests:

### TaskComplexity.SIMPLE
**Use for:** Basic queries, error detection, simple checks
**Recommended models:** qwen2.5-coder:1.5b, llama3.2, mistral
**Expected time:** 3-10s

```python
async def test_missing_file(ollama_provider, ollama_model_name, temp_workspace):
    with assert_time_context(ollama_model_name, TaskComplexity.SIMPLE):
        response = await orchestrator.chat("Read /tmp/nonexistent.txt")
        assert "not found" in response.content.lower()
```

### TaskComplexity.LOW
**Use for:** Single tool calls, simple file reads, basic operations
**Recommended models:** qwen2.5-coder:7b, llama3.1:8b
**Expected time:** 5-20s

```python
async def test_read_file(ollama_provider, ollama_model_name, temp_workspace):
    with assert_time_context(ollama_model_name, TaskComplexity.LOW):
        response = await orchestrator.chat(f"Read {test_file}")
        assert response.content is not None
```

### TaskComplexity.MEDIUM
**Use for:** Multi-step tasks, tool orchestration, counting lines
**Recommended models:** qwen2.5-coder:7b, deepseek-coder-v2:16b
**Expected time:** 10-45s

```python
async def test_count_lines(ollama_provider, ollama_model_name, temp_workspace):
    with assert_time_context(ollama_model_name, TaskComplexity.MEDIUM):
        response = await orchestrator.chat(
            f"Read {large_file} and count how many lines it has."
        )
        assert "100" in response.content
```

### TaskComplexity.HIGH
**Use for:** Complex reasoning, multi-file operations, refactoring
**Recommended models:** qwen3-coder-tools:30b, deepseek-coder:33b
**Expected time:** 20-90s

```python
async def test_refactor_code(ollama_provider, ollama_model_name, temp_workspace):
    with assert_time_context(ollama_model_name, TaskComplexity.HIGH):
        response = await orchestrator.chat(
            f"Refactor the code in {project_dir} to use async/await."
        )
        assert "async def" in response.content
```

### TaskComplexity.MAXIMUM
**Use for:** Maximum capability required (rare)
**Recommended models:** deepseek-r1:70b, llama3.3:70b
**Expected time:** 40-180s

```python
async def test_complex_architecture(ollama_provider, ollama_model_name, temp_workspace):
    with assert_time_context(ollama_model_name, TaskComplexity.MAXIMUM):
        response = await orchestrator.chat(
            f"Analyze and redesign the entire architecture in {project_dir}."
        )
        assert response.content is not None
```

## Installing Faster Models

To get faster tests, install these models:

### Ultra-fast (600MB - 2GB)
```bash
ollama pull qwen2.5-coder:1.5b  # 986 MB - FASTEST
ollama pull llama3.2:latest      # 2.0 GB
ollama pull mistral:latest       # 4.1 GB
```

### Fast (4GB - 5GB)
```bash
ollama pull qwen2.5-coder:7b    # 4.7 GB - Great for coding
ollama pull llama3.1:8b         # 4.9 GB
ollama pull mistral:7b-instruct  # 4.4 GB
```

### Balanced (8GB - 9GB)
```bash
ollama pull deepseek-coder-v2:16b         # 8.9 GB - Fast for coding
ollama pull qwen25-coder-tools:14b-64K    # 9.0 GB - Tool-optimized
ollama pull deepseek-r1:14b               # 9.0 GB - Reasoning
```

### Capable (18GB - 20GB) - For complex tasks
```bash
ollama pull qwen3-coder-tools:30b-64K     # 18 GB - EXCELLENT
ollama pull deepseek-coder:33b            # 18 GB
ollama pull deepseek-r1:32b               # 19 GB
```

## Performance Debugging

If tests are slow, use TimingContext to identify bottlenecks:

```python
async def test_slow_operation(ollama_provider, ollama_model_name, temp_workspace):
    with TimingContext("test_slow_operation") as timing:
        # Phase 1: Setup
        settings = Settings()
        orchestrator = AgentOrchestrator(settings=settings, provider=ollama_provider)
        timing.checkpoint("setup")

        # Phase 2: File creation
        test_file = Path(temp_workspace) / "test.py"
        test_file.write_text("print('hello')")
        timing.checkpoint("file_creation")

        # Phase 3: LLM call
        response = await orchestrator.chat(f"Read {test_file}")
        timing.checkpoint("llm_call")

        # Output will show:
        # ⏱️  [CHECKPOINT] setup: 0.52s
        # ⏱️  [CHECKPOINT] file_creation: 0.01s
        # ⏱️  [CHECKPOINT] llm_call: 15.34s  <- Bottleneck here!
```

## Expected Performance by Model Size

| Model Size | Examples | Simple Task | Low Task | Medium Task | High Task |
|------------|----------|-------------|----------|-------------|-----------|
| Ultra-fast (0.5B-3B) | qwen2.5:1.5b, llama3.2 | 3-5s | 5-8s | 7-10s | 10-15s |
| Fast (7B-8B) | qwen2.5-coder:7b, llama3.1:8b | 5-8s | 8-12s | 12-20s | 20-30s |
| Balanced (14B-20B) | qwen2.5-coder:14b, deepseek-coder-v2:16b | 10-15s | 15-22s | 22-45s | 45-67s |
| Capable (30B+) | qwen3-coder:30b, deepseek-coder:33b | 20-30s | 30-45s | 45-90s | 90-135s |
| Large (70B+) | deepseek-r1:70b, llama3.3:70b | 40-60s | 60-90s | 90-180s | 180-360s |

## CI/CD Optimization

For CI/CD, use only ultra-fast and fast models:

```bash
# Install only fast models
ollama pull qwen2.5-coder:1.5b
ollama pull qwen2.5-coder:7b

# Run tests with timeout
pytest tests/integration/real_execution/ -v --timeout=300
```

## Troubleshooting

### Test is timing out

1. **Check model size**: `ollama list` to see which model is being used
2. **Use faster model**: Install qwen2.5-coder:1.5b for simple tasks
3. **Adjust task complexity**: Use `TaskComplexity.SIMPLE` instead of `MEDIUM`
4. **Increase timeout**: `@pytest.mark.timeout(600)` for 10 minutes

### Test is slow but passing

1. **Add timing**: Use `TimingContext` to identify bottleneck
2. **Check model**: Ultra-fast models should complete in < 10s
3. **Optimize test**: Reduce file size, simplify prompts

### Model not found

1. **List available models**: `ollama list` or `curl http://localhost:1234/v1/models`
2. **Install model**: `ollama pull <model_name>`
3. **Check fixture**: Ensure model is in PROVIDER_MODELS in conftest.py

## Examples

See `test_real_error_scenarios.py` for complete examples of optimized tests.
