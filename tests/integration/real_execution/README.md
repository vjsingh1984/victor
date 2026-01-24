# Real Execution Integration Tests

Integration tests that execute with real LLM providers and real file operations.

## Overview

These tests perform **actual execution** without mocking:
- Real LLM API calls (Ollama with qwen3-coder-tools)
- Real file operations (Read, Edit, Write, Shell, Grep)
- Real error scenarios
- Real multi-turn conversations

## Purpose

Unlike traditional mocked integration tests, these tests catch real integration issues:
- Actual API failures and network errors
- Real tool execution behavior
- Performance characteristics
- Multi-turn conversation flows
- Cross-component integration bugs

## Requirements

### Hardware
- **Recommended**: M1 Max (or equivalent)
- **Minimum**: 8GB RAM, 4 CPU cores

### Software
```bash
# Install Ollama
brew install ollama  # macOS
# Or download from https://ollama.com

# Start Ollama service
ollama serve

# Pull recommended model
ollama pull qwen3-coder-tools:30b
# Alternative (smaller/faster):
ollama pull qwen2.5-coder:7b
```

### Verification
```bash
# Verify Ollama is running
curl http://localhost:11434/api/tags

# Verify model is available
ollama list | grep qwen
```

## Running Tests

### Run All Real Execution Tests
```bash
# From repository root
pytest tests/integration/real_execution/ -v -m "real_execution"

# With coverage
pytest tests/integration/real_execution/ -v --cov=victor --cov-report=term-missing

# Stop on first failure
pytest tests/integration/real_execution/ -v -x
```

### Run Specific Test Files
```bash
# Core tool execution
pytest tests/integration/real_execution/test_real_tool_execution.py -v

# Conversation flow
pytest tests/integration/real_execution/test_real_conversation_flow.py -v

# Error scenarios
pytest tests/integration/real_execution/test_real_error_scenarios.py -v
```

### Run Individual Tests
```bash
pytest tests/integration/real_execution/test_real_tool_execution.py::test_real_read_tool_execution -v

# With verbose output
pytest tests/integration/real_execution/test_real_tool_execution.py::test_real_read_tool_execution -vv -s
```

## Test Structure

### Test Suite 1: Core Tool Execution (6 tests)
```python
test_real_read_tool_execution()      # Read file content
test_real_edit_tool_execution()      # Edit/modify files
test_real_shell_tool_execution()     # Execute shell commands
test_real_multi_tool_execution()     # Multiple tools in sequence
test_real_grep_tool_execution()      # Search file content
test_real_write_tool_execution()     # Create new files
```

**Expected Time**: 3-5 minutes

### Test Suite 2: Conversation Flow (6 tests)
```python
test_conversation_context_preservation()        # Context across turns
test_conversation_stage_transitions()           # Stage transitions
test_conversation_error_recovery()              # Error handling
test_conversation_multi_turn_task_completion()  # Complex multi-turn
test_conversation_tool_calling_accuracy()       # Tool accuracy
test_conversation_memory_efficiency()           # Memory efficiency
```

**Expected Time**: 4-6 minutes

### Test Suite 3: Error Scenarios (8 tests)
```python
test_missing_file_error_handling()         # File not found
test_invalid_syntax_error_recovery()       # Syntax errors
test_permission_denied_error_handling()    # Permission errors
test_timeout_on_long_operation()           # Timeouts
test_empty_file_handling()                 # Empty files
test_special_characters_in_content()       # Unicode/special chars
test_very_long_response_handling()         # Long responses
test_concurrent_operations_stability()     # Stability
```

**Expected Time**: 3-4 minutes

**Total Expected Time**: 10-15 minutes (full suite)

## Expected Results

### Successful Run
```
tests/integration/real_execution/test_real_tool_execution.py::test_real_read_tool_execution PASSED
tests/integration/real_execution/test_real_tool_execution.py::test_real_edit_tool_execution PASSED
tests/integration/real_execution/test_real_tool_execution.py::test_real_shell_tool_execution PASSED
tests/integration/real_execution/test_real_tool_execution.py::test_real_multi_tool_execution PASSED
tests/integration/real_execution/test_real_tool_execution.py::test_real_grep_tool_execution PASSED
tests/integration/real_execution/test_real_tool_execution.py::test_real_write_tool_execution PASSED
tests/integration/real_execution/test_real_conversation_flow.py::test_conversation_context_preservation PASSED
tests/integration/real_execution/test_real_conversation_flow.py::test_conversation_stage_transitions PASSED
tests/integration/real_execution/test_real_conversation_flow.py::test_conversation_error_recovery PASSED
tests/integration/real_execution/test_real_conversation_flow.py::test_conversation_multi_turn_task_completion PASSED
tests/integration/real_execution/test_real_conversation_flow.py::test_conversation_tool_calling_accuracy PASSED
tests/integration/real_execution/test_real_conversation_flow.py::test_conversation_memory_efficiency PASSED
tests/integration/real_execution/test_real_error_scenarios.py::test_missing_file_error_handling PASSED
tests/integration/real_execution/test_real_error_scenarios.py::test_invalid_syntax_error_recovery PASSED
tests/integration/real_execution/test_real_error_scenarios.py::test_permission_denied_error_handling PASSED
tests/integration/real_execution/test_real_error_scenarios.py::test_timeout_on_long_operation PASSED
tests/integration/real_execution/test_real_error_scenarios.py::test_empty_file_handling PASSED
tests/integration/real_execution/test_real_error_scenarios.py::test_special_characters_in_content PASSED
tests/integration/real_execution/test_real_error_scenarios.py::test_very_long_response_handling PASSED
tests/integration/real_execution/test_real_error_scenarios.py::test_concurrent_operations_stability PASSED

=== 20 passed in 245.67s ===
```

### Skipping Tests (Ollama Not Available)
```
tests/integration/real_execution/test_real_tool_execution.py::test_real_read_tool_execution SKIPPED (Ollama not available)
...
=== 20 skipped in 0.52s ===
```

## Troubleshooting

### Issue: "Ollama not available"
**Solution**: Start Ollama service
```bash
ollama serve
```

### Issue: "No suitable Ollama model found"
**Solution**: Pull recommended model
```bash
ollama pull qwen3-coder-tools:30b
# Or smaller/faster:
ollama pull qwen2.5-coder:7b
```

### Issue: Tests timeout
**Solution**: Increase timeout in conftest.py
```python
TIMEOUT_LONG = 180  # Increase from 120 to 180
```

### Issue: Slow execution
**Solution**:
1. Use smaller model: `qwen2.5-coder:7b`
2. Close other applications
3. Check Ollama model is loaded in memory (first run after restart is slower)

### Issue: Permission errors on macOS
**Solution**: Grant Terminal full disk access
1. System Settings → Privacy & Security
2. Full Disk Access → Add Terminal

## Performance Expectations

| Hardware | Model | Time per Test | Full Suite |
|----------|-------|---------------|------------|
| M1 Max | qwen3-coder-tools:30b | 10-15s | ~4 min |
| M1 Max | qwen2.5-coder:7b | 5-8s | ~2 min |
| M1 Pro | qwen2.5-coder:7b | 8-12s | ~3 min |
| Intel i7 | qwen2.5-coder:7b | 15-20s | ~5 min |

## Contributing

### Adding New Tests
1. Follow naming convention: `test_real_*`
2. Add `@pytest.mark.real_execution` decorator
3. Include availability check (if applicable)
4. Document expected execution time
5. Update this README

### Test Template
```python
@pytest.mark.real_execution
@pytest.mark.asyncio
async def test_real_your_feature(ollama_provider, temp_workspace):
    """Test description.

    Verifies:
    - What is being tested
    - Expected outcome
    """
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.config.settings import Settings

    settings = Settings()
    settings.provider = "ollama"
    settings.model = "qwen3-coder-tools:30b"
    settings.working_dir = temp_workspace
    settings.airgapped_mode = True

    orchestrator = AgentOrchestrator(
        settings=settings,
        provider=ollama_provider,
        model=settings.model,
    )

    # Your test code here
    response = await orchestrator.chat(
        messages=[Message(role="user", content="Your test prompt")]
    )

    # Assertions
    assert response.content is not None
```

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Real Execution Tests

on: [push, pull_request]

jobs:
  real-execution:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install Ollama
        run: brew install ollama
      - name: Start Ollama
        run: ollama serve &
      - name: Pull Model
        run: ollama pull qwen2.5-coder:7b
      - name: Run Tests
        run: |
          pytest tests/integration/real_execution/ -v \
            -m "real_execution" \
            --timeout=300
```

## References

- **Test Plan**: `docs/REAL_EXECUTION_TEST_PLAN.md`
- **Ollama Provider**: `victor/providers/ollama_provider.py`
- **Orchestrator**: `victor/agent/orchestrator.py`
- **Configuration**: `victor/config/settings.py`

---

**Last Updated**: 2025-01-24
**Maintainers**: Victor Framework Team
