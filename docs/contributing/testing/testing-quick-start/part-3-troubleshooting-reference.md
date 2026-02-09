# Testing Quick Start - Part 3

**Part 3 of 3:** Troubleshooting and Quick Reference

---

## Navigation

- [Part 1: Templates & Patterns](part-1-templates-patterns-mocks.md)
- [Part 2: Coverage & Coordinators](part-2-coverage-coordinators.md)
- **[Part 3: Troubleshooting & Reference](#)** (Current)
- [**Complete Guide**](../TESTING_QUICK_START.md)

---

### Common Issues and Solutions

#### Issue: Async tests not running

```bash
# Problem: Tests hang or don't execute
# Solution: Install pytest-asyncio
pip install pytest-asyncio

# Add to pyproject.toml if needed
[tool.pytest.ini_options]
asyncio_mode = "auto"
```text

#### Issue: Mock not being called

```python
# Problem: Mock.assert_called() fails
# Solution: Use AsyncMock for async methods

# WRONG
coordinator._dependency = Mock(return_value="result")

# CORRECT
coordinator._dependency = AsyncMock(return_value="result")
```

#### Issue: Coverage report missing lines

```bash
# Problem: Coverage shows 0% or missing lines
# Solution: Ensure source is in PYTHONPATH

export PYTHONPATH=/Users/vijaysingh/code/codingagent:$PYTHONPATH
pytest --cov=victor.agent.coordinators.chat_coordinator tests/
```text

#### Issue: Tests pass locally but fail in CI

```bash
# Problem: Environment-specific test behavior
# Solution: Use environment isolation

# Mark tests that require specific environment
@pytest.mark.skipif(not is_ollama_available(), reason="Ollama not available")
async def test_with_ollama():
    pass
```

#### Issue: Slow test execution

```bash
# Problem: Tests take too long
# Solution: Run tests in parallel

pytest tests/unit/agent/coordinators/ -n auto

# Or skip slow tests
pytest -m "not slow" tests/unit/agent/coordinators/
```text

#### Issue: Fixture not found

```python
# Problem: pytestfixture not found error
# Solution: Check fixture scope and location

# Make sure fixtures are in conftest.py or test file
# For shared fixtures, use tests/conftest.py

@pytest.fixture
def shared_fixture():
    return "value"
```

### Debug Commands

```bash
# Run with verbose output
pytest tests/unit/agent/coordinators/test_chat_coordinator.py -vv

# Show print statements
pytest tests/unit/agent/coordinators/ -s

# Stop on first failure and drop into debugger
pytest tests/unit/agent/coordinators/ -x --pdb

# Show local variables on failure
pytest tests/unit/agent/coordinators/ -l

# Run specific test with maximum verbosity
pytest tests/unit/agent/coordinators/test_chat_coordinator.py::TestChatCoordinatorChat::test_chat_with_tools -vvvl -s
```text

### Getting Help

- Check [TESTING_GUIDE.md](TESTING_GUIDE.md) for detailed guidance
- Review existing test files in `tests/unit/agent/coordinators/`
- Consult pytest documentation: https://docs.pytest.org/
- Check Victor architecture docs: `docs/architecture/overview.md`

## Quick Reference Card

### Test Structure

```python
# 1. Import dependencies
import pytest
from unittest.mock import Mock, AsyncMock

# 2. Import coordinator
from victor.agent.coordinators.coordinator_name import CoordinatorName

# 3. Create fixtures
@pytest.fixture
def coordinator():
    return CoordinatorName(orchestrator=Mock())

# 4. Write tests
@pytest.mark.asyncio
async def test_something(coordinator):
    result = await coordinator.method()
    assert result is not None
```

### Common Assertions

```python
# Equality
assert result == expected

# Boolean
assert result.success is True
assert result.error is None

# Exceptions
with pytest.raises(ValueError):
    coordinator.method()

# Mock calls
mock_method.assert_called_once()
mock_method.assert_called_with(arg1, arg2)

# State
assert coordinator.state == "running"
```text

### Common Mocks

```python
# Async mock
AsyncMock(return_value="result")

# Mock with side effect
Mock(side_effect=ValueError("Error"))

# Async generator
async def stream_gen():
    yield StreamChunk(content="chunk")
```

### Running Tests

```bash
# Run all
pytest tests/unit/agent/coordinators/

# Run one file
pytest tests/unit/agent/coordinators/test_chat_coordinator.py

# Run with coverage
pytest --cov=victor.agent.coordinators.chat_coordinator tests/

# Run in parallel
pytest tests/unit/agent/coordinators/ -n 4
```text

## Next Steps

1. **Create your test file**: Copy the appropriate template
2. **Write your first test**: Start with a simple success case
3. **Add edge cases**: Test error handling and boundary conditions
4. **Check coverage**: Ensure you meet the >75% target
5. **Run tests locally**: Verify everything passes
6. **Submit PR**: Include coverage report in PR description

For detailed guidance, see [TESTING_GUIDE.md](TESTING_GUIDE.md).

---

**Last Updated:** February 01, 2026
**Reading Time:** 2 min
