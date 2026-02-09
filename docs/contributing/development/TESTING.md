# Testing Guide

This guide covers testing strategies, configurations, and troubleshooting for the Victor codebase.

## Quick Start

```bash
# Run unit tests (fastest, no coverage)
make test

# Run all tests (no coverage - prevents hanging on async tests)
make test-all

# Run multimodal integration tests
make test-multimodal

# Run tests with coverage
make test-cov

# Run tests with CI configuration (includes coverage)
make test-ci
```

## Test Configurations

### pytest.ini (Default - Development)
- **Purpose**: Local development, fast iteration
- **Coverage**: Disabled by default (prevents hanging on async tests)
- **Usage**: `pytest` or `make test`

### pytest-ci.ini (CI/CD)
- **Purpose**: CI/CD pipelines, coverage reporting
- **Coverage**: Enabled with branch coverage
- **Usage**: `pytest -c pytest-ci.ini` or `make test-ci`

## Test Targets

| Target | Description | Coverage | Speed |
|--------|-------------|----------|-------|
| `make test` | Unit tests only | ❌ No | ⚡ Fast |
| `make test-all` | All tests (unit + integration) | ❌ No | 🔄 Medium |
| `make test-integration` | Integration tests only | ❌ No | 🔄 Medium |
| `make test-multimodal` | Multimodal/async tests | ❌ No | 🔄 Medium |
| `make test-cov` | Unit tests with coverage | ✅ Yes | 🐢 Slow |
| `make test-cov-all` | All tests with coverage | ✅ Yes | 🐢 Slowest |
| `make test-ci` | CI configuration (coverage) | ✅ Yes | 🐢 Slow |
| `make test-fast` | Stop on first failure | ❌ No | ⚡ Fast |

## Running Specific Tests

```bash
# Run specific test file
pytest tests/integration/agent/multimodal/test_multimodal_integration.py -v

# Run specific test class
pytest tests/integration/agent/multimodal/test_multimodal_integration.py::TestVisionAgentIntegration -v

# Run specific test
pytest
 
 
 
 
 
 
 
  tests/integration/agent/multimodal/test_multimodal_integration.py::TestVisionAgentIntegration::test_vision_with_mock_provider
  -v

# Run tests matching pattern
pytest -k "multimodal" -v

# Run tests without coverage (to avoid hanging)
pytest tests/integration/agent/multimodal/ --no-cov -v
```

## Coverage

### Enable Coverage for Specific Runs

```bash
# Run specific test with coverage
pytest tests/unit/test_file.py --cov=victor --cov-report=html

# Generate coverage report
make test-cov

# Full coverage with branch checking
make test-cov-all
```

### Coverage Reports

- **HTML**: `htmlcov/index.html` - Interactive browser report
- **Terminal**: Shown in console after test run
- **XML**: `coverage.xml` - For CI/CD integration

### Exclusions

The following are excluded from coverage:
- Test files (`*/tests/*`)
- `__init__.py` files
- CLI entry points (`victor/ui/cli.py`, `victor/ui/commands.py`)

## Async Test Handling

### The Hanging Issue

**Problem**: Coverage plugin can cause pytest to hang after async tests complete due to:
1. Coverage trying to trace async code with lingering event loops
2. Singleton reset happening while coverage is still collecting data
3. Threading issues in coverage with async test cleanup

**Solution**: Coverage is disabled by default in `pytest.ini`. Use `pytest-ci.ini` for CI or `--cov` flag explicitly when needed.

### Best Practices for Async Tests

```python
# ✅ GOOD: Clean async test
@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result == expected
    # No lingering event loops

# ❌ BAD: Async test with potential hanging
@pytest.mark.asyncio
async def test_async_with_coverage():
    # Don't rely on coverage for async integration tests
    # Run them with --no-cov flag
    result = await async_function()
    assert result == expected
```

## Test Organization

```
tests/
├── unit/              # Fast, isolated tests (no external services)
│   ├── core/          # Core functionality tests
│   ├── agents/        # Agent tests
│   └── tools/         # Tool tests
├── integration/      # Integration tests (may require mocking)
│   ├── agent/
│   │   └── multimodal/ # Async/multimodal tests
│   └── real_execution/
├── load_test/         # Load and scalability tests
└── benchmark/         # Performance benchmarks
```

## Test Fixtures

Key fixtures in `tests/conftest.py`:

- `reset_singletons`: Resets all singletons before/after each test
- `isolate_environment_variables`: Isolates from .env files and API keys
- `auto_mock_docker_for_orchestrator`: Auto-mocks Docker for orchestrator tests

## Markers

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Exclude slow tests
pytest -m "not slow"

# Run tests requiring specific dependencies
pytest -m requires_docker
```

## Troubleshooting

### Tests Hanging After Completion

**Symptom**: Tests pass but pytest hangs at 100%

**Solution 1**: Run without coverage
```bash
pytest --no-cov -v
```

**Solution 2**: Use CI config for coverage
```bash
pytest -c pytest-ci.ini -v
```

**Solution 3**: Kill hanging pytest processes
```bash
pkill -9 pytest
```

### Slow Tests

**Symptom**: Tests take too long

**Solution**: Exclude slow tests
```bash
pytest -m "not slow" -v
```

### Import Errors

**Symptom**: `ModuleNotFoundError` during tests

**Solution**: Install with dev dependencies
```bash
pip install -e ".[dev]"
```

## CI/CD Integration

### GitHub Actions

Use `pytest-ci.ini` for CI pipelines:

```yaml
- name: Run tests with coverage
  run: |
    pytest -c pytest-ci.ini -v
    # Coverage reports will be generated automatically
```

### Coverage Thresholds

Set minimum coverage in `.coveragerc` or `pyproject.toml`:

```toml
[tool.coverage.report]
fail_under = 80
```

## Performance Tips

1. **Use `--no-cov` for development**: Faster iteration
2. **Run specific tests**: `pytest tests/path/to/test.py`
3. **Use `-x` flag**: Stop on first failure
4. **Parallel execution**: `pytest -n auto` (requires pytest-xdist)
5. **Exclude slow tests**: `pytest -m "not slow"`

## See Also

- [Makefile](../Makefile) - Test targets and commands
- [pytest.ini](../pytest.ini) - Default test configuration
- [pytest-ci.ini](../pytest-ci.ini) - CI test configuration
- [conftest.py](../tests/conftest.py) - Test fixtures and configuration

---

**Last Updated:** February 01, 2026
**Reading Time:** 2 min
