# Quick Test Commands Reference

## Problem: Running `pytest` with no arguments is TOO SLOW

When you run just `pytest`, it tries to run **30,047 tests** which takes a very long time!

## Solution: Run Specific Test Suites

```bash
# Run only integration tests (recommended)
pytest tests/integration/ -v

# Run specific test file
pytest tests/integration/agent/multimodal/test_multimodal_integration.py -v

# Run specific test
pytest tests/integration/agent/multimodal/test_multimodal_integration.py::TestVisionAgentIntegration::test_vision_with_mock_provider -v

# Run only unit tests (fast)
pytest tests/unit/ -v

# Run with marker filter
pytest -m "not slow" -v
```

## Test Counts

| Suite | Count | Time |
|-------|--------|------|
| **All tests** (just `pytest`) | **30,047** | ~30+ minutes |
| Unit tests only | ~2,000 | ~2 minutes |
| Integration tests only | ~2,662 | ~3-4 minutes |
| Multimodal tests | 22 | **3 seconds** ✅ |
| Agent tests | ~800 | ~1 minute |
| Framework tests | ~500 | ~45 seconds |

## Make Targets (Recommended)

```bash
make test           # Unit tests (fast)
make test-all       # All tests (slow)
make test-multimodal # Multimodal tests (fast)
make test-fast      # Stop on first failure
```

## What NOT to Do

```bash
# ❌ DON'T DO THIS - runs 30k tests!
pytest

# ✅ DO THIS INSTEAD - runs specific tests
pytest tests/integration/agent/multimodal/
```

## Test Timeouts

Tests have a **60 second default timeout** to prevent hanging. If a test hangs, pytest-timeout will kill it:

```bash
# Default timeout (60s per test)
pytest tests/integration/ -v

# Custom timeout
pytest tests/integration/ -v --timeout=30

# Disable timeout for specific test
@pytest.mark.timeout(0)  # or @pytest.mark.timeout(300) for 5 min
def test_long_running():
    ...
```

### If Tests Appear Stuck

1. **Wait for timeout** - The test will fail after 60 seconds
2. **Check for background tasks** - Some tests create `asyncio.create_task()` without cleanup
3. **Run with verbose output** - `pytest -v --tb=short` to see exactly where it hangs
4. **Use smaller timeout** - `pytest --timeout=10` to fail faster

## Summary

The tests are NOT stuck - you're just running way too many tests!

When you see 1% progress and it's not moving, you're probably running the full test suite. Run a smaller subset instead.

## Understanding "1% Progress" with Integration Tests

When running `pytest tests/integration/`, you'll see progress at 1% for a while because:

- **2,662 tests** are being executed
- 1% = ~26 tests completed
- Average test time: ~0.06 seconds
- Total expected time: **~3-4 minutes**

This is **normal behavior** - the tests are running, not hung.

### Running Integration Test Subsets

Instead of running all integration tests, run specific subsets:

```bash
# Agent integration tests (~800 tests, ~1 minute)
pytest tests/integration/agent/ -v

# Framework integration tests (~500 tests, ~45 seconds)
pytest tests/integration/framework/ -v

# Workflow integration tests (~400 tests, ~30 seconds)
pytest tests/integration/workflows/ -v

# Provider integration tests (~200 tests, ~15 seconds)
pytest tests/integration/providers/ -v

# Vertical integration tests (~300 tests, ~25 seconds)
pytest tests/integration/verticals/ -v

# Multiple specific directories
pytest tests/integration/agent/multimodal/ tests/integration/agent/improvement/ -v
```

### Progress Indicators

Watch for these patterns to know tests are running:

```
PASSED [  0%]  ← Test 1-26 completed (1%)
PASSED [  1%]  ← Test 27-53 completed (2%)
PASSED [  2%]  ← Test 54-79 completed (3%)
...
```

If you see **"PASSED"** or **"FAILED"** messages, tests are running normally.
