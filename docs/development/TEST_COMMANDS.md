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
| Integration tests only | ~5,000 | ~5 minutes |
| Multimodal tests | 22 | **3 seconds** ✅ |

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

## Summary

The tests are NOT stuck - you're just running way too many tests!

When you see 1% progress and it's not moving, you're probably running the full test suite. Run a smaller subset instead.
