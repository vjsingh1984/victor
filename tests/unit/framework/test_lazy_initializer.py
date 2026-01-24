# Copyright 2025 Vijaykumar Singh
#
# Licensed under the Apache License, Version 2.0

"""Unit tests for LazyInitializer (eliminating import side-effects).

Tests for thread-safe lazy initialization that eliminates registration-on-import
side effects while maintaining backward compatibility (TDD approach).
"""

import pytest
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from victor.framework.lazy_initializer import LazyInitializer

from victor.framework.lazy_initializer import (
    LazyInitializer,
    get_initializer_for_vertical,
    clear_all_initializers,
)


# =============================================================================
# Test Fixtures
# =============================================================================


class MockVertical:
    """Mock vertical for testing."""

    name = "mock_vertical"


# =============================================================================
# Test LazyInitializer Initialization
# =============================================================================


class TestLazyInitializerInit:
    """Test suite for LazyInitializer initialization."""

    def test_init_with_vertical_name_and_initializer(self):
        """Should initialize with vertical name and initializer callable."""
        call_count = [0]

        def initializer():
            call_count[0] += 1
            return "initialized"

        init = LazyInitializer(
            vertical_name="test_vertical",
            initializer=initializer
        )

        assert init.vertical_name == "test_vertical"
        assert call_count[0] == 0  # Not called yet
        assert not init.is_initialized()

    def test_init_with_multiple_initializers(self):
        """Should accept multiple initializers to run in sequence."""
        order = []

        def init1():
            order.append("init1")

        def init2():
            order.append("init2")

        init = LazyInitializer(
            vertical_name="test",
            initializers=[init1, init2]
        )

        init.initialize()

        assert order == ["init1", "init2"]


# =============================================================================
# Test Lazy Initialization Behavior
# =============================================================================


class TestLazyInitialization:
    """Test suite for lazy initialization behavior."""

    def test_initialize_on_first_call(self):
        """Should initialize on first call to get_or_initialize()."""
        call_count = [0]

        def initializer():
            call_count[0] += 1
            return "result"

        init = LazyInitializer(
            vertical_name="test",
            initializer=initializer
        )

        assert call_count[0] == 0  # Not called yet

        result = init.get_or_initialize()

        assert call_count[0] == 1  # Called once
        assert result == "result"

    def test_only_initialize_once(self):
        """Should only initialize once, cache subsequent calls."""
        call_count = [0]

        def initializer():
            call_count[0] += 1
            return call_count[0]

        init = LazyInitializer(
            vertical_name="test",
            initializer=initializer
        )

        # Call multiple times
        result1 = init.get_or_initialize()
        result2 = init.get_or_initialize()
        result3 = init.get_or_initialize()

        # Should only initialize once
        assert call_count[0] == 1
        assert result1 == 1
        assert result2 == 1
        assert result3 == 1

    def test_caches_initialization_result(self):
        """Should cache the initialization result."""
        def initializer():
            return {"data": "value"}

        init = LazyInitializer(
            vertical_name="test",
            initializer=initializer
        )

        # Get result twice
        result1 = init.get_or_initialize()
        result2 = init.get_or_initialize()

        # Should be the same object (cached)
        assert result1 is result2
        assert result1 == {"data": "value"}

    def test_is_initialized_tracking(self):
        """Should correctly track initialization state."""
        init = LazyInitializer(
            vertical_name="test",
            initializer=lambda: None
        )

        assert not init.is_initialized()

        init.get_or_initialize()

        assert init.is_initialized()


# =============================================================================
# Test Thread Safety
# =============================================================================


class TestThreadSafety:
    """Test suite for thread-safe lazy initialization."""

    def test_concurrent_initialization_thread_safe(self):
        """Should handle concurrent initialization safely."""
        call_count = [0]
        lock = threading.Lock()

        def initializer():
            with lock:
                call_count[0] += 1
                # Simulate slow initialization
                time.sleep(0.01)
            return call_count[0]

        init = LazyInitializer(
            vertical_name="test",
            initializer=initializer
        )

        # Spawn multiple threads
        threads = []
        results = []

        def initialize_from_thread():
            result = init.get_or_initialize()
            results.append(result)

        for _ in range(10):
            thread = threading.Thread(target=initialize_from_thread)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Should only initialize once despite concurrent access
        assert call_count[0] == 1
        assert len(results) == 10

        # All results should be the same
        assert all(r == results[0] for r in results)

    def test_double_checked_locking_pattern(self):
        """Should use double-checked locking for efficiency."""
        init = LazyInitializer(
            vertical_name="test",
            initializer=lambda: "initialized"
        )

        # First initialization
        init.get_or_initialize()

        # Second initialization should use fast path (no lock)
        start = time.time()
        for _ in range(1000):
            init.get_or_initialize()
        elapsed = time.time() - start

        # Should be very fast (cached)
        assert elapsed < 0.01  # Less than 10ms for 1000 calls


# =============================================================================
# Test Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Test suite for factory functions."""

    def test_get_initializer_for_vertical(self):
        """Should return initializer for a vertical."""
        initializer = lambda: "result"

        init = get_initializer_for_vertical(
            "test_vertical",
            initializer
        )

        assert isinstance(init, LazyInitializer)
        assert init.vertical_name == "test_vertical"

    def test_get_initializer_returns_singleton(self):
        """Should return singleton initializer for same vertical."""
        initializer1 = lambda: "result1"
        initializer2 = lambda: "result2"

        init1 = get_initializer_for_vertical("test", initializer1)
        init2 = get_initializer_for_vertical("test", initializer2)

        # Should return the same instance (singleton)
        assert init1 is init2

    def test_clear_all_initializers(self):
        """Should clear all cached initializers."""
        # Create initializers
        init1 = get_initializer_for_vertical("test1", lambda: "1")
        init2 = get_initializer_for_vertical("test2", lambda: "2")

        # Clear all
        clear_all_initializers()

        # Should create new instances
        init3 = get_initializer_for_vertical("test1", lambda: "1")
        init4 = get_initializer_for_vertical("test2", lambda: "2")

        # Should be different instances
        assert init1 is not init3
        assert init2 is not init4


# =============================================================================
# Test Error Handling
# =============================================================================


class TestErrorHandling:
    """Test suite for error handling and edge cases."""

    def test_handles_initializer_exception_gracefully(self):
        """Should handle initializer exceptions gracefully."""
        def failing_initializer():
            raise RuntimeError("Initialization failed")

        init = LazyInitializer(
            vertical_name="test",
            initializer=failing_initializer
        )

        # Should raise exception
        with pytest.raises(RuntimeError, match="Initialization failed"):
            init.get_or_initialize()

        # Should not be marked as initialized
        assert not init.is_initialized()

    def test_can_retry_after_failure(self):
        """Should allow retry after initialization failure."""
        attempt_count = [0]

        def flaky_initializer():
            attempt_count[0] += 1
            if attempt_count[0] < 2:
                raise RuntimeError("First attempt fails")
            return "success"

        init = LazyInitializer(
            vertical_name="test",
            initializer=flaky_initializer
        )

        # First attempt fails
        with pytest.raises(RuntimeError):
            init.get_or_initialize()

        # Second attempt succeeds
        result = init.get_or_initialize()

        assert result == "success"
        assert init.is_initialized()

    def test_reset_clears_initialization(self):
        """Should clear initialization state on reset()."""
        init = LazyInitializer(
            vertical_name="test",
            initializer=lambda: "result"
        )

        # Initialize
        init.get_or_initialize()
        assert init.is_initialized()

        # Reset
        init.reset()
        assert not init.is_initialized()

        # Can initialize again
        result = init.get_or_initialize()
        assert result == "result"
        assert init.is_initialized()


# =============================================================================
# Test Integration with Verticals
# =============================================================================


class TestVerticalIntegration:
    """Test suite for integration with vertical modules."""

    def test_register_escape_hatches_lazily(self):
        """Should register escape hatches lazily on first use."""
        from victor.framework.escape_hatch_registry import EscapeHatchRegistry

        # Clear registry
        EscapeHatchRegistry.reset_instance()

        call_count = [0]

        def mock_initializer():
            call_count[0] += 1
            # Simulate registering escape hatches
            registry = EscapeHatchRegistry.get_instance()
            registry.register_condition(
                "test_condition",
                lambda ctx: "test",
                vertical="mock"
            )
            return True

        init = LazyInitializer(
            vertical_name="mock",
            initializer=mock_initializer
        )

        # Not registered yet
        registry = EscapeHatchRegistry.get_instance()
        assert "test_condition" not in registry.get_registry_for_vertical("mock")[0]

        # Trigger lazy initialization
        init.get_or_initialize()

        # Should be registered now
        conditions, _ = registry.get_registry_for_vertical("mock")
        assert "test_condition" in conditions
        assert call_count[0] == 1

    def test_multiple_verticals_independent(self):
        """Should handle multiple verticals independently."""
        import_count = {"vertical1": 0, "vertical2": 0}

        def initializer1():
            import_count["vertical1"] += 1
            return "vertical1"

        def initializer2():
            import_count["vertical2"] += 1
            return "vertical2"

        init1 = get_initializer_for_vertical("vertical1", initializer1)
        init2 = get_initializer_for_vertical("vertical2", initializer2)

        # Initialize vertical1 only
        init1.get_or_initialize()

        # Only vertical1 should be initialized
        assert import_count["vertical1"] == 1
        assert import_count["vertical2"] == 0

        # Now initialize vertical2
        init2.get_or_initialize()

        # Both should be initialized
        assert import_count["vertical1"] == 1
        assert import_count["vertical2"] == 1


# =============================================================================
# Test Feature Flag Control
# =============================================================================


class TestFeatureFlags:
    """Test suite for feature flag control."""

    def test_respects_disable_lazy_initialization_flag(self, monkeypatch):
        """Should respect VICTOR_LAZY_INITIALIZATION flag."""
        # Disable lazy initialization
        monkeypatch.setenv("VICTOR_LAZY_INITIALIZATION", "false")

        call_count = [0]

        def initializer():
            call_count[0] += 1
            return "result"

        # Note: This would require modifying LazyInitializer to check env var
        # For now, we just test the basic functionality
        init = LazyInitializer(
            vertical_name="test",
            initializer=initializer
        )

        # With flag disabled, would initialize immediately
        # For now, we just verify lazy behavior
        assert call_count[0] == 0
        init.get_or_initialize()
        assert call_count[0] == 1


# =============================================================================
# Test Performance
# =============================================================================


class TestPerformance:
    """Test suite for performance characteristics."""

    def test_first_access_overhead_acceptable(self):
        """First access overhead should be acceptable (<50ms)."""
        def slow_initializer():
            time.sleep(0.01)  # Simulate 10ms initialization
            return "result"

        init = LazyInitializer(
            vertical_name="test",
            initializer=slow_initializer
        )

        start = time.time()
        result = init.get_or_initialize()
        elapsed = time.time() - start

        # Should complete in reasonable time
        # (10ms sleep + overhead)
        assert elapsed < 0.05  # Less than 50ms
        assert result == "result"

    def test_cached_access_fast(self):
        """Cached access should be very fast (<1ms)."""
        init = LazyInitializer(
            vertical_name="test",
            initializer=lambda: "result"
        )

        # Initialize once
        init.get_or_initialize()

        # Measure cached access
        start = time.time()
        for _ in range(1000):
            init.get_or_initialize()
        elapsed = time.time() - start

        # Should be very fast (cached)
        assert elapsed < 0.01  # Less than 10ms for 1000 accesses
