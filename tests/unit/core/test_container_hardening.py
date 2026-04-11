# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""TDD hardening tests for ServiceContainer.

Tests for 6 identified gaps:
1. dispose() TOCTOU race
2. _resolving thread-safety for transient services
3. freeze() after bootstrap
4. validate() for early cycle detection
5. @service decorator dependency injection
6. Async disposal support
"""

import threading
import time

import pytest

from victor.core.container import (
    Disposable,
    ServiceContainer,
    ServiceLifetime,
    ServiceNotFoundError,
    ServiceResolutionError,
)


# =========================================================================
# Helper classes
# =========================================================================


class DisposableCounter:
    """Tracks how many times dispose() is called."""

    def __init__(self):
        self.dispose_count = 0
        self._lock = threading.Lock()

    def dispose(self) -> None:
        with self._lock:
            self.dispose_count += 1


class DisposableRaiseOnDouble:
    """Raises on second dispose() call."""

    def __init__(self):
        self._disposed = False

    def dispose(self) -> None:
        if self._disposed:
            raise RuntimeError("Already disposed!")
        self._disposed = True


# =========================================================================
# Step 1: Gap 2 — dispose() TOCTOU Race
# =========================================================================


class TestDisposeRace:
    """Verify dispose() is atomic — no double-disposal under concurrency."""

    def test_concurrent_dispose_services_disposed_exactly_once(self):
        """10 threads call dispose() simultaneously. Service disposed once."""
        container = ServiceContainer()
        counter = DisposableCounter()
        container.register_instance(DisposableCounter, counter)

        barrier = threading.Barrier(10)
        errors = []

        def dispose_thread():
            try:
                barrier.wait(timeout=5)
                container.dispose()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=dispose_thread) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Unexpected errors: {errors}"
        assert counter.dispose_count == 1

    def test_concurrent_dispose_no_exceptions(self):
        """Service that raises on double-dispose should not surface errors."""
        container = ServiceContainer()
        svc = DisposableRaiseOnDouble()
        container.register_instance(DisposableRaiseOnDouble, svc)

        barrier = threading.Barrier(10)
        errors = []

        def dispose_thread():
            try:
                barrier.wait(timeout=5)
                container.dispose()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=dispose_thread) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # No errors should propagate — dispose should be atomic
        assert not errors

    def test_dispose_idempotent_single_thread(self):
        """Calling dispose() multiple times is safe, service disposed once."""
        container = ServiceContainer()
        counter = DisposableCounter()
        container.register_instance(DisposableCounter, counter)

        container.dispose()
        container.dispose()
        container.dispose()

        assert counter.dispose_count == 1
