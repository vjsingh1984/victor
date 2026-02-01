# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pytest fixtures for unit tests."""

import logging

import pytest


@pytest.fixture(autouse=True)
def cleanup_event_bus():
    """Clean up event bus to prevent async task leaks between tests.

    The event bus may have pending tasks from previous tests that can
    cause 'Task was destroyed but it is pending' warnings.
    """
    yield

    # Cleanup after test
    try:
        from victor.observability.event_bus import EventBus

        EventBus.reset_instance()
    except ImportError:
        pass

    # Also cleanup core event backends
    try:
        import asyncio
        from victor.core.events.backends import InMemoryEventBackend

        # Get all instances that might be running and disconnect them
        # This helps prevent "Task was destroyed but it is pending" warnings
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Can't await in sync fixture, but we can schedule the cleanup
            pass
    except (ImportError, RuntimeError):
        pass


@pytest.fixture(autouse=True)
def reset_api_keys_logger():
    """Reset the api_keys logger to ensure log propagation works for caplog.

    This fixes test isolation issues where prior tests may have configured
    the logger differently (e.g., added handlers, changed propagate flag).
    """
    # Get the logger used by api_keys module
    logger = logging.getLogger("victor.config.api_keys")

    # Store original state
    original_handlers = logger.handlers.copy()
    original_level = logger.level
    original_propagate = logger.propagate

    # Reset to clean state for test (propagate=True ensures caplog captures)
    logger.handlers.clear()
    logger.propagate = True
    logger.setLevel(logging.DEBUG)

    yield

    # Restore original state after test
    logger.handlers = original_handlers
    logger.level = original_level
    logger.propagate = original_propagate
