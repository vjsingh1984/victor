"""Integration test fixtures.

Provides autouse fixtures for test isolation in integration tests.
"""

import pytest


@pytest.fixture(autouse=True)
def reset_service_container():
    """Reset DI service container between integration tests.

    Prevents ServiceAlreadyRegisteredError from container state
    leaking between tests.
    """
    yield
    import sys

    if "victor.core.container" in sys.modules:
        try:
            from victor.core.container import reset_container

            reset_container()
        except ImportError:
            pass
