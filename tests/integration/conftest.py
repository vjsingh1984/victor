"""Integration test fixtures.

Provides autouse fixtures for test isolation in integration tests.
"""

import pytest


@pytest.fixture(autouse=True)
def clear_rich_formatting_flag_override():
    """Clear Rich formatting feature flag override between integration tests.

    Prevents feature flag state from leaking between tests and causing
    unexpected behavior. Only clears the specific flag override, not
    the entire manager.
    """
    import sys

    # Clear override before test
    if "victor.core.feature_flags" in sys.modules:
        try:
            from victor.core.feature_flags import get_feature_flag_manager, FeatureFlag

            manager = get_feature_flag_manager()
            manager.clear_runtime_override(FeatureFlag.USE_RICH_FORMATTING)
        except (ImportError, Exception):
            pass

    yield

    # Clear override after test
    if "victor.core.feature_flags" in sys.modules:
        try:
            from victor.core.feature_flags import get_feature_flag_manager, FeatureFlag

            manager = get_feature_flag_manager()
            manager.clear_runtime_override(FeatureFlag.USE_RICH_FORMATTING)
        except (ImportError, Exception):
            pass


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
