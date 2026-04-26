"""Integration test fixtures.

Provides autouse fixtures for test isolation in integration tests.
"""

from pathlib import Path
from unittest.mock import PropertyMock, patch

import pytest


@pytest.fixture
def isolated_integration_project_victor_dir(tmp_path) -> Path:
    """Provide an isolated project-local .victor directory for integration tests."""
    temp_victor = tmp_path / ".victor"
    temp_victor.mkdir(exist_ok=True)
    return temp_victor


@pytest.fixture(autouse=True)
def isolate_integration_project_victor_storage(isolated_integration_project_victor_dir):
    """Redirect integration-test project-local persistence into temp .victor storage.

    Integration tests should exercise the real SQLite/filesystem behavior without
    writing into the repository's shared .victor/ directory.
    """
    from victor.config.settings import ProjectPaths

    temp_project_db = isolated_integration_project_victor_dir / "project.db"

    with (
        patch.object(
            ProjectPaths,
            "project_victor_dir",
            new_callable=PropertyMock,
            return_value=isolated_integration_project_victor_dir,
        ),
        patch.object(
            ProjectPaths,
            "project_db",
            new_callable=PropertyMock,
            return_value=temp_project_db,
        ),
    ):
        yield


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
