"""Integration test fixtures.

Provides autouse fixtures for test isolation in integration tests.
"""

import sys
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


@pytest.fixture
def isolated_integration_home(tmp_path) -> Path:
    """Provide an isolated HOME directory for integration tests."""
    home_dir = tmp_path / "home"
    home_dir.mkdir(exist_ok=True)
    return home_dir


@pytest.fixture
def isolated_integration_global_victor_dir(isolated_integration_home) -> Path:
    """Provide an isolated global ~/.victor directory for integration tests."""
    victor_dir = isolated_integration_home / ".victor"
    victor_dir.mkdir(parents=True, exist_ok=True)
    return victor_dir


@pytest.fixture(autouse=True)
def isolate_integration_global_victor_storage(
    monkeypatch,
    isolated_integration_home,
    isolated_integration_global_victor_dir,
):
    """Redirect integration-test global Victor storage into a temp HOME/.victor.

    This keeps global test artifacts such as victor.db, profiles, logs, and plan
    files out of the developer's real ~/.victor directory.
    """
    import victor.config.secure_paths as secure_paths_module
    import victor.config.settings as settings_module
    from victor.core.database import reset_all_databases

    monkeypatch.setenv("HOME", str(isolated_integration_home))
    monkeypatch.setenv("USERPROFILE", str(isolated_integration_home))
    monkeypatch.setattr(
        secure_paths_module,
        "get_secure_home",
        lambda: isolated_integration_home,
    )
    monkeypatch.setattr(
        secure_paths_module,
        "get_victor_dir",
        lambda: isolated_integration_global_victor_dir,
    )
    monkeypatch.setattr(
        settings_module,
        "GLOBAL_VICTOR_DIR",
        isolated_integration_global_victor_dir,
    )

    config_loaders_module = sys.modules.get("victor.config.config_loaders")
    if config_loaders_module is not None:
        monkeypatch.setattr(
            config_loaders_module,
            "USER_CONFIG_DIR",
            isolated_integration_global_victor_dir,
        )

    reset_all_databases()
    yield
    reset_all_databases()


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
