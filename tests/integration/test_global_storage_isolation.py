"""Integration regressions for global Victor storage isolation."""

from pathlib import Path

import pytest


@pytest.mark.integration
def test_global_paths_use_isolated_integration_home(
    isolated_integration_home,
    isolated_integration_global_victor_dir,
):
    """Global Victor paths should resolve inside the integration test home."""
    from victor.config.settings import get_project_paths

    assert Path.home() == isolated_integration_home
    assert get_project_paths().global_victor_dir == isolated_integration_global_victor_dir


@pytest.mark.integration
def test_global_database_uses_isolated_integration_victor_dir(
    isolated_integration_global_victor_dir,
):
    """Default global database should not use the developer's real ~/.victor."""
    from victor.core.database import get_database, reset_database

    database = get_database()
    try:
        assert database.db_path == isolated_integration_global_victor_dir / "victor.db"
    finally:
        reset_database()
