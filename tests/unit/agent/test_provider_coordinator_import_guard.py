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

"""Import boundary guard tests for ProviderCoordinator and ProviderSwitchCoordinator.

Migration Note (2026-05-01):
- ProviderCoordinator removed from internal runtime components
- ProviderSwitchCoordinator removed from internal runtime components
- ProviderService is the canonical owner for provider operations
- Both coordinator classes remain as external compatibility surfaces only

These tests prevent regression by ensuring no new internal code imports
ProviderCoordinator or ProviderSwitchCoordinator.
"""

import ast
import os
from pathlib import Path


def _get_python_files(directory: Path) -> list[Path]:
    """Get all Python files in a directory recursively."""
    files = []
    for root, dirs, filenames in os.walk(directory):
        # Skip test files and __pycache__
        dirs[:] = [d for d in dirs if d not in ["__pycache__", "test", "tests"]]

        for filename in filenames:
            if filename.endswith(".py"):
                files.append(Path(root) / filename)
    return files


def _check_provider_coordinator_import(file_path: Path) -> bool:
    """Check if a file imports ProviderCoordinator or ProviderSwitchCoordinator.

    Returns True if the forbidden import is found.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source, filename=str(file_path))

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                # Check ProviderCoordinator imports
                if node.module == "victor.agent.provider.coordinator":
                    for alias in node.names:
                        if alias.name in ("ProviderCoordinator", "create_provider_coordinator"):
                            return True
                # Check ProviderSwitchCoordinator imports
                if node.module == "victor.agent.provider.switch_coordinator":
                    for alias in node.names:
                        if alias.name in (
                            "ProviderSwitchCoordinator",
                            "create_provider_switch_coordinator",
                        ):
                            return True
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in (
                        "victor.agent.provider.coordinator",
                        "victor.agent.provider.switch_coordinator",
                    ):
                        return True

        return False
    except (SyntaxError, UnicodeDecodeError):
        # Skip files that can't be parsed
        return False


def test_internal_code_does_not_import_provider_coordinator():
    """Test that internal production code does not import ProviderCoordinator or ProviderSwitchCoordinator.

    Both coordinators are deprecated for internal use (2026-05-01).
    ProviderService is the canonical owner for provider operations.

    This is an AST-based guard test to prevent regression.
    """
    repo_root = Path(__file__).resolve().parents[3]
    victor_dir = repo_root / "victor"

    # Exclude only exact compatibility surfaces.
    allowed_files = {
        Path("victor/agent/provider/__init__.py"),
        Path("victor/agent/provider/coordinator.py"),
        Path("victor/agent/provider/switch_coordinator.py"),
        Path("victor/agent/orchestrator_properties.py"),
        Path("victor/agent/facades/provider_facade.py"),
    }

    violations = []

    for py_file in _get_python_files(victor_dir):
        relative_path = py_file.relative_to(repo_root)
        if relative_path in allowed_files:
            continue

        if _check_provider_coordinator_import(py_file):
            violations.append(str(relative_path))

    assert not violations, (
        f"Found {len(violations)} file(s) importing deprecated provider coordinators. "
        f"Use ProviderService instead.\n"
        f"Violations: {violations}"
    )


def test_provider_runtime_does_not_create_provider_coordinator():
    """Test that provider_runtime module no longer creates deprecated coordinators.

    Migration (2026-05-01): Both ProviderCoordinator and ProviderSwitchCoordinator removed.
    """
    from victor.agent.runtime.provider_runtime import create_provider_runtime_components

    from unittest.mock import MagicMock

    manager = MagicMock()
    manager._provider_switcher = object()
    manager._health_monitor = object()
    settings = MagicMock()
    settings.max_rate_limit_retries = 3
    settings.provider_health_checks = True
    settings.feature_flags.use_provider_pooling = False

    runtime = create_provider_runtime_components(
        settings=settings,
        provider_manager=manager,
    )

    # Both coordinators should not be in runtime components
    assert not hasattr(runtime, "provider_coordinator"), (
        "ProviderCoordinator should have been removed from ProviderRuntimeComponents. "
        "Use ProviderService instead."
    )

    assert not hasattr(runtime, "provider_switch_coordinator"), (
        "ProviderSwitchCoordinator should have been removed from ProviderRuntimeComponents. "
        "Use ProviderService instead."
    )
