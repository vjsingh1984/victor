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

"""Tests to enforce architectural boundaries between Victor components.

These tests verify that:
1. Verticals only import from public API (not internal modules)
2. victor-sdk has zero dependencies on victor-ai
3. Protocols are imported from canonical locations
4. No circular import chains exist

Run with: pytest tests/unit/architecture/test_import_boundaries.py -v
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest


class TestVerticalImportBoundaries:
    """Verify verticals only import from public API.

    IMPORTANT: This test only applies to EXTERNAL verticals (separate packages).

    Built-in verticals (victor.verticals.contrib.*) are part of the victor-ai
    codebase and CAN import from internal modules like victor.core.verticals.*.

    External verticals (separate PyPI packages like victor-coding) MUST import from:
    - victor_sdk.* (protocol definitions, types)
    - victor.framework.* (public API)
    - victor.protocols.team.* (canonical team protocols)
    - victor.config.* (configuration)
    - victor.storage.* (storage interfaces)

    External verticals MUST NOT import from:
    - victor.agent.* (internal orchestrator/coordinators)
    - victor.core.verticals.* (internal vertical management)
    - victor.teams.protocols.* (use victor.protocols.team instead)
    """

    # This test is meant for external verticals, not built-in ones
    # Built-in verticals are part of the same codebase and can use internal modules
    BUILTIN_VERTICALS = []  # Disabled - built-in verticals can use internal modules

    # These are public APIs in victor.framework that are safe to import
    ALLOWED_INTERNAL_IMPORTS = [
        "victor.framework.",
        "victor.protocols.",
        "victor.config.",
        "victor.storage.",
        "victor.coordination.",
    ]

    # Internal modules that external verticals must not import
    FORBIDDEN_IMPORTS = [
        "victor.agent.",
        "victor.core.verticals.",
        "victor.teams.protocols",  # Use victor.protocols.team instead
        "victor.core.bootstrap.",
        "victor.core.container.",
    ]

    @pytest.mark.parametrize("vertical", BUILTIN_VERTICALS)
    def test_vertical_no_internal_imports(self, vertical: str) -> None:
        """Verify vertical doesn't import internal modules."""
        module_path = f"victor.verticals.contrib.{vertical}"
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            pytest.skip(f"Vertical {vertical} not installed or not found")

        # Get all Python files in vertical
        vertical_path = Path(module.__file__).parent
        py_files = [f for f in vertical_path.rglob("*.py") if "__pycache__" not in str(f)]

        errors = []
        for py_file in py_files:
            content = py_file.read_text()
            for forbidden in self.FORBIDDEN_IMPORTS:
                if forbidden in content:
                    # Check if it's actually used or just in a comment
                    lines = content.split("\n")
                    for i, line in enumerate(lines, 1):
                        if forbidden in line and not line.strip().startswith("#"):
                            errors.append(
                                f"{py_file.relative_to(vertical_path)}:{i} "
                                f"imports forbidden module {forbidden}"
                            )

        if errors:
            pytest.fail("\n".join(errors))

    def test_builtins_dont_import_from_external_verticals(self) -> None:
        """Verify built-in verticals don't import from external verticals."""
        # Built-in verticals should be self-contained
        # They can import from victor.framework and victor_sdk
        # But not from external packages like victor-coding (if separate)
        pass  # Placeholder for future check


class TestVictorSDKNoDependencies:
    """Verify victor-sdk has zero dependencies on victor-ai.

    victor-sdk is designed to be a standalone package with only
    typing-extensions as a runtime dependency. This ensures that:
    1. victor-sdk can be used independently
    2. victor-sdk version doesn't need to match victor-ai exactly
    3. victor-sdk can be imported by external tools without pulling in victor-ai
    """

    def test_victor_sdk_no_victor_ai_imports(self) -> None:
        """Verify victor-sdk doesn't import from victor-ai."""
        sdk_path = Path("victor-sdk/victor_sdk")
        if not sdk_path.exists():
            pytest.skip("victor-sdk not found in expected location")

        py_files = [f for f in sdk_path.rglob("*.py") if "__pycache__" not in str(f)]

        errors = []
        for py_file in py_files:
            content = py_file.read_text()

            # Check for imports from victor package (not victor_sdk)
            # Only check actual import statements, not docstrings or comments
            lines = content.split("\n")
            for i, line in enumerate(lines, 1):
                # Skip empty lines, comments, and docstrings
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or '"""' in line or "'''" in line:
                    continue

                # Check for actual import statements from victor
                if (
                    line.startswith("from victor.") or line.startswith("import victor.")
                ) and not line.startswith("from victor_sdk."):
                    errors.append(
                        f"{py_file.relative_to(sdk_path)}:{i} "
                        f"imports from victor-ai, but victor-sdk "
                        "must have zero dependencies"
                    )

        if errors:
            pytest.fail("\n".join(errors))

    def test_victor_sdk_dependencies(self) -> None:
        """Verify victor-sdk only has minimal dependencies."""
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib

        pyproject_path = Path("victor-sdk/pyproject.toml")
        if not pyproject_path.exists():
            pytest.skip("victor-sdk pyproject.toml not found")

        with open(pyproject_path, "rb") as f:
            pyproject = tomllib.load(f)

        dependencies = pyproject.get("project", {}).get("dependencies", [])

        # victor-sdk should only have typing-extensions
        allowed_deps = ["typing-extensions"]
        for dep in dependencies:
            dep_name = dep.split(">=")[0].split("==")[0].strip()
            assert dep_name in allowed_deps, f"victor-sdk has unexpected dependency: {dep}"


class TestCanonicalProtocolImports:
    """Verify protocols are imported from canonical locations.

    Canonical locations:
    - Team protocols: victor.protocols.team (NOT victor.teams.protocols)
    - Framework protocols: victor.framework.protocols (correct)
    """

    def test_teams_protocols_from_canonical_location(self) -> None:
        """victor.teams.protocols should re-export from victor.protocols.team."""
        import victor.protocols.team as canonical
        import victor.teams.protocols as teams_protocols

        # Verify all exports come from canonical location
        if hasattr(teams_protocols, "__all__"):
            for name in teams_protocols.__all__:
                canonical_attr = getattr(canonical, name, None)
                teams_attr = getattr(teams_protocols, name, None)

                # They should be the same object (re-export)
                # OR teams_attr should be None (not re-exported)
                assert (
                    canonical_attr is teams_attr or teams_attr is None
                ), f"{name} in victor.teams.protocols should come from victor.protocols.team"

    def test_no_direct_imports_from_teams_protocols(self) -> None:
        """Verify code imports from canonical location, not via re-export."""
        # Check that no files in victor/ import from victor.teams.protocols
        # (except for the re-export shim itself)
        victor_path = Path("victor")
        if not victor_path.exists():
            pytest.skip("victor directory not found")

        py_files = [f for f in victor_path.rglob("*.py") if "__pycache__" not in str(f)]

        errors = []
        for py_file in py_files:
            # Skip the re-export shim itself
            if "teams/protocols.py" in str(py_file):
                continue

            content = py_file.read_text()
            if "from victor.teams.protocols import" in content:
                # Allow it in comments
                lines = content.split("\n")
                for i, line in enumerate(lines, 1):
                    if "from victor.teams.protocols import" in line and not line.strip().startswith(
                        "#"
                    ):
                        errors.append(
                            f"{py_file.relative_to(victor_path)}:{i} "
                            "imports from victor.teams.protocols, should use "
                            "victor.protocols.team (canonical location)"
                        )

        if errors:
            pytest.fail("\n".join(errors))


class TestNoCircularImports:
    """Verify there are no circular import chains.

    This test attempts to import all major modules in different orders
    to detect circular dependencies.
    """

    def test_can_import_all_modules(self) -> None:
        """Verify all top-level modules can be imported."""
        modules_to_test = [
            "victor_sdk",
            "victor.framework",
            "victor.protocols",
            "victor.config",
            "victor.storage",
            "victor.tools",
            "victor.teams",
            "victor.workflows",
        ]

        failed = []
        for module_name in modules_to_test:
            try:
                # Fresh import by removing from sys.modules first
                if module_name in sys.modules:
                    del sys.modules[module_name]
                importlib.import_module(module_name)
            except Exception as e:
                failed.append(f"{module_name}: {e}")

        if failed:
            pytest.fail("Failed to import modules:\n" + "\n".join(failed))

    def test_reverse_import_order(self) -> None:
        """Test importing modules in reverse order."""
        modules_to_test = [
            "victor.workflows",
            "victor.teams",
            "victor.tools",
            "victor.storage",
            "victor.config",
            "victor.protocols",
            "victor.framework",
            "victor_sdk",
        ]

        failed = []
        for module_name in modules_to_test:
            try:
                importlib.import_module(module_name)
            except Exception as e:
                failed.append(f"{module_name}: {e}")

        if failed:
            pytest.fail("Failed to import in reverse order:\n" + "\n".join(failed))


class TestPublicAPIExports:
    """Verify public API modules export expected symbols.

    This ensures that imports that should work actually do work.
    """

    def test_victor_framework_public_api(self) -> None:
        """Verify victor.framework exports expected public API."""
        from victor import framework

        # These should be importable from victor.framework
        expected_exports = [
            "Agent",
            "AgentBuilder",
            "StateGraph",
            "WorkflowEngine",
            # ToolRegistry is in victor.tools, not victor.framework
        ]

        missing = []
        for export in expected_exports:
            if not hasattr(framework, export):
                missing.append(export)

        if missing:
            pytest.fail(f"victor.framework missing exports: {', '.join(missing)}")

    def test_victor_sdk_public_api(self) -> None:
        """Verify victor_sdk exports expected public API."""
        import victor_sdk

        # These should be importable from victor_sdk
        expected_exports = [
            "VerticalBase",
            "ExtensionManifest",
            "ExtensionType",
            "CapabilityProvider",
        ]

        missing = []
        for export in expected_exports:
            if not hasattr(victor_sdk, export):
                missing.append(export)

        if missing:
            pytest.fail(f"victor_sdk missing exports: {', '.join(missing)}")

    def test_victor_protocols_team_public_api(self) -> None:
        """Verify victor.protocols.team exports expected protocols."""
        from victor.protocols import team

        # These should be importable from victor.protocols.team
        expected_exports = [
            "IAgent",
            "ITeamMember",
            "ITeamCoordinator",
        ]

        missing = []
        for export in expected_exports:
            if not hasattr(team, export):
                missing.append(export)

        if missing:
            pytest.fail(f"victor.protocols.team missing exports: {', '.join(missing)}")


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])
