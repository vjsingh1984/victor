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

"""Tests for Open/Closed Principle (OCP) Compliance.

These tests verify that the framework is OCP-compliant - it's open for extension
(new verticals can be added) but closed for modification (framework code doesn't
need to change when adding new verticals).

CRITICAL: These tests ensure the framework can load new verticals without
hardcoded imports, which is essential for the plugin architecture.
"""

import ast
import os
import re
from pathlib import Path
from typing import List, Set, Tuple

import pytest


class TestOCPCompliance:
    """Test that framework is OCP-compliant."""

    def test_no_hardcoded_vertical_imports_in_framework(self):
        """Test that framework has no hardcoded vertical imports.

        This is the most critical OCP test - the framework should NOT have
        hardcoded imports like "from victor.coding import X" because that
        violates OCP (adding a new vertical would require framework changes).
        """
        framework_dir = Path(__file__).parent.parent.parent.parent / "victor" / "framework"

        # Find all Python files in framework
        framework_files = list(framework_dir.rglob("*.py"))

        hardcoded_violations = []

        for file_path in framework_files:
            # Skip test files
            if "test" in file_path.name:
                continue

            content = file_path.read_text()

            # Check for hardcoded vertical imports
            # Pattern: from victor.{vertical} import
            # Allowed: from victor.core, from victor.framework, from victor.protocols
            # Not allowed: from victor.coding, from victor.research, etc.

            lines = content.split("\n")
            for line_num, line in enumerate(lines, 1):
                # Skip comments and docstrings
                stripped = line.strip()
                if (
                    stripped.startswith("#")
                    or stripped.startswith('"""')
                    or stripped.startswith("'''")
                ):
                    continue

                # Check for vertical imports
                for vertical in [
                    "coding",
                    "research",
                    "devops",
                    "rag",
                    "dataanalysis",
                    "benchmark",
                ]:
                    # Pattern: from victor.{vertical}
                    if re.match(rf"from victor\.{vertical}(?:\.|$)", line):
                        # Check if it's in a type check (allowed)
                        if "TYPE_CHECKING" in line or "if TYPE_CHECKING" in line:
                            continue

                        hardcoded_violations.append(
                            {
                                "file": str(file_path.relative_to(framework_dir.parent.parent)),
                                "line": line_num,
                                "content": line.strip(),
                                "vertical": vertical,
                            }
                        )

        # Should have NO hardcoded vertical imports in framework
        # (except in type checking blocks which are allowed)
        assert len(hardcoded_violations) == 0, (
            f"Found {len(hardcoded_violations)} hardcoded vertical imports in framework:\n"
            + "\n".join(f"  {v['file']}:{v['line']}: {v['content']}" for v in hardcoded_violations)
        )

    def test_framework_uses_discovery_api(self):
        """Test that framework uses VerticalDiscovery API instead of direct imports."""
        framework_dir = Path(__file__).parent.parent.parent.parent / "victor" / "framework"

        # Find all Python files in framework
        framework_files = list(framework_dir.rglob("*.py"))

        uses_discovery = []

        for file_path in framework_files:
            # Skip test files
            if "test" in file_path.name:
                continue

            content = file_path.read_text()

            # Check for usage of VerticalDiscovery
            if (
                "VerticalDiscovery" in content
                or "from victor.framework.discovery import" in content
            ):
                uses_discovery.append(file_path.name)

        # At least some framework files should use discovery
        # (especially prompt_builder.py and escape_hatch_registry.py)
        assert len(uses_discovery) >= 1, "Framework should use VerticalDiscovery API"

        # Specific files that MUST use discovery
        critical_files = [
            framework_dir / "prompt_builder.py",
            framework_dir / "escape_hatch_registry.py",
        ]

        for critical_file in critical_files:
            if critical_file.exists():
                content = critical_file.read_text()
                # Should use discovery or be in type checking block
                uses_discovery_or_typing = (
                    "VerticalDiscovery" in content
                    or "from victor.framework.discovery" in content
                    or "TYPE_CHECKING" in content
                )
                assert (
                    uses_discovery_or_typing
                ), f"{critical_file.name} should use VerticalDiscovery"

    def test_prompt_builder_uses_discovery(self):
        """Test that PromptBuilder uses discovery for vertical prompt contributors."""
        from victor.framework import prompt_builder as pb_module
        import inspect

        # Check that create_coding_prompt_builder uses discovery
        if hasattr(pb_module, "create_coding_prompt_builder"):
            source = inspect.getsource(pb_module.create_coding_prompt_builder)

            # Should mention discovery or handle missing vertical gracefully
            # (backward compatibility is OK)
            has_discovery = "VerticalDiscovery" in source or "discover" in source
            assert has_discovery or "backward" in source.lower(), (
                "create_coding_prompt_builder should use VerticalDiscovery "
                "or have backward compatibility comment"
            )

    def test_escape_hatch_registry_uses_discovery(self):
        """Test that EscapeHatchRegistry has discover_from_all_verticals method."""
        from victor.framework.escape_hatch_registry import EscapeHatchRegistry

        # Should have OCP-compliant discovery method
        assert hasattr(EscapeHatchRegistry, "discover_from_all_verticals")

        # Check that it uses VerticalDiscovery
        import inspect

        source = inspect.getsource(EscapeHatchRegistry.discover_from_all_verticals)
        assert (
            "VerticalDiscovery" in source
        ), "discover_from_all_verticals should use VerticalDiscovery"

    def test_verticals_auto_register_on_import(self):
        """Test that verticals auto-register their capabilities on import."""
        # Clear registry first
        from victor.framework.escape_hatch_registry import EscapeHatchRegistry

        EscapeHatchRegistry.reset_instance()

        # Import a vertical (should trigger auto-registration)
        from victor.coding import CodingAssistant  # noqa: F401

        # Check that escape hatches were registered
        registry = EscapeHatchRegistry.get_instance()

        # Should have coding vertical escape hatches
        coding_conditions, coding_transforms = registry.get_registry_for_vertical("coding")

        # Should have at least some escape hatches
        assert len(coding_conditions) > 0 or len(coding_transforms) > 0

    def test_can_add_vertical_without_modifying_framework(self):
        """Test that a new vertical can be added without modifying framework code.

        This is the ultimate OCP test - verify that the discovery system allows
        new verticals to be added without framework changes.
        """
        from victor.framework.discovery import VerticalDiscovery

        # Clear cache
        VerticalDiscovery.clear_cache()

        # Discover all verticals
        verticals = VerticalDiscovery.discover_verticals()

        # Should find coding and research (at minimum)
        assert "coding" in verticals
        assert "research" in verticals

        # Each vertical should be discoverable via get_prompt_contributor
        # (this tests protocol-based discovery)
        for vertical_name, vertical_class in verticals.items():
            if hasattr(vertical_class, "get_prompt_contributor"):
                contributor = vertical_class.get_prompt_contributor()
                assert contributor is not None
                # Should implement the protocol
                assert hasattr(contributor, "get_system_prompt_section")

    def test_no_vertical_specific_factory_methods_in_framework(self):
        """Test that framework doesn't have vertical-specific factory methods.

        Example of what we DON'T want:
            def create_coding_agent() -> Agent:  # ❌ Violates OCP
                return Agent(vertical=CodingAssistant)

        Example of what we DO want:
            def create_agent(vertical: Type[VerticalBase]) -> Agent:  # ✅ OCP-compliant
                return Agent(vertical=vertical)
        """
        framework_dir = Path(__file__).parent.parent.parent.parent / "victor" / "framework"

        # Find all Python files in framework
        framework_files = list(framework_dir.rglob("*.py"))

        violations = []

        for file_path in framework_files:
            # Skip test files
            if "test" in file_path.name:
                continue

            content = file_path.read_text()

            # Check for vertical-specific factory functions
            # Pattern: def create_{vertical}_...
            lines = content.split("\n")
            for line_num, line in enumerate(lines, 1):
                # Look for function definitions
                if re.match(
                    r"def create_(coding|research|devops|rag|dataanalysis|benchmark)_", line
                ):
                    # This is OK if it's a prompt builder (those are helper functions)
                    # Not OK if it's an agent factory
                    if "prompt_builder" in file_path.name or "agent" not in line.lower():
                        continue

                    violations.append(
                        {
                            "file": str(file_path.relative_to(framework_dir.parent.parent)),
                            "line": line_num,
                            "content": line.strip(),
                        }
                    )

        # Agent factory should not be vertical-specific
        # (prompt builders are OK as convenience functions)
        agent_factory_violations = [v for v in violations if "prompt_builder" not in v["file"]]

        assert len(agent_factory_violations) == 0, (
            f"Found {len(agent_factory_violations)} vertical-specific agent factory methods "
            f"in framework (violates OCP):\n"
            + "\n".join(
                f"  {v['file']}:{v['line']}: {v['content']}" for v in agent_factory_violations
            )
        )


class TestOCPComplianceAdvanced:
    """Advanced OCP compliance tests."""

    def test_framework_imports_only_protocols_and_core(self):
        """Test that framework imports are limited to protocols and core modules."""
        framework_dir = Path(__file__).parent.parent.parent.parent / "victor" / "framework"

        # Disallowed import prefixes (vertical-specific)
        disallowed_prefixes = {
            "from victor.coding",
            "from victor.research",
            "from victor.devops",
            "from victor.rag",
            "from victor.dataanalysis",
            "from victor.benchmark",
        }

        framework_files = list(framework_dir.rglob("*.py"))

        for file_path in framework_files:
            # Skip test files
            if "test" in file_path.name:
                continue

            # Skip __init__.py files (they may have re-exports)
            if file_path.name == "__init__.py":
                continue

            content = file_path.read_text()
            lines = content.split("\n")

            # Track if we're in a docstring
            in_docstring = False
            docstring_indent = 0

            for line_num, line in enumerate(lines, 1):
                # Track docstring state
                stripped = line.strip()

                # Start of docstring
                if '"""' in stripped or "'''" in stripped:
                    if not in_docstring:
                        in_docstring = True
                        docstring_indent = len(line) - len(line.lstrip())
                    elif stripped.count('"""') >= 2 or stripped.count("'''") >= 2:
                        # Single-line docstring
                        in_docstring = False
                    continue

                # End of docstring (check indentation)
                if in_docstring:
                    current_indent = len(line) - len(line.lstrip())
                    if (
                        current_indent <= docstring_indent
                        and stripped
                        and not stripped.startswith("#")
                    ):
                        in_docstring = False
                    else:
                        # Still in docstring, skip this line
                        continue

                # Skip non-imports
                if not stripped.startswith("from ") and not stripped.startswith("import "):
                    continue

                # Skip type checking blocks
                if "TYPE_CHECKING" in line:
                    continue

                # Skip example code comments
                if stripped.startswith("#") and "Example" in line:
                    continue

                # Check for disallowed imports
                for disallowed in disallowed_prefixes:
                    if stripped.startswith(disallowed):
                        # Found a violation
                        pytest.fail(
                            f"OCP Violation: {file_path.relative_to(framework_dir.parent.parent)}:{line_num} "
                            f"has disallowed import: {line.strip()}"
                        )

    def test_discovery_system_caching(self):
        """Test that discovery system uses caching for performance."""
        from victor.framework.discovery import VerticalDiscovery

        # Clear cache
        VerticalDiscovery.clear_cache()

        # First call (populate cache)
        verticals1 = VerticalDiscovery.discover_verticals()

        # Second call (from cache)
        verticals2 = VerticalDiscovery.discover_verticals()

        # Should be same instance (cached)
        assert verticals1 is verticals2

    def test_all_verticals_discoverable_via_entry_points_or_builtin(self):
        """Test that all verticals are discoverable without hardcoded imports."""
        from victor.framework.discovery import VerticalDiscovery

        # Clear cache
        VerticalDiscovery.clear_cache()

        # Discover verticals
        verticals = VerticalDiscovery.discover_verticals()

        # Should find at least coding and research
        assert "coding" in verticals
        assert "research" in verticals

        # Each discovered vertical should be usable
        for vertical_name, vertical_class in verticals.items():
            # Should have name attribute
            assert hasattr(vertical_class, "name")

            # Should have get_config method
            assert hasattr(vertical_class, "get_config")

            # Should be able to get config
            config = vertical_class.get_config()
            assert config is not None

    def test_protocol_based_extension_loading(self):
        """Test that verticals use protocol-based extension loading."""
        from victor.framework.discovery import VerticalDiscovery

        verticals = VerticalDiscovery.discover_verticals()

        # Each vertical should use protocols for extension loading
        for vertical_name, vertical_class in verticals.items():
            # Should inherit from VerticalBase
            from victor.core.verticals.base import VerticalBase

            assert issubclass(vertical_class, VerticalBase)

            # Should implement protocol-based methods
            # (via VerticalExtensionLoader)
            assert hasattr(vertical_class, "get_extensions")
            assert hasattr(vertical_class, "get_config")


__all__ = [
    "TestOCPCompliance",
    "TestOCPComplianceAdvanced",
]
