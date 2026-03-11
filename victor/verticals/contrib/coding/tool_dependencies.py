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

"""Coding-specific tool dependencies and sequences.

This module provides the canonical entry-point provider factory for
coding-vertical tool dependencies loaded from YAML configuration.

Usage::

    from victor.verticals.contrib.coding.tool_dependencies import get_provider

    provider = get_provider()
    deps = provider.get_dependencies()
    sequence = provider.get_recommended_sequence("edit")

Removed in E5 M3:
    - ``CodingToolDependencyProvider`` class (use ``get_provider()`` or
      ``create_vertical_tool_dependency_provider("coding")``)
    - Legacy constants ``CODING_TOOL_DEPENDENCIES``, ``CODING_TOOL_TRANSITIONS``,
      ``CODING_TOOL_CLUSTERS``, ``CODING_TOOL_SEQUENCES``, ``CODING_REQUIRED_TOOLS``,
      ``CODING_OPTIONAL_TOOLS`` (use provider methods instead)
"""

from __future__ import annotations

from pathlib import Path

from victor.core.tool_dependency_loader import (
    YAMLToolDependencyProvider,
)

# Path to the YAML configuration file
_YAML_CONFIG_PATH = Path(__file__).parent / "tool_dependencies.yaml"


def get_provider() -> YAMLToolDependencyProvider:
    """Entry point provider factory for coding vertical.

    This function is registered as an entry point in pyproject.toml:
        [project.entry-points."victor.tool_dependencies"]
        coding = "victor_coding.tool_dependencies:get_provider"

    Returns:
        A configured tool dependency provider for the coding vertical.

    Example:
        # Framework usage via entry points:
        from importlib.metadata import entry_points
        eps = entry_points(group="victor.tool_dependencies")
        for ep in eps:
            if ep.name == "coding":
                provider_factory = ep.load()
                provider = provider_factory()
                deps = provider.get_dependencies()
    """
    return YAMLToolDependencyProvider(
        yaml_path=_YAML_CONFIG_PATH,
        canonicalize=True,
    )


__all__ = [
    "get_provider",
]
