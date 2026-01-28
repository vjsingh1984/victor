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

"""Research Tool Dependencies - Tool relationships for research workflows.

Migrated to YAML-based configuration for declarative tool dependency management.
Configuration is loaded from tool_dependencies.yaml in this directory.

Uses canonical tool names from ToolNames to ensure consistent naming
across RL Q-values, workflow patterns, and vertical configurations.

The YAML-based approach provides:
- Declarative configuration that's easier to read and modify
- Consistent schema validation via Pydantic
- Automatic tool name canonicalization
- Caching for performance
- Auto-inference of vertical name from module path (no duplication)

Use the canonical tool dependency provider:
    from victor.core.tool_dependency_loader import create_vertical_tool_dependency_provider
    provider = create_vertical_tool_dependency_provider()  # Auto-infers "research"
"""

from pathlib import Path

from victor.core.tool_dependency_loader import create_vertical_tool_dependency_provider

# Path to the YAML configuration file
_YAML_PATH = Path(__file__).parent / "tool_dependencies.yaml"

# Create canonical provider for research vertical
# Vertical name is auto-inferred from module path (victor.research.tool_dependencies -> research)
ResearchToolDependencyProvider = create_vertical_tool_dependency_provider()

__all__ = [
    "ResearchToolDependencyProvider",
]
