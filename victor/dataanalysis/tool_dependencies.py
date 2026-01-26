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

"""Data Analysis Tool Dependencies - Tool relationships for data analysis workflows.

Migrated to YAML-based configuration for declarative tool dependency management.
Configuration is loaded from tool_dependencies.yaml in this directory.

Uses canonical tool names from ToolNames to ensure consistent naming
across RL Q-values, workflow patterns, and vertical configurations.

The YAML-based approach provides:
- Declarative configuration that's easier to read and modify
- Consistent schema validation via Pydantic
- Automatic tool name canonicalization
- Caching for performance

Use the canonical tool dependency provider:
    from victor.core.tool_dependency_loader import create_vertical_tool_dependency_provider
    provider = create_vertical_tool_dependency_provider("dataanalysis")
"""

from pathlib import Path

from victor.core.tool_dependency_loader import create_vertical_tool_dependency_provider

# Create canonical provider for data analysis vertical
DataAnalysisToolDependencyProvider = create_vertical_tool_dependency_provider("dataanalysis")

__all__ = [
    "DataAnalysisToolDependencyProvider",
]
