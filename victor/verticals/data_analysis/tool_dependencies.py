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

"""Data Analysis Tool Dependencies - Tool relationships for analysis workflows.

Extends the core BaseToolDependencyProvider with data analysis-specific data.

Uses canonical tool names from ToolNames to ensure consistent naming
across RL Q-values, workflow patterns, and vertical configurations.
"""

from typing import Dict, List, Set, Tuple

from victor.core.tool_dependency_base import BaseToolDependencyProvider, ToolDependencyConfig
from victor.core.tool_types import ToolDependency
from victor.framework.tool_naming import ToolNames


# Tool dependency graph for data analysis workflows
# Uses canonical ToolNames constants for consistency
DATA_ANALYSIS_TOOL_TRANSITIONS: Dict[str, List[Tuple[str, float]]] = {
    # Reading data files
    ToolNames.READ: [
        (ToolNames.SHELL, 0.5),  # Run Python for analysis
        (ToolNames.WRITE, 0.2),  # Save processed data
        (ToolNames.GREP, 0.2),  # Find related analysis
        (ToolNames.READ, 0.1),  # Read more files
    ],
    # Python/shell for analysis
    ToolNames.SHELL: [
        (ToolNames.WRITE, 0.3),  # Save results/charts
        (ToolNames.SHELL, 0.3),  # Continue analysis
        (ToolNames.READ, 0.2),  # Check outputs
        (ToolNames.EDIT, 0.2),  # Modify scripts
    ],
    # Writing results
    ToolNames.WRITE: [
        (ToolNames.SHELL, 0.4),  # Run more analysis
        (ToolNames.READ, 0.3),  # Verify written
        (ToolNames.WRITE, 0.2),  # Write more outputs
        (ToolNames.EDIT, 0.1),  # Refine
    ],
    # Code search for patterns
    ToolNames.GREP: [
        (ToolNames.READ, 0.5),  # Read found code
        (ToolNames.SHELL, 0.3),  # Run found analysis
        (ToolNames.CODE_SEARCH, 0.2),  # Refine search
    ],
    # Web for documentation
    ToolNames.WEB_SEARCH: [
        (ToolNames.WEB_FETCH, 0.6),  # Fetch documentation
        (ToolNames.SHELL, 0.2),  # Try example code
        (ToolNames.WEB_SEARCH, 0.2),  # Refine search
    ],
    ToolNames.WEB_FETCH: [
        (ToolNames.SHELL, 0.4),  # Apply learnings
        (ToolNames.WRITE, 0.3),  # Save reference
        (ToolNames.WEB_FETCH, 0.2),  # Fetch more
        (ToolNames.WEB_SEARCH, 0.1),  # Find related
    ],
}

# Tools that work well together in data analysis
# Uses canonical ToolNames constants for consistency
DATA_ANALYSIS_TOOL_CLUSTERS: Dict[str, Set[str]] = {
    "file_operations": {ToolNames.READ, ToolNames.WRITE, ToolNames.EDIT, ToolNames.LS},
    "code_execution": {ToolNames.SHELL},  # Python runs through shell
    "code_exploration": {ToolNames.GREP, ToolNames.CODE_SEARCH, ToolNames.OVERVIEW},
    "documentation": {ToolNames.WEB_SEARCH, ToolNames.WEB_FETCH},
}

# Recommended sequences for common analysis patterns
# Uses canonical ToolNames constants for consistency
DATA_ANALYSIS_TOOL_SEQUENCES: Dict[str, List[str]] = {
    "data_profiling": [ToolNames.READ, ToolNames.SHELL, ToolNames.SHELL, ToolNames.WRITE],
    "visualization": [ToolNames.READ, ToolNames.SHELL, ToolNames.WRITE],
    "statistical_test": [ToolNames.READ, ToolNames.SHELL, ToolNames.SHELL, ToolNames.WRITE],
    "ml_pipeline": [ToolNames.READ, ToolNames.SHELL, ToolNames.SHELL, ToolNames.SHELL, ToolNames.WRITE],
    "report_generation": [ToolNames.READ, ToolNames.SHELL, ToolNames.WRITE, ToolNames.WRITE],
}

# Tool dependencies for data analysis
# Uses canonical ToolNames constants for consistency
DATA_ANALYSIS_TOOL_DEPENDENCIES: List[ToolDependency] = [
    ToolDependency(
        tool_name=ToolNames.SHELL,
        depends_on={ToolNames.READ},
        enables={ToolNames.WRITE, ToolNames.SHELL},
        weight=0.6,
    ),
    ToolDependency(
        tool_name=ToolNames.WRITE,
        depends_on={ToolNames.SHELL},
        enables={ToolNames.READ},
        weight=0.5,
    ),
    ToolDependency(
        tool_name=ToolNames.GREP,
        depends_on=set(),
        enables={ToolNames.READ, ToolNames.SHELL},
        weight=0.4,
    ),
]

# Required tools for data analysis
# Uses canonical ToolNames constants for consistency
DATA_ANALYSIS_REQUIRED_TOOLS: Set[str] = {ToolNames.READ, ToolNames.WRITE, ToolNames.SHELL}

# Optional tools that enhance data analysis
# Uses canonical ToolNames constants for consistency
DATA_ANALYSIS_OPTIONAL_TOOLS: Set[str] = {
    ToolNames.GREP,
    ToolNames.CODE_SEARCH,
    ToolNames.EDIT,
    ToolNames.LS,
    ToolNames.WEB_SEARCH,
    ToolNames.WEB_FETCH,
}


class DataAnalysisToolDependencyProvider(BaseToolDependencyProvider):
    """Tool dependency provider for data analysis vertical.

    Extends BaseToolDependencyProvider with data analysis-specific tool
    relationships for data profiling, visualization, and ML workflows.

    Uses canonical ToolNames constants for consistency.
    """

    def __init__(self):
        """Initialize the provider with data analysis-specific config."""
        super().__init__(
            ToolDependencyConfig(
                dependencies=DATA_ANALYSIS_TOOL_DEPENDENCIES,
                transitions=DATA_ANALYSIS_TOOL_TRANSITIONS,
                clusters=DATA_ANALYSIS_TOOL_CLUSTERS,
                sequences=DATA_ANALYSIS_TOOL_SEQUENCES,
                required_tools=DATA_ANALYSIS_REQUIRED_TOOLS,
                optional_tools=DATA_ANALYSIS_OPTIONAL_TOOLS,
                default_sequence=[ToolNames.READ, ToolNames.SHELL, ToolNames.WRITE],
            )
        )


__all__ = [
    "DataAnalysisToolDependencyProvider",
    "DATA_ANALYSIS_TOOL_DEPENDENCIES",
    "DATA_ANALYSIS_TOOL_TRANSITIONS",
    "DATA_ANALYSIS_TOOL_CLUSTERS",
    "DATA_ANALYSIS_TOOL_SEQUENCES",
    "DATA_ANALYSIS_REQUIRED_TOOLS",
    "DATA_ANALYSIS_OPTIONAL_TOOLS",
]
