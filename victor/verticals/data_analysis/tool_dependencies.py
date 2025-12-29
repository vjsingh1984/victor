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
"""

from typing import Dict, List, Set, Tuple

from victor.core.tool_dependency_base import BaseToolDependencyProvider, ToolDependencyConfig
from victor.core.tool_types import ToolDependency


# Tool dependency graph for data analysis workflows
DATA_ANALYSIS_TOOL_TRANSITIONS: Dict[str, List[Tuple[str, float]]] = {
    # Reading data files
    "read_file": [
        ("bash", 0.5),  # Run Python for analysis
        ("write_file", 0.2),  # Save processed data
        ("code_search", 0.2),  # Find related analysis
        ("read_file", 0.1),  # Read more files
    ],
    # Python/bash for analysis
    "bash": [
        ("write_file", 0.3),  # Save results/charts
        ("bash", 0.3),  # Continue analysis
        ("read_file", 0.2),  # Check outputs
        ("edit_files", 0.2),  # Modify scripts
    ],
    # Writing results
    "write_file": [
        ("bash", 0.4),  # Run more analysis
        ("read_file", 0.3),  # Verify written
        ("write_file", 0.2),  # Write more outputs
        ("edit_files", 0.1),  # Refine
    ],
    # Code search for patterns
    "code_search": [
        ("read_file", 0.5),  # Read found code
        ("bash", 0.3),  # Run found analysis
        ("semantic_code_search", 0.2),  # Refine search
    ],
    # Web for documentation
    "web_search": [
        ("web_fetch", 0.6),  # Fetch documentation
        ("bash", 0.2),  # Try example code
        ("web_search", 0.2),  # Refine search
    ],
    "web_fetch": [
        ("bash", 0.4),  # Apply learnings
        ("write_file", 0.3),  # Save reference
        ("web_fetch", 0.2),  # Fetch more
        ("web_search", 0.1),  # Find related
    ],
}

# Tools that work well together in data analysis
DATA_ANALYSIS_TOOL_CLUSTERS: Dict[str, Set[str]] = {
    "file_operations": {"read_file", "write_file", "edit_files", "list_directory"},
    "code_execution": {"bash"},  # Python runs through bash
    "code_exploration": {"code_search", "semantic_code_search", "codebase_overview"},
    "documentation": {"web_search", "web_fetch"},
}

# Recommended sequences for common analysis patterns
DATA_ANALYSIS_TOOL_SEQUENCES: Dict[str, List[str]] = {
    "data_profiling": ["read_file", "bash", "bash", "write_file"],
    "visualization": ["read_file", "bash", "write_file"],
    "statistical_test": ["read_file", "bash", "bash", "write_file"],
    "ml_pipeline": ["read_file", "bash", "bash", "bash", "write_file"],
    "report_generation": ["read_file", "bash", "write_file", "write_file"],
}

# Tool dependencies for data analysis
DATA_ANALYSIS_TOOL_DEPENDENCIES: List[ToolDependency] = [
    ToolDependency(
        tool_name="bash",
        depends_on={"read_file"},
        enables={"write_file", "bash"},
        weight=0.6,
    ),
    ToolDependency(
        tool_name="write_file",
        depends_on={"bash"},
        enables={"read_file"},
        weight=0.5,
    ),
    ToolDependency(
        tool_name="code_search",
        depends_on=set(),
        enables={"read_file", "bash"},
        weight=0.4,
    ),
]

# Required tools for data analysis
DATA_ANALYSIS_REQUIRED_TOOLS: Set[str] = {"read_file", "write_file", "bash"}

# Optional tools that enhance data analysis
DATA_ANALYSIS_OPTIONAL_TOOLS: Set[str] = {
    "code_search",
    "semantic_code_search",
    "edit_files",
    "list_directory",
    "web_search",
    "web_fetch",
}


class DataAnalysisToolDependencyProvider(BaseToolDependencyProvider):
    """Tool dependency provider for data analysis vertical.

    Extends BaseToolDependencyProvider with data analysis-specific tool
    relationships for data profiling, visualization, and ML workflows.
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
                default_sequence=["read_file", "bash", "write_file"],
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
