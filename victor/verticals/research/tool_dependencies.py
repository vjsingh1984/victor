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

Extends the core BaseToolDependencyProvider with research-specific data.
"""

from typing import Dict, List, Set, Tuple

from victor.core.tool_dependency_base import BaseToolDependencyProvider, ToolDependencyConfig
from victor.core.tool_types import ToolDependency


# Tool dependency graph for research workflows
RESEARCH_TOOL_TRANSITIONS: Dict[str, List[Tuple[str, float]]] = {
    # Search tools lead to fetch/read
    "web_search": [
        ("web_fetch", 0.7),  # Usually fetch after finding results
        ("web_search", 0.2),  # Sometimes refine search
        ("read_file", 0.1),  # Check local context
    ],
    # Fetch leads to more fetch or synthesis
    "web_fetch": [
        ("web_fetch", 0.4),  # Fetch more pages
        ("web_search", 0.3),  # Search for related info
        ("write_file", 0.2),  # Write findings
        ("read_file", 0.1),  # Check local notes
    ],
    # Reading local files
    "read_file": [
        ("web_search", 0.4),  # Research based on file content
        ("write_file", 0.3),  # Update notes
        ("edit_files", 0.2),  # Modify content
        ("list_directory", 0.1),  # Find more files
    ],
    # Writing outputs
    "write_file": [
        ("read_file", 0.4),  # Review what was written
        ("edit_files", 0.3),  # Refine content
        ("web_search", 0.2),  # Verify/expand
        ("web_fetch", 0.1),  # Get more sources
    ],
    # Code search for technical research
    "code_search": [
        ("read_file", 0.5),  # Read found code
        ("semantic_code_search", 0.3),  # Semantic follow-up
        ("code_search", 0.2),  # Refine search
    ],
    "semantic_code_search": [
        ("read_file", 0.5),  # Read found code
        ("code_search", 0.3),  # Keyword follow-up
        ("codebase_overview", 0.2),  # Get context
    ],
}

# Tools that should typically be used together
RESEARCH_TOOL_CLUSTERS: Dict[str, Set[str]] = {
    "web_research": {"web_search", "web_fetch"},
    "file_operations": {"read_file", "write_file", "edit_files", "list_directory"},
    "code_research": {"code_search", "semantic_code_search", "codebase_overview"},
}

# Recommended tool sequences for common research patterns
RESEARCH_TOOL_SEQUENCES: Dict[str, List[str]] = {
    "fact_check": ["web_search", "web_fetch", "web_search", "web_fetch"],
    "literature_review": ["web_search", "web_fetch", "web_fetch", "web_fetch", "write_file"],
    "technical_lookup": ["web_search", "web_fetch", "code_search", "read_file"],
    "report_writing": ["read_file", "web_search", "web_fetch", "write_file", "edit_files"],
}

# Tool dependencies for research
RESEARCH_TOOL_DEPENDENCIES: List[ToolDependency] = [
    ToolDependency(
        tool_name="web_fetch",
        depends_on={"web_search"},
        enables={"write_file", "web_fetch"},
        weight=0.7,
    ),
    ToolDependency(
        tool_name="write_file",
        depends_on={"web_fetch", "read_file"},
        enables={"read_file", "edit_files"},
        weight=0.5,
    ),
    ToolDependency(
        tool_name="code_search",
        depends_on=set(),
        enables={"read_file", "semantic_code_search"},
        weight=0.5,
    ),
]

# Required tools for research
RESEARCH_REQUIRED_TOOLS: Set[str] = {"web_search", "web_fetch", "read_file", "write_file"}

# Optional tools that enhance research
RESEARCH_OPTIONAL_TOOLS: Set[str] = {
    "code_search",
    "semantic_code_search",
    "codebase_overview",
    "list_directory",
    "edit_files",
}


class ResearchToolDependencyProvider(BaseToolDependencyProvider):
    """Tool dependency provider for research vertical.

    Extends BaseToolDependencyProvider with research-specific tool
    relationships for fact-checking, literature review, and report writing.
    """

    def __init__(self):
        """Initialize the provider with research-specific config."""
        super().__init__(
            ToolDependencyConfig(
                dependencies=RESEARCH_TOOL_DEPENDENCIES,
                transitions=RESEARCH_TOOL_TRANSITIONS,
                clusters=RESEARCH_TOOL_CLUSTERS,
                sequences=RESEARCH_TOOL_SEQUENCES,
                required_tools=RESEARCH_REQUIRED_TOOLS,
                optional_tools=RESEARCH_OPTIONAL_TOOLS,
                default_sequence=["web_search", "web_fetch"],
            )
        )


__all__ = [
    "ResearchToolDependencyProvider",
    "RESEARCH_TOOL_DEPENDENCIES",
    "RESEARCH_TOOL_TRANSITIONS",
    "RESEARCH_TOOL_CLUSTERS",
    "RESEARCH_TOOL_SEQUENCES",
    "RESEARCH_REQUIRED_TOOLS",
    "RESEARCH_OPTIONAL_TOOLS",
]
