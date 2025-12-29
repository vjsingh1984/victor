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

Uses canonical tool names from ToolNames to ensure consistent naming
across RL Q-values, workflow patterns, and vertical configurations.
"""

from typing import Dict, List, Set, Tuple

from victor.core.tool_dependency_base import BaseToolDependencyProvider, ToolDependencyConfig
from victor.core.tool_types import ToolDependency
from victor.framework.tool_naming import ToolNames


# Tool dependency graph for research workflows
# Uses canonical ToolNames constants for consistency
RESEARCH_TOOL_TRANSITIONS: Dict[str, List[Tuple[str, float]]] = {
    # Search tools lead to fetch/read
    ToolNames.WEB_SEARCH: [
        (ToolNames.WEB_FETCH, 0.7),  # Usually fetch after finding results
        (ToolNames.WEB_SEARCH, 0.2),  # Sometimes refine search
        (ToolNames.READ, 0.1),  # Check local context
    ],
    # Fetch leads to more fetch or synthesis
    ToolNames.WEB_FETCH: [
        (ToolNames.WEB_FETCH, 0.4),  # Fetch more pages
        (ToolNames.WEB_SEARCH, 0.3),  # Search for related info
        (ToolNames.WRITE, 0.2),  # Write findings
        (ToolNames.READ, 0.1),  # Check local notes
    ],
    # Reading local files
    ToolNames.READ: [
        (ToolNames.WEB_SEARCH, 0.4),  # Research based on file content
        (ToolNames.WRITE, 0.3),  # Update notes
        (ToolNames.EDIT, 0.2),  # Modify content
        (ToolNames.LS, 0.1),  # Find more files
    ],
    # Writing outputs
    ToolNames.WRITE: [
        (ToolNames.READ, 0.4),  # Review what was written
        (ToolNames.EDIT, 0.3),  # Refine content
        (ToolNames.WEB_SEARCH, 0.2),  # Verify/expand
        (ToolNames.WEB_FETCH, 0.1),  # Get more sources
    ],
    # Code search for technical research
    ToolNames.GREP: [
        (ToolNames.READ, 0.5),  # Read found code
        (ToolNames.CODE_SEARCH, 0.3),  # Semantic follow-up
        (ToolNames.GREP, 0.2),  # Refine search
    ],
    ToolNames.CODE_SEARCH: [
        (ToolNames.READ, 0.5),  # Read found code
        (ToolNames.GREP, 0.3),  # Keyword follow-up
        (ToolNames.OVERVIEW, 0.2),  # Get context
    ],
}

# Tools that should typically be used together
# Uses canonical ToolNames constants for consistency
RESEARCH_TOOL_CLUSTERS: Dict[str, Set[str]] = {
    "web_research": {ToolNames.WEB_SEARCH, ToolNames.WEB_FETCH},
    "file_operations": {ToolNames.READ, ToolNames.WRITE, ToolNames.EDIT, ToolNames.LS},
    "code_research": {ToolNames.GREP, ToolNames.CODE_SEARCH, ToolNames.OVERVIEW},
}

# Recommended tool sequences for common research patterns
# Uses canonical ToolNames constants for consistency
RESEARCH_TOOL_SEQUENCES: Dict[str, List[str]] = {
    "fact_check": [ToolNames.WEB_SEARCH, ToolNames.WEB_FETCH, ToolNames.WEB_SEARCH, ToolNames.WEB_FETCH],
    "literature_review": [ToolNames.WEB_SEARCH, ToolNames.WEB_FETCH, ToolNames.WEB_FETCH, ToolNames.WEB_FETCH, ToolNames.WRITE],
    "technical_lookup": [ToolNames.WEB_SEARCH, ToolNames.WEB_FETCH, ToolNames.GREP, ToolNames.READ],
    "report_writing": [ToolNames.READ, ToolNames.WEB_SEARCH, ToolNames.WEB_FETCH, ToolNames.WRITE, ToolNames.EDIT],
}

# Tool dependencies for research
# Uses canonical ToolNames constants for consistency
RESEARCH_TOOL_DEPENDENCIES: List[ToolDependency] = [
    ToolDependency(
        tool_name=ToolNames.WEB_FETCH,
        depends_on={ToolNames.WEB_SEARCH},
        enables={ToolNames.WRITE, ToolNames.WEB_FETCH},
        weight=0.7,
    ),
    ToolDependency(
        tool_name=ToolNames.WRITE,
        depends_on={ToolNames.WEB_FETCH, ToolNames.READ},
        enables={ToolNames.READ, ToolNames.EDIT},
        weight=0.5,
    ),
    ToolDependency(
        tool_name=ToolNames.GREP,
        depends_on=set(),
        enables={ToolNames.READ, ToolNames.CODE_SEARCH},
        weight=0.5,
    ),
]

# Required tools for research
# Uses canonical ToolNames constants for consistency
RESEARCH_REQUIRED_TOOLS: Set[str] = {ToolNames.WEB_SEARCH, ToolNames.WEB_FETCH, ToolNames.READ, ToolNames.WRITE}

# Optional tools that enhance research
# Uses canonical ToolNames constants for consistency
RESEARCH_OPTIONAL_TOOLS: Set[str] = {
    ToolNames.GREP,
    ToolNames.CODE_SEARCH,
    ToolNames.OVERVIEW,
    ToolNames.LS,
    ToolNames.EDIT,
}


class ResearchToolDependencyProvider(BaseToolDependencyProvider):
    """Tool dependency provider for research vertical.

    Extends BaseToolDependencyProvider with research-specific tool
    relationships for fact-checking, literature review, and report writing.

    Uses canonical ToolNames constants for consistency.
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
                default_sequence=[ToolNames.WEB_SEARCH, ToolNames.WEB_FETCH],
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
