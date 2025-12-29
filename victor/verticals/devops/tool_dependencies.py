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

"""DevOps Tool Dependencies - Tool relationships for infrastructure workflows.

Extends the core BaseToolDependencyProvider with DevOps-specific data.
"""

from typing import Dict, List, Set, Tuple

from victor.core.tool_dependency_base import BaseToolDependencyProvider, ToolDependencyConfig
from victor.core.tool_types import ToolDependency


# Tool dependency graph for DevOps workflows
DEVOPS_TOOL_TRANSITIONS: Dict[str, List[Tuple[str, float]]] = {
    # Reading leads to more analysis or writing
    "read_file": [
        ("bash", 0.3),  # Execute what we read
        ("edit_files", 0.3),  # Modify configuration
        ("code_search", 0.2),  # Find related configs
        ("write_file", 0.2),  # Write new config
    ],
    # Code search for finding configurations
    "code_search": [
        ("read_file", 0.5),  # Read found files
        ("semantic_code_search", 0.2),  # Refine search
        ("code_search", 0.2),  # Continue searching
        ("codebase_overview", 0.1),  # Get context
    ],
    # Bash for infrastructure commands
    "bash": [
        ("read_file", 0.3),  # Check results
        ("bash", 0.3),  # Chain commands
        ("write_file", 0.2),  # Save output
        ("git_status", 0.2),  # Check git state
    ],
    # Writing configurations
    "write_file": [
        ("bash", 0.4),  # Validate/apply config
        ("read_file", 0.3),  # Review what was written
        ("git_status", 0.2),  # Check changes
        ("edit_files", 0.1),  # Refine
    ],
    # Editing configurations
    "edit_files": [
        ("bash", 0.4),  # Test changes
        ("read_file", 0.3),  # Verify edit
        ("git_diff", 0.2),  # Review changes
        ("git_status", 0.1),  # Check status
    ],
    # Git operations
    "git_status": [
        ("git_diff", 0.4),  # See what changed
        ("git_commit", 0.3),  # Commit changes
        ("read_file", 0.2),  # Check files
        ("bash", 0.1),  # Run commands
    ],
    "git_diff": [
        ("git_commit", 0.4),  # Commit if ok
        ("edit_files", 0.3),  # Fix issues
        ("git_status", 0.2),  # Recheck
        ("read_file", 0.1),  # Review files
    ],
    # Web for documentation
    "web_search": [
        ("web_fetch", 0.6),  # Fetch found docs
        ("web_search", 0.2),  # Refine search
        ("read_file", 0.1),  # Check local files
        ("write_file", 0.1),  # Document findings
    ],
    "web_fetch": [
        ("write_file", 0.3),  # Apply learnings
        ("web_search", 0.3),  # Find more
        ("web_fetch", 0.2),  # Fetch related
        ("edit_files", 0.2),  # Update configs
    ],
}

# Tools that work well together in DevOps
DEVOPS_TOOL_CLUSTERS: Dict[str, Set[str]] = {
    "file_operations": {"read_file", "write_file", "edit_files", "list_directory"},
    "git_operations": {"git_status", "git_diff", "git_commit"},
    "code_exploration": {"code_search", "semantic_code_search", "codebase_overview"},
    "infrastructure": {"bash", "read_file", "write_file"},
    "documentation": {"web_search", "web_fetch"},
}

# Recommended sequences for common DevOps tasks
DEVOPS_TOOL_SEQUENCES: Dict[str, List[str]] = {
    "dockerfile_create": ["read_file", "write_file", "bash"],
    "ci_cd_setup": ["list_directory", "code_search", "read_file", "write_file", "bash"],
    "kubernetes_deploy": ["read_file", "write_file", "bash", "bash"],
    "terraform_apply": ["read_file", "bash", "bash", "git_commit"],
    "config_update": ["read_file", "edit_files", "bash", "git_status", "git_commit"],
}

# Tool dependencies for DevOps
DEVOPS_TOOL_DEPENDENCIES: List[ToolDependency] = [
    ToolDependency(
        tool_name="bash",
        depends_on={"read_file"},
        enables={"write_file", "git_status"},
        weight=0.6,
    ),
    ToolDependency(
        tool_name="edit_files",
        depends_on={"read_file"},
        enables={"bash", "git_diff"},
        weight=0.5,
    ),
    ToolDependency(
        tool_name="git_commit",
        depends_on={"git_status", "git_diff"},
        enables=set(),
        weight=0.4,
    ),
    ToolDependency(
        tool_name="write_file",
        depends_on={"read_file"},
        enables={"bash", "git_status"},
        weight=0.5,
    ),
]

# Required tools for DevOps
DEVOPS_REQUIRED_TOOLS: Set[str] = {"read_file", "write_file", "edit_files", "bash"}

# Optional tools that enhance DevOps
DEVOPS_OPTIONAL_TOOLS: Set[str] = {
    "code_search",
    "semantic_code_search",
    "git_status",
    "git_diff",
    "git_commit",
    "web_search",
    "web_fetch",
}


class DevOpsToolDependencyProvider(BaseToolDependencyProvider):
    """Tool dependency provider for DevOps vertical.

    Extends BaseToolDependencyProvider with DevOps-specific tool
    relationships for infrastructure and CI/CD workflows.
    """

    def __init__(self):
        """Initialize the provider with DevOps-specific config."""
        super().__init__(
            ToolDependencyConfig(
                dependencies=DEVOPS_TOOL_DEPENDENCIES,
                transitions=DEVOPS_TOOL_TRANSITIONS,
                clusters=DEVOPS_TOOL_CLUSTERS,
                sequences=DEVOPS_TOOL_SEQUENCES,
                required_tools=DEVOPS_REQUIRED_TOOLS,
                optional_tools=DEVOPS_OPTIONAL_TOOLS,
                default_sequence=["read_file", "write_file", "bash"],
            )
        )


__all__ = [
    "DevOpsToolDependencyProvider",
    "DEVOPS_TOOL_DEPENDENCIES",
    "DEVOPS_TOOL_TRANSITIONS",
    "DEVOPS_TOOL_CLUSTERS",
    "DEVOPS_TOOL_SEQUENCES",
    "DEVOPS_REQUIRED_TOOLS",
    "DEVOPS_OPTIONAL_TOOLS",
]
