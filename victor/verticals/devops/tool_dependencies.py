"""DevOps Tool Dependencies - Tool relationships for infrastructure workflows."""

from typing import Dict, List, Set, Tuple

from victor.verticals.protocols import ToolDependencyProviderProtocol, ToolDependency


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


class DevOpsToolDependencyProvider(ToolDependencyProviderProtocol):
    """Provides tool dependency information for DevOps workflows."""

    def get_dependencies(self) -> List[ToolDependency]:
        """Get tool dependencies as ToolDependency objects."""
        return [
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
        ]

    def get_tool_transitions(self) -> Dict[str, List[Tuple[str, float]]]:
        """Return tool transition probabilities."""
        return DEVOPS_TOOL_TRANSITIONS

    def get_tool_clusters(self) -> Dict[str, Set[str]]:
        """Return tool clusters that work well together."""
        return DEVOPS_TOOL_CLUSTERS

    def get_recommended_sequence(self, task_type: str) -> List[str]:
        """Get recommended tool sequence for a task type."""
        return DEVOPS_TOOL_SEQUENCES.get(task_type, ["read_file", "write_file", "bash"])

    def get_required_tools(self) -> Set[str]:
        """Return tools that are essential for DevOps."""
        return {"read_file", "write_file", "edit_files", "bash"}

    def get_optional_tools(self) -> Set[str]:
        """Return tools that enhance DevOps but aren't essential."""
        return {
            "code_search",
            "semantic_code_search",
            "git_status",
            "git_diff",
            "git_commit",
            "web_search",
            "web_fetch",
        }

    def get_tool_sequences(self) -> List[List[str]]:
        """Return recommended tool sequences for DevOps workflows."""
        return [list(seq) for seq in DEVOPS_TOOL_SEQUENCES.values()]

    def suggest_next_tool(self, current_tool: str, used_tools: List[str]) -> str:
        """Suggest the next tool based on current tool and history."""
        transitions = self.get_tool_transitions()
        if current_tool not in transitions:
            return "read_file"  # Default to reading

        # Get transition probabilities
        candidates = transitions[current_tool]

        # Prefer tools not recently used (avoid loops)
        recent = set(used_tools[-3:]) if len(used_tools) >= 3 else set(used_tools)

        for tool, _prob in candidates:
            if tool not in recent:
                return tool

        # Fall back to highest probability
        return candidates[0][0] if candidates else "read_file"
