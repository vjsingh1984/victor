"""Research Tool Dependencies - Tool relationships and sequencing for research workflows."""

from typing import Dict, List, Set, Tuple

from victor.verticals.protocols import ToolDependencyProviderProtocol, ToolDependency


# Tool dependency graph for research workflows
# Format: tool_name -> list of tools that commonly follow
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


class ResearchToolDependencyProvider(ToolDependencyProviderProtocol):
    """Provides tool dependency information for research workflows."""

    def get_dependencies(self) -> List[ToolDependency]:
        """Get tool dependencies as ToolDependency objects."""
        return [
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

    def get_tool_transitions(self) -> Dict[str, List[Tuple[str, float]]]:
        """Return tool transition probabilities.

        Returns:
            Dict mapping tool_name to list of (next_tool, probability) tuples.
        """
        return RESEARCH_TOOL_TRANSITIONS

    def get_tool_clusters(self) -> Dict[str, Set[str]]:
        """Return tool clusters that work well together.

        Returns:
            Dict mapping cluster_name to set of tool_names.
        """
        return RESEARCH_TOOL_CLUSTERS

    def get_recommended_sequence(self, task_type: str) -> List[str]:
        """Get recommended tool sequence for a task type.

        Args:
            task_type: Type of research task.

        Returns:
            List of tool names in recommended order.
        """
        return RESEARCH_TOOL_SEQUENCES.get(task_type, ["web_search", "web_fetch"])

    def get_required_tools(self) -> Set[str]:
        """Return tools that are essential for research.

        Returns:
            Set of tool names that should always be available.
        """
        return {"web_search", "web_fetch", "read_file", "write_file"}

    def get_optional_tools(self) -> Set[str]:
        """Return tools that enhance research but aren't essential.

        Returns:
            Set of optional tool names.
        """
        return {"code_search", "semantic_code_search", "codebase_overview", "list_directory", "edit_files"}

    def get_tool_sequences(self) -> List[List[str]]:
        """Return recommended tool sequences for research workflows.

        Returns:
            List of tool sequences (each sequence is a list of tool names).
        """
        return [
            list(seq) for seq in RESEARCH_TOOL_SEQUENCES.values()
        ]

    def suggest_next_tool(self, current_tool: str, used_tools: List[str]) -> str:
        """Suggest the next tool based on current tool and history.

        Args:
            current_tool: The tool that was just used.
            used_tools: List of tools used so far in the session.

        Returns:
            Suggested next tool name.
        """
        transitions = self.get_tool_transitions()
        if current_tool not in transitions:
            return "web_search"  # Default to search

        # Get transition probabilities
        candidates = transitions[current_tool]

        # Prefer tools not recently used (avoid loops)
        recent = set(used_tools[-3:]) if len(used_tools) >= 3 else set(used_tools)

        for tool, _prob in candidates:
            if tool not in recent:
                return tool

        # Fall back to highest probability
        return candidates[0][0] if candidates else "web_search"
