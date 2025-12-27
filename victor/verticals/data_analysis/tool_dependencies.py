"""Data Analysis Tool Dependencies - Tool relationships for analysis workflows."""

from typing import Dict, List, Set, Tuple

from victor.verticals.protocols import ToolDependencyProviderProtocol, ToolDependency


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


class DataAnalysisToolDependencyProvider(ToolDependencyProviderProtocol):
    """Provides tool dependency information for data analysis workflows."""

    def get_dependencies(self) -> List[ToolDependency]:
        """Get tool dependencies as ToolDependency objects."""
        return [
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

    def get_tool_transitions(self) -> Dict[str, List[Tuple[str, float]]]:
        """Return tool transition probabilities."""
        return DATA_ANALYSIS_TOOL_TRANSITIONS

    def get_tool_clusters(self) -> Dict[str, Set[str]]:
        """Return tool clusters that work well together."""
        return DATA_ANALYSIS_TOOL_CLUSTERS

    def get_recommended_sequence(self, task_type: str) -> List[str]:
        """Get recommended tool sequence for a task type."""
        return DATA_ANALYSIS_TOOL_SEQUENCES.get(task_type, ["read_file", "bash", "write_file"])

    def get_required_tools(self) -> Set[str]:
        """Return tools that are essential for data analysis."""
        return {"read_file", "write_file", "bash"}

    def get_optional_tools(self) -> Set[str]:
        """Return tools that enhance analysis but aren't essential."""
        return {
            "code_search",
            "semantic_code_search",
            "edit_files",
            "list_directory",
            "web_search",
            "web_fetch",
        }

    def get_tool_sequences(self) -> List[List[str]]:
        """Return recommended tool sequences for data analysis workflows."""
        return [list(seq) for seq in DATA_ANALYSIS_TOOL_SEQUENCES.values()]

    def suggest_next_tool(self, current_tool: str, used_tools: List[str]) -> str:
        """Suggest the next tool based on current tool and history."""
        transitions = self.get_tool_transitions()
        if current_tool not in transitions:
            return "bash"  # Default to Python execution

        # Get transition probabilities
        candidates = transitions[current_tool]

        # Prefer tools not recently used (avoid loops)
        recent = set(used_tools[-3:]) if len(used_tools) >= 3 else set(used_tools)

        for tool, _prob in candidates:
            if tool not in recent:
                return tool

        # Fall back to highest probability
        return candidates[0][0] if candidates else "bash"
