"""Protocol implementations for the minimal vertical.

This module demonstrates how to implement SDK protocols
for enhanced vertical functionality.
"""

from typing import Dict, Any, List

from victor_sdk.verticals.protocols import ToolProvider, SafetyProvider


class MinimalToolProvider(ToolProvider):
    """Tool provider for the minimal vertical."""

    def get_tools(self) -> List[str]:
        """Return list of tool names."""
        return ["read", "write", "search"]

    def get_tools_for_stage(self, stage: str, task_type: str) -> List[str]:
        """Return optimized tools for a specific stage and task type.

        Args:
            stage: Workflow stage (planning, execution, verification)
            task_type: Type of task (e.g., code_edit, analysis)

        Returns:
            List of tool names optimized for this stage/task.
        """
        if stage == "planning":
            return ["search", "read"]
        elif stage == "execution":
            return ["read", "write"]
        elif stage == "verification":
            return ["read"]
        else:
            return self.get_tools()


class MinimalSafetyProvider(SafetyProvider):
    """Safety provider for the minimal vertical."""

    def __init__(self):
        """Initialize safety rules."""
        self._rules = {
            "max_file_size": 1024 * 1024,  # 1MB
            "allowed_extensions": [".py", ".txt", ".md", ".json"],
            "blocked_paths": ["/etc", "/sys", "/proc"],
        }

    def get_safety_rules(self) -> Dict[str, Any]:
        """Return safety rules."""
        return self._rules.copy()

    def validate_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> bool:
        """Validate a tool call before execution.

        Args:
            tool_name: Name of the tool being called
            arguments: Arguments being passed to the tool

        Returns:
            True if the tool call is safe, False otherwise.
        """
        # Check if path is blocked
        if "path" in arguments:
            path = arguments["path"]
            for blocked in self._rules["blocked_paths"]:
                if path.startswith(blocked):
                    return False

        return True

    def validate_prompt(self, prompt: str) -> bool:
        """Validate a user prompt before processing.

        Args:
            prompt: User's input prompt

        Returns:
            True if the prompt is safe, False otherwise.
        """
        # Basic safety check
        dangerous_patterns = ["rm -rf", "format c:", "del /s"]
        prompt_lower = prompt.lower()
        return not any(pattern in prompt_lower for pattern in dangerous_patterns)
