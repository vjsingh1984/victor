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

"""Tool Selection Coordinator - Extracted from AgentOrchestrator.

This module extracts tool selection, routing, and classification logic
from the orchestrator following SRP (Single Responsibility Principle).

Extracts ~650 lines of tool selection logic including:
- Tool selection and routing (semantic vs keyword matching)
- Task classification (analysis, action, creation)
- Tool mention detection from prompts
- Required files/outputs extraction
- Tool capability checking

Design Philosophy:
- Single Responsibility: Only handles tool selection decisions
- Stateless: Selection logic doesn't depend on internal state
- Protocol-compliant: Implements IToolSelectionCoordinator
- Testable: All methods can be unit tested in isolation
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from victor.agent.protocols import IToolSelectionCoordinator, AgentToolSelectionContext

if TYPE_CHECKING:
    from victor.tools import ToolRegistry

logger = logging.getLogger(__name__)


class ToolSelectionCoordinator(IToolSelectionCoordinator):
    """Coordinator for intelligent tool selection and routing.

    Extracts tool selection logic from orchestrator, implementing
    IToolSelectionCoordinator protocol for SRP compliance.

    Attributes:
        tool_registry: Tool registry for available tools
    """

    # Keywords for task classification
    ANALYSIS_KEYWORDS = {
        "explain",
        "understand",
        "analyze",
        "review",
        "what",
        "how",
        "why",
        "describe",
        "summarize",
        "compare",
        "find",
        "search",
        "locate",
        "identify",
        "check",
        "verify",
        "validate",
        "document",
        "read",
        "show",
        "display",
        "list",
        "get",
        "examine",
        "inspect",
    }

    ACTION_KEYWORDS = {
        "fix",
        "repair",
        "resolve",
        "debug",
        "correct",
        "patch",
        "solve",
        "implement",
        "apply",
        "execute",
        "run",
        "perform",
        "do",
        "make",
        "change",
        "update",
        "modify",
        "edit",
        "refactor",
        "optimize",
        "improve",
        "enhance",
        "adjust",
        "configure",
        "setup",
        "deploy",
    }

    CREATION_KEYWORDS = {
        "create",
        "generate",
        "build",
        "write",
        "add",
        "new",
        "make",
        "develop",
        "implement",
        "design",
        "construct",
        "produce",
        "author",
        "draft",
        "compose",
        "formulate",
        "establish",
        "initiate",
        "start",
    }

    # Tool name patterns for detection
    TOOL_PATTERNS = {
        "grep": r"\bgrep\b",
        "ls": r"\bls\b",
        "read": r"\bread\b",
        "write": r"\bwrite\b",
        "web_search": r"\bweb[_\s]?search\b",
        "semantic_search": r"\bsemantic[_\s]?search\b",
        "code_search": r"\bcode[_\s]?search\b",
        "bash": r"\bbash\b",
        "shell": r"\bshell\b",
    }

    def __init__(self, tool_registry: "ToolRegistry"):
        """Initialize the coordinator.

        Args:
            tool_registry: Tool registry for available tools
        """
        self._tool_registry = tool_registry

    def get_recommended_search_tool(
        self,
        query: str,
        context: Optional["AgentToolSelectionContext"] = None,
    ) -> Optional[str]:
        """Get recommended search tool for a query.

        Analyzes the query to determine which search tool would be most
        appropriate based on query characteristics.

        Args:
            query: Search query string
            context: Optional selection context (stage, task type, history)

        Returns:
            Recommended tool name or None if no recommendation
        """
        if not query or not query.strip():
            return None

        query_lower = query.lower()

        # Check for semantic search indicators
        semantic_indicators = ["similar", "related", "like", "analogous"]
        if any(indicator in query_lower for indicator in semantic_indicators):
            return "semantic_search"

        # Check for file pattern indicators
        file_pattern_indicators = [
            "files ending",
            "files starting",
            "pattern",
            "extension",
            "glob",
            "wildcard",
        ]
        if any(indicator in query_lower for indicator in file_pattern_indicators):
            return "ls"

        # Check for web search indicators
        web_indicators = [
            "latest",
            "current",
            "recent",
            "news",
            "external",
            "internet",
            "online",
            "documentation",
            "docs",
        ]
        if any(indicator in query_lower for indicator in web_indicators):
            return "web_search"

        # Default to grep for code searches
        if "find" in query_lower or "search" in query_lower:
            return "grep"

        return None

    def route_search_query(
        self,
        query: str,
        available_tools: Set[str],
    ) -> str:
        """Route a search query to the appropriate tool.

        Determines the best search tool based on query characteristics
        and available tools.

        Args:
            query: Search query string
            available_tools: Set of available tool names

        Returns:
            Selected tool name
        """
        # Get recommendation
        recommended = self.get_recommended_search_tool(query)

        # If recommended tool is available, use it
        if recommended and recommended in available_tools:
            return recommended

        # Fallback to grep if available
        if "grep" in available_tools:
            return "grep"

        # Fallback to ls if grep not available
        if "ls" in available_tools:
            return "ls"

        # Return first available tool
        if available_tools:
            return next(iter(available_tools))

        # Default fallback
        return "grep"

    def detect_mentioned_tools(
        self,
        prompt: str,
        available_tools: Optional[Set[str]] = None,
    ) -> Set[str]:
        """Detect tools mentioned in a prompt.

        Scans the prompt for explicit tool mentions.

        Args:
            prompt: Prompt text to scan
            available_tools: Optional set of available tools (defaults to all)

        Returns:
            Set of detected tool names
        """
        if not prompt:
            return set()

        detected = set()
        prompt_lower = prompt.lower()

        # Check for explicit tool mentions
        for tool_name, pattern in self.TOOL_PATTERNS.items():
            if re.search(pattern, prompt_lower, re.IGNORECASE):
                detected.add(tool_name)

        # If available_tools provided, only return those
        if available_tools is not None:
            detected = detected.intersection(available_tools)

        return detected

    def classify_task_keywords(
        self,
        task: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Classify task type using keyword analysis.

        Determines if a task is primarily analysis, action, or creation
        based on keyword presence.

        Args:
            task: Task description
            conversation_history: Optional conversation history for context

        Returns:
            Task type: "analysis", "action", or "creation"
        """
        if not task:
            return "analysis"

        task_lower = task.lower()

        # Count keywords for each category
        analysis_count = sum(1 for kw in self.ANALYSIS_KEYWORDS if kw in task_lower)
        action_count = sum(1 for kw in self.ACTION_KEYWORDS if kw in task_lower)
        creation_count = sum(1 for kw in self.CREATION_KEYWORDS if kw in task_lower)

        # Determine category with most matches
        if creation_count > action_count and creation_count > analysis_count:
            return "creation"
        elif action_count > analysis_count:
            return "action"
        else:
            return "analysis"

    def classify_task_with_context(
        self,
        task: str,
        context: Optional["AgentToolSelectionContext"] = None,
    ) -> str:
        """Classify task type with full context.

        Enhanced task classification using conversation stage, recent tools,
        and other context information.

        Args:
            task: Task description
            context: Selection context with stage, history, recent tools

        Returns:
            Task type: "analysis", "action", or "creation"
        """
        # Start with keyword classification
        task_type = self.classify_task_keywords(task)

        # Adjust based on context if provided
        if context:
            # EXECUTING stage suggests action
            if context.stage and "EXECUTING" in str(context.stage).upper():
                if action_count := sum(1 for kw in self.ACTION_KEYWORDS if kw in task.lower()):
                    if action_count > 0:
                        return "action"

            # Recent tools can indicate continuing work
            if context.recent_tools:
                recent_write = any(
                    tool in ["write", "edit", "bash"] for tool in context.recent_tools
                )
                if recent_write and task_type != "analysis":
                    return "action"

        return task_type

    def should_use_tools(
        self,
        message: str,
        model_supports_tools: bool = True,
    ) -> bool:
        """Determine if tools should be used for a message.

        Analyzes the message to determine if tool use is appropriate.

        Args:
            message: User message
            model_supports_tools: Whether the model supports tool calling

        Returns:
            True if tools should be used
        """
        if not message or not message.strip():
            return False

        # Check for explicit tool mentions
        mentioned = self.detect_mentioned_tools(message)
        if mentioned:
            return True

        # Check for tool-related keywords
        tool_keywords = [
            "use",
            "run",
            "execute",
            "call",
            "invoke",
            "search",
            "find",
            "read",
            "write",
            "list",
            "check",
        ]
        message_lower = message.lower()
        if any(keyword in message_lower for keyword in tool_keywords):
            return True

        # Check if it's a question (may not need tools)
        if message.strip().endswith("?"):
            # Still might need tools (e.g., "Find the file?")
            return any(keyword in message_lower for keyword in ["find", "search", "list"])

        return False

    def extract_required_files(
        self,
        prompt: str,
    ) -> Set[str]:
        """Extract required files from a prompt.

        Parses the prompt to find file paths that are explicitly mentioned
        or implied as dependencies.

        Args:
            prompt: Prompt text to parse

        Returns:
            Set of required file paths
        """
        if not prompt:
            return set()

        files = set()

        # Pattern for file paths (Unix and Windows)
        file_patterns = [
            r'[\'"]?([a-zA-Z0-9_\-./\\]+\.[a-zA-Z0-9]+)[\'"]?',  # files.ext
            r'[\'"]?([a-zA-Z0-9_\-./\\]+/)[\'"]?',  # paths/
        ]

        for pattern in file_patterns:
            matches = re.findall(pattern, prompt)
            files.update(matches)

        # Filter to reasonable file paths
        valid_files = set()
        for f in files:
            # Must have reasonable length and contain path separator or extension
            if len(f) < 200 and ("/" in f or "\\" in f or "." in f):
                valid_files.add(f)

        return valid_files

    def extract_required_outputs(
        self,
        prompt: str,
    ) -> Set[str]:
        """Extract required outputs from a prompt.

        Parses the prompt to find output specifications (file paths,
        variable names, etc.) that the task should produce.

        Args:
            prompt: Prompt text to parse

        Returns:
            Set of required output identifiers
        """
        if not prompt:
            return set()

        outputs = set()

        # Pattern for "save to X", "write to X", "output to X"
        output_patterns = [
            r'(?:save|write|output|store|export)\s+(?:to|in|as)\s+["\']?([a-zA-Z0-9_\-./\\]+)["\']?',
            r'(?:create|generate|produce)\s+["\']?([a-zA-Z0-9_\-./\\]+)["\']?',
        ]

        for pattern in output_patterns:
            matches = re.findall(pattern, prompt, re.IGNORECASE)
            outputs.update(matches)

        return outputs


__all__ = [
    "ToolSelectionCoordinator",
    "IToolSelectionCoordinator",
]
