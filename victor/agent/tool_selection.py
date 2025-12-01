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

"""Tool selection configuration and utilities for Victor.

This module centralizes tool selection configuration including:
- Core tool definitions (always-available tools)
- Tool categories and their associated tools
- Keyword-to-category mappings
- Small model indicators for adaptive selection
- ToolSelector class for unified tool selection logic
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from victor.agent.conversation_state import ConversationStateMachine
    from victor.providers.base import ToolDefinition
    from victor.tools.base import ToolRegistry
    from victor.tools.semantic_selector import SemanticToolSelector

logger = logging.getLogger(__name__)


# Core tools that are almost always useful (filesystem, bash, editor)
CORE_TOOLS: Set[str] = {
    "read_file",
    "write_file",
    "list_directory",
    "execute_bash",
    "edit_files",
}

# Tool categories by use case
TOOL_CATEGORIES: Dict[str, List[str]] = {
    "git": ["git", "git_suggest_commit", "git_create_pr"],
    "testing": ["testing_generate", "testing_run", "testing_coverage"],
    "refactor": [
        "refactor_extract_function",
        "refactor_inline_variable",
        "refactor_organize_imports",
    ],
    "security": ["security_scan"],
    "docs": ["generate_docs", "analyze_docs"],
    "review": ["code_review"],
    "web": ["web_search", "web_fetch", "web_summarize"],
    "docker": ["docker"],
    "metrics": ["analyze_metrics"],
    "batch": ["batch"],
    "cicd": ["cicd"],
    "scaffold": ["scaffold"],
    "plan": ["plan_files"],
    "search": ["code_search"],
}

# Web-related tools that should be included when web search is detected
WEB_TOOLS: Set[str] = {"web_search", "web_summarize", "web_fetch"}

# Keyword mappings to categories
CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "git": ["git", "commit", "branch", "merge", "repository"],
    "testing": ["test", "pytest", "unittest", "coverage"],
    "refactor": ["refactor", "rename", "extract", "reorganize"],
    "security": ["security", "vulnerability", "secret", "scan"],
    "docs": ["document", "docstring", "readme", "api doc"],
    "review": ["review", "analyze code", "check code", "code quality"],
    "web": [
        "search web",
        "search the web",
        "look up",
        "find online",
        "search for",
        "web search",
        "online search",
    ],
    "docker": ["docker", "container", "image"],
    "metrics": ["complexity", "metrics", "maintainability", "technical debt"],
    "batch": ["batch", "bulk", "multiple files", "search files", "replace across"],
    "cicd": [
        "ci/cd",
        "cicd",
        "pipeline",
        "github actions",
        "gitlab ci",
        "circleci",
        "workflow",
    ],
    "scaffold": ["scaffold", "template", "boilerplate", "new project", "create project"],
    "plan": ["plan", "which files", "pick files", "where to start"],
    "search": ["search code", "code search", "find file", "locate code", "where is"],
}

# Web-related keywords for explicit web tool inclusion
WEB_KEYWORDS: List[str] = ["search", "web", "online", "lookup", "http", "https"]

# Small model indicators (for limiting tools)
SMALL_MODEL_INDICATORS: List[str] = [":0.5b", ":1.5b", ":3b"]

# Default thresholds for semantic selection
DEFAULT_THRESHOLD: float = 0.18
DEFAULT_MAX_TOOLS: int = 10
MIN_THRESHOLD: float = 0.10
MAX_THRESHOLD: float = 0.40
MIN_TOOLS: int = 5
MAX_TOOLS: int = 15


@dataclass
class ToolSelectionStats:
    """Statistics for tool selection tracking."""

    semantic_selections: int = 0
    keyword_selections: int = 0
    fallback_selections: int = 0
    total_tools_selected: int = 0
    total_tools_executed: int = 0

    def record_selection(self, method: str, num_tools: int) -> None:
        """Record a tool selection event.

        Args:
            method: Selection method ('semantic', 'keyword', 'fallback')
            num_tools: Number of tools selected
        """
        if method == "semantic":
            self.semantic_selections += 1
        elif method == "keyword":
            self.keyword_selections += 1
        elif method == "fallback":
            self.fallback_selections += 1

        self.total_tools_selected += num_tools

    def record_execution(self) -> None:
        """Record a tool execution event."""
        self.total_tools_executed += 1

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary."""
        return {
            "semantic_selections": self.semantic_selections,
            "keyword_selections": self.keyword_selections,
            "fallback_selections": self.fallback_selections,
            "total_tools_selected": self.total_tools_selected,
            "total_tools_executed": self.total_tools_executed,
        }


def detect_categories_from_message(message: str) -> Set[str]:
    """Detect relevant tool categories from user message.

    Args:
        message: User message text

    Returns:
        Set of detected category names
    """
    message_lower = message.lower()
    selected_categories: Set[str] = set()

    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in message_lower for kw in keywords):
            selected_categories.add(category)

    return selected_categories


def get_tools_for_categories(categories: Set[str]) -> Set[str]:
    """Get tool names for the given categories.

    Args:
        categories: Set of category names

    Returns:
        Set of tool names from all categories
    """
    tools: Set[str] = set()
    for category in categories:
        if category in TOOL_CATEGORIES:
            tools.update(TOOL_CATEGORIES[category])
    return tools


def is_small_model(model_name: str, provider_name: str = "") -> bool:
    """Check if a model is considered small (for tool limiting).

    Args:
        model_name: Model name/identifier
        provider_name: Optional provider name

    Returns:
        True if model is small
    """
    model_lower = model_name.lower()

    # Only apply small model logic for Ollama currently
    if provider_name.lower() == "ollama":
        return any(indicator in model_lower for indicator in SMALL_MODEL_INDICATORS)

    return False


def needs_web_tools(message: str) -> bool:
    """Check if message suggests web tools are needed.

    Args:
        message: User message text

    Returns:
        True if web tools should be included
    """
    message_lower = message.lower()
    return any(kw in message_lower for kw in WEB_KEYWORDS)


def calculate_adaptive_threshold(
    model_name: str,
    query_word_count: int,
    conversation_depth: int,
    base_threshold: float = DEFAULT_THRESHOLD,
    base_max_tools: int = DEFAULT_MAX_TOOLS,
) -> Tuple[float, int]:
    """Calculate adaptive threshold and max_tools based on context.

    Args:
        model_name: Model name for size-based adjustments
        query_word_count: Number of words in query
        conversation_depth: Number of messages in conversation
        base_threshold: Starting threshold value
        base_max_tools: Starting max tools value

    Returns:
        Tuple of (threshold, max_tools)
    """
    threshold = base_threshold
    max_tools = base_max_tools
    model_lower = model_name.lower() if model_name else ""

    # Factor 1: Model size estimation
    if any(size in model_lower for size in [":70b", ":72b", ":65b", "qwen3-coder:30b"]):
        threshold -= 0.05  # Larger models can handle more
        max_tools = min(MAX_TOOLS, max_tools + 3)
    elif any(size in model_lower for size in [":7b", ":8b"]):
        threshold += 0.03  # Medium models
    elif any(size in model_lower for size in [":3b", ":1b", ":0.5b"]):
        threshold += 0.08  # Small models need stricter selection
        max_tools = max(MIN_TOOLS, max_tools - 3)

    # Factor 2: Query complexity
    if query_word_count < 5:
        threshold += 0.05  # Vague query → stricter
    elif query_word_count > 20:
        threshold -= 0.05  # Detailed query → looser
        max_tools = min(MAX_TOOLS, max_tools + 2)

    # Factor 3: Conversation depth
    if conversation_depth > 15:
        threshold -= 0.05  # Deep conversation → looser
        max_tools = min(MAX_TOOLS, max_tools + 1)
    elif conversation_depth > 8:
        threshold -= 0.03  # Moderate conversation

    # Clamp values
    threshold = max(MIN_THRESHOLD, min(MAX_THRESHOLD, threshold))
    max_tools = max(MIN_TOOLS, min(MAX_TOOLS, max_tools))

    return threshold, max_tools


def select_tools_by_keywords(
    message: str,
    all_tool_names: Set[str],
    is_small: bool = False,
    max_tools_for_small: int = 10,
) -> Set[str]:
    """Select tools using keyword matching.

    Args:
        message: User message
        all_tool_names: Set of all available tool names
        is_small: Whether this is a small model
        max_tools_for_small: Max tools for small models

    Returns:
        Set of selected tool names
    """
    # Start with core tools
    selected = CORE_TOOLS.copy()

    # Add category-based tools
    categories = detect_categories_from_message(message)
    category_tools = get_tools_for_categories(categories)
    selected.update(category_tools)

    # Filter to only available tools
    selected = selected.intersection(all_tool_names)

    # Limit for small models
    if is_small and len(selected) > max_tools_for_small:
        # Keep core tools, limit others
        core_in_selected = selected.intersection(CORE_TOOLS)
        others = list(selected - CORE_TOOLS)
        max_others = max(0, max_tools_for_small - len(core_in_selected))
        selected = core_in_selected.union(set(others[:max_others]))

    return selected


class ToolSelector:
    """Unified tool selection with semantic and keyword-based approaches.

    This class encapsulates tool selection logic extracted from AgentOrchestrator,
    providing a cleaner separation of concerns and easier testing.

    Supports:
    - Semantic selection using embeddings (via SemanticToolSelector)
    - Keyword-based selection using category matching
    - Stage-aware prioritization using ConversationStateMachine
    - Adaptive thresholds based on model size and query complexity

    Example:
        selector = ToolSelector(
            tools=tool_registry,
            semantic_selector=semantic_selector,
            model="qwen2.5-coder:7b",
        )
        tools = await selector.select_tools(user_message, use_semantic=True)
    """

    def __init__(
        self,
        tools: "ToolRegistry",
        semantic_selector: Optional["SemanticToolSelector"] = None,
        conversation_state: Optional["ConversationStateMachine"] = None,
        model: str = "",
        provider_name: str = "",
        tool_selection_config: Optional[Dict[str, Any]] = None,
        fallback_max_tools: int = 8,
        on_selection_recorded: Optional[Callable[[str, int], None]] = None,
    ):
        """Initialize the tool selector.

        Args:
            tools: Tool registry containing available tools
            semantic_selector: Optional semantic selector for embedding-based selection
            conversation_state: Optional conversation state machine for stage detection
            model: Model name (used for adaptive thresholds)
            provider_name: Provider name (used for small model detection)
            tool_selection_config: Optional config with base_threshold, base_max_tools
            fallback_max_tools: Max tools for fallback selection
            on_selection_recorded: Optional callback for recording selection stats
        """
        self.tools = tools
        self.semantic_selector = semantic_selector
        self.conversation_state = conversation_state
        self.model = model
        self.provider_name = provider_name
        self.tool_selection_config = tool_selection_config or {}
        self.fallback_max_tools = fallback_max_tools
        self._on_selection_recorded = on_selection_recorded

        # Track whether embeddings have been initialized
        self._embeddings_initialized = False

        # Selection statistics
        self.stats = ToolSelectionStats()

    def _record_selection(self, method: str, num_tools: int) -> None:
        """Record a selection event.

        Args:
            method: Selection method used
            num_tools: Number of tools selected
        """
        self.stats.record_selection(method, num_tools)
        if self._on_selection_recorded:
            self._on_selection_recorded(method, num_tools)

    def get_adaptive_threshold(
        self, user_message: str, conversation_depth: int = 0
    ) -> Tuple[float, int]:
        """Calculate adaptive similarity threshold and max_tools based on context.

        Adapts based on:
        1. Model size (from config or detected from model name)
        2. Query specificity (vague queries need stricter thresholds)
        3. Conversation depth (deeper conversations can be more permissive)

        Args:
            user_message: The user's input message
            conversation_depth: Number of messages in conversation

        Returns:
            Tuple of (similarity_threshold, max_tools)
        """
        # Factor 1: Model size - Check configuration first, then fall back to detection
        if self.tool_selection_config and "base_threshold" in self.tool_selection_config:
            base_threshold = self.tool_selection_config.get("base_threshold", DEFAULT_THRESHOLD)
            base_max_tools = self.tool_selection_config.get("base_max_tools", DEFAULT_MAX_TOOLS)
            logger.debug(
                f"Using configured tool selection: threshold={base_threshold:.2f}, "
                f"max_tools={base_max_tools}"
            )
        else:
            # Fall back to model name pattern detection
            model_lower = self.model.lower()

            # Detect model size from common naming patterns
            if any(size in model_lower for size in [":0.5b", ":1b", ":1.5b", ":3b"]):
                base_threshold = 0.35
                base_max_tools = 5
            elif any(size in model_lower for size in [":7b", ":8b"]):
                base_threshold = 0.25
                base_max_tools = 7
            elif any(size in model_lower for size in [":13b", ":14b", ":15b"]):
                base_threshold = 0.20
                base_max_tools = 10
            elif any(size in model_lower for size in [":30b", ":32b", ":34b", ":70b", ":72b"]):
                base_threshold = 0.15
                base_max_tools = 12
            else:
                base_threshold = DEFAULT_THRESHOLD
                base_max_tools = DEFAULT_MAX_TOOLS

            logger.debug(
                f"Detected tool selection from model name '{model_lower}': "
                f"threshold={base_threshold:.2f}, max_tools={base_max_tools}"
            )

        # Factor 2: Query specificity
        word_count = len(user_message.split())

        if word_count < 5:
            base_threshold += 0.10
            base_max_tools = max(MIN_TOOLS, base_max_tools - 2)
        elif word_count < 10:
            base_threshold += 0.05
        elif word_count > 20:
            base_threshold -= 0.05
            base_max_tools = min(MAX_TOOLS, base_max_tools + 2)

        # Factor 3: Conversation depth
        if conversation_depth > 15:
            base_threshold -= 0.05
            base_max_tools = min(MAX_TOOLS, base_max_tools + 1)
        elif conversation_depth > 8:
            base_threshold -= 0.03

        # Clamp values
        threshold = max(MIN_THRESHOLD, min(MAX_THRESHOLD, base_threshold))
        max_tools = max(MIN_TOOLS, min(MAX_TOOLS, base_max_tools))

        logger.debug(
            f"Adaptive threshold: {threshold:.2f}, max_tools: {max_tools} "
            f"(model: {self.model}, words: {word_count}, depth: {conversation_depth})"
        )

        return threshold, max_tools

    async def select_tools(
        self,
        user_message: str,
        use_semantic: bool = True,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        conversation_depth: int = 0,
        planned_tools: Optional[List["ToolDefinition"]] = None,
    ) -> List["ToolDefinition"]:
        """Select tools using the best available method.

        Main entry point for tool selection. Uses semantic selection if available
        and enabled, otherwise falls back to keyword-based selection.

        Args:
            user_message: The user's input message
            use_semantic: Whether to use semantic selection (if available)
            conversation_history: Optional conversation history for context
            conversation_depth: Number of messages in conversation
            planned_tools: Optional pre-planned tools to include

        Returns:
            List of relevant ToolDefinition objects
        """
        if use_semantic and self.semantic_selector:
            return await self.select_semantic(
                user_message,
                conversation_history=conversation_history,
                conversation_depth=conversation_depth,
                planned_tools=planned_tools,
            )
        else:
            return self.select_keywords(user_message, planned_tools=planned_tools)

    async def select_semantic(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        conversation_depth: int = 0,
        planned_tools: Optional[List["ToolDefinition"]] = None,
    ) -> List["ToolDefinition"]:
        """Select tools using embedding-based semantic similarity.

        Args:
            user_message: The user's input message
            conversation_history: Optional conversation history for context
            conversation_depth: Number of messages in conversation
            planned_tools: Optional pre-planned tools to include

        Returns:
            List of relevant ToolDefinition objects based on semantic similarity
        """
        # Import here to avoid circular imports
        from victor.providers.base import ToolDefinition

        if not self.semantic_selector:
            return self.select_keywords(user_message, planned_tools=planned_tools)

        # Initialize embeddings on first call
        if not self._embeddings_initialized:
            logger.info("Initializing tool embeddings (one-time operation)...")
            await self.semantic_selector.initialize_tool_embeddings(self.tools)
            self._embeddings_initialized = True

        # Get adaptive threshold and max_tools
        threshold, max_tools = self.get_adaptive_threshold(user_message, conversation_depth)

        # Select tools with context awareness
        tools = await self.semantic_selector.select_relevant_tools_with_context(
            user_message=user_message,
            tools=self.tools,
            conversation_history=conversation_history,
            max_tools=max_tools,
            similarity_threshold=threshold,
        )

        # Blend with keyword-selected tools
        keyword_tools = self.select_keywords(user_message, planned_tools=planned_tools)
        if keyword_tools:
            existing = {t.name for t in tools}
            tools.extend([t for t in keyword_tools if t.name not in existing])

        # Ensure web tools if explicitly mentioned
        message_lower = user_message.lower()
        if any(kw in message_lower for kw in WEB_KEYWORDS):
            must_have = WEB_TOOLS
            existing = {t.name for t in tools}
            for tool in self.tools.list_tools():
                if tool.name in must_have and tool.name not in existing:
                    tools.append(
                        ToolDefinition(
                            name=tool.name,
                            description=tool.description,
                            parameters=tool.parameters,
                        )
                    )

        # Deduplicate
        dedup: Dict[str, ToolDefinition] = {}
        for t in tools:
            dedup[t.name] = t
        tools = list(dedup.values())

        logger.info(
            f"Semantic+keyword tools selected ({len(tools)}): "
            f"{', '.join(t.name for t in tools)}"
        )

        # Smart fallback if 0 tools
        if not tools:
            logger.warning(
                "Semantic selection returned 0 tools. "
                "Using smart fallback: core tools + keyword matching."
            )
            tools = self._get_fallback_tools(user_message)
            self._record_selection("fallback", len(tools))
        else:
            self._record_selection("semantic", len(tools))

        return tools

    def select_keywords(
        self,
        user_message: str,
        planned_tools: Optional[List["ToolDefinition"]] = None,
    ) -> List["ToolDefinition"]:
        """Select tools using keyword-based category matching.

        Args:
            user_message: The user's input message
            planned_tools: Optional pre-planned tools to include

        Returns:
            List of relevant ToolDefinition objects
        """
        from victor.providers.base import ToolDefinition

        all_tools = list(self.tools.list_tools())

        # Start with planned tools if provided
        selected_tools: List[ToolDefinition] = list(planned_tools) if planned_tools else []
        existing_names = {t.name for t in selected_tools}

        # Detect categories from message
        selected_categories = detect_categories_from_message(user_message)

        # Build selected tool names: core tools + category-specific tools
        selected_tool_names = CORE_TOOLS.copy()
        selected_tool_names.update(get_tools_for_categories(selected_categories))

        # Check if this is a small model
        small_model = is_small_model(self.model, self.provider_name)

        # Filter tools
        for tool in all_tools:
            if tool.name in selected_tool_names and tool.name not in existing_names:
                selected_tools.append(
                    ToolDefinition(
                        name=tool.name,
                        description=tool.description,
                        parameters=tool.parameters,
                    )
                )
                existing_names.add(tool.name)

        # For small models, limit to max 10 tools
        if small_model and len(selected_tools) > 10:
            core_tools = [t for t in selected_tools if t.name in CORE_TOOLS]
            other_tools = [t for t in selected_tools if t.name not in CORE_TOOLS]
            selected_tools = core_tools + other_tools[: max(0, 10 - len(core_tools))]

        tool_names = [t.name for t in selected_tools]
        logger.info(
            f"Selected {len(selected_tools)} tools (small_model={small_model}): "
            f"{', '.join(tool_names)}"
        )

        self._record_selection("keyword", len(selected_tools))
        return selected_tools

    def prioritize_by_stage(
        self,
        user_message: str,
        tools: Optional[List["ToolDefinition"]],
    ) -> Optional[List["ToolDefinition"]]:
        """Stage-aware pruning of tool list to keep it focused per step.

        Uses ConversationStateMachine for intelligent stage detection.

        Args:
            user_message: The user's message
            tools: List of tool definitions to filter

        Returns:
            Filtered list of tools appropriate for current stage
        """
        if not tools:
            return tools

        if not self.conversation_state:
            return tools

        # Record user message for stage detection
        self.conversation_state.record_message(user_message, is_user=True)

        # Use conversation state machine for stage detection
        current_stage = self.conversation_state.get_stage()
        stage_tools = self.conversation_state.get_stage_tools()

        logger.debug(f"Stage detection: {current_stage.name}, " f"recommended tools: {stage_tools}")

        # Core tools always included
        core = CORE_TOOLS

        # Web tools check
        web_tools = WEB_TOOLS if needs_web_tools(user_message) else set()

        # Combine stage-specific tools with core and web tools
        keep = stage_tools | core | web_tools

        # Also get tools from adjacent stages for flexibility
        for tool in tools:
            if self.conversation_state.should_include_tool(tool.name):
                keep.add(tool.name)

        # Apply priority boost based on stage
        boosted_tools: List[Tuple["ToolDefinition", float]] = []
        for tool in tools:
            boost = self.conversation_state.get_tool_priority_boost(tool.name)
            if tool.name in keep or boost > 0:
                boosted_tools.append((tool, boost))

        # Sort by boost (descending) and filter
        boosted_tools.sort(key=lambda x: x[1], reverse=True)
        pruned = [t for t, _ in boosted_tools if t.name in keep]

        if pruned:
            logger.info(
                f"Stage-pruned tools ({current_stage.name}): "
                f"{len(pruned)} tools kept from {len(tools)}"
            )
            return pruned

        # Fallback to core tools
        core_fallback = CORE_TOOLS
        fallback_tools = [t for t in tools if t.name in core_fallback]

        if fallback_tools:
            logger.info(f"Stage pruning fallback: {len(fallback_tools)} core tools")
            return fallback_tools

        # Last resort: return a small prefix
        logger.warning(f"Stage pruning: last resort fallback to {self.fallback_max_tools} tools")
        return tools[: self.fallback_max_tools]

    def _get_fallback_tools(self, user_message: str) -> List["ToolDefinition"]:
        """Get fallback tools when semantic selection returns 0 results.

        Args:
            user_message: The user's input message

        Returns:
            List of core + keyword-selected tools
        """
        from victor.providers.base import ToolDefinition

        all_tools_map = {tool.name: tool for tool in self.tools.list_tools()}
        tools: List[ToolDefinition] = []

        # Add core tools first
        for tool_name in CORE_TOOLS:
            if tool_name in all_tools_map:
                tool = all_tools_map[tool_name]
                tools.append(
                    ToolDefinition(
                        name=tool.name,
                        description=tool.description,
                        parameters=tool.parameters,
                    )
                )

        # Add keyword-selected tools (avoiding duplicates)
        keyword_tools = self.select_keywords(user_message)
        existing_names = {t.name for t in tools}
        for keyword_tool in keyword_tools:
            if keyword_tool.name not in existing_names:
                tools.append(keyword_tool)

        logger.info(
            f"Smart fallback selected {len(tools)} tools: " f"{', '.join(t.name for t in tools)}"
        )

        return tools
