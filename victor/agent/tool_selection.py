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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Union

import yaml

from victor.agent.conversation_state import ConversationStage
from victor.tools.enums import SchemaLevel
from victor.protocols.mode_aware import ModeAwareMixin
from victor.tools.base import AccessMode, ExecutionCategory

# Rust-accelerated pattern matching (with Python fallback)
_RUST_PATTERN_MATCHING = False
try:
    from victor.processing.native import find_all_patterns

    _RUST_PATTERN_MATCHING = True
except ImportError:
    pass

if TYPE_CHECKING:
    from victor.agent.conversation_state import ConversationStateMachine
    from victor.agent.milestone_monitor import TaskMilestoneMonitor, TaskToolConfigLoader
    from victor.agent.unified_task_tracker import UnifiedTaskTracker
    from victor.agent.vertical_context import VerticalContext
    from victor.core.verticals.protocols import (
        ToolSelectionContext,
        ToolSelectionResult,
        ToolSelectionStrategyProtocol,
    )
    from victor.providers.base import ToolDefinition
    from victor.tools.base import ToolRegistry
    from victor.tools.semantic_selector import SemanticToolSelector

logger = logging.getLogger(__name__)


# Fallback critical tools for cases where registry is unavailable.
# Critical tools are detected via priority=Priority.CRITICAL in @tool decorator.
_FALLBACK_CRITICAL_TOOLS: Set[str] = {
    "read",  # read_file → read
    "write",  # write_file → write
    "ls",  # list_directory → ls
    "shell",  # execute_bash → shell
    "edit",  # edit_files → edit
    "search",  # code_search → search (always needed for code exploration)
}


def get_critical_tools(registry: Optional["ToolRegistry"] = None) -> Set[str]:
    """Dynamically discover critical tools from registry using priority-based detection.

    Critical tools are those with priority=Priority.CRITICAL in their @tool decorator.
    These tools are always available for selection regardless of task type.
    Falls back to hardcoded list if registry is unavailable.

    Args:
        registry: Optional tool registry to query

    Returns:
        Set of canonical tool names that are critical priority tools
    """
    if registry is None:
        return _FALLBACK_CRITICAL_TOOLS.copy()

    critical_tools: Set[str] = set()
    for tool in registry.list_tools(only_enabled=True):
        # Use is_critical property which checks priority=Priority.CRITICAL
        if hasattr(tool, "is_critical") and tool.is_critical:
            critical_tools.add(tool.name)

    # Fallback if no critical tools found (shouldn't happen with proper setup)
    if not critical_tools:
        logger.warning(
            "No tools with priority=Priority.CRITICAL found in registry. "
            "Using fallback critical tools. Check tool decorator definitions."
        )
        return _FALLBACK_CRITICAL_TOOLS.copy()

    return critical_tools


def get_tools_by_category(
    registry: Optional["ToolRegistry"] = None, category: str = ""
) -> Set[str]:
    """Dynamically discover tools in a specific category from registry.

    Tools declare their category via @tool(category="...") decorator.

    Args:
        registry: Tool registry to query
        category: Category name to filter by (e.g., "git", "web", "testing")

    Returns:
        Set of tool names in the specified category
    """
    if registry is None or not category:
        return set()

    tools: Set[str] = set()
    for tool in registry.list_tools(only_enabled=True):
        if hasattr(tool, "category") and tool.category == category:
            tools.add(tool.name)

    return tools


def get_all_categories(registry: Optional["ToolRegistry"] = None) -> Set[str]:
    """Get all unique categories from tools in registry.

    Args:
        registry: Tool registry to query

    Returns:
        Set of unique category names
    """
    if registry is None:
        return set()

    categories: Set[str] = set()
    for tool in registry.list_tools(only_enabled=True):
        if hasattr(tool, "category") and tool.category:
            categories.add(tool.category)

    return categories


def get_category_to_tools_map(
    registry: Optional["ToolRegistry"] = None,
) -> Dict[str, Set[str]]:
    """Build a mapping from categories to tool names.

    Dynamically discovers categories from tool metadata. Tools must declare
    their category via @tool(category="...") decorator.

    Args:
        registry: Tool registry to query (required for proper operation)

    Returns:
        Dict mapping category name to set of tool names.
        Empty dict if registry is None.

    Raises:
        ValueError: If registry is provided but no tools have categories
    """
    if registry is None:
        logger.warning(
            "get_category_to_tools_map called without registry. "
            "Pass a ToolRegistry for proper category discovery."
        )
        return {}

    category_map: Dict[str, Set[str]] = {}
    for tool in registry.list_tools(only_enabled=True):
        if hasattr(tool, "category") and tool.category:
            category = tool.category
            if category not in category_map:
                category_map[category] = set()
            category_map[category].add(tool.name)

    if not category_map:
        logger.warning(
            "No tools with categories found in registry. "
            "Add category metadata to @tool decorators."
        )

    return category_map


# Fallback category keywords for goal inference.
# DEPRECATED: Use registry.detect_categories_from_text() which builds this
# dynamically from @tool(keywords=[...]) decorators.
_FALLBACK_CATEGORY_KEYWORDS: Dict[str, Set[str]] = {
    "security": {"security", "vulnerability", "scan", "audit", "cve", "exploit", "owasp"},
    "metrics": {"metrics", "complexity", "coverage", "analyze", "statistics", "cyclomatic"},
    "testing": {"test", "unittest", "pytest", "spec", "coverage", "mock"},
    "git": {"git", "commit", "branch", "merge", "push", "pull", "rebase"},
    "docker": {"docker", "container", "dockerfile", "compose", "image"},
    "web": {"web", "http", "api", "fetch", "url", "search"},
}

# Alias for backward compatibility
CATEGORY_KEYWORDS = _FALLBACK_CATEGORY_KEYWORDS


def detect_categories_from_message(message: str) -> Set[str]:
    """Detect relevant tool categories from keywords in a message.

    Merges registry-based detection (decorator-driven keywords) with
    static fallback keywords to ensure comprehensive category coverage.

    Args:
        message: User message text to analyze

    Returns:
        Set of category names that match keywords in the message

    Example:
        >>> detect_categories_from_message("run a security scan")
        {'security'}
        >>> detect_categories_from_message("analyze code complexity and metrics")
        {'metrics'}
    """
    detected: Set[str] = set()

    # Try registry-based detection (decorator-driven)
    try:
        from victor.tools.metadata_registry import detect_categories_from_text

        registry_detected = detect_categories_from_text(message)
        if registry_detected:
            logger.debug(f"Registry detected categories: {registry_detected}")
            detected.update(registry_detected)
    except Exception as e:
        logger.debug(f"Registry category detection failed: {e}")

    # Always check fallback keywords to ensure coverage
    # (registry may not have all category keywords defined)
    # Use Rust Aho-Corasick when available for O(text_len) matching
    for category, keywords in _FALLBACK_CATEGORY_KEYWORDS.items():
        keywords_list = list(keywords)
        if _RUST_PATTERN_MATCHING:
            matches = find_all_patterns(message, keywords_list, case_insensitive=True)
            if matches:
                detected.add(category)
        else:
            message_lower = message.lower()
            if any(kw in message_lower for kw in keywords):
                detected.add(category)

    if detected:
        logger.debug(f"Detected categories (merged): {detected}")

    return detected


def get_tools_for_categories(
    categories: Set[str], registry: Optional["ToolRegistry"] = None
) -> Set[str]:
    """Get tool names for a set of categories.

    Aggregates tools from multiple categories into a single set.
    Uses the ToolMetadataRegistry for lookup if registry is not provided.

    Args:
        categories: Set of category names (e.g., {"security", "metrics"})
        registry: Optional tool registry to query

    Returns:
        Set of tool names belonging to any of the specified categories
    """
    tools: Set[str] = set()

    # Try registry-based lookup first
    if registry is not None:
        for category in categories:
            category_tools = get_tools_by_category(registry, category)
            tools.update(category_tools)
    else:
        # Fall back to metadata registry
        try:
            from victor.tools.metadata_registry import get_global_registry

            meta_registry = get_global_registry()
            for category in categories:
                category_tools = meta_registry.get_tools_by_category(category)
                tools.update(category_tools)
        except Exception as e:
            logger.debug(f"Failed to get tools for categories via registry: {e}")

    return tools


def get_web_tools(registry: Optional["ToolRegistry"] = None) -> Set[str]:
    """Get web-related tools from registry.

    Web tools are those with category='web'. Tools must declare their category
    via @tool(category="web") decorator.

    Args:
        registry: Tool registry to query (required for proper operation)

    Returns:
        Set of web-related tool names. Empty set if registry is None.
    """
    if registry is None:
        logger.warning(
            "get_web_tools called without registry. "
            "Pass a ToolRegistry for proper tool discovery."
        )
        return set()

    # Get tools with category="web"
    return get_tools_by_category(registry, "web")


def get_tools_with_keywords(
    registry: Optional["ToolRegistry"] = None,
    match_keywords: Optional[Set[str]] = None,
) -> Set[str]:
    """Find tools that have any of the specified keywords in their metadata.

    Args:
        registry: Tool registry to query
        match_keywords: Set of keywords to match against tool metadata

    Returns:
        Set of tool names that match any of the keywords
    """
    if registry is None or not match_keywords:
        return set()

    matching_tools: Set[str] = set()
    match_keywords_lower = {k.lower() for k in match_keywords}

    for tool in registry.list_tools(only_enabled=True):
        # Check tool metadata for keywords
        metadata = getattr(tool, "metadata", None)
        if metadata and hasattr(metadata, "keywords"):
            tool_keywords = {k.lower() for k in (metadata.keywords or [])}
            if tool_keywords & match_keywords_lower:
                matching_tools.add(tool.name)

        # Also check description for keyword matches
        description = getattr(tool, "description", "").lower()
        for kw in match_keywords_lower:
            if kw in description:
                matching_tools.add(tool.name)
                break

    return matching_tools


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


def get_tools_from_message(message: str) -> Set[str]:
    """Get tool names that match keywords in the user message.

    Uses ToolMetadataRegistry to find tools whose @tool(keywords=[...])
    match the user's message. This is the preferred method for keyword-based
    tool selection as it uses the single source of truth from tool decorators.

    Args:
        message: User message text

    Returns:
        Set of tool names whose keywords match the message
    """
    try:
        from victor.tools.metadata_registry import get_global_registry

        registry = get_global_registry()
        tools = registry.get_tools_matching_text(message)
        if tools:
            logger.debug(f"Registry keyword match found tools: {tools}")
        return tools
    except Exception as e:
        logger.debug(f"Registry unavailable for keyword matching: {e}")
        return set()


def get_tools_from_message_scored(
    message: str,
    min_score: float = 0.0,
    max_results: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """Get tool names with relevance scores from the user message.

    Enhanced version of get_tools_from_message() that returns scored results.
    Uses ToolMetadataRegistry.get_tools_matching_text_scored() for intelligent
    ranking based on:
    - Number of matching keywords
    - Keyword specificity (longer = more specific)
    - Tool priority (CRITICAL tools get boost)

    Args:
        message: User message text
        min_score: Minimum score threshold (0.0 to 1.0)
        max_results: Maximum number of results to return

    Returns:
        List of (tool_name, score) tuples sorted by score descending
    """
    try:
        from victor.tools.metadata_registry import get_global_registry

        registry = get_global_registry()
        results = registry.get_tools_matching_text_scored(
            text=message,
            min_score=min_score,
            max_results=max_results,
            use_fallback=True,
        )
        scored_tools = [(r.tool_name, r.total_score) for r in results]
        if scored_tools:
            logger.debug(f"Scored keyword match: {[(t, f'{s:.2f}') for t, s in scored_tools[:5]]}")
        return scored_tools
    except Exception as e:
        logger.debug(f"Registry unavailable for scored matching: {e}")
        return []


def get_keyword_matching_metrics() -> Optional[Dict[str, Any]]:
    """Get keyword matching metrics for observability.

    Returns:
        Dictionary with matching statistics, or None if unavailable
    """
    try:
        from victor.tools.metadata_registry import get_global_registry

        registry = get_global_registry()
        return registry.metrics.to_dict()
    except Exception:
        return None


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
    """Select tools using keyword matching via ToolMetadataRegistry.

    Uses keywords defined in @tool decorators for tool selection.
    This is the recommended approach - tools define their own keywords.

    Args:
        message: User message
        all_tool_names: Set of all available tool names
        is_small: Whether this is a small model
        max_tools_for_small: Max tools for small models

    Returns:
        Set of selected tool names
    """
    # Start with critical tools (dynamic discovery)
    core_tools = get_critical_tools()
    selected = core_tools.copy()

    # Add tools matching keywords from @tool decorators
    keyword_matches = get_tools_from_message(message)
    selected.update(keyword_matches)

    # Filter to only available tools
    selected = selected.intersection(all_tool_names)

    # Limit for small models
    if is_small and len(selected) > max_tools_for_small:
        # Keep core tools, limit others
        core_in_selected = selected.intersection(core_tools)
        others = list(selected - core_tools)
        max_others = max(0, max_tools_for_small - len(core_in_selected))
        selected = core_in_selected.union(set(others[:max_others]))

    return selected


class ToolSelector(ModeAwareMixin):
    """Unified tool selection with semantic and keyword-based approaches.

    This class encapsulates tool selection logic extracted from AgentOrchestrator,
    providing a cleaner separation of concerns and easier testing.

    Supports:
    - Semantic selection using embeddings (via SemanticToolSelector)
    - Keyword-based selection using category matching
    - Stage-aware prioritization using ConversationStateMachine
    - Adaptive thresholds based on model size and query complexity
    - Mode-aware filtering (BUILD/PLAN/EXPLORE modes)

    Uses ModeAwareMixin for consistent mode controller access (via self.is_build_mode,
    self.exploration_multiplier, etc.).

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
        task_tracker: Optional[Union["TaskMilestoneMonitor", "UnifiedTaskTracker"]] = None,
        model: str = "",
        provider_name: str = "",
        tool_selection_config: Optional[Dict[str, Any]] = None,
        fallback_max_tools: int = 8,
        on_selection_recorded: Optional[Callable[[str, int], None]] = None,
        vertical_context: Optional["VerticalContext"] = None,
    ):
        """Initialize the tool selector.

        Args:
            tools: Tool registry containing available tools
            semantic_selector: Optional semantic selector for embedding-based selection
            conversation_state: Optional conversation state machine for stage detection
            task_tracker: Optional task progress tracker (TaskMilestoneMonitor or UnifiedTaskTracker)
            model: Model name (used for adaptive thresholds)
            provider_name: Provider name (used for small model detection)
            tool_selection_config: Optional config with base_threshold, base_max_tools
            fallback_max_tools: Max tools for fallback selection
            on_selection_recorded: Optional callback for recording selection stats
            vertical_context: Optional vertical context for vertical-specific tool selection
        """
        self.tools = tools
        self.semantic_selector = semantic_selector
        self.conversation_state = conversation_state
        self.task_tracker = task_tracker
        self.model = model
        self.provider_name = provider_name
        self.tool_selection_config = tool_selection_config or {}
        self.fallback_max_tools = fallback_max_tools
        self._on_selection_recorded = on_selection_recorded

        # Vertical context for vertical-specific tool selection (DIP)
        self._vertical_context: Optional["VerticalContext"] = vertical_context

        # Task tool config loader for YAML-based configuration
        self._task_config_loader: Optional["TaskToolConfigLoader"] = None

        # Track whether embeddings have been initialized
        self._embeddings_initialized = False

        # Selection statistics
        self.stats = ToolSelectionStats()

        # Cache last selection to avoid redundant logging
        self._last_selection: Optional[Set[str]] = None

        # Cached core and web tools (lazy loaded for dynamic discovery)
        self._cached_core_tools: Optional[Set[str]] = None
        self._cached_core_readonly: Optional[Set[str]] = None
        self._cached_web_tools: Optional[Set[str]] = None

        # Enabled tools filter (set by vertical, None = all tools)
        self._enabled_tools: Optional[Set[str]] = None

        # Populate global metadata registry for keyword-based tool selection
        self._populate_metadata_registry()

    def _populate_metadata_registry(self) -> None:
        """Populate the global metadata registry with tools from this selector.

        This enables keyword-based tool lookup via get_tools_for_categories().
        Only runs once per ToolSelector instance.
        """
        try:
            from victor.tools.metadata_registry import get_global_registry

            registry = get_global_registry()

            # Register all tools from our tool registry
            for tool in self.tools.list_tools(only_enabled=True):
                registry.register(tool)

            logger.debug(
                f"Populated metadata registry with {len(registry)} tools "
                f"({len(registry.get_all_keywords())} keywords)"
            )
        except Exception as e:
            logger.warning(f"Failed to populate metadata registry: {e}")

    def _load_vertical_config(self) -> Dict[str, Any]:
        """Load vertical tool configurations from YAML.

        Returns:
            Dict with vertical configurations, or empty dict if not found
        """
        if not hasattr(self, "_vertical_config_cache"):
            self._vertical_config_cache: Optional[Dict[str, Any]] = None

        if self._vertical_config_cache is not None:
            return self._vertical_config_cache

        config_path = Path(__file__).parent.parent / "config" / "vertical_tools.yaml"
        try:
            if config_path.exists():
                self._vertical_config_cache = yaml.safe_load(config_path.read_text())
                logger.debug(f"Loaded vertical tools config from {config_path}")
            else:
                self._vertical_config_cache = {}
                logger.debug("No vertical_tools.yaml found, using defaults")
        except Exception as e:
            logger.warning(f"Failed to load vertical_tools.yaml: {e}")
            self._vertical_config_cache = {}

        return self._vertical_config_cache

    def _get_vertical_core_tools(self, vertical: Optional[str] = None) -> Set[str]:
        """Get core tools for a specific vertical.

        Args:
            vertical: Vertical name (coding, devops, research, dataanalysis, rag)

        Returns:
            Set of tool names that are core for this vertical
        """
        config = self._load_vertical_config()
        if not config:
            return self._get_core_tools_cached()

        # Get vertical-specific config or use default
        vert_config = config.get("verticals", {}).get(vertical or "", {})
        if not vert_config:
            vert_config = config.get("default", {})

        core = set(vert_config.get("core_tools", []))
        core.update(vert_config.get("vertical_tools", []))
        return core

    def get_tools_with_levels(
        self,
        user_message: str,
        vertical: Optional[str] = None,
    ) -> Dict[str, SchemaLevel]:
        """Return tools with their schema levels for token-efficient broadcasting.

        This method implements tiered schema selection:
        - Core tools get FULL schema (complete description, all params)
        - Vertical-specific tools get COMPACT schema (shorter descriptions, ~20% reduction)
        - Semantic pool tools get STUB schema (minimal description, required params)

        Args:
            user_message: The user's message for semantic matching
            vertical: Optional vertical name (coding, devops, research, etc.)

        Returns:
            Dict mapping tool_name -> SchemaLevel
        """
        config = self._load_vertical_config()

        # Get vertical-specific config or default
        vert_config = config.get("verticals", {}).get(vertical or "", {})
        if not vert_config:
            vert_config = config.get("default", {})

        # Core tools always get FULL schema
        core_tools = set(vert_config.get("core_tools", []))
        if not core_tools:
            # Fallback to dynamic discovery
            core_tools = self._get_core_tools_cached()

        # Vertical-specific tools get COMPACT schema (~20% token reduction)
        vertical_core = set(vert_config.get("vertical_tools", []))

        # Semantic pool gets STUB schema
        semantic_pool = set(vert_config.get("semantic_pool", []))
        max_semantic = vert_config.get("max_semantic_pool", 5)

        # Build levels dict
        levels: Dict[str, SchemaLevel] = {}

        # FULL for core tools only
        for tool in core_tools:
            levels[tool] = SchemaLevel.FULL

        # COMPACT for vertical tools (~20% reduction from FULL)
        for tool in vertical_core:
            if tool not in levels:  # Don't override core tools
                levels[tool] = SchemaLevel.COMPACT

        # STUB for semantic pool (limited by max_semantic_pool)
        # Filter semantic pool by keyword/semantic matching
        message_lower = user_message.lower()
        matched_semantic = []

        for tool_name in semantic_pool:
            if tool_name in levels:
                continue  # Skip if already in core/vertical

            # Simple keyword matching for semantic pool
            tool = None
            for t in self.tools.list_tools():
                if t.name == tool_name:
                    tool = t
                    break

            if tool:
                # Check keywords from metadata
                should_include = False
                if hasattr(tool, "metadata") and tool.metadata:
                    keywords = getattr(tool.metadata, "keywords", []) or []
                    if any(kw.lower() in message_lower for kw in keywords):
                        should_include = True

                # Check tool description keywords
                if not should_include:
                    desc_words = tool.description.lower().split()[:10]
                    if any(word in message_lower for word in desc_words if len(word) > 4):
                        should_include = True

                if should_include:
                    matched_semantic.append(tool_name)

        # Limit semantic pool
        for tool_name in matched_semantic[:max_semantic]:
            levels[tool_name] = SchemaLevel.STUB

        logger.debug(
            f"Tool schema levels for vertical={vertical}: "
            f"FULL={len([t for t, l in levels.items() if l == SchemaLevel.FULL])}, "
            f"COMPACT={len([t for t, l in levels.items() if l == SchemaLevel.COMPACT])}, "
            f"STUB={len([t for t, l in levels.items() if l == SchemaLevel.STUB])}"
        )

        return levels

    def get_tools_for_broadcast(
        self,
        user_message: str,
        vertical: Optional[str] = None,
    ) -> List[Tuple["ToolDefinition", SchemaLevel]]:
        """Get tools with their schema levels for LLM broadcasting.

        This is the main entry point for token-efficient tool broadcasting.
        Returns tools paired with their schema levels so adapters can
        generate appropriately sized schemas.

        Args:
            user_message: The user's message
            vertical: Optional vertical name

        Returns:
            List of (ToolDefinition, SchemaLevel) tuples
        """
        from victor.providers.base import ToolDefinition

        levels = self.get_tools_with_levels(user_message, vertical)

        result: List[Tuple[ToolDefinition, SchemaLevel]] = []
        all_tools_map = {tool.name: tool for tool in self.tools.list_tools()}

        for tool_name, level in levels.items():
            if tool_name in all_tools_map:
                tool = all_tools_map[tool_name]
                tool_def = ToolDefinition(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.parameters,
                )
                result.append((tool_def, level))

        # Sort: FULL schema tools first, then STUB
        result.sort(key=lambda x: (0 if x[1] == SchemaLevel.FULL else 1, x[0].name))

        return result

    def _get_core_tools_cached(self) -> Set[str]:
        """Get core tools with caching for performance.

        Uses dynamic discovery via get_critical_tools() on first call,
        then caches the result for subsequent calls.

        Returns:
            Set of canonical core tool names
        """
        if self._cached_core_tools is None:
            self._cached_core_tools = get_critical_tools(self.tools)
        return self._cached_core_tools

    def _get_core_readonly_cached(self) -> Set[str]:
        """Get core read-only tools with caching."""
        if self._cached_core_readonly is None:
            try:
                from victor.tools.metadata_registry import get_core_readonly_tools

                self._cached_core_readonly = set(get_core_readonly_tools())
            except Exception:
                # Fallback to an empty set; caller will layer other tools.
                self._cached_core_readonly = set()
        return self._cached_core_readonly

    def _get_stage_core_tools(self, stage: Optional[ConversationStage]) -> Set[str]:
        """Choose core set based on stage (safe for exploration/analysis)."""
        if stage in {
            ConversationStage.INITIAL,
            ConversationStage.PLANNING,
            ConversationStage.READING,
            ConversationStage.ANALYSIS,
        }:
            return self._get_core_readonly_cached()
        return self._get_core_tools_cached()

    def _is_readonly_tool(self, tool_name: str) -> bool:
        """Check if a tool is readonly via metadata registry."""
        try:
            from victor.tools.metadata_registry import get_global_registry

            entry = get_global_registry().get(tool_name)
            if not entry:
                return False
            return (
                entry.access_mode == AccessMode.READONLY
                or entry.execution_category == ExecutionCategory.READ_ONLY
            )
        except Exception:
            return False

    def _filter_tools_for_stage(
        self, tools: List["ToolDefinition"], stage: Optional[ConversationStage]
    ) -> List["ToolDefinition"]:
        """Remove write/execute tools during exploration/analysis stages.

        Note: Vertical core tools (from TieredToolConfig) are ALWAYS preserved
        since they are essential for the vertical's operation even in early stages.
        For example, DevOps needs 'docker' and 'shell', Research needs 'web_search'.
        """
        # Skip stage filtering in BUILD mode (allow_all_tools=True)
        # Uses ModeAwareMixin for consistent mode controller access
        if self.is_build_mode:
            logger.debug("Stage filtering skipped: BUILD mode allows all tools")
            return tools

        if stage not in {
            ConversationStage.INITIAL,
            ConversationStage.PLANNING,
            ConversationStage.READING,
            ConversationStage.ANALYSIS,
        }:
            return tools

        # Get vertical core tools that should be preserved regardless of stage
        preserved_tools: Set[str] = set()
        tiered_config = getattr(self, "_tiered_config", None)
        if tiered_config:
            # Always preserve mandatory and vertical_core tools
            preserved_tools.update(tiered_config.mandatory)
            preserved_tools.update(tiered_config.vertical_core)
            # If vertical has readonly_only_for_analysis=False, don't filter at all
            if (
                hasattr(tiered_config, "readonly_only_for_analysis")
                and not tiered_config.readonly_only_for_analysis
            ):
                logger.debug(
                    "Stage filtering skipped: vertical has readonly_only_for_analysis=False"
                )
                return tools

        # Filter to readonly tools, but always keep vertical core tools
        filtered = [t for t in tools if self._is_readonly_tool(t.name) or t.name in preserved_tools]

        if filtered:
            if preserved_tools:
                logger.debug(
                    f"Stage filtering preserved vertical tools: {preserved_tools & {t.name for t in filtered}}"
                )
            return filtered

        # Fallback to core readonly if filtering removed everything
        readonly_core = self._get_stage_core_tools(stage)
        fallback: List["ToolDefinition"] = []
        for tool in self.tools.list_tools():
            if tool.name in readonly_core:
                from victor.providers.base import ToolDefinition

                fallback.append(
                    ToolDefinition(
                        name=tool.name, description=tool.description, parameters=tool.parameters
                    )
                )
        return fallback or tools

    def _get_web_tools_cached(self) -> Set[str]:
        """Get web tools with caching for performance.

        Uses dynamic discovery via get_web_tools() on first call,
        then caches the result for subsequent calls.

        Returns:
            Set of canonical web tool names
        """
        if self._cached_web_tools is None:
            self._cached_web_tools = get_web_tools(self.tools)
        return self._cached_web_tools

    def invalidate_tool_cache(self) -> None:
        """Invalidate cached tool sets to force re-discovery.

        Call this method when:
        - Tools are dynamically added/removed at runtime
        - Plugin tools are loaded/unloaded
        - Tool categories are modified

        This enables hot-reload of tool configurations without
        recreating the ToolSelector instance.
        """
        self._cached_core_tools = None
        self._cached_core_readonly = None
        self._cached_web_tools = None
        logger.debug("Tool selection cache invalidated - will re-discover on next access")

    def set_enabled_tools(self, tools: Optional[Set[str]]) -> None:
        """Set which tools are enabled for selection (vertical filter).

        When set, only tools in this set will be considered for selection.
        Pass None to allow all tools.

        Args:
            tools: Set of tool names to enable, or None for all tools
        """
        self._enabled_tools = tools
        if tools:
            logger.info(f"Tool selector enabled tools filter: {sorted(tools)}")
        else:
            logger.debug("Tool selector enabled tools filter cleared")

    def get_enabled_tools(self) -> Optional[Set[str]]:
        """Get the enabled tools filter.

        Returns:
            Set of enabled tool names, or None if all tools are enabled
        """
        return self._enabled_tools

    def set_tiered_config(self, config: Any) -> None:
        """Set tiered tool configuration for intelligent selection.

        When set, tool selection uses the tiered approach:
        1. Mandatory + Vertical Core are always included
        2. Semantic pool is selected based on query similarity
        3. Stage tools are added for specific stages

        Args:
            config: TieredToolConfig instance
        """
        self._tiered_config = config
        if config:
            logger.info(
                f"Tiered tool config set: "
                f"mandatory={sorted(config.mandatory)}, "
                f"vertical_core={sorted(config.vertical_core)}, "
                f"semantic_pool={sorted(config.semantic_pool)}"
            )
        else:
            logger.debug("Tiered tool config cleared")

    def set_vertical_context(self, context: Optional["VerticalContext"]) -> None:
        """Set vertical context for vertical-specific tool selection.

        The vertical context provides the tool_selection_strategy which can
        prioritize, weight, or exclude tools based on vertical domain knowledge.

        Args:
            context: VerticalContext instance, or None to disable
        """
        self._vertical_context = context
        if context and context.has_tool_selection_strategy:
            logger.info(
                f"Vertical tool selection strategy set for vertical: {context.vertical_name}"
            )
        else:
            logger.debug("Vertical context set (no tool selection strategy)")

    def _apply_vertical_strategy(
        self,
        tools: List["ToolDefinition"],
        user_message: str,
        task_type: str = "unknown",
    ) -> List["ToolDefinition"]:
        """Apply vertical-specific tool selection strategy to reorder/filter tools.

        This implements the Strategy Pattern (OCP) allowing verticals to customize
        tool selection without modifying the core selection logic.

        Args:
            tools: List of selected tools
            user_message: The user's message
            task_type: Detected task type

        Returns:
            Reordered/filtered list of tools based on vertical strategy
        """
        if not self._vertical_context or not self._vertical_context.has_tool_selection_strategy:
            return tools

        strategy = self._vertical_context.tool_selection_strategy
        if not strategy:
            return tools

        try:
            # Import here to avoid circular imports
            from victor.core.verticals.protocols import ToolSelectionContext

            # Get conversation stage if available
            stage = "exploration"
            if self.conversation_state:
                stage = self.conversation_state.current_stage.name.lower()

            # Build context for strategy
            context = ToolSelectionContext(
                task_type=task_type,
                user_message=user_message,
                conversation_stage=stage,
                available_tools={t.name for t in tools},
                recent_tools=[],  # Could be populated from history
                metadata={
                    "model": self.model,
                    "provider": self.provider_name,
                },
            )

            # Apply strategy
            result = strategy.select_tools(context)

            # Reorder based on priority_tools
            if result.priority_tools:
                priority_set = set(result.priority_tools)
                priority_ordered = [t for t in tools if t.name in priority_set]
                others = [t for t in tools if t.name not in priority_set]

                # Sort priority tools by their position in priority_tools list
                priority_order = {name: i for i, name in enumerate(result.priority_tools)}
                priority_ordered.sort(key=lambda t: priority_order.get(t.name, 999))

                tools = priority_ordered + others
                logger.debug(f"Vertical strategy prioritized tools: {result.priority_tools}")

            # Exclude tools
            if result.excluded_tools:
                tools = [t for t in tools if t.name not in result.excluded_tools]
                logger.debug(f"Vertical strategy excluded tools: {sorted(result.excluded_tools)}")

            # Log reasoning if provided
            if result.reasoning:
                logger.info(f"Vertical tool selection: {result.reasoning}")

            return tools

        except Exception as e:
            logger.warning(f"Failed to apply vertical tool selection strategy: {e}")
            return tools

    def get_tiered_config(self) -> Optional[Any]:
        """Get the tiered tool configuration.

        Returns:
            TieredToolConfig or None if not set
        """
        return getattr(self, "_tiered_config", None)

    def select_tiered_tools(
        self,
        user_message: str,
        stage: Optional[str] = None,
        is_analysis_task: bool = False,
    ) -> List["ToolDefinition"]:
        """Select tools using tiered configuration.

        This method provides context-efficient tool selection by using:
        1. Always: mandatory + vertical_core tools
        2. If not analysis or readonly_only_for_analysis=False: write/execute tools
        3. Stage-specific tools if stage is provided
        4. Semantic pool tools matched to the query

        Args:
            user_message: User's query for semantic matching
            stage: Optional stage name for stage-specific tools
            is_analysis_task: If True and config.readonly_only_for_analysis,
                              exclude write/execute tools

        Returns:
            List of ToolDefinition objects
        """
        from victor.providers.base import ToolDefinition

        config = self.get_tiered_config()
        if not config:
            # Fall back to regular selection
            return self.select_keywords(user_message)

        all_tools_map = {tool.name: tool for tool in self.tools.list_tools()}
        selected: Dict[str, ToolDefinition] = {}

        # Tier 1: Mandatory tools (always included)
        for name in config.mandatory:
            if name in all_tools_map:
                tool = all_tools_map[name]
                selected[name] = ToolDefinition(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.parameters,
                )

        # Tier 2: Vertical core tools (always included for this vertical)
        for name in config.vertical_core:
            if name in all_tools_map and name not in selected:
                tool = all_tools_map[name]
                # Skip write/execute tools for analysis tasks if configured
                if is_analysis_task and config.readonly_only_for_analysis:
                    if not self._is_readonly_tool(name):
                        continue
                selected[name] = ToolDefinition(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.parameters,
                )

        # Tier 3: Stage-specific tools
        # Use config helper method that prefers registry metadata over static stage_tools
        if stage:
            stage_tools = (
                config.get_tools_for_stage_from_registry(stage)
                if hasattr(config, "get_tools_for_stage_from_registry")
                else config.stage_tools.get(stage, set())
            )
            for name in stage_tools:
                if name in all_tools_map and name not in selected:
                    tool = all_tools_map[name]
                    # Skip write/execute for analysis tasks
                    if is_analysis_task and config.readonly_only_for_analysis:
                        if not self._is_readonly_tool(name):
                            continue
                    selected[name] = ToolDefinition(
                        name=tool.name,
                        description=tool.description,
                        parameters=tool.parameters,
                    )

        # Tier 4: Semantic pool (selected based on query)
        # Use config helper method that prefers registry metadata over static semantic_pool
        semantic_pool = (
            config.get_effective_semantic_pool()
            if hasattr(config, "get_effective_semantic_pool")
            else config.semantic_pool
        )
        if semantic_pool:
            # Simple keyword matching for semantic pool
            message_lower = user_message.lower()
            for name in semantic_pool:
                if name in all_tools_map and name not in selected:
                    tool = all_tools_map[name]

                    # Check if query suggests need for this tool
                    should_include = False

                    # Check tool keywords from metadata
                    if hasattr(tool, "metadata") and tool.metadata:
                        keywords = getattr(tool.metadata, "keywords", []) or []
                        if any(kw.lower() in message_lower for kw in keywords):
                            should_include = True

                    # Check tool description keywords
                    desc_words = tool.description.lower().split()[:10]
                    if any(word in message_lower for word in desc_words if len(word) > 4):
                        should_include = True

                    # Skip write/execute for analysis tasks
                    if should_include and is_analysis_task and config.readonly_only_for_analysis:
                        if not self._is_readonly_tool(name):
                            should_include = False

                    if should_include:
                        selected[name] = ToolDefinition(
                            name=tool.name,
                            description=tool.description,
                            parameters=tool.parameters,
                        )

        result = list(selected.values())
        logger.info(
            f"Tiered selection: {len(result)} tools "
            f"(mandatory={len(config.mandatory)}, "
            f"core={len(config.vertical_core)}, "
            f"stage={stage or 'None'}, "
            f"analysis={is_analysis_task}): "
            f"{', '.join(t.name for t in result)}"
        )
        self._record_selection("tiered", len(result))
        return result

    def _filter_by_enabled(self, tools: List["ToolDefinition"]) -> List["ToolDefinition"]:
        """Filter tools by the enabled tools set.

        Args:
            tools: List of tool definitions to filter

        Returns:
            Filtered list containing only enabled tools
        """
        if not self._enabled_tools:
            return tools

        filtered = [t for t in tools if t.name in self._enabled_tools]
        if len(filtered) < len(tools):
            logger.debug(
                f"Filtered tools by enabled set: {len(filtered)} from {len(tools)} "
                f"(enabled: {sorted(self._enabled_tools)})"
            )
        return filtered

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

        # Ensure web tools if explicitly mentioned (dynamic discovery)
        message_lower = user_message.lower()
        if any(kw in message_lower for kw in WEB_KEYWORDS):
            must_have = self._get_web_tools_cached()
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

        # Stage-aware filtering (keep read-only tools for exploration/analysis)
        stage = self.conversation_state.get_stage() if self.conversation_state else None
        tools = self._filter_tools_for_stage(tools, stage)

        logger.debug(
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

        # Cap to fallback_max_tools to avoid broadcasting too many tools
        if len(tools) > self.fallback_max_tools:
            tools = tools[: self.fallback_max_tools]

        # Apply vertical-specific tool selection strategy (DIP/OCP)
        # This allows verticals to prioritize/reorder tools based on domain knowledge
        tools = self._apply_vertical_strategy(tools, user_message)

        return tools

    def select_keywords(
        self,
        user_message: str,
        planned_tools: Optional[List["ToolDefinition"]] = None,
    ) -> List["ToolDefinition"]:
        """Select tools using keyword-based category matching.

        If enabled_tools filter is set (from vertical), returns all enabled tools.
        Otherwise falls back to core tools + keyword matching.

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

        # If vertical has set enabled tools, use those directly
        # This allows verticals like "research" to specify web_search, web_fetch, etc.
        if self._enabled_tools:
            logger.info(
                f"Using vertical enabled tools ({len(self._enabled_tools)}): "
                f"{sorted(self._enabled_tools)}"
            )
            for tool in all_tools:
                if tool.name in self._enabled_tools and tool.name not in existing_names:
                    selected_tools.append(
                        ToolDefinition(
                            name=tool.name,
                            description=tool.description,
                            parameters=tool.parameters,
                        )
                    )
                    existing_names.add(tool.name)

            # Apply stage filtering and return
            stage = self.conversation_state.get_stage() if self.conversation_state else None
            selected_tools = self._filter_tools_for_stage(selected_tools, stage)

            tool_names = [t.name for t in selected_tools]
            logger.info(
                f"Selected {len(selected_tools)} tools from vertical filter: "
                f"{', '.join(tool_names)}"
            )
            self._record_selection("vertical", len(selected_tools))
            return selected_tools

        # Fallback: Build selected tool names using core tools + registry keyword matches
        # Uses keywords from @tool decorators as single source of truth
        stage = self.conversation_state.get_stage() if self.conversation_state else None
        selected_tool_names = self._get_stage_core_tools(stage).copy()

        # Use registry-based keyword matching (from @tool decorators)
        registry_matches = get_tools_from_message(user_message)
        selected_tool_names.update(registry_matches)

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
            core_tools_set = self._get_stage_core_tools(stage)
            core_tools = [t for t in selected_tools if t.name in core_tools_set]
            other_tools = [t for t in selected_tools if t.name not in core_tools_set]
            selected_tools = core_tools + other_tools[: max(0, 10 - len(core_tools))]

        # Enforce read-only set during exploration/analysis stages
        selected_tools = self._filter_tools_for_stage(selected_tools, stage)

        # Apply vertical-specific tool selection strategy (DIP/OCP)
        # This allows verticals to prioritize/reorder tools based on domain knowledge
        selected_tools = self._apply_vertical_strategy(selected_tools, user_message)

        tool_names = [t.name for t in selected_tools]
        tool_names_set = set(tool_names)

        # Only log at INFO level if selection changed, otherwise DEBUG
        if self._last_selection != tool_names_set:
            self._last_selection = tool_names_set
            logger.info(
                f"Selected {len(selected_tools)} tools (small_model={small_model}): "
                f"{', '.join(tool_names)}"
            )
        else:
            logger.debug(f"Tool selection unchanged: {len(selected_tools)} tools")

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

        # Core tools always included (stage-aware, read-only for explore/analysis)
        core = self._get_stage_core_tools(current_stage)

        # Web tools check (dynamic discovery)
        web_tools = self._get_web_tools_cached() if needs_web_tools(user_message) else set()

        # Combine stage-specific tools with core and web tools
        keep = stage_tools | core | web_tools

        # Always include vertical_core tools from tiered config (GAP-4 fix)
        # These are essential tools for the vertical (e.g., docker for DevOps, web_search for Research)
        tiered_config = getattr(self, "_tiered_config", None)
        if tiered_config:
            keep.update(tiered_config.mandatory)
            keep.update(tiered_config.vertical_core)

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
            logger.debug(
                f"Stage-pruned tools ({current_stage.name}): "
                f"{len(pruned)} tools kept from {len(tools)}"
            )
            return pruned

        # Fallback to core tools (stage-aware) + vertical_core tools
        core_fallback = self._get_stage_core_tools(current_stage)
        # Also include vertical_core tools in fallback (GAP-4 fix)
        if tiered_config:
            core_fallback = core_fallback | tiered_config.mandatory | tiered_config.vertical_core
        fallback_tools = [t for t in tools if t.name in core_fallback]

        if fallback_tools:
            logger.debug(f"Stage pruning fallback: {len(fallback_tools)} core+vertical tools")
            return fallback_tools

        # Last resort: return a small prefix
        logger.warning(f"Stage pruning: last resort fallback to {self.fallback_max_tools} tools")
        return tools[: self.fallback_max_tools]

    def get_task_aware_tools(
        self,
        stage: str = "initial",
    ) -> Set[str]:
        """Get tools appropriate for the current task type and stage.

        Uses TaskMilestoneMonitor to determine task type and TaskToolConfigLoader
        to get stage-specific tool recommendations from YAML configuration.

        Args:
            stage: Current conversation stage (initial, reading, executing, verifying)

        Returns:
            Set of tool names appropriate for the task and stage
        """
        stage_enum: Optional[ConversationStage] = None
        try:
            stage_enum = ConversationStage[stage.upper()]
        except Exception:
            stage_enum = None

        if not self.task_tracker:
            return self._get_stage_core_tools(stage_enum).copy()

        # Lazy load the config loader
        if self._task_config_loader is None:
            from victor.agent.milestone_monitor import TaskToolConfigLoader

            self._task_config_loader = TaskToolConfigLoader()

        # Get task type from tracker
        task_type = self.task_tracker.progress.task_type.value

        # Get stage-specific tools from YAML config
        stage_tools = self._task_config_loader.get_stage_tools(task_type, stage)

        # Always include required tools for the task type
        required = self.task_tracker.get_required_tools()

        # Combine stage tools with required tools
        tools = set(stage_tools) | required

        logger.debug(f"Task-aware tools for {task_type}/{stage}: " f"{sorted(tools)}")

        return tools

    def get_force_action_hint(self) -> Optional[str]:
        """Get hint message if LLM should be forced to take action.

        Returns:
            Hint message if action should be forced, None otherwise
        """
        if not self.task_tracker:
            return None

        should_force, hint = self.task_tracker.should_force_action()
        return hint if should_force else None

    def prioritize_by_task(
        self,
        tools: List["ToolDefinition"],
        stage: str = "initial",
    ) -> List["ToolDefinition"]:
        """Filter and prioritize tools based on task type and stage.

        Args:
            tools: List of tool definitions to filter
            stage: Current conversation stage

        Returns:
            Filtered list of tools appropriate for task and stage
        """
        if not tools:
            return tools

        if not self.task_tracker:
            return tools

        # Get task-aware tools
        task_tools = self.get_task_aware_tools(stage)

        # Always include core tools (dynamic discovery)
        try:
            stage_enum = ConversationStage[stage.upper()]
        except Exception:
            stage_enum = None
        allowed = task_tools | self._get_stage_core_tools(stage_enum)

        # Check if we need to force action tools
        if self.task_tracker.progress.task_type.value == "edit":
            # For EDIT tasks after target read, ensure edit is included
            from victor.agent.milestone_monitor import Milestone

            if Milestone.TARGET_READ in self.task_tracker.progress.milestones:
                allowed.add("edit")  # edit_files → edit (canonical name)
                logger.debug("Added edit for EDIT task after TARGET_READ")

        # Filter tools to only those allowed
        filtered = [t for t in tools if t.name in allowed]

        if filtered:
            logger.info(
                f"Task-aware tool filtering: {len(filtered)} tools from {len(tools)} "
                f"(task={self.task_tracker.progress.task_type.value}, stage={stage})"
            )
            return filtered

        # Fallback to original tools
        logger.warning("Task-aware filtering removed all tools, keeping originals")
        return tools

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

        stage = self.conversation_state.get_stage() if self.conversation_state else None
        # Add core tools first (dynamic discovery)
        for tool_name in self._get_stage_core_tools(stage):
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

        # Cap to fallback_max_tools to avoid broadcasting too many
        if len(tools) > self.fallback_max_tools:
            tools = tools[: self.fallback_max_tools]

        logger.info(
            f"Smart fallback selected {len(tools)} tools: " f"{', '.join(t.name for t in tools)}"
        )

        return tools
