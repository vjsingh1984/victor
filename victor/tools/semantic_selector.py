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

"""Semantic tool selection using embeddings for intelligent, context-aware tool matching."""

import hashlib
import logging
import os
import pickle
import re
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

# Disable tokenizers parallelism BEFORE importing sentence_transformers
# This prevents "bad value(s) in fds_to_keep" errors in async contexts
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import httpx
import numpy as np

from victor.providers.base import ToolDefinition
from victor.tools.base import CostTier, ToolMetadataRegistry, ToolRegistry
from victor.embeddings.service import EmbeddingService

# Import for type checking only (avoid circular imports)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from victor.agent.unified_classifier import ClassificationResult, TaskType

logger = logging.getLogger(__name__)


# Cost tier warning messages for user visibility
COST_TIER_WARNINGS = {
    CostTier.HIGH: "HIGH COST: This tool may use significant resources or make external API calls",
    CostTier.MEDIUM: "MEDIUM COST: This tool performs moderately expensive operations",
}


class SemanticToolSelector:
    """Select relevant tools using embedding-based semantic similarity.

    Instead of keyword matching, this uses embeddings to find tools
    semantically related to the user's request.

    Benefits:
    - Handles synonyms automatically (test → verify, validate, check)
    - Understands context and intent
    - No hardcoded keyword lists
    - Self-improving with better tool descriptions
    """

    # Class-level cache for tool knowledge loaded from YAML
    _tool_knowledge: ClassVar[Optional[Dict[str, Dict[str, Any]]]] = None
    _tool_knowledge_loaded: ClassVar[bool] = False

    @classmethod
    def _load_tool_knowledge(cls) -> Dict[str, Dict[str, Any]]:
        """Load tool knowledge from YAML file (DEPRECATED).

        NOTE: This is a legacy fallback for tools that don't implement
        the ToolMetadataProvider contract (get_metadata()). New tools
        should use inline metadata via @tool decorator or metadata property.

        The tool_knowledge.yaml file has been archived. All tools should
        now provide metadata via get_metadata() which auto-generates from
        tool properties if not explicitly defined.

        Returns:
            Dictionary mapping tool names to their knowledge (use_cases, keywords, examples)
        """
        if cls._tool_knowledge_loaded:
            # Return cached value (don't use `or {}` as empty dict is falsy)
            return cls._tool_knowledge if cls._tool_knowledge is not None else {}

        # tool_knowledge.yaml is deprecated and archived
        # Return empty dict - all metadata now comes from get_metadata()
        cls._tool_knowledge = {}
        cls._tool_knowledge_loaded = True
        logger.debug(
            "tool_knowledge.yaml is deprecated. All tools should use get_metadata() "
            "for metadata discovery (auto-generated or explicit)."
        )
        return cls._tool_knowledge

    @classmethod
    def _build_use_case_text(cls, tool_name: str) -> str:
        """Build use case text from loaded tool knowledge.

        Args:
            tool_name: Name of the tool

        Returns:
            Formatted use case text for embedding
        """
        knowledge = cls._load_tool_knowledge()

        if tool_name not in knowledge:
            return ""

        tool_data = knowledge[tool_name]
        parts = []

        # Add use cases
        use_cases = tool_data.get("use_cases", [])
        if use_cases:
            parts.append(f"Use for: {', '.join(use_cases)}.")

        # Add keywords
        keywords = tool_data.get("keywords", [])
        if keywords:
            parts.append(f"Common requests: {', '.join(keywords)}.")

        # Add examples
        examples = tool_data.get("examples", [])
        if examples:
            parts.append(f"Examples: {', '.join(examples)}.")

        return " ".join(parts)

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_provider: str = "sentence-transformers",
        ollama_base_url: str = "http://localhost:11434",
        cache_embeddings: bool = True,
        cache_dir: Optional[Path] = None,
        cost_aware_selection: bool = True,
        cost_penalty_factor: float = 0.05,
    ):
        """Initialize semantic tool selector.

        Args:
            embedding_model: Model to use for embeddings
                - sentence-transformers: "all-MiniLM-L6-v2" (default, 80MB, ~5ms)
                - ollama: "nomic-embed-text", "qwen3-embedding:8b", etc.
            embedding_provider: Provider (sentence-transformers, ollama, vllm, lmstudio)
                Default: "sentence-transformers" (local, fast, bundled)
            ollama_base_url: Ollama/vLLM/LMStudio API base URL
            cache_embeddings: Cache tool embeddings (recommended)
            cache_dir: Directory to store embedding cache (default: ~/.victor/embeddings/)
            cost_aware_selection: Deprioritize high-cost tools (default: True)
            cost_penalty_factor: Penalty per cost weight (default: 0.05)
        """
        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider
        self.ollama_base_url = ollama_base_url
        self.cache_embeddings = cache_embeddings
        self.cost_aware_selection = cost_aware_selection
        self.cost_penalty_factor = cost_penalty_factor

        # Cache directory
        if cache_dir is None:
            from victor.config.settings import get_project_paths

            cache_dir = get_project_paths().global_embeddings_dir
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache file path (includes model name for version control)
        cache_filename = (
            f"tool_embeddings_{embedding_model.replace(':', '_').replace('/', '_')}.pkl"
        )
        self.cache_file = self.cache_dir / cache_filename

        # In-memory cache: tool_name → embedding vector
        self._tool_embedding_cache: Dict[str, np.ndarray] = {}

        # Tool version hash (to detect when tools change)
        self._tools_hash: Optional[str] = None

        # Note: sentence-transformers model is managed by shared EmbeddingService singleton
        # This reduces memory usage by sharing the model with IntentClassifier

        # HTTP client for Ollama/vLLM/LMStudio
        self._client = None
        if embedding_provider in ["ollama", "vllm", "lmstudio"]:
            self._client = httpx.AsyncClient(base_url=ollama_base_url, timeout=30.0)

        # Phase 3: Tool usage tracking and learning
        self._usage_cache_file = self.cache_dir / "tool_usage_stats.pkl"
        self._tool_usage_cache: Dict[str, Dict[str, Any]] = {}
        self._load_usage_cache()

        # Phase 6: Store last cost warnings for retrieval
        self._last_cost_warnings: List[str] = []
        # Track which tools have already been warned about (warn once per session)
        self._warned_tools: set = set()

    async def initialize_tool_embeddings(self, tools: ToolRegistry) -> None:
        """Pre-compute embeddings for all tools (called once at startup).

        Loads from pickle cache if available and tools haven't changed.
        Otherwise, computes embeddings and saves to cache.

        Also refreshes the ToolMetadataRegistry with metadata from all tools.

        Args:
            tools: Tool registry with all available tools
        """
        # Refresh the centralized ToolMetadataRegistry (smart reindexing)
        # This collects metadata from all tools (explicit or auto-generated)
        # Uses hash-based change detection to skip reindexing if tools haven't changed
        metadata_registry = ToolMetadataRegistry.get_instance()
        tool_list = tools.list_tools()
        reindexed = metadata_registry.refresh_from_tools(tool_list)

        if reindexed:
            stats = metadata_registry.get_statistics()
            logger.info(
                f"ToolMetadataRegistry reindexed: {stats['total_tools']} tools, "
                f"{stats['total_categories']} categories, {stats['total_keywords']} keywords"
            )
        else:
            logger.debug("ToolMetadataRegistry cache valid, skipping reindex")

        # Pre-initialize usage stats for ALL tools (not just used ones)
        # This ensures show_tool_stats.py reports all 47 tools, not just used ones
        self._initialize_all_tool_stats(tool_list)

        if not self.cache_embeddings:
            return

        # Calculate hash of all tool definitions
        tools_hash = self._calculate_tools_hash(tools)

        # Try to load from cache
        if self._load_from_cache(tools_hash):
            logger.info(
                f"Loaded tool embeddings from cache for {len(self._tool_embedding_cache)} tools"
            )
            return

        # Cache miss or tools changed - recompute
        logger.info(
            f"Computing tool embeddings for {len(tools.list_tools())} tools "
            f"(model: {self.embedding_model})"
        )

        for tool in tools.list_tools():
            # Create semantic description of tool
            tool_text = self._create_tool_text(tool)

            # Generate embedding
            embedding = await self._get_embedding(tool_text)

            # Cache it in memory
            self._tool_embedding_cache[tool.name] = embedding

        # Save to disk
        self._save_to_cache(tools_hash)

        logger.info(
            f"Tool embeddings computed and cached for {len(self._tool_embedding_cache)} tools"
        )

    def _calculate_tools_hash(self, tools: ToolRegistry) -> str:
        """Calculate hash of all tool definitions to detect changes.

        Args:
            tools: Tool registry

        Returns:
            SHA256 hash of tool definitions
        """
        # Create deterministic string from all tool definitions
        tool_strings = []
        for tool in sorted(tools.list_tools(), key=lambda t: t.name):
            tool_string = f"{tool.name}:{tool.description}:{tool.parameters}"
            tool_strings.append(tool_string)

        combined = "|".join(tool_strings)
        return hashlib.sha256(combined.encode()).hexdigest()

    def _load_from_cache(self, tools_hash: str) -> bool:
        """Load embeddings from pickle cache if valid.

        Args:
            tools_hash: Current hash of tool definitions

        Returns:
            True if loaded successfully, False otherwise
        """
        if not self.cache_file.exists():
            return False

        try:
            with open(self.cache_file, "rb") as f:
                cache_data = pickle.load(f)

            # Verify cache is for same tools
            if cache_data.get("tools_hash") != tools_hash:
                logger.info("Tool definitions changed, cache invalidated")
                return False

            # Verify cache is for same embedding model
            if cache_data.get("embedding_model") != self.embedding_model:
                logger.info("Embedding model changed, cache invalidated")
                return False

            # Load embeddings
            self._tool_embedding_cache = cache_data["embeddings"]
            self._tools_hash = tools_hash

            return True

        except Exception as e:
            logger.warning(f"Failed to load embedding cache: {e}")
            return False

    def _save_to_cache(self, tools_hash: str) -> None:
        """Save embeddings to pickle cache.

        Args:
            tools_hash: Hash of tool definitions
        """
        try:
            cache_data = {
                "embedding_model": self.embedding_model,
                "tools_hash": tools_hash,
                "embeddings": self._tool_embedding_cache,
            }

            with open(self.cache_file, "wb") as f:
                pickle.dump(cache_data, f)

            cache_size = self.cache_file.stat().st_size / 1024  # KB
            logger.info(f"Saved embedding cache to {self.cache_file} ({cache_size:.1f} KB)")

        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")

    # Category alias mappings: maps semantic category names to registry categories
    # This enables using logical names (file_ops, git_ops) that map to
    # auto-generated metadata categories (filesystem, git, etc.)
    CATEGORY_ALIASES = {
        "file_ops": ["filesystem", "code"],
        "git_ops": ["git", "merge"],
        "analysis": ["code", "pipeline", "security", "audit", "code_quality"],
        "refactoring": ["refactoring", "code"],
        "generation": ["generation", "search", "code"],
        "execution": ["code", "testing"],
        "code_intel": ["search", "code", "lsp"],
        "web": ["web"],
        "workflows": ["pipeline", "code"],
        "pipeline": ["pipeline"],
        "merge": ["merge", "git"],
        "security": ["security"],
        "audit": ["audit"],
    }

    # Fallback tools for each logical category (used when registry has no matches)
    # These are deprecated and will be removed once all tools have proper metadata
    FALLBACK_CATEGORY_TOOLS = {
        "file_ops": ["read_file", "write_file", "edit_files", "list_directory"],
        "git_ops": ["execute_bash", "git_suggest_commit", "git_create_pr"],
        "analysis": ["analyze_docs", "analyze_metrics"],
        "refactoring": ["refactor_extract_function", "refactor_inline_variable", "rename_symbol"],
        "generation": ["generate_docs", "semantic_code_search", "code_search"],
        "execution": ["execute_bash", "execute_python_in_sandbox", "run_tests"],
        "code_intel": ["find_symbol", "find_references", "semantic_code_search", "code_search"],
        "web": ["web_search", "web_fetch", "web_summarize"],
        "workflows": ["run_workflow", "batch", "cicd"],
    }

    def get_tools_for_logical_category(self, logical_category: str) -> List[str]:
        """Get tools for a logical category using the registry.

        Maps logical category names (file_ops, git_ops) to registry categories
        and returns the combined list of tools. Falls back to hardcoded lists
        if registry has no matches.

        Args:
            logical_category: Logical category name (file_ops, git_ops, etc.)

        Returns:
            List of tool names for the category
        """
        # Get registry categories for this logical category
        registry_categories = self.CATEGORY_ALIASES.get(logical_category, [])

        # Collect tools from all registry categories
        tools = set()
        registry = ToolMetadataRegistry.get_instance()
        for category in registry_categories:
            tools.update(registry.get_tools_by_category(category))

        # If registry has tools, use them
        if tools:
            return list(tools)

        # Fallback to hardcoded list (deprecated)
        return self.FALLBACK_CATEGORY_TOOLS.get(logical_category, [])

    # Mandatory tools for specific keywords (Phase 1)
    MANDATORY_TOOL_KEYWORDS = {
        "diff": ["execute_bash"],
        "show changes": ["execute_bash"],
        "git diff": ["execute_bash"],
        "show diff": ["execute_bash"],
        "compare": ["execute_bash"],
        "commit": ["git_suggest_commit", "execute_bash"],
        "pull request": ["git_create_pr"],
        "pr": ["git_create_pr"],
        "test": ["execute_bash", "run_tests"],
        "run": ["execute_bash"],
        "execute": ["execute_bash"],
        "install": ["execute_bash"],
        # Prefer semantic search, but keep keyword search available as a fallback
        "search": ["web_search", "semantic_code_search", "code_search"],
        "find": ["find_symbol", "find_references"],
        "refactor": ["refactor_extract_function", "refactor_inline_variable", "rename_symbol"],
        "security": ["security_scan"],
        "scan": ["security_scan"],
        "review": ["code_review"],
        "document": ["generate_docs"],
        "docs": ["generate_docs", "analyze_docs"],
        # File explanation requires reading the file first - prevents hallucination
        "explain": ["read_file"],
        "describe": ["read_file"],
        "what does": ["read_file"],
        # Count operations are more efficient with bash
        "count": ["execute_bash", "list_directory"],
        "how many": ["execute_bash", "list_directory"],
    }

    # Conceptual query patterns that strongly prefer semantic_code_search over code_search
    # When these patterns are detected, code_search is excluded from mandatory tools
    CONCEPTUAL_QUERY_PATTERNS = [
        "inherit",  # "classes that inherit from"
        "implement",  # "classes implementing interface"
        "extend",  # "classes that extend"
        "subclass",  # "subclasses of"
        "pattern",  # "find patterns", "error handling patterns"
        "similar",  # "similar code", "similar to"
        "related",  # "related functionality"
        "usage",  # "find usages", "usage patterns"
        "example",  # "examples of", "example code"
        "all classes",  # "find all classes"
        "all functions",  # "all functions that"
        "all methods",  # "all methods"
        "how is",  # "how is X done"
        "where is",  # "where is X implemented"
        "what classes",  # "what classes"
        "which files",  # "which files handle"
        "error handling",  # conceptual search
        "exception",  # conceptual search
        "try catch",  # conceptual search
        "logging",  # conceptual search
        "caching",  # conceptual search
        "authentication",  # conceptual search
        "validation",  # conceptual search
        "similar to",  # conceptual search
        "related to",  # conceptual search
    ]

    # Tools for conceptual queries - forces semantic_code_search as primary
    # Excludes list_directory to prevent LLM from exploring instead of searching
    CONCEPTUAL_FALLBACK_TOOLS: List[str] = [
        "semantic_code_search",  # MUST be first - primary tool for conceptual queries
        "read_file",  # To examine results after search
    ]

    # Common fallback tools - used when semantic selection returns too few results
    # These are the most universally useful tools
    COMMON_FALLBACK_TOOLS: List[str] = [
        "read_file",
        "code_search",
        "semantic_code_search",
        "list_directory",
        "execute_bash",
        "write_file",
        "edit_file",
    ]

    def _get_fallback_tools(
        self, tools: "ToolRegistry", max_tools: int = 5, query: str = ""
    ) -> List[str]:
        """Get fallback tools when semantic selection returns too few results.

        Instead of broadcasting ALL tools (which wastes tokens), return a
        curated list of common, universally useful tools.

        For conceptual queries (inheritance, patterns, etc.), uses a restricted
        tool set that forces semantic_code_search as the primary tool.

        Args:
            tools: Tool registry
            max_tools: Maximum fallback tools to return
            query: User query (used to detect conceptual queries)

        Returns:
            List of fallback tool names
        """
        # For conceptual queries, use restricted tool set that forces semantic search
        if query and self._is_conceptual_query(query):
            fallback = []
            for tool_name in self.CONCEPTUAL_FALLBACK_TOOLS:
                if tools.is_tool_enabled(tool_name) and tools.get(tool_name):
                    fallback.append(tool_name)
                if len(fallback) >= max_tools:
                    break
            logger.debug(f"Using CONCEPTUAL fallback tools ({len(fallback)}): {fallback}")
            return fallback

        # Standard fallback for non-conceptual queries
        fallback = []
        for tool_name in self.COMMON_FALLBACK_TOOLS:
            if tools.is_tool_enabled(tool_name) and tools.get(tool_name):
                fallback.append(tool_name)
            if len(fallback) >= max_tools:
                break

        logger.debug(f"Using fallback tools ({len(fallback)}): {fallback}")
        return fallback

    def _is_conceptual_query(self, query: str) -> bool:
        """Check if query is conceptual and should prefer semantic_code_search.

        Args:
            query: User query

        Returns:
            True if query matches conceptual patterns
        """
        query_lower = query.lower()
        for pattern in self.CONCEPTUAL_QUERY_PATTERNS:
            if pattern in query_lower:
                logger.debug(f"Conceptual query detected via pattern: '{pattern}'")
                return True
        return False

    def _get_mandatory_tools(self, query: str) -> List[str]:
        """Get tools that MUST be included based on keywords.

        Args:
            query: User query

        Returns:
            List of tool names that are mandatory for this query
        """
        mandatory = []
        query_lower = query.lower()

        # Check if this is a conceptual query that should strongly prefer semantic search
        is_conceptual = self._is_conceptual_query(query)

        for keyword, tools in self.MANDATORY_TOOL_KEYWORDS.items():
            if self._keyword_in_text(query_lower, keyword):
                # For conceptual queries, exclude code_search from search-related tools
                # to force the LLM to use semantic_code_search instead
                if is_conceptual and keyword in ["search", "find"]:
                    filtered_tools = [t for t in tools if t != "code_search"]
                    mandatory.extend(filtered_tools)
                    logger.debug(
                        f"Mandatory tools for '{keyword}' (conceptual, excluding code_search): {filtered_tools}"
                    )
                else:
                    mandatory.extend(tools)
                    logger.debug(f"Mandatory tools for '{keyword}': {tools}")

        return list(set(mandatory))

    def _get_relevant_categories(self, query: str) -> List[str]:
        """Determine which tool categories are relevant for this query.

        Uses ToolMetadataRegistry dynamically for category lookups, falling
        back to hardcoded lists only when registry is empty.

        Args:
            query: User query

        Returns:
            List of relevant tool names from categories
        """
        query_lower = query.lower()
        relevant_tools = []

        # Multi-step tasks need file_ops and git_ops
        if any(sep in query for sep in [";", "then", "after", "next", "and then"]):
            relevant_tools.extend(self.get_tools_for_logical_category("file_ops"))
            relevant_tools.extend(self.get_tools_for_logical_category("git_ops"))
            logger.debug("Multi-step task detected, including file_ops and git_ops")

        # Analysis keywords
        if any(kw in query_lower for kw in ["analyze", "review", "check", "scan", "audit"]):
            relevant_tools.extend(self.get_tools_for_logical_category("analysis"))
            logger.debug("Analysis task detected")

        # Editing keywords
        if any(kw in query_lower for kw in ["edit", "modify", "change", "update", "fix"]):
            relevant_tools.extend(self.get_tools_for_logical_category("file_ops"))
            relevant_tools.extend(self.get_tools_for_logical_category("refactoring"))
            logger.debug("Editing task detected")

        # Git/diff keywords
        if any(
            self._keyword_in_text(query_lower, kw)
            for kw in ["diff", "commit", "pr", "git", "pull request"]
        ):
            relevant_tools.extend(self.get_tools_for_logical_category("git_ops"))
            logger.debug("Git operation detected")

        # Code navigation
        if any(kw in query_lower for kw in ["find", "locate", "search", "where"]):
            code_intel_tools = self.get_tools_for_logical_category("code_intel")
            # For conceptual queries, exclude code_search to prefer semantic_code_search
            if self._is_conceptual_query(query):
                code_intel_tools = [t for t in code_intel_tools if t != "code_search"]
                logger.debug("Code navigation detected (conceptual - excluding code_search)")
            else:
                logger.debug("Code navigation detected")
            relevant_tools.extend(code_intel_tools)

        # Generation/creation
        if any(kw in query_lower for kw in ["create", "generate", "make", "write new"]):
            relevant_tools.extend(self.get_tools_for_logical_category("file_ops"))
            relevant_tools.extend(self.get_tools_for_logical_category("generation"))
            logger.debug("Generation task detected")

        # Default: file_ops + execution
        if not relevant_tools:
            relevant_tools.extend(self.get_tools_for_logical_category("file_ops"))
            relevant_tools.extend(self.get_tools_for_logical_category("execution"))
            logger.debug("Using default categories: file_ops + execution")

        return list(set(relevant_tools))

    def _extract_pending_actions(self, conversation_history: List[Dict[str, Any]]) -> List[str]:
        """Extract actions mentioned in original request but not yet completed.

        Args:
            conversation_history: List of conversation messages

        Returns:
            List of pending action types
        """
        if not conversation_history:
            return []

        # Get the original user request (first user message)
        original_request = None
        for msg in conversation_history:
            if msg.get("role") == "user":
                original_request = msg.get("content", "")
                break

        if not original_request:
            return []

        pending = []

        # Action keywords to check
        action_patterns = {
            "edit": ["edit", "modify", "change", "update"],
            "show_diff": ["diff", "show changes", "show diff", "compare"],
            "read": ["read", "examine", "look at", "check"],
            "propose": ["propose", "suggest", "recommend"],
            "create": ["create", "generate", "make"],
            "test": ["test", "verify", "validate"],
            "commit": ["commit"],
            "pr": ["pull request", "pr"],
        }

        # Check which actions were requested
        original_lower = original_request.lower()
        for action_type, keywords in action_patterns.items():
            if any(self._keyword_in_text(original_lower, kw) for kw in keywords):
                # Check if this action was completed by looking at tool results
                completed = self._was_action_completed(action_type, conversation_history)
                if not completed:
                    pending.append(action_type)
                    logger.debug(f"Pending action detected: {action_type}")

        return pending

    def _was_action_completed(self, action: str, history: List[Dict[str, Any]]) -> bool:
        """Check if an action was completed based on conversation history.

        Args:
            action: Action type to check
            history: Conversation history

        Returns:
            True if action was completed, False otherwise
        """
        # Look for tool results in assistant messages
        for msg in history:
            if msg.get("role") == "assistant":
                content = str(msg.get("content", "")).lower()

                # Check based on action type
                if action == "show_diff":
                    if "diff" in content or "git diff" in content:
                        return True
                elif action == "edit":
                    if "modified" in content or "edited" in content or "updated" in content:
                        # Check if the specific file mentioned was edited
                        return True
                elif action == "read":
                    if "read" in content or "file contents" in content:
                        return True
                elif action == "create":
                    if "created" in content or "written" in content:
                        return True
                elif action == "test":
                    if "test" in content and ("passed" in content or "failed" in content):
                        return True
                elif action == "commit":
                    if "committed" in content or "commit" in content:
                        return True

        return False

    def _keyword_in_text(self, text: str, keyword: str) -> bool:
        """Return True if keyword is present in text with sane boundaries."""
        if len(keyword) <= 3:
            return re.search(rf"\b{re.escape(keyword)}\b", text) is not None
        return keyword in text

    def _build_contextual_query(
        self,
        current_query: str,
        conversation_history: List[Dict[str, Any]],
        pending_actions: List[str],
    ) -> str:
        """Build enhanced query with conversation context.

        Args:
            current_query: Current user message
            conversation_history: Full conversation history
            pending_actions: List of pending action types

        Returns:
            Enhanced query with context
        """
        # Get last 2 user messages for context
        recent_context = []
        for msg in reversed(conversation_history[-6:]):  # Last 6 messages (3 exchanges)
            if msg.get("role") == "user":
                recent_context.append(msg.get("content", ""))
                if len(recent_context) >= 2:
                    break

        # Build enhanced query
        context_parts = []

        if recent_context and len(recent_context) > 1:
            context_parts.append(f"Context: {recent_context[-1]}")  # Previous request

        if pending_actions:
            context_parts.append(f"Incomplete: {', '.join(pending_actions)}")

        context_parts.append(f"Now: {current_query}")

        enhanced = " | ".join(context_parts)
        logger.debug(f"Contextual query: {enhanced}")
        return enhanced

    # ========================================================================
    # Phase 3: Tool Usage Tracking and Learning
    # ========================================================================

    def _load_usage_cache(self) -> None:
        """Load tool usage statistics from disk cache (Phase 3)."""
        if not self._usage_cache_file.exists():
            logger.debug("No usage cache found - starting fresh")
            return

        try:
            with open(self._usage_cache_file, "rb") as f:
                self._tool_usage_cache = pickle.load(f)
            logger.info(f"Loaded usage stats for {len(self._tool_usage_cache)} tools")
        except Exception as e:
            logger.warning(f"Failed to load usage cache: {e}")
            self._tool_usage_cache = {}

    def _save_usage_cache(self) -> None:
        """Save tool usage statistics to disk cache (Phase 3)."""
        try:
            with open(self._usage_cache_file, "wb") as f:
                pickle.dump(self._tool_usage_cache, f)
            logger.debug(f"Saved usage stats for {len(self._tool_usage_cache)} tools")
        except Exception as e:
            logger.warning(f"Failed to save usage cache: {e}")

    def _initialize_all_tool_stats(self, tools: List[Any]) -> None:
        """Pre-initialize usage stats for ALL tools, not just used ones.

        This ensures that show_tool_stats.py and other analytics tools can
        report on all available tools, even those that haven't been used yet.
        Tools are initialized with 0 usage count but preserve existing stats
        for tools that have been used.

        Args:
            tools: List of BaseTool instances from ToolRegistry
        """
        initialized_count = 0
        for tool in tools:
            if tool.name not in self._tool_usage_cache:
                self._tool_usage_cache[tool.name] = {
                    "usage_count": 0,
                    "success_count": 0,
                    "last_used": 0,
                    "recent_contexts": [],
                }
                initialized_count += 1

        if initialized_count > 0:
            logger.info(
                f"Pre-initialized usage stats for {initialized_count} new tools "
                f"(total: {len(self._tool_usage_cache)} tools)"
            )
            self._save_usage_cache()

    def _record_tool_usage(self, tool_name: str, query: str, success: bool = True) -> None:
        """Record tool usage for learning (Phase 3).

        Args:
            tool_name: Name of the tool that was used
            query: The query context where it was used
            success: Whether the tool was successfully used
        """
        import time

        if tool_name not in self._tool_usage_cache:
            self._tool_usage_cache[tool_name] = {
                "usage_count": 0,
                "success_count": 0,
                "last_used": 0,
                "recent_contexts": [],
            }

        stats = self._tool_usage_cache[tool_name]
        stats["usage_count"] += 1
        if success:
            stats["success_count"] += 1
        stats["last_used"] = time.time()

        # Keep last 50 query contexts for semantic matching (better pattern recognition)
        query_summary = query[:100]  # Truncate long queries
        stats["recent_contexts"].append(query_summary)
        if len(stats["recent_contexts"]) > 50:
            stats["recent_contexts"] = stats["recent_contexts"][-50:]

        # Save periodically (every 5 uses)
        if sum(s["usage_count"] for s in self._tool_usage_cache.values()) % 5 == 0:
            self._save_usage_cache()

    async def _get_usage_boost(self, tool_name: str, query: str) -> float:
        """Calculate similarity boost based on usage history (Phase 3).

        Args:
            tool_name: Name of the tool
            query: Current query

        Returns:
            Boost value (0.0 to 0.2) to add to similarity score
        """
        if tool_name not in self._tool_usage_cache:
            return 0.0

        stats = self._tool_usage_cache[tool_name]

        # Base boost from usage frequency
        usage_boost = min(0.05, stats["usage_count"] * 0.01)  # Max 0.05

        # Boost from success rate
        success_rate = (
            stats["success_count"] / stats["usage_count"] if stats["usage_count"] > 0 else 0
        )
        success_boost = success_rate * 0.05  # Max 0.05

        # Boost from recency (tools used recently get slight preference)
        import time

        time_since_use = time.time() - stats["last_used"]
        days_since_use = time_since_use / 86400  # Convert to days
        recency_boost = max(0, 0.05 - (days_since_use * 0.01))  # Max 0.05

        # Context similarity boost (compare with recent contexts)
        context_boost = 0.0
        if stats["recent_contexts"]:
            try:
                query_emb = await self._get_embedding(query)
                context_similarities = []

                for ctx in stats["recent_contexts"][-3:]:  # Check last 3 contexts
                    ctx_emb = await self._get_embedding(ctx)
                    sim = self._cosine_similarity(query_emb, ctx_emb)
                    context_similarities.append(sim)

                if context_similarities:
                    avg_context_sim = sum(context_similarities) / len(context_similarities)
                    context_boost = avg_context_sim * 0.05  # Max 0.05

            except Exception as e:
                logger.debug(f"Context boost calculation failed: {e}")

        total_boost = usage_boost + success_boost + recency_boost + context_boost
        logger.debug(
            f"Usage boost for {tool_name}: {total_boost:.3f} "
            f"(usage={usage_boost:.3f}, success={success_boost:.3f}, "
            f"recency={recency_boost:.3f}, context={context_boost:.3f})"
        )

        return min(0.2, total_boost)  # Cap total boost at 0.2

    def _get_cost_penalty(self, tool: Any, tools: ToolRegistry) -> float:
        """Calculate cost penalty for a tool based on its cost tier.

        Higher-cost tools receive a penalty to deprioritize them when
        lower-cost alternatives with similar relevance exist.

        Args:
            tool: Tool object
            tools: Tool registry to lookup cost tiers

        Returns:
            Penalty value (0.0 to 0.15) to subtract from similarity score
        """
        if not self.cost_aware_selection:
            return 0.0

        cost_tier = tools.get_tool_cost(tool.name)
        if cost_tier is None:
            return 0.0

        # Calculate penalty: weight * factor
        # FREE (0) = 0.0 penalty
        # LOW (1) = 0.05 penalty
        # MEDIUM (2) = 0.10 penalty
        # HIGH (3) = 0.15 penalty
        penalty = cost_tier.weight * self.cost_penalty_factor

        if penalty > 0:
            logger.debug(f"Cost penalty for {tool.name}: -{penalty:.3f} (tier={cost_tier.value})")

        return penalty

    def _generate_cost_warnings(
        self, selected_tools: List[Tuple[Any, float]], tools: ToolRegistry
    ) -> List[str]:
        """Generate user-facing warnings for high-cost tools in selection.

        Args:
            selected_tools: List of (tool, score) tuples
            tools: Tool registry to lookup cost tiers

        Returns:
            List of warning messages for display to user
        """
        if not self.cost_aware_selection:
            return []

        warnings = []
        for tool, _ in selected_tools:
            # Skip if we've already warned about this tool (warn once per session)
            if tool.name in self._warned_tools:
                continue

            cost_tier = tools.get_tool_cost(tool.name)
            if cost_tier and cost_tier in COST_TIER_WARNINGS:
                warning_msg = f"[{tool.name}] {COST_TIER_WARNINGS[cost_tier]}"
                warnings.append(warning_msg)
                self._warned_tools.add(tool.name)
                logger.debug(f"Cost warning: {warning_msg}")

        return warnings

    def get_last_cost_warnings(self) -> List[str]:
        """Get cost warnings from the last tool selection.

        Returns:
            List of warning messages about high-cost tools selected.
            Empty list if no high-cost tools were selected.
        """
        return self._last_cost_warnings.copy()

    def clear_cost_warnings(self) -> None:
        """Clear stored cost warnings and reset warned tools tracking."""
        self._last_cost_warnings = []
        self._warned_tools.clear()

    async def select_relevant_tools_with_context(
        self,
        user_message: str,
        tools: ToolRegistry,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        max_tools: int = 5,
        similarity_threshold: float = 0.15,
    ) -> List[ToolDefinition]:
        """Select tools with full conversation context awareness (Phase 2).

        This method enhances tool selection by:
        - Tracking pending actions from the original request
        - Including conversation context in semantic search
        - Ensuring mandatory tools for pending actions

        Args:
            user_message: Current user message
            tools: Tool registry
            conversation_history: Full conversation history (Phase 2)
            max_tools: Maximum tools to return
            similarity_threshold: Minimum similarity score

        Returns:
            List of relevant tools with context-aware selection
        """
        # Phase 2: Extract pending actions
        pending_actions = []
        if conversation_history:
            pending_actions = self._extract_pending_actions(conversation_history)
            if pending_actions:
                logger.debug(f"Pending actions: {pending_actions}")

        # Phase 2: Build contextual query
        enhanced_query = user_message
        if conversation_history and pending_actions:
            enhanced_query = self._build_contextual_query(
                user_message, conversation_history, pending_actions
            )

        # Phase 1: Get mandatory tools (including those for pending actions)
        mandatory_tool_names = self._get_mandatory_tools(enhanced_query)

        # Add mandatory tools for pending actions
        pending_action_tools = {
            "show_diff": ["execute_bash"],
            "edit": ["edit_files", "read_file"],
            "commit": ["git_suggest_commit", "execute_bash"],
            "pr": ["git_create_pr"],
            "test": ["execute_bash", "run_tests"],
        }

        for action in pending_actions:
            if action in pending_action_tools:
                mandatory_tool_names.extend(pending_action_tools[action])
                logger.debug(
                    f"Added mandatory tools for pending '{action}': {pending_action_tools[action]}"
                )

        mandatory_tool_names = list(set(mandatory_tool_names))
        if mandatory_tool_names:
            logger.debug(f"Total mandatory tools: {mandatory_tool_names}")

        # Phase 1: Get relevant categories
        category_tools = self._get_relevant_categories(enhanced_query)
        logger.debug(f"Category tools ({len(category_tools)}): {category_tools[:5]}...")

        # Get embedding for enhanced query
        query_embedding = await self._get_embedding(enhanced_query)

        # Calculate similarity scores for tools in relevant categories
        similarities: List[Tuple[Any, float]] = []

        for tool in tools.list_tools():
            # Skip if not in relevant categories and not mandatory
            if tool.name not in category_tools and tool.name not in mandatory_tool_names:
                continue

            # Get cached embedding or compute on-demand
            if tool.name in self._tool_embedding_cache:
                tool_embedding = self._tool_embedding_cache[tool.name]
            else:
                tool_text = self._create_tool_text(tool)
                tool_embedding = await self._get_embedding(tool_text)

            # Cosine similarity
            similarity = self._cosine_similarity(query_embedding, tool_embedding)

            # Boost mandatory tools
            if tool.name in mandatory_tool_names:
                similarity = max(similarity, 0.9)  # Ensure mandatory tools rank high

            # Phase 3: Apply usage boost based on learning
            usage_boost = await self._get_usage_boost(tool.name, enhanced_query)
            similarity += usage_boost

            # Phase 5: Apply cost penalty for high-cost tools
            cost_penalty = self._get_cost_penalty(tool, tools)
            similarity -= cost_penalty

            if similarity >= similarity_threshold:
                similarities.append((tool, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Ensure all mandatory tools are included
        mandatory_tools = [tool for tool in tools.list_tools() if tool.name in mandatory_tool_names]

        # Combine mandatory + top semantic matches
        selected_tools = []
        selected_names = set()

        # First, add all mandatory tools
        for tool in mandatory_tools:
            if tool.name not in selected_names:
                selected_tools.append((tool, 0.9))
                selected_names.add(tool.name)

        # Then add top semantic matches
        for tool, score in similarities:
            if tool.name not in selected_names and len(selected_tools) < max_tools:
                selected_tools.append((tool, score))
                selected_names.add(tool.name)

        # Phase 8: Smart fallback - if too few tools selected, add common fallback tools
        MIN_TOOLS_THRESHOLD = 2
        if len(selected_tools) < MIN_TOOLS_THRESHOLD:
            fallback_names = self._get_fallback_tools(
                tools, max_tools - len(selected_tools), query=user_message
            )
            for fallback_name in fallback_names:
                if fallback_name not in selected_names:
                    fallback_tool = tools.get(fallback_name)
                    if fallback_tool:
                        selected_tools.append((fallback_tool, 0.5))
                        selected_names.add(fallback_name)
            logger.debug(
                f"Added {len(fallback_names)} fallback tools (selection returned < {MIN_TOOLS_THRESHOLD})"
            )

        # Log selection
        tool_names = [t.name for t, _ in selected_tools]
        scores = [f"{s:.3f}" for _, s in selected_tools]
        logger.debug(
            f"Context-aware selection: {len(selected_tools)} tools (mandatory={len(mandatory_tools)}, "
            f"pending_actions={len(pending_actions)}): "
            f"{', '.join(f'{name}({score})' for name, score in zip(tool_names, scores, strict=False))}"
        )

        # Phase 3: Record tool usage for learning
        for tool_name in tool_names:
            self._record_tool_usage(tool_name, user_message, success=True)

        # Phase 6: Generate and store cost warnings for high-cost tools
        self._last_cost_warnings = self._generate_cost_warnings(selected_tools, tools)
        if self._last_cost_warnings:
            logger.debug(f"Cost info: {len(self._last_cost_warnings)} high-cost tools selected")

        # Convert to ToolDefinition
        return [
            ToolDefinition(name=tool.name, description=tool.description, parameters=tool.parameters)
            for tool, _ in selected_tools
        ]

    async def select_relevant_tools(
        self,
        user_message: str,
        tools: ToolRegistry,
        max_tools: int = 5,
        similarity_threshold: float = 0.15,
    ) -> List[ToolDefinition]:
        """Select relevant tools using semantic similarity with category filtering.

        Enhanced with Phase 1 features:
        - Mandatory tool selection for specific keywords
        - Category-based filtering for better relevance

        Args:
            user_message: User's input message
            tools: Tool registry
            max_tools: Maximum number of tools to return
            similarity_threshold: Minimum similarity score (0-1)

        Returns:
            List of relevant ToolDefinition objects, sorted by relevance
        """
        # Phase 1: Get mandatory tools (always included)
        mandatory_tool_names = self._get_mandatory_tools(user_message)
        if mandatory_tool_names:
            logger.debug(f"Mandatory tools: {mandatory_tool_names}")

        # Phase 1: Get relevant categories
        category_tools = self._get_relevant_categories(user_message)
        logger.debug(f"Category tools ({len(category_tools)}): {category_tools[:5]}...")

        # Get embedding for user message
        query_embedding = await self._get_embedding(user_message)

        # Calculate similarity scores for tools in relevant categories
        similarities: List[Tuple[Any, float]] = []

        for tool in tools.list_tools():
            # Skip if not in relevant categories and not mandatory
            if tool.name not in category_tools and tool.name not in mandatory_tool_names:
                continue

            # Get cached embedding or compute on-demand
            if tool.name in self._tool_embedding_cache:
                tool_embedding = self._tool_embedding_cache[tool.name]
            else:
                tool_text = self._create_tool_text(tool)
                tool_embedding = await self._get_embedding(tool_text)

            # Cosine similarity
            similarity = self._cosine_similarity(query_embedding, tool_embedding)

            # Boost mandatory tools
            if tool.name in mandatory_tool_names:
                similarity = max(similarity, 0.9)  # Ensure mandatory tools rank high

            # Phase 5: Apply cost penalty for high-cost tools
            cost_penalty = self._get_cost_penalty(tool, tools)
            similarity -= cost_penalty

            if similarity >= similarity_threshold:
                similarities.append((tool, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Ensure all mandatory tools are included
        mandatory_tools = [tool for tool in tools.list_tools() if tool.name in mandatory_tool_names]

        # Combine mandatory + top semantic matches
        selected_tools = []
        selected_names = set()

        # First, add all mandatory tools
        for tool in mandatory_tools:
            if tool.name not in selected_names:
                selected_tools.append((tool, 0.9))
                selected_names.add(tool.name)

        # Then add top semantic matches
        for tool, score in similarities:
            if tool.name not in selected_names and len(selected_tools) < max_tools:
                selected_tools.append((tool, score))
                selected_names.add(tool.name)

        # Phase 8: Smart fallback - if too few tools selected, add common fallback tools
        # This prevents broadcasting ALL tools (which wastes tokens)
        MIN_TOOLS_THRESHOLD = 2
        if len(selected_tools) < MIN_TOOLS_THRESHOLD:
            fallback_names = self._get_fallback_tools(
                tools, max_tools - len(selected_tools), query=user_message
            )
            for fallback_name in fallback_names:
                if fallback_name not in selected_names:
                    fallback_tool = tools.get(fallback_name)
                    if fallback_tool:
                        selected_tools.append((fallback_tool, 0.5))  # Default score for fallback
                        selected_names.add(fallback_name)
            logger.debug(
                f"Added {len(fallback_names)} fallback tools (semantic selection returned < {MIN_TOOLS_THRESHOLD})"
            )

        # Log selection
        tool_names = [t.name for t, _ in selected_tools]
        scores = [f"{s:.3f}" for _, s in selected_tools]
        logger.debug(
            f"Selected {len(selected_tools)} tools (mandatory={len(mandatory_tools)}): "
            f"{', '.join(f'{name}({score})' for name, score in zip(tool_names, scores, strict=False))}"
        )

        # Phase 6: Generate and store cost warnings for high-cost tools
        self._last_cost_warnings = self._generate_cost_warnings(selected_tools, tools)
        if self._last_cost_warnings:
            logger.debug(f"Cost info: {len(self._last_cost_warnings)} high-cost tools selected")

        # Convert to ToolDefinition
        return [
            ToolDefinition(name=tool.name, description=tool.description, parameters=tool.parameters)
            for tool, _ in selected_tools
        ]

    async def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        if self.embedding_provider == "sentence-transformers":
            return await self._get_sentence_transformer_embedding(text)
        elif self.embedding_provider in ["ollama", "vllm", "lmstudio"]:
            return await self._get_api_embedding(text)
        else:
            raise NotImplementedError(f"Provider {self.embedding_provider} not yet supported")

    async def _get_sentence_transformer_embedding(self, text: str) -> np.ndarray:
        """Get embedding from sentence-transformers using shared EmbeddingService.

        Uses the shared EmbeddingService singleton for memory efficiency
        (single model instance shared with IntentClassifier and other components).

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        try:
            # Use shared embedding service (singleton)
            embedding_service = EmbeddingService.get_instance(model_name=self.embedding_model)
            return await embedding_service.embed_text(text)

        except Exception as e:
            logger.warning(f"Failed to get embedding from EmbeddingService: {e}")
            # Fall back to random embedding (better than crashing)
            logger.warning("Falling back to random embedding")
            return np.random.randn(384).astype(np.float32)

    async def _get_api_embedding(self, text: str) -> np.ndarray:
        """Get embedding from Ollama/vLLM/LMStudio API.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        try:
            response = await self._client.post(
                "/api/embeddings",
                json={"model": self.embedding_model, "prompt": text},
            )
            response.raise_for_status()
            data = response.json()
            return np.array(data["embedding"], dtype=np.float32)

        except Exception as e:
            logger.warning(f"Failed to get embedding from {self.embedding_provider}: {e}")
            # Fallback to random embedding (better than crashing)
            return np.random.randn(768).astype(np.float32)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            a: First vector
            b: Second vector

        Returns:
            Similarity score (0-1)
        """
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    @classmethod
    def _create_tool_text(cls, tool: Any) -> str:
        """Create semantic description of tool for embedding.

        Combines tool name, description, parameter names, and use cases to create
        a rich semantic representation that matches user queries better.

        Uses the ToolMetadataProvider contract: tool.get_metadata() is guaranteed
        to return valid ToolMetadata (either explicit or auto-generated from
        tool properties). Falls back to YAML only for legacy tools without
        get_metadata().

        Args:
            tool: Tool object

        Returns:
            Semantic text description enriched with use cases
        """
        # Start with name (important for matching)
        parts = [tool.name.replace("_", " ")]

        # Add description
        if tool.description:
            parts.append(tool.description)

        # Add parameter names (provide additional context)
        if hasattr(tool, "parameters") and tool.parameters:
            params = tool.parameters.get("properties", {})
            if params:
                param_names = ", ".join(params.keys())
                parts.append(f"Parameters: {param_names}")

        # Use get_metadata() for ToolMetadataProvider contract
        # This guarantees valid metadata (explicit or auto-generated)
        if hasattr(tool, "get_metadata"):
            metadata = tool.get_metadata()
            if metadata.use_cases:
                parts.append(f"Use for: {', '.join(metadata.use_cases)}.")
            if metadata.keywords:
                parts.append(f"Common requests: {', '.join(metadata.keywords)}.")
            if metadata.examples:
                parts.append(f"Examples: {', '.join(metadata.examples)}.")
        else:
            # Fallback for legacy tools without get_metadata()
            use_cases = cls._get_tool_use_cases(tool.name)
            if use_cases:
                parts.append(use_cases)

        return ". ".join(parts)

    @classmethod
    def _get_tool_use_cases(cls, tool_name: str) -> str:
        """Get common use cases for a tool to improve semantic matching.

        Loads tool knowledge from YAML configuration file (tool_knowledge.yaml)
        which is more maintainable than hardcoded dictionaries.

        Args:
            tool_name: Name of the tool

        Returns:
            String describing common use cases with rich keywords and examples
        """
        # Try to get from YAML-loaded knowledge first
        use_case_text = cls._build_use_case_text(tool_name)
        if use_case_text:
            return use_case_text

        # Fallback to empty string if not found in YAML
        # The tool's description from the tool definition will still be used
        return ""

    # ========================================================================
    # Classification-Aware Tool Selection (UnifiedTaskClassifier Integration)
    # ========================================================================

    # Task type to logical category mapping
    TASK_TYPE_CATEGORIES = {
        "analysis": ["analysis", "code_intel", "file_ops"],
        "action": ["execution", "git_ops", "file_ops"],
        "generation": ["file_ops", "generation", "refactoring"],
        "search": ["code_intel", "file_ops"],
        "edit": ["file_ops", "refactoring", "git_ops"],
        "default": ["file_ops", "execution"],
    }

    # Tools to exclude based on negated keywords
    KEYWORD_TOOL_MAPPING = {
        "analyze": ["analyze_docs", "analyze_metrics", "code_review"],
        "review": ["code_review", "analyze_docs"],
        "test": ["run_tests", "execute_bash"],
        "run": ["execute_bash", "run_tests"],
        "execute": ["execute_bash", "execute_python_in_sandbox"],
        "search": ["code_search", "semantic_code_search", "web_search"],
        "find": ["find_symbol", "find_references", "code_search"],
        "create": ["write_file", "generate_docs"],
        "generate": ["generate_docs", "write_file"],
        "refactor": ["refactor_extract_function", "refactor_inline_variable", "rename_symbol"],
        "edit": ["edit_files", "edit_file", "write_file"],
        "commit": ["git_suggest_commit", "execute_bash"],
        "deploy": ["execute_bash"],
    }

    def _get_tools_for_task_type(self, task_type_str: str) -> List[str]:
        """Get relevant tools based on task type.

        Args:
            task_type_str: Task type as string (e.g., "analysis", "action")

        Returns:
            List of tool names relevant to this task type
        """
        categories = self.TASK_TYPE_CATEGORIES.get(task_type_str, ["file_ops", "execution"])
        tools = []
        for category in categories:
            tools.extend(self.get_tools_for_logical_category(category))
        return list(set(tools))

    def _get_excluded_tools_from_negations(
        self,
        negated_keywords: List[Any],
    ) -> set:
        """Get tools that should be excluded based on negated keywords.

        Args:
            negated_keywords: List of KeywordMatch objects with negated keywords

        Returns:
            Set of tool names to exclude from selection
        """
        excluded = set()
        for match in negated_keywords:
            keyword = match.keyword if hasattr(match, "keyword") else str(match)
            if keyword in self.KEYWORD_TOOL_MAPPING:
                excluded.update(self.KEYWORD_TOOL_MAPPING[keyword])
                logger.debug(f"Excluding tools for negated '{keyword}': {self.KEYWORD_TOOL_MAPPING[keyword]}")
        return excluded

    def _adjust_threshold_by_confidence(
        self,
        base_threshold: float,
        classification_confidence: float,
    ) -> float:
        """Adjust similarity threshold based on classification confidence.

        Higher confidence = stricter threshold (more focused selection)
        Lower confidence = looser threshold (broader selection)

        Args:
            base_threshold: Base similarity threshold
            classification_confidence: Confidence from classifier (0.0 - 1.0)

        Returns:
            Adjusted threshold
        """
        # High confidence (>0.7): tighten threshold by up to 0.05
        # Low confidence (<0.3): loosen threshold by up to 0.05
        confidence_adjustment = (classification_confidence - 0.5) * 0.1
        adjusted = base_threshold + confidence_adjustment
        # Clamp between reasonable bounds
        return max(0.1, min(0.3, adjusted))

    async def select_tools_with_classification(
        self,
        user_message: str,
        tools: ToolRegistry,
        classification_result: "ClassificationResult",
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        max_tools: int = 5,
        base_similarity_threshold: float = 0.15,
    ) -> List[ToolDefinition]:
        """Select tools using classification result for smarter selection.

        Integrates with UnifiedTaskClassifier to:
        1. Use task type for category filtering
        2. Exclude tools related to negated keywords
        3. Adjust thresholds based on classification confidence
        4. Include context from conversation history

        Args:
            user_message: Current user message
            tools: Tool registry
            classification_result: Result from UnifiedTaskClassifier
            conversation_history: Optional conversation history
            max_tools: Maximum tools to return
            base_similarity_threshold: Base minimum similarity score

        Returns:
            List of relevant ToolDefinition objects
        """
        # Get task type and confidence
        task_type_str = classification_result.task_type.value
        confidence = classification_result.confidence

        logger.debug(
            f"Classification-aware selection: type={task_type_str}, "
            f"confidence={confidence:.2f}"
        )

        # Get tools excluded by negated keywords
        excluded_tools = self._get_excluded_tools_from_negations(
            classification_result.negated_keywords
        )
        if excluded_tools:
            logger.debug(f"Tools excluded by negation: {excluded_tools}")

        # Adjust threshold based on confidence
        similarity_threshold = self._adjust_threshold_by_confidence(
            base_similarity_threshold, confidence
        )
        logger.debug(f"Adjusted similarity threshold: {similarity_threshold:.3f}")

        # Get task-type-specific tools
        task_tools = self._get_tools_for_task_type(task_type_str)
        logger.debug(f"Task-type tools ({len(task_tools)}): {task_tools[:5]}...")

        # Get mandatory tools from keywords
        mandatory_tool_names = self._get_mandatory_tools(user_message)

        # Remove negated tools from mandatory
        mandatory_tool_names = [t for t in mandatory_tool_names if t not in excluded_tools]

        # Get query embedding
        query_embedding = await self._get_embedding(user_message)

        # Calculate similarity scores
        similarities: List[Tuple[Any, float]] = []

        for tool in tools.list_tools():
            # Skip excluded tools
            if tool.name in excluded_tools:
                continue

            # Skip if not in task-type tools or mandatory (unless default type)
            if (
                task_type_str != "default"
                and tool.name not in task_tools
                and tool.name not in mandatory_tool_names
            ):
                continue

            # Get cached embedding or compute on-demand
            if tool.name in self._tool_embedding_cache:
                tool_embedding = self._tool_embedding_cache[tool.name]
            else:
                tool_text = self._create_tool_text(tool)
                tool_embedding = await self._get_embedding(tool_text)

            # Cosine similarity
            similarity = self._cosine_similarity(query_embedding, tool_embedding)

            # Boost mandatory tools
            if tool.name in mandatory_tool_names:
                similarity = max(similarity, 0.9)

            # Apply usage boost
            usage_boost = await self._get_usage_boost(tool.name, user_message)
            similarity += usage_boost

            # Apply cost penalty
            cost_penalty = self._get_cost_penalty(tool, tools)
            similarity -= cost_penalty

            # Boost tools matching task type
            if tool.name in task_tools:
                similarity += 0.05  # Small boost for task-type alignment

            if similarity >= similarity_threshold:
                similarities.append((tool, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Build final selection
        selected_tools = []
        selected_names = set()

        # Add mandatory tools first
        mandatory_tools = [tool for tool in tools.list_tools() if tool.name in mandatory_tool_names]
        for tool in mandatory_tools:
            if tool.name not in selected_names:
                selected_tools.append((tool, 0.9))
                selected_names.add(tool.name)

        # Add top semantic matches
        for tool, score in similarities:
            if tool.name not in selected_names and len(selected_tools) < max_tools:
                selected_tools.append((tool, score))
                selected_names.add(tool.name)

        # Smart fallback if too few tools
        MIN_TOOLS = 2
        if len(selected_tools) < MIN_TOOLS:
            fallback_names = self._get_fallback_tools(
                tools, max_tools - len(selected_tools), query=user_message
            )
            for fallback_name in fallback_names:
                if fallback_name not in selected_names and fallback_name not in excluded_tools:
                    fallback_tool = tools.get(fallback_name)
                    if fallback_tool:
                        selected_tools.append((fallback_tool, 0.5))
                        selected_names.add(fallback_name)

        # Log selection
        tool_names = [t.name for t, _ in selected_tools]
        scores = [f"{s:.3f}" for _, s in selected_tools]
        logger.info(
            f"Classification-aware selection: {len(selected_tools)} tools "
            f"(type={task_type_str}, excluded={len(excluded_tools)}): "
            f"{', '.join(f'{name}({score})' for name, score in zip(tool_names, scores, strict=False))}"
        )

        # Record usage for learning
        for tool_name in tool_names:
            self._record_tool_usage(tool_name, user_message, success=True)

        # Generate cost warnings
        self._last_cost_warnings = self._generate_cost_warnings(selected_tools, tools)

        # Convert to ToolDefinition
        return [
            ToolDefinition(name=tool.name, description=tool.description, parameters=tool.parameters)
            for tool, _ in selected_tools
        ]

    def get_classification_tool_stats(self) -> Dict[str, Any]:
        """Get statistics about classification-aware tool selection.

        Returns:
            Dictionary with selection statistics
        """
        return {
            "task_type_categories": len(self.TASK_TYPE_CATEGORIES),
            "keyword_tool_mappings": len(self.KEYWORD_TOOL_MAPPING),
            "usage_cache_size": len(self._tool_usage_cache),
            "embedding_cache_size": len(self._tool_embedding_cache),
        }

    async def close(self) -> None:
        """Close HTTP client and save usage cache (Phase 3)."""
        # Phase 3: Save usage statistics before shutdown
        self._save_usage_cache()

        if self._client:
            await self._client.aclose()
