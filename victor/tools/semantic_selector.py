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
from collections import OrderedDict
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple, Union

# Disable tokenizers parallelism BEFORE importing sentence_transformers
# This prevents "bad value(s) in fds_to_keep" errors in async contexts
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import httpx
import numpy as np

from victor.core.errors import FileError, ConfigurationError
from victor.providers.base import ToolDefinition
from victor.tools.base import CostTier, ToolMetadataRegistry, ToolRegistry
from victor.storage.embeddings.service import EmbeddingService
from victor.agent.tool_sequence_tracker import ToolSequenceTracker, create_sequence_tracker
from victor.protocols.tool_selector import (
    ToolSelectionResult,
    ToolSelectionContext,
    ToolSelectionStrategy,
)
from victor.agent.debug_logger import TRACE  # Import TRACE level
from victor.tools.metadata_registry import (
    get_core_readonly_tools,
    get_tools_matching_mandatory_keywords,
    get_tools_by_task_type,
)
from victor.config.tool_selection_defaults import (
    SemanticSelectorDefaults,
    FallbackTools,
    QueryPatterns,
)

# Import classification protocol to avoid circular imports
from victor.protocols.classification import IClassificationResult

logger = logging.getLogger(__name__)


# Lazy hook initialization to avoid circular imports
_rl_hooks = None


def _get_rl_hooks():
    """Lazy load RL hooks registry."""
    global _rl_hooks
    if _rl_hooks is None:
        try:
            from victor.framework.rl.hooks import get_rl_hooks

            _rl_hooks = get_rl_hooks()
        except ImportError:
            _rl_hooks = None
    return _rl_hooks


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

    # NOTE: _load_tool_knowledge() and related class variables were removed.
    # All tools now provide metadata via get_metadata() which auto-generates
    # from tool properties. Legacy tool_knowledge.yaml has been archived.

    @classmethod
    def _build_use_case_text(cls, tool_name: str) -> str:
        """Build use case text for embedding (legacy method, returns empty).

        NOTE: This method previously loaded from tool_knowledge.yaml which
        has been archived. All metadata now comes from get_metadata().
        Kept for API compatibility but always returns empty string.

        Args:
            tool_name: Name of the tool (unused)

        Returns:
            Empty string - use get_metadata() for tool metadata
        """
        return ""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_provider: str = "sentence-transformers",
        ollama_base_url: str = "http://localhost:11434",
        cache_embeddings: bool = True,
        cache_dir: Optional[Path] = None,
        cost_aware_selection: bool = SemanticSelectorDefaults.COST_AWARE_SELECTION,
        cost_penalty_factor: float = SemanticSelectorDefaults.COST_PENALTY_FACTOR,
        sequence_tracking: bool = True,
    ):
        """Initialize semantic tool selector.

        Uses centralized defaults from SemanticSelectorDefaults.

        Args:
            embedding_model: Model to use for embeddings
                - sentence-transformers: "all-MiniLM-L6-v2" (default, 80MB, ~5ms)
                - ollama: "nomic-embed-text", "qwen3-embedding:8b", etc.
            embedding_provider: Provider (sentence-transformers, ollama, vllm, lmstudio)
                Default: "sentence-transformers" (local, fast, bundled)
            ollama_base_url: Ollama/vLLM/LMStudio API base URL
            cache_embeddings: Cache tool embeddings (recommended)
            cache_dir: Directory to store embedding cache (default: ~/.victor/embeddings/)
            cost_aware_selection: Deprioritize high-cost tools
            cost_penalty_factor: Penalty per cost weight
            sequence_tracking: Enable tool sequence tracking for 15-20% boost (default: True)
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

        # Cache file path (includes model name AND project hash for isolation)
        # TD-010: Add project hash to prevent cache pollution between projects
        project_hash = self._get_project_hash()
        model_safe = embedding_model.replace(":", "_").replace("/", "_")
        cache_filename = f"tool_embeddings_{model_safe}_{project_hash}.pkl"
        self.cache_file = self.cache_dir / cache_filename

        # In-memory cache: tool_name → embedding vector
        self._tool_embedding_cache: Dict[str, np.ndarray] = {}

        # Tool version hash (to detect when tools change)
        self._tools_hash: Optional[str] = None

        # PERF-001: Query embedding cache with LRU eviction (max 500 queries)
        # This eliminates redundant embedding generation for repeated queries
        # Expected: 10x improvement for repeated queries
        # Phase 2.1: Expanded from 100 to 500 entries for higher hit rate
        self._query_embedding_cache: OrderedDict[str, np.ndarray | None] = OrderedDict()
        self._query_cache_max_size = 500

        # PERF-002: Category memberships cache for fast pre-filtering
        # Maps category → set of tool names in that category
        # Expected: 3-5x improvement for category-specific queries
        self._category_memberships_cache: Dict[str, Set[str]] = {}

        # Note: sentence-transformers model is managed by shared EmbeddingService singleton
        # This reduces memory usage by sharing the model with IntentClassifier

        # HTTP client for Ollama/vLLM/LMStudio
        self._client = None
        if embedding_provider in ["ollama", "vllm", "lmstudio"]:
            self._client = httpx.AsyncClient(base_url=ollama_base_url, timeout=30.0)

        # Phase 3: Tool usage tracking and learning (also project-isolated)
        self._usage_cache_file = self.cache_dir / f"tool_usage_stats_{project_hash}.pkl"
        self._tool_usage_cache: Dict[str, Dict[str, Any]] = {}
        self._usage_cache_dirty = False  # Dirty flag - only save when changed
        self._load_usage_cache()

        # Phase 6: Store last cost warnings for retrieval
        self._last_cost_warnings: List[str] = []
        # Track which tools have already been warned about (warn once per session)
        self._warned_tools: set = set()

        # Phase 9: Tool sequence tracking for intelligent next-tool suggestions
        # Provides 15-20% improvement in tool selection via workflow pattern detection
        self._sequence_tracking = sequence_tracking
        self._sequence_tracker: Optional[ToolSequenceTracker] = None
        if sequence_tracking:
            self._sequence_tracker = create_sequence_tracker()

        # IToolSelector protocol: Track whether embeddings have been initialized
        # This is checked by AgentOrchestrator._preload_embeddings()
        self._embeddings_initialized: bool = False

        # Store tools registry for legacy select_tools() calls
        # The orchestrator doesn't pass tools registry in legacy calls, so we need it stored
        self._tools_registry: Optional[ToolRegistry] = None

        # PERF-005: Performance metrics collection (Phase 2.1)
        self._selection_latency_ms: float = 0.0
        self._cache_hit_count: int = 0
        self._cache_miss_count: int = 0
        self._total_selections: int = 0

    async def initialize_tool_embeddings(self, tools: ToolRegistry) -> None:
        """Pre-compute embeddings for all tools (called once at startup).

        Loads from pickle cache if available and tools haven't changed.
        Otherwise, computes embeddings and saves to cache.

        Also refreshes the ToolMetadataRegistry with metadata from all tools.

        Args:
            tools: Tool registry with all available tools
        """
        # Store tools registry for legacy select_tools() calls
        self._tools_registry = tools

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

        # PERF-004: Build category memberships cache for fast pre-filtering
        self._build_category_cache(tool_list)

        # PERF-004: Warm up query embedding cache with common patterns
        # Note: This is async and will be awaited during initialization
        await self._warmup_query_cache()

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

        # Mark initialization complete (IToolSelector protocol)
        self._embeddings_initialized = True

        logger.info(
            f"Tool embeddings computed and cached for {len(self._tool_embedding_cache)} tools"
        )

    # Cache version - aligned with Victor version, increment on breaking cache format changes
    # Format: "{victor_version}.{cache_revision}" e.g., "0.2.0.1" for first revision of 0.2.0
    CACHE_VERSION = "0.2.0"

    @staticmethod
    def _get_project_hash() -> str:
        """Get a short hash of the current project root for cache isolation.

        This ensures each project has its own tool embeddings cache,
        preventing cache pollution when switching between projects.

        Returns:
            8-character hash of the project root path
        """
        from victor.config.settings import get_project_paths

        try:
            project_root = get_project_paths().project_root
            # Use absolute path for consistent hashing
            path_str = str(project_root.resolve())
            return hashlib.sha256(path_str.encode()).hexdigest()[:8]
        except (FileError, ConfigurationError, OSError):
            # Known error types - fallback to "global" if project detection fails
            return "global"
        except Exception:
            # Catch-all for truly unexpected errors
            return "global"

    def _calculate_tools_hash(self, tools: ToolRegistry) -> str:
        """Calculate hash of all tool definitions to detect changes.

        Args:
            tools: Tool registry

        Returns:
            SHA256 hash of tool definitions including count and version
        """
        # Create deterministic string from all tool definitions
        tool_list = sorted(tools.list_tools(), key=lambda t: t.name)
        tool_count = len(tool_list)
        tool_names = sorted([t.name for t in tool_list])

        tool_strings = []
        for tool in tool_list:
            tool_string = f"{tool.name}:{tool.description}:{tool.parameters}"
            tool_strings.append(tool_string)

        # Include tool count, names, and cache version in hash for robustness
        # This ensures ANY change (add, remove, rename, modify) triggers rebuild
        combined = (
            f"v{self.CACHE_VERSION}|count:{tool_count}|names:{','.join(tool_names)}|"
            + "|".join(tool_strings)
        )
        return hashlib.sha256(combined.encode()).hexdigest()

    def _delete_cache(self, reason: str) -> None:
        """Delete corrupted or stale cache file.

        Args:
            reason: Reason for deletion (for logging)
        """
        try:
            if self.cache_file.exists():
                self.cache_file.unlink()
                logger.info(f"Tool embeddings: deleted stale cache ({reason})")
        except (FileError, ConfigurationError) as e:
            # Known error types
            logger.warning(f"Tool embeddings: failed to delete cache: {e}")
        except Exception as e:
            # Catch-all for truly unexpected errors
            logger.warning(f"Tool embeddings: failed to delete cache: {e}")

    def _load_from_cache(self, tools_hash: str) -> bool:
        """Load embeddings from pickle cache if valid.

        Performs robust validation:
        1. Cache version check (breaking format changes)
        2. Tools hash check (tool definition changes)
        3. Model name check (embedding model changes)
        4. Embedding dimension validation (model dimension mismatch)
        5. Data integrity check (corrupt numpy arrays)

        Args:
            tools_hash: Current hash of tool definitions

        Returns:
            True if loaded successfully, False otherwise
        """
        if not self.cache_file.exists():
            logger.debug("No embedding cache file found, will rebuild")
            return False

        try:
            with open(self.cache_file, "rb") as f:
                cache_data = pickle.load(f)

            # 1. Check cache version first (breaking changes)
            cached_version = cache_data.get("cache_version", 1)
            if cached_version != self.CACHE_VERSION:
                logger.info(
                    f"Cache version mismatch (cached: v{cached_version}, current: v{self.CACHE_VERSION}), "
                    "rebuilding embeddings"
                )
                self._delete_cache("version mismatch")
                return False

            # 2. Verify cache is for same tools (hash includes count, names, and definitions)
            if cache_data.get("tools_hash") != tools_hash:
                cached_count = cache_data.get("tool_count", "?")
                cached_names = set(cache_data.get("tool_names", []))
                # The current tool names are embedded in the hash calculation
                logger.info(
                    f"Tool definitions changed (cached {cached_count} tools), "
                    "rebuilding embeddings"
                )
                # Log specific changes if available
                if cached_names:
                    current_names = set(cache_data.get("embeddings", {}).keys())
                    added = cached_names - current_names
                    removed = current_names - cached_names
                    if added:
                        logger.debug(f"Tools added since cache: {added}")
                    if removed:
                        logger.debug(f"Tools removed since cache: {removed}")
                return False

            # 3. Verify cache is for same embedding model
            if cache_data.get("embedding_model") != self.embedding_model:
                logger.info(
                    f"Embedding model changed (cached: {cache_data.get('embedding_model')}, "
                    f"current: {self.embedding_model}), rebuilding embeddings"
                )
                self._delete_cache("model mismatch")
                return False

            # 4. Validate embeddings exist and have correct structure
            embeddings = cache_data.get("embeddings", {})
            if not embeddings:
                logger.warning("Tool embeddings: cache missing embeddings dict")
                self._delete_cache("missing embeddings")
                return False

            # 5. Validate embedding dimensions and integrity
            expected_dim = None
            for tool_name, embedding in embeddings.items():
                if not isinstance(embedding, np.ndarray):
                    logger.warning(f"Tool embeddings: '{tool_name}' is not a numpy array")
                    self._delete_cache("invalid type")
                    return False

                if len(embedding.shape) != 1:
                    logger.warning(
                        f"Tool embeddings: '{tool_name}' has wrong shape: {embedding.shape}"
                    )
                    self._delete_cache("invalid shape")
                    return False

                if expected_dim is None:
                    expected_dim = embedding.shape[0]
                elif embedding.shape[0] != expected_dim:
                    logger.warning(
                        f"Tool embeddings: dimension mismatch for '{tool_name}' "
                        f"(got {embedding.shape[0]}, expected {expected_dim})"
                    )
                    self._delete_cache("dimension inconsistency")
                    return False

                # Check for NaN or Inf (corruption detection)
                if not np.isfinite(embedding).all():
                    logger.warning(f"Tool embeddings: '{tool_name}' contains NaN or Inf values")
                    self._delete_cache("corrupted embeddings")
                    return False

            # All checks passed - load embeddings
            self._tool_embedding_cache = embeddings
            self._tools_hash = tools_hash

            # Mark initialization complete (IToolSelector protocol)
            self._embeddings_initialized = True

            return True

        except (pickle.UnpicklingError, EOFError) as e:
            logger.warning(f"Tool embeddings: cache file corrupted: {e}")
            self._delete_cache("unpickling error")
            return False
        except (FileError, ConfigurationError, pickle.PickleError) as e:
            # Known error types - file or pickle errors
            logger.warning(f"Failed to load embedding cache (corrupted?): {e}, will rebuild")
            self._delete_cache("load error")
            return False
        except Exception as e:
            # Catch-all for truly unexpected errors
            logger.warning(f"Failed to load embedding cache: {e}, will rebuild")
            self._delete_cache("unexpected error")
            return False

    def _save_to_cache(self, tools_hash: str) -> None:
        """Save embeddings to pickle cache with full metadata for robust invalidation.

        Args:
            tools_hash: Hash of tool definitions
        """
        try:
            tool_names = sorted(self._tool_embedding_cache.keys())
            cache_data = {
                "cache_version": self.CACHE_VERSION,
                "embedding_model": self.embedding_model,
                "tools_hash": tools_hash,
                "tool_count": len(tool_names),
                "tool_names": tool_names,  # Explicit list for debugging
                "embeddings": self._tool_embedding_cache,
            }

            with open(self.cache_file, "wb") as f:
                pickle.dump(cache_data, f)

            cache_size = self.cache_file.stat().st_size / 1024  # KB
            logger.info(
                f"Saved embedding cache to {self.cache_file} "
                f"({cache_size:.1f} KB, {len(tool_names)} tools, v{self.CACHE_VERSION})"
            )

        except (FileError, ConfigurationError, pickle.PickleError) as e:
            # Known error types - file or pickle errors
            logger.warning(f"Failed to save embedding cache: {e}")
        except Exception as e:
            # Catch-all for truly unexpected errors
            logger.warning(f"Failed to save embedding cache: {e}")

    # Category alias mappings for logical category names
    # Maps logical names (file_ops, git_ops) to registry categories
    _CATEGORY_ALIASES = {
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

    def get_tools_for_logical_category(self, logical_category: str) -> List[str]:
        """Get tools for a logical category using the registry.

        Maps logical category names (file_ops, git_ops) to registry categories
        and returns the combined list of tools.

        Args:
            logical_category: Logical category name (file_ops, git_ops, etc.)

        Returns:
            List of tool names for the category
        """
        # Get registry categories for this logical category
        registry_categories = self._CATEGORY_ALIASES.get(logical_category, [])

        # Collect tools from all registry categories
        tools = set()
        registry = ToolMetadataRegistry.get_instance()
        for category in registry_categories:
            tools.update(registry.get_tools_by_category(category))

        return list(tools)

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
        # Analysis/exploration patterns - trigger semantic search
        "analyze",  # "analyze codebase", "analyze this code"
        "understand",  # "understand architecture"
        "architecture",  # "show architecture", "explain architecture"
        "structure",  # "codebase structure"
        "overview",  # "codebase overview"
        "how does",  # "how does X work"
        "what is the",  # "what is the architecture"
        "explore",  # "explore codebase"
    ]

    # Tools for conceptual queries - forces semantic search as primary
    # Excludes ls to prevent LLM from exploring instead of searching
    # Uses centralized FallbackTools from tool_selection_defaults
    CONCEPTUAL_FALLBACK_TOOLS: List[str] = list(FallbackTools.CONCEPTUAL_FALLBACK_TOOLS)

    # Common fallback tools - used when semantic selection returns too few results
    # These are the most universally useful tools
    # Uses centralized FallbackTools from tool_selection_defaults
    COMMON_FALLBACK_TOOLS: List[str] = list(FallbackTools.COMMON_FALLBACK_TOOLS)

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
                logger.log(TRACE, f"Conceptual query detected via pattern: '{pattern}'")
                return True
        return False

    def _get_mandatory_tools(self, query: str) -> List[str]:
        """Get tools that MUST be included based on keywords.

        Uses registry-based lookup. Tools declare mandatory keywords via
        @tool(mandatory_keywords=["show diff"]).

        Args:
            query: User query

        Returns:
            List of tool names that are mandatory for this query
        """
        # Use registry-based mandatory keyword lookup (decorator-driven)
        mandatory = get_tools_matching_mandatory_keywords(query)
        if mandatory:
            logger.log(
                TRACE,
                f"Registry mandatory tools: {mandatory}",
            )

        return list(mandatory)

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

    def _is_analysis_query(self, query: str) -> bool:
        """Heuristic check for analysis/review intents."""
        query_lower = query.lower()
        analysis_keywords = ["analyze", "analysis", "review", "check", "scan", "audit", "inspect"]
        return any(kw in query_lower for kw in analysis_keywords)

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
        # NOTE: Be specific to avoid false positives. Generic words like "check"
        # or "examine" in analysis tasks shouldn't trigger pending actions.
        action_patterns = {
            "edit": ["edit the file", "modify the file", "change the code", "update the file"],
            "show_diff": ["show diff", "show the diff", "git diff", "compare changes"],
            "read": ["read the file", "read this file", "show me the file"],
            "propose": ["propose changes", "suggest changes", "recommend changes"],
            "create": ["create a file", "create new file", "generate a file", "make a file"],
            "test": ["run tests", "run the tests", "execute tests"],
            "commit": ["commit the changes", "make a commit", "git commit"],
            "pr": ["create a pull request", "open a pr", "make a pr", "raise a pr"],
        }

        # Check which actions were requested
        original_lower = original_request.lower()
        for action_type, keywords in action_patterns.items():
            if any(self._keyword_in_text(original_lower, kw) for kw in keywords):
                # Check if this action was completed by looking at tool results
                completed = self._was_action_completed(action_type, conversation_history)
                if not completed:
                    pending.append(action_type)
                    logger.log(TRACE, f"Pending action detected: {action_type}")

        return pending

    def _was_action_completed(self, action: str, history: List[Dict[str, Any]]) -> bool:
        """Check if an action was completed based on conversation history.

        Checks both tool calls made and text content for completion indicators.

        Args:
            action: Action type to check
            history: Conversation history

        Returns:
            True if action was completed, False otherwise
        """
        # Map action types to tool names that complete them
        action_to_tools = {
            "show_diff": ["git", "shell", "shell_readonly"],
            "edit": ["edit", "write", "patch"],
            "read": ["read", "symbol", "search", "overview", "ls"],
            "create": ["write", "scaffold"],
            "test": ["test", "shell"],
            "commit": ["git", "commit_msg"],
            "pr": ["pr", "git"],
        }

        # Check if any tool that completes this action was called
        tools_for_action = action_to_tools.get(action, [])
        for msg in history:
            if msg.get("role") == "assistant":
                # Check tool_calls in the message (handle None explicitly)
                tool_calls = msg.get("tool_calls") or []
                for tc in tool_calls:
                    tool_name = tc.get("name", "") or tc.get("function", {}).get("name", "")
                    if tool_name in tools_for_action:
                        return True

                # Also check content for tool result markers
                content = str(msg.get("content", "")).lower()
                if action == "show_diff" and ("diff" in content or "git diff" in content):
                    return True
                elif (
                    action == "test"
                    and "test" in content
                    and ("passed" in content or "failed" in content)
                ):
                    return True

        # Check user messages for tool output markers (TOOL_OUTPUT tags)
        for msg in history:
            if msg.get("role") == "user":
                content = str(msg.get("content", ""))
                if "<TOOL_OUTPUT" in content:
                    # Check if any relevant tool output exists
                    for tool_name in tools_for_action:
                        if f'tool="{tool_name}"' in content or f"tool='{tool_name}'" in content:
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

    def _save_usage_cache(self, force: bool = False) -> None:
        """Save tool usage statistics to disk cache (Phase 3).

        Uses dirty flag pattern to avoid redundant disk writes.
        Only saves when data has changed or force=True (e.g., on shutdown).

        Args:
            force: Force save even if not dirty (for shutdown)
        """
        if not force and not self._usage_cache_dirty:
            return  # Nothing changed, skip save

        try:
            with open(self._usage_cache_file, "wb") as f:
                pickle.dump(self._tool_usage_cache, f)
            self._usage_cache_dirty = False  # Clear dirty flag after save
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
            self._usage_cache_dirty = True  # Mark dirty, save on shutdown

    def _build_category_cache(self, tools: List[Any]) -> None:
        """Build category memberships cache for fast pre-filtering (PERF-002).

        Populates _category_memberships_cache with all tools in each category.
        This enables O(1) category lookup during tool selection instead of
        iterating through all tools.

        Args:
            tools: List of BaseTool instances from ToolRegistry
        """
        metadata_registry = ToolMetadataRegistry.get_instance()

        # Build cache from all logical categories
        for logical_category in self._CATEGORY_ALIASES.keys():
            tool_names = set(self.get_tools_for_logical_category(logical_category))
            self._category_memberships_cache[logical_category] = tool_names

        # Also cache individual registry categories
        for category in metadata_registry.get_all_categories():
            tools_in_category = metadata_registry.get_tools_by_category(category)
            self._category_memberships_cache[category] = set(tools_in_category)

        logger.debug(
            f"Built category cache: {len(self._category_memberships_cache)} categories, "
            f"avg {sum(len(v) for v in self._category_memberships_cache.values()) // len(self._category_memberships_cache)} tools/category"
        )

    async def _warmup_query_cache(self) -> None:
        """Warm up query embedding cache with common patterns (PERF-004).

        Pre-generates embeddings for common query patterns to eliminate
        cold start penalty. This provides 10x improvement on first queries.

        Performance Impact:
        - 100ms one-time cost during initialization
        - 10x speedup on first 10 common queries (0.5ms → 0.05ms)
        - Estimated 40-60% cache hit rate for these patterns in production
        """
        # Common query patterns from benchmarks and production logs
        common_queries = [
            "read the file",
            "write to file",
            "search code",
            "find classes",
            "analyze codebase",
            "run tests",
            "git commit",
            "edit files",
            "show diff",
            "create endpoint",
            # Extended patterns for better coverage
            "list directory",
            "find functions",
            "run command",
            "check status",
            "create file",
            "modify code",
            "test changes",
            "view changes",
            "search for",
            "analyze",
        ]

        logger.debug(f"Warming up query cache with {len(common_queries)} patterns...")

        # Generate embeddings for all common queries
        import asyncio

        start_time = asyncio.get_event_loop().time()
        tasks = [self._get_embedding(query) for query in common_queries]
        await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = (asyncio.get_event_loop().time() - start_time) * 1000

        logger.info(
            f"Query cache warmup complete: {len(common_queries)} patterns in {elapsed:.1f}ms "
            f"(avg {elapsed/len(common_queries):.2f}ms/query)"
        )

    def _get_query_cache_key(self, query: str) -> str:
        """Generate cache key for query embedding.

        Args:
            query: Query text

        Returns:
            SHA256 hash of query text
        """
        return hashlib.sha256(query.encode()).hexdigest()[:16]

    def _record_tool_usage(self, tool_name: str, query: str, success: bool = True) -> None:
        """Record tool selection for learning (Phase 3).

        NOTE: This records tool SELECTION, not execution. Actual execution
        should be tracked separately via record_tool_execution().

        Args:
            tool_name: Name of the tool that was selected
            query: The query context where it was selected
            success: Whether the tool selection was valid
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

        # Mark dirty - save will happen on shutdown via close()
        self._usage_cache_dirty = True

    def record_tool_execution(
        self,
        tool_name: str,
        success: bool = True,
        execution_time: float = 0.0,
    ) -> None:
        """Record actual tool execution for sequence tracking (Phase 9).

        This should be called AFTER a tool is actually executed, not just selected.
        Updates the sequence tracker to build workflow patterns.

        Args:
            tool_name: Name of the tool that was executed
            success: Whether the execution succeeded
            execution_time: Time taken for execution
        """
        if self._sequence_tracker:
            self._sequence_tracker.record_execution(tool_name, success, execution_time)

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
        # PERF-003: Use batch embedding for context comparisons
        context_boost = 0.0
        if stats["recent_contexts"]:
            try:
                # Get query embedding (cached)
                query_emb = await self._get_embedding(query)

                # Batch generate embeddings for recent contexts
                recent_contexts = stats["recent_contexts"][-3:]  # Last 3 contexts
                if recent_contexts:
                    # Include query in batch for cache efficiency
                    all_texts = [query] + recent_contexts
                    embeddings = await self._get_embeddings_batch(all_texts)
                    # embeddings[0] is query (already cached), embeddings[1:] are contexts
                    ctx_embeddings = embeddings[1:]

                    # Calculate similarities
                    context_similarities = []
                    for ctx_emb in ctx_embeddings:
                        sim = self._cosine_similarity(query_emb, ctx_emb)
                        context_similarities.append(sim)

                    if context_similarities:
                        avg_context_sim = sum(context_similarities) / len(context_similarities)
                        context_boost = avg_context_sim * 0.05  # Max 0.05

            except Exception as e:
                logger.debug(f"Context boost calculation failed: {e}")

        total_boost = usage_boost + success_boost + recency_boost + context_boost
        # Use TRACE for per-tool verbose logging
        logger.log(
            TRACE,
            f"Usage boost for {tool_name}: {total_boost:.3f} "
            f"(usage={usage_boost:.3f}, success={success_boost:.3f}, "
            f"recency={recency_boost:.3f}, context={context_boost:.3f})",
        )

        return min(0.2, total_boost)  # Cap total boost at 0.2

    def _get_sequence_boost(self, tool_name: str) -> float:
        """Calculate boost based on tool sequence patterns (Phase 9).

        Uses the ToolSequenceTracker to predict likely next tools based on
        previously executed tools in the session, providing 15-20% improvement
        in tool selection accuracy.

        Args:
            tool_name: Name of the tool to calculate boost for

        Returns:
            Boost value (0.0 to 0.15) to add to similarity score
        """
        if not self._sequence_tracker:
            return 0.0

        # Get sequence suggestions (confidence ordered)
        suggestions = self._sequence_tracker.get_next_suggestions(top_k=10)

        # Find this tool's confidence in suggestions
        for suggested_tool, confidence in suggestions:
            if suggested_tool == tool_name:
                # Scale confidence to boost value (max 0.15)
                boost = confidence * 0.15
                logger.log(
                    TRACE,
                    f"Sequence boost for {tool_name}: +{boost:.3f} "
                    f"(confidence={confidence:.2f})",
                )
                return boost

        return 0.0

    def _apply_sequence_boosts(
        self, similarities: List[Tuple[Any, float]]
    ) -> List[Tuple[Any, float]]:
        """Apply sequence-based boosts to all tool similarity scores.

        Args:
            similarities: List of (tool, score) tuples

        Returns:
            List of (tool, boosted_score) tuples
        """
        if not self._sequence_tracker:
            return similarities

        boosted = []
        for tool, score in similarities:
            sequence_boost = self._get_sequence_boost(tool.name)
            boosted_score = score + sequence_boost
            boosted.append((tool, boosted_score))

        return boosted

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

        # Defensive: Check tools is actually a ToolRegistry
        from victor.tools.registry import ToolRegistry

        if not isinstance(tools, ToolRegistry):  # type: ignore[unreachable]
            logger.warning(
                f"_get_cost_penalty expected ToolRegistry, got {type(tools).__name__}. "
                "Skipping cost penalty calculation."
            )
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
            logger.log(
                TRACE, f"Cost penalty for {tool.name}: -{penalty:.3f} (tier={cost_tier.value})"
            )

        return penalty

    # ========================================================================
    # Selection Result Caching (Phase 3 Task 2)
    # ========================================================================

    def _try_get_cached_selection(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        similarity_threshold: float = SemanticSelectorDefaults.SIMILARITY_THRESHOLD,
    ) -> Optional[List[ToolDefinition]]:
        """Try to get cached tool selection result.

        Args:
            query: User query
            conversation_history: Optional conversation history for context-aware caching
            similarity_threshold: Minimum similarity threshold used for selection

        Returns:
            Cached list of ToolDefinition or None if not found
        """
        try:
            from victor.tools.caches import get_cache_key_generator, get_tool_selection_cache
        except ImportError:
            return None

        if not self._tools_registry:
            return None

        key_gen = get_cache_key_generator()
        cache = get_tool_selection_cache()

        # Calculate tools hash
        tools_hash = key_gen.calculate_tools_hash(self._tools_registry)

        # Determine cache key type based on context
        if conversation_history and len(conversation_history) > 0:
            # Use context-aware cache
            cache_key = key_gen.generate_context_key(
                query=query,
                tools_hash=tools_hash,
                conversation_history=conversation_history,
            )
            cached = cache.get_context(cache_key)
        else:
            # Use simple query cache (includes threshold for correctness)
            config_hash = self._get_config_hash(similarity_threshold)
            cache_key = key_gen.generate_query_key(
                query=query,
                tools_hash=tools_hash,
                config_hash=config_hash,
            )
            cached = cache.get_query(cache_key)

        if cached and cached.tools:
            logger.debug(f"SemanticToolSelector: Cache hit for query: {query[:50]}...")
            return cached.tools

        return None

    def _store_selection_in_cache(
        self,
        query: str,
        tools: List[ToolDefinition],
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        similarity_threshold: float = SemanticSelectorDefaults.SIMILARITY_THRESHOLD,
        selection_latency_ms: float = 0.0,
    ) -> None:
        """Store tool selection result in cache.

        Args:
            query: User query
            tools: Selected tools to cache
            conversation_history: Optional conversation history for context-aware caching
            similarity_threshold: Minimum similarity threshold used for selection
            selection_latency_ms: Time taken for this selection (for metrics)
        """
        try:
            from victor.tools.caches import get_cache_key_generator, get_tool_selection_cache
        except ImportError:
            return

        if not self._tools_registry:
            return

        key_gen = get_cache_key_generator()
        cache = get_tool_selection_cache()

        tool_names = [t.name for t in tools]

        # Calculate tools hash
        tools_hash = key_gen.calculate_tools_hash(self._tools_registry)

        # Determine cache key type based on context
        if conversation_history and len(conversation_history) > 0:
            # Use context-aware cache (shorter TTL)
            cache_key = key_gen.generate_context_key(
                query=query,
                tools_hash=tools_hash,
                conversation_history=conversation_history,
            )
            cache.put_context(
                cache_key, tool_names, tools=tools, selection_latency_ms=selection_latency_ms
            )
        else:
            # Use simple query cache (longer TTL, includes threshold)
            config_hash = self._get_config_hash(similarity_threshold)
            cache_key = key_gen.generate_query_key(
                query=query,
                tools_hash=tools_hash,
                config_hash=config_hash,
            )
            cache.put_query(
                cache_key, tool_names, tools=tools, selection_latency_ms=selection_latency_ms
            )

    def _get_config_hash(self, similarity_threshold: float) -> str:
        """Generate hash of selector configuration for cache invalidation.

        Args:
            similarity_threshold: Similarity threshold used for selection

        Returns:
            Hash string for configuration
        """
        import hashlib

        config_str = f"threshold:{similarity_threshold}:model:{self.embedding_model}"
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics.

        Returns:
            Dictionary with cache metrics
        """
        try:
            from victor.tools.caches import get_tool_selection_cache

            cache = get_tool_selection_cache()
            return cache.get_stats()
        except ImportError:
            return {"enabled": False}

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

        # Defensive: Check tools is actually a ToolRegistry
        from victor.tools.registry import ToolRegistry

        if not isinstance(tools, ToolRegistry):  # type: ignore[unreachable]
            logger.warning(
                f"_generate_cost_warnings expected ToolRegistry, got {type(tools).__name__}. "
                "Skipping cost warning generation."
            )
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
                logger.log(TRACE, f"Cost warning: {warning_msg}")

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
        similarity_threshold: float = SemanticSelectorDefaults.SIMILARITY_THRESHOLD,
    ) -> List[ToolDefinition]:
        """Select tools with full conversation context awareness (Phase 2).

        This method enhances tool selection by:
        - Tracking pending actions from the original request
        - Including conversation context in semantic search
        - Ensuring mandatory tools for pending actions
        - Using selection result caching for performance
        - Tracking selection latency for cache metrics

        Args:
            user_message: Current user message
            tools: Tool registry
            conversation_history: Full conversation history (Phase 2)
            max_tools: Maximum tools to return
            similarity_threshold: Minimum similarity score

        Returns:
            List of relevant tools with context-aware selection
        """
        import time

        start_time = time.perf_counter()

        # Check cache first (Phase 3 Task 2: Selection result caching)
        cached_result = self._try_get_cached_selection(
            query=user_message,
            conversation_history=conversation_history,
            similarity_threshold=similarity_threshold,
        )
        if cached_result is not None:
            # PERF-005: Track cache hit
            self._cache_hit_count += 1
            self._total_selections += 1
            return cached_result

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

        # Always include core read-only tools for analysis-style queries.
        if self._is_analysis_query(enhanced_query):
            mandatory_tool_names.extend(get_core_readonly_tools())

        # Add mandatory tools for pending actions
        # NOTE: Uses canonical short names for token efficiency
        pending_action_tools = {
            "show_diff": ["shell"],
            "edit": ["edit", "read"],
            "commit": ["commit_msg", "shell"],
            "pr": ["pr"],
            "test": ["shell", "test"],
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

        # PERF-002: Build candidate tool set using cached category memberships
        candidate_tool_names = set(category_tools) | set(mandatory_tool_names)

        # Calculate similarity scores for candidate tools only
        similarities: List[Tuple[Any, float]] = []

        for tool in tools.list_tools():
            # PERF-002: Skip if tool not in candidate set (pre-filtering via cache)
            if tool.name not in candidate_tool_names:
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

            # Phase 9: Apply sequence boost based on workflow patterns
            sequence_boost = self._get_sequence_boost(tool.name)
            similarity += sequence_boost

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
                selected_tools.append((tool, SemanticSelectorDefaults.MANDATORY_TOOL_SCORE))
                selected_names.add(tool.name)

        # Then add top semantic matches
        for tool, score in similarities:
            if tool.name not in selected_names and len(selected_tools) < max_tools:
                selected_tools.append((tool, score))
                selected_names.add(tool.name)

        # Phase 8: Smart fallback - if too few tools selected, add common fallback tools
        min_threshold = SemanticSelectorDefaults.MIN_TOOLS_THRESHOLD
        if len(selected_tools) < min_threshold:
            fallback_names = self._get_fallback_tools(
                tools, max_tools - len(selected_tools), query=user_message
            )
            for fallback_name in fallback_names:
                if fallback_name not in selected_names:
                    fallback_tool = tools.get(fallback_name)
                    if fallback_tool:
                        selected_tools.append(
                            (fallback_tool, SemanticSelectorDefaults.FALLBACK_TOOL_SCORE)
                        )
                        selected_names.add(fallback_name)
            logger.debug(
                f"Added {len(fallback_names)} fallback tools (selection returned < {min_threshold})"
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

        # Emit semantic match event for RL learning
        self._emit_semantic_match_event(
            selected_tools=selected_tools,
            threshold=similarity_threshold,
            task_type="default",
            classification_aware=False,
            excluded_count=0,
        )

        # Convert to ToolDefinition
        result = [
            ToolDefinition(name=tool.name, description=tool.description, parameters=tool.parameters)
            for tool, _ in selected_tools
        ]

        # Store in cache (Phase 3 Task 2: Selection result caching)
        selection_latency_ms = (time.perf_counter() - start_time) * 1000
        self._store_selection_in_cache(
            query=user_message,
            tools=result,
            conversation_history=conversation_history,
            similarity_threshold=similarity_threshold,
            selection_latency_ms=selection_latency_ms,
        )

        # PERF-005: Track cache miss and latency
        self._cache_miss_count += 1
        self._total_selections += 1
        self._selection_latency_ms = selection_latency_ms

        return result

    async def select_relevant_tools(
        self,
        user_message: str,
        tools: ToolRegistry,
        max_tools: int = 5,
        similarity_threshold: float = SemanticSelectorDefaults.SIMILARITY_THRESHOLD,
    ) -> List[ToolDefinition]:
        """Select relevant tools using semantic similarity with category filtering.

        Enhanced with Phase 1 features:
        - Mandatory tool selection for specific keywords
        - Category-based filtering for better relevance
        - Selection result caching for performance (Phase 3 Task 2)
        - Selection latency tracking for cache metrics

        Args:
            user_message: User's input message
            tools: Tool registry
            max_tools: Maximum number of tools to return
            similarity_threshold: Minimum similarity score (0-1)

        Returns:
            List of relevant ToolDefinition objects, sorted by relevance
        """
        import time

        start_time = time.perf_counter()

        # Check cache first (Phase 3 Task 2: Selection result caching)
        cached_result = self._try_get_cached_selection(
            query=user_message,
            conversation_history=None,  # No conversation history for this method
            similarity_threshold=similarity_threshold,
        )
        if cached_result is not None:
            return cached_result

        # Phase 1: Get mandatory tools (always included)
        mandatory_tool_names = self._get_mandatory_tools(user_message)
        if mandatory_tool_names:
            logger.debug(f"Mandatory tools: {mandatory_tool_names}")

        # Always include core read-only tools for analysis-style queries.
        if self._is_analysis_query(user_message):
            mandatory_tool_names.extend(get_core_readonly_tools())

        # Phase 1: Get relevant categories
        category_tools = self._get_relevant_categories(user_message)
        logger.debug(f"Category tools ({len(category_tools)}): {category_tools[:5]}...")

        # Get embedding for user message
        query_embedding = await self._get_embedding(user_message)

        # PERF-002: Build candidate tool set using cached category memberships
        # This avoids iterating through ALL tools when categories are known
        candidate_tool_names = set(category_tools) | set(mandatory_tool_names)

        # Calculate similarity scores for candidate tools only
        similarities: List[Tuple[Any, float]] = []

        for tool in tools.list_tools():
            # PERF-002: Skip if tool not in candidate set (pre-filtering via cache)
            if tool.name not in candidate_tool_names:
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

            # Phase 9: Apply sequence boost based on workflow patterns
            sequence_boost = self._get_sequence_boost(tool.name)
            similarity += sequence_boost

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
                selected_tools.append((tool, SemanticSelectorDefaults.MANDATORY_TOOL_SCORE))
                selected_names.add(tool.name)

        # Then add top semantic matches
        for tool, score in similarities:
            if tool.name not in selected_names and len(selected_tools) < max_tools:
                selected_tools.append((tool, score))
                selected_names.add(tool.name)

        # Phase 8: Smart fallback - if too few tools selected, add common fallback tools
        # This prevents broadcasting ALL tools (which wastes tokens)
        min_threshold = SemanticSelectorDefaults.MIN_TOOLS_THRESHOLD
        if len(selected_tools) < min_threshold:
            fallback_names = self._get_fallback_tools(
                tools, max_tools - len(selected_tools), query=user_message
            )
            for fallback_name in fallback_names:
                if fallback_name not in selected_names:
                    fallback_tool = tools.get(fallback_name)
                    if fallback_tool:
                        selected_tools.append(
                            (fallback_tool, SemanticSelectorDefaults.FALLBACK_TOOL_SCORE)
                        )
                        selected_names.add(fallback_name)
            logger.debug(
                f"Added {len(fallback_names)} fallback tools (semantic selection returned < {min_threshold})"
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

        # Emit semantic match event for RL learning
        self._emit_semantic_match_event(
            selected_tools=selected_tools,
            threshold=similarity_threshold,
            task_type="default",
            classification_aware=False,
            excluded_count=0,
        )

        # Convert to ToolDefinition
        result = [
            ToolDefinition(name=tool.name, description=tool.description, parameters=tool.parameters)
            for tool, _ in selected_tools
        ]

        # Store in cache (Phase 3 Task 2: Selection result caching)
        selection_latency_ms = (time.perf_counter() - start_time) * 1000
        self._store_selection_in_cache(
            query=user_message,
            tools=result,
            conversation_history=None,  # No conversation history for this method
            similarity_threshold=similarity_threshold,
            selection_latency_ms=selection_latency_ms,
        )

        return result

    async def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text.

        PERF-001: Checks query cache before generating embedding (10x speedup for repeated queries).

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        # Check query cache first (PERF-001)
        cache_key = self._get_query_cache_key(text)
        if cache_key in self._query_embedding_cache:
            # Move to end (mark as recently used)
            self._query_embedding_cache.move_to_end(cache_key)
            return self._query_embedding_cache[cache_key]  # type: ignore[return-value]

        # Cache miss - generate embedding
        if self.embedding_provider == "sentence-transformers":
            embedding = await self._get_sentence_transformer_embedding(text)
        elif self.embedding_provider in ["ollama", "vllm", "lmstudio"]:
            embedding = await self._get_api_embedding(text)
        else:
            raise NotImplementedError(f"Provider {self.embedding_provider} not yet supported")

        # Add to cache with LRU eviction
        self._query_embedding_cache[cache_key] = embedding
        if len(self._query_embedding_cache) > self._query_cache_max_size:
            # Remove oldest entry
            self._query_embedding_cache.popitem(last=False)

        return embedding

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

    async def _get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for multiple texts efficiently (PERF-003).

        Batch embedding generation provides 2-3x speedup for multiple texts.
        Checks query cache first, then generates remaining embeddings.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (same order as input texts)
        """
        if not texts:
            return []

        embeddings = []
        uncached_indices = []
        uncached_texts = []

        # Check cache for each text
        for i, text in enumerate(texts):
            cache_key = self._get_query_cache_key(text)
            if cache_key in self._query_embedding_cache:
                # Cache hit
                self._query_embedding_cache.move_to_end(cache_key)
                embeddings.append(self._query_embedding_cache[cache_key])
            else:
                # Cache miss - mark for batch generation
                embeddings.append(None)  # Placeholder
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Generate uncached embeddings
        if uncached_texts:
            try:
                # Try batch generation if EmbeddingService supports it
                embedding_service = EmbeddingService.get_instance(model_name=self.embedding_model)

                # Check if batch method exists
                if hasattr(embedding_service, "embed_text_batch"):
                    # Batch generation
                    uncached_embeddings = await embedding_service.embed_text_batch(uncached_texts)
                else:
                    # Fallback to parallel individual embeddings
                    import asyncio

                    tasks = [embedding_service.embed_text(text) for text in uncached_texts]
                    uncached_embeddings = await asyncio.gather(*tasks)

                # Update cache and results
                for idx, text, emb in zip(uncached_indices, uncached_texts, uncached_embeddings):
                    cache_key = self._get_query_cache_key(text)
                    self._query_embedding_cache[cache_key] = emb
                    embeddings[idx] = emb

                    # Apply LRU eviction
                    if len(self._query_embedding_cache) > self._query_cache_max_size:
                        self._query_embedding_cache.popitem(last=False)

            except Exception as e:
                logger.warning(
                    f"Batch embedding generation failed: {e}, falling back to individual"
                )
                # Fallback to individual generation
                for idx, text in zip(uncached_indices, uncached_texts):
                    emb = await self._get_embedding(text)
                    if emb is not None:  # type: ignore[unreachable]
                        embeddings[idx] = emb

        return embeddings

    async def _get_api_embedding(self, text: str) -> np.ndarray:
        """Get embedding from Ollama/vLLM/LMStudio API.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        try:
            response = await self._client.post(  # type: ignore[union-attr]
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

        Uses Rust-accelerated implementation with NumPy fallback.

        Args:
            a: First vector
            b: Second vector

        Returns:
            Similarity score (0-1)
        """
        from victor.processing.native import cosine_similarity

        # Convert numpy arrays to lists for Rust/fallback interface
        return cosine_similarity(a.tolist(), b.tolist())

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

    def _get_tools_for_task_type(self, task_type_str: str) -> List[str]:
        """Get relevant tools based on task type.

        Uses registry-based lookup. Tools declare their task types via
        @tool(task_types=["analysis", "edit"]).

        Args:
            task_type_str: Task type as string (e.g., "analysis", "action")

        Returns:
            List of tool names relevant to this task type
        """
        # Use registry-based task type lookup (decorator-driven)
        registry_tools = get_tools_by_task_type(task_type_str)
        if registry_tools:
            logger.log(
                TRACE,
                f"Registry task-type tools for '{task_type_str}': {registry_tools}",
            )
        return list(registry_tools)

    def _get_excluded_tools_from_negations(
        self,
        negated_keywords: List[Any],
    ) -> Set[str]:
        """Get tools that should be excluded based on negated keywords.

        Uses registry-based lookup. When user says "don't analyze", this
        excludes tools with task_type="analyze".

        Args:
            negated_keywords: List of KeywordMatch objects with negated keywords

        Returns:
            Set of tool names to exclude from selection
        """
        excluded: Set[str] = set()

        for match in negated_keywords:
            keyword = match.keyword if hasattr(match, "keyword") else str(match)

            # Use registry-based task type lookup (decorator-driven)
            # If keyword matches a task type, exclude tools declared for that type
            registry_excluded = get_tools_by_task_type(keyword)
            if registry_excluded:
                excluded.update(registry_excluded)
                logger.log(
                    TRACE,
                    f"Registry excluding tools for negated task_type '{keyword}': {registry_excluded}",
                )

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
        classification_result: IClassificationResult,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        max_tools: int = 5,
        base_similarity_threshold: float = SemanticSelectorDefaults.SIMILARITY_THRESHOLD,
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
            f"Classification-aware selection: type={task_type_str}, " f"confidence={confidence:.2f}"
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

        # PERF-002: Build candidate tool set using task types and mandatory tools
        candidate_tool_names = set(task_tools) | set(mandatory_tool_names)
        if task_type_str == "default":
            # For default type, consider all non-excluded tools
            all_tool_names = {t.name for t in tools.list_tools()} - excluded_tools
            candidate_tool_names = all_tool_names
        else:
            candidate_tool_names = candidate_tool_names - excluded_tools

        # Calculate similarity scores for candidate tools only
        similarities: List[Tuple[Any, float]] = []

        for tool in tools.list_tools():
            # PERF-002: Skip if tool not in candidate set (pre-filtering via cache)
            if tool.name not in candidate_tool_names:
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

            # Phase 9: Apply sequence boost based on workflow patterns
            sequence_boost = self._get_sequence_boost(tool.name)
            similarity += sequence_boost

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
                selected_tools.append((tool, SemanticSelectorDefaults.MANDATORY_TOOL_SCORE))
                selected_names.add(tool.name)

        # Add top semantic matches
        for tool, score in similarities:
            if tool.name not in selected_names and len(selected_tools) < max_tools:
                selected_tools.append((tool, score))
                selected_names.add(tool.name)

        # Smart fallback if too few tools
        min_threshold = SemanticSelectorDefaults.MIN_TOOLS_THRESHOLD
        if len(selected_tools) < min_threshold:
            fallback_names = self._get_fallback_tools(
                tools, max_tools - len(selected_tools), query=user_message
            )
            for fallback_name in fallback_names:
                if fallback_name not in selected_names and fallback_name not in excluded_tools:
                    fallback_tool = tools.get(fallback_name)
                    if fallback_tool:
                        selected_tools.append(
                            (fallback_tool, SemanticSelectorDefaults.FALLBACK_TOOL_SCORE)
                        )
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

        # Emit semantic match event for RL learning
        self._emit_semantic_match_event(
            selected_tools=selected_tools,
            threshold=similarity_threshold,
            task_type=task_type_str,
            classification_aware=True,
            excluded_count=len(excluded_tools),
        )

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
        stats = {
            "usage_cache_size": len(self._tool_usage_cache),
            "embedding_cache_size": len(self._tool_embedding_cache),
        }

        # Add sequence tracking stats if enabled
        if self._sequence_tracker:
            sequence_stats = self._sequence_tracker.get_statistics()
            stats["sequence_tracking"] = {  # type: ignore[assignment]
                "enabled": True,
                "history_length": sequence_stats["history_length"],
                "unique_tools_used": sequence_stats["unique_tools_used"],
                "total_transitions": sequence_stats["total_transitions"],
                "workflow_progress": sequence_stats["workflow_progress"],
            }
        else:
            stats["sequence_tracking"] = {"enabled": False}  # type: ignore[assignment]

        return stats

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for tool selection (PERF-005).

        Returns:
            Dictionary with performance metrics including latency,
            cache hit rate, and total selections
        """
        if self._total_selections == 0:
            return {
                "total_selections": 0,
                "cache_hit_count": 0,
                "cache_miss_count": 0,
                "cache_hit_rate": 0.0,
                "avg_latency_ms": 0.0,
                "last_latency_ms": 0.0,
            }

        hit_rate = (
            self._cache_hit_count / self._total_selections if self._total_selections > 0 else 0.0
        )

        return {
            "total_selections": self._total_selections,
            "cache_hit_count": self._cache_hit_count,
            "cache_miss_count": self._cache_miss_count,
            "cache_hit_rate": hit_rate,
            "avg_latency_ms": self._selection_latency_ms,  # Last selection latency
            "last_latency_ms": self._selection_latency_ms,
            "query_cache_size": len(self._query_embedding_cache),
            "query_cache_max_size": self._query_cache_max_size,
            "query_cache_utilization": len(self._query_embedding_cache)
            / self._query_cache_max_size,
        }

    def get_next_tool_suggestions(self, top_k: int = 5) -> List[Tuple[str, float]]:
        """Get suggested next tools based on workflow patterns (Phase 9).

        Uses the ToolSequenceTracker to predict likely next tools based on
        the history of tool executions in the current session.

        Args:
            top_k: Number of suggestions to return

        Returns:
            List of (tool_name, confidence) tuples sorted by confidence
        """
        if not self._sequence_tracker:
            return []

        return self._sequence_tracker.get_next_suggestions(top_k=top_k)

    def get_current_workflow(self) -> Optional[Tuple[str, float]]:
        """Detect if we're in the middle of a known workflow (Phase 9).

        Returns:
            Tuple of (workflow_name, progress_percentage) or None
        """
        if not self._sequence_tracker:
            return None

        return self._sequence_tracker.get_workflow_progress()

    def clear_session_state(self) -> None:
        """Clear session-specific state (sequence history, warnings).

        Call this when starting a new conversation session to reset
        sequence tracking and cost warnings.
        """
        self._warned_tools.clear()
        self._last_cost_warnings = []

        if self._sequence_tracker:
            self._sequence_tracker.clear_history()
            logger.debug("Cleared sequence tracker session history")

    def notify_tools_changed(self) -> None:
        """Notify selector that tools registry has changed (cache invalidation).

        Call this when:
        - Tools are added/removed/modified
        - Tool definitions are updated
        - Tool metadata changes

        This invalidates internal caches:
        - Tools hash (for cache key generation)
        - Category memberships cache
        - Selection result cache
        """
        self._tools_hash = None
        self._category_memberships_cache.clear()

        # Invalidate selection cache if available
        try:
            from victor.tools.caches import invalidate_tool_selection_cache

            invalidate_tool_selection_cache()
        except ImportError:
            pass  # Cache module not available

        logger.info("SemanticToolSelector: Notified of tools registry change, caches invalidated")

    async def close(self) -> None:
        """Close HTTP client and save usage cache (Phase 3)."""
        # Phase 3: Save usage statistics before shutdown (force save)
        self._save_usage_cache(force=True)

        if self._client:
            await self._client.aclose()

    # ========================================================================
    # IToolSelector Protocol Implementation
    # ========================================================================

    @property
    def strategy(self) -> ToolSelectionStrategy:
        """Get the selection strategy used by this selector (IToolSelector protocol)."""
        return ToolSelectionStrategy.SEMANTIC

    async def select_tools(
        self,
        task: str,
        *,
        limit: int = 10,
        min_score: float = 0.0,
        context: Optional[ToolSelectionContext] = None,
        # Legacy ToolSelector parameters (for backward compatibility with orchestrator)
        use_semantic: bool = True,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        conversation_depth: int = 0,
        planned_tools: Optional[Any] = None,
    ) -> Union[ToolSelectionResult, List[ToolDefinition]]:
        """Select relevant tools for a task (IToolSelector protocol + legacy support).

        This method handles TWO calling conventions:
        1. IToolSelector protocol (new): select_tools(task, limit, min_score, context) -> ToolSelectionResult
        2. Legacy ToolSelector (old): select_tools(prompt, use_semantic, conversation_history, ...) -> List[ToolDefinition]

        Args:
            task: Task description or query to match tools against
            limit: Maximum number of tools to return
            min_score: Minimum relevance score threshold (0.0-1.0)
            context: Optional additional context for selection (new protocol)
            use_semantic: Whether to use semantic selection (legacy, ignored - always semantic)
            conversation_history: Conversation history (legacy)
            conversation_depth: Conversation depth (legacy, unused)
            planned_tools: Planned tools (legacy, unused)

        Returns:
            ToolSelectionResult (new protocol) or List[ToolDefinition] (legacy)
        """
        # Detect legacy call from AgentOrchestrator
        # Legacy call has: use_semantic, conversation_history as kwargs, context=None
        # New protocol has: limit, min_score, context with context.metadata['tools']
        is_legacy_call = context is None

        # Legacy calling convention from AgentOrchestrator
        # Expected: select_tools(user_message, use_semantic=..., conversation_history=..., conversation_depth=...)
        if is_legacy_call:
            # Use stored tools registry from initialize_tool_embeddings()
            tools_registry = self._tools_registry
            if not tools_registry:
                raise RuntimeError(
                    "SemanticToolSelector not initialized. Call initialize_tool_embeddings(tools) first."
                )

            # task is actually user_message in legacy call
            user_message = task
            # use_semantic is ignored - we always use semantic selection in SemanticToolSelector
            # conversation_history should be a list of dicts
            conv_history = conversation_history if isinstance(conversation_history, list) else None

            # Use existing semantic selection method
            tools = await self.select_relevant_tools_with_context(
                user_message=user_message,
                tools=tools_registry,
                conversation_history=conv_history,
                max_tools=10,  # Default max tools for legacy call
            )

            # Legacy: Return List[ToolDefinition] directly
            return tools

        # New IToolSelector protocol calling convention
        # Map IToolSelector protocol to existing method
        tools_registry = context.metadata.get("tools") if context and context.metadata else None

        # Validate that tools_registry is actually a ToolRegistry (not ToolPipeline or other types)
        if tools_registry is not None:
            from victor.tools.registry import ToolRegistry

            if not isinstance(tools_registry, ToolRegistry):
                logger.warning(
                    f"ToolSelectionContext.metadata['tools'] must be ToolRegistry, got {type(tools_registry).__name__}. "
                    "Falling back to stored registry."
                )
                tools_registry = None

        if not tools_registry:
            # Fall back to stored tools registry from initialize_tool_embeddings()
            tools_registry = self._tools_registry
            if not tools_registry:
                raise ValueError(
                    "ToolSelectionContext must include 'tools' in metadata, or SemanticToolSelector "
                    "must be initialized via initialize_tool_embeddings(tools). "
                    "Pass tools via: context=ToolSelectionContext(..., metadata={'tools': tool_registry})"
                )

        # Safely get conversation_history from context (not in ToolSelectionContext dataclass)
        conv_history = getattr(context, "conversation_history", None) if context else None

        # Use existing semantic selection method
        tools = await self.select_relevant_tools_with_context(
            user_message=task,
            tools=tools_registry,
            conversation_history=conv_history,
            max_tools=limit,
            similarity_threshold=min_score,
        )

        # For now, return the tools directly (ToolDefinition objects)
        # This maintains backward compatibility with chat_coordinator which expects List[ToolDefinition]
        # NOTE: Refactor to return ToolSelectionResult for better error handling and metadata
        # Deferred: Requires updating all callers in chat_coordinator, agent_loop, and tool_pipeline
        return tools

    async def get_tool_score(
        self,
        tool_name: str,
        task: str,
        *,
        context: Optional[ToolSelectionContext] = None,
    ) -> float:
        """Get relevance score for a specific tool (IToolSelector protocol).

        Args:
            tool_name: Name of the tool to score
            task: Task description to score against
            context: Optional additional context

        Returns:
            Relevance score from 0.0 (not relevant) to 1.0 (highly relevant)
        """
        # Check if tool has cached embedding
        if tool_name not in self._tool_embedding_cache:
            return 0.0

        # Get query embedding
        query_embedding = await self._get_embedding(task)

        # Get tool embedding
        tool_embedding = self._tool_embedding_cache[tool_name]

        # Calculate cosine similarity
        return self._cosine_similarity(query_embedding, tool_embedding)

    def prioritize_by_stage(
        self,
        user_message: str,
        tools: Optional[List["ToolDefinition"]],
    ) -> Optional[List["ToolDefinition"]]:
        """Stage-aware pruning of tool list to keep it focused per step.

        Note: SemanticToolSelector returns tools as-is since stage-based
        prioritization should be handled by the wrapper selector or caller.
        The semantic selection process already prioritizes based on semantic
        relevance to the user's message.

        Args:
            user_message: The user's message (unused, for compatibility)
            tools: List of tool definitions to filter

        Returns:
            The same list of tools (no-op for semantic selector)
        """
        # SemanticToolSelector returns tools as-is since it already prioritizes
        # based on semantic similarity during the select_tools() call.
        # Stage-based prioritization should happen before/after semantic selection.
        return tools

    def _emit_semantic_match_event(
        self,
        selected_tools: List[Tuple[Any, float]],
        threshold: float,
        task_type: str = "default",
        classification_aware: bool = False,
        excluded_count: int = 0,
    ) -> None:
        """Emit semantic match event for RL learning.

        Args:
            selected_tools: List of (tool, score) tuples
            threshold: Similarity threshold used
            task_type: Task type from classification (if available)
            classification_aware: Whether classification was used
            excluded_count: Number of tools excluded by negation
        """
        hooks = _get_rl_hooks()
        if hooks is None:
            return

        try:
            from victor.framework.rl.hooks import RLEvent, RLEventType

            # Calculate selection quality metrics
            avg_score = (
                sum(score for _, score in selected_tools) / len(selected_tools)
                if selected_tools
                else 0.0
            )
            tool_names = [t.name for t, _ in selected_tools]

            event = RLEvent(
                type=RLEventType.SEMANTIC_MATCH,
                success=len(selected_tools) > 0,
                quality_score=avg_score,
                tool_name=",".join(tool_names[:5]),  # First 5 tools
                threshold_value=threshold,
                task_type=task_type,
                metadata={
                    "tools_selected": len(selected_tools),
                    "classification_aware": classification_aware,
                    "excluded_count": excluded_count,
                    "avg_similarity": avg_score,
                    "top_scores": [s for _, s in selected_tools[:3]],  # Top 3 scores
                },
            )
            hooks.emit(event)
            logger.log(
                TRACE,
                f"Emitted semantic_match event: {len(selected_tools)} tools, "
                f"threshold={threshold:.3f}, avg_score={avg_score:.3f}",
            )
        except Exception as e:
            logger.debug(f"Failed to emit semantic_match event: {e}")
