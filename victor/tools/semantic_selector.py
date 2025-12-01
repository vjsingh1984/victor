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
import pickle
import re
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import httpx
import numpy as np
import yaml

from victor.providers.base import ToolDefinition
from victor.tools.base import CostTier, ToolRegistry

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
        """Load tool knowledge from YAML file.

        Returns:
            Dictionary mapping tool names to their knowledge (use_cases, keywords, examples)
        """
        if cls._tool_knowledge_loaded:
            return cls._tool_knowledge or {}

        # Find the tool_knowledge.yaml file
        config_dir = Path(__file__).parent.parent / "config"
        yaml_path = config_dir / "tool_knowledge.yaml"

        if not yaml_path.exists():
            logger.warning(f"Tool knowledge file not found: {yaml_path}")
            cls._tool_knowledge = {}
            cls._tool_knowledge_loaded = True
            return {}

        try:
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f) or {}

            cls._tool_knowledge = data
            cls._tool_knowledge_loaded = True
            logger.info(f"Loaded tool knowledge for {len(data)} tools from {yaml_path}")
            return data

        except Exception as e:
            logger.warning(f"Failed to load tool knowledge from {yaml_path}: {e}")
            cls._tool_knowledge = {}
            cls._tool_knowledge_loaded = True
            return {}

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
            cache_dir = Path.home() / ".victor" / "embeddings"
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

        # Sentence-transformers model (loaded on demand)
        self._sentence_model = None

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

    async def initialize_tool_embeddings(self, tools: ToolRegistry) -> None:
        """Pre-compute embeddings for all tools (called once at startup).

        Loads from pickle cache if available and tools haven't changed.
        Otherwise, computes embeddings and saves to cache.

        Args:
            tools: Tool registry with all available tools
        """
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

    # Tool categories for intelligent selection
    TOOL_CATEGORIES = {
        "file_ops": ["read_file", "write_file", "edit_files", "list_directory"],
        "git_ops": ["execute_bash", "git_suggest_commit", "git_create_pr", "git_analyze_conflicts"],
        "analysis": ["analyze_docs", "analyze_metrics", "code_review", "security_scan"],
        "refactoring": [
            "refactor_extract_function",
            "refactor_inline_variable",
            "rename_symbol",
            "refactor_organize_imports",
        ],
        "generation": [
            "generate_docs",
            "semantic_code_search",
            "code_search",
            "plan_files",
            "scaffold",
        ],
        "execution": ["execute_bash", "execute_python_in_sandbox", "run_tests"],
        "code_intel": ["find_symbol", "find_references", "semantic_code_search", "code_search"],
        "web": ["web_search", "web_fetch", "web_summarize"],
        "workflows": ["run_workflow", "batch", "cicd"],
    }

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
    }

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

    def _get_fallback_tools(self, tools: "ToolRegistry", max_tools: int = 5) -> List[str]:
        """Get fallback tools when semantic selection returns too few results.

        Instead of broadcasting ALL tools (which wastes tokens), return a
        curated list of common, universally useful tools.

        Args:
            tools: Tool registry
            max_tools: Maximum fallback tools to return

        Returns:
            List of fallback tool names
        """
        fallback = []
        for tool_name in self.COMMON_FALLBACK_TOOLS:
            if tools.is_tool_enabled(tool_name) and tools.get(tool_name):
                fallback.append(tool_name)
            if len(fallback) >= max_tools:
                break

        logger.info(f"Using fallback tools ({len(fallback)}): {fallback}")
        return fallback

    def _get_mandatory_tools(self, query: str) -> List[str]:
        """Get tools that MUST be included based on keywords.

        Args:
            query: User query

        Returns:
            List of tool names that are mandatory for this query
        """
        mandatory = []
        query_lower = query.lower()

        for keyword, tools in self.MANDATORY_TOOL_KEYWORDS.items():
            if self._keyword_in_text(query_lower, keyword):
                mandatory.extend(tools)
                logger.debug(f"Mandatory tools for '{keyword}': {tools}")

        return list(set(mandatory))

    def _get_relevant_categories(self, query: str) -> List[str]:
        """Determine which tool categories are relevant for this query.

        Args:
            query: User query

        Returns:
            List of relevant tool names from categories
        """
        query_lower = query.lower()
        relevant_tools = []

        # Multi-step tasks need file_ops and git_ops
        if any(sep in query for sep in [";", "then", "after", "next", "and then"]):
            relevant_tools.extend(self.TOOL_CATEGORIES.get("file_ops", []))
            relevant_tools.extend(self.TOOL_CATEGORIES.get("git_ops", []))
            logger.debug("Multi-step task detected, including file_ops and git_ops")

        # Analysis keywords
        if any(kw in query_lower for kw in ["analyze", "review", "check", "scan", "audit"]):
            relevant_tools.extend(self.TOOL_CATEGORIES.get("analysis", []))
            logger.debug("Analysis task detected")

        # Editing keywords
        if any(kw in query_lower for kw in ["edit", "modify", "change", "update", "fix"]):
            relevant_tools.extend(self.TOOL_CATEGORIES.get("file_ops", []))
            relevant_tools.extend(self.TOOL_CATEGORIES.get("refactoring", []))
            logger.debug("Editing task detected")

        # Git/diff keywords
        if any(
            self._keyword_in_text(query_lower, kw)
            for kw in ["diff", "commit", "pr", "git", "pull request"]
        ):
            relevant_tools.extend(self.TOOL_CATEGORIES.get("git_ops", []))
            logger.debug("Git operation detected")

        # Code navigation
        if any(kw in query_lower for kw in ["find", "locate", "search", "where"]):
            relevant_tools.extend(self.TOOL_CATEGORIES.get("code_intel", []))
            logger.debug("Code navigation detected")

        # Generation/creation
        if any(kw in query_lower for kw in ["create", "generate", "make", "write new"]):
            relevant_tools.extend(self.TOOL_CATEGORIES.get("file_ops", []))
            relevant_tools.extend(self.TOOL_CATEGORIES.get("generation", []))
            logger.debug("Generation task detected")

        # Default: file_ops + execution
        if not relevant_tools:
            relevant_tools.extend(self.TOOL_CATEGORIES.get("file_ops", []))
            relevant_tools.extend(self.TOOL_CATEGORIES.get("execution", []))
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
            cost_tier = tools.get_tool_cost(tool.name)
            if cost_tier and cost_tier in COST_TIER_WARNINGS:
                warning_msg = f"[{tool.name}] {COST_TIER_WARNINGS[cost_tier]}"
                warnings.append(warning_msg)
                logger.info(f"Cost warning for user: {warning_msg}")

        return warnings

    def get_last_cost_warnings(self) -> List[str]:
        """Get cost warnings from the last tool selection.

        Returns:
            List of warning messages about high-cost tools selected.
            Empty list if no high-cost tools were selected.
        """
        return self._last_cost_warnings.copy()

    def clear_cost_warnings(self) -> None:
        """Clear stored cost warnings."""
        self._last_cost_warnings = []

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
            logger.info(f"Pending actions: {pending_actions}")

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
                logger.info(
                    f"Added mandatory tools for pending '{action}': {pending_action_tools[action]}"
                )

        mandatory_tool_names = list(set(mandatory_tool_names))
        logger.info(f"Total mandatory tools: {mandatory_tool_names}")

        # Phase 1: Get relevant categories
        category_tools = self._get_relevant_categories(enhanced_query)
        logger.info(f"Category tools ({len(category_tools)}): {category_tools[:5]}...")

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
            fallback_names = self._get_fallback_tools(tools, max_tools - len(selected_tools))
            for fallback_name in fallback_names:
                if fallback_name not in selected_names:
                    fallback_tool = tools.get(fallback_name)
                    if fallback_tool:
                        selected_tools.append((fallback_tool, 0.5))
                        selected_names.add(fallback_name)
            logger.info(
                f"Added {len(fallback_names)} fallback tools (selection returned < {MIN_TOOLS_THRESHOLD})"
            )

        # Log selection
        tool_names = [t.name for t, _ in selected_tools]
        scores = [f"{s:.3f}" for _, s in selected_tools]
        logger.info(
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
            logger.warning(
                f"Cost warnings for selected tools: {len(self._last_cost_warnings)} high-cost tools"
            )

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
        logger.info(f"Mandatory tools: {mandatory_tool_names}")

        # Phase 1: Get relevant categories
        category_tools = self._get_relevant_categories(user_message)
        logger.info(f"Category tools ({len(category_tools)}): {category_tools[:5]}...")

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
            fallback_names = self._get_fallback_tools(tools, max_tools - len(selected_tools))
            for fallback_name in fallback_names:
                if fallback_name not in selected_names:
                    fallback_tool = tools.get(fallback_name)
                    if fallback_tool:
                        selected_tools.append((fallback_tool, 0.5))  # Default score for fallback
                        selected_names.add(fallback_name)
            logger.info(
                f"Added {len(fallback_names)} fallback tools (semantic selection returned < {MIN_TOOLS_THRESHOLD})"
            )

        # Log selection
        tool_names = [t.name for t, _ in selected_tools]
        scores = [f"{s:.3f}" for _, s in selected_tools]
        logger.info(
            f"Selected {len(selected_tools)} tools (mandatory={len(mandatory_tools)}): "
            f"{', '.join(f'{name}({score})' for name, score in zip(tool_names, scores, strict=False))}"
        )

        # Phase 6: Generate and store cost warnings for high-cost tools
        self._last_cost_warnings = self._generate_cost_warnings(selected_tools, tools)
        if self._last_cost_warnings:
            logger.warning(
                f"Cost warnings for selected tools: {len(self._last_cost_warnings)} high-cost tools"
            )

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
        """Get embedding from sentence-transformers (local, fast).

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        try:
            # Lazy load sentence-transformers model
            if self._sentence_model is None:
                try:
                    from sentence_transformers import SentenceTransformer

                    logger.info(f"Loading sentence-transformers model: {self.embedding_model}")
                    self._sentence_model = SentenceTransformer(self.embedding_model)
                    logger.info("Model loaded successfully (local, ~5ms per embedding)")
                except ImportError:
                    raise ImportError(
                        "sentence-transformers not installed. "
                        "Install with: pip install sentence-transformers"
                    )

            # Run in thread pool to avoid blocking event loop
            import asyncio

            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None, lambda: self._sentence_model.encode(text, convert_to_numpy=True)
            )
            return embedding.astype(np.float32)

        except Exception as e:
            logger.warning(f"Failed to get embedding from sentence-transformers: {e}")
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

        # Enrich with use cases based on tool name (improves semantic matching)
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

    async def close(self) -> None:
        """Close HTTP client and save usage cache (Phase 3)."""
        # Phase 3: Save usage statistics before shutdown
        self._save_usage_cache()

        if self._client:
            await self._client.aclose()
