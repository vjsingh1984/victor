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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np

from victor.providers.base import ToolDefinition
from victor.tools.base import ToolRegistry

logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        embedding_model: str = "nomic-embed-text",
        embedding_provider: str = "ollama",
        ollama_base_url: str = "http://localhost:11434",
        cache_embeddings: bool = True,
        cache_dir: Optional[Path] = None,
    ):
        """Initialize semantic tool selector.

        Args:
            embedding_model: Model to use for embeddings
            embedding_provider: Provider (ollama, openai, sentence-transformers)
            ollama_base_url: Ollama API base URL
            cache_embeddings: Cache tool embeddings (recommended)
            cache_dir: Directory to store embedding cache (default: ~/.victor/embeddings/)
        """
        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider
        self.ollama_base_url = ollama_base_url
        self.cache_embeddings = cache_embeddings

        # Cache directory
        if cache_dir is None:
            cache_dir = Path.home() / ".victor" / "embeddings"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache file path (includes model name for version control)
        cache_filename = f"tool_embeddings_{embedding_model.replace(':', '_')}.pkl"
        self.cache_file = self.cache_dir / cache_filename

        # In-memory cache: tool_name → embedding vector
        self._tool_embedding_cache: Dict[str, np.ndarray] = {}

        # Tool version hash (to detect when tools change)
        self._tools_hash: Optional[str] = None

        # HTTP client for Ollama
        self._client = httpx.AsyncClient(base_url=ollama_base_url, timeout=30.0)

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

    async def select_relevant_tools(
        self,
        user_message: str,
        tools: ToolRegistry,
        max_tools: int = 10,
        similarity_threshold: float = 0.3,
    ) -> List[ToolDefinition]:
        """Select relevant tools using semantic similarity.

        Args:
            user_message: User's input message
            tools: Tool registry
            max_tools: Maximum number of tools to return
            similarity_threshold: Minimum similarity score (0-1)

        Returns:
            List of relevant ToolDefinition objects, sorted by relevance
        """
        # Get embedding for user message
        query_embedding = await self._get_embedding(user_message)

        # Calculate similarity scores for all tools
        similarities: List[Tuple[Any, float]] = []

        for tool in tools.list_tools():
            # Get cached embedding or compute on-demand
            if tool.name in self._tool_embedding_cache:
                tool_embedding = self._tool_embedding_cache[tool.name]
            else:
                tool_text = self._create_tool_text(tool)
                tool_embedding = await self._get_embedding(tool_text)

            # Cosine similarity
            similarity = self._cosine_similarity(query_embedding, tool_embedding)

            if similarity >= similarity_threshold:
                similarities.append((tool, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Take top-K
        top_tools = similarities[:max_tools]

        # Log selection
        tool_names = [t.name for t, _ in top_tools]
        scores = [f"{s:.3f}" for _, s in top_tools]
        logger.info(
            f"Selected {len(top_tools)} tools by semantic similarity: "
            f"{', '.join(f'{name}({score})' for name, score in zip(tool_names, scores))}"
        )

        # Convert to ToolDefinition
        return [
            ToolDefinition(name=tool.name, description=tool.description, parameters=tool.parameters)
            for tool, _ in top_tools
        ]

    async def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        if self.embedding_provider == "ollama":
            return await self._get_ollama_embedding(text)
        else:
            raise NotImplementedError(f"Provider {self.embedding_provider} not yet supported")

    async def _get_ollama_embedding(self, text: str) -> np.ndarray:
        """Get embedding from Ollama.

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
            logger.warning(f"Failed to get embedding from Ollama: {e}")
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

    @staticmethod
    def _create_tool_text(tool: Any) -> str:
        """Create semantic description of tool for embedding.

        Combines tool name, description, and parameter names to create
        a rich semantic representation.

        Args:
            tool: Tool object

        Returns:
            Semantic text description
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

        return ". ".join(parts)

    async def close(self) -> None:
        """Close HTTP client."""
        await self._client.aclose()
