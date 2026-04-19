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

"""MIPROv2 prompt optimization strategy with KNNFewShot selection.

Inspired by DSPy's KNNFewShot optimizer (arXiv:2604.04869), this strategy
selects few-shot demonstrations that are most similar to the current query
using embedding-based cosine similarity. Different queries get different
demonstrations, improving prompt relevance.

The strategy:
1. Embeds trace descriptions (task type, tools used, failures) into vectors
2. Embeds the current user query
3. Uses cosine similarity to find the most relevant traces
4. Falls back to score-based selection when embeddings are unavailable
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MIPROv2Strategy:
    """MIPROv2-style few-shot demonstration mining strategy.

    Uses embedding-based KNN selection to find the most relevant
    traces for a given query, producing input-aware few-shot examples.

    Attributes:
        _max_examples: Maximum number of few-shot examples to include.
    """

    def __init__(self, max_examples: int = 3):
        """Initialize the MIPROv2 strategy.

        Args:
            max_examples: Maximum number of few-shot examples to produce.
        """
        self._max_examples = max_examples

    def select_similar_traces(self, traces: List[Any], query: str, top_k: int = 3) -> List[Any]:
        """KNNFewShot: select traces most similar to the current query.

        Embeds trace descriptions and the query, then returns the top-k
        most similar traces by cosine similarity. Falls back to returning
        the first top_k traces if embedding is unavailable.

        Args:
            traces: List of trace objects with task_type, tool_call_details,
                and tool_failures attributes.
            query: The current user query to match against.
            top_k: Number of traces to return.

        Returns:
            List of selected traces, length <= top_k.
        """
        if not traces:
            return []

        embedded = self._embed_traces(traces)
        if not embedded:
            # Fallback: return first top_k traces (score-based ordering
            # is the caller's responsibility)
            return traces[:top_k]

        try:
            from victor.storage.embeddings.service import EmbeddingService
            import numpy as np

            svc = EmbeddingService.get_instance()
            query_emb = svc.embed_text_sync(query)
            corpus_embs = np.array([e for _, e in embedded])
            similarities = EmbeddingService.cosine_similarity_matrix(
                np.array(query_emb), corpus_embs
            )
            top_indices = np.argsort(similarities)[::-1][:top_k]
            return [embedded[i][0] for i in top_indices]
        except Exception:
            logger.debug("KNNFewShot: embedding similarity failed, falling back to top-k")
            return traces[:top_k]

    def _embed_traces(self, traces: List[Any]) -> List[Tuple[Any, Any]]:
        """Embed trace descriptions for KNN search.

        Builds a text description for each trace from its task type,
        tool names, and failure keys, then batch-embeds them.

        Args:
            traces: List of trace objects.

        Returns:
            List of (trace, embedding) tuples, or empty list on failure.
        """
        try:
            from victor.storage.embeddings.service import EmbeddingService

            svc = EmbeddingService.get_instance()
        except Exception:
            return []

        descriptions = []
        for trace in traces:
            task = getattr(trace, "task_type", "default")
            tools = [
                getattr(d, "tool_name", "") for d in getattr(trace, "tool_call_details", [])[:5]
            ]
            failures = list(getattr(trace, "tool_failures", {}).keys())
            desc = f"{task}: {' '.join(tools)} failures={','.join(failures)}"
            descriptions.append(desc)

        try:
            embeddings = svc.embed_batch_sync(descriptions)
            return list(zip(traces, embeddings))
        except Exception:
            return []

    def reflect(
        self,
        traces: List[Any],
        section_name: str,
        current_text: str,
        **kwargs: Any,
    ) -> str:
        """Reflect on traces to produce updated prompt section text.

        Filters to successful traces, optionally narrows them using
        KNN selection when a query is provided, and formats them
        into few-shot example text.

        Args:
            traces: List of trace objects with success and completion_score.
            section_name: Name of the prompt section being optimized.
            current_text: Current text of the prompt section.
            **kwargs: Optional keyword arguments. Supports:
                - query (str): Current user query for KNN selection.

        Returns:
            Updated prompt section text with few-shot examples.
        """
        query = kwargs.get("query")

        # Filter to successful traces
        successful = [
            t
            for t in traces
            if getattr(t, "success", False) and getattr(t, "completion_score", 0) > 0.5
        ]

        if not successful:
            return current_text

        # Use KNN selection when query is available and we have more
        # traces than needed
        if query and len(successful) > self._max_examples:
            successful = self.select_similar_traces(successful, query, self._max_examples)

        # Limit to max_examples
        selected = successful[: self._max_examples]

        # Format few-shot examples from selected traces
        examples = []
        for i, trace in enumerate(selected, 1):
            task = getattr(trace, "task_type", "unknown")
            tools = [
                getattr(d, "tool_name", "unknown")
                for d in getattr(trace, "tool_call_details", [])[:5]
            ]
            score = getattr(trace, "completion_score", 0)
            examples.append(
                f"Example {i} ({task}, score={score:.1f}): " f"tools=[{', '.join(tools)}]"
            )

        if not examples:
            return current_text

        example_text = "\n".join(examples)
        return f"{current_text}\n\n--- Few-shot demonstrations ---\n{example_text}"
