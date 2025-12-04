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

"""Intent classification using semantic embeddings.

This module provides intent classification for detecting:
- Continuation intent: Model wants to continue exploring/analyzing
- Completion intent: Model has finished and is providing a structured response

Uses semantic similarity with embeddings instead of hardcoded phrase matching.
This reduces false positives and handles variations better.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

from victor.embeddings.collections import CollectionItem, StaticEmbeddingCollection
from victor.embeddings.service import EmbeddingService

logger = logging.getLogger(__name__)


class IntentType(Enum):
    """Types of intent that can be classified."""

    CONTINUATION = "continuation"  # Model wants to continue (future-looking)
    COMPLETION = "completion"  # Model has finished (summary/conclusion)
    NEUTRAL = "neutral"  # Neither clear continuation nor completion


@dataclass
class IntentResult:
    """Result of intent classification."""

    intent: IntentType
    confidence: float  # 0-1 confidence score
    top_matches: List[Tuple[str, float]]  # Top matching intents with scores


# Canonical phrases for continuation intent (future-looking)
CONTINUATION_PHRASES = [
    # "Let me" patterns
    "Let me examine the code",
    "Let me look at the file",
    "Let me read this",
    "Let me check the implementation",
    "Let me analyze this further",
    "Let me investigate",
    "Let me start by reading",
    "Let me begin with",
    # "I'll" patterns
    "I'll examine the code next",
    "I'll look at the implementation",
    "I'll read the file",
    "I'll check this",
    "I'll analyze this",
    "I'll investigate further",
    # "Now let's" patterns
    "Now let's look at",
    "Now let's examine",
    "Now let's check",
    "Now let's analyze",
    # "Next" patterns
    "Next I'll examine",
    "Next, I'll look at",
    "Next, let me check",
    # In-progress patterns
    "I'm going to read",
    "I'm going to examine",
    "I'm going to check",
    "I need to look at",
    "I should examine",
]

# Canonical phrases for completion intent (finished/summarizing)
COMPLETION_PHRASES = [
    # Summary patterns
    "In summary, here are the findings",
    "To summarize the analysis",
    "Here is a summary of",
    "In conclusion, the analysis shows",
    # "Here are" patterns
    "Here are the main issues",
    "Here are the improvements",
    "Here are my recommendations",
    "Here are the key findings",
    "Here is the complete analysis",
    # List/structured patterns
    "The following improvements are recommended",
    "These are the main areas for improvement",
    "The key points are",
    "Based on my analysis, here are",
    # Conclusion patterns
    "Overall, the code shows",
    "In total, there are",
    "The analysis reveals",
    "After reviewing the code",
    # Numbered section starts (common in structured responses)
    "## 1. First finding",
    "## 2. Second finding",
    "1. The first issue is",
    "2. The second issue is",
]


class IntentClassifier:
    """Semantic intent classifier using embeddings.

    Replaces hardcoded phrase matching with semantic similarity.
    Benefits:
    - Handles variations and paraphrases
    - Reduces false positives
    - More robust to model output variations

    Usage:
        classifier = IntentClassifier()
        await classifier.initialize()

        result = await classifier.classify_intent(
            "Let me start by examining the orchestrator.py file"
        )
        print(result.intent)  # IntentType.CONTINUATION
        print(result.confidence)  # 0.85
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        embedding_service: Optional[EmbeddingService] = None,
        continuation_threshold: float = 0.30,
        completion_threshold: float = 0.30,
    ):
        """Initialize intent classifier.

        Args:
            cache_dir: Directory for cache files
            embedding_service: Shared embedding service
            continuation_threshold: Minimum similarity for continuation intent (default 0.30)
                Note: Semantic similarity scores typically range 0.3-0.9 for related texts.
                Lower values (0.25-0.35) catch more matches but may have false positives.
                Higher values (0.5+) are more conservative but may miss legitimate cases.
            completion_threshold: Minimum similarity for completion intent (default 0.30)
        """
        self.cache_dir = cache_dir or Path.home() / ".victor" / "embeddings"
        self.embedding_service = embedding_service or EmbeddingService.get_instance()
        self.continuation_threshold = continuation_threshold
        self.completion_threshold = completion_threshold

        # Create collections
        self._continuation_collection = StaticEmbeddingCollection(
            name="intent_continuation",
            cache_dir=self.cache_dir,
            embedding_service=self.embedding_service,
        )
        self._completion_collection = StaticEmbeddingCollection(
            name="intent_completion",
            cache_dir=self.cache_dir,
            embedding_service=self.embedding_service,
        )

        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if classifier is initialized."""
        return self._initialized

    async def initialize(self) -> None:
        """Initialize intent collections with canonical phrases."""
        if self._initialized:
            return

        # Build continuation collection
        continuation_items = [
            CollectionItem(
                id=f"cont_{i}",
                text=phrase,
                metadata={"intent": "continuation"},
            )
            for i, phrase in enumerate(CONTINUATION_PHRASES)
        ]
        await self._continuation_collection.initialize(continuation_items)

        # Build completion collection
        completion_items = [
            CollectionItem(
                id=f"comp_{i}",
                text=phrase,
                metadata={"intent": "completion"},
            )
            for i, phrase in enumerate(COMPLETION_PHRASES)
        ]
        await self._completion_collection.initialize(completion_items)

        self._initialized = True
        logger.info(
            f"IntentClassifier initialized with {len(CONTINUATION_PHRASES)} continuation "
            f"and {len(COMPLETION_PHRASES)} completion phrases"
        )

    def initialize_sync(self) -> None:
        """Initialize intent collections (sync version)."""
        if self._initialized:
            return

        # Build continuation collection
        continuation_items = [
            CollectionItem(
                id=f"cont_{i}",
                text=phrase,
                metadata={"intent": "continuation"},
            )
            for i, phrase in enumerate(CONTINUATION_PHRASES)
        ]
        self._continuation_collection.initialize_sync(continuation_items)

        # Build completion collection
        completion_items = [
            CollectionItem(
                id=f"comp_{i}",
                text=phrase,
                metadata={"intent": "completion"},
            )
            for i, phrase in enumerate(COMPLETION_PHRASES)
        ]
        self._completion_collection.initialize_sync(completion_items)

        self._initialized = True
        logger.info(
            f"IntentClassifier initialized with {len(CONTINUATION_PHRASES)} continuation "
            f"and {len(COMPLETION_PHRASES)} completion phrases"
        )

    async def classify_intent(
        self,
        text: str,
        top_k: int = 3,
    ) -> IntentResult:
        """Classify the intent of text.

        Args:
            text: Text to classify (usually model's response start)
            top_k: Number of top matches to return

        Returns:
            IntentResult with classified intent, confidence, and top matches
        """
        if not self._initialized:
            await self.initialize()

        # Search both collections
        continuation_results = await self._continuation_collection.search(text, top_k=top_k)
        completion_results = await self._completion_collection.search(text, top_k=top_k)

        # Get best scores
        best_continuation = continuation_results[0][1] if continuation_results else 0.0
        best_completion = completion_results[0][1] if completion_results else 0.0

        # Build top matches for debugging
        top_matches = []
        for item, score in continuation_results[:2]:
            top_matches.append((f"cont:{item.text[:50]}", score))
        for item, score in completion_results[:2]:
            top_matches.append((f"comp:{item.text[:50]}", score))

        # Determine intent based on thresholds and relative scores
        if best_continuation >= self.continuation_threshold and best_continuation > best_completion:
            return IntentResult(
                intent=IntentType.CONTINUATION,
                confidence=best_continuation,
                top_matches=top_matches,
            )
        elif best_completion >= self.completion_threshold and best_completion > best_continuation:
            return IntentResult(
                intent=IntentType.COMPLETION,
                confidence=best_completion,
                top_matches=top_matches,
            )
        else:
            # Neither meets threshold or too close to call
            return IntentResult(
                intent=IntentType.NEUTRAL,
                confidence=max(best_continuation, best_completion),
                top_matches=top_matches,
            )

    def classify_intent_sync(
        self,
        text: str,
        top_k: int = 3,
    ) -> IntentResult:
        """Classify the intent of text (sync version).

        Args:
            text: Text to classify
            top_k: Number of top matches to return

        Returns:
            IntentResult with classified intent, confidence, and top matches
        """
        if not self._initialized:
            self.initialize_sync()

        # Search both collections
        continuation_results = self._continuation_collection.search_sync(text, top_k=top_k)
        completion_results = self._completion_collection.search_sync(text, top_k=top_k)

        # Get best scores
        best_continuation = continuation_results[0][1] if continuation_results else 0.0
        best_completion = completion_results[0][1] if completion_results else 0.0

        # Build top matches for debugging
        top_matches = []
        for item, score in continuation_results[:2]:
            top_matches.append((f"cont:{item.text[:50]}", score))
        for item, score in completion_results[:2]:
            top_matches.append((f"comp:{item.text[:50]}", score))

        # Determine intent based on thresholds and relative scores
        if best_continuation >= self.continuation_threshold and best_continuation > best_completion:
            return IntentResult(
                intent=IntentType.CONTINUATION,
                confidence=best_continuation,
                top_matches=top_matches,
            )
        elif best_completion >= self.completion_threshold and best_completion > best_continuation:
            return IntentResult(
                intent=IntentType.COMPLETION,
                confidence=best_completion,
                top_matches=top_matches,
            )
        else:
            return IntentResult(
                intent=IntentType.NEUTRAL,
                confidence=max(best_continuation, best_completion),
                top_matches=top_matches,
            )

    def intends_to_continue(self, text: str) -> bool:
        """Quick check if text indicates continuation intent.

        This is a convenience method for the orchestrator to use
        instead of hardcoded phrase matching.

        Args:
            text: Text to check

        Returns:
            True if text indicates continuation intent
        """
        result = self.classify_intent_sync(text)
        return result.intent == IntentType.CONTINUATION

    def is_complete_response(self, text: str) -> bool:
        """Quick check if text indicates a complete response.

        Args:
            text: Text to check

        Returns:
            True if text indicates completion intent
        """
        result = self.classify_intent_sync(text)
        return result.intent == IntentType.COMPLETION

    def clear_cache(self) -> None:
        """Clear cached collections."""
        self._continuation_collection.clear_cache()
        self._completion_collection.clear_cache()
        self._initialized = False
        logger.info("IntentClassifier cache cleared")
