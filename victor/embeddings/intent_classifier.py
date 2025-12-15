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
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

from victor.embeddings.collections import CollectionItem, StaticEmbeddingCollection
from victor.embeddings.service import EmbeddingService

logger = logging.getLogger(__name__)


# Compiled regex patterns for explicit continuation detection
# These are strong signals that the model wants to continue working
CONTINUATION_HEURISTIC_PATTERNS = [
    # "Let me read/check/examine/look at..."
    re.compile(
        r"\blet\s+me\s+(read|check|examine|look\s+at|analyze|review|explore|investigate)\b",
        re.IGNORECASE,
    ),
    # "First, let me..." or "Now let me..."
    re.compile(r"\b(first|now|next),?\s+let\s+me\b", re.IGNORECASE),
    # "I'll read/check/examine..."
    re.compile(
        r"\bi['']ll\s+(read|check|examine|look\s+at|analyze|review|explore)\b", re.IGNORECASE
    ),
    # "I need to read/check..."
    re.compile(r"\bi\s+need\s+to\s+(read|check|examine|look\s+at|analyze|review)\b", re.IGNORECASE),
    # "I should read/check..."
    re.compile(r"\bi\s+should\s+(read|check|examine|look\s+at|analyze|review)\b", re.IGNORECASE),
    # "Let me start by..."
    re.compile(r"\blet\s+me\s+start\s+by\b", re.IGNORECASE),
]


# Patterns that indicate the model is STUCK in a loop (says what it will do but doesn't)
# These should be detected to prevent continuation loops
STUCK_LOOP_HEURISTIC_PATTERNS = [
    # "I'm going to" without tool call
    re.compile(r"\bi[''`]?m\s+going\s+to\s+(read|examine|check|call|use)\b", re.IGNORECASE),
    # "I will now" without action
    re.compile(r"\bi\s+will\s+now\s+(read|examine|check|call|use)\b", re.IGNORECASE),
    # Long preamble about what will be done (indicates stalling)
    re.compile(
        r"(first|to\s+begin|to\s+start),?\s+i[''`]?(ll|'ll|m\s+going\s+to)",
        re.IGNORECASE,
    ),
    # Repeating tool name like "I'll use search... I'll use read..."
    re.compile(r"\bi[''`]?ll\s+use\s+\w+.*\bi[''`]?ll\s+use\s+\w+", re.IGNORECASE | re.DOTALL),
]


def _has_stuck_loop_heuristic(text: str) -> bool:
    """Check if text indicates model is stuck in a loop (planning but not executing).

    This detects patterns where the model keeps saying what it will do
    but never actually makes tool calls.

    Args:
        text: Text to check

    Returns:
        True if text shows stuck loop patterns
    """
    # Check if multiple planning statements exist (strong stuck signal)
    planning_count = 0
    for pattern in STUCK_LOOP_HEURISTIC_PATTERNS:
        matches = pattern.findall(text)
        planning_count += len(matches)

    if planning_count >= 2:
        logger.debug(f"Stuck loop heuristic: found {planning_count} planning statements")
        return True

    return False


# Compiled regex patterns for explicit asking-for-input detection
# These are strong signals that should override semantic classification
ASKING_INPUT_HEURISTIC_PATTERNS = [
    # "Would you like me to..." with question context
    re.compile(r"\bwould\s+you\s+like\s+(me\s+to|to\s+see)\b", re.IGNORECASE),
    # "Do you want me to..."
    re.compile(r"\bdo\s+you\s+want\s+(me\s+to|to\s+see)\b", re.IGNORECASE),
    # "Should I..." at start of sentence or after newline/period
    re.compile(r"(?:^|[.\n])\s*should\s+i\s+\w+", re.IGNORECASE | re.MULTILINE),
    # "Shall I..."
    re.compile(r"(?:^|[.\n])\s*shall\s+i\s+\w+", re.IGNORECASE | re.MULTILINE),
    # "Let me know if you..."
    re.compile(r"\blet\s+me\s+know\s+if\s+you\b", re.IGNORECASE),
    # "Which.*would you like" or "What.*would you like"
    re.compile(r"\b(which|what)\s+.*\bwould\s+you\s+like\b", re.IGNORECASE),
    # "How would you like me to..."
    re.compile(r"\bhow\s+would\s+you\s+like\s+(me\s+to|to)\b", re.IGNORECASE),
]


# Compiled regex patterns for detecting "task complete + optional elaboration" offers
# These questions indicate the task is DONE and model is just offering to go deeper
# Should be treated as COMPLETION, not ASKING_INPUT
COMPLETION_OFFER_HEURISTIC_PATTERNS = [
    # "Would you like me to elaborate/expand/go deeper..."
    re.compile(
        r"\bwould\s+you\s+like\s+(me\s+to\s+)?(elaborate|expand|go\s+deeper|dive\s+deeper|explain\s+further|provide\s+more\s+detail)\b",
        re.IGNORECASE,
    ),
    # "Do you want me to elaborate/expand..."
    re.compile(
        r"\bdo\s+you\s+want\s+(me\s+to\s+)?(elaborate|expand|go\s+deeper|dive\s+deeper|explain\s+further)\b",
        re.IGNORECASE,
    ),
    # "Let me know if you'd like more details on any of these"
    re.compile(
        r"\blet\s+me\s+know\s+if\s+you.*(more\s+details?|elaborate|go\s+deeper)\b", re.IGNORECASE
    ),
    # "Would you like more details on any of..."
    re.compile(
        r"\bwould\s+you\s+like\s+more\s+(details?|information)\s+on\s+(any|these|this)\b",
        re.IGNORECASE,
    ),
    # "I can elaborate on any of these if you'd like"
    re.compile(
        r"\bi\s+can\s+(elaborate|expand|explain\s+further|provide\s+more\s+detail)\b", re.IGNORECASE
    ),
]


def _has_completion_offer_heuristic(text: str) -> bool:
    """Check if text contains 'task complete + optional elaboration' patterns.

    These patterns indicate the model has finished the task and is offering
    to elaborate further. This should be treated as COMPLETION, not ASKING_INPUT,
    because the core task is done.

    Args:
        text: Text to check (usually last ~500 chars of response)

    Returns:
        True if text contains completion offer patterns
    """
    # Quick check: must have a question mark or "if you" to be offering something
    if "?" not in text and "if you" not in text.lower():
        return False

    # Check all heuristic patterns
    for pattern in COMPLETION_OFFER_HEURISTIC_PATTERNS:
        if pattern.search(text):
            logger.debug(f"Completion-offer heuristic matched: {pattern.pattern[:40]}...")
            return True

    return False


def _has_continuation_heuristic(text: str) -> bool:
    """Check if text contains strong continuation heuristic patterns.

    These patterns are unambiguous indicators that the model wants to continue
    exploring/reading, so they should override semantic classification when detected.

    Args:
        text: Text to check (usually last ~500 chars of response)

    Returns:
        True if text contains strong continuation patterns
    """
    # Check all heuristic patterns
    for pattern in CONTINUATION_HEURISTIC_PATTERNS:
        if pattern.search(text):
            logger.debug(f"Continuation heuristic matched: {pattern.pattern[:40]}...")
            return True

    return False


def _has_asking_input_heuristic(text: str) -> bool:
    """Check if text contains strong asking-for-input heuristic patterns.

    These patterns are unambiguous indicators that the model is asking for user input,
    so they should override semantic classification when detected.

    Args:
        text: Text to check (usually last ~500 chars of response)

    Returns:
        True if text contains strong asking-for-input patterns
    """
    # Quick check: must have a question mark to be asking something
    if "?" not in text:
        return False

    # Check all heuristic patterns
    for pattern in ASKING_INPUT_HEURISTIC_PATTERNS:
        if pattern.search(text):
            logger.debug(f"Asking-input heuristic matched: {pattern.pattern[:40]}...")
            return True

    return False


class IntentType(Enum):
    """Types of intent that can be classified."""

    CONTINUATION = "continuation"  # Model wants to continue (future-looking)
    COMPLETION = "completion"  # Model has finished (summary/conclusion)
    ASKING_INPUT = "asking_input"  # Model is asking for user input/confirmation
    STUCK_LOOP = "stuck_loop"  # Model is stuck in a loop (planning but not executing)
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
    "Let me review the file",
    "Let me review the contents",
    "Let me fetch the contents",
    # "I'll" patterns
    "I'll examine the code next",
    "I'll look at the implementation",
    "I'll read the file",
    "I'll check this",
    "I'll analyze this",
    "I'll investigate further",
    "I'll review the contents",
    "I'll review the file",
    "I'll fetch the contents",
    "I'll review this",
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

# Canonical phrases for asking for user input (questions/confirmation requests)
ASKING_INPUT_PHRASES = [
    # "Would you like" patterns
    "Would you like me to help you",
    "Would you like me to implement",
    "Would you like me to continue",
    "Would you like me to develop",
    "Would you like me to create",
    "Would you like me to proceed",
    "Would you like me to explore",
    "Would you like me to explain",
    "Would you like more details",
    "Would you like to see",
    # "Do you want" patterns
    "Do you want me to help",
    "Do you want me to implement",
    "Do you want me to continue",
    "Do you want me to proceed",
    "Do you want me to create",
    "Do you want me to develop",
    "Do you want more information",
    # "Should I" patterns
    "Should I continue with",
    "Should I implement",
    "Should I proceed",
    "Should I help you",
    "Should I develop",
    "Should I create",
    # "Shall I" patterns
    "Shall I continue",
    "Shall I implement",
    "Shall I proceed",
    "Shall I help",
    # Question patterns asking for direction
    "What would you like me to focus on",
    "Which component should I start with",
    "How would you like me to proceed",
    "Where should I begin",
    "What area should I explore first",
    # "Let me know" patterns
    "Let me know if you would like",
    "Let me know if you want",
    "Let me know if you need",
    "Please let me know",
    # Offer patterns
    "I can help you with",
    "I can implement this if you",
    "I can proceed with",
]


class IntentClassifier:
    """Semantic intent classifier using embeddings.

    Replaces hardcoded phrase matching with semantic similarity.
    Benefits:
    - Handles variations and paraphrases
    - Reduces false positives
    - More robust to model output variations

    Usage:
        # Use singleton to avoid duplicate initialization
        classifier = IntentClassifier.get_instance()
        await classifier.initialize()

        result = await classifier.classify_intent(
            "Let me start by examining the orchestrator.py file"
        )
        print(result.intent)  # IntentType.CONTINUATION
        print(result.confidence)  # 0.85
    """

    _instance: Optional["IntentClassifier"] = None
    _lock = __import__("threading").Lock()

    @classmethod
    def get_instance(
        cls,
        cache_dir: Optional[Path] = None,
        embedding_service: Optional[EmbeddingService] = None,
        continuation_threshold: float = 0.30,
        completion_threshold: float = 0.30,
        asking_input_threshold: float = 0.35,
    ) -> "IntentClassifier":
        """Get or create the singleton IntentClassifier instance.

        Args:
            cache_dir: Directory for cache files (only used on first call)
            embedding_service: Shared embedding service (only used on first call)
            continuation_threshold: Min similarity for continuation (only used on first call)
            completion_threshold: Min similarity for completion (only used on first call)
            asking_input_threshold: Min similarity for asking input (only used on first call)

        Returns:
            The singleton IntentClassifier instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(
                        cache_dir=cache_dir,
                        embedding_service=embedding_service,
                        continuation_threshold=continuation_threshold,
                        completion_threshold=completion_threshold,
                        asking_input_threshold=asking_input_threshold,
                    )
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (mainly for testing)."""
        with cls._lock:
            cls._instance = None
            logger.debug("Reset IntentClassifier singleton")

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        embedding_service: Optional[EmbeddingService] = None,
        continuation_threshold: float = 0.30,
        completion_threshold: float = 0.30,
        asking_input_threshold: float = 0.35,
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
            asking_input_threshold: Minimum similarity for asking input intent (default 0.35)
                Slightly higher threshold to avoid false positives on questions.
        """
        from victor.config.settings import get_project_paths

        self.cache_dir = cache_dir or get_project_paths().global_embeddings_dir
        self.embedding_service = embedding_service or EmbeddingService.get_instance()
        self.continuation_threshold = continuation_threshold
        self.completion_threshold = completion_threshold
        self.asking_input_threshold = asking_input_threshold

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
        self._asking_input_collection = StaticEmbeddingCollection(
            name="intent_asking_input",
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

        # Build asking_input collection
        asking_input_items = [
            CollectionItem(
                id=f"ask_{i}",
                text=phrase,
                metadata={"intent": "asking_input"},
            )
            for i, phrase in enumerate(ASKING_INPUT_PHRASES)
        ]
        await self._asking_input_collection.initialize(asking_input_items)

        self._initialized = True
        logger.info(
            f"IntentClassifier initialized with {len(CONTINUATION_PHRASES)} continuation, "
            f"{len(COMPLETION_PHRASES)} completion, and {len(ASKING_INPUT_PHRASES)} asking_input phrases"
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

        # Build asking_input collection
        asking_input_items = [
            CollectionItem(
                id=f"ask_{i}",
                text=phrase,
                metadata={"intent": "asking_input"},
            )
            for i, phrase in enumerate(ASKING_INPUT_PHRASES)
        ]
        self._asking_input_collection.initialize_sync(asking_input_items)

        self._initialized = True
        logger.info(
            f"IntentClassifier initialized with {len(CONTINUATION_PHRASES)} continuation, "
            f"{len(COMPLETION_PHRASES)} completion, and {len(ASKING_INPUT_PHRASES)} asking_input phrases"
        )

    async def classify_intent(
        self,
        text: str,
        top_k: int = 3,
    ) -> IntentResult:
        """Classify the intent of text.

        Uses a hybrid approach:
        1. Semantic similarity matching against canonical phrases
        2. Heuristic pattern matching as override for unambiguous cases

        Args:
            text: Text to classify (usually model's response start)
            top_k: Number of top matches to return

        Returns:
            IntentResult with classified intent, confidence, and top matches
        """
        if not self._initialized:
            await self.initialize()

        # Search all collections
        continuation_results = await self._continuation_collection.search(text, top_k=top_k)
        completion_results = await self._completion_collection.search(text, top_k=top_k)
        asking_input_results = await self._asking_input_collection.search(text, top_k=top_k)

        # Get best scores
        best_continuation = continuation_results[0][1] if continuation_results else 0.0
        best_completion = completion_results[0][1] if completion_results else 0.0
        best_asking_input = asking_input_results[0][1] if asking_input_results else 0.0

        # Build top matches for debugging
        top_matches = []
        for item, score in continuation_results[:2]:
            top_matches.append((f"cont:{item.text[:50]}", score))
        for item, score in completion_results[:2]:
            top_matches.append((f"comp:{item.text[:50]}", score))
        for item, score in asking_input_results[:2]:
            top_matches.append((f"ask:{item.text[:50]}", score))

        # HEURISTIC OVERRIDE 0: Check for stuck loop patterns FIRST
        # If model keeps saying what it will do with multiple planning statements,
        # it's likely stuck and not actually making progress
        if _has_stuck_loop_heuristic(text):
            top_matches.append(("heuristic:stuck_loop_pattern", 0.85))
            logger.debug(
                "STUCK_LOOP detected via heuristic override (conf=0.85)"
            )
            return IntentResult(
                intent=IntentType.STUCK_LOOP,
                confidence=0.85,
                top_matches=top_matches,
            )

        # HEURISTIC OVERRIDE 1: Check for explicit continuation patterns
        # Patterns like "let me read", "let me check" are unambiguous
        if _has_continuation_heuristic(text):
            heuristic_confidence = max(best_continuation, 0.75)
            top_matches.append(("heuristic:continuation_pattern", heuristic_confidence))
            logger.debug(
                f"CONTINUATION detected via heuristic override (conf={heuristic_confidence:.2f})"
            )
            return IntentResult(
                intent=IntentType.CONTINUATION,
                confidence=heuristic_confidence,
                top_matches=top_matches,
            )

        # HEURISTIC OVERRIDE 2: Check for "task done + optional elaboration" patterns
        # These indicate the model has FINISHED and is just offering to explain more
        # Must check BEFORE asking_input since patterns overlap (both use "Would you like")
        if _has_completion_offer_heuristic(text):
            heuristic_confidence = max(best_completion, 0.85)
            top_matches.append(("heuristic:completion_offer_pattern", heuristic_confidence))
            logger.debug(
                f"COMPLETION detected via elaboration-offer heuristic (conf={heuristic_confidence:.2f})"
            )
            return IntentResult(
                intent=IntentType.COMPLETION,
                confidence=heuristic_confidence,
                top_matches=top_matches,
            )

        # HEURISTIC OVERRIDE 3: Check for explicit asking-for-input patterns
        # These patterns are unambiguous, so they override semantic classification
        if _has_asking_input_heuristic(text):
            # Use higher confidence when heuristic matches
            heuristic_confidence = max(best_asking_input, 0.75)
            top_matches.append(("heuristic:asking_input_pattern", heuristic_confidence))
            logger.debug(
                f"ASKING_INPUT detected via heuristic override (conf={heuristic_confidence:.2f})"
            )
            return IntentResult(
                intent=IntentType.ASKING_INPUT,
                confidence=heuristic_confidence,
                top_matches=top_matches,
            )

        # Determine intent based on thresholds and relative scores
        # If scores are too close (within 0.05), it's ambiguous -> NEUTRAL
        TIE_THRESHOLD = 0.05
        max_score = max(best_continuation, best_completion, best_asking_input)
        scores_are_tied = (
            abs(best_continuation - max_score) < TIE_THRESHOLD
            and abs(best_completion - max_score) < TIE_THRESHOLD
        )

        if scores_are_tied:
            # Scores too close to call
            return IntentResult(
                intent=IntentType.NEUTRAL,
                confidence=max_score,
                top_matches=top_matches,
            )

        # Check asking_input first (highest priority if it matches well)
        if (
            best_asking_input >= self.asking_input_threshold
            and best_asking_input >= best_continuation
            and best_asking_input >= best_completion
        ):
            return IntentResult(
                intent=IntentType.ASKING_INPUT,
                confidence=best_asking_input,
                top_matches=top_matches,
            )
        elif (
            best_continuation >= self.continuation_threshold and best_continuation > best_completion
        ):
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
                confidence=max(best_continuation, best_completion, best_asking_input),
                top_matches=top_matches,
            )

    def classify_intent_sync(
        self,
        text: str,
        top_k: int = 3,
    ) -> IntentResult:
        """Classify the intent of text (sync version).

        Uses a hybrid approach:
        1. Semantic similarity matching against canonical phrases
        2. Heuristic pattern matching as override for unambiguous cases

        Args:
            text: Text to classify
            top_k: Number of top matches to return

        Returns:
            IntentResult with classified intent, confidence, and top matches
        """
        if not self._initialized:
            self.initialize_sync()

        # Search all collections
        continuation_results = self._continuation_collection.search_sync(text, top_k=top_k)
        completion_results = self._completion_collection.search_sync(text, top_k=top_k)
        asking_input_results = self._asking_input_collection.search_sync(text, top_k=top_k)

        # Get best scores
        best_continuation = continuation_results[0][1] if continuation_results else 0.0
        best_completion = completion_results[0][1] if completion_results else 0.0
        best_asking_input = asking_input_results[0][1] if asking_input_results else 0.0

        # Build top matches for debugging
        top_matches = []
        for item, score in continuation_results[:2]:
            top_matches.append((f"cont:{item.text[:50]}", score))
        for item, score in completion_results[:2]:
            top_matches.append((f"comp:{item.text[:50]}", score))
        for item, score in asking_input_results[:2]:
            top_matches.append((f"ask:{item.text[:50]}", score))

        # HEURISTIC OVERRIDE 0: Check for stuck loop patterns FIRST
        # If model keeps saying what it will do with multiple planning statements,
        # it's likely stuck and not actually making progress
        if _has_stuck_loop_heuristic(text):
            top_matches.append(("heuristic:stuck_loop_pattern", 0.85))
            logger.debug(
                "STUCK_LOOP detected via heuristic override (conf=0.85)"
            )
            return IntentResult(
                intent=IntentType.STUCK_LOOP,
                confidence=0.85,
                top_matches=top_matches,
            )

        # HEURISTIC OVERRIDE 1: Check for explicit continuation patterns
        # Patterns like "let me read", "let me check" are unambiguous
        if _has_continuation_heuristic(text):
            heuristic_confidence = max(best_continuation, 0.75)
            top_matches.append(("heuristic:continuation_pattern", heuristic_confidence))
            logger.debug(
                f"CONTINUATION detected via heuristic override (conf={heuristic_confidence:.2f})"
            )
            return IntentResult(
                intent=IntentType.CONTINUATION,
                confidence=heuristic_confidence,
                top_matches=top_matches,
            )

        # HEURISTIC OVERRIDE 2: Check for "task done + optional elaboration" patterns
        # These indicate the model has FINISHED and is just offering to explain more
        # Must check BEFORE asking_input since patterns overlap (both use "Would you like")
        if _has_completion_offer_heuristic(text):
            heuristic_confidence = max(best_completion, 0.85)
            top_matches.append(("heuristic:completion_offer_pattern", heuristic_confidence))
            logger.debug(
                f"COMPLETION detected via elaboration-offer heuristic (conf={heuristic_confidence:.2f})"
            )
            return IntentResult(
                intent=IntentType.COMPLETION,
                confidence=heuristic_confidence,
                top_matches=top_matches,
            )

        # HEURISTIC OVERRIDE 3: Check for explicit asking-for-input patterns
        # These patterns are unambiguous, so they override semantic classification
        if _has_asking_input_heuristic(text):
            # Use higher confidence when heuristic matches
            heuristic_confidence = max(best_asking_input, 0.75)
            top_matches.append(("heuristic:asking_input_pattern", heuristic_confidence))
            logger.debug(
                f"ASKING_INPUT detected via heuristic override (conf={heuristic_confidence:.2f})"
            )
            return IntentResult(
                intent=IntentType.ASKING_INPUT,
                confidence=heuristic_confidence,
                top_matches=top_matches,
            )

        # Determine intent based on thresholds and relative scores
        # If scores are too close (within 0.05), it's ambiguous -> NEUTRAL
        TIE_THRESHOLD = 0.05
        max_score = max(best_continuation, best_completion, best_asking_input)
        scores_are_tied = (
            abs(best_continuation - max_score) < TIE_THRESHOLD
            and abs(best_completion - max_score) < TIE_THRESHOLD
        )

        if scores_are_tied:
            # Scores too close to call
            return IntentResult(
                intent=IntentType.NEUTRAL,
                confidence=max_score,
                top_matches=top_matches,
            )

        # Check asking_input first (highest priority if it matches well)
        if (
            best_asking_input >= self.asking_input_threshold
            and best_asking_input >= best_continuation
            and best_asking_input >= best_completion
        ):
            return IntentResult(
                intent=IntentType.ASKING_INPUT,
                confidence=best_asking_input,
                top_matches=top_matches,
            )
        elif (
            best_continuation >= self.continuation_threshold and best_continuation > best_completion
        ):
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
                confidence=max(best_continuation, best_completion, best_asking_input),
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

    def is_asking_for_input(self, text: str) -> bool:
        """Quick check if text indicates the model is asking for user input.

        This detects patterns like "Would you like me to...", "Should I...", etc.
        Useful for auto-continuing in one-shot mode.

        Args:
            text: Text to check

        Returns:
            True if text indicates asking for user input
        """
        result = self.classify_intent_sync(text)
        return result.intent == IntentType.ASKING_INPUT

    def clear_cache(self) -> None:
        """Clear cached collections."""
        self._continuation_collection.clear_cache()
        self._completion_collection.clear_cache()
        self._asking_input_collection.clear_cache()
        self._initialized = False
        logger.info("IntentClassifier cache cleared")
