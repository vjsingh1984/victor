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

"""Continuation service implementation.

Handles continuation prompt generation, continuation decision logic,
and multi-turn coordination.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from victor.providers.base import CompletionResponse

logger = logging.getLogger(__name__)


class ContinuationServiceConfig:
    """Configuration for ContinuationService.

    Attributes:
        max_continuation_prompts: Maximum continuation prompts per request
        continuation_threshold: Minimum length for continuation (chars)
        enable_smart_continuation: Enable LLM-based continuation decisions
    """

    def __init__(
        self,
        max_continuation_prompts: int = 3,
        continuation_threshold: int = 500,
        enable_smart_continuation: bool = False,
    ):
        self.max_continuation_prompts = max_continuation_prompts
        self.continuation_threshold = continuation_threshold
        self.enable_smart_continuation = enable_smart_continuation


class ContinuationService:
    """Service for continuation prompt management.

    Responsible for:
    - Determining when continuation is needed
    - Generating continuation prompts
    - Managing multi-turn conversation coordination
    - Continuation iteration tracking

    This service does NOT handle:
    - Chat flow control (delegated to ChatFlowService)
    - Streaming (delegated to StreamingService)
    - Response aggregation (delegated to ResponseAggregationService)

    Example:
        config = ContinuationServiceConfig()
        service = ContinuationService(config=config)

        # Check if continuation needed
        if service.needs_continuation(response):
            prompt = await service.create_continuation_prompt(response)
            # Use prompt for next iteration
    """

    def __init__(self, config: ContinuationServiceConfig):
        """Initialize ContinuationService.

        Args:
            config: Service configuration
        """
        self.config = config
        self._continuation_count = 0
        self._total_continuations = 0

    def needs_continuation(self, response: "CompletionResponse") -> bool:
        """Determine if a response needs continuation.

        A response needs continuation if:
        - Stop reason is not "stop" (e.g., "length", "max_tokens")
        - Content is too short (below threshold)
        - Content ends mid-sentence
        - Content is truncated

        Args:
            response: The response to evaluate

        Returns:
            True if continuation is needed, False otherwise
        """
        if not response:
            return False

        # Check stop reason
        if response.stop_reason in ("length", "max_tokens", "token_limit"):
            logger.debug(f"Continuation needed: stop_reason={response.stop_reason}")
            return True

        # Check content length
        if response.content and len(response.content) < self.config.continuation_threshold:
            logger.debug(
                f"Continuation needed: content length {len(response.content)} "
                f"< threshold {self.config.continuation_threshold}"
            )
            return True

        # Check for incomplete sentences
        if self._ends_mid_sentence(response.content):
            logger.debug("Continuation needed: content ends mid-sentence")
            return True

        return False

    async def create_continuation_prompt(
        self, response: "CompletionResponse", context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a continuation prompt for incomplete responses.

        Args:
            response: The incomplete response
            context: Optional context for continuation

        Returns:
            Continuation prompt string
        """
        self._continuation_count += 1
        self._total_continuations += 1

        # Get the last part of the response for context
        last_content = response.content or ""
        context_snippet = last_content[-200:] if len(last_content) > 200 else last_content

        # Create continuation prompt
        prompt = self._build_continuation_prompt(context_snippet, context)

        logger.debug(f"Created continuation prompt #{self._continuation_count}")

        return prompt

    def should_stop_continuation(self, iteration: int, max_iterations: int) -> bool:
        """Determine if continuation should stop.

        Args:
            iteration: Current iteration number
            max_iterations: Maximum allowed iterations

        Returns:
            True if should stop, False to continue
        """
        # Check max continuation prompts
        if self._continuation_count >= self.config.max_continuation_prompts:
            logger.debug(
                f"Stopping continuation: reached max " f"({self.config.max_continuation_prompts})"
            )
            return True

        # Check max iterations
        if iteration >= max_iterations:
            logger.debug(f"Stopping continuation: reached max iterations ({max_iterations})")
            return True

        return False

    def get_continuation_count(self) -> int:
        """Get the number of continuation prompts used.

        Returns:
            Number of continuation prompts
        """
        return self._continuation_count

    def reset_continuation_count(self) -> None:
        """Reset the continuation count for a new request."""
        self._continuation_count = 0

    def get_continuation_metrics(self) -> Dict[str, Any]:
        """Get continuation metrics.

        Returns:
            Dictionary with continuation metrics
        """
        return {
            "current_continuation_count": self._continuation_count,
            "total_continuations": self._total_continuations,
            "max_continuation_prompts": self.config.max_continuation_prompts,
            "continuation_threshold": self.config.continuation_threshold,
        }

    def _build_continuation_prompt(
        self, context_snippet: str, context: Optional[Dict[str, Any]]
    ) -> str:
        """Build a continuation prompt.

        Args:
            context_snippet: Snippet of previous content for context
            context: Optional additional context

        Returns:
            Continuation prompt string
        """
        # Build prompt based on context
        if context and "task" in context:
            task = context["task"]
            prompt = f"Please continue with the task: {task}\n\n"
        else:
            prompt = "Please continue your response.\n\n"

        # Add context snippet if available
        if context_snippet:
            prompt += f"Context: ...{context_snippet}\n\n"

        prompt += "Continue from where you left off."

        return prompt

    def _ends_mid_sentence(self, content: Optional[str]) -> bool:
        """Check if content ends mid-sentence.

        Args:
            content: The content to check

        Returns:
            True if ends mid-sentence, False otherwise
        """
        if not content:
            return False

        content = content.strip()

        # Check for sentence-ending punctuation (must be at the very end)
        if content[-1] in (".", "!", "?"):
            return False

        # Check for common sentence structures
        if any(content.endswith(ending) for ending in (":", ";", "...")):
            return True

        # Check if ends with lowercase letter (likely mid-sentence)
        if content[-1].islower():
            return True

        return False
