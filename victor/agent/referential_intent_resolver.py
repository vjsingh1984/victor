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

"""Referential intent resolution for anaphoric user messages.

Detects vague referential messages ("do it", "apply the changes", "update as
discussed") and enriches them with recent actionable context from the session
ledger so the LLM has concrete items to act on.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from victor.config.orchestrator_constants import (
    ReferentialIntentConfig,
    REFERENTIAL_INTENT_CONFIG,
)

logger = logging.getLogger(__name__)

# Default referential patterns — word-boundary anchored to reduce false positives
DEFAULT_REFERENTIAL_PATTERNS = [
    re.compile(r"\bdo\s+it\b", re.IGNORECASE),
    re.compile(r"\bdo\s+that\b", re.IGNORECASE),
    re.compile(
        r"\bapply\s+(?:the\s+)?(?:changes|recommendations|fixes|updates)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bupdate\s+(?:as\s+)?(?:discussed|recommended|suggested|per)\b", re.IGNORECASE),
    re.compile(r"\bas\s+(?:discussed|recommended|suggested|mentioned)\b", re.IGNORECASE),
    re.compile(r"\bgo\s+ahead\b", re.IGNORECASE),
    re.compile(r"\bmake\s+(?:the\s+)?(?:changes|updates|fixes)\b", re.IGNORECASE),
    re.compile(r"\bproceed\b", re.IGNORECASE),
    re.compile(r"\byes,?\s*(?:do|please|go)\b", re.IGNORECASE),
    re.compile(r"\bimplement\s+(?:it|that|them|those)\b", re.IGNORECASE),
    re.compile(r"\bfix\s+(?:it|that|them|those)\b", re.IGNORECASE),
    re.compile(r"\bupdate\s+(?:it|that|them|those)\b", re.IGNORECASE),
]


class ReferentialIntentResolver:
    """Detects and enriches referential user messages.

    When a user says "do it" or "apply the changes", this resolver appends
    concrete context from the session ledger so the LLM knows what to do.
    Returns the message unchanged if not referential or no ledger available.
    """

    def __init__(
        self,
        config: Optional[ReferentialIntentConfig] = None,
        session_ledger: Optional[object] = None,
        patterns: Optional[List[re.Pattern]] = None,
    ):
        self._config = config or REFERENTIAL_INTENT_CONFIG
        self._session_ledger = session_ledger
        self._patterns = patterns or DEFAULT_REFERENTIAL_PATTERNS

    @property
    def config(self) -> ReferentialIntentConfig:
        return self._config

    def is_referential(self, message: str) -> bool:
        if not self._config.enabled:
            return False
        if not message or len(message.strip()) > 200:
            # Long messages are unlikely to be purely referential
            return False
        for pattern in self._patterns:
            if pattern.search(message):
                return True
        return False

    def enrich(self, message: str) -> str:
        if not self._config.enabled:
            return message
        if not self.is_referential(message):
            return message
        if self._session_ledger is None:
            return message

        get_items = getattr(self._session_ledger, "get_recent_actionable_items", None)
        if not get_items:
            return message

        items = get_items(limit=5)
        if not items:
            return message

        # Build context block
        context_lines = ["[Context: The user is referring to these recent items:"]
        for item in items:
            summary = getattr(item, "summary", str(item))
            category = getattr(item, "category", "item")
            context_lines.append(f"  - [{category}] {summary}")
        context_lines.append("]")

        context_block = "\n".join(context_lines)
        # Truncate if over budget
        if len(context_block) > self._config.max_enrichment_chars:
            context_block = context_block[: self._config.max_enrichment_chars - 3] + "...]"

        return f"{message}\n\n{context_block}"
