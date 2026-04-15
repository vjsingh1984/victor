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

"""Thread-safe token usage tracker.

Centralizes all token usage accumulation to prevent divergence
between TurnExecutor and SessionStateManager.
"""

from __future__ import annotations

import threading
from typing import Dict


class TokenTracker:
    """Thread-safe token usage tracker.

    Centralizes token accumulation from TurnExecutor,
    ChatCoordinator, and MetricsCoordinator into a single
    source of truth.
    """

    _KEYS = (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
    )

    def __init__(self) -> None:
        self._usage: Dict[str, int] = dict.fromkeys(self._KEYS, 0)
        self._lock = threading.Lock()

    def accumulate(self, usage: Dict[str, int]) -> None:
        """Add token counts from a response (thread-safe).

        Args:
            usage: Dictionary with token count keys to add.
        """
        with self._lock:
            for key in self._KEYS:
                self._usage[key] += usage.get(key, 0)

    def get_usage(self) -> Dict[str, int]:
        """Return a copy of current usage (thread-safe)."""
        with self._lock:
            return dict(self._usage)

    def reset(self) -> None:
        """Reset all counters to zero (thread-safe)."""
        with self._lock:
            self._usage = dict.fromkeys(self._KEYS, 0)

    @property
    def total_tokens(self) -> int:
        """Return total token count."""
        with self._lock:
            return self._usage["total_tokens"]

    @property
    def prompt_tokens(self) -> int:
        """Return prompt token count."""
        with self._lock:
            return self._usage["prompt_tokens"]

    @property
    def completion_tokens(self) -> int:
        """Return completion token count."""
        with self._lock:
            return self._usage["completion_tokens"]


__all__ = [
    "TokenTracker",
]
