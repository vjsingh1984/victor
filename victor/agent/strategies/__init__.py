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

<<<<<<<< HEAD:victor/agent/strategies/__init__.py
"""Strategy implementations for extensibility.

This package provides strategy pattern implementations for provider
classification and other extensibility concerns, following the
Open/Closed Principle (OCP).
"""

from victor.agent.strategies.provider_strategies import (
    DefaultProviderClassificationStrategy,
    ConfigurableProviderClassificationStrategy,
)

__all__ = [
    "DefaultProviderClassificationStrategy",
    "ConfigurableProviderClassificationStrategy",
]
========
"""Core search types shared across verticals and storage layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class SearchHit:
    """Canonical search hit representation for cross-layer interchange."""

    file_path: str
    content: str
    score: float
    line_number: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


__all__ = ["SearchHit"]
>>>>>>>> origin/develop:victor/core/search_types.py
