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

"""Plugin marketplace registry for discovery and search.

Provides a local registry of known plugins that can be searched
and installed via ``victor plugin search`` and ``victor plugin install``.

The registry can be extended with remote sources in the future.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Built-in marketplace entries. External developers can contribute by
# adding entries here or by hosting a remote registry.
_MARKETPLACE_ENTRIES: List[Dict[str, str]] = [
    {
        "name": "victor-coding",
        "description": "Coding assistant vertical with LSP, git, refactoring, and testing tools",
        "source": "https://github.com/vjsingh1984/victor-coding.git",
        "source_type": "pip",
        "install_cmd": "pip install victor-coding",
        "category": "development",
    },
    {
        "name": "victor-devops",
        "description": "DevOps vertical with Docker, Kubernetes, Terraform, and CI/CD tools",
        "source": "https://github.com/vjsingh1984/victor-devops.git",
        "source_type": "pip",
        "install_cmd": "pip install victor-devops",
        "category": "infrastructure",
    },
    {
        "name": "victor-research",
        "description": "Research assistant vertical for literature review and analysis",
        "source": "https://github.com/vjsingh1984/victor-research.git",
        "source_type": "pip",
        "install_cmd": "pip install victor-research",
        "category": "research",
    },
    {
        "name": "victor-rag",
        "description": "RAG vertical with vector search, semantic matching, and knowledge graphs",
        "source": "https://github.com/vjsingh1984/victor-rag.git",
        "source_type": "pip",
        "install_cmd": "pip install victor-rag",
        "category": "data",
    },
    {
        "name": "victor-dataanalysis",
        "description": "Data analysis vertical with pandas, statistical analysis, and visualization",
        "source": "https://github.com/vjsingh1984/victor-dataanalysis.git",
        "source_type": "pip",
        "install_cmd": "pip install victor-dataanalysis",
        "category": "data",
    },
    {
        "name": "victor-invest",
        "description": "Investment analysis vertical with SEC fundamentals and technical analysis",
        "source": "https://github.com/vjsingh1984/victor-invest.git",
        "source_type": "pip",
        "install_cmd": "pip install victor-invest",
        "category": "finance",
    },
]


def search_marketplace(
    query: str,
    category: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Search the plugin marketplace registry.

    Args:
        query: Search term (matched against name and description).
        category: Optional category filter.

    Returns:
        List of matching marketplace entries.
    """
    query_lower = query.lower()
    results = []

    for entry in _MARKETPLACE_ENTRIES:
        if category and entry.get("category") != category:
            continue

        name_match = query_lower in entry["name"].lower()
        desc_match = query_lower in entry["description"].lower()
        cat_match = query_lower in entry.get("category", "").lower()

        if name_match or desc_match or cat_match:
            results.append(entry)

    return results


def list_marketplace(
    category: Optional[str] = None,
) -> List[Dict[str, str]]:
    """List all marketplace entries.

    Args:
        category: Optional category filter.

    Returns:
        All marketplace entries, optionally filtered.
    """
    if category:
        return [e for e in _MARKETPLACE_ENTRIES if e.get("category") == category]
    return list(_MARKETPLACE_ENTRIES)


def get_categories() -> List[str]:
    """Return unique categories in the marketplace."""
    return sorted({e.get("category", "") for e in _MARKETPLACE_ENTRIES if e.get("category")})
