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

"""Victor Verticals - Domain-specific assistant templates.

Verticals are pre-configured assistant templates optimized for specific
domains. Each vertical defines:
- Tool sets appropriate for the domain
- Stage definitions and transitions
- System prompts with domain expertise
- Evaluation criteria

Available Verticals:
- CodingAssistant: Software development (current Victor default)
- ResearchAssistant: Web research and document analysis (Perplexity AI competitor)
- DevOpsAssistant: Infrastructure and deployment (Docker Desktop AI competitor)
- DataAnalysisAssistant: Data exploration and ML (ChatGPT Data Analysis competitor)

Architecture:
    Uses the Template Method pattern - VerticalBase defines the skeleton,
    concrete verticals override specific steps.

Example:
    from victor.verticals import CodingAssistant

    # Create agent with vertical configuration
    config = CodingAssistant.get_config()
    agent = await Agent.create(**config)
"""

from typing import List, Optional, Type

from victor.verticals.base import (
    StageDefinition,
    VerticalBase,
    VerticalConfig,
    VerticalRegistry,
)
from victor.verticals.coding import CodingAssistant
from victor.verticals.devops import DevOpsAssistant
from victor.verticals.research import ResearchAssistant
from victor.verticals.data_analysis import DataAnalysisAssistant
from victor.verticals.rag import RAGAssistant

__all__ = [
    # Base classes
    "VerticalBase",
    "VerticalConfig",
    "VerticalRegistry",
    "StageDefinition",
    # Verticals
    "CodingAssistant",
    "DevOpsAssistant",
    "ResearchAssistant",
    "DataAnalysisAssistant",
    "RAGAssistant",
    # Helper functions
    "get_vertical",
    "list_verticals",
]


def get_vertical(name: Optional[str]) -> Optional[Type[VerticalBase]]:
    """Look up a vertical by name.

    Convenience function for CLI usage with case-insensitive matching.

    Args:
        name: Vertical name (case-insensitive), or None.

    Returns:
        Vertical class or None if not found.

    Example:
        vertical = get_vertical("coding")
        if vertical:
            config = vertical.get_config()
    """
    if name is None:
        return None

    # Try exact match first
    result = VerticalRegistry.get(name)
    if result:
        return result

    # Try case-insensitive match
    name_lower = name.lower()
    for registered_name in VerticalRegistry.list_names():
        if registered_name.lower() == name_lower:
            return VerticalRegistry.get(registered_name)

    return None


def list_verticals() -> List[str]:
    """List all available vertical names.

    Returns:
        List of registered vertical names.

    Example:
        print(f"Available verticals: {list_verticals()}")
        # Output: Available verticals: ['coding', 'research', 'devops']
    """
    return VerticalRegistry.list_names()


# Register built-in verticals
VerticalRegistry.register(CodingAssistant)
VerticalRegistry.register(DevOpsAssistant)
VerticalRegistry.register(ResearchAssistant)
VerticalRegistry.register(DataAnalysisAssistant)
VerticalRegistry.register(RAGAssistant)

__version__ = "0.2.0"
