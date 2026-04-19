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

"""Content registry for prompt composition.

Classifies all prompt content items by lifecycle category (STATIC, SEMI_STATIC,
DYNAMIC, EPHEMERAL) so the PromptComposer can route each item to the correct
placement (system prompt, user prefix, or omitted) based on provider tier.

Research basis:
- arXiv:2601.06007 — System-prompt-only caching is optimal (41-80% cost reduction)
- arXiv:2404.13208 — Safety/guardrails belong in system prompt (instruction hierarchy)
- arXiv:2410.14826 — System prompt optimization yields ~10% gains (SPRIG)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class ContentCategory(Enum):
    """Lifecycle category for prompt content."""

    STATIC = "static"  # Never changes within a session
    SEMI_STATIC = "semi_static"  # Changes per workspace/task but not per turn
    DYNAMIC = "dynamic"  # Can change per turn (optimization outputs)
    EPHEMERAL = "ephemeral"  # Single-turn only (continuations, pivots)


@dataclass
class ContentItem:
    """A registered piece of prompt content with classification metadata.

    Attributes:
        name: Unique identifier (e.g., "GROUNDING_RULES")
        category: Lifecycle category
        default_text: Static fallback text (empty for dynamic-only items)
        token_estimate: Approximate token count (~chars/4)
        evolvable: Whether GEPA/MIPROv2 can evolve this section
        required: Must always be included regardless of tier/edge selection
        section_group: Logical grouping for edge model selection
    """

    name: str
    category: ContentCategory
    default_text: str = ""
    token_estimate: int = 0
    evolvable: bool = False
    required: bool = False
    section_group: str = "general"


class ContentRegistry:
    """Registry of all prompt content items.

    Provides a single source of truth for what content exists, how it's
    classified, and its metadata. Used by ContentRouter and PromptComposer
    to make placement decisions.
    """

    def __init__(self) -> None:
        self._items: Dict[str, ContentItem] = {}

    def register(self, item: ContentItem) -> None:
        """Register a content item."""
        self._items[item.name] = item

    def get(self, name: str) -> Optional[ContentItem]:
        """Get a content item by name."""
        return self._items.get(name)

    def get_all(self) -> List[ContentItem]:
        """Get all registered items."""
        return list(self._items.values())

    def get_by_category(self, category: ContentCategory) -> List[ContentItem]:
        """Get all items in a category."""
        return [i for i in self._items.values() if i.category == category]

    def get_by_group(self, group: str) -> List[ContentItem]:
        """Get all items in a section group."""
        return [i for i in self._items.values() if i.section_group == group]

    def get_evolvable(self) -> List[ContentItem]:
        """Get all items that can be evolved by GEPA/MIPROv2."""
        return [i for i in self._items.values() if i.evolvable]


def create_default_registry() -> ContentRegistry:
    """Create the default content registry with all standard prompt sections.

    Imports static constants from prompt_builder to avoid duplication.
    Token estimates are approximate (~chars/4).
    """
    from victor.agent.prompt_builder import (
        ASI_TOOL_EFFECTIVENESS_GUIDANCE,
        COMPLETION_GUIDANCE,
        CONCISE_MODE_GUIDANCE,
        GROUNDING_RULES,
        GROUNDING_RULES_EXTENDED,
        LARGE_FILE_PAGINATION_GUIDANCE,
        PARALLEL_READ_GUIDANCE,
    )

    registry = ContentRegistry()

    # --- STATIC: Never changes within a session ---

    registry.register(
        ContentItem(
            name="GROUNDING_RULES",
            category=ContentCategory.STATIC,
            default_text=GROUNDING_RULES,
            token_estimate=40,
            evolvable=True,
            required=True,
            section_group="grounding",
        )
    )

    registry.register(
        ContentItem(
            name="GROUNDING_RULES_EXTENDED",
            category=ContentCategory.STATIC,
            default_text=GROUNDING_RULES_EXTENDED,
            token_estimate=140,
            evolvable=False,
            required=False,
            section_group="grounding",
        )
    )

    registry.register(
        ContentItem(
            name="COMPLETION_GUIDANCE",
            category=ContentCategory.STATIC,
            default_text=COMPLETION_GUIDANCE,
            token_estimate=218,
            evolvable=True,
            required=True,
            section_group="completion",
        )
    )

    registry.register(
        ContentItem(
            name="ASI_TOOL_EFFECTIVENESS_GUIDANCE",
            category=ContentCategory.STATIC,
            default_text=ASI_TOOL_EFFECTIVENESS_GUIDANCE,
            token_estimate=280,
            evolvable=True,
            required=True,
            section_group="tool_guidance",
        )
    )

    registry.register(
        ContentItem(
            name="CONCISE_MODE_GUIDANCE",
            category=ContentCategory.STATIC,
            default_text=CONCISE_MODE_GUIDANCE,
            token_estimate=91,
            evolvable=False,
            required=False,
            section_group="concise_mode",
        )
    )

    registry.register(
        ContentItem(
            name="PARALLEL_READ_GUIDANCE",
            category=ContentCategory.STATIC,
            default_text=PARALLEL_READ_GUIDANCE,
            token_estimate=89,
            evolvable=False,
            required=False,
            section_group="tool_guidance",
        )
    )

    registry.register(
        ContentItem(
            name="LARGE_FILE_PAGINATION_GUIDANCE",
            category=ContentCategory.STATIC,
            default_text=LARGE_FILE_PAGINATION_GUIDANCE,
            token_estimate=212,
            evolvable=False,
            required=False,
            section_group="tool_guidance",
        )
    )

    # --- SEMI-STATIC: Changes per workspace/task ---

    registry.register(
        ContentItem(
            name="PROJECT_CONTEXT",
            category=ContentCategory.SEMI_STATIC,
            token_estimate=500,  # Variable, 200-2000
            evolvable=False,
            required=False,
            section_group="context",
        )
    )

    registry.register(
        ContentItem(
            name="TASK_DESCRIPTION",
            category=ContentCategory.SEMI_STATIC,
            token_estimate=200,  # Variable
            evolvable=False,
            required=False,
            section_group="context",
        )
    )

    registry.register(
        ContentItem(
            name="VERTICAL_PROMPT",
            category=ContentCategory.SEMI_STATIC,
            token_estimate=150,  # Variable
            evolvable=False,
            required=False,
            section_group="vertical",
        )
    )

    # --- DYNAMIC: Can change per turn (optimization outputs) ---

    registry.register(
        ContentItem(
            name="GEPA_EVOLVED_ASI_TOOL",
            category=ContentCategory.DYNAMIC,
            token_estimate=300,
            evolvable=True,
            required=False,
            section_group="tool_guidance",
        )
    )

    registry.register(
        ContentItem(
            name="GEPA_EVOLVED_GROUNDING",
            category=ContentCategory.DYNAMIC,
            token_estimate=100,
            evolvable=True,
            required=False,
            section_group="grounding",
        )
    )

    registry.register(
        ContentItem(
            name="GEPA_EVOLVED_COMPLETION",
            category=ContentCategory.DYNAMIC,
            token_estimate=200,
            evolvable=True,
            required=False,
            section_group="completion",
        )
    )

    registry.register(
        ContentItem(
            name="MIPROV2_FEW_SHOTS",
            category=ContentCategory.DYNAMIC,
            token_estimate=350,
            evolvable=False,
            required=False,
            section_group="few_shot",
        )
    )

    registry.register(
        ContentItem(
            name="FAILURE_HINTS",
            category=ContentCategory.DYNAMIC,
            token_estimate=100,
            evolvable=False,
            required=False,
            section_group="tool_guidance",
        )
    )

    registry.register(
        ContentItem(
            name="CONTEXT_REMINDERS",
            category=ContentCategory.DYNAMIC,
            token_estimate=80,
            evolvable=False,
            required=False,
            section_group="reminders",
        )
    )

    registry.register(
        ContentItem(
            name="ACTIVE_SKILL",
            category=ContentCategory.DYNAMIC,
            token_estimate=200,
            evolvable=False,
            required=False,
            section_group="skills",
        )
    )

    # --- EPHEMERAL: Single-turn only ---

    registry.register(
        ContentItem(
            name="APPROACH_PIVOT",
            category=ContentCategory.EPHEMERAL,
            token_estimate=60,
            evolvable=False,
            required=False,
            section_group="ephemeral",
        )
    )

    registry.register(
        ContentItem(
            name="COMPACTION_NOTICE",
            category=ContentCategory.EPHEMERAL,
            token_estimate=80,
            evolvable=False,
            required=False,
            section_group="ephemeral",
        )
    )

    return registry
