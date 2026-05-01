"""Unified section registry for prompt construction.

Provides single source of truth for all prompt sections across
legacy and framework builders, with consistent naming and metadata.

Research basis:
- arXiv:2601.06007 — System-prompt-only caching is optimal (41-80% cost reduction)
- arXiv:2404.13208 — Safety/guardrails belong in system prompt (instruction hierarchy)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Set

logger = logging.getLogger(__name__)


class SectionCategory(str, Enum):
    """Categories for organizing prompt sections.

    Used for edge model selection and content routing.
    """

    GROUNDING = "grounding"
    TOOL_GUIDANCE = "tool_guidance"
    COMPLETION = "completion"
    TASK_HINTS = "task_hints"
    FEW_SHOT = "few_shot"
    SYNTHESIS = "synthesis"
    CONTEXT = "context"


@dataclass(frozen=True)
class SectionDefinition:
    """Definition of a prompt section with all metadata.

    Attributes:
        name: Canonical uppercase name (e.g., "ASI_TOOL_EFFECTIVENESS_GUIDANCE")
        aliases: Alternative names for backward compatibility (e.g., {"tool_effectiveness_guidance"})
        category: Section category for organization
        default_text: Static fallback text
        evolvable: Whether GEPA/MIPROv2 can evolve this section
        required: Must always be included regardless of tier/edge selection
        priority: For ordering in prompt (lower = earlier)
    """

    name: str
    aliases: Set[str]
    category: SectionCategory
    default_text: str
    evolvable: bool
    required: bool
    priority: int

    def resolve_name(self, name: str) -> bool:
        """Check if a name refers to this section.

        Args:
            name: Name to check (case-insensitive for canonical, exact for aliases)

        Returns:
            True if name matches this section
        """
        return name.upper() == self.name or name in self.aliases


class UnifiedSectionRegistry:
    """Registry of all prompt sections with consistent naming.

    Provides single source of truth for section definitions,
    supporting both uppercase canonical names and lowercase aliases
    for backward compatibility.

    Singleton pattern — use get_section_registry() to access.
    """

    def __init__(self) -> None:
        self._sections: Dict[str, SectionDefinition] = {}

    def register(self, section: SectionDefinition) -> None:
        """Register a section definition.

        Args:
            section: Section definition to register
        """
        self._sections[section.name] = section
        logger.debug(
            f"Registered prompt section: {section.name} with {len(section.aliases)} aliases"
        )

    def get(self, name: str) -> Optional[SectionDefinition]:
        """Get section by name or alias.

        Args:
            name: Section name (canonical or alias)

        Returns:
            SectionDefinition if found, None otherwise
        """
        # Try exact match first (canonical uppercase)
        section = self._sections.get(name.upper())
        if section:
            return section

        # Try aliases
        for section in self._sections.values():
            if section.resolve_name(name):
                return section

        return None

    def get_all(self) -> list[SectionDefinition]:
        """Get all registered sections.

        Returns:
            List of all section definitions
        """
        return list(self._sections.values())

    def get_evolvable_sections(self) -> Set[str]:
        """Get all evolvable section names.

        Returns:
            Set of canonical section names that can be evolved
        """
        return {name for name, section in self._sections.items() if section.evolvable}

    def get_by_category(self, category: SectionCategory) -> list[SectionDefinition]:
        """Get all sections in a category.

        Args:
            category: Section category

        Returns:
            List of sections in the category
        """
        return [section for section in self._sections.values() if section.category == category]


# Singleton instance
_registry: Optional[UnifiedSectionRegistry] = None


def get_section_registry() -> UnifiedSectionRegistry:
    """Get the singleton section registry.

    Initializes default sections on first call.

    Returns:
        UnifiedSectionRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = UnifiedSectionRegistry()
        _initialize_default_sections(_registry)
    return _registry


def _initialize_default_sections(registry: UnifiedSectionRegistry) -> None:
    """Initialize default prompt sections.

    Args:
        registry: Registry to populate with default sections
    """
    from victor.agent.prompt_builder import (
        ASI_TOOL_EFFECTIVENESS_GUIDANCE,
        GROUNDING_RULES,
        COMPLETION_GUIDANCE,
    )

    default_sections = [
        SectionDefinition(
            name="ASI_TOOL_EFFECTIVENESS_GUIDANCE",
            aliases={"tool_effectiveness_guidance", "tool_hints", "asi_tool_guidance"},
            category=SectionCategory.TOOL_GUIDANCE,
            default_text=ASI_TOOL_EFFECTIVENESS_GUIDANCE,
            evolvable=True,
            required=True,
            priority=50,
        ),
        SectionDefinition(
            name="GROUNDING_RULES",
            aliases={"grounding", "grounding_rules", "safety_rules"},
            category=SectionCategory.GROUNDING,
            default_text=GROUNDING_RULES,
            evolvable=True,
            required=True,
            priority=80,
        ),
        SectionDefinition(
            name="COMPLETION_GUIDANCE",
            aliases={"completion_guidance", "completion", "task_completion"},
            category=SectionCategory.COMPLETION,
            default_text=COMPLETION_GUIDANCE,
            evolvable=True,
            required=True,
            priority=60,
        ),
    ]

    for section in default_sections:
        registry.register(section)

    logger.info(f"Initialized {len(default_sections)} default prompt sections")
