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
from typing import TYPE_CHECKING, Dict, Iterable, Optional, Set

if TYPE_CHECKING:
    from victor.core.verticals.protocols.prompt_provider import (
        PromptSectionContribution,
    )

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
    default_strategies: tuple[str, ...] = ()

    def resolve_name(self, name: str) -> bool:
        """Check if a name refers to this section.

        Args:
            name: Name to check (case-insensitive for canonical, exact for aliases)

        Returns:
            True if name matches this section
        """
        return name.upper() == self.name or name in self.aliases


@dataclass(frozen=True)
class EdgeFocusSection:
    """User-facing prompt-focus selector backed by canonical prompt sections."""

    name: str
    description: str
    section_names: Set[str]


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

    def register_runtime_sections(self, sections: Iterable["PromptSectionContribution"]) -> None:
        """Register contributor-owned prompt sections into the shared registry."""
        for contribution in sections:
            name = str(getattr(contribution, "name", "") or "").strip().upper()
            text = str(getattr(contribution, "text", "") or "")
            if not name or not text.strip():
                continue
            category = _coerce_section_category(getattr(contribution, "category", "context"))
            self.register(
                SectionDefinition(
                    name=name,
                    aliases=set(getattr(contribution, "aliases", set()) or set()),
                    category=category,
                    default_text=text,
                    evolvable=bool(getattr(contribution, "evolvable", False)),
                    required=bool(getattr(contribution, "required", False)),
                    priority=int(getattr(contribution, "priority", 50)),
                    default_strategies=tuple(
                        str(strategy).strip()
                        for strategy in (getattr(contribution, "default_strategies", ()) or ())
                        if str(strategy).strip()
                    ),
                )
            )


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


def register_prompt_contributor_sections(contributors: Iterable[object]) -> None:
    """Register named prompt sections supplied by contributors.

    Legacy contributors that only expose ``get_system_prompt_section()`` are
    normalized into non-evolvable context sections.
    """
    from victor.core.verticals.protocols.prompt_provider import (
        collect_prompt_section_contributions,
    )

    registry = get_section_registry()
    for contributor in contributors:
        registry.register_runtime_sections(collect_prompt_section_contributions(contributor))


def get_required_evolvable_sections() -> list[SectionDefinition]:
    """Return required evolvable sections ordered for prompt construction."""
    registry = get_section_registry()
    sections = [section for section in registry.get_all() if section.required and section.evolvable]
    return sorted(sections, key=lambda section: section.priority)


def get_required_evolvable_section_names() -> list[str]:
    """Return canonical names for required evolvable sections in prompt order."""
    return [section.name for section in get_required_evolvable_sections()]


def get_default_section_strategies() -> dict[str, list[str]]:
    """Return registry-backed default optimization strategies by section name."""
    registry = get_section_registry()
    strategy_map: dict[str, list[str]] = {}
    for section in registry.get_all():
        if section.default_strategies:
            strategy_map[section.name] = list(section.default_strategies)
    return strategy_map


def build_fallback_map(section_names: Iterable[str]) -> dict[str, str]:
    """Return static fallback text for the requested canonical sections."""
    registry = get_section_registry()
    fallback_map: dict[str, str] = {}
    for name in section_names:
        section = registry.get(name)
        if section is None:
            continue
        fallback_map[section.name] = section.default_text
    return fallback_map


def get_edge_focus_sections() -> list[EdgeFocusSection]:
    """Return the canonical edge-model prompt-focus catalog."""
    return [
        EdgeFocusSection(
            name="grounding",
            description="Base rules (always respond from tool output)",
            section_names={"GROUNDING_RULES", "GROUNDING_RULES_EXTENDED"},
        ),
        EdgeFocusSection(
            name="completion",
            description="Task completion signal markers (**DONE**, **SUMMARY**)",
            section_names={"COMPLETION_GUIDANCE"},
        ),
        EdgeFocusSection(
            name="tool_guidance",
            description="How to call tools correctly",
            section_names={"ASI_TOOL_EFFECTIVENESS_GUIDANCE"},
        ),
        EdgeFocusSection(
            name="file_pagination",
            description="Large file reading hints",
            section_names={"LARGE_FILE_PAGINATION_GUIDANCE"},
        ),
        EdgeFocusSection(
            name="concise_mode",
            description="Output brevity directives",
            section_names={"CONCISE_MODE_GUIDANCE"},
        ),
        EdgeFocusSection(
            name="parallel_read",
            description="Batch file reading optimization",
            section_names={"PARALLEL_READ_GUIDANCE"},
        ),
    ]


def build_edge_focus_prompt_options_text() -> str:
    """Render the edge-model prompt-focus catalog as bullet text."""
    return "\n".join(
        f'- "{section.name}": {section.description}' for section in get_edge_focus_sections()
    )


def get_edge_focus_selector_index() -> dict[str, Set[str]]:
    """Map canonical section names to all supported edge-focus selectors."""
    index: dict[str, Set[str]] = {}
    registry = get_section_registry()

    for focus in get_edge_focus_sections():
        for section_name in focus.section_names:
            selectors = index.setdefault(section_name, set())
            selectors.add(focus.name)
            section = registry.get(section_name)
            if section is not None:
                selectors.add(section.name)
                selectors.add(section.name.lower())
                selectors.update(section.aliases)
                selectors.add(section.category.value)
    return index


def _initialize_default_sections(registry: UnifiedSectionRegistry) -> None:
    """Initialize default prompt sections.

    Args:
        registry: Registry to populate with default sections
    """
    from victor.agent.prompt_section_texts import (
        ASI_TOOL_EFFECTIVENESS_GUIDANCE,
        CONCISE_MODE_GUIDANCE,
        COMPLETION_GUIDANCE,
        GROUNDING_RULES,
        GROUNDING_RULES_EXTENDED,
        LARGE_FILE_PAGINATION_GUIDANCE,
        PARALLEL_READ_GUIDANCE,
    )
    from victor.framework.init_synthesizer import SYNTHESIS_RULES

    default_sections = [
        SectionDefinition(
            name="ASI_TOOL_EFFECTIVENESS_GUIDANCE",
            aliases={"tool_effectiveness_guidance", "tool_hints", "asi_tool_guidance"},
            category=SectionCategory.TOOL_GUIDANCE,
            default_text=ASI_TOOL_EFFECTIVENESS_GUIDANCE,
            evolvable=True,
            required=True,
            priority=50,
            default_strategies=("gepa", "cot_distillation"),
        ),
        SectionDefinition(
            name="GROUNDING_RULES",
            aliases={"grounding", "grounding_rules", "safety_rules"},
            category=SectionCategory.GROUNDING,
            default_text=GROUNDING_RULES,
            evolvable=True,
            required=True,
            priority=80,
            default_strategies=("gepa", "prefpo"),
        ),
        SectionDefinition(
            name="COMPLETION_GUIDANCE",
            aliases={"completion_guidance", "completion", "task_completion"},
            category=SectionCategory.COMPLETION,
            default_text=COMPLETION_GUIDANCE,
            evolvable=True,
            required=True,
            priority=60,
            default_strategies=("gepa", "prefpo"),
        ),
        SectionDefinition(
            name="CONCISE_MODE_GUIDANCE",
            aliases={"concise_mode_guidance", "concise_mode", "brevity_guidance"},
            category=SectionCategory.TASK_HINTS,
            default_text=CONCISE_MODE_GUIDANCE,
            evolvable=True,
            required=False,
            priority=30,
            default_strategies=("prefpo",),
        ),
        SectionDefinition(
            name="PARALLEL_READ_GUIDANCE",
            aliases={"parallel_read_guidance", "parallel_reads", "read_parallelism"},
            category=SectionCategory.TOOL_GUIDANCE,
            default_text=PARALLEL_READ_GUIDANCE,
            evolvable=True,
            required=False,
            priority=40,
            default_strategies=("gepa",),
        ),
        SectionDefinition(
            name="LARGE_FILE_PAGINATION_GUIDANCE",
            aliases={
                "large_file_guidance",
                "pagination_guidance",
                "truncation_guidance",
            },
            category=SectionCategory.TOOL_GUIDANCE,
            default_text=LARGE_FILE_PAGINATION_GUIDANCE,
            evolvable=True,
            required=False,
            priority=45,
            default_strategies=("gepa",),
        ),
        SectionDefinition(
            name="GROUNDING_RULES_EXTENDED",
            aliases={
                "grounding_rules_extended",
                "extended_grounding",
                "tool_output_grounding",
            },
            category=SectionCategory.GROUNDING,
            default_text=GROUNDING_RULES_EXTENDED,
            evolvable=True,
            required=False,
            priority=85,
            default_strategies=("gepa",),
        ),
        SectionDefinition(
            name="FEW_SHOT_EXAMPLES",
            aliases={"few_shot_examples", "few_shots", "demonstrations"},
            category=SectionCategory.FEW_SHOT,
            default_text="",
            evolvable=True,
            required=False,
            priority=90,
            default_strategies=("miprov2",),
        ),
        SectionDefinition(
            name="INIT_SYNTHESIS_RULES",
            aliases={"init_synthesis_rules", "synthesis_rules", "init_rules"},
            category=SectionCategory.SYNTHESIS,
            default_text=SYNTHESIS_RULES,
            evolvable=True,
            required=False,
            priority=100,
            default_strategies=("gepa",),
        ),
    ]

    for section in default_sections:
        registry.register(section)

    logger.info(f"Initialized {len(default_sections)} default prompt sections")


def _coerce_section_category(value: object) -> SectionCategory:
    """Map loosely typed contributor categories into the registry enum."""
    normalized = str(value or "").strip().lower()
    for category in SectionCategory:
        if category.value == normalized:
            return category
    return SectionCategory.CONTEXT
