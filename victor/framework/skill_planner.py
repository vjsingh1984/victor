"""Framework-level skill planning helpers.

This module keeps skill-aware planning logic in a shared framework surface so
chat/planning entry points can reuse the same decomposition and prompt-building
behavior instead of carrying local variants.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Mapping, Optional

from victor_sdk.skills import SkillDefinition

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SkillDecomposition:
    """Ordered skill sequence derived from the shared matcher."""

    skills: List[str]
    confidence: float
    rationale: str


def coerce_skill_catalog(skills: Any) -> Dict[str, SkillDefinition]:
    """Normalize skill inputs into a name -> definition mapping."""
    if not skills:
        return {}

    if isinstance(skills, Mapping):
        return {
            str(name): skill for name, skill in skills.items() if isinstance(skill, SkillDefinition)
        }

    if isinstance(skills, list):
        return {skill.name: skill for skill in skills if isinstance(skill, SkillDefinition)}

    return {}


def build_skill_decomposition(
    user_request: str,
    matcher: Any,
    *,
    max_skills: int = 3,
) -> Optional[SkillDecomposition]:
    """Build an ordered skill sequence from the shared SkillMatcher.

    The matcher already owns the ranking and phase ordering logic. This helper
    turns those matches into a reusable planning/decomposition artifact.
    """
    if matcher is None or not getattr(
        matcher, "initialized", getattr(matcher, "_initialized", False)
    ):
        return None

    match_multiple = getattr(matcher, "match_multiple_sync", None)
    if not callable(match_multiple):
        return None

    matches = match_multiple(user_request, max_skills=max_skills)
    if not matches:
        return None

    ordered_skills = [skill.name for skill, _score in matches]
    mean_confidence = sum(score for _skill, score in matches) / max(len(matches), 1)
    rationale = (
        "Ordered by framework skill matcher phase sequence"
        if len(ordered_skills) > 1
        else "Matched primary framework skill"
    )
    return SkillDecomposition(
        skills=ordered_skills,
        confidence=round(mean_confidence, 3),
        rationale=rationale,
    )


def enrich_plan_with_skills(
    steps: List[List],
    skills: Dict[str, SkillDefinition],
) -> List[Dict[str, Any]]:
    """Map plan steps to matching skills.

    For each step, tries to match a skill by:
    1. Step type matching against skill tags (e.g., "bugfix" matches tag "bugfix")
    2. Step description keyword matching against skill tags and name

    Args:
        steps: Plan steps as [[id, type, description, tools, ...], ...]
        skills: Available skills keyed by name

    Returns:
        List of enriched step dicts with matched_skill field.
    """
    if not steps:
        return []

    result = []
    for step_data in steps:
        step_id = step_data[0] if len(step_data) > 0 else 0
        step_type = str(step_data[1]).lower() if len(step_data) > 1 else ""
        step_desc = str(step_data[2]) if len(step_data) > 2 else ""
        step_tools = str(step_data[3]) if len(step_data) > 3 else ""

        matched = _match_skill_for_step(step_type, step_desc, skills)

        result.append(
            {
                "id": step_id,
                "type": step_type,
                "description": step_desc,
                "tools": step_tools,
                "matched_skill": matched.name if matched else None,
                "skill_prompt": matched.prompt_fragment if matched else None,
            }
        )

    return result


def _match_skill_for_step(
    step_type: str,
    step_desc: str,
    skills: Dict[str, SkillDefinition],
) -> Optional[SkillDefinition]:
    """Find the best matching skill for a plan step.

    Matching priority:
    1. Step type matches a skill tag exactly
    2. Step description contains a skill tag keyword
    3. Step type/description matches skill name
    """
    if not skills:
        return None

    step_type_lower = step_type.lower()
    desc_lower = step_desc.lower()
    desc_words = set(desc_lower.split())

    best_match = None
    best_score = 0

    for skill in skills.values():
        score = 0
        all_keywords = set(skill.tags) | {skill.name.lower()}

        # Type matches a tag
        if step_type_lower in all_keywords:
            score += 3

        # Tags appear in description
        for tag in all_keywords:
            if tag in desc_lower:
                score += 2
            if tag in desc_words:
                score += 1

        if score > best_score:
            best_score = score
            best_match = skill

    return best_match if best_score >= 2 else None


def build_skill_aware_plan_prompt(
    user_request: str,
    skills: Dict[str, SkillDefinition],
    *,
    selected_skills: Optional[List[str]] = None,
    decomposition_confidence: Optional[float] = None,
) -> str:
    """Build a plan generation prompt that includes available skills.

    The LLM sees the skill catalog and can reference skills in its plan.

    Args:
        user_request: The user's task description
        skills: Available skills keyed by name

    Returns:
        Prompt string for plan generation
    """
    normalized_skills = coerce_skill_catalog(skills)
    skill_catalog = "\n".join(
        f"  - {name}: {skill.description} (phase: {getattr(skill, 'phase', 'action')})"
        for name, skill in sorted(normalized_skills.items())
    )
    decomposition_hint = ""
    if selected_skills:
        decomposition_hint = (
            "\nSuggested ordered skill sequence from the shared framework matcher:\n"
            f"  - {' -> '.join(selected_skills)}"
        )
        if decomposition_confidence is not None:
            decomposition_hint += f" (confidence={decomposition_confidence:.2f})"
        decomposition_hint += "\n"

    return (
        f"Generate a structured task plan for the following request.\n\n"
        f"Available skills (use these as step types when applicable):\n"
        f"{skill_catalog}\n\n"
        f"{decomposition_hint}"
        f"Task: {user_request}\n\n"
        f"Return a JSON plan with steps. Each step can reference a skill "
        f"name as its type if it matches."
    )
