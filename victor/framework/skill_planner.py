"""Skill-aware task planning — enriches plan steps with matching skills.

Maps plan step types and descriptions to registered skills so the agent
gets skill-specific guidance for each step during plan execution.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from victor_sdk.skills import SkillDefinition

logger = logging.getLogger(__name__)


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
) -> str:
    """Build a plan generation prompt that includes available skills.

    The LLM sees the skill catalog and can reference skills in its plan.

    Args:
        user_request: The user's task description
        skills: Available skills keyed by name

    Returns:
        Prompt string for plan generation
    """
    skill_catalog = "\n".join(
        f"  - {name}: {skill.description} (phase: {getattr(skill, 'phase', 'action')})"
        for name, skill in sorted(skills.items())
    )

    return (
        f"Generate a structured task plan for the following request.\n\n"
        f"Available skills (use these as step types when applicable):\n"
        f"{skill_catalog}\n\n"
        f"Task: {user_request}\n\n"
        f"Return a JSON plan with steps. Each step can reference a skill "
        f"name as its type if it matches."
    )
