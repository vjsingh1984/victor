"""Service-owned skill auto-selection runtime helper."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class SkillRuntime:
    """Bridge skill-selection orchestration off the concrete orchestrator."""

    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime

    def apply_skill_for_turn(self, user_message: str) -> None:
        """Apply skill auto-selection for a chat turn."""
        runtime = self._runtime
        runtime.clear_active_skills()

        matcher = getattr(runtime, "_skill_matcher", None)
        if (
            matcher is None
            or not getattr(matcher, "_initialized", False)
            or getattr(runtime, "_skill_auto_disabled", False)
            or getattr(runtime, "_manual_skill_active", False)
        ):
            return

        try:
            matches = matcher.match_multiple_sync(user_message)
            analytics = getattr(runtime, "_skill_analytics", None)

            if matches:
                if len(matches) == 1:
                    skill, score = matches[0]
                    logger.info("Auto-selected skill: %s (score=%.2f)", skill.name, score)
                    runtime.inject_skill(skill)
                    runtime._last_skill_match_info = {
                        "auto_skill": skill.name,
                        "auto_skill_score": round(score, 2),
                    }
                    if analytics:
                        analytics.record_selection(skill.name, score)
                    return

                names = [skill.name for skill, _ in matches]
                logger.info("Auto-selected %d skills: %s", len(matches), " → ".join(names))
                runtime.inject_skills(matches)
                runtime._last_skill_match_info = {
                    "auto_skills": [
                        {"name": skill.name, "score": round(score, 2)}
                        for skill, score in matches
                    ],
                }
                if analytics:
                    analytics.record_multi_selection(
                        [(skill.name, score) for skill, score in matches]
                    )
                return

            if analytics:
                analytics.record_miss()
            runtime._last_skill_match_info = None
        except Exception:
            logger.debug("Skill auto-selection failed", exc_info=True)

    def get_last_skill_match_info(self) -> Optional[Dict[str, Any]]:
        """Return metadata about the last skill match for response attachment."""
        return getattr(self._runtime, "_last_skill_match_info", None)
