"""Skill registry for Victor framework.

Discovers and manages SkillDefinition instances from verticals,
plugins, entry points, and YAML files. Provides lookup, search,
and enumeration.
"""

from __future__ import annotations

import logging
import os
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from victor_sdk.skills import SkillDefinition

logger = logging.getLogger(__name__)

SKILLS_ENTRY_POINT_GROUP = "victor.skills"
USER_SKILLS_DIR = os.path.join(Path.home(), ".victor", "skills")


class SkillRegistry:
    """Registry for skill definitions.

    Skills can be registered directly, loaded from verticals via
    ``from_vertical()``, or discovered from entry points via
    ``from_entry_points()``.
    """

    def __init__(self) -> None:
        self._skills: Dict[str, SkillDefinition] = {}

    def register(self, skill: SkillDefinition) -> None:
        """Register a skill definition."""
        self._skills[skill.name] = skill
        logger.debug("Registered skill: %s (category=%s)", skill.name, skill.category)

    def get(self, name: str) -> SkillDefinition:
        """Get a skill by name. Raises KeyError if not found."""
        try:
            return self._skills[name]
        except KeyError:
            raise KeyError(
                f"Skill '{name}' not found. " f"Available: {sorted(self._skills.keys())}"
            )

    def get_optional(self, name: str) -> Optional[SkillDefinition]:
        """Get a skill by name, returning None if not found."""
        return self._skills.get(name)

    def list_all(self) -> List[SkillDefinition]:
        """Return all registered skills."""
        return list(self._skills.values())

    def search(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
    ) -> List[SkillDefinition]:
        """Search skills by query string and/or category.

        Query matches against name, description, and tags (case-insensitive).
        """
        results = list(self._skills.values())

        if category is not None:
            results = [s for s in results if s.category == category]

        if query is not None:
            query_lower = query.lower()
            results = [
                s
                for s in results
                if query_lower in s.name.lower()
                or query_lower in s.description.lower()
                or any(query_lower in tag.lower() for tag in s.tags)
            ]

        return results

    def from_vertical(self, vertical_cls: Type[Any]) -> None:
        """Load skills from a vertical class's get_skills() method."""
        get_skills = getattr(vertical_cls, "get_skills", None)
        if get_skills is None:
            return
        try:
            skills = get_skills()
            for skill in skills:
                if isinstance(skill, SkillDefinition):
                    self.register(skill)
        except Exception:
            logger.warning(
                "Failed to load skills from vertical %s",
                getattr(vertical_cls, "name", vertical_cls),
                exc_info=True,
            )

    def from_entry_points(self) -> None:
        """Discover and register skills from victor.skills entry points."""
        try:
            eps = entry_points(group=SKILLS_ENTRY_POINT_GROUP)
        except TypeError:
            eps = entry_points().get(SKILLS_ENTRY_POINT_GROUP, [])

        for ep in eps:
            try:
                loaded = ep.load()
                if isinstance(loaded, SkillDefinition):
                    self.register(loaded)
                elif callable(loaded):
                    result = loaded()
                    if isinstance(result, list):
                        for skill in result:
                            if isinstance(skill, SkillDefinition):
                                self.register(skill)
                    elif isinstance(result, SkillDefinition):
                        self.register(result)
                logger.info("Loaded skill from entry point: %s", ep.name)
            except Exception:
                logger.warning(
                    "Failed to load skill from entry point: %s",
                    ep.name,
                    exc_info=True,
                )

    def from_yaml_directory(self, directory: str) -> None:
        """Load skill definitions from YAML files in a directory.

        Each ``.yaml`` or ``.yml`` file should contain a single skill
        definition with fields matching ``SkillDefinition.from_dict()``.
        """
        if not os.path.isdir(directory):
            logger.debug("Skills directory does not exist: %s", directory)
            return

        import yaml

        for filename in sorted(os.listdir(directory)):
            if not filename.endswith((".yaml", ".yml")):
                continue
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath) as f:
                    data = yaml.safe_load(f)
                if isinstance(data, dict) and "name" in data:
                    skill = SkillDefinition.from_dict(data)
                    self.register(skill)
                    logger.debug("Loaded skill from YAML: %s", filepath)
            except Exception:
                logger.warning(
                    "Failed to load skill from YAML: %s",
                    filepath,
                    exc_info=True,
                )

    def from_user_skills(self) -> None:
        """Load skills from the user's ~/.victor/skills/ directory."""
        self.from_yaml_directory(USER_SKILLS_DIR)
