"""Skill provider protocol for Victor SDK.

Defines the structural typing protocol that any skill provider
(verticals, plugins, external packages) must satisfy.
"""

from __future__ import annotations

from typing import List, runtime_checkable

from typing_extensions import Protocol

from victor_sdk.skills.definition import SkillDefinition


@runtime_checkable
class SkillProvider(Protocol):
    """Protocol for components that provide skill definitions.

    Any class with a ``get_skills()`` method returning a list of
    SkillDefinition objects satisfies this protocol.
    """

    def get_skills(self) -> List[SkillDefinition]: ...
