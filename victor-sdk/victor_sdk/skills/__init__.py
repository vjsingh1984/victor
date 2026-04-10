"""Skills package for Victor SDK.

Provides the SkillDefinition contract and SkillProvider protocol
for declaring composable agent expertise.
"""

from victor_sdk.skills.definition import SkillDefinition
from victor_sdk.skills.provider import SkillProvider

__all__ = ["SkillDefinition", "SkillProvider"]
