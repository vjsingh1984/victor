"""VerticalBase mixins — opt-in capability groups.

Each mixin provides a logical group of methods that VerticalBase inherits.
Verticals can also import individual mixins for type-narrowing or
documentation purposes.
"""

from victor_sdk.verticals.mixins.extension_provider_mixin import ExtensionProviderMixin
from victor_sdk.verticals.mixins.prompt_metadata_mixin import PromptMetadataMixin
from victor_sdk.verticals.mixins.rl_mixin import RLMixin
from victor_sdk.verticals.mixins.team_mixin import TeamMixin
from victor_sdk.verticals.mixins.workflow_metadata_mixin import WorkflowMetadataMixin

__all__ = [
    "ExtensionProviderMixin",
    "PromptMetadataMixin",
    "RLMixin",
    "TeamMixin",
    "WorkflowMetadataMixin",
]
