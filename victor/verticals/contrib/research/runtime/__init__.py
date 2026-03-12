"""Runtime-owned helpers for the Research vertical."""

from victor.verticals.contrib.research.runtime.capabilities import ResearchCapabilityProvider
from victor.verticals.contrib.research.runtime.mode_config import ResearchModeConfigProvider
from victor.verticals.contrib.research.runtime.rl import (
    ResearchRLConfig,
    ResearchRLHooks,
    get_default_config,
    get_research_rl_hooks,
)
from victor.verticals.contrib.research.runtime.safety import ResearchSafetyExtension
from victor.verticals.contrib.research.runtime.safety_enhanced import (
    EnhancedResearchSafetyExtension,
    ResearchSafetyRules,
)
from victor.verticals.contrib.research.runtime.team_personas import (
    RESEARCH_PERSONAS,
    ResearchPersona,
    ResearchPersonaTraits,
)
from victor.verticals.contrib.research.runtime.teams import (
    RESEARCH_TEAM_SPECS,
    ResearchTeamSpecProvider,
    register_research_teams,
)
from victor.verticals.contrib.research.runtime.tool_dependencies import get_provider
from victor.verticals.contrib.research.runtime.workflows import ResearchWorkflowProvider

__all__ = [
    "EnhancedResearchSafetyExtension",
    "RESEARCH_PERSONAS",
    "RESEARCH_TEAM_SPECS",
    "ResearchCapabilityProvider",
    "ResearchModeConfigProvider",
    "ResearchPersona",
    "ResearchPersonaTraits",
    "ResearchRLConfig",
    "ResearchRLHooks",
    "ResearchSafetyExtension",
    "ResearchSafetyRules",
    "ResearchTeamSpecProvider",
    "ResearchWorkflowProvider",
    "get_default_config",
    "get_provider",
    "get_research_rl_hooks",
    "register_research_teams",
]
