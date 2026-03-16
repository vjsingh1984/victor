"""Research Vertical Package - Complete implementation with extensions.

Competitive use case: Perplexity AI, ChatGPT Research Mode, Google Gemini Deep Research.

This vertical provides:
- Web research and fact-checking capabilities
- Academic and technical literature synthesis
- Structured report generation
- Source verification and citation management
"""

import warnings

warnings.warn(
    "victor.verticals.contrib.research is deprecated and will be removed in v0.7.0. "
    "Install the victor-research package instead.",
    DeprecationWarning,
    stacklevel=2,
)

from victor.framework.vertical_runtime_adapter import VerticalRuntimeAdapter
from victor.verticals.contrib.research.assistant import (
    ResearchAssistant as ResearchAssistantDefinition,
)
from victor.verticals.contrib.research.prompts import ResearchPromptContributor
from victor.verticals.contrib.research.runtime.capabilities import ResearchCapabilityProvider
from victor.verticals.contrib.research.runtime.mode_config import ResearchModeConfigProvider
from victor.verticals.contrib.research.runtime.safety import ResearchSafetyExtension
from victor.verticals.contrib.research.runtime.tool_dependencies import get_provider

ResearchAssistant = VerticalRuntimeAdapter.as_runtime_vertical_class(ResearchAssistantDefinition)

__all__ = [
    "ResearchAssistant",
    "ResearchAssistantDefinition",
    "ResearchPromptContributor",
    "ResearchModeConfigProvider",
    "ResearchSafetyExtension",
    "ResearchCapabilityProvider",  # Capability provider
    "get_provider",  # Tool dependency provider factory
]

# Enhanced features with new coordinators
from victor.verticals.contrib.research.runtime.safety_enhanced import (
    ResearchSafetyRules,
    EnhancedResearchSafetyExtension,
)
from victor.verticals.contrib.research.conversation_enhanced import (
    ResearchContext,
    EnhancedResearchConversationManager,
)

__all__.extend(
    [
        "ResearchSafetyRules",
        "EnhancedResearchSafetyExtension",
        "ResearchContext",
        "EnhancedResearchConversationManager",
    ]
)
