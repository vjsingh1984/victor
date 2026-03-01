"""Research Vertical Package - Complete implementation with extensions.

Competitive use case: Perplexity AI, ChatGPT Research Mode, Google Gemini Deep Research.

This vertical provides:
- Web research and fact-checking capabilities
- Academic and technical literature synthesis
- Structured report generation
- Source verification and citation management
"""

from victor.verticals.contrib.research.assistant import ResearchAssistant
from victor.verticals.contrib.research.prompts import ResearchPromptContributor
from victor.verticals.contrib.research.mode_config import ResearchModeConfigProvider
from victor.verticals.contrib.research.safety import ResearchSafetyExtension
from victor.verticals.contrib.research.capabilities import ResearchCapabilityProvider

# Import canonical tool dependency provider instead of deprecated class
from victor.core.tool_dependency_loader import create_vertical_tool_dependency_provider

# Create canonical provider for research vertical
ResearchToolDependencyProvider = create_vertical_tool_dependency_provider("research")

__all__ = [
    "ResearchAssistant",
    "ResearchPromptContributor",
    "ResearchModeConfigProvider",
    "ResearchSafetyExtension",
    "ResearchCapabilityProvider",  # Capability provider
    "ResearchToolDependencyProvider",  # Now uses canonical provider
]

# Enhanced features with new coordinators
from victor.verticals.contrib.research.safety_enhanced import (
    ResearchSafetyRules,
    EnhancedResearchSafetyExtension,
)
from victor.verticals.contrib.research.conversation_enhanced import (
    ResearchContext,
    EnhancedResearchConversationManager,
)

__all__.extend([
    "ResearchSafetyRules",
    "EnhancedResearchSafetyExtension",
    "ResearchContext",
    "EnhancedResearchConversationManager",
])
