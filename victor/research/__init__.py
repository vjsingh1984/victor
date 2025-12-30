"""Research Vertical Package - Complete implementation with extensions.

Competitive use case: Perplexity AI, ChatGPT Research Mode, Google Gemini Deep Research.

This vertical provides:
- Web research and fact-checking capabilities
- Academic and technical literature synthesis
- Structured report generation
- Source verification and citation management
"""

from victor.research.assistant import ResearchAssistant
from victor.research.prompts import ResearchPromptContributor
from victor.research.mode_config import ResearchModeConfigProvider
from victor.research.safety import ResearchSafetyExtension
from victor.research.tool_dependencies import ResearchToolDependencyProvider

__all__ = [
    "ResearchAssistant",
    "ResearchPromptContributor",
    "ResearchModeConfigProvider",
    "ResearchSafetyExtension",
    "ResearchToolDependencyProvider",
]
