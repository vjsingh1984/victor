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
from victor.research.capabilities import ResearchCapabilityProvider

# Import canonical tool dependency provider instead of deprecated class
from victor.core.tool_dependency_loader import create_vertical_tool_dependency_provider

# Create canonical provider for research vertical
ResearchToolDependencyProvider = create_vertical_tool_dependency_provider("research")


# Auto-register escape hatches (OCP-compliant)
def _register_escape_hatches() -> None:
    """Register research vertical's escape hatches with the global registry.

    This function runs on module import to automatically register escape hatches.
    Using the registry's discover_from_all_verticals() method is the OCP-compliant
    approach, as it doesn't require the framework to know about specific verticals.
    """
    try:
        from victor.framework.escape_hatch_registry import EscapeHatchRegistry

        # Import escape hatches module to trigger side-effect registration
        from victor.research import escape_hatches  # noqa: F401

        # Register with global registry
        registry = EscapeHatchRegistry.get_instance()
        registry.register_from_vertical(
            "research",
            conditions=escape_hatches.CONDITIONS,
            transforms=escape_hatches.TRANSFORMS,
        )
    except Exception:
        # If registration fails, it's not critical
        pass


# Register on import
_register_escape_hatches()

__all__ = [
    "ResearchAssistant",
    "ResearchPromptContributor",
    "ResearchModeConfigProvider",
    "ResearchSafetyExtension",
    "ResearchCapabilityProvider",  # Capability provider
    "ResearchToolDependencyProvider",  # Now uses canonical provider
]
