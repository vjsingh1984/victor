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

# Import lazy initializer for eliminating import side-effects
from victor.framework.lazy_initializer import get_initializer_for_vertical


# Auto-register escape hatches (OCP-compliant)
def _register_escape_hatches() -> None:
    """Register research vertical's escape hatches with the global registry.

    Phase 5 Import Side-Effects Remediation:
    This function now uses lazy initialization via LazyInitializer to eliminate
    import-time side effects. Registration occurs on first use, not on import.
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


# Create lazy initializer (no import side-effect)
_lazy_init = get_initializer_for_vertical("research", _register_escape_hatches)

__all__ = [
    "ResearchAssistant",
    "ResearchPromptContributor",
    "ResearchModeConfigProvider",
    "ResearchSafetyExtension",
    "ResearchCapabilityProvider",  # Capability provider
    "ResearchToolDependencyProvider",  # Now uses canonical provider
]
