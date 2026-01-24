"""Data Analysis Vertical Package - Complete implementation with extensions.

Competitive use case: ChatGPT Data Analysis, Claude Artifacts, Jupyter AI, Code Interpreter.

This vertical provides:
- Data exploration and profiling
- Statistical analysis and visualization
- Machine learning model training
- Report generation with insights
- CSV/Excel/JSON data processing
"""

from victor.dataanalysis.assistant import DataAnalysisAssistant
from victor.dataanalysis.prompts import DataAnalysisPromptContributor
from victor.dataanalysis.mode_config import DataAnalysisModeConfigProvider
from victor.dataanalysis.safety import DataAnalysisSafetyExtension
from victor.dataanalysis.capabilities import DataAnalysisCapabilityProvider

# Import canonical tool dependency provider instead of deprecated class
from victor.core.tool_dependency_loader import create_vertical_tool_dependency_provider

# Create canonical provider for data analysis vertical
DataAnalysisToolDependencyProvider = create_vertical_tool_dependency_provider("data_analysis")

# Import lazy initializer for eliminating import side-effects
from victor.framework.lazy_initializer import get_initializer_for_vertical


# Auto-register escape hatches (OCP-compliant)
def _register_escape_hatches() -> None:
    """Register data analysis vertical's escape hatches with the global registry.

    Phase 5 Import Side-Effects Remediation:
    This function now uses lazy initialization via LazyInitializer to eliminate
    import-time side effects. Registration occurs on first use, not on import.
    """
    try:
        from victor.framework.escape_hatch_registry import EscapeHatchRegistry

        # Import escape hatches module to trigger side-effect registration
        from victor.dataanalysis import escape_hatches  # noqa: F401

        # Register with global registry
        registry = EscapeHatchRegistry.get_instance()
        registry.register_from_vertical(
            "data_analysis",
            conditions=escape_hatches.CONDITIONS,
            transforms=escape_hatches.TRANSFORMS,
        )
    except Exception:
        # If registration fails, it's not critical
        pass


# Create lazy initializer (no import side-effect)
_lazy_init = get_initializer_for_vertical("data_analysis", _register_escape_hatches)

__all__ = [
    "DataAnalysisAssistant",
    "DataAnalysisPromptContributor",
    "DataAnalysisModeConfigProvider",
    "DataAnalysisSafetyExtension",
    "DataAnalysisCapabilityProvider",
    "DataAnalysisToolDependencyProvider",  # Now uses canonical provider
]
