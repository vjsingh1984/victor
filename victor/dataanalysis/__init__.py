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

# Auto-register escape hatches (OCP-compliant)
def _register_escape_hatches() -> None:
    """Register data analysis vertical's escape hatches with the global registry.

    This function runs on module import to automatically register escape hatches.
    Using the registry's discover_from_all_verticals() method is the OCP-compliant
    approach, as it doesn't require the framework to know about specific verticals.
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


# Register on import
_register_escape_hatches()

__all__ = [
    "DataAnalysisAssistant",
    "DataAnalysisPromptContributor",
    "DataAnalysisModeConfigProvider",
    "DataAnalysisSafetyExtension",
    "DataAnalysisCapabilityProvider",
    "DataAnalysisToolDependencyProvider",  # Now uses canonical provider
]
