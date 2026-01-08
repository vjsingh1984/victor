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
from victor.dataanalysis.tool_dependencies import DataAnalysisToolDependencyProvider
from victor.dataanalysis.capabilities import DataAnalysisCapabilityProvider

__all__ = [
    "DataAnalysisAssistant",
    "DataAnalysisPromptContributor",
    "DataAnalysisModeConfigProvider",
    "DataAnalysisSafetyExtension",
    "DataAnalysisToolDependencyProvider",
    "DataAnalysisCapabilityProvider",
]
