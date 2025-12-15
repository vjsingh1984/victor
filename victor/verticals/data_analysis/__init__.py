"""Data Analysis Vertical Package - Complete implementation with extensions.

Competitive use case: ChatGPT Data Analysis, Claude Artifacts, Jupyter AI, Code Interpreter.

This vertical provides:
- Data exploration and profiling
- Statistical analysis and visualization
- Machine learning model training
- Report generation with insights
- CSV/Excel/JSON data processing
"""

from victor.verticals.data_analysis.assistant import DataAnalysisAssistant
from victor.verticals.data_analysis.prompts import DataAnalysisPromptContributor
from victor.verticals.data_analysis.mode_config import DataAnalysisModeConfigProvider
from victor.verticals.data_analysis.safety import DataAnalysisSafetyExtension
from victor.verticals.data_analysis.tool_dependencies import DataAnalysisToolDependencyProvider

__all__ = [
    "DataAnalysisAssistant",
    "DataAnalysisPromptContributor",
    "DataAnalysisModeConfigProvider",
    "DataAnalysisSafetyExtension",
    "DataAnalysisToolDependencyProvider",
]
