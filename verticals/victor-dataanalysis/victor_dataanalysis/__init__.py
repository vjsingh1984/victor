"""Data Analysis Vertical Package - Complete implementation with extensions.

Competitive use case: ChatGPT Data Analysis, Claude Artifacts, Jupyter AI, Code Interpreter.

This vertical provides:
- Data exploration and profiling
- Statistical analysis and visualization
- Machine learning model training
- Report generation with insights
- CSV/Excel/JSON data processing
"""

from victor_dataanalysis.assistant import DataAnalysisAssistant
from victor_dataanalysis.prompts import DataAnalysisPromptContributor
from victor_dataanalysis.mode_config import DataAnalysisModeConfigProvider
from victor_dataanalysis.safety import DataAnalysisSafetyExtension
from victor_dataanalysis.tool_dependencies import DataAnalysisToolDependencyProvider
from victor_dataanalysis.capabilities import DataAnalysisCapabilityProvider

__all__ = [
    "DataAnalysisAssistant",
    "DataAnalysisPromptContributor",
    "DataAnalysisModeConfigProvider",
    "DataAnalysisSafetyExtension",
    "DataAnalysisToolDependencyProvider",
    "DataAnalysisCapabilityProvider",
]

# Enhanced features with new coordinators
from victor_dataanalysis.safety_enhanced import (
    DataAnalysisSafetyRules,
    EnhancedDataAnalysisSafetyExtension,
)
from victor_dataanalysis.conversation_enhanced import (
    DataAnalysisContext,
    EnhancedDataAnalysisConversationManager,
)

__all__.extend([
    "DataAnalysisSafetyRules",
    "EnhancedDataAnalysisSafetyExtension",
    "DataAnalysisContext",
    "EnhancedDataAnalysisConversationManager",
])
