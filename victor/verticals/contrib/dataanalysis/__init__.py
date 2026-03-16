"""Data Analysis Vertical Package - Complete implementation with extensions.

Competitive use case: ChatGPT Data Analysis, Claude Artifacts, Jupyter AI, Code Interpreter.

This vertical provides:
- Data exploration and profiling
- Statistical analysis and visualization
- Machine learning model training
- Report generation with insights
- CSV/Excel/JSON data processing
"""

import warnings

warnings.warn(
    "victor.verticals.contrib.dataanalysis is deprecated and will be removed in v0.7.0. "
    "Install the victor-dataanalysis package instead.",
    DeprecationWarning,
    stacklevel=2,
)

from victor.framework.vertical_runtime_adapter import VerticalRuntimeAdapter
from victor.verticals.contrib.dataanalysis.assistant import (
    DataAnalysisAssistant as DataAnalysisAssistantDefinition,
)
from victor.verticals.contrib.dataanalysis.prompts import DataAnalysisPromptContributor
from victor.verticals.contrib.dataanalysis.runtime.capabilities import (
    DataAnalysisCapabilityProvider,
)
from victor.verticals.contrib.dataanalysis.runtime.mode_config import DataAnalysisModeConfigProvider
from victor.verticals.contrib.dataanalysis.runtime.safety import DataAnalysisSafetyExtension
from victor.verticals.contrib.dataanalysis.runtime.tool_dependencies import get_provider

DataAnalysisAssistant = VerticalRuntimeAdapter.as_runtime_vertical_class(
    DataAnalysisAssistantDefinition
)

__all__ = [
    "DataAnalysisAssistant",
    "DataAnalysisAssistantDefinition",
    "DataAnalysisPromptContributor",
    "DataAnalysisModeConfigProvider",
    "DataAnalysisSafetyExtension",
    "get_provider",
    "DataAnalysisCapabilityProvider",
]

# Enhanced features with new coordinators
from victor.verticals.contrib.dataanalysis.conversation_enhanced import (
    DataAnalysisContext,
    EnhancedDataAnalysisConversationManager,
)
from victor.verticals.contrib.dataanalysis.runtime.safety_enhanced import (
    DataAnalysisSafetyRules,
    EnhancedDataAnalysisSafetyExtension,
)

__all__.extend(
    [
        "DataAnalysisSafetyRules",
        "EnhancedDataAnalysisSafetyExtension",
        "DataAnalysisContext",
        "EnhancedDataAnalysisConversationManager",
    ]
)
