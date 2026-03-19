# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data Analysis Vertical Package.

This vertical provides data exploration, statistical analysis, and visualization.
"""

import warnings
from typing import Optional

import typer
from victor_sdk import PluginContext, VictorPlugin

# Deprecation warning deferred to register() to avoid firing during discovery scan


class DataAnalysisPlugin(VictorPlugin):
    """Victor Plugin for DataAnalysis vertical."""

    @property
    def name(self) -> str:
        return "dataanalysis"

    def register(self, context: PluginContext) -> None:
        """Register DataAnalysis vertical."""
        from victor.verticals.contrib._compat import external_package_installed

        if external_package_installed("victor_dataanalysis"):
            return

        warnings.warn(
            "victor.verticals.contrib.dataanalysis is deprecated and will be removed in v0.7.0. "
            "Install the victor-dataanalysis package instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Use lazy imports inside register to minimize startup overhead
        from victor.verticals.contrib.dataanalysis.assistant import DataAnalysisAssistant

        # Register vertical
        context.register_vertical(DataAnalysisAssistant)

    def get_cli_app(self) -> Optional[typer.Typer]:
        """Return the DataAnalysis-specific CLI application if any."""
        return None


# Instantiate plugin for discovery
plugin = DataAnalysisPlugin()


# Lazy loading for members to avoid heavy imports on package discovery
def __getattr__(name: str):
    if name == "DataAnalysisAssistant":
        from victor.framework.vertical_runtime_adapter import VerticalRuntimeAdapter
        from victor.verticals.contrib.dataanalysis.assistant import (
            DataAnalysisAssistant as Definition,
        )

        return VerticalRuntimeAdapter.as_runtime_vertical_class(Definition)

    if name == "DataAnalysisAssistantDefinition":
        from victor.verticals.contrib.dataanalysis.assistant import DataAnalysisAssistant

        return DataAnalysisAssistant

    if name == "DataAnalysisPromptContributor":
        from victor.verticals.contrib.dataanalysis.prompts import DataAnalysisPromptContributor

        return DataAnalysisPromptContributor

    if name == "DataAnalysisModeConfigProvider":
        from victor.verticals.contrib.dataanalysis.runtime.mode_config import (
            DataAnalysisModeConfigProvider,
        )

        return DataAnalysisModeConfigProvider

    if name == "DataAnalysisSafetyExtension":
        from victor.verticals.contrib.dataanalysis.runtime.safety import DataAnalysisSafetyExtension

        return DataAnalysisSafetyExtension

    if name == "DataAnalysisCapabilityProvider":
        from victor.verticals.contrib.dataanalysis.runtime.capabilities import (
            DataAnalysisCapabilityProvider,
        )

        return DataAnalysisCapabilityProvider

    if name == "get_provider":
        from victor.verticals.contrib.dataanalysis.runtime.tool_dependencies import get_provider

        return get_provider

    if name == "DataAnalysisSafetyRules":
        from victor.verticals.contrib.dataanalysis.runtime.safety_enhanced import (
            DataAnalysisSafetyRules,
        )

        return DataAnalysisSafetyRules

    if name == "EnhancedDataAnalysisSafetyExtension":
        from victor.verticals.contrib.dataanalysis.runtime.safety_enhanced import (
            EnhancedDataAnalysisSafetyExtension,
        )

        return EnhancedDataAnalysisSafetyExtension

    if name == "DataAnalysisContext":
        from victor.verticals.contrib.dataanalysis.conversation_enhanced import DataAnalysisContext

        return DataAnalysisContext

    if name == "EnhancedDataAnalysisConversationManager":
        from victor.verticals.contrib.dataanalysis.conversation_enhanced import (
            EnhancedDataAnalysisConversationManager,
        )

        return EnhancedDataAnalysisConversationManager

    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = [
    "DataAnalysisAssistant",
    "DataAnalysisAssistantDefinition",
    "DataAnalysisPromptContributor",
    "DataAnalysisModeConfigProvider",
    "DataAnalysisSafetyExtension",
    "DataAnalysisCapabilityProvider",
    "get_provider",
    "DataAnalysisSafetyRules",
    "EnhancedDataAnalysisSafetyExtension",
    "DataAnalysisContext",
    "EnhancedDataAnalysisConversationManager",
]
