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

"""Research Vertical Package.

This vertical provides web research and synthesis capabilities.
"""

import warnings
from typing import Optional

import typer
from victor_sdk import PluginContext, VictorPlugin

# Deprecation warning deferred to register() to avoid firing during discovery scan


class ResearchPlugin(VictorPlugin):
    """Victor Plugin for Research vertical."""

    @property
    def name(self) -> str:
        return "research"

    def register(self, context: PluginContext) -> None:
        """Register Research vertical."""
        from victor.verticals.contrib._compat import external_package_installed

        if external_package_installed("victor_research"):
            return

        warnings.warn(
            "victor.verticals.contrib.research is deprecated and will be removed in v0.7.0. "
            "Install the victor-research package instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Use lazy imports inside register to minimize startup overhead
        from victor.verticals.contrib.research.assistant import ResearchAssistant

        # Register vertical
        context.register_vertical(ResearchAssistant)

    def get_cli_app(self) -> Optional[typer.Typer]:
        """Return the Research-specific CLI application if any."""
        return None


# Instantiate plugin for discovery
plugin = ResearchPlugin()


# Lazy loading for members to avoid heavy imports on package discovery
def __getattr__(name: str):
    if name == "ResearchAssistant":
        from victor.framework.vertical_runtime_adapter import VerticalRuntimeAdapter
        from victor.verticals.contrib.research.assistant import ResearchAssistant as Definition

        return VerticalRuntimeAdapter.as_runtime_vertical_class(Definition)

    if name == "ResearchAssistantDefinition":
        from victor.verticals.contrib.research.assistant import ResearchAssistant

        return ResearchAssistant

    if name == "ResearchPromptContributor":
        from victor.verticals.contrib.research.prompts import ResearchPromptContributor

        return ResearchPromptContributor

    if name == "ResearchCapabilityProvider":
        from victor.verticals.contrib.research.runtime.capabilities import (
            ResearchCapabilityProvider,
        )

        return ResearchCapabilityProvider

    if name == "ResearchModeConfigProvider":
        from victor.verticals.contrib.research.runtime.mode_config import ResearchModeConfigProvider

        return ResearchModeConfigProvider

    if name == "ResearchSafetyExtension":
        from victor.verticals.contrib.research.runtime.safety import ResearchSafetyExtension

        return ResearchSafetyExtension

    if name == "get_provider":
        from victor.verticals.contrib.research.runtime.tool_dependencies import get_provider

        return get_provider

    if name == "ResearchSafetyRules":
        from victor.verticals.contrib.research.runtime.safety_enhanced import ResearchSafetyRules

        return ResearchSafetyRules

    if name == "EnhancedResearchSafetyExtension":
        from victor.verticals.contrib.research.runtime.safety_enhanced import (
            EnhancedResearchSafetyExtension,
        )

        return EnhancedResearchSafetyExtension

    if name == "ResearchContext":
        from victor.verticals.contrib.research.conversation_enhanced import ResearchContext

        return ResearchContext

    if name == "EnhancedResearchConversationManager":
        from victor.verticals.contrib.research.conversation_enhanced import (
            EnhancedResearchConversationManager,
        )

        return EnhancedResearchConversationManager

    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = [
    "ResearchAssistant",
    "ResearchAssistantDefinition",
    "ResearchPromptContributor",
    "ResearchModeConfigProvider",
    "ResearchSafetyExtension",
    "ResearchCapabilityProvider",
    "get_provider",
    "ResearchSafetyRules",
    "EnhancedResearchSafetyExtension",
    "ResearchContext",
    "EnhancedResearchConversationManager",
]
