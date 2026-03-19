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

"""DevOps Vertical Package.

This vertical provides infrastructure automation and CI/CD capabilities.
"""

import warnings
from typing import Optional

import typer
from victor_sdk import PluginContext, VictorPlugin

# Deprecation warning deferred to register() to avoid firing during discovery scan


class DevOpsPlugin(VictorPlugin):
    """Victor Plugin for DevOps vertical."""

    @property
    def name(self) -> str:
        return "devops"

    def register(self, context: PluginContext) -> None:
        """Register DevOps vertical."""
        from victor.verticals.contrib._compat import external_package_installed

        if external_package_installed("victor_devops"):
            return

        warnings.warn(
            "victor.verticals.contrib.devops is deprecated and will be removed in v0.7.0. "
            "Install the victor-devops package instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Use lazy imports inside register to minimize startup overhead
        from victor.verticals.contrib.devops.assistant import DevOpsAssistant

        # Register vertical
        context.register_vertical(DevOpsAssistant)

    def get_cli_app(self) -> Optional[typer.Typer]:
        """Return the DevOps-specific CLI application if any."""
        return None


# Instantiate plugin for discovery
plugin = DevOpsPlugin()


# Lazy loading for members to avoid heavy imports on package discovery
def __getattr__(name: str):
    if name == "DevOpsAssistant":
        from victor.framework.vertical_runtime_adapter import VerticalRuntimeAdapter
        from victor.verticals.contrib.devops.assistant import DevOpsAssistant as Definition

        return VerticalRuntimeAdapter.as_runtime_vertical_class(Definition)

    if name == "DevOpsAssistantDefinition":
        from victor.verticals.contrib.devops.assistant import DevOpsAssistant

        return DevOpsAssistant

    if name == "DevOpsPromptContributor":
        from victor.verticals.contrib.devops.prompts import DevOpsPromptContributor

        return DevOpsPromptContributor

    if name == "DevOpsModeConfigProvider":
        from victor.verticals.contrib.devops.mode_config import DevOpsModeConfigProvider

        return DevOpsModeConfigProvider

    if name == "DevOpsSafetyExtension":
        from victor.verticals.contrib.devops.safety import DevOpsSafetyExtension

        return DevOpsSafetyExtension

    if name == "EnhancedDevOpsSafetyExtension":
        from victor.verticals.contrib.devops.safety_enhanced import EnhancedDevOpsSafetyExtension

        return EnhancedDevOpsSafetyExtension

    if name == "DevOpsSafetyRules":
        from victor.verticals.contrib.devops.safety_enhanced import DevOpsSafetyRules

        return DevOpsSafetyRules

    if name == "EnhancedDevOpsConversationManager":
        from victor.verticals.contrib.devops.conversation_enhanced import (
            EnhancedDevOpsConversationManager,
        )

        return EnhancedDevOpsConversationManager

    if name == "DevOpsContext":
        from victor.verticals.contrib.devops.conversation_enhanced import DevOpsContext

        return DevOpsContext

    if name == "get_provider":
        from victor.verticals.contrib.devops.tool_dependencies import get_provider

        return get_provider

    if name == "DevOpsCapabilityProvider":
        from victor.verticals.contrib.devops.capabilities import DevOpsCapabilityProvider

        return DevOpsCapabilityProvider

    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = [
    "DevOpsAssistant",
    "DevOpsAssistantDefinition",
    "DevOpsPromptContributor",
    "DevOpsModeConfigProvider",
    "DevOpsSafetyExtension",
    "EnhancedDevOpsSafetyExtension",
    "DevOpsSafetyRules",
    "EnhancedDevOpsConversationManager",
    "DevOpsContext",
    "get_provider",
    "DevOpsCapabilityProvider",
]
