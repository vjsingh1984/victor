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

"""Coding Vertical Package.

Victor's primary vertical for software development.
"""

import warnings
from typing import Optional

import typer
from victor_sdk import PluginContext, VictorPlugin

# Deprecation warning deferred to register() to avoid firing during discovery scan


class CodingPlugin(VictorPlugin):
    """Victor Plugin for Coding vertical."""

    @property
    def name(self) -> str:
        return "coding"

    def register(self, context: PluginContext) -> None:
        """Register coding vertical and its specialized strategies."""
        from victor.verticals.contrib._compat import external_package_installed

        if external_package_installed("victor_coding"):
            return

        warnings.warn(
            "victor.verticals.contrib.coding is deprecated and will be removed in v0.7.0. "
            "Install the victor-coding package instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Use lazy imports inside register to minimize startup overhead
        from victor.verticals.contrib.coding.assistant import CodingAssistant
        from victor.verticals.contrib.coding.codebase.chunker import CodeChunker

        # Register vertical
        context.register_vertical(CodingAssistant)

        # Register specialized chunker
        context.register_chunker(CodeChunker())

        # Register coding-specific tools
        try:
            from victor.verticals.contrib.coding.tools.language_analyzer import language_analyzer
            from victor.verticals.contrib.coding.tools.graph_tool import graph
            from victor.verticals.contrib.coding.tools.code_review_tool import code_review
            from victor.verticals.contrib.coding.tools.architecture_summary import (
                architecture_summary,
            )

            context.register_tool(language_analyzer)
            context.register_tool(graph)
            context.register_tool(code_review)
            context.register_tool(architecture_summary)
        except ImportError:
            pass

        # Register domain command
        from victor.verticals.contrib.coding.commands.analyze import app as analyze_app

        context.register_command("analyze", analyze_app)

    def get_cli_app(self) -> Optional[typer.Typer]:
        """Return the coding-specific CLI application (legacy hook)."""
        from victor.verticals.contrib.coding.commands.analyze import app as analyze_app

        return analyze_app


# Instantiate plugin for discovery
plugin = CodingPlugin()


# Lazy loading for members to avoid heavy imports on package discovery
def __getattr__(name: str):
    if name == "CodingAssistant":
        from victor.framework.vertical_runtime_adapter import VerticalRuntimeAdapter
        from victor.verticals.contrib.coding.assistant import CodingAssistant as Definition

        return VerticalRuntimeAdapter.as_runtime_vertical_class(Definition)

    if name == "CodingMiddleware":
        from victor.verticals.contrib.coding.middleware import CodingMiddleware

        return CodingMiddleware

    if name == "CodeCorrectionMiddleware":
        from victor.verticals.contrib.coding.middleware import CodeCorrectionMiddleware

        return CodeCorrectionMiddleware

    if name == "CodingSafetyExtension":
        from victor.verticals.contrib.coding.safety import CodingSafetyExtension

        return CodingSafetyExtension

    if name == "EnhancedCodingSafetyExtension":
        from victor.verticals.contrib.coding.safety_enhanced import EnhancedCodingSafetyExtension

        return EnhancedCodingSafetyExtension

    if name == "EnhancedCodingConversationManager":
        from victor.verticals.contrib.coding.conversation_enhanced import (
            EnhancedCodingConversationManager,
        )

        return EnhancedCodingConversationManager

    if name == "CodingToolDependencyProvider":
        from victor.core.tool_dependency_loader import create_vertical_tool_dependency_provider

        return create_vertical_tool_dependency_provider("coding")

    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = [
    "CodingAssistant",
    "CodingMiddleware",
    "CodeCorrectionMiddleware",
    "CodingSafetyExtension",
    "EnhancedCodingSafetyExtension",
    "EnhancedCodingConversationManager",
    "CodingToolDependencyProvider",
]
