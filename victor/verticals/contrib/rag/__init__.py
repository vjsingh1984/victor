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

"""RAG (Retrieval-Augmented Generation) Vertical Package.

This vertical provides a complete RAG implementation.
"""

import warnings
from typing import Optional

import typer
from victor_sdk import PluginContext, VictorPlugin

# Deprecation warning deferred to register() to avoid firing during discovery scan


class RAGPlugin(VictorPlugin):
    """Victor Plugin for RAG vertical."""

    @property
    def name(self) -> str:
        return "rag"

    def register(self, context: PluginContext) -> None:
        """Register RAG vertical and its specialized strategies."""
        from victor.verticals.contrib._compat import external_package_installed

        if external_package_installed("victor_rag"):
            return

        warnings.warn(
            "victor.verticals.contrib.rag is deprecated and will be removed in v0.7.0. "
            "Install the victor-rag package instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Use lazy imports inside register to minimize startup overhead
        from victor.verticals.contrib.rag.assistant import RAGAssistant
        from victor.verticals.contrib.rag.chunker import RAGChunkingStrategy

        # Register vertical
        context.register_vertical(RAGAssistant)

        # Register specialized chunker
        context.register_chunker(RAGChunkingStrategy())

        # Register RAG-specific tools
        try:
            from victor.verticals.contrib.rag.tools.rag_tools import (
                rag_search,
                rag_index,
                rag_list,
                rag_delete,
                rag_stats,
            )

            context.register_tool(rag_search)
            context.register_tool(rag_index)
            context.register_tool(rag_list)
            context.register_tool(rag_delete)
            context.register_tool(rag_stats)
        except ImportError:
            pass

        # Register domain command
        try:
            from victor.verticals.contrib.rag.commands.rag import rag_app

            context.register_command("rag", rag_app)
        except ImportError:
            pass

    def get_cli_app(self) -> Optional[typer.Typer]:
        """Return the RAG-specific CLI application (legacy hook)."""
        try:
            from victor.verticals.contrib.rag.commands.rag import rag_app

            return rag_app
        except ImportError:
            return None


# Instantiate plugin for discovery
plugin = RAGPlugin()


# Lazy loading for members to avoid heavy imports on package discovery
def __getattr__(name: str):
    if name == "RAGAssistant":
        from victor.framework.vertical_runtime_adapter import VerticalRuntimeAdapter
        from victor.verticals.contrib.rag.assistant import RAGAssistant as Definition

        return VerticalRuntimeAdapter.as_runtime_vertical_class(Definition)

    if name == "RAGAssistantDefinition":
        from victor.verticals.contrib.rag.assistant import RAGAssistant

        return RAGAssistant

    if name in [
        "Document",
        "DocumentChunk",
        "DocumentSearchResult",
        "DocumentStore",
        "DocumentStoreConfig",
    ]:
        from victor.verticals.contrib.rag.document_store import (
            Document,
            DocumentChunk,
            DocumentSearchResult,
            DocumentStore,
            DocumentStoreConfig,
        )

        return locals()[name]

    if name in ["DocumentChunker", "ChunkingConfig"]:
        from victor.verticals.contrib.rag.chunker import DocumentChunker, ChunkingConfig

        return locals()[name]

    if name == "RAGPromptContributor":
        from victor.verticals.contrib.rag.prompts import RAGPromptContributor

        return RAGPromptContributor

    if name == "RAGModeConfigProvider":
        from victor.verticals.contrib.rag.runtime.mode_config import RAGModeConfigProvider

        return RAGModeConfigProvider

    if name == "RAGCapabilityProvider":
        from victor.verticals.contrib.rag.runtime.capabilities import RAGCapabilityProvider

        return RAGCapabilityProvider

    if name in [
        "RAGIngestTool",
        "RAGSearchTool",
        "RAGQueryTool",
        "RAGListTool",
        "RAGDeleteTool",
        "RAGStatsTool",
    ]:
        from victor.verticals.contrib.rag.tools import (
            RAGIngestTool,
            RAGSearchTool,
            RAGQueryTool,
            RAGListTool,
            RAGDeleteTool,
            RAGStatsTool,
        )

        return locals()[name]

    if name == "RAGSafetyRules":
        from victor.verticals.contrib.rag.runtime.safety_enhanced import RAGSafetyRules

        return RAGSafetyRules

    if name == "EnhancedRAGSafetyExtension":
        from victor.verticals.contrib.rag.runtime.safety_enhanced import EnhancedRAGSafetyExtension

        return EnhancedRAGSafetyExtension

    if name == "RAGContext":
        from victor.verticals.contrib.rag.conversation_enhanced import RAGContext

        return RAGContext

    if name == "EnhancedRAGConversationManager":
        from victor.verticals.contrib.rag.conversation_enhanced import (
            EnhancedRAGConversationManager,
        )

        return EnhancedRAGConversationManager

    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = [
    "RAGAssistant",
    "RAGAssistantDefinition",
    "Document",
    "DocumentChunk",
    "DocumentSearchResult",
    "DocumentStore",
    "DocumentStoreConfig",
    "DocumentChunker",
    "ChunkingConfig",
    "RAGPromptContributor",
    "RAGModeConfigProvider",
    "RAGCapabilityProvider",
    "RAGIngestTool",
    "RAGSearchTool",
    "RAGQueryTool",
    "RAGListTool",
    "RAGDeleteTool",
    "RAGStatsTool",
    "RAGSafetyRules",
    "EnhancedRAGSafetyExtension",
    "RAGContext",
    "EnhancedRAGConversationManager",
]
