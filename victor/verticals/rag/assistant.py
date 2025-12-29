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

"""RAGAssistant - Retrieval-Augmented Generation vertical.

This module defines the RAGAssistant vertical showcasing a complete RAG
implementation with document ingestion, vector search, and query generation.

Features:
- Document ingestion from multiple formats (PDF, Markdown, Text, Code)
- LanceDB vector storage (embedded, no server)
- Hybrid search combining vector + full-text
- Semantic chunking with configurable overlap
- Interactive TUI for document management
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from victor.verticals.base import StageDefinition, VerticalBase, VerticalConfig
from victor.verticals.protocols import (
    MiddlewareProtocol,
    SafetyExtensionProtocol,
    PromptContributorProtocol,
    ModeConfigProviderProtocol,
    ToolDependencyProviderProtocol,
    TieredToolConfig,
    VerticalExtensions,
)


class RAGAssistant(VerticalBase):
    """Retrieval-Augmented Generation assistant vertical.

    A complete RAG implementation demonstrating:
    - Document ingestion and indexing
    - Vector search with LanceDB
    - Query processing with context retrieval
    - Source attribution and citations

    Example:
        from victor.verticals.rag import RAGAssistant

        config = RAGAssistant.get_config()
        agent = await Agent.create(
            tools=config.tools,
            vertical=RAGAssistant,
        )
    """

    name = "rag"
    description = "Retrieval-Augmented Generation assistant for document Q&A"
    version = "1.0.0"

    @classmethod
    def get_tools(cls) -> List[str]:
        """Get tools for RAG operations.

        Returns:
            List of RAG-specific tool names
        """
        return [
            # RAG-specific tools
            "rag_ingest",      # Ingest documents into the store
            "rag_search",      # Search for relevant chunks
            "rag_query",       # Query with context retrieval
            "rag_list",        # List indexed documents
            "rag_delete",      # Delete documents
            "rag_stats",       # Get store statistics
            # Filesystem for document access
            "read",
            "ls",
            # Web for fetching web content
            "web_fetch",
            # Shell for document processing
            "shell",
        ]

    @classmethod
    def get_system_prompt(cls) -> str:
        """Get RAG-focused system prompt.

        Returns:
            System prompt optimized for RAG operations
        """
        return """You are Victor, a Retrieval-Augmented Generation (RAG) assistant.

Your capabilities:
- Ingest documents from files (PDF, Markdown, Text, Code)
- Index documents with semantic embeddings using LanceDB
- Search for relevant information using hybrid search
- Answer questions with source citations
- Manage the document knowledge base

Guidelines:
- Always search the knowledge base before answering questions
- Cite sources by referencing document names and page/section numbers
- If information isn't in the knowledge base, say so clearly
- Use rag_ingest to add new documents when requested
- Use rag_search for exploratory queries
- Use rag_query for direct Q&A with automatic context retrieval

Workflow:
1. For questions: Use rag_query to search and get relevant context
2. For exploration: Use rag_search to find related chunks
3. For adding docs: Use rag_ingest with file paths or URLs
4. Always cite your sources in answers

Example interaction:
User: "What does the documentation say about authentication?"
You: [Use rag_query tool with query="authentication"]
     Then synthesize the results with citations.
"""

    @classmethod
    def get_stages(cls) -> Dict[str, StageDefinition]:
        """Get RAG-specific workflow stages.

        Returns:
            Stage definitions for RAG workflow
        """
        return {
            "INITIAL": StageDefinition(
                name="INITIAL",
                description="Ready to accept RAG queries",
                tools={"rag_search", "rag_query", "rag_list", "rag_stats"},
                next_stages={"INGESTING", "SEARCHING", "QUERYING"},
            ),
            "INGESTING": StageDefinition(
                name="INGESTING",
                description="Ingesting documents into knowledge base",
                tools={"rag_ingest", "read", "ls", "web_fetch"},
                next_stages={"INITIAL", "SEARCHING"},
            ),
            "SEARCHING": StageDefinition(
                name="SEARCHING",
                description="Searching knowledge base",
                tools={"rag_search", "rag_query"},
                next_stages={"INITIAL", "QUERYING", "SYNTHESIZING"},
            ),
            "QUERYING": StageDefinition(
                name="QUERYING",
                description="Processing query with retrieved context",
                tools={"rag_query"},
                next_stages={"SYNTHESIZING", "SEARCHING"},
            ),
            "SYNTHESIZING": StageDefinition(
                name="SYNTHESIZING",
                description="Synthesizing answer from retrieved context",
                tools=set(),  # LLM response only
                next_stages={"INITIAL"},
            ),
        }

    @classmethod
    def get_tiered_tool_config(cls) -> TieredToolConfig:
        """Get tiered tool configuration for RAG.

        Returns:
            Tool configuration with tiers
        """
        return TieredToolConfig(
            always_enabled=[
                "rag_search",
                "rag_query",
                "rag_list",
                "rag_stats",
            ],
            stage_tools={
                "INGESTING": ["rag_ingest", "read", "ls", "web_fetch"],
                "SEARCHING": ["rag_search"],
                "QUERYING": ["rag_query"],
            },
            cost_tiers={
                "rag_search": "low",
                "rag_query": "low",
                "rag_list": "free",
                "rag_stats": "free",
                "rag_ingest": "medium",
                "rag_delete": "low",
            },
        )

    @classmethod
    def get_provider_hints(cls) -> Dict[str, Any]:
        """Get provider hints for RAG.

        RAG works best with models that follow instructions well.

        Returns:
            Provider hints dictionary
        """
        return {
            "preferred_providers": ["anthropic", "openai", "google"],
            "min_context_window": 8000,  # Need context for retrieved chunks
            "features": ["tool_calling"],
            "temperature": 0.3,  # Lower temperature for factual answers
        }

    @classmethod
    def get_evaluation_criteria(cls) -> List[str]:
        """Get evaluation criteria for RAG.

        Returns:
            List of criteria for evaluating RAG performance
        """
        return [
            "Answer is grounded in retrieved documents",
            "Sources are properly cited",
            "No hallucination of facts not in documents",
            "Relevant documents were retrieved",
            "Answer is coherent and well-structured",
        ]

    @classmethod
    def get_extensions(cls) -> VerticalExtensions:
        """Get RAG vertical extensions.

        Returns:
            Extension implementations for framework integration
        """
        # Import extension implementations
        from victor.verticals.rag.prompts import RAGPromptContributor
        from victor.verticals.rag.mode_config import RAGModeConfigProvider

        return VerticalExtensions(
            middleware=[],  # No special middleware for RAG
            safety_extensions=[],  # No special safety for RAG
            prompt_contributors=[RAGPromptContributor()],
            mode_config_provider=RAGModeConfigProvider(),
            tool_dependency_provider=None,
            workflow_provider=None,
            service_provider=None,
        )
