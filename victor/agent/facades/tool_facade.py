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

"""Tool domain facade for orchestrator decomposition.

Groups tool registry, execution pipeline, selection strategy, caching,
deduplication, output formatting, and budgeting components behind a
single interface.

This facade wraps already-initialized components from the orchestrator,
providing a coherent grouping without changing initialization ordering.
The orchestrator delegates property access through this facade.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ToolFacade:
    """Groups tool registry, execution, and management components.

    Satisfies ``ToolFacadeProtocol`` structurally.  The orchestrator creates
    this facade after all tool-domain components are initialized, passing
    references to the already-created instances.

    Components managed:
        - tools: SharedToolRegistry for tool storage and lookup
        - tool_pipeline: ToolPipeline for execution orchestration
        - tool_executor: ToolExecutor for individual tool invocation
        - tool_selector: ToolSelector for semantic + keyword selection
        - tool_cache: Optional tool result cache
        - tool_graph: ToolDependencyGraph for planning
        - tool_registrar: ToolRegistrar for dynamic tool discovery
        - tool_budget: Maximum tool calls per session
        - tool_output_formatter: LLM-context-aware output formatting
        - deduplication_tracker: Optional tracker for preventing duplicates
        - argument_normalizer: Handles malformed tool arguments
        - parallel_executor: Concurrent independent tool calls
        - safety_checker: Validates tool execution safety
        - middleware_chain: Vertical-specific tool processing
        - tool_access_controller: Unified tool access control
        - budget_manager: Unified budget tracking
        - search_router: Search query routing
        - semantic_selector: Optional semantic tool selection
    """

    def __init__(
        self,
        *,
        tools: Any,
        tool_pipeline: Any,
        tool_executor: Any,
        tool_selector: Any,
        tool_cache: Optional[Any] = None,
        tool_graph: Optional[Any] = None,
        tool_registrar: Optional[Any] = None,
        tool_budget: int = 50,
        tool_output_formatter: Optional[Any] = None,
        deduplication_tracker: Optional[Any] = None,
        argument_normalizer: Optional[Any] = None,
        parallel_executor: Optional[Any] = None,
        safety_checker: Optional[Any] = None,
        auto_committer: Optional[Any] = None,
        middleware_chain: Optional[Any] = None,
        code_correction_middleware: Optional[Any] = None,
        tool_access_controller: Optional[Any] = None,
        budget_manager: Optional[Any] = None,
        search_router: Optional[Any] = None,
        semantic_selector: Optional[Any] = None,
        task_classifier: Optional[Any] = None,
        sequence_tracker: Optional[Any] = None,
        unified_tracker: Optional[Any] = None,
        plugin_manager: Optional[Any] = None,
    ) -> None:
        self._tools = tools
        self._tool_pipeline = tool_pipeline
        self._tool_executor = tool_executor
        self._tool_selector = tool_selector
        self._tool_cache = tool_cache
        self._tool_graph = tool_graph
        self._tool_registrar = tool_registrar
        self._tool_budget = tool_budget
        self._tool_output_formatter = tool_output_formatter
        self._deduplication_tracker = deduplication_tracker
        self._argument_normalizer = argument_normalizer
        self._parallel_executor = parallel_executor
        self._safety_checker = safety_checker
        self._auto_committer = auto_committer
        self._middleware_chain = middleware_chain
        self._code_correction_middleware = code_correction_middleware
        self._tool_access_controller = tool_access_controller
        self._budget_manager = budget_manager
        self._search_router = search_router
        self._semantic_selector = semantic_selector
        self._task_classifier = task_classifier
        self._sequence_tracker = sequence_tracker
        self._unified_tracker = unified_tracker
        self._plugin_manager = plugin_manager

        logger.debug(
            "ToolFacade initialized (cache=%s, dedup=%s, semantic=%s, budget=%d)",
            tool_cache is not None,
            deduplication_tracker is not None,
            semantic_selector is not None,
            tool_budget,
        )

    # ------------------------------------------------------------------
    # Properties (satisfy ToolFacadeProtocol)
    # ------------------------------------------------------------------

    @property
    def tools(self) -> Any:
        """Tool registry (SharedToolRegistry instance)."""
        return self._tools

    @property
    def tool_registry(self) -> Any:
        """Alias for tools (backward compatibility)."""
        return self._tools

    @property
    def tool_pipeline(self) -> Any:
        """ToolPipeline for execution orchestration."""
        return self._tool_pipeline

    @property
    def tool_executor(self) -> Any:
        """ToolExecutor for individual tool invocation."""
        return self._tool_executor

    @property
    def tool_selector(self) -> Any:
        """Tool selection strategy (semantic + keyword)."""
        return self._tool_selector

    @property
    def tool_cache(self) -> Optional[Any]:
        """Optional tool result cache."""
        return self._tool_cache

    @property
    def tool_graph(self) -> Optional[Any]:
        """ToolDependencyGraph for planning."""
        return self._tool_graph

    @property
    def tool_registrar(self) -> Optional[Any]:
        """ToolRegistrar for dynamic tool discovery."""
        return self._tool_registrar

    @property
    def tool_budget(self) -> int:
        """Maximum tool calls per session."""
        return self._tool_budget

    @tool_budget.setter
    def tool_budget(self, value: int) -> None:
        """Update the tool budget."""
        self._tool_budget = value

    @property
    def tool_output_formatter(self) -> Optional[Any]:
        """ToolOutputFormatter for LLM-context-aware output formatting."""
        return self._tool_output_formatter

    @property
    def deduplication_tracker(self) -> Optional[Any]:
        """Optional tracker for preventing redundant tool calls."""
        return self._deduplication_tracker

    @property
    def argument_normalizer(self) -> Optional[Any]:
        """ArgumentNormalizer for handling malformed tool arguments."""
        return self._argument_normalizer

    @property
    def parallel_executor(self) -> Optional[Any]:
        """ParallelToolExecutor for concurrent independent calls."""
        return self._parallel_executor

    @property
    def safety_checker(self) -> Optional[Any]:
        """SafetyChecker for validating tool execution."""
        return self._safety_checker

    @property
    def auto_committer(self) -> Optional[Any]:
        """AutoCommitter for AI-assisted commits."""
        return self._auto_committer

    @property
    def middleware_chain(self) -> Optional[Any]:
        """Middleware chain for vertical-specific tool processing."""
        return self._middleware_chain

    @middleware_chain.setter
    def middleware_chain(self, value: Any) -> None:
        """Update the middleware chain."""
        self._middleware_chain = value

    @property
    def code_correction_middleware(self) -> Optional[Any]:
        """Code correction middleware for automatic validation/fixing."""
        return self._code_correction_middleware

    @property
    def tool_access_controller(self) -> Optional[Any]:
        """ToolAccessController for unified tool access control."""
        return self._tool_access_controller

    @property
    def budget_manager(self) -> Optional[Any]:
        """BudgetManager for unified budget tracking."""
        return self._budget_manager

    @property
    def search_router(self) -> Optional[Any]:
        """SearchRouter for query routing."""
        return self._search_router

    @property
    def semantic_selector(self) -> Optional[Any]:
        """Optional SemanticToolSelector."""
        return self._semantic_selector

    @property
    def task_classifier(self) -> Optional[Any]:
        """Complexity classifier."""
        return self._task_classifier

    @property
    def sequence_tracker(self) -> Optional[Any]:
        """ToolSequenceTracker for intelligent next-tool suggestions."""
        return self._sequence_tracker

    @property
    def unified_tracker(self) -> Optional[Any]:
        """UnifiedTaskTracker for task progress and loop detection."""
        return self._unified_tracker

    @property
    def plugin_manager(self) -> Optional[Any]:
        """ToolPluginRegistry for extensible tools."""
        return self._plugin_manager
