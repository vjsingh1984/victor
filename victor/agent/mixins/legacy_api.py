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

"""Legacy API mixin for backward compatibility.

This mixin contains deprecated methods from the AgentOrchestrator that
are maintained for backward compatibility but will be removed in v0.7.0.

All methods in this mixin issue deprecation warnings when called and
provide clear migration paths to the new APIs.

Migration Timeline:
- Deprecated in: v0.5.1
- Removal target: v0.7.0
- Grace period: Users should migrate during this period

To migrate away from this mixin:
1. Run your code with deprecation warnings enabled
2. Replace deprecated method calls with recommended alternatives
3. Remove dependency on LegacyAPIMixin from your classes
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from victor.agent.decorators import deprecated

if TYPE_CHECKING:
    from victor.core.state import ConversationStage
    from victor.core.events.vertical_context import VerticalContext
    from victor.agent.stream_handler import StreamMetrics
    from victor.evaluation.protocol import TokenUsage

logger = logging.getLogger(__name__)


class LegacyAPIMixin:
    """Mixin containing deprecated orchestrator methods for backward compatibility.

    This mixin consolidates 500+ lines of legacy code into a single location,
    making the main orchestrator class cleaner and more maintainable.

    All methods issue deprecation warnings and provide clear migration paths.

    Type Attributes:
        The following attributes are expected to be provided by classes using this mixin:
        - _mode_workflow_team_coordinator: Optional[ModeWorkflowTeamCoordinator]
        - _configuration_manager: Optional[ConfigurationManager]
        - _tool_access_controller: Optional[ToolAccessController]
        - _project_context: Optional[ProjectContext]
        - _vertical_integration_adapter: Optional[VerticalIntegrationAdapter]
        - _team_coordinator: Optional[TeamCoordinator]
        - _metrics_coordinator: Optional[MetricsCoordinator]
        - _cumulative_token_usage: Dict[str, int]
        - _state_coordinator: Optional[StateCoordinator]
        - _context_compactor: Optional[ContextCompactor]
        - _usage_analytics: Optional[UsageAnalytics]
        - _sequence_tracker: Optional[SequenceTracker]
        - _code_correction_middleware: Optional[CodeCorrectionMiddleware]
        - _safety_checker: Optional[SafetyChecker]
        - _auto_committer: Optional[AutoCommitter]
        - search_router: Optional[SearchRouter]
        - unified_tracker: Optional[UnifiedTaskTracker]
        - provider_name: str
        - model: str
        - _provider_manager: Optional[ProviderManager]
        - _tool_access_coordinator: Optional[ToolAccessConfigCoordinator]
        - prompt_builder: Optional[SystemPromptBuilder]
        - tools: Optional[ToolRegistryProtocol]
        - conversation: Optional[ConversationStore]
        - _search_coordinator: Optional[SearchCoordinator]
        - tool_selector: Optional[ToolSelector]
        - conversation_state: Optional[ConversationState]
        - _vertical_context: Optional[VerticalContext]
        - mode_workflow_team_coordinator: Optional[ModeWorkflowTeamCoordinator]
    """

    # Type stubs for attributes expected on classes using this mixin
    # These are annotated but not initialized, as the owning class provides them
    _mode_workflow_team_coordinator: Optional[Any]
    _configuration_manager: Optional[Any]
    _tool_access_controller: Optional[Any]
    _project_context: Optional[Any]
    _vertical_integration_adapter: Optional[Any]
    _team_coordinator: Optional[Any]
    _metrics_coordinator: Optional[Any]
    _cumulative_token_usage: Dict[str, int]
    _state_coordinator: Optional[Any]
    _context_compactor: Optional[Any]
    _usage_analytics: Optional[Any]
    _sequence_tracker: Optional[Any]
    _code_correction_middleware: Optional[Any]
    _safety_checker: Optional[Any]
    _auto_committer: Optional[Any]
    search_router: Optional[Any]
    unified_tracker: Optional[Any]
    provider_name: str
    model: str
    _provider_manager: Optional[Any]
    _tool_access_coordinator: Optional[Any]
    prompt_builder: Optional[Any]
    tools: Optional[Any]
    conversation: Optional[Any]
    _search_coordinator: Optional[Any]
    tool_selector: Optional[Any]
    conversation_state: Optional[Any]
    _vertical_context: Optional[Any]
    _vertical_middleware: List[Any]
    _vertical_safety_patterns: List[Any]
    mode_workflow_team_coordinator: Optional[Any]

    # =========================================================================
    # Category 1: Vertical Configuration Methods (Deprecated)
    # These methods have been replaced by protocol-based APIs
    # =========================================================================

    @deprecated(
        version="0.5.1",
        replacement="vertical_context property or VerticalContext.set_context()",
        remove_version="0.7.0",
        reason="Direct setter violates encapsulation - use VerticalContext protocol",
    )
    def set_vertical_context(self, context: "VerticalContext") -> None:
        """Set the vertical context (OrchestratorVerticalProtocol).

        This replaces direct assignment to _vertical_context and provides
        a proper API for framework integration.

        DEPRECATED: Use VerticalContext.set_context() instead.

        Args:
            context: VerticalContext to set
        """
        self._vertical_context = context

        # Sync coordinator with new vertical context (if already initialized)
        if self._mode_workflow_team_coordinator is not None:
            self._mode_workflow_team_coordinator.set_vertical_context(context)
            logger.debug(f"Coordinator synced with vertical context: {context.vertical_name}")

        logger.debug(f"Vertical context set: {context.vertical_name}")

    @deprecated(
        version="0.5.1",
        replacement="ConfigurationManager.set_tiered_tool_config()",
        remove_version="0.7.0",
        reason="ConfigurationManager centralizes all tool config management",
    )
    def set_tiered_tool_config(self, config: Any) -> None:
        """Set tiered tool configuration (Phase 1: Gap fix).

        DEPRECATED: ConfigurationManager handles this internally.

        Args:
            config: TieredToolConfig from the active vertical
        """
        # Delegate to ConfigurationManager for centralized management
        if self._configuration_manager is not None:
            self._configuration_manager.set_tiered_tool_config(
                config=config,
                vertical_context=self._vertical_context,
                tool_access_controller=self._tool_access_controller,
            )
            logger.debug("Tiered tool config set via ConfigurationManager")

    @deprecated(
        version="0.5.1",
        replacement="set_project_root() from victor.config.settings",
        remove_version="0.7.0",
        reason="Project context should be managed globally, not per orchestrator",
    )
    def set_workspace(self, workspace_dir: Path) -> None:
        """Set the workspace directory for task execution.

        DEPRECATED: Use set_project_root() from settings module instead.

        Args:
            workspace_dir: Path to the workspace directory
        """
        from victor.config.settings import set_project_root

        # Update global project root
        set_project_root(workspace_dir)
        logger.info(f"Project root set to: {workspace_dir}")

        # Update project context if available
        if hasattr(self, "_project_context") and self._project_context:
            from victor.context.project_context import ProjectContext

            self._project_context = ProjectContext(str(workspace_dir))
            logger.debug(f"Project context updated: {workspace_dir}")

    # =========================================================================
    # Category 2: Vertical Storage Protocol Methods (Deprecated)
    # These methods have been replaced by coordinator-based APIs
    # =========================================================================

    @deprecated(
        version="0.5.1",
        replacement="VerticalIntegrationAdapter.apply_middleware()",
        remove_version="0.7.0",
        reason="Middleware management is now centralized in VerticalIntegrationAdapter",
    )
    def apply_vertical_middleware(self, middleware_list: List[Any]) -> None:
        """Apply middleware from vertical extensions.

        DEPRECATED: VerticalIntegrationAdapter handles this automatically.

        Args:
            middleware_list: List of MiddlewareProtocol implementations.
        """
        if self._vertical_integration_adapter is not None:
            self._vertical_integration_adapter.apply_middleware(middleware_list)

    @deprecated(
        version="0.5.1",
        replacement="VerticalIntegrationAdapter.apply_safety_patterns()",
        remove_version="0.7.0",
        reason="Safety pattern management is now centralized",
    )
    def apply_vertical_safety_patterns(self, patterns: List[Any]) -> None:
        """Apply safety patterns from vertical extensions.

        DEPRECATED: VerticalIntegrationAdapter handles this automatically.

        Args:
            patterns: List of SafetyPattern objects.
        """
        if self._vertical_integration_adapter is not None:
            self._vertical_integration_adapter.apply_safety_patterns(patterns)

    @deprecated(
        version="0.5.1",
        replacement="StateCoordinator or direct property access",
        remove_version="0.7.0",
        reason="Direct middleware chain access breaks encapsulation",
    )
    def get_middleware_chain(self) -> Optional[Any]:
        """Get the middleware chain for tool execution.

        DEPRECATED: Use StateCoordinator or access via properties.

        Returns:
            MiddlewareChain instance or None if not initialized.
        """
        return getattr(self, "_middleware_chain", None)

    @deprecated(
        version="0.5.1",
        replacement="VerticalIntegrationAdapter for middleware management",
        remove_version="0.7.0",
        reason="Direct storage access violates DIP principle",
    )
    def set_middleware(self, middleware: List[Any]) -> None:
        """Store middleware configuration.

        DEPRECATED: VerticalIntegrationAdapter manages middleware internally.

        Args:
            middleware: List of MiddlewareProtocol implementations
        """
        self._vertical_middleware = middleware

    @deprecated(
        version="0.5.1",
        replacement="VerticalIntegrationAdapter for middleware management",
        remove_version="0.7.0",
        reason="Direct storage access violates DIP principle",
    )
    def get_middleware(self) -> List[Any]:
        """Retrieve middleware configuration.

        DEPRECATED: VerticalIntegrationAdapter manages middleware internally.

        Returns:
            List of middleware instances, or empty list if not set
        """
        return getattr(self, "_vertical_middleware", [])

    @deprecated(
        version="0.5.1",
        replacement="VerticalIntegrationAdapter for safety pattern management",
        remove_version="0.7.0",
        reason="Direct storage access violates DIP principle",
    )
    def set_safety_patterns(self, patterns: List[Any]) -> None:
        """Store safety patterns.

        DEPRECATED: VerticalIntegrationAdapter manages patterns internally.

        Args:
            patterns: List of SafetyPattern instances from vertical extensions
        """
        self._vertical_safety_patterns = patterns

    @deprecated(
        version="0.5.1",
        replacement="VerticalIntegrationAdapter for safety pattern management",
        remove_version="0.7.0",
        reason="Direct storage access violates DIP principle",
    )
    def get_safety_patterns(self) -> List[Any]:
        """Retrieve safety patterns.

        DEPRECATED: VerticalIntegrationAdapter manages patterns internally.

        Returns:
            List of safety pattern instances, or empty list if not set
        """
        return getattr(self, "_vertical_safety_patterns", [])

    @deprecated(
        version="0.5.1",
        replacement="TeamCoordinator.set_team_specs() or TeamCoordinator.get_team_specs()",
        remove_version="0.7.0",
        reason="Team specification management is now in TeamCoordinator",
    )
    def set_team_specs(self, specs: Dict[str, Any]) -> None:
        """Store team specifications.

        DEPRECATED: TeamCoordinator manages team specs internally.

        Args:
            specs: Dictionary mapping team names to TeamSpec instances
        """
        # Ensure team coordinator is initialized
        if self._team_coordinator is None:
            _ = self.mode_workflow_team_coordinator

        if self._team_coordinator is not None:
            self._team_coordinator.set_team_specs(specs)

    @deprecated(
        version="0.5.1",
        replacement="TeamCoordinator.get_team_specs()",
        remove_version="0.7.0",
        reason="Team specification management is now in TeamCoordinator",
    )
    def get_team_specs(self) -> Dict[str, Any]:
        """Retrieve team specifications.

        DEPRECATED: TeamCoordinator manages team specs internally.

        Returns:
            Dictionary of team specs, or empty dict if not set
        """
        # Ensure team coordinator is initialized
        if self._team_coordinator is None:
            _ = self.mode_workflow_team_coordinator

        if self._team_coordinator is not None:
            return self._team_coordinator.get_team_specs()  # type: ignore[no-any-return]
        return {}

    # =========================================================================
    # Category 3: Metrics and Analytics Methods (Deprecated)
    # These methods have been replaced by MetricsCoordinator
    # =========================================================================

    @deprecated(
        version="0.5.1",
        replacement="MetricsCoordinator.get_tool_usage_stats()",
        remove_version="0.7.0",
        reason="All metrics should be accessed through MetricsCoordinator",
    )
    def get_tool_usage_stats(self) -> Dict[str, Any]:
        """Get comprehensive tool usage statistics.

        DEPRECATED: Use MetricsCoordinator.get_tool_usage_stats() instead.

        Returns:
            Dictionary with usage analytics
        """
        if self._metrics_coordinator is not None and self.conversation_state is not None:
            return self._metrics_coordinator.get_tool_usage_stats(  # type: ignore[no-any-return]
                conversation_state_summary=self.conversation_state.get_state_summary()
            )
        return {}

    @deprecated(
        version="0.5.1",
        replacement="MetricsCoordinator.get_token_usage()",
        remove_version="0.7.0",
        reason="All metrics should be accessed through MetricsCoordinator",
    )
    def get_token_usage(self) -> "TokenUsage":
        """Get cumulative token usage for evaluation tracking.

        DEPRECATED: Use MetricsCoordinator.get_token_usage() instead.

        Returns:
            TokenUsage dataclass with input/output/total token counts
        """
        if self._metrics_coordinator is not None:
            return self._metrics_coordinator.get_token_usage()  # type: ignore[no-any-return]
        # Return default TokenUsage if coordinator not available
        from victor.evaluation.protocol import TokenUsage

        return TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0)

    @deprecated(
        version="0.5.1",
        replacement="MetricsCoordinator.reset_token_usage()",
        remove_version="0.7.0",
        reason="All metrics should be managed through MetricsCoordinator",
    )
    def reset_token_usage(self) -> None:
        """Reset cumulative token usage tracking.

        DEPRECATED: Use MetricsCoordinator.reset_token_usage() instead.
        """
        # Reset through coordinator (which updates the cumulative dict)
        for key in self._cumulative_token_usage:
            self._cumulative_token_usage[key] = 0

    @deprecated(
        version="0.5.1",
        replacement="MetricsCoordinator.get_last_stream_metrics()",
        remove_version="0.7.0",
        reason="All metrics should be accessed through MetricsCoordinator",
    )
    def get_last_stream_metrics(self) -> Optional["StreamMetrics"]:
        """Get metrics from the last streaming session.

        DEPRECATED: Use MetricsCoordinator.get_last_stream_metrics() instead.

        Returns:
            StreamMetrics from the last session or None
        """
        if self._metrics_coordinator is not None:
            return self._metrics_coordinator.get_last_stream_metrics()  # type: ignore[no-any-return]
        return None

    @deprecated(
        version="0.5.1",
        replacement="MetricsCoordinator.get_streaming_metrics_summary()",
        remove_version="0.7.0",
        reason="All metrics should be accessed through MetricsCoordinator",
    )
    def get_streaming_metrics_summary(self) -> Optional[Dict[str, Any]]:
        """Get comprehensive streaming metrics summary.

        DEPRECATED: Use MetricsCoordinator.get_streaming_metrics_summary() instead.

        Returns:
            Dictionary with aggregated metrics or None
        """
        if self._metrics_coordinator is not None:
            return self._metrics_coordinator.get_streaming_metrics_summary()  # type: ignore[no-any-return]
        return None

    @deprecated(
        version="0.5.1",
        replacement="MetricsCoordinator.get_streaming_metrics_history()",
        remove_version="0.7.0",
        reason="All metrics should be accessed through MetricsCoordinator",
    )
    def get_streaming_metrics_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent streaming metrics history.

        DEPRECATED: Use MetricsCoordinator.get_streaming_metrics_history() instead.

        Args:
            limit: Maximum number of recent metrics to return

        Returns:
            List of recent metrics dictionaries
        """
        if self._metrics_coordinator is not None:
            return self._metrics_coordinator.get_streaming_metrics_history(limit)  # type: ignore[no-any-return]
        return []

    @deprecated(
        version="0.5.1",
        replacement="MetricsCoordinator.get_session_cost_summary()",
        remove_version="0.7.0",
        reason="All cost tracking should use MetricsCoordinator",
    )
    def get_session_cost_summary(self) -> Dict[str, Any]:
        """Get session cost summary.

        DEPRECATED: Use MetricsCoordinator.get_session_cost_summary() instead.

        Returns:
            Dictionary with session cost statistics
        """
        if self._metrics_coordinator is not None:
            return self._metrics_coordinator.get_session_cost_summary()  # type: ignore[no-any-return]
        return {}

    @deprecated(
        version="0.5.1",
        replacement="MetricsCoordinator.get_session_cost_formatted()",
        remove_version="0.7.0",
        reason="All cost tracking should use MetricsCoordinator",
    )
    def get_session_cost_formatted(self) -> str:
        """Get formatted session cost string.

        DEPRECATED: Use MetricsCoordinator.get_session_cost_formatted() instead.

        Returns:
            Cost string like "$0.0123" or "cost n/a"
        """
        if self._metrics_coordinator is not None:
            return self._metrics_coordinator.get_session_cost_formatted()  # type: ignore[no-any-return]
        return "cost n/a"

    @deprecated(
        version="0.5.1",
        replacement="MetricsCoordinator.export_session_costs()",
        remove_version="0.7.0",
        reason="All cost tracking should use MetricsCoordinator",
    )
    def export_session_costs(self, path: str, format: str = "json") -> None:
        """Export session costs to file.

        DEPRECATED: Use MetricsCoordinator.export_session_costs() instead.

        Args:
            path: Output file path
            format: Export format ("json" or "csv")
        """
        if self._metrics_coordinator is not None:
            self._metrics_coordinator.export_session_costs(path, format)

    # =========================================================================
    # Category 4: State and Conversation Methods (Deprecated)
    # These methods have been replaced by StateCoordinator
    # =========================================================================

    @deprecated(
        version="0.5.1",
        replacement="StateCoordinator.get_stage()",
        remove_version="0.7.0",
        reason="All state access should go through StateCoordinator",
    )
    def get_conversation_stage(self) -> "ConversationStage":
        """Get the current conversation stage.

        DEPRECATED: Use StateCoordinator.get_stage() instead.

        Returns:
            Current ConversationStage enum value
        """
        from victor.core.state import ConversationStage

        if self._state_coordinator is not None:
            # StateCoordinator returns stage name, convert to enum
            stage_name = self._state_coordinator.get_stage()
            if stage_name:
                return ConversationStage[stage_name]
        return ConversationStage.INITIAL

    @deprecated(
        version="0.5.1",
        replacement="StateCoordinator.get_stage_tools()",
        remove_version="0.7.0",
        reason="All state access should go through StateCoordinator",
    )
    def get_stage_recommended_tools(self) -> Set[str]:
        """Get tools recommended for the current conversation stage.

        DEPRECATED: Use StateCoordinator.get_stage_tools() instead.

        Returns:
            Set of tool names recommended for current stage
        """
        if self._state_coordinator is not None:
            return self._state_coordinator.get_stage_tools()  # type: ignore[no-any-return]
        return set()

    @deprecated(
        version="0.5.1",
        replacement="AnalyticsCoordinator.get_optimization_status()",
        remove_version="0.7.0",
        reason="Optimization status reporting is now in AnalyticsCoordinator",
    )
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all integrated optimization components.

        DEPRECATED: Use AnalyticsCoordinator.get_optimization_status() instead.

        Returns:
            Dictionary with component status and statistics
        """
        from victor.agent.coordinators.analytics_coordinator import AnalyticsCoordinator

        coordinator = AnalyticsCoordinator(exporters=[])
        return coordinator.get_optimization_status(
            context_compactor=self._context_compactor,
            usage_analytics=self._usage_analytics,
            sequence_tracker=self._sequence_tracker,
            code_correction_middleware=self._code_correction_middleware,
            safety_checker=self._safety_checker,
            auto_committer=self._auto_committer,
            search_router=self.search_router,
        )

    @deprecated(
        version="0.5.1",
        replacement="StateCoordinator.observed_files property",
        remove_version="0.7.0",
        reason="All state access should go through StateCoordinator",
    )
    def get_observed_files(self) -> Set[str]:
        """Get files observed/read during conversation.

        DEPRECATED: Use StateCoordinator.observed_files instead.

        Returns:
            Set of absolute file paths
        """
        if self._state_coordinator is not None:
            return self._state_coordinator.observed_files  # type: ignore[no-any-return]
        return set()

    @deprecated(
        version="0.5.1",
        replacement="conversation_state.state.modified_files",
        remove_version="0.7.0",
        reason="Direct access to conversation state is preferred",
    )
    def get_modified_files(self) -> Set[str]:
        """Get files modified during conversation.

        DEPRECATED: Access conversation_state.state.modified_files directly.

        Returns:
            Set of absolute file paths
        """
        if self.conversation_state and hasattr(self.conversation_state, "state"):
            return set(getattr(self.conversation_state.state, "modified_files", []))
        return set()

    # =========================================================================
    # Category 5: Protocol Method Implementations (Deprecated)
    # These methods implement protocols but should use coordinator delegates
    # =========================================================================

    @deprecated(
        version="0.5.1",
        replacement="unified_tracker.tool_calls_used or UnifiedTaskTracker",
        remove_version="0.7.0",
        reason="Tracker access should be through UnifiedTaskTracker protocol",
    )
    def get_tool_calls_count(self) -> int:
        """Get total tool calls made (protocol method).

        DEPRECATED: Use unified_tracker.tool_calls_used instead.

        Returns:
            Non-negative count of tool calls in this session
        """
        if self.unified_tracker:
            return self.unified_tracker.tool_calls_used  # type: ignore[no-any-return]
        return getattr(self, "tool_calls_used", 0)

    @deprecated(
        version="0.5.1",
        replacement="unified_tracker.tool_budget or UnifiedTaskTracker",
        remove_version="0.7.0",
        reason="Tracker access should be through UnifiedTaskTracker protocol",
    )
    def get_tool_budget(self) -> int:
        """Get tool call budget (protocol method).

        DEPRECATED: Use unified_tracker.tool_budget instead.

        Returns:
            Maximum allowed tool calls
        """
        if self.unified_tracker:
            return self.unified_tracker.tool_budget  # type: ignore[no-any-return]
        return getattr(self, "tool_budget", 50)

    @deprecated(
        version="0.5.1",
        replacement="unified_tracker.iteration_count or UnifiedTaskTracker",
        remove_version="0.7.0",
        reason="Tracker access should be through UnifiedTaskTracker protocol",
    )
    def get_iteration_count(self) -> int:
        """Get current agent loop iteration count (protocol method).

        DEPRECATED: Use unified_tracker.iteration_count instead.

        Returns:
            Non-negative iteration count
        """
        if self.unified_tracker:
            return self.unified_tracker.iteration_count  # type: ignore[no-any-return]
        return 0

    @deprecated(
        version="0.5.1",
        replacement="unified_tracker.max_iterations or UnifiedTaskTracker",
        remove_version="0.7.0",
        reason="Tracker access should be through UnifiedTaskTracker protocol",
    )
    def get_max_iterations(self) -> int:
        """Get maximum allowed iterations (protocol method).

        DEPRECATED: Use unified_tracker.max_iterations instead.

        Returns:
            Max iteration limit
        """
        if self.unified_tracker:
            return self.unified_tracker.max_iterations  # type: ignore[no-any-return]
        return 25

    # =========================================================================
    # Category 6: Provider and Model Methods (Deprecated)
    # Properties replaced by direct attribute access
    # =========================================================================

    @deprecated(
        version="0.5.1",
        replacement="orchestrator.provider_name attribute",
        remove_version="0.7.0",
        reason="Property is unnecessary - use direct attribute access",
    )
    def current_provider(self) -> str:
        """Get current provider name (protocol property).

        DEPRECATED: Use orchestrator.provider_name instead.

        Returns:
            Provider identifier (e.g., "anthropic", "openai")
        """
        return self.provider_name

    @deprecated(
        version="0.5.1",
        replacement="orchestrator.model attribute",
        remove_version="0.7.0",
        reason="Property is unnecessary - use direct attribute access",
    )
    def current_model(self) -> str:
        """Get current model name (protocol property).

        DEPRECATED: Use orchestrator.model instead.

        Returns:
            Model identifier
        """
        return self.model

    @deprecated(
        version="0.5.1",
        replacement="ProviderManager.get_info() or provider_name + model attributes",
        remove_version="0.7.0",
        reason="Provider info should come from ProviderManager",
    )
    def get_current_provider_info(self) -> Dict[str, Any]:
        """Get information about the current provider and model.

        DEPRECATED: Use ProviderManager.get_info() instead.

        Returns:
            Dictionary with provider/model info and capabilities
        """
        # Get base info from ProviderManager
        info = {}
        if self._provider_manager is not None:
            info = self._provider_manager.get_info()

        # Add orchestrator-specific runtime state
        info["tool_budget"] = self.get_tool_budget()
        info["tool_calls_used"] = self.get_tool_calls_count()

        return info

    # =========================================================================
    # Category 7: Tool Access Methods (Deprecated)
    # Replaced by ToolAccessConfigCoordinator
    # =========================================================================

    @deprecated(
        version="0.5.1",
        replacement="tools.list_tools() or ToolRegistryProtocol",
        remove_version="0.7.0",
        reason="Tool registry access should go through ToolRegistryProtocol",
    )
    def get_available_tools(self) -> Set[str]:
        """Get all registered tool names (protocol method).

        DEPRECATED: Use tools.list_tools() instead.

        Returns:
            Set of tool names available in registry
        """
        if self.tools:
            return set(self.tools.list_tools())
        return set()

    @deprecated(
        version="0.5.1",
        replacement="ToolAccessConfigCoordinator.is_tool_enabled()",
        remove_version="0.7.0",
        reason="Tool access control is now in ToolAccessConfigCoordinator",
    )
    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a specific tool is enabled (protocol method).

        DEPRECATED: Use ToolAccessConfigCoordinator.is_tool_enabled() instead.

        Args:
            tool_name: Name of tool to check

        Returns:
            True if tool is enabled
        """
        if self._tool_access_coordinator is not None:
            return self._tool_access_coordinator.is_tool_enabled(tool_name)  # type: ignore[no-any-return]
        return False

    # =========================================================================
    # Category 8: System Prompt Methods (Deprecated)
    # Replaced by prompt builder protocols
    # =========================================================================

    @deprecated(
        version="0.5.1",
        replacement="prompt_builder.build() or SystemPromptBuilder",
        remove_version="0.7.0",
        reason="Prompt building should use SystemPromptBuilder protocol",
    )
    def get_system_prompt(self) -> str:
        """Get current system prompt (protocol method).

        DEPRECATED: Use prompt_builder.build() instead.

        Returns:
            Complete system prompt string
        """
        if self.prompt_builder:
            return self.prompt_builder.build()  # type: ignore[no-any-return]
        return ""

    @deprecated(
        version="0.5.1",
        replacement="prompt_builder.set_custom_prompt()",
        remove_version="0.7.0",
        reason="Prompt customization should use prompt builder methods",
    )
    def set_system_prompt(self, prompt: str) -> None:
        """Set custom system prompt (protocol method).

        DEPRECATED: Use prompt_builder.set_custom_prompt() instead.

        Args:
            prompt: New system prompt (replaces existing)
        """
        if self.prompt_builder and hasattr(self.prompt_builder, "set_custom_prompt"):
            self.prompt_builder.set_custom_prompt(prompt)

    @deprecated(
        version="0.5.1",
        replacement="prompt_builder.append_content()",
        remove_version="0.7.0",
        reason="Prompt customization should use prompt builder methods",
    )
    def append_to_system_prompt(self, content: str) -> None:
        """Append content to system prompt (protocol method).

        DEPRECATED: Use prompt_builder.append_content() instead.

        Args:
            content: Content to append
        """
        current = self.get_system_prompt()
        self.set_system_prompt(current + "\n\n" + content)

    # =========================================================================
    # Category 9: Message Access Methods (Deprecated)
    # Replaced by ConversationStore protocols
    # =========================================================================

    @deprecated(
        version="0.5.1",
        replacement="conversation.messages or ConversationStoreProtocol",
        remove_version="0.7.0",
        reason="Message access should go through ConversationStore protocol",
    )
    def get_messages(self) -> List[Dict[str, Any]]:
        """Get conversation messages (protocol method).

        DEPRECATED: Use conversation.messages instead.

        Returns:
            List of message dictionaries
        """
        if self.conversation is not None:
            return [{"role": m.role, "content": m.content} for m in self.conversation.messages]
        return []

    @deprecated(
        version="0.5.1",
        replacement="len(conversation.messages) or ConversationStoreProtocol",
        remove_version="0.7.0",
        reason="Message access should go through ConversationStore protocol",
    )
    def get_message_count(self) -> int:
        """Get message count (protocol method).

        DEPRECATED: Use len(conversation.messages) instead.

        Returns:
            Number of messages in conversation
        """
        if self.conversation is not None:
            return len(self.conversation.messages)
        return 0

    # =========================================================================
    # Category 10: Search Routing Methods (Deprecated)
    # Replaced by SearchCoordinator
    # =========================================================================

    @deprecated(
        version="0.5.1",
        replacement="SearchCoordinator.route_search_query()",
        remove_version="0.7.0",
        reason="Search routing is now in SearchCoordinator",
    )
    def route_search_query(self, query: str) -> Dict[str, Any]:
        """Route a search query to the optimal search tool using SearchRouter.

        DEPRECATED: Use SearchCoordinator.route_search_query() instead.

        Args:
            query: The search query

        Returns:
            Dictionary with routing recommendation
        """
        if self._search_coordinator is not None:
            return self._search_coordinator.route_search_query(query)  # type: ignore[no-any-return]
        return {"tool": "unknown", "confidence": 0.0}

    @deprecated(
        version="0.5.1",
        replacement="SearchCoordinator.get_recommended_search_tool()",
        remove_version="0.7.0",
        reason="Search routing is now in SearchCoordinator",
    )
    def get_recommended_search_tool(self, query: str) -> str:
        """Get the recommended search tool name for a query.

        DEPRECATED: Use SearchCoordinator.get_recommended_search_tool() instead.

        Args:
            query: The search query

        Returns:
            Tool name: "code_search", "semantic_code_search", or "both"
        """
        if self._search_coordinator is not None:
            return self._search_coordinator.get_recommended_search_tool(query)  # type: ignore[no-any-return]
        return "code_search"

    # =========================================================================
    # Category 11: Health Check Methods (Deprecated)
    # Replaced by dedicated health check protocols
    # =========================================================================

    @deprecated(
        version="0.5.1",
        replacement="ToolSelector.get_health() or dedicated health check protocols",
        remove_version="0.7.0",
        reason="Health checks should use dedicated health check protocols",
    )
    def check_tool_selector_health(self) -> Dict[str, Any]:
        """Check health of the tool selector subsystem.

        DEPRECATED: Use tool_selector.get_health() instead.

        Returns:
            Health status dictionary
        """
        health = {
            "status": "healthy",
            "tool_selector": self.tool_selector.__class__.__name__,
            "tool_selector_initialized": self.tool_selector is not None,
        }

        # Check if semantic selector is ready
        if hasattr(self, "_semantic_selector"):
            health["semantic_selector_ready"] = (
                hasattr(self._semantic_selector, "_is_ready") and self._semantic_selector._is_ready
            )

        # Check embeddings initialization status
        if self.tool_selector is not None and hasattr(
            self.tool_selector, "_embeddings_initialized"
        ):
            health["embeddings_initialized"] = self.tool_selector._embeddings_initialized
        elif hasattr(self, "_semantic_selector"):
            health["embeddings_initialized"] = getattr(
                self._semantic_selector, "_embeddings_initialized", False
            )

        # Tool registry health
        if self.tools:
            health["tool_registry_initialized"] = self.tools.is_initialized()
            health["available_tools_count"] = len(self.tools.list_tools())

        return health


__all__ = ["LegacyAPIMixin"]
