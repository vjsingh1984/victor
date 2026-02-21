"""Victor Framework - Simplified API for AI coding agents.

This module provides a "golden path" API that covers 90% of use cases
with 5 core concepts:

1. **Agent** - Single entry point for creating agents
2. **Task** - What the agent should accomplish (TaskResult)
3. **Tools** - Available capabilities (ToolSet with presets)
4. **State** - Observable conversation state (Stage enum)
5. **Event** - Stream of observations (EventType enum)

Quick Start:
    from victor.framework import Agent

    # Simple use case
    agent = await Agent.create(provider="anthropic")
    result = await agent.run("Write a hello world function")
    print(result.content)

Streaming:
    async for event in agent.stream("Refactor this code"):
        if event.type == EventType.CONTENT:
            print(event.content, end="")
        elif event.type == EventType.TOOL_CALL:
            print(f"Using {event.tool_name}")

Tools:
    # Default tools (recommended)
    agent = await Agent.create(tools=ToolSet.default())

    # Minimal for simple tasks
    agent = await Agent.create(tools=ToolSet.minimal())

    # Full access to all 45+ tools
    agent = await Agent.create(tools=ToolSet.full())

State Observation:
    print(f"Stage: {agent.state.stage}")
    print(f"Tools used: {agent.state.tool_calls_used}/{agent.state.tool_budget}")

    agent.on_state_change(lambda old, new: print(f"{old.stage} -> {new.stage}"))

Escape Hatch:
    # Access full AgentOrchestrator for advanced use cases
    orchestrator = agent.get_orchestrator()
    # Now you have access to all 27+ internal components
"""

from victor.framework.agent import Agent, ChatSession
from victor.framework.config import AgentConfig
from victor.framework.protocols import (
    ChunkType,
    ConversationStateProtocol,
    MessagesProtocol,
    OrchestratorProtocol,
    OrchestratorStreamChunk,
    ProviderProtocol,
    StreamingProtocol,
    SystemPromptProtocol,
    ToolsProtocol,
    verify_protocol_conformance,
)
from victor.framework.errors import (
    AgentError,
    BudgetExhaustedError,
    CancellationError,
    ConfigurationError,
    ProviderError,
    StateTransitionError,
    ToolError,
)
from victor.framework.events import (
    AgentExecutionEvent,
    EventType,
    content_event,
    error_event,
    milestone_event,
    progress_event,
    stage_change_event,
    stream_end_event,
    stream_start_event,
    thinking_event,
    tool_call_event,
    tool_error_event,
    tool_result_event,
)
from victor.framework.state import Stage, State, StateHooks, StateObserver
from victor.framework.task import FrameworkTaskType, Task, TaskResult
from victor.framework.shim import FrameworkShim, get_vertical, list_verticals
from victor.framework.tools import ToolCategory, Tools, ToolSet, ToolsInput

# CQRS integration (optional - lazy loaded)
try:
    from victor.framework.cqrs_bridge import (
        CQRSBridge,
        FrameworkEventAdapter,
        ObservabilityToCQRSBridge,
        create_cqrs_bridge,
        create_event_adapter,
        cqrs_event_to_framework,
        framework_event_to_cqrs,
        framework_event_to_observability,
        observability_event_to_framework,
    )

    _CQRS_EXPORTS = [
        "CQRSBridge",
        "FrameworkEventAdapter",
        "ObservabilityToCQRSBridge",
        "create_cqrs_bridge",
        "create_event_adapter",
        "cqrs_event_to_framework",
        "framework_event_to_cqrs",
        "framework_event_to_observability",
        "observability_event_to_framework",
    ]
except ImportError:
    _CQRS_EXPORTS = []

# Event Registry (Phase 7.3 - Unified event conversion)
try:
    from victor.framework.event_registry import (
        BaseEventConverter,
        EventConverterProtocol,
        EventRegistry,
        EventTarget,
        convert_from_cqrs,
        convert_from_observability,
        convert_to_cqrs,
        convert_to_observability,
        get_event_registry,
    )

    _EVENT_REGISTRY_EXPORTS = [
        "BaseEventConverter",
        "EventConverterProtocol",
        "EventRegistry",
        "EventTarget",
        "convert_from_cqrs",
        "convert_from_observability",
        "convert_to_cqrs",
        "convert_to_observability",
        "get_event_registry",
    ]
except ImportError:
    _EVENT_REGISTRY_EXPORTS = []

# Agent Components (Phase 7.4 - Builder/Session/Bridge decomposition)
# Phase 8.3 - Added SessionLifecycleHooks, SessionMetrics for lifecycle management
try:
    from victor.framework.agent_components import (
        AgentBridge,
        AgentBuilder,
        AgentBuildOptions,
        AgentSession,
        BridgeConfiguration,
        BuilderPreset,
        SessionContext,
        SessionLifecycleHooks,
        SessionMetrics,
        SessionState,
        create_bridge,
        create_builder,
        create_session,
    )

    _AGENT_COMPONENTS_EXPORTS = [
        "AgentBridge",
        "AgentBuilder",
        "AgentBuildOptions",
        "AgentSession",
        "BridgeConfiguration",
        "BuilderPreset",
        "SessionContext",
        "SessionLifecycleHooks",
        "SessionMetrics",
        "SessionState",
        "create_bridge",
        "create_builder",
        "create_session",
    ]
except ImportError:
    _AGENT_COMPONENTS_EXPORTS = []

# Tool Configuration (Phase 7.5 - Externalized tool configuration)
try:
    from victor.framework.tool_config import (
        AirgappedFilter,
        CostTierFilter,
        SecurityFilter,
        ToolConfig,
        ToolConfigBuilder,
        ToolConfigEntry,
        ToolConfigMode,
        ToolConfigResult,
        ToolConfigurator,
        ToolFilterProtocol,
        configure_tools,
        configure_tools_from_toolset,
        get_tool_configurator,
    )

    # Note: ToolCategory is already exported from tools.py
    _TOOL_CONFIG_EXPORTS = [
        "AirgappedFilter",
        "CostTierFilter",
        "SecurityFilter",
        "ToolConfig",
        "ToolConfigBuilder",
        "ToolConfigEntry",
        "ToolConfigMode",
        "ToolConfigResult",
        "ToolConfigurator",
        "ToolFilterProtocol",
        "configure_tools",
        "configure_tools_from_toolset",
        "get_tool_configurator",
    ]
except ImportError:
    _TOOL_CONFIG_EXPORTS = []

# Service Provider (Phase 8 - DI Container Integration)
try:
    from victor.framework.service_provider import (
        AgentBuilderService,
        AgentSessionService,
        EventRegistryService,
        FrameworkScope,
        FrameworkServiceProvider,
        ToolConfiguratorService,
        configure_framework_services,
        create_builder,
        create_framework_scope,
    )

    _SERVICE_PROVIDER_EXPORTS = [
        "AgentBuilderService",
        "AgentSessionService",
        "EventRegistryService",
        "FrameworkScope",
        "FrameworkServiceProvider",
        "ToolConfiguratorService",
        "configure_framework_services",
        "create_builder",
        "create_framework_scope",
    ]
except ImportError:
    _SERVICE_PROVIDER_EXPORTS = []

# Resilience Facade (Phase 9.1 - Re-exports from providers/core modules)
try:
    from victor.framework.resilience import (
        # Circuit Breaker (Standalone)
        CircuitBreaker,
        CircuitBreakerError,
        CircuitBreakerRegistry,
        CircuitState,
        # Resilient Provider
        CircuitBreakerConfig,
        CircuitBreakerState,
        CircuitOpenError,
        ProviderUnavailableError,
        ResilientProvider,
        ProviderRetryConfig,
        RetryExhaustedError,
        ProviderRetryStrategy,
        # Unified Retry Strategies
        ExponentialBackoffStrategy,
        FixedDelayStrategy,
        LinearBackoffStrategy,
        NoRetryStrategy,
        RetryContext,
        RetryExecutor,
        RetryOutcome,
        RetryResult,
        BaseRetryStrategy,
        connection_retry_strategy,
        provider_retry_strategy,
        tool_retry_strategy,
        with_retry,
        with_retry_sync,
    )

    _RESILIENCE_EXPORTS = [
        # Circuit Breaker (Standalone)
        "CircuitBreaker",
        "CircuitBreakerError",
        "CircuitBreakerRegistry",
        "CircuitState",
        # Resilient Provider
        "CircuitBreakerConfig",
        "CircuitBreakerState",
        "CircuitOpenError",
        "ProviderUnavailableError",
        "ResilientProvider",
        "ProviderRetryConfig",
        "RetryExhaustedError",
        "ProviderRetryStrategy",
        # Unified Retry Strategies
        "ExponentialBackoffStrategy",
        "FixedDelayStrategy",
        "LinearBackoffStrategy",
        "NoRetryStrategy",
        "RetryContext",
        "RetryExecutor",
        "RetryOutcome",
        "RetryResult",
        "BaseRetryStrategy",
        "connection_retry_strategy",
        "provider_retry_strategy",
        "tool_retry_strategy",
        "with_retry",
        "with_retry_sync",
    ]
except ImportError:
    _RESILIENCE_EXPORTS = []

# Health Facade (Phase 9.2 - Re-exports from core/providers modules)
try:
    from victor.framework.health import (
        # Core Health Check System
        BaseHealthCheck,
        CacheHealthCheck,
        CallableHealthCheck,
        ComponentHealth,
        HealthCheckProtocol,
        HealthChecker,
        HealthReport,
        HealthStatus,
        MemoryHealthCheck,
        ProviderHealthCheck,
        ToolHealthCheck,
        create_default_health_checker,
        # Provider-Specific Health
        HealthCheckResult,
        ProviderHealthStatus,
        ProviderHealthChecker,
        ProviderHealthReport,
        get_provider_health_checker,
        reset_provider_health_checker,
    )

    _HEALTH_EXPORTS = [
        # Core Health Check System
        "BaseHealthCheck",
        "CacheHealthCheck",
        "CallableHealthCheck",
        "ComponentHealth",
        "HealthCheckProtocol",
        "HealthChecker",
        "HealthReport",
        "HealthStatus",
        "MemoryHealthCheck",
        "ProviderHealthCheck",
        "ToolHealthCheck",
        "create_default_health_checker",
        # Provider-Specific Health
        "HealthCheckResult",
        "ProviderHealthStatus",
        "ProviderHealthChecker",
        "ProviderHealthReport",
        "get_provider_health_checker",
        "reset_provider_health_checker",
    ]
except ImportError:
    _HEALTH_EXPORTS = []

# Metrics Facade (Phase 9.3 - Re-exports from observability/telemetry modules)
try:
    from victor.framework.metrics import (
        # Metrics System
        Counter,
        Gauge,
        Histogram,
        Metric,
        MetricLabels,
        MetricsCollector,
        MetricsRegistry,
        Timer,
        TimerContext,
        # Telemetry
        get_meter,
        get_tracer,
        is_telemetry_enabled,
        setup_opentelemetry,
    )

    _METRICS_EXPORTS = [
        # Metrics System
        "Counter",
        "Gauge",
        "Histogram",
        "Metric",
        "MetricLabels",
        "MetricsCollector",
        "MetricsRegistry",
        "Timer",
        "TimerContext",
        # Telemetry
        "get_meter",
        "get_tracer",
        "is_telemetry_enabled",
        "setup_opentelemetry",
    ]
except ImportError:
    _METRICS_EXPORTS = []

# Teams (Phase 4 - Multi-Agent Teams Exposure)
# NOTE: Canonical team types (TeamFormation, MemberResult, TeamResult, etc.)
# are now imported from victor.teams. Framework-specific types remain in
# victor.framework.teams.
try:
    from victor.teams import (
        MemberResult,
        TeamFormation,
        TeamResult,
    )
    from victor.framework.teams import (
        AgentTeam,
        TeamEvent,
        TeamEventType,
        TeamMemberSpec,
        member_complete_event,
        member_start_event,
        team_complete_event,
        team_start_event,
    )
    from victor.teams.types import (
        TeamConfig,
        TeamMember,
    )

    _TEAMS_EXPORTS = [
        "AgentTeam",
        "MemberResult",
        "TeamConfig",
        "TeamEvent",
        "TeamEventType",
        "TeamFormation",
        "TeamMember",
        "TeamMemberSpec",
        "TeamResult",
        "member_complete_event",
        "member_start_event",
        "team_complete_event",
        "team_start_event",
    ]
except ImportError:
    _TEAMS_EXPORTS = []

# Reinforcement Learning (Phase 5 - RL + Capability Registry)
try:
    from victor.framework.rl import (
        BaseLearner,
        LearnerStats,
        LearnerType,
        RLCoordinator,
        RLManager,
        RLOutcome,
        RLRecommendation,
        RLStats,
        create_outcome,
        get_rl_coordinator,
        record_tool_success,
    )

    _RL_EXPORTS = [
        "BaseLearner",
        "LearnerStats",
        "LearnerType",
        "RLCoordinator",
        "RLManager",
        "RLOutcome",
        "RLRecommendation",
        "RLStats",
        "create_outcome",
        "get_rl_coordinator",
        "record_tool_success",
    ]
except ImportError:
    _RL_EXPORTS = []

# Team Registry (Framework-level team spec management)
try:
    from victor.framework.team_registry import (
        TeamSpecRegistry,
        TeamSpecEntry,
        get_team_registry,
        register_team_spec,
        get_team_spec,
    )

    _TEAM_REGISTRY_EXPORTS = [
        "TeamSpecRegistry",
        "TeamSpecEntry",
        "get_team_registry",
        "register_team_spec",
        "get_team_spec",
    ]
except ImportError:
    _TEAM_REGISTRY_EXPORTS = []

# Graph Workflow Engine (LangGraph-compatible StateGraph)
try:
    from victor.framework.graph import (
        StateGraph,
        CompiledGraph,
        Node,
        Edge,
        EdgeType,
        FrameworkNodeStatus,
        GraphExecutionResult,
        GraphConfig,
        WorkflowCheckpoint,
        CheckpointerProtocol,
        MemoryCheckpointer,
        RLCheckpointerAdapter,
        StateProtocol,
        NodeFunctionProtocol,
        ConditionFunctionProtocol,
        END,
        START,
        create_graph,
    )

    _GRAPH_EXPORTS = [
        "StateGraph",
        "CompiledGraph",
        "Node",
        "Edge",
        "EdgeType",
        "FrameworkNodeStatus",
        "GraphExecutionResult",
        "GraphConfig",
        "WorkflowCheckpoint",
        "CheckpointerProtocol",
        "MemoryCheckpointer",
        "RLCheckpointerAdapter",  # Integrates with existing RL CheckpointStore
        "StateProtocol",
        "NodeFunctionProtocol",
        "ConditionFunctionProtocol",
        "END",
        "START",
        "create_graph",
    ]
except ImportError:
    _GRAPH_EXPORTS = []

# Module Loader (Shared infrastructure for dynamic module loading)
try:
    from victor.framework.module_loader import (
        CachedEntryPoints,
        DebouncedReloadTimer,
        DynamicModuleLoader,
        EntryPointCache,
        get_entry_point_cache,
    )

    _MODULE_LOADER_EXPORTS = [
        "CachedEntryPoints",
        "DebouncedReloadTimer",
        "DynamicModuleLoader",
        "EntryPointCache",
        "get_entry_point_cache",
    ]
except ImportError:
    _MODULE_LOADER_EXPORTS = []

# Capability Loader (Phase 4.4 - Dynamic capability loading for plugins)
try:
    from victor.framework.capability_loader import (
        CapabilityLoader,
        CapabilityEntry,
        CapabilityLoadError,
        capability,
        create_capability_loader,
        get_default_capability_loader,
    )

    _CAPABILITY_LOADER_EXPORTS = [
        "CapabilityLoader",
        "CapabilityEntry",
        "CapabilityLoadError",
        "capability",
        "create_capability_loader",
        "get_default_capability_loader",
    ]
except ImportError:
    _CAPABILITY_LOADER_EXPORTS = []

# Tool Naming (Canonical tool name enforcement)
try:
    from victor.framework.tool_naming import (
        CANONICAL_TO_ALIASES,
        TOOL_ALIASES,
        ToolNameEntry,
        ToolNames,
        canonicalize_dependencies,
        canonicalize_tool_dict,
        canonicalize_tool_list,
        canonicalize_tool_set,
        canonicalize_transitions,
        get_aliases,
        get_all_canonical_names,
        get_canonical_name,
        get_legacy_names_report,
        get_name_mapping,
        is_valid_tool_name,
        validate_tool_names,
    )

    _TOOL_NAMING_EXPORTS = [
        "CANONICAL_TO_ALIASES",
        "TOOL_ALIASES",
        "ToolNameEntry",
        "ToolNames",
        "canonicalize_dependencies",
        "canonicalize_tool_dict",
        "canonicalize_tool_list",
        "canonicalize_tool_set",
        "canonicalize_transitions",
        "get_aliases",
        "get_all_canonical_names",
        "get_canonical_name",
        "get_legacy_names_report",
        "get_name_mapping",
        "is_valid_tool_name",
        "validate_tool_names",
    ]
except ImportError:
    _TOOL_NAMING_EXPORTS = []

# Framework Middleware (common middleware for all verticals)
try:
    from victor.framework.middleware import (
        GitSafetyMiddleware,
        LoggingMiddleware,
        MetricsMiddleware,
        SecretMaskingMiddleware,
        ToolMetrics,
    )

    _MIDDLEWARE_EXPORTS = [
        "GitSafetyMiddleware",
        "LoggingMiddleware",
        "MetricsMiddleware",
        "SecretMaskingMiddleware",
        "ToolMetrics",
    ]
except ImportError:
    _MIDDLEWARE_EXPORTS = []

# Task Type Registry (Unified task type definitions with hints/budgets)
try:
    from victor.framework.task_types import (
        TaskCategory,
        TaskTypeDefinition,
        TaskTypeRegistry,
        get_task_budget,
        get_task_hint,
        get_task_type_registry,
        register_vertical_task_type,
        register_coding_task_types,
        register_data_analysis_task_types,
        register_devops_task_types,
        register_research_task_types,
        setup_vertical_task_types,
    )

    _TASK_TYPES_EXPORTS = [
        "TaskCategory",
        "TaskTypeDefinition",
        "TaskTypeRegistry",
        "get_task_budget",
        "get_task_hint",
        "get_task_type_registry",
        "register_vertical_task_type",
        "register_coding_task_types",
        "register_data_analysis_task_types",
        "register_devops_task_types",
        "register_research_task_types",
        "setup_vertical_task_types",
    ]
except ImportError:
    _TASK_TYPES_EXPORTS = []

# Checkpointing is handled by victor.agent.rl.checkpoint_store.CheckpointStore
# which provides versioning, rollback, and diff capabilities.
# WorkflowExecutor and StateGraph integrate with it via optional parameters.
_CHECKPOINTER_EXPORTS = []

# Multi-agent teams are handled by AgentTeam with extended TeamMember
# supporting backstory, memory, cache attributes (CrewAI-compatible features).
# Use victor.framework.teams.AgentTeam instead of a separate Crew class.
_CREW_EXPORTS = []

__all__ = (
    [
        # Core classes (the 5 concepts)
        "Agent",
        "Task",
        "Tools",
        "State",
        "AgentExecutionEvent",
        # Agent
        "ChatSession",
        # Task
        "TaskResult",
        "FrameworkTaskType",
        # Tools
        "ToolSet",
        "ToolCategory",
        "ToolsInput",
        # State
        "Stage",
        "StateHooks",
        "StateObserver",
        # Protocols (Phase 7 - Framework-Orchestrator Interface)
        "OrchestratorProtocol",
        "ConversationStateProtocol",
        "ProviderProtocol",
        "ToolsProtocol",
        "SystemPromptProtocol",
        "MessagesProtocol",
        "StreamingProtocol",
        "OrchestratorStreamChunk",
        "ChunkType",
        "verify_protocol_conformance",
        # Events
        "EventType",
        "content_event",
        "thinking_event",
        "tool_call_event",
        "tool_result_event",
        "tool_error_event",
        "stage_change_event",
        "stream_start_event",
        "stream_end_event",
        "error_event",
        "progress_event",
        "milestone_event",
        # Config
        "AgentConfig",
        # Shim (backward compatibility)
        "FrameworkShim",
        "get_vertical",
        "list_verticals",
        # Errors
        "AgentError",
        "ProviderError",
        "ToolError",
        "ConfigurationError",
        "BudgetExhaustedError",
        "CancellationError",
        "StateTransitionError",
    ]
    + _CQRS_EXPORTS
    + _EVENT_REGISTRY_EXPORTS
    + _AGENT_COMPONENTS_EXPORTS
    + _TOOL_CONFIG_EXPORTS
    + _SERVICE_PROVIDER_EXPORTS
    + _RESILIENCE_EXPORTS
    + _HEALTH_EXPORTS
    + _METRICS_EXPORTS
    + _TEAMS_EXPORTS
    + _RL_EXPORTS
    + _TEAM_REGISTRY_EXPORTS
    + _GRAPH_EXPORTS
    + _CHECKPOINTER_EXPORTS
    + _CREW_EXPORTS
    + _MODULE_LOADER_EXPORTS
    + _CAPABILITY_LOADER_EXPORTS
    + _TOOL_NAMING_EXPORTS
    + _MIDDLEWARE_EXPORTS
    + _TASK_TYPES_EXPORTS
    + ["discover"]  # Capability discovery function
)


def discover() -> dict:
    """Discover all Victor framework capabilities programmatically.

    Returns a dictionary containing all available tools, verticals, personas,
    teams, chains, handlers, task types, providers, and events.

    Example:
        from victor.framework import discover

        caps = discover()
        print(f"Available tools: {len(caps['tools'])}")
        print(f"Available verticals: {caps['verticals']}")

    Returns:
        dict: Capability manifest with keys:
            - tools: List of tool names
            - tool_categories: List of category names
            - verticals: List of vertical names
            - personas: List of persona names
            - teams: List of team names
            - chains: List of chain/workflow names
            - handlers: List of handler names
            - task_types: List of task type names
            - providers: List of provider names
            - events: List of event type names
            - summary: Aggregated counts
    """
    from victor.ui.commands.capabilities import get_capability_discovery

    discovery = get_capability_discovery()
    manifest = discovery.discover_all()
    return manifest.to_dict()


# Vertical Integration (Phase 3.1 - Step Handlers)
try:
    from victor.framework.vertical_integration import (
        IntegrationResult,
        OrchestratorVerticalProtocol,
        VerticalIntegrationPipeline,
        apply_vertical,
        create_integration_pipeline,
        create_integration_pipeline_with_handlers,
    )
    from victor.framework.vertical_cache_policy import (
        InMemoryLRUVerticalIntegrationCachePolicy,
        VerticalIntegrationCachePolicy,
    )
    from victor.framework.step_handlers import (
        BaseStepHandler,
        ConfigStepHandler,
        ContextStepHandler,
        ExtensionsStepHandler,
        FrameworkStepHandler,
        MiddlewareStepHandler,
        PromptStepHandler,
        SafetyStepHandler,
        StepHandlerProtocol,
        StepHandlerRegistry,
        ToolStepHandler,
    )

    _VERTICAL_INTEGRATION_EXPORTS = [
        # Integration pipeline
        "IntegrationResult",
        "OrchestratorVerticalProtocol",
        "VerticalIntegrationPipeline",
        "apply_vertical",
        "create_integration_pipeline",
        "create_integration_pipeline_with_handlers",
        "VerticalIntegrationCachePolicy",
        "InMemoryLRUVerticalIntegrationCachePolicy",
        # Step handlers (Phase 3.1)
        "BaseStepHandler",
        "ConfigStepHandler",
        "ContextStepHandler",
        "ExtensionsStepHandler",
        "FrameworkStepHandler",
        "MiddlewareStepHandler",
        "PromptStepHandler",
        "SafetyStepHandler",
        "StepHandlerProtocol",
        "StepHandlerRegistry",
        "ToolStepHandler",
    ]

    __all__ = list(__all__) + _VERTICAL_INTEGRATION_EXPORTS
except ImportError:
    pass

# HITL Protocol (Human-in-the-Loop)
try:
    from victor.framework.hitl import (
        ApprovalHandler,
        ApprovalRequest,
        ApprovalStatus,
        HITLCheckpoint,
        HITLController,
    )

    _HITL_EXPORTS = [
        "ApprovalHandler",
        "ApprovalRequest",
        "ApprovalStatus",
        "HITLCheckpoint",
        "HITLController",
    ]

    __all__ = list(__all__) + _HITL_EXPORTS
except ImportError:
    pass

# Persona System
try:
    from victor.framework.personas import (
        Persona,
        PERSONA_REGISTRY,
        get_persona,
        register_persona,
        list_personas,
    )

    _PERSONA_EXPORTS = [
        "Persona",
        "PERSONA_REGISTRY",
        "get_persona",
        "register_persona",
        "list_personas",
    ]

    __all__ = list(__all__) + _PERSONA_EXPORTS
except ImportError:
    pass

# Agent Protocols (Multi-Agent Orchestration)
try:
    from victor.framework.agent_protocols import (
        AgentCapability,
        AgentMessage,
        IAgentPersona,
        IAgentRole,
        ITeamCoordinator,
        ITeamMember,
        MessageType,
        TeamFormation as AgentTeamFormation,  # Alias to avoid conflict with teams.py
    )

    _AGENT_PROTOCOLS_EXPORTS = [
        "AgentCapability",
        "AgentMessage",
        "IAgentPersona",
        "IAgentRole",
        "ITeamCoordinator",
        "ITeamMember",
        "MessageType",
        "AgentTeamFormation",
    ]

    __all__ = list(__all__) + _AGENT_PROTOCOLS_EXPORTS
except ImportError:
    pass

# Agent Roles (Built-in role definitions)
try:
    from victor.framework.agent_roles import (
        ExecutorRole,
        ManagerRole,
        ResearcherRole,
        ReviewerRole,
        ROLE_REGISTRY,
        get_role,
    )

    _AGENT_ROLES_EXPORTS = [
        "ExecutorRole",
        "ManagerRole",
        "ResearcherRole",
        "ReviewerRole",
        "ROLE_REGISTRY",
        "get_role",
    ]

    __all__ = list(__all__) + _AGENT_ROLES_EXPORTS
except ImportError:
    pass

# Team Coordinator (Multi-Agent Coordination)
# NOTE: FrameworkTeamCoordinator has been removed.
# Use victor.teams.create_coordinator() instead.
# See victor/teams/MIGRATION_GUIDE.md for migration instructions.
try:
    from victor.teams import (
        MemberResult as CoordinatorMemberResult,  # Alias to avoid conflict
        TeamResult as CoordinatorTeamResult,  # Alias to avoid conflict
    )

    _TEAM_COORDINATOR_EXPORTS = [
        "CoordinatorMemberResult",
        "CoordinatorTeamResult",
    ]

    __all__ = list(__all__) + _TEAM_COORDINATOR_EXPORTS
except ImportError:
    pass

# Chain Registry (Cross-vertical chain discovery)
try:
    from victor.framework.chain_registry import (
        ChainMetadata,
        ChainRegistry,
        get_chain,
        get_chain_registry,
        register_chain,
    )

    _CHAIN_REGISTRY_EXPORTS = [
        "ChainMetadata",
        "ChainRegistry",
        "get_chain",
        "get_chain_registry",
        "register_chain",
    ]

    __all__ = list(__all__) + _CHAIN_REGISTRY_EXPORTS
except ImportError:
    pass

# Persona Registry (Cross-vertical persona discovery)
try:
    from victor.framework.persona_registry import (
        PersonaRegistry,
        PersonaSpec,
        get_persona_registry,
        get_persona_spec,
        register_persona_spec,
    )

    _PERSONA_REGISTRY_EXPORTS = [
        "PersonaRegistry",
        "PersonaSpec",
        "get_persona_registry",
        "get_persona_spec",
        "register_persona_spec",
    ]

    __all__ = list(__all__) + _PERSONA_REGISTRY_EXPORTS
except ImportError:
    pass

# Prompt Builder (WS-B - Consolidated prompt building)
try:
    from victor.framework.prompt_builder import (
        PromptBuilder,
        PromptSection,
        ToolHint,
        create_coding_prompt_builder,
        create_data_analysis_prompt_builder,
        create_devops_prompt_builder,
        create_research_prompt_builder,
    )

    _PROMPT_BUILDER_EXPORTS = [
        "PromptBuilder",
        "PromptSection",
        "ToolHint",
        "create_coding_prompt_builder",
        "create_data_analysis_prompt_builder",
        "create_devops_prompt_builder",
        "create_research_prompt_builder",
    ]

    __all__ = list(__all__) + _PROMPT_BUILDER_EXPORTS
except ImportError:
    pass

# Prompt Sections (WS-B - Reusable prompt templates)
try:
    from victor.framework.prompt_sections import (
        # Grounding
        GROUNDING_RULES_EXTENDED,
        GROUNDING_RULES_MINIMAL,
        PARALLEL_READ_GUIDANCE,
        # Coding
        CODING_GUIDELINES,
        CODING_IDENTITY,
        CODING_TOOL_USAGE,
        # DevOps
        DEVOPS_COMMON_PITFALLS,
        DEVOPS_GROUNDING,
        DEVOPS_IDENTITY,
        DEVOPS_SECURITY_CHECKLIST,
        # Research
        RESEARCH_GROUNDING,
        RESEARCH_IDENTITY,
        RESEARCH_QUALITY_CHECKLIST,
        RESEARCH_SOURCE_HIERARCHY,
        # Data Analysis
        DATA_ANALYSIS_GROUNDING,
        DATA_ANALYSIS_IDENTITY,
        DATA_ANALYSIS_LIBRARIES,
        DATA_ANALYSIS_OPERATIONS,
    )

    _PROMPT_SECTIONS_EXPORTS = [
        # Grounding
        "GROUNDING_RULES_EXTENDED",
        "GROUNDING_RULES_MINIMAL",
        "PARALLEL_READ_GUIDANCE",
        # Coding
        "CODING_GUIDELINES",
        "CODING_IDENTITY",
        "CODING_TOOL_USAGE",
        # DevOps
        "DEVOPS_COMMON_PITFALLS",
        "DEVOPS_GROUNDING",
        "DEVOPS_IDENTITY",
        "DEVOPS_SECURITY_CHECKLIST",
        # Research
        "RESEARCH_GROUNDING",
        "RESEARCH_IDENTITY",
        "RESEARCH_QUALITY_CHECKLIST",
        "RESEARCH_SOURCE_HIERARCHY",
        # Data Analysis
        "DATA_ANALYSIS_GROUNDING",
        "DATA_ANALYSIS_IDENTITY",
        "DATA_ANALYSIS_LIBRARIES",
        "DATA_ANALYSIS_OPERATIONS",
    ]

    __all__ = list(__all__) + _PROMPT_SECTIONS_EXPORTS
except ImportError:
    pass

# Stage Manager (Framework-level stage management)
try:
    from victor.framework.stage_manager import (
        StageDefinition,
        StageManager,
        StageManagerConfig,
        StageManagerProtocol,
        StageTransition,
        create_stage_manager,
        get_coding_stages,
        get_data_analysis_stages,
        get_research_stages,
    )

    _STAGE_MANAGER_EXPORTS = [
        "StageDefinition",
        "StageManager",
        "StageManagerConfig",
        "StageManagerProtocol",
        "StageTransition",
        "create_stage_manager",
        "get_coding_stages",
        "get_data_analysis_stages",
        "get_research_stages",
    ]
except ImportError:
    _STAGE_MANAGER_EXPORTS = []

# LSP Protocols (Cross-vertical language intelligence)
try:
    from victor.framework.lsp_protocols import (
        LSPCompletionItem,
        LSPDiagnostic,
        LSPHoverInfo,
        LSPLocation,
        LSPPoolProtocol,
        LSPPosition,
        LSPRange,
        LSPServiceProtocol,
        LSPSymbol,
    )

    _LSP_PROTOCOLS_EXPORTS = [
        "LSPCompletionItem",
        "LSPDiagnostic",
        "LSPHoverInfo",
        "LSPLocation",
        "LSPPoolProtocol",
        "LSPPosition",
        "LSPRange",
        "LSPServiceProtocol",
        "LSPSymbol",
    ]

    __all__ = list(__all__) + _LSP_PROTOCOLS_EXPORTS
except ImportError:
    pass

# Add Stage Manager exports
if _STAGE_MANAGER_EXPORTS:
    __all__ = list(__all__) + _STAGE_MANAGER_EXPORTS

# Workflow Engine (High-level workflow execution facade)
try:
    from victor.framework.workflow_engine import (
        WorkflowExecutionResult,
        WorkflowEngine,
        WorkflowEngineConfig,
        WorkflowEngineProtocol,
        WorkflowEvent,
        create_workflow_engine,
        run_graph_workflow,
        run_yaml_workflow,
    )

    _WORKFLOW_ENGINE_EXPORTS = [
        "WorkflowExecutionResult",
        "WorkflowEngine",
        "WorkflowEngineConfig",
        "WorkflowEngineProtocol",
        "WorkflowEvent",
        "create_workflow_engine",
        "run_graph_workflow",
        "run_yaml_workflow",
    ]
    __all__ = list(__all__) + _WORKFLOW_ENGINE_EXPORTS
except ImportError:
    _WORKFLOW_ENGINE_EXPORTS = []

# Observability (Phase 4 - Unified metrics and dashboard)
try:
    from victor.framework.observability import (
        # Manager
        ObservabilityManager,
        ObservabilityConfig,
        DashboardData,
        # Metrics
        Metric,
        MetricType,
        MetricLabel,
        CounterMetric,
        GaugeMetric,
        HistogramBucket,
        HistogramMetric,
        SummaryMetric,
        MetricsSnapshot,
        MetricsCollection,
        # Protocols
        MetricSource,
        # Constants
        MetricNames,
    )

    _OBSERVABILITY_EXPORTS = [
        # Manager
        "ObservabilityManager",
        "ObservabilityConfig",
        "DashboardData",
        # Metrics
        "Metric",
        "MetricType",
        "MetricLabel",
        "CounterMetric",
        "GaugeMetric",
        "HistogramBucket",
        "HistogramMetric",
        "SummaryMetric",
        "MetricsSnapshot",
        "MetricsCollection",
        # Protocols
        "MetricSource",
        # Constants
        "MetricNames",
    ]
    __all__ = list(__all__) + _OBSERVABILITY_EXPORTS
except ImportError:
    _OBSERVABILITY_EXPORTS = []

# Version sourced from pyproject.toml via importlib.metadata
try:
    from importlib.metadata import version as _get_version

    __version__ = _get_version("victor-ai")
except Exception:
    __version__ = "0.0.0"
