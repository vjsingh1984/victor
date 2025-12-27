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
    ProviderProtocol,
    StreamChunk,
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
    Event,
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
from victor.framework.task import Task, TaskResult, TaskType
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
        RetryConfig,
        RetryExhaustedError,
        ResilientRetryStrategy,
        # Unified Retry Strategies
        ExponentialBackoffStrategy,
        FixedDelayStrategy,
        LinearBackoffStrategy,
        NoRetryStrategy,
        RetryContext,
        RetryExecutor,
        RetryOutcome,
        RetryResult,
        RetryStrategy,
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
        "RetryConfig",
        "RetryExhaustedError",
        "ResilientRetryStrategy",
        # Unified Retry Strategies
        "ExponentialBackoffStrategy",
        "FixedDelayStrategy",
        "LinearBackoffStrategy",
        "NoRetryStrategy",
        "RetryContext",
        "RetryExecutor",
        "RetryOutcome",
        "RetryResult",
        "RetryStrategy",
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

__all__ = (
    [
        # Core classes (the 5 concepts)
        "Agent",
        "Task",
        "Tools",
        "State",
        "Event",
        # Agent
        "ChatSession",
        # Task
        "TaskResult",
        "TaskType",
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
        "StreamChunk",
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
)

# Version of the framework API
__version__ = "0.2.0"
