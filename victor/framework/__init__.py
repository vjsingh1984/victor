"""Victor Framework - Simplified API for AI coding agents.

This module provides a "golden path" API that covers 90% of use cases
with 5 core concepts:

1. **Agent** - Single entry point for creating agents
2. **Task** - What the agent should accomplish (TaskResult)
3. **Tools** - Available capabilities (ToolSet with presets)
4. **State** - Observable conversation state (ConversationStage enum)
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

LAZY LOADING: Optional subsystems (CQRS, resilience, health, metrics, services, etc.)
are loaded on-demand via __getattr__ to reduce startup time. This improves Victor's
import performance by ~0.4s (25% reduction). Optional subsystems are only imported
when actually accessed (e.g., when using specific features like circuit breakers).
"""

from __future__ import annotations

from typing import Any

# Core imports (always loaded - these are the 5 essential concepts)
from victor.framework.agent import Agent, ChatSession
from victor.framework.config import AgentConfig
from victor.framework.protocols import (
    ChatResult,
    ChatResultProtocol,
    ChatStateProtocol,
    ChunkType,
    ConversationStateProtocol,
    MessagesProtocol,
    MutableChatState,
    OrchestratorProtocol,
    OrchestratorStreamChunk,
    ProviderProtocol,
    StreamingProtocol,
    SystemPromptProtocol,
    ToolsProtocol,
    verify_protocol_conformance,
    WorkflowChatProtocol,
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
from victor.framework.state import (
    ConversationStage,
    State,
    StateHooks,
    StateObserver,
)
from victor.framework.task import FrameworkTaskType, Task, TaskResult
from victor.framework.checkpoint import CheckpointBackend
from victor.framework.shim import FrameworkShim, get_vertical, list_verticals
from victor.framework.tools import ToolCategory, Tools, ToolSet, ToolsInput
from victor.framework.prompt_builder import (
    PromptBuilder,
    PromptSection,
    ToolHint,
    create_coding_prompt_builder,
    create_data_analysis_prompt_builder,
    create_devops_prompt_builder,
    create_research_prompt_builder,
)
from victor.framework.prompt_builder_template import PromptBuilderTemplate
from victor.framework.prompt_sections import (
    GroundingSection,
    RuleSection,
    ChecklistSection,
    VerticalPromptSection,
)
from victor.framework.prompt_sections_legacy import (
    # Grounding rules
    GROUNDING_RULES_MINIMAL,
    GROUNDING_RULES_EXTENDED,
    PARALLEL_READ_GUIDANCE,
    # Coding sections
    CODING_IDENTITY,
    CODING_GUIDELINES,
    CODING_TOOL_USAGE,
    # DevOps sections
    DEVOPS_IDENTITY,
    DEVOPS_SECURITY_CHECKLIST,
    DEVOPS_COMMON_PITFALLS,
    DEVOPS_GROUNDING,
    # Research sections
    RESEARCH_IDENTITY,
    RESEARCH_QUALITY_CHECKLIST,
    RESEARCH_SOURCE_HIERARCHY,
    RESEARCH_GROUNDING,
    # Data Analysis sections
    DATA_ANALYSIS_IDENTITY,
    DATA_ANALYSIS_LIBRARIES,
    DATA_ANALYSIS_OPERATIONS,
    DATA_ANALYSIS_GROUNDING,
    # Task hints
    TASK_HINT_CODE_GENERATION,
    TASK_HINT_CREATE_SIMPLE,
    TASK_HINT_CREATE,
    TASK_HINT_EDIT,
    TASK_HINT_SEARCH,
    TASK_HINT_ACTION,
    TASK_HINT_ANALYSIS_DEEP,
    TASK_HINT_ANALYZE,
    TASK_HINT_DESIGN,
    TASK_HINT_GENERAL,
    TASK_HINT_REFACTOR,
    TASK_HINT_DEBUG,
    TASK_HINT_TEST,
    # Provider-specific guidance
    DEEPSEEK_TOOL_EFFICIENCY,
    XAI_GROK_GUIDANCE,
    OLLAMA_STRICT_GUIDANCE,
)

# Lazy-loaded optional subsystems
# These 81 subsystems are only loaded when accessed via __getattr__
# This improves startup time by ~0.4s (25% reduction)

# Module mappings for lazy loading
_LAZY_MODULES = {
    # CQRS integration
    "CQRSBridge": "victor.framework.cqrs_bridge",
    "FrameworkEventAdapter": "victor.framework.cqrs_bridge",
    "ObservabilityToCQRSBridge": "victor.framework.cqrs_bridge",
    "create_cqrs_bridge": "victor.framework.cqrs_bridge",
    "create_event_adapter": "victor.framework.cqrs_bridge",
    "cqrs_event_to_framework": "victor.framework.cqrs_bridge",
    "framework_event_to_cqrs": "victor.framework.cqrs_bridge",
    "framework_event_to_observability": "victor.framework.cqrs_bridge",
    "observability_event_to_framework": "victor.framework.cqrs_bridge",
    # Event Registry
    "BaseEventConverter": "victor.framework.event_registry",
    "EventConverterProtocol": "victor.framework.event_registry",
    "EventRegistry": "victor.framework.event_registry",
    "EventTarget": "victor.framework.event_registry",
    "convert_from_cqrs": "victor.framework.event_registry",
    "convert_from_observability": "victor.framework.event_registry",
    "convert_to_cqrs": "victor.framework.event_registry",
    "convert_to_observability": "victor.framework.event_registry",
    "get_event_registry": "victor.framework.event_registry",
    # Agent Components
    "AgentBridge": "victor.framework.agent_components",
    "AgentBuilder": "victor.framework.agent_components",
    "AgentBuildOptions": "victor.framework.agent_components",
    "AgentSession": "victor.framework.agent_components",
    "BridgeConfiguration": "victor.framework.agent_components",
    "BuilderPreset": "victor.framework.agent_components",
    "SessionContext": "victor.framework.agent_components",
    "SessionLifecycleHooks": "victor.framework.agent_components",
    "SessionMetrics": "victor.framework.agent_components",
    "SessionState": "victor.framework.agent_components",
    "create_bridge": "victor.framework.agent_components",
    "create_builder": "victor.framework.agent_components",
    "create_session": "victor.framework.agent_components",
    # Tool Configuration
    "AirgappedFilter": "victor.framework.tool_config",
    "CostTierFilter": "victor.framework.tool_config",
    "SecurityFilter": "victor.framework.tool_config",
    "ToolConfig": "victor.framework.tool_config",
    "ToolConfigBuilder": "victor.framework.tool_config",
    "ToolConfigEntry": "victor.framework.tool_config",
    "ToolConfigMode": "victor.framework.tool_config",
    "ToolConfigResult": "victor.framework.tool_config",
    "ToolConfigurator": "victor.framework.tool_config",
    "ToolFilterProtocol": "victor.framework.tool_config",
    "configure_tools": "victor.framework.tool_config",
    "configure_tools_from_toolset": "victor.framework.tool_config",
    "get_tool_configurator": "victor.framework.tool_config",
    # Service Provider
    "AgentBuilderService": "victor.framework.service_provider",
    "AgentSessionService": "victor.framework.service_provider",
    "EventRegistryService": "victor.framework.service_provider",
    "FrameworkScope": "victor.framework.service_provider",
    "FrameworkServiceProvider": "victor.framework.service_provider",
    "ToolConfiguratorService": "victor.framework.service_provider",
    "configure_framework_services": "victor.framework.service_provider",
    "create_framework_scope": "victor.framework.service_provider",
    # Resilience
    "CircuitBreaker": "victor.framework.resilience",
    "CircuitBreakerError": "victor.framework.resilience",
    "CircuitBreakerRegistry": "victor.framework.resilience",
    "CircuitState": "victor.framework.resilience",
    "CircuitBreakerConfig": "victor.framework.resilience",
    "CircuitBreakerState": "victor.framework.resilience",
    "CircuitOpenError": "victor.framework.resilience",
    "ProviderUnavailableError": "victor.framework.resilience",
    "ResilientProvider": "victor.framework.resilience",
    "ProviderRetryConfig": "victor.framework.resilience",
    "RetryExhaustedError": "victor.framework.resilience",
    "ProviderRetryStrategy": "victor.framework.resilience",
    "ExponentialBackoffStrategy": "victor.framework.resilience",
    "FixedDelayStrategy": "victor.framework.resilience",
    "LinearBackoffStrategy": "victor.framework.resilience",
    "NoRetryStrategy": "victor.framework.resilience",
    "RetryContext": "victor.framework.resilience",
    "RetryExecutor": "victor.framework.resilience",
    "RetryOutcome": "victor.framework.resilience",
    "RetryResult": "victor.framework.resilience",
    "BaseRetryStrategy": "victor.framework.resilience",
    "connection_retry_strategy": "victor.framework.resilience",
    "provider_retry_strategy": "victor.framework.resilience",
    "tool_retry_strategy": "victor.framework.resilience",
    "with_retry": "victor.framework.resilience",
    "with_retry_sync": "victor.framework.resilience",
    "DatabaseRetryHandler": "victor.framework.resilience",
    "FRAMEWORK_RETRY_HANDLERS": "victor.framework.resilience",
    "NetworkRetryHandler": "victor.framework.resilience",
    "RateLimitRetryHandler": "victor.framework.resilience",
    "register_framework_retry_handlers": "victor.framework.resilience",
    "retry_with_backoff": "victor.framework.resilience",
    "retry_with_backoff_sync": "victor.framework.resilience",
    "RetryConfig": "victor.framework.resilience",
    "RetryHandler": "victor.framework.resilience",
    "RetryHandlerConfig": "victor.framework.resilience",
    "with_exponential_backoff": "victor.framework.resilience",
    "with_exponential_backoff_sync": "victor.framework.resilience",
    # Health
    "BaseHealthCheck": "victor.framework.health",
    "CacheHealthCheck": "victor.framework.health",
    "CallableHealthCheck": "victor.framework.health",
    "ComponentHealth": "victor.framework.health",
    "HealthCheckProtocol": "victor.framework.health",
    "HealthChecker": "victor.framework.health",
    "HealthReport": "victor.framework.health",
    "HealthStatus": "victor.framework.health",
    "MemoryHealthCheck": "victor.framework.health",
    "ProviderHealthCheck": "victor.framework.health",
    "ToolHealthCheck": "victor.framework.health",
    "create_default_health_checker": "victor.framework.health",
    "HealthCheckResult": "victor.framework.health",
    "ProviderHealthStatus": "victor.framework.health",
    "ProviderHealthChecker": "victor.framework.health",
    "ProviderHealthReport": "victor.framework.health",
    "get_provider_health_checker": "victor.framework.health",
    "reset_provider_health_checker": "victor.framework.health",
    # Metrics
    "Counter": "victor.framework.metrics",
    "Gauge": "victor.framework.metrics",
    "Histogram": "victor.framework.metrics",
    "Metric": "victor.framework.metrics",
    "MetricLabels": "victor.framework.metrics",
    "MetricsCollector": "victor.framework.metrics",
    "MetricsRegistry": "victor.framework.metrics",
    "Timer": "victor.framework.metrics",
    "TimerContext": "victor.framework.metrics",
    "get_meter": "victor.framework.metrics",
    "get_tracer": "victor.framework.metrics",
    "is_telemetry_enabled": "victor.framework.metrics",
    "setup_opentelemetry": "victor.framework.metrics",
    # Services
    "ServiceLifecycleProtocol": "victor.framework.services",
    "ServiceConfigurable": "victor.framework.services",
    "ServiceTypeHandler": "victor.framework.services",
    "ServiceState": "victor.framework.services",
    "ServiceMetadata": "victor.framework.services",
    "BaseServiceConfig": "victor.framework.services",
    "SQLiteServiceConfig": "victor.framework.services",
    "DockerServiceConfig": "victor.framework.services",
    "HTTPServiceConfig": "victor.framework.services",
    "ExternalServiceConfig": "victor.framework.services",
    "ServiceStartError": "victor.framework.services",
    "ServiceStopError": "victor.framework.services",
    "BaseService": "victor.framework.services",
    "SQLiteServiceHandler": "victor.framework.services",
    "DockerServiceHandler": "victor.framework.services",
    "HTTPServiceHandler": "victor.framework.services",
    "ExternalServiceHandler": "victor.framework.services",
    "ServiceRegistry": "victor.framework.services",
    "ServiceManager": "victor.framework.services",
    "create_sqlite_service": "victor.framework.services",
    "create_http_service": "victor.framework.services",
    # Teams
    "MemberResult": "victor.framework.teams",
    "TeamFormation": "victor.framework.teams",
    "TeamResult": "victor.framework.teams",
    "AgentTeam": "victor.framework.teams",
    "TeamEvent": "victor.framework.teams",
    "TeamEventType": "victor.framework.teams",
    "TeamMemberSpec": "victor.framework.teams",
    "member_complete_event": "victor.framework.teams",
    "member_start_event": "victor.framework.teams",
    "team_complete_event": "victor.framework.teams",
    "team_start_event": "victor.framework.teams",
    "TeamConfig": "victor.framework.teams",
    "TeamMember": "victor.framework.teams",
    # Reinforcement Learning
    "BaseLearner": "victor.framework.rl",
    "LearnerStats": "victor.framework.rl",
    "LearnerType": "victor.framework.rl",
    "RLCoordinator": "victor.framework.rl",
    "RLManager": "victor.framework.rl",
    "RLOutcome": "victor.framework.rl",
    "RLRecommendation": "victor.framework.rl",
    "RLStats": "victor.framework.rl",
    "create_outcome": "victor.framework.rl",
    "get_rl_coordinator": "victor.framework.rl",
    "record_tool_success": "victor.framework.rl",
    # Team Registry
    "TeamSpecRegistry": "victor.framework.team_registry",
    "TeamSpecEntry": "victor.framework.team_registry",
    "get_team_registry": "victor.framework.team_registry",
    "register_team_spec": "victor.framework.team_registry",
    "get_team_spec": "victor.framework.team_registry",
    # Graph Workflow Engine
    "StateGraph": "victor.framework.graph",
    "CompiledGraph": "victor.framework.graph",
    "Node": "victor.framework.graph",
    "Edge": "victor.framework.graph",
    "EdgeType": "victor.framework.graph",
    "FrameworkNodeStatus": "victor.framework.graph",
    "GraphExecutionResult": "victor.framework.graph",
    "GraphConfig": "victor.framework.graph",
    "WorkflowCheckpoint": "victor.framework.graph",
    "CheckpointerProtocol": "victor.framework.graph",
    "MemoryCheckpointer": "victor.framework.graph",
    "RLCheckpointerAdapter": "victor.framework.graph",
    "StateProtocol": "victor.framework.graph",
    "NodeFunctionProtocol": "victor.framework.graph",
    "ConditionFunctionProtocol": "victor.framework.graph",
    "END": "victor.framework.graph",
    "START": "victor.framework.graph",
    "create_graph": "victor.framework.graph",
    # Module Loader
    "CachedEntryPoints": "victor.framework.module_loader",
    "DebouncedReloadTimer": "victor.framework.module_loader",
    "DynamicModuleLoader": "victor.framework.module_loader",
    "EntryPointCache": "victor.framework.module_loader",
    "get_entry_point_cache": "victor.framework.module_loader",
    # Capability Loader
    "CapabilityLoader": "victor.framework.capability_loader",
    "CapabilityEntry": "victor.framework.capability_loader",
    "CapabilityLoadError": "victor.framework.capability_loader",
    "capability": "victor.framework.capability_loader",
    "create_capability_loader": "victor.framework.capability_loader",
    "get_default_capability_loader": "victor.framework.capability_loader",
    # Tool Naming
    "CANONICAL_TO_ALIASES": "victor.framework.tool_naming",
    "TOOL_ALIASES": "victor.framework.tool_naming",
    "ToolNameEntry": "victor.framework.tool_naming",
    "ToolNames": "victor.framework.tool_naming",
    "canonicalize_dependencies": "victor.framework.tool_naming",
    "canonicalize_tool_dict": "victor.framework.tool_naming",
    "canonicalize_tool_list": "victor.framework.tool_naming",
    "canonicalize_tool_set": "victor.framework.tool_naming",
    "canonicalize_transitions": "victor.framework.tool_naming",
    "get_aliases": "victor.framework.tool_naming",
    "get_all_canonical_names": "victor.framework.tool_naming",
    "get_canonical_name": "victor.framework.tool_naming",
    "get_legacy_names_report": "victor.framework.tool_naming",
    "get_name_mapping": "victor.framework.tool_naming",
    "is_valid_tool_name": "victor.framework.tool_naming",
    "validate_tool_names": "victor.framework.tool_naming",
    # Middleware
    "GitSafetyMiddleware": "victor.framework.middleware",
    "LoggingMiddleware": "victor.framework.middleware",
    "MetricsMiddleware": "victor.framework.middleware",
    "SecretMaskingMiddleware": "victor.framework.middleware",
    "ToolMetrics": "victor.framework.middleware",
    # Middleware Protocols
    "IMiddleware": "victor.framework.middleware_protocols",
    "MiddlewarePhase": "victor.framework.middleware_protocols",
    "BaseMiddleware": "victor.framework.middleware_base",
    "MiddlewarePriority": "victor.core.vertical_types",
    "MiddlewareResult": "victor.core.vertical_types",
    # Task Types
    "TaskCategory": "victor.framework.task_types",
    "TaskTypeDefinition": "victor.framework.task_types",
    "TaskTypeRegistry": "victor.framework.task_types",
    "get_task_budget": "victor.framework.task_types",
    "get_task_hint": "victor.framework.task_types",
    "get_task_type_registry": "victor.framework.task_types",
    "register_vertical_task_type": "victor.framework.task_types",
    "register_coding_task_types": "victor.framework.task_types",
    "register_data_analysis_task_types": "victor.framework.task_types",
    "register_devops_task_types": "victor.framework.task_types",
    "register_research_task_types": "victor.framework.task_types",
    "setup_vertical_task_types": "victor.framework.task_types",
    # Vertical Templates
    "VerticalTemplate": "victor.framework.vertical_template",
    "VerticalMetadata": "victor.framework.vertical_template",
    "ExtensionSpecs": "victor.framework.vertical_template",
    "MiddlewareSpec": "victor.framework.vertical_template",
    "SafetyPatternSpec": "victor.framework.vertical_template",
    "PromptHintSpec": "victor.framework.vertical_template",
    "WorkflowSpec": "victor.framework.vertical_template",
    "TeamSpec": "victor.framework.vertical_template",
    "TeamRoleSpec": "victor.framework.vertical_template",
    "CapabilitySpec": "victor.framework.vertical_template",
    "VerticalTemplateRegistry": "victor.framework.vertical_template_registry",
    "get_template_registry": "victor.framework.vertical_template_registry",
    "register_template": "victor.framework.vertical_template_registry",
    "get_template": "victor.framework.vertical_template_registry",
    "list_templates": "victor.framework.vertical_template_registry",
    "VerticalExtractor": "victor.framework.vertical_extractor",
    "VerticalGenerator": "victor.framework.vertical_extractor",
    "migrate_vertical_to_template": "victor.framework.vertical_extractor",
    # Workflow Chat (Phase 1 - Domain-Agnostic Workflow Chat)
    "ChatResultProtocol": "victor.framework.protocols.workflow_chat",
    "ChatStateProtocol": "victor.framework.protocols.workflow_chat",
    "WorkflowChatProtocol": "victor.framework.protocols.workflow_chat",
    "WorkflowChatExecutorProtocol": "victor.framework.protocols.workflow_chat",
    "MutableChatState": "victor.framework.protocols.chat_state",
    "ChatResult": "victor.framework.protocols.chat_state",
    "WorkflowOrchestrator": "victor.framework.workflow_orchestrator",
    "WorkflowChatCoordinator": "victor.framework.coordinators.workflow_chat_coordinator",
    "ChatExecutionEvent": "victor.framework.coordinators.workflow_chat_coordinator",
    "ChatEventType": "victor.framework.coordinators.workflow_chat_coordinator",
    "ChatExecutionConfig": "victor.framework.coordinators.workflow_chat_coordinator",
}

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
        "AgentConfig",
        # Task
        "TaskResult",
        "FrameworkTaskType",
        # Tools
        "ToolSet",
        "ToolCategory",
        "ToolsInput",
        # State
        "ConversationStage",
        "StateHooks",
        "StateObserver",
        # Checkpoint
        "CheckpointBackend",
        # Prompt Builder
        "PromptBuilder",
        "PromptSection",
        "ToolHint",
        "create_coding_prompt_builder",
        "create_data_analysis_prompt_builder",
        "create_devops_prompt_builder",
        "create_research_prompt_builder",
        "PromptBuilderTemplate",
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
        # Protocols (Phase 1 - Workflow Chat)
        "ChatStateProtocol",
        "ChatResultProtocol",
        "WorkflowChatProtocol",
        # Chat State Implementations (Phase 1)
        "ChatResult",
        "MutableChatState",
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
        # Prompt Sections (legacy constants)
        "GROUNDING_RULES_MINIMAL",
        "GROUNDING_RULES_EXTENDED",
        "PARALLEL_READ_GUIDANCE",
        "CODING_IDENTITY",
        "CODING_GUIDELINES",
        "CODING_TOOL_USAGE",
        "DEVOPS_IDENTITY",
        "DEVOPS_SECURITY_CHECKLIST",
        "DEVOPS_COMMON_PITFALLS",
        "DEVOPS_GROUNDING",
        "RESEARCH_IDENTITY",
        "RESEARCH_QUALITY_CHECKLIST",
        "RESEARCH_SOURCE_HIERARCHY",
        "RESEARCH_GROUNDING",
        "DATA_ANALYSIS_IDENTITY",
        "DATA_ANALYSIS_LIBRARIES",
        "DATA_ANALYSIS_OPERATIONS",
        "DATA_ANALYSIS_GROUNDING",
        "TASK_HINT_CODE_GENERATION",
        "TASK_HINT_CREATE_SIMPLE",
        "TASK_HINT_CREATE",
        "TASK_HINT_EDIT",
        "TASK_HINT_SEARCH",
        "TASK_HINT_ACTION",
        "TASK_HINT_ANALYSIS_DEEP",
        "TASK_HINT_ANALYZE",
        "TASK_HINT_DESIGN",
        "TASK_HINT_GENERAL",
        "TASK_HINT_REFACTOR",
        "TASK_HINT_DEBUG",
        "TASK_HINT_TEST",
        "DEEPSEEK_TOOL_EFFICIENCY",
        "XAI_GROK_GUIDANCE",
        "OLLAMA_STRICT_GUIDANCE",
        # Prompt Section Classes (Phase 7)
        "GroundingSection",
        "RuleSection",
        "ChecklistSection",
        "VerticalPromptSection",
        # Vertical Templates
        "VerticalTemplate",
        "VerticalMetadata",
        "ExtensionSpecs",
        "MiddlewareSpec",
        "SafetyPatternSpec",
        "PromptHintSpec",
        "WorkflowSpec",
        "TeamSpec",
        "TeamRoleSpec",
        "CapabilitySpec",
        "VerticalTemplateRegistry",
        "get_template_registry",
        "register_template",
        "get_template",
        "list_templates",
        "VerticalExtractor",
        "VerticalGenerator",
        "migrate_vertical_to_template",
        # Workflow Chat (Phase 1)
        "ChatResultProtocol",
        "ChatStateProtocol",
        "WorkflowChatProtocol",
        "WorkflowChatExecutorProtocol",
        "MutableChatState",
        "ChatResult",
        "WorkflowOrchestrator",
        "WorkflowChatCoordinator",
        "ChatExecutionEvent",
        "ChatEventType",
        "ChatExecutionConfig",
    ]
    + list(_LAZY_MODULES.keys())  # All lazy-loaded exports
    + ["discover"]  # Capability discovery function
)


def __getattr__(name: str) -> Any:
    """Lazy load optional framework subsystems on first access.

    This function is called by Python when an attribute is not found in the module.
    It lazy-loads the optional subsystem to improve startup performance.

    Args:
        name: Name of the attribute being accessed

    Returns:
        The requested attribute from the lazy-loaded module

    Raises:
        AttributeError: If the requested attribute doesn't exist
    """
    if name in _LAZY_MODULES:
        # Import the module and get the attribute
        import importlib

        module_path = _LAZY_MODULES[name]
        module = importlib.import_module(module_path)

        # Cache the attribute in this module's dict so future accesses are fast
        import sys

        # Get the actual module object for this module
        this_module = sys.modules[__name__]

        # Cache the attribute
        setattr(this_module, name, getattr(module, name))

        return getattr(module, name)

    if name == "discover":
        # Special case for discover function
        from victor.ui.commands.capabilities import get_capability_discovery

        def discover() -> dict[str, Any]:
            """Discover all Victor framework capabilities programmatically.

            Returns a dictionary containing all available tools, verticals, personas,
            teams, chains, handlers, task types, providers, and events.

            Example:
                from victor.framework import discover

                caps = discover()
                print(f"Available tools: {len(caps['tools'])}")
                print(f"Available verticals: {caps['verticals']}")

            Returns: dict[str, Any]: Capability manifest with keys:
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
            discovery = get_capability_discovery()
            manifest = discovery.discover_all()
            return manifest.to_dict()

        # Cache the function
        import sys

        this_module = sys.modules[__name__]
        this_module.discover = discover  # type: ignore[attr-defined]

        return discover

    raise AttributeError(f"module 'victor.framework' has no attribute '{name}'")


# Version of the framework API
__version__ = "0.5.0"
