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

"""Workflow definition and execution system.

This package provides a LangGraph-like workflow DSL for defining
reusable multi-agent workflows as Python code or YAML.

Features:
- ComputeNode: LLM-free execution with constraints
- AgentNode: LLM-powered execution for complex reasoning
- Handlers: Extensible compute handlers for domain logic
- Isolation: Per-vertical sandbox configuration
- YAML Support: Declarative workflow definitions

Example (Python):
    from victor.workflows import WorkflowBuilder, workflow, WorkflowRegistry

    @workflow("code_review", "Review code quality")
    def code_review_workflow():
        return (
            WorkflowBuilder("code_review")
            .add_agent("analyze", "researcher", "Find code patterns")
            .add_agent("review", "reviewer", "Review quality")
            .add_agent("report", "planner", "Summarize findings")
            .build()
        )

Example (YAML):
    workflows:
      eda_pipeline:
        nodes:
          - id: load
            type: compute
            tools: [read]
            constraints:
              llm_allowed: false
          - id: analyze
            type: agent
            role: analyst
            goal: "Analyze data patterns"
"""

from typing import Optional

from victor.workflows.base import BaseWorkflow
from victor.workflows.definition import (
    WorkflowNodeType,
    WorkflowNode,
    AgentNode,
    ConditionNode,
    ParallelNode,
    TransformNode,
    WorkflowDefinition,
    WorkflowBuilder,
    workflow,
    get_registered_workflows,
)
from victor.workflows.registry import (
    WorkflowMetadata,
    WorkflowRegistry,
    get_global_registry,
)
from victor.workflows.executor import (
    ExecutorNodeStatus,
    NodeResult,
    WorkflowContext,
    WorkflowResult,
    WorkflowExecutor,
)
from victor.workflows.protocols import (
    RetryPolicy,
    IWorkflowNode,
    IWorkflowEdge,
    IWorkflowGraph,
    ICheckpointStore,
    IWorkflowExecutor,
    IStreamingWorkflowExecutor,
    # Re-export ProtocolNodeStatus and NodeResult from protocols for graph API users
    ProtocolNodeStatus as GraphNodeStatus,
    NodeResult as GraphNodeResult,
)
from victor.workflows.streaming import (
    WorkflowEventType,
    WorkflowStreamChunk,
    WorkflowStreamContext,
)
from victor.workflows.streaming_executor import (
    StreamingWorkflowExecutor,
)
from victor.workflows.yaml_loader import (
    YAMLWorkflowError,
    YAMLWorkflowConfig,
    YAMLWorkflowProvider,
    load_workflow_from_dict,
    load_workflow_from_yaml,
    load_workflow_from_file,
    load_workflows_from_directory,
    load_and_validate,
)
from victor.workflows.cache import (
    WorkflowCacheConfig as _LegacyWorkflowCacheConfig,
    WorkflowNodeCacheEntry,
    WorkflowCache as _LegacyWorkflowCache,
    WorkflowCacheManager as _LegacyWorkflowCacheManager,
    get_workflow_cache_manager as _get_workflow_cache_manager,
    configure_workflow_cache as _configure_workflow_cache,
)


class WorkflowCacheConfig(_LegacyWorkflowCacheConfig):
    """Package-level defaults with cache disabled unless explicitly enabled."""

    def __init__(
        self,
        enabled: bool = False,
        ttl_seconds: int = 3600,
        max_entries: int = 500,
        **kwargs: object,
    ) -> None:
        if "max_size" in kwargs and "max_entries" not in kwargs:
            max_entries = kwargs.pop("max_size")
        super().__init__(
            enabled=enabled,
            ttl_seconds=ttl_seconds,
            max_entries=max_entries,
            **kwargs,
        )


class WorkflowCache(_LegacyWorkflowCache):
    """WorkflowCache with package-level defaults."""

    def __init__(self, config: Optional[_LegacyWorkflowCacheConfig] = None) -> None:
        super().__init__(config or WorkflowCacheConfig())


class WorkflowCacheManager(_LegacyWorkflowCacheManager):
    """WorkflowCacheManager with package-level defaults."""

    def __init__(self, default_config: Optional[_LegacyWorkflowCacheConfig] = None) -> None:
        super().__init__(default_config or WorkflowCacheConfig())

    def get_cache(
        self,
        workflow_name: str,
        config: Optional[_LegacyWorkflowCacheConfig] = None,
    ) -> Optional[WorkflowCache]:
        """Get or create cache for a workflow (package-level behavior)."""
        with self._lock:
            cache = self._caches.get(workflow_name)
            if cache is None:
                cache_config = config or self._default_config
                cache = WorkflowCache(cache_config)
                self._caches[workflow_name] = cache
            return cache


_package_workflow_cache_manager: Optional[WorkflowCacheManager] = None


def get_workflow_cache_manager() -> WorkflowCacheManager:
    """Get the package-level workflow cache manager."""
    global _package_workflow_cache_manager
    if _package_workflow_cache_manager is None:
        _package_workflow_cache_manager = WorkflowCacheManager()
    return _package_workflow_cache_manager


def configure_workflow_cache(config: WorkflowCacheConfig) -> None:
    """Configure the package-level workflow cache manager."""
    global _package_workflow_cache_manager
    _configure_workflow_cache(config)
    _package_workflow_cache_manager = WorkflowCacheManager(config)


from victor.workflows.graph_dsl import (
    State,
    WorkflowGraph,  # Typed workflow graph DSL (compiles to WorkflowDefinition)
    GraphNode,
    GraphNodeType,
    NodeFunc,
    RouterFunc,
    Compilable,
    create_graph,
    compile_graph,
)
from victor.workflows.graph import (
    WorkflowNode as GraphWorkflowNode,
    WorkflowEdge,
    ConditionalEdge,
    BasicWorkflowGraph,  # Basic graph container
    DuplicateNodeError,
    InvalidEdgeError,
    GraphValidationError,
)

# New modules for extended workflow capabilities
from victor.workflows.definition import (
    ComputeNode,
    TaskConstraints,
    ConstraintsProtocol,
    AirgappedConstraints,
    ComputeOnlyConstraints,
    FullAccessConstraints,
)
from victor.workflows.executor import (
    ComputeHandler,
    TemporalContext,
    register_compute_handler,
    get_compute_handler,
    list_compute_handlers,
)
from victor.workflows.isolation import (
    SandboxType,
    ResourceLimits,
    IsolationConfig,
    IsolationMapper,
)
from victor.workflows.cost_router import (
    CostTier,
    ModelConfig,
    RoutingDecision,
    CostAwareRouter,
    get_default_router,
    route_for_cost,
)
from victor.workflows.sandbox_executor import (
    SandboxExecutionResult,
    SandboxedExecutor,
    get_sandboxed_executor,
)
from victor.workflows.handlers import (
    ParallelToolsHandler,
    SequentialToolsHandler,
    RetryBackoffHandler,
    DataTransformHandler,
    ConditionalBranchHandler,
    FRAMEWORK_HANDLERS,
    register_framework_handlers,
)
from victor.workflows.batch_executor import (
    BatchRetryStrategy,
    BatchConfig,
    BatchItemResult,
    BatchProgress,
    BatchResult,
    BatchWorkflowExecutor,
)
from victor.workflows.yaml_to_graph_compiler import (
    WorkflowState,
    GraphNodeResult as CompiledGraphNodeResult,
    CompilerConfig,
    YAMLToStateGraphCompiler,
    NodeExecutorFactory,
    ConditionEvaluator,
    compile_yaml_workflow,
    execute_yaml_workflow,
)
from victor.workflows.unified_executor import (
    ExecutorConfig,
    ExecutorResult,
    StateGraphExecutor,
    get_executor,
    execute_workflow,
)
from victor.workflows.deployment import (
    DeploymentTarget,
    DockerConfig,
    DockerComposeConfig,
    KubernetesConfig,
    ECSConfig,
    CloudRunConfig,
    AzureContainerConfig,
    LambdaConfig,
    RemoteConfig,
    SandboxConfig,
    DeploymentConfig,
    DeploymentHandler,
    get_deployment_handler,
)
from victor.workflows.service_lifecycle import (
    ServiceConfig,
    ServiceLifecycle,
    ServiceManager,
    PostgresService,
    RedisService,
    RabbitMQService,
    KafkaService,
    HTTPClientService,
    S3Service,
)
from victor.workflows.runtime import (
    HITLServerConfig,
    RuntimeConfig,
    RuntimeResult,
    WorkflowRuntime,
    run_workflow,
)
from victor.workflows.hitl import (
    HITLNodeType,
    HITLFallback,
    HITLMode,
    HITLCategory,
    HITL_MODE_CATEGORIES,
    HITLStatus,
    HITLRequest,
    HITLResponse,
    HITLHandler,
    HITLNode,
    HITLExecutor,
    DefaultHITLHandler,
    DEPLOYMENT_HITL_COMPATIBILITY,
    get_supported_hitl_modes,
    validate_hitl_deployment_compatibility,
)
from victor.workflows.hitl_transports import (
    # Base classes
    BaseTransportConfig,
    BaseTransport,
    HITLTransportProtocol,
    # Config classes
    EmailConfig,
    SMSConfig,
    SlackConfig,
    TeamsConfig,
    GitHubConfig,
    GitLabConfig,
    JiraConfig,
    PagerDutyConfig,
    TerraformCloudConfig,
    CustomHookConfig,
    # Transport classes
    EmailTransport,
    SMSTransport,
    SlackTransport,
    GitHubPRTransport,
    GitHubCheckTransport,
    CustomHookTransport,
    # Registry functions
    register_transport,
    get_transport,
    list_available_transports,
)
from victor.workflows.hitl_api import (
    StoredRequest,
    HITLStore,
    get_global_store,
    SQLiteHITLStore,
    get_sqlite_store,
    APIHITLHandler,
    create_hitl_router,
    create_hitl_app,
    run_hitl_server,
)
from victor.workflows.context import (
    ExecutionContext,
    create_execution_context,
    ExecutionContextWrapper,
    from_workflow_context,
    to_workflow_context,
    from_compiler_workflow_state,
    to_compiler_workflow_state,
    from_adapter_workflow_state,
    to_adapter_workflow_state,
)
from victor.workflows.node_runners import (
    BaseNodeRunner,
    AgentNodeRunner,
    ComputeNodeRunner,
    TransformNodeRunner,
    HITLNodeRunner,
    ConditionNodeRunner,
    ParallelNodeRunner,
    NodeRunnerRegistry,
)
from victor.workflows.team_node_runner import (
    TeamNodeRunner,
)
from victor.workflows.protocols import (
    NodeRunner,
    NodeRunnerResult,
)
from victor.workflows.graph_compiler import (
    CompilerConfig as GraphCompilerConfig,
    NodeRunnerWrapper,
    WorkflowGraphCompiler,
    WorkflowDefinitionCompiler,
    compile_workflow_graph,
    compile_workflow_definition,
)
from victor.workflows.observability import (
    StreamingObserver,
    AsyncStreamingObserver,
    FunctionObserver,
    ObservabilityEmitter,
    create_emitter,
    create_logging_observer,
)
from victor.workflows.unified_compiler import (
    UnifiedWorkflowCompiler,
    create_unified_compiler,
)
from victor.workflows.create import (
    create_compiler,
    register_scheme_alias,
    list_supported_schemes,
    is_scheme_supported,
    get_default_scheme,
    set_default_scheme,
)
from victor.workflows.compiler_registry import (
    WorkflowCompilerRegistry,
    register_compiler,
    unregister_compiler,
    get_compiler,
    is_registered,
    list_compilers,
)
from victor.workflows.team_metrics import (
    TeamMetricsCollector,
    MemberExecutionMetrics,
    TeamExecutionMetrics,
    MetricPriority,
    get_team_metrics_collector,
    record_team_execution,
)
from victor.workflows.team_tracing import (
    TeamTracer,
    TraceSpan,
    SpanAttributes,
    SpanEvents,
    SpanKind,
    get_team_tracer,
    trace_team_execution,
    trace_member_execution,
    trace_workflow_execution,
    get_current_trace_id,
    set_trace_id,
    export_trace_to_dict,
    get_all_traces,
)
from victor.workflows.template_registry import (
    WorkflowTemplateRegistry,
    get_workflow_template_registry,
    register_default_templates,
)

# Register framework handlers on module load
# Domain-specific handlers are registered by each vertical when loaded
register_framework_handlers()

__all__ = [
    # Base
    "BaseWorkflow",
    # Node types
    "WorkflowNodeType",
    "WorkflowNode",
    "AgentNode",
    "ComputeNode",
    "ConditionNode",
    "ParallelNode",
    "TransformNode",
    # Constraints
    "TaskConstraints",
    "ConstraintsProtocol",
    "AirgappedConstraints",
    "ComputeOnlyConstraints",
    "FullAccessConstraints",
    # Definition
    "WorkflowDefinition",
    "WorkflowBuilder",
    "workflow",
    "get_registered_workflows",
    # Registry
    "WorkflowMetadata",
    "WorkflowRegistry",
    "get_global_registry",
    # Executor
    "ExecutorNodeStatus",
    "NodeResult",
    "WorkflowContext",
    "WorkflowResult",
    "WorkflowExecutor",
    "TemporalContext",
    # Compute Handlers
    "ComputeHandler",
    "register_compute_handler",
    "get_compute_handler",
    "list_compute_handlers",
    # Framework handlers (domain handlers are provided by each vertical)
    "ParallelToolsHandler",
    "SequentialToolsHandler",
    "RetryBackoffHandler",
    "DataTransformHandler",
    "ConditionalBranchHandler",
    "FRAMEWORK_HANDLERS",
    "register_framework_handlers",
    # Isolation
    "SandboxType",
    "ResourceLimits",
    "IsolationConfig",
    "IsolationMapper",
    # Cost Routing
    "CostTier",
    "ModelConfig",
    "RoutingDecision",
    "CostAwareRouter",
    "get_default_router",
    "route_for_cost",
    # Sandbox Execution
    "SandboxExecutionResult",
    "SandboxedExecutor",
    "get_sandboxed_executor",
    # Batch Execution
    "BatchRetryStrategy",
    "BatchConfig",
    "BatchItemResult",
    "BatchProgress",
    "BatchResult",
    "BatchWorkflowExecutor",
    # Protocols (Graph API)
    "RetryPolicy",
    "IWorkflowNode",
    "IWorkflowEdge",
    "IWorkflowGraph",
    "ICheckpointStore",
    "IWorkflowExecutor",
    "IStreamingWorkflowExecutor",
    "GraphNodeStatus",
    "GraphNodeResult",
    # Streaming
    "WorkflowEventType",
    "WorkflowStreamChunk",
    "WorkflowStreamContext",
    "StreamingWorkflowExecutor",
    # YAML Loader
    "YAMLWorkflowError",
    "YAMLWorkflowConfig",
    "YAMLWorkflowProvider",
    "load_workflow_from_dict",
    "load_workflow_from_yaml",
    "load_workflow_from_file",
    "load_workflows_from_directory",
    "load_and_validate",
    # Cache
    "WorkflowCacheConfig",
    "WorkflowNodeCacheEntry",
    "WorkflowCache",
    "WorkflowCacheManager",
    "get_workflow_cache_manager",
    "configure_workflow_cache",
    # WorkflowGraph DSL (compiles to WorkflowDefinition)
    "State",
    "WorkflowGraph",  # Typed workflow graph DSL
    "GraphNode",
    "GraphNodeType",
    "NodeFunc",
    "RouterFunc",
    "Compilable",
    "create_graph",
    "compile_graph",
    # Graph implementation (basic graph container)
    "GraphWorkflowNode",
    "WorkflowEdge",
    "ConditionalEdge",
    "BasicWorkflowGraph",  # Basic graph container (renamed from WorkflowGraph)
    "DuplicateNodeError",
    "InvalidEdgeError",
    "GraphValidationError",
    # YAML to StateGraph Compiler
    "WorkflowState",
    "CompiledGraphNodeResult",
    "CompilerConfig",
    "YAMLToStateGraphCompiler",
    "NodeExecutorFactory",
    "ConditionEvaluator",
    "compile_yaml_workflow",
    "execute_yaml_workflow",
    # StateGraph Executor
    "ExecutorConfig",
    "ExecutorResult",
    "StateGraphExecutor",
    "get_executor",
    "execute_workflow",
    # Deployment
    "DeploymentTarget",
    "DockerConfig",
    "DockerComposeConfig",
    "KubernetesConfig",
    "ECSConfig",
    "CloudRunConfig",
    "AzureContainerConfig",
    "LambdaConfig",
    "RemoteConfig",
    "SandboxConfig",
    "DeploymentConfig",
    "DeploymentHandler",
    "get_deployment_handler",
    # Services
    "ServiceConfig",
    "ServiceLifecycle",
    "ServiceManager",
    "PostgresService",
    "RedisService",
    "RabbitMQService",
    "KafkaService",
    "HTTPClientService",
    "S3Service",
    # Runtime (serverless-like)
    "HITLServerConfig",
    "RuntimeConfig",
    "RuntimeResult",
    "WorkflowRuntime",
    "run_workflow",
    # HITL (Human-in-the-Loop)
    "HITLNodeType",
    "HITLFallback",
    "HITLMode",
    "HITLCategory",
    "HITL_MODE_CATEGORIES",
    "HITLStatus",
    "HITLRequest",
    "HITLResponse",
    "HITLHandler",
    "HITLNode",
    "HITLExecutor",
    "DefaultHITLHandler",
    "DEPLOYMENT_HITL_COMPATIBILITY",
    "get_supported_hitl_modes",
    "validate_hitl_deployment_compatibility",
    # HITL API (remote deployments)
    "StoredRequest",
    "HITLStore",
    "get_global_store",
    "SQLiteHITLStore",
    "get_sqlite_store",
    "APIHITLHandler",
    "create_hitl_router",
    "create_hitl_app",
    "run_hitl_server",
    # HITL Transports (external integrations)
    "BaseTransportConfig",
    "BaseTransport",
    "HITLTransportProtocol",
    "EmailConfig",
    "SMSConfig",
    "SlackConfig",
    "TeamsConfig",
    "GitHubConfig",
    "GitLabConfig",
    "JiraConfig",
    "PagerDutyConfig",
    "TerraformCloudConfig",
    "CustomHookConfig",
    "EmailTransport",
    "SMSTransport",
    "SlackTransport",
    "GitHubPRTransport",
    "GitHubCheckTransport",
    "CustomHookTransport",
    "register_transport",
    "get_transport",
    "list_available_transports",
    # Unified Execution Context
    "ExecutionContext",
    "create_execution_context",
    "ExecutionContextWrapper",
    "from_workflow_context",
    "to_workflow_context",
    "from_compiler_workflow_state",
    "to_compiler_workflow_state",
    "from_adapter_workflow_state",
    "to_adapter_workflow_state",
    # NodeRunner Protocol and Implementations (ISP + DIP)
    "NodeRunner",
    "NodeRunnerResult",
    "BaseNodeRunner",
    "AgentNodeRunner",
    "ComputeNodeRunner",
    "TransformNodeRunner",
    "HITLNodeRunner",
    "ConditionNodeRunner",
    "ParallelNodeRunner",
    "TeamNodeRunner",
    "NodeRunnerRegistry",
    # Graph Compilers (Single Execution Engine - Phase 4)
    "GraphCompilerConfig",
    "NodeRunnerWrapper",
    "WorkflowGraphCompiler",
    "WorkflowDefinitionCompiler",
    "compile_workflow_graph",
    "compile_workflow_definition",
    # Observability (Unified Streaming - Phase 5)
    "StreamingObserver",
    "AsyncStreamingObserver",
    "FunctionObserver",
    "ObservabilityEmitter",
    "create_emitter",
    "create_logging_observer",
    # Unified Compiler (Consistent Compilation and Caching)
    "UnifiedWorkflowCompiler",
    "create_unified_compiler",
    # Plugin Creation API
    "create_compiler",
    "register_scheme_alias",
    "list_supported_schemes",
    "is_scheme_supported",
    "get_default_scheme",
    "set_default_scheme",
    # Plugin Registry
    "WorkflowCompilerRegistry",
    "register_compiler",
    "unregister_compiler",
    "get_compiler",
    "is_registered",
    "list_compilers",
    # Team Metrics
    "TeamMetricsCollector",
    "MemberExecutionMetrics",
    "TeamExecutionMetrics",
    "MetricPriority",
    "get_team_metrics_collector",
    "record_team_execution",
    # Team Tracing
    "TeamTracer",
    "TraceSpan",
    "SpanAttributes",
    "SpanEvents",
    "SpanKind",
    "get_team_tracer",
    "trace_team_execution",
    "trace_member_execution",
    "trace_workflow_execution",
    "get_current_trace_id",
    "set_trace_id",
    "export_trace_to_dict",
    "get_all_traces",
    # Template Registry (Phase 5: Workflow Consolidation)
    "WorkflowTemplateRegistry",
    "get_workflow_template_registry",
    "register_default_templates",
]
