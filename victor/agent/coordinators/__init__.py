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

"""Agent coordinators for application-specific orchestration.

This package provides coordinators that manage the AI agent conversation lifecycle,
following SOLID principles (Single Responsibility, Interface Segregation,
Dependency Inversion, Open/Closed).

Architecture Layer: APPLICATION
-------------------------------
This is the APPLICATION LAYER of Victor's two-layer coordinator architecture:
- Application Layer (this package): Victor-specific orchestration logic
- Framework Layer (victor.framework.coordinators): Domain-agnostic workflow infrastructure

These coordinators handle Victor's business logic:
- How to manage conversation state
- When to switch providers or modes
- How to select and execute tools
- How to manage context windows
- How to collect session analytics

Key Coordinators:
-----------------
ConfigCoordinator:
    Configuration loading and validation from multiple sources (settings, env)
    Example: Load orchestrator config with validation

PromptCoordinator:
    System prompt building from composable contributors
    Example: Build prompts with system prompt + task hint + mode-specific content

ContextCoordinator:
    Context management and compaction strategies
    Example: Compact context window using truncation, summarization, or semantic strategies

AnalyticsCoordinator:
    Session analytics and metrics collection
    Example: Track tool usage, token consumption, performance metrics

ChatCoordinator:
    Chat and streaming operations with agentic loop
    Example: Manage LLM chat with tool calling, error recovery, and streaming

ToolCoordinator:
    Tool validation, execution coordination, and budget enforcement
    Example: Select tools semantically, execute tool calls, enforce budgets

ToolSelectionCoordinator:
    Semantic tool selection using hybrid strategies
    Example: Select relevant tools based on query context and embeddings

SessionCoordinator:
    Conversation session lifecycle management
    Example: Create, update, and close user sessions

ProviderCoordinator:
    Provider switching and management
    Example: Switch between Anthropic and OpenAI mid-conversation

ModeCoordinator:
    Agent modes (build, plan, explore)
    Example: Switch to plan mode for 2.5x exploration budget

CheckpointCoordinator:
    Workflow checkpoint management for state persistence
    Example: Save and restore workflow execution state

EvaluationCoordinator:
    LLM evaluation and benchmarking coordination
    Example: Run evaluations with different providers and models

MetricsCoordinator:
    System metrics collection and reporting
    Example: Track performance metrics, token usage, and costs

WorkflowCoordinator:
    High-level workflow orchestration (deprecated - use framework layer)
    Example: Orchestrate complex multi-step workflows

Design Principles:
------------------
1. Single Responsibility: Each coordinator has one reason to change
2. Interface Segregation: Focused protocols (e.g., IToolCoordinator)
3. Dependency Inversion: Depend on protocols, not concrete implementations
4. Open/Closed: Extensible through strategies and contributors

SOLID Compliance:
-----------------
- SRP: ChatCoordinator only handles chat (not config, tools, or context)
- ISP: IToolCoordinator defines only tool-related methods
- DIP: Coordinators depend on IAgentOrchestrator protocol
- OCP: New compaction strategies via BaseCompactionStrategy

Layer Interaction:
------------------
Application layer coordinators USE framework layer coordinators:
    ChatCoordinator --> GraphExecutionCoordinator (for workflow execution)
    ToolCoordinator --> GraphExecutionCoordinator (for tool workflows)

Framework coordinators are domain-agnostic and reusable across verticals.

Migration from Monolithic Orchestrator:
----------------------------------------
Before: AgentOrchestrator with 2000+ lines handling everything
After: AgentOrchestrator as facade delegating to 10+ specialized coordinators

Benefits:
---------
- Testability: Each coordinator can be tested independently
- Maintainability: Changes are localized to specific coordinators
- Reusability: Framework coordinators used across all verticals
- Readability: Clear separation of concerns

For framework layer coordinators, see victor.framework.coordinators.
For detailed architecture documentation, see docs/architecture/coordinator_separation.md
"""

from victor.agent.coordinators.base_config import BaseCoordinatorConfig
from victor.agent.coordinators.config_coordinator import (
    ConfigCoordinator,
    ValidationResult,
    OrchestratorConfig,
    SettingsConfigProvider,
    EnvironmentConfigProvider,
)

from victor.agent.coordinators.prompt_coordinator import (
    PromptCoordinator,
    PromptBuildError,
    BasePromptContributor,
    SystemPromptContributor,
    TaskHintContributor,
    IPromptBuilderCoordinator,
    PromptBuilderCoordinator,
)

from victor.agent.coordinators.context_coordinator import (
    ContextCoordinator,
    ContextCompactionError,
    BaseCompactionStrategy,
    TruncationCompactionStrategy,
    SummarizationCompactionStrategy,
    SemanticCompactionStrategy,
    HybridCompactionStrategy,
)

from victor.agent.coordinators.analytics_coordinator import (
    AnalyticsCoordinator,
    SessionAnalytics,
    BaseAnalyticsExporter,
    ConsoleAnalyticsExporter,
    FileAnalyticsExporter,
)

from victor.agent.coordinators.chat_coordinator import ChatCoordinator
from victor.agent.coordinators.tool_coordinator import (
    ToolCoordinator,
    ToolCoordinatorConfig,
    IToolCoordinator,
    TaskContext,
    create_tool_coordinator,
)

from victor.agent.coordinators.tool_selection_coordinator import (
    ToolSelectionCoordinator,
    IToolSelectionCoordinator,
)
from victor.agent.coordinators.session_coordinator import (
    SessionCoordinator,
    SessionInfo,
    SessionCostSummary,
)
from victor.agent.coordinators.provider_coordinator import (
    ProviderCoordinator,
    ProviderCoordinatorConfig,
    RateLimitInfo,
)
from victor.agent.coordinators.mode_coordinator import (
    ModeCoordinator,
)
from victor.agent.coordinators.checkpoint_coordinator import (
    CheckpointCoordinator,
)
from victor.agent.coordinators.evaluation_coordinator import (
    EvaluationCoordinator,
)
from victor.agent.coordinators.metrics_coordinator import (
    MetricsCoordinator,
)
from victor.agent.coordinators.workflow_coordinator import (
    WorkflowCoordinator,
)
from victor.agent.coordinators.state_coordinator import (
    StateCoordinator,
    StateScope,
    StateChange,
    StateObserver,
    create_state_coordinator,
)
from victor.agent.coordinators.response_coordinator import (
    ResponseCoordinator,
    IResponseCoordinator,
    ResponseCoordinatorConfig,
    ProcessedResponse,
    ChunkProcessResult,
    ToolCallValidationResult,
)
from victor.agent.coordinators.tool_execution_coordinator import (
    ToolExecutionCoordinator,
    ToolExecutionConfig,
    ToolCallResult as ToolExecutionCallResult,
    ToolExecutionStats,
    ExecutionContext,
    ToolAccessDecision,
    create_tool_execution_coordinator,
)
from victor.agent.coordinators.validation_coordinator import (
    ValidationCoordinator,
    ValidationCoordinatorConfig,
    ValidationResult as ValidationCoordinatorResult,
    IntelligentValidationResult,
    ToolCallValidationResult as ValidationToolCallResult,
    ContextValidationResult,
)

__all__ = [
    # Base configuration
    "BaseCoordinatorConfig",
    # ConfigCoordinator
    "ConfigCoordinator",
    "ValidationResult",
    "OrchestratorConfig",
    "SettingsConfigProvider",
    "EnvironmentConfigProvider",
    # PromptCoordinator
    "PromptCoordinator",
    "PromptBuildError",
    "BasePromptContributor",
    "SystemPromptContributor",
    "TaskHintContributor",
    "IPromptBuilderCoordinator",
    "PromptBuilderCoordinator",
    # ContextCoordinator
    "ContextCoordinator",
    "ContextCompactionError",
    "BaseCompactionStrategy",
    "TruncationCompactionStrategy",
    "SummarizationCompactionStrategy",
    "SemanticCompactionStrategy",
    "HybridCompactionStrategy",
    # AnalyticsCoordinator
    "AnalyticsCoordinator",
    "SessionAnalytics",
    "BaseAnalyticsExporter",
    "ConsoleAnalyticsExporter",
    "FileAnalyticsExporter",
    # ChatCoordinator
    "ChatCoordinator",
    # ToolCoordinator
    "ToolCoordinator",
    "ToolCoordinatorConfig",
    "IToolCoordinator",
    "TaskContext",
    "create_tool_coordinator",
    # ToolSelectionCoordinator
    "ToolSelectionCoordinator",
    "IToolSelectionCoordinator",
    # SessionCoordinator
    "SessionCoordinator",
    "SessionInfo",
    "SessionCostSummary",
    # ProviderCoordinator
    "ProviderCoordinator",
    "ProviderCoordinatorConfig",
    "RateLimitInfo",
    # ModeCoordinator
    "ModeCoordinator",
    # CheckpointCoordinator
    "CheckpointCoordinator",
    # EvaluationCoordinator
    "EvaluationCoordinator",
    # MetricsCoordinator
    "MetricsCoordinator",
    # WorkflowCoordinator
    "WorkflowCoordinator",
    # StateCoordinator
    "StateCoordinator",
    "StateScope",
    "StateChange",
    "StateObserver",
    "create_state_coordinator",
    # ResponseCoordinator
    "ResponseCoordinator",
    "IResponseCoordinator",
    "ResponseCoordinatorConfig",
    "ProcessedResponse",
    "ChunkProcessResult",
    "ToolCallValidationResult",
    # ToolExecutionCoordinator
    "ToolExecutionCoordinator",
    "ToolExecutionConfig",
    "ToolExecutionCallResult",
    "ToolExecutionStats",
    "ExecutionContext",
    "ToolAccessDecision",
    "create_tool_execution_coordinator",
    # ValidationCoordinator
    "ValidationCoordinator",
    "ValidationCoordinatorConfig",
    "ValidationCoordinatorResult",
    "IntelligentValidationResult",
    "ValidationToolCallResult",
    "ContextValidationResult",
]
