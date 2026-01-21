# Victor AI Architecture

Comprehensive system architecture documentation for Victor AI, including components, data flows, integration points, and extension mechanisms.

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Component Diagram](#component-diagram)
- [Core Components](#core-components)
- [Data Flows](#data-flows)
- [Integration Points](#integration-points)
- [Extension Points](#extension-points)
- [Technology Stack](#technology-stack)
- [Design Patterns](#design-patterns)
- [SOLID Principles](#solid-principles)

## Overview

Victor AI is an open-source AI coding assistant with a modular, event-driven architecture supporting 21 LLM providers, 55 specialized tools, and 5 domain verticals.

**Key Architectural Principles:**
- **Provider Agnostic**: Unified interface for 21+ LLM providers
- **Event-Driven**: Pluggable event backends for scalability
- **Protocol-First**: Loose coupling via protocols (interfaces)
- **SOLID-Compliant**: Following SRP, OCP, LSP, ISP, DIP
- **Vertical Architecture**: Self-contained domain verticals
- **Dependency Injection**: ServiceContainer manages 55+ services

**Architecture Layers:**
```
┌─────────────────────────────────────────────────────────┐
│                     Clients Layer                       │
│  CLI │ TUI │ VS Code │ MCP Server │ API Server         │
└─────────────────────┬───────────────────────────────────┘
                      │ (HTTP/stdio/messaging)
                      ▼
┌─────────────────────────────────────────────────────────┐
│                ServiceContainer (DI)                     │
│  55+ registered services (singleton, scoped, transient) │
└─────────────────────┬───────────────────────────────────┘
                      │ (Dependency resolution)
                      ▼
┌─────────────────────────────────────────────────────────┐
│              AgentOrchestrator (Facade)                  │
│  Public API: chat(), stream_chat(), execute_tool()      │
└─────────────────────┬───────────────────────────────────┘
                      │ (Delegation)
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
┌──────────────┐ ┌──────────┐ ┌──────────────┐
│ Coordinators │ │ Handlers │ │  Services    │
└──────────────┘ └──────────┘ └──────────────┘
        │             │             │
        └─────────────┼─────────────┘
                      │ (Events/Messages)
                      ▼
┌─────────────────────────────────────────────────────────┐
│                   Event Bus (Pub/Sub)                    │
│  Topics: tool.*, agent.*, workflow.*, error.*           │
└─────────────────────┬───────────────────────────────────┘
                      │ (Backend abstraction)
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Event Backends (Pluggable)                  │
│  In-Memory │ Kafka │ SQS │ RabbitMQ │ Redis              │
└─────────────────────────────────────────────────────────┘
```

## System Architecture

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         CLIENTS                                  │
├──────────────────────────────────────────────────────────────────┤
│  CLI/TUI (victor/chat)  │  VS Code (HTTP)  │  MCP (stdio)       │
│  - Rich CLI             │  - Language Server│  - Tool Server    │
│  - Textual TUI          │  - Code Lens      │  - Resources      │
└────────────┬─────────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────────┐
│                     SERVICE CONTAINER                            │
│  Manages 55+ services:                                           │
│  - Tool Registry (55 tools)                                      │
│  - Provider Manager (21 providers)                               │
│  - Event Bus (5 backends)                                        │
│  - Memory Systems (SQLite, LanceDB)                              │
│  - Coordinators (20+ coordinators)                               │
└────────────┬─────────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────────┐
│                  AGENT ORCHESTRATOR (Facade)                     │
│  - Public API: chat(), stream_chat(), execute_tool()             │
│  - Delegates to coordinators                                     │
│  - No business logic                                             │
└────────────┬─────────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────────┐
│                       COORDINATORS                               │
│  ┌──────────────┬──────────────┬──────────────┬─────────────┐  │
│  │ Tool         │ Prompt       │ State        │ Chat        │  │
│  │ Coordinator  │ Coordinator  │ Coordinator  │ Coordinator │  │
│  └──────────────┴──────────────┴──────────────┴─────────────┘  │
│  ┌──────────────┬──────────────┬──────────────┬─────────────┐  │
│  │ Memory       │ Config       │ Analytics    │ Workflow    │  │
│  │ Coordinator  │ Coordinator  │ Coordinator  │ Coordinator │  │
│  └──────────────┴──────────────┴──────────────┴─────────────┘  │
└────────────┬─────────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────────┐
│                    SERVICES & PROVIDERS                          │
│  ┌──────────────┬──────────────┬──────────────┬─────────────┐  │
│  │ Tool         │ Provider     │ Event        │ Vector      │  │
│  │ Registry     │ Pool         │ Bus          │ Stores      │  │
│  └──────────────┴──────────────┴──────────────┴─────────────┘  │
│  ┌──────────────┬──────────────┬──────────────┬─────────────┐  │
│  │ Memory       │ Checkpoint   │ Embedding    │ RL          │  │
│  │ Manager      │ Manager      │ Service      │ Manager     │  │
│  └──────────────┴──────────────┴──────────────┴─────────────┘  │
└────────────┬─────────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────────┐
│                    EVENT BUS LAYER                              │
│  - In-Memory (default)                                          │
│  - SQLite (persistent)                                          │
│  - Redis (distributed)                                          │
│  - Kafka (high-throughput)                                      │
│  - SQS (serverless)                                             │
│  - RabbitMQ (traditional)                                       │
└────────────┬─────────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────────┐
│                      INTEGRATIONS                                │
│  ┌──────────────┬──────────────┬──────────────┬─────────────┐  │
│  │ LLM          │ Vector       │ Database     │ Cache       │  │
│  │ Providers    │ Stores       │ (SQLite/PG)  │ (Disk/Redis)│  │
│  └──────────────┴──────────────┴──────────────┴─────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

## Component Diagram

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                     victor/agent/                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  AgentOrchestrator (Facade)                                 │
│  ├── tool_pipeline.py: Tool execution pipeline              │
│  ├── conversation_controller.py: Conversation management    │
│  ├── streaming_controller.py: Streaming response handler    │
│  ├── provider_manager.py: Provider lifecycle management     │
│  └── tool_registrar.py: Tool registration                   │
│                                                              │
│  coordinators/ (SRP - Single Responsibility)                │
│  ├── tool_coordinator.py: Tool selection & execution        │
│  ├── prompt_coordinator.py: Prompt building & assembly      │
│  ├── state_coordinator.py: Conversation state management    │
│  ├── chat_coordinator.py: Chat message handling             │
│  ├── memory_coordinator.py: Memory retrieval & storage      │
│  ├── config_coordinator.py: Configuration management        │
│  ├── analytics_coordinator.py: Metrics & analytics          │
│  └── workflow_coordinator.py: Workflow execution            │
│                                                              │
│  protocols/ (ISP - Interface Segregation)                    │
│  ├── ToolExecutorProtocol: Tool execution interface         │
│  ├── PromptBuilderProtocol: Prompt building interface       │
│  ├── StateManagerProtocol: State management interface       │
│  ├── ChatHandlerProtocol: Chat handling interface           │
│  └── 15+ more protocols                                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    victor/providers/                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  BaseProvider (Abstract Base)                                │
│  ├── chat(): Send message, get response                     │
│  ├── stream_chat(): Stream response chunks                  │
│  ├── supports_tools(): Check tool calling support           │
│  └── name(): Provider identifier                            │
│                                                              │
│  Provider Implementations (DIP - Dependency Inversion)      │
│  ├── anthropic.py: Anthropic Claude                         │
│  ├── openai.py: OpenAI GPT                                  │
│  ├── google.py: Google Gemini                               │
│  ├── ollama.py: Ollama (local)                              │
│  ├── lmstudio.py: LM Studio (local)                         │
│  └── 16 more providers                                     │
│                                                              │
│  tool_calling/ (Adapter Pattern)                            │
│  ├── base.py: Tool calling adapter interface               │
│  ├── anthropic_adapter.py: Anthropic tool calling           │
│  ├── openai_adapter.py: OpenAI tool calling                 │
│  └── 15+ more adapters                                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      victor/tools/                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  BaseTool (Abstract Base)                                    │
│  ├── name: Tool identifier                                   │
│  ├── description: Tool description                          │
│  ├── parameters: JSON Schema                                │
│  ├── cost_tier: FREE/LOW/MEDIUM/HIGH                        │
│  └── execute(): Tool execution logic                        │
│                                                              │
│  Tool Categories (OCP - Open/Closed Principle)              │
│  ├── filesystem/: File operations                           │
│  ├── search/: Code search                                   │
│  ├── analysis/: Code analysis                               │
│  ├── editing/: Code editing                                 │
│  ├── testing/: Test generation & execution                  │
│  └── 10 more categories                                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     victor/core/                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  container.py (Dependency Injection)                         │
│  ├── ServiceContainer: DI container                         │
│  ├── register(): Register service                          │
│  ├── get(): Resolve service                                │
│  └── 55+ registered services                                │
│                                                              │
│  events/ (Event-Driven Architecture)                        │
│  ├── __init__.py: Event backend factory                     │
│  ├── messaging.py: Messaging event types                    │
│  ├── backends/: Event backend implementations               │
│  │   ├── in_memory.py: In-memory backend                   │
│  │   ├── sqlite.py: SQLite backend                         │
│  │   ├── redis.py: Redis backend                           │
│  │   ├── kafka.py: Kafka backend                           │
│  │   ├── sqs.py: AWS SQS backend                           │
│  │   └── rabbitmq.py: RabbitMQ backend                     │
│  └── protocols.py: Event interfaces                        │
│                                                              │
│  registries/ (Universal Registry System)                    │
│  ├── universal_registry.py: Type-safe registry             │
│  ├── CacheStrategy: TTL, LRU, Manual, None                 │
│  └── 15+ specialized registries                             │
│                                                              │
│  verticals/ (Vertical Architecture)                         │
│  ├── base.py: VerticalBase abstract class                  │
│  ├── lazy_extensions.py: Lazy loading support              │
│  └── 5 domain verticals                                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   victor/framework/                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  graph.py: StateGraph DSL (LangGraph-compatible)           │
│  ├── StateGraph: Workflow definition                       │
│  ├── CompiledGraph: Compiled workflow                      │
│  ├── Node: Workflow node                                   │
│  └── Edge: Workflow edge                                   │
│                                                              │
│  coordinators/: Framework coordinators                       │
│  ├── YAMLWorkflowCoordinator: YAML workflows               │
│  ├── HITLCoordinator: Human-in-the-loop                    │
│  ├── CacheCoordinator: Workflow caching                    │
│  └── GraphCoordinator: StateGraph workflows                │
│                                                              │
│  capabilities/: Generic capability providers                │
│  ├── ConfigurationCapabilityProvider: Config management    │
│  ├── FileOperationsCapability: File operations             │
│  ├── PrivacyCapabilityProvider: Privacy controls           │
│  └── PromptContributionCapability: Prompt sections         │
│                                                              │
│  resilience/: Unified retry framework                       │
│  ├── ExponentialBackoffStrategy: Retry strategy            │
│  ├── CircuitBreaker: Circuit breaker pattern               │
│  └── with_retry: Retry decorator                           │
│                                                              │
│  parallel/: Generic parallel execution                      │
│  ├── ParallelExecutor: Concurrent task execution           │
│  ├── JoinStrategy: Result aggregation                      │
│  └── ErrorStrategy: Error handling                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. AgentOrchestrator (Facade)

**Purpose**: Central facade coordinating all agent operations

**Responsibilities**:
- Public API: `chat()`, `stream_chat()`, `execute_tool()`
- Delegate to coordinators (no business logic)
- Request/response handling
- Error handling and recovery

**Location**: `victor/agent/orchestrator.py`

**Key Methods**:
```python
class AgentOrchestrator:
    async def chat(self, query: str) -> str:
        """Process a single query and return response."""

    async def stream_chat(self, query: str) -> AsyncIterator[StreamChunk]:
        """Stream response chunks in real-time."""

    async def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute a specific tool."""
```

**Dependencies** (via protocols):
- `ToolCoordinatorProtocol`: Tool operations
- `PromptCoordinatorProtocol`: Prompt building
- `StateCoordinatorProtocol`: State management
- `ChatCoordinatorProtocol`: Chat handling

### 2. ToolCoordinator

**Purpose**: Coordinate tool selection and execution

**Responsibilities**:
- Tool selection based on query and context
- Budget checking and enforcement
- Tool execution with caching
- Result aggregation

**Location**: `victor/agent/coordinators/tool_coordinator.py`

**Key Methods**:
```python
class ToolCoordinator:
    async def select_and_execute(
        self,
        query: str,
        context: AgentToolSelectionContext,
    ) -> ToolResult:
        """Select and execute tools for a query."""

    async def select_tools(
        self,
        query: str,
        context: AgentToolSelectionContext,
    ) -> List[BaseTool]:
        """Select relevant tools."""

    async def execute_tools(
        self,
        tools: List[BaseTool],
        context: Dict[str, Any],
    ) -> List[ToolResult]:
        """Execute selected tools."""
```

### 3. PromptCoordinator

**Purpose**: Coordinate prompt building and assembly

**Responsibilities**:
- Prompt template management
- Context injection
- Tool hint generation
- Prompt enrichment

**Location**: `victor/agent/coordinators/prompt_coordinator.py`

**Key Methods**:
```python
class PromptCoordinator:
    async def build_prompt(
        self,
        query: str,
        tools: List[BaseTool],
        context: PromptContext,
    ) -> str:
        """Build complete prompt for LLM."""

    async def enrich_prompt(
        self,
        prompt: str,
        context: PromptContext,
    ) -> str:
        """Enrich prompt with contextual information."""
```

### 4. StateCoordinator

**Purpose**: Coordinate conversation state management

**Responsibilities**:
- Stage transitions (INITIAL → PLANNING → READING → ...)
- State persistence and recovery
- Context aggregation
- Memory retrieval

**Location**: `victor/agent/coordinators/state_coordinator.py`

**Key Methods**:
```python
class StateCoordinator:
    async def get_state(self) -> ConversationState:
        """Get current conversation state."""

    async def transition_stage(
        self,
        new_stage: ConversationStage,
    ) -> None:
        """Transition to new conversation stage."""

    async def persist_state(self) -> None:
        """Persist state to storage."""
```

### 5. ProviderManager

**Purpose**: Manage LLM provider lifecycle

**Responsibilities**:
- Provider instantiation
- Provider switching
- Health monitoring
- Circuit breaking

**Location**: `victor/agent/provider_manager.py`

**Key Methods**:
```python
class ProviderManager:
    def get_provider(self, provider_name: str) -> BaseProvider:
        """Get provider instance by name."""

    async def switch_provider(
        self,
        new_provider: str,
        new_model: str,
    ) -> None:
        """Switch to different provider."""

    async def check_health(self, provider: BaseProvider) -> bool:
        """Check provider health."""
```

### 6. ToolRegistry

**Purpose**: Register and discover tools

**Responsibilities**:
- Tool registration
- Tool discovery by category
- Tool metadata management
- Tool alias resolution

**Location**: `victor/agent/tool_registrar.py`

**Key Methods**:
```python
class SharedToolRegistry:
    def register_tool(self, tool: BaseTool) -> None:
        """Register a new tool."""

    def get_tool(self, name: str) -> BaseTool:
        """Get tool by name."""

    def get_tools_by_category(
        self,
        category: str,
    ) -> List[BaseTool]:
        """Get all tools in a category."""

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
```

### 7. EventBus

**Purpose**: Event-driven communication

**Responsibilities**:
- Event publishing
- Event subscription
- Backend abstraction
- Event filtering

**Location**: `victor/core/events/`

**Key Methods**:
```python
class IEventBackend(Protocol):
    async def publish(self, event: MessagingEvent) -> None:
        """Publish event to a topic."""

    async def subscribe(
        self,
        topic: str,
        handler: EventHandler,
    ) -> None:
        """Subscribe to events on a topic."""

    async def connect(self) -> None:
        """Connect to backend."""
```

## Data Flows

### Chat Request Flow

```
┌─────────┐
│ Client  │
└────┬────┘
     │ 1. chat(query)
     ▼
┌─────────────────────┐
│ AgentOrchestrator   │
└────┬────────────────┘
     │ 2. delegate to coordinators
     ├────────────────────────────────┐
     │                                │
     ▼                                ▼
┌──────────────────┐        ┌──────────────────┐
│ ToolCoordinator  │        │ StateCoordinator │
└────┬─────────────┘        └──────────────────┘
     │ 3. select_tools()
     ▼
┌──────────────────┐
│ ToolRegistry     │
└────┬─────────────┘
     │ 4. return tools
     ▼
┌─────────────────────┐
│ ToolCoordinator     │
└────┬────────────────┘
     │ 5. tools selected
     ▼
┌──────────────────┐
│ PromptCoordinator│
└────┬─────────────┘
     │ 6. build_prompt()
     ▼
┌──────────────────┐
│ ProviderManager  │
└────┬─────────────┘
     │ 7. chat(prompt)
     ▼
┌──────────────────┐
│ LLM Provider     │
└────┬─────────────┘
     │ 8. response
     ▼
┌─────────────────────┐
│ AgentOrchestrator   │
└────┬────────────────┘
     │ 9. parse tool_calls
     ▼
┌─────────────────────┐
│ ToolCoordinator     │
└────┬────────────────┘
     │ 10. execute_tools()
     ▼
┌──────────────────┐
│ Tools            │
└────┬─────────────┘
     │ 11. tool_results
     ▼
┌─────────────────────┐
│ AgentOrchestrator   │
└────┬────────────────┘
     │ 12. final_response
     ▼
┌─────────┐
│ Client  │
└─────────┘
```

### Tool Execution Flow

```
┌──────────────────┐
│ ToolCoordinator  │
└────┬─────────────┘
     │ 1. execute_tool(tool, args)
     ▼
┌─────────────────────┐
│ BudgetManager       │
└────┬────────────────┘
     │ 2. check_budget()
     ├────✓ allowed
     │
     ▼
┌─────────────────────┐
│ ToolCache           │
└────┬────────────────┘
     │ 3. check_cache()
     ├────✗ cache miss
     │
     ▼
┌─────────────────────┐
│ ToolDeduplication   │
└────┬────────────────┘
     │ 4. check_duplicates()
     ├────✗ not duplicate
     │
     ▼
┌─────────────────────┐
│ ToolExecutor        │
└────┬────────────────┘
     │ 5. execute(**args)
     ▼
┌─────────────────────┐
│ Tool                │
└────┬────────────────┘
     │ 6. result
     ▼
┌─────────────────────┐
│ ToolCache           │
└────┬────────────────┘
     │ 7. cache_result()
     ▼
┌──────────────────┐
│ ToolCoordinator  │
└────┬─────────────┘
     │ 8. return result
     ▼
┌──────────────────┐
│ EventBus         │
└────┬─────────────┘
     │ 9. publish("tool.complete")
     ▼
┌──────────────────┐
│ Subscribers      │
└──────────────────┘
```

### Event Flow

```
┌─────────────────────┐
│ Event Publisher    │
│ (any component)    │
└────┬────────────────┘
     │ 1. publish(event)
     ▼
┌─────────────────────┐
│ EventBus           │
└────┬────────────────┘
     │ 2. route_to_backend()
     │
     ├─────────────────────┬─────────────────────┐
     │                     │                     │
     ▼                     ▼                     ▼
┌──────────┐      ┌──────────┐      ┌──────────┐
│ In-Memory│      │  SQLite  │      │  Redis   │
└──────────┘      └──────────┘      └──────────┘
     │                     │                     │
     └─────────────────────┴─────────────────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Event Subscribers│
                  │ - Analytics      │
                  │ - Logging        │
                  │ - Monitoring     │
                  └──────────────────┘
```

## Integration Points

### 1. Provider Integration

**Add New LLM Provider**:

1. Create provider class inheriting `BaseProvider`:
```python
# victor/providers/my_provider.py
from victor.providers.base import BaseProvider

class MyProvider(BaseProvider):
    async def chat(self, prompt: str) -> str:
        # Implementation
        pass

    async def stream_chat(self, prompt: str) -> AsyncIterator[StreamChunk]:
        # Implementation
        pass

    def supports_tools(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "my_provider"
```

2. Register in `ProviderRegistry`:
```python
# victor/providers/registry.py
from victor.providers.my_provider import MyProvider

register_provider("my_provider", MyProvider)
```

3. Add tool calling adapter (if supported):
```python
# victor/agent/tool_calling/my_provider_adapter.py
from victor.agent.tool_calling.base import BaseToolCallingAdapter

class MyProviderToolCallingAdapter(BaseToolCallingAdapter):
    def convert_tools(self, tools: List[BaseTool]) -> List[Dict]:
        # Convert tools to provider format
        pass

    def parse_tool_calls(self, response: str) -> List[ToolCall]:
        # Parse tool calls from response
        pass
```

4. Add model capabilities:
```yaml
# victor/config/model_capabilities.yaml
models:
  "my-model*":
    training:
      tool_calling: true
    providers:
      my_provider:
        native_tool_calls: true
        parallel_tool_calls: true
```

### 2. Tool Integration

**Add New Tool**:

1. Create tool class inheriting `BaseTool`:
```python
# victor/tools/my_tool.py
from victor.tools.base import BaseTool

class MyTool(BaseTool):
    name = "my_tool"
    description = "Does something useful"

    parameters = {
        "type": "object",
        "properties": {
            "param1": {
                "type": "string",
                "description": "First parameter",
            },
        },
        "required": ["param1"],
    }

    cost_tier = CostTier.LOW

    async def execute(self, **kwargs) -> str:
        # Tool logic
        return "result"
```

2. Register in tool registry:
```python
# victor/tools/__init__.py
from victor.tools.my_tool import MyTool

SharedToolRegistry.register_tool(MyTool())
```

3. Add to tool catalog:
```bash
python scripts/generate_tool_catalog.py
```

### 3. Vertical Integration

**Add New Domain Vertical**:

1. Create vertical class inheriting `VerticalBase`:
```python
# victor/my_vertical/__init__.py
from victor.core.verticals.base import VerticalBase

class MyVertical(VerticalBase):
    name = "my_vertical"
    description = "My custom vertical"

    @classmethod
    def get_tools(cls) -> List[str]:
        return ["tool1", "tool2", "tool3"]

    @classmethod
    def get_system_prompt(cls) -> str:
        return "You are an expert in my domain..."
```

2. Register via entry point:
```toml
# pyproject.toml
[project.entry-points."victor.verticals"]
my_vertical = "victor.my_vertical:MyVertical"
```

3. Create mode config (optional):
```yaml
# victor/config/modes/my_vertical_modes.yaml
vertical_name: my_vertical
default_mode: standard
modes:
  standard:
    name: standard
    exploration: standard
    edit_permission: full
```

### 4. Event Backend Integration

**Add New Event Backend**:

1. Implement `IEventBackend` protocol:
```python
# victor/core/events/backends/my_backend.py
from victor.core.events.protocols import IEventBackend

class MyEventBackend:
    async def publish(self, event: MessagingEvent) -> None:
        # Publish to my backend
        pass

    async def subscribe(
        self,
        topic: str,
        handler: EventHandler,
    ) -> None:
        # Subscribe to events
        pass

    async def connect(self) -> None:
        # Connect to backend
        pass
```

2. Register in factory:
```python
# victor/core/events/__init__.py
def create_event_backend(config: BackendConfig) -> IEventBackend:
    if config.backend_type == "my_backend":
        return MyEventBackend(config)
    # ... other backends
```

## Extension Points

### 1. Custom Coordinators

Create specialized coordinators for complex operations:

```python
# victor/agent/coordinators/my_coordinator.py
from victor.agent.coordinators.base_config import CoordinatorConfig

class MyCoordinator:
    def __init__(self, config: CoordinatorConfig):
        self._config = config

    async def execute(self, context: Dict[str, Any]) -> Any:
        # Custom coordination logic
        pass
```

### 2. Custom Middleware

Add middleware for preprocessing/postprocessing:

```python
# victor/agent/middleware/my_middleware.py
from victor.agent.middleware_chain import MiddlewareChain

class MyMiddleware:
    async def process(
        self,
        request: Any,
        call_next: Callable,
    ) -> Any:
        # Preprocess
        result = await call_next(request)
        # Postprocess
        return result
```

### 3. Custom Prompt Sections

Add reusable prompt sections:

```python
# victor/framework/prompt_sections.py
from victor.framework.prompt_builder import PromptSection

class MyPromptSection(PromptSection):
    def render(self, context: Dict[str, Any]) -> str:
        return """
# My Custom Section

Custom instructions here...
"""
```

### 4. Custom Workflow Nodes

Create custom workflow nodes for StateGraph:

```python
# victor/my_vertical/workflows/nodes.py
from victor.framework import Task, TaskResult

async def my_node(state: AgentState) -> TaskResult:
    # Custom node logic
    return TaskResult(
        status="success",
        data={"key": "value"},
    )
```

## Technology Stack

### Core Technologies

- **Language**: Python 3.10+ (type hints, async/await)
- **Async Framework**: asyncio, aiohttp, httpx
- **Configuration**: Pydantic, Pydantic-Settings, python-dotenv
- **CLI**: Typer, Rich, Prompt-Toolkit, Textual
- **Testing**: pytest, pytest-asyncio, pytest-cov

### LLM Integration

- **Anthropic**: anthropic>=0.34
- **OpenAI**: openai>=1.40
- **Google**: google-genai>=1.0 (optional)
- **Local**: Ollama, LM Studio, vLLM (no API key required)

### Data Storage

- **Vector Stores**: LanceDB (default), ChromaDB (alternative)
- **Databases**: SQLite (default), PostgreSQL (optional)
- **Caching**: diskcache, cachetools
- **Embeddings**: sentence-transformers (local), optional APIs

### Code Analysis

- **AST Parsing**: tree-sitter (Python + Rust accelerator)
- **Language Support**: 15+ tree-sitter grammars
- **Syntax Highlighting**: Pygments

### Observability

- **Logging**: Python logging, structured logging (JSONL)
- **Metrics**: OpenTelemetry, Prometheus (optional)
- **Tracing**: OpenTelemetry (optional)
- **Profiling**: built-in performance profiler

## Design Patterns

### 1. Facade Pattern

**AgentOrchestrator** provides a simplified interface to complex subsystems:

```python
# Facade hides complexity
orchestrator = AgentOrchestrator()
response = await orchestrator.chat("Hello")  # Simple interface

# Internally delegates to multiple coordinators
# - ToolCoordinator
# - PromptCoordinator
# - StateCoordinator
# - ProviderManager
# - etc.
```

### 2. Strategy Pattern

**Tool Selection Strategy** allows different algorithms:

```python
# Strategy interface
class IToolSelector(Protocol):
    async def select_tools(self, query: str) -> List[BaseTool]:
        ...

# Different strategies
class KeywordToolSelector:
    # Keyword-based selection

class SemanticToolSelector:
    # Semantic similarity selection

class HybridToolSelector:
    # Hybrid approach
```

### 3. Adapter Pattern

**Tool Calling Adapters** normalize provider-specific APIs:

```python
# Adapter interface
class BaseToolCallingAdapter:
    def convert_tools(self, tools: List[BaseTool]) -> List[Dict]:
        ...

# Provider-specific adapters
class AnthropicToolCallingAdapter(BaseToolCallingAdapter):
    # Anthropic-specific format

class OpenAIToolCallingAdapter(BaseToolCallingAdapter):
    # OpenAI-specific format
```

### 4. Observer Pattern

**Event Bus** implements pub/sub messaging:

```python
# Publisher
await event_bus.publish(
    MessagingEvent(
        topic="tool.complete",
        data={"tool": "read_file", "result": "..."},
    )
)

# Subscriber
await event_bus.subscribe("tool.*", my_handler)
```

### 5. Dependency Injection

**ServiceContainer** manages dependencies:

```python
# Register service
container.register(
    ToolCoordinatorProtocol,
    lambda c: ToolCoordinator(tool_registry=c.get(ToolRegistryProtocol)),
    ServiceLifetime.SINGLETON,
)

# Resolve service
tool_coordinator = container.get(ToolCoordinatorProtocol)
```

### 6. Template Method Pattern

**BaseYAMLWorkflowProvider** defines workflow structure:

```python
class BaseYAMLWorkflowProvider:
    def compile_workflow(self, name: str) -> CompiledGraph:
        # Template method defining workflow compilation
        definition = self._load_definition(name)
        graph = self._build_graph(definition)
        return self._compile_graph(graph)

    def _load_definition(self, name: str) -> Dict:
        # Subclass implements
        pass

    def _build_graph(self, definition: Dict) -> StateGraph:
        # Subclass implements
        pass
```

### 7. Chain of Responsibility

**Middleware Chain** processes requests sequentially:

```python
class MiddlewareChain:
    async def process(self, request: Any) -> Any:
        for middleware in self._middlewares:
            request = await middleware.process(
                request,
                lambda r: self._next_middleware(r, middleware),
            )
        return request
```

### 8. Factory Pattern

**Event Backend Factory** creates backends:

```python
def create_event_backend(config: BackendConfig) -> IEventBackend:
    if config.backend_type == "in_memory":
        return InMemoryBackend(config)
    elif config.backend_type == "sqlite":
        return SQLiteBackend(config)
    elif config.backend_type == "redis":
        return RedisBackend(config)
    # ... etc
```

## SOLID Principles

### Single Responsibility Principle (SRP)

Each component has one reason to change:

- **ToolCoordinator**: Only tool operations
- **PromptCoordinator**: Only prompt building
- **StateCoordinator**: Only state management
- **ChatCoordinator**: Only chat handling

### Open/Closed Principle (OCP)

System is open for extension, closed for modification:

- **New Providers**: Inherit `BaseProvider`, no core changes needed
- **New Tools**: Inherit `BaseTool`, register in registry
- **New Verticals**: Inherit `VerticalBase`, register via entry point
- **New Coordinators**: Implement protocol, inject via DI

### Liskov Substitution Principle (LSP)

Subtypes are substitutable for their base types:

- **Providers**: Any `BaseProvider` can be used interchangeably
- **Tools**: Any `BaseTool` works with tool pipeline
- **Coordinators**: Any coordinator protocol implementation works

### Interface Segregation Principle (ISP)

Clients depend only on interfaces they use:

- **ToolExecutorProtocol**: Only tool execution methods
- **PromptBuilderProtocol**: Only prompt building methods
- **StateManagerProtocol**: Only state management methods
- **15+ more protocols**: Fine-grained, focused interfaces

### Dependency Inversion Principle (DIP)

High-level modules depend on abstractions, not concretions:

```python
# High-level module depends on protocol
class ToolCoordinator:
    def __init__(
        self,
        tool_executor: ToolExecutorProtocol,  # Protocol, not concrete
    ):
        self._tool_executor = tool_executor

# Any implementation can be injected
executor = ToolExecutor()  # Concrete implementation
coordinator = ToolCoordinator(tool_executor=executor)
```

## Additional Resources

- **Production Deployment**: See [PRODUCTION_DEPLOYMENT_GUIDE.md](PRODUCTION_DEPLOYMENT_GUIDE.md)
- **Feature Guide**: See [FEATURE_GUIDE.md](FEATURE_GUIDE.md)
- **Migration Guide**: See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)
- **API Reference**: See [API_REFERENCE.md](API_REFERENCE.md)

## Support

- **GitHub Issues**: https://github.com/vijayksingh/victor/issues
- **Discord Community**: https://discord.gg/victor-ai
- **Documentation**: https://docs.victor-ai.com
