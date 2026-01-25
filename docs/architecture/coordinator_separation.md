# Coordinator Architecture: Application vs Framework Layer

## Executive Summary

Victor employs a **two-layer coordinator architecture** that separates application-specific orchestration from framework-agnostic workflow infrastructure. This design is **intentional and beneficial**, not harmful duplication.

- **Application Layer** (`victor/agent/coordinators/`): Manages AI agent conversation lifecycle
- **Framework Layer** (`victor/framework/coordinators/`): Provides reusable workflow infrastructure

This separation enables:
- **Single Responsibility Principle** (SRP): Each coordinator has one clear purpose
- **Layered Architecture**: Application logic builds on framework foundation
- **Reusability**: Framework coordinators work across all verticals
- **Testability**: Coordinators can be tested independently
- **Maintainability**: Clear boundaries reduce coupling

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     APPLICATION LAYER                            │
│                  victor/agent/coordinators/                     │
│                                                                   │
│  Manages AI agent conversation lifecycle and orchestration      │
│                                                                   │
│  • ChatCoordinator: LLM chat & streaming                       │
│  • ToolCoordinator: Tool validation & execution                 │
│  • ContextCoordinator: Context management                       │
│  • AnalyticsCoordinator: Session metrics                        │
│  • PromptCoordinator: System prompt building                    │
│  • SessionCoordinator: Session lifecycle                        │
│  • ProviderCoordinator: Provider switching                      │
│  • ModeCoordinator: Agent modes (build/plan/explore)            │
│  • ConfigCoordinator: Configuration loading                     │
│  • ToolSelectionCoordinator: Semantic tool selection            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ uses
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FRAMEWORK LAYER                             │
│                 victor/framework/coordinators/                  │
│                                                                   │
│  Provides domain-agnostic workflow infrastructure                │
│                                                                   │
│  • YAMLWorkflowCoordinator: YAML workflow execution             │
│  • GraphExecutionCoordinator: StateGraph execution              │
│  • HITLCoordinator: Human-in-the-loop integration               │
│  • CacheCoordinator: Workflow caching                           │
└─────────────────────────────────────────────────────────────────┘
```

## Application Layer: victor/agent/coordinators/

### Purpose

Application-specific orchestration for the AI agent conversation lifecycle. These coordinators manage the high-level logic of how Victor interacts with users and orchestrates the complete conversation flow.

### Key Coordinators

#### ChatCoordinator
**Responsibility**: Manages LLM chat operations and streaming

```python
from victor.agent.coordinators import ChatCoordinator

coordinator = ChatCoordinator(orchestrator)
response = await coordinator.chat(
    messages=[...],
    tools=[...],
    stream=True
)
```

**Key Features**:
- Non-streaming chat with agentic loop
- Streaming chat with iteration management
- Tool calling and execution coordination
- Response validation and error recovery
- Intent classification and continuation handling

**SOLID Compliance**:
- **SRP**: Only handles chat operations (not configuration, tools, or context)
- **DIP**: Depends on orchestrator protocol, not concrete implementation

---

#### ToolCoordinator
**Responsibility**: Tool validation, execution coordination, and budget enforcement

```python
from victor.agent.coordinators import ToolCoordinator

coordinator = ToolCoordinator(
    tool_pipeline=pipeline,
    tool_selector=selector,
    tool_registry=registry,
    budget_manager=budget_mgr
)

# Select tools for current context
tools = await coordinator.select_tools(context)

# Execute tool calls
results = await coordinator.execute_tool_calls(tool_calls)
```

**Key Features**:
- Semantic + keyword-based tool selection
- Tool budget management and enforcement
- Tool access control and enable/disable
- Tool execution coordination via ToolPipeline
- Tool alias resolution and shell variant handling
- Tool call parsing and validation

**SOLID Compliance**:
- **SRP**: Coordinates all tool-related operations
- **ISP**: Focused interfaces (IToolCoordinator)
- **OCP**: Extensible through new tool strategies

---

#### ContextCoordinator
**Responsibility**: Context management and compaction strategies

```python
from victor.agent.coordinators import ContextCoordinator

coordinator = ContextCoordinator(
    max_tokens=8000,
    strategy="hybrid"
)

# Compact context when needed
compacted = await coordinator.compact_context(messages)
```

**Key Features**:
- Multiple compaction strategies (truncation, summarization, semantic, hybrid)
- Context window management
- Message prioritization and importance scoring
- Token counting and budget enforcement

**SOLID Compliance**:
- **SRP**: Only manages context (not chat, tools, or configuration)
- **OCP**: New compaction strategies via BaseCompactionStrategy

---

#### AnalyticsCoordinator
**Responsibility**: Session analytics and metrics collection

```python
from victor.agent.coordinators import AnalyticsCoordinator

coordinator = AnalyticsCoordinator()

# Track session metrics
coordinator.track_tool_call(tool_name, duration, success)
coordinator.track_token_usage(tokens)

# Export analytics
exporter = FileAnalyticsExporter("session.json")
await coordinator.export(exporter)
```

**Key Features**:
- Tool usage tracking
- Token consumption monitoring
- Performance metrics collection
- Multiple export formats (console, file, custom)

**SOLID Compliance**:
- **SRP**: Only collects analytics (not business logic)
- **OCP**: Custom exporters via BaseAnalyticsExporter

---

#### PromptCoordinator
**Responsibility**: System prompt building from contributors

```python
from victor.agent.coordinators import PromptCoordinator

coordinator = PromptCoordinator()

# Build system prompt from contributors
prompt = await coordinator.build_prompt(
    contributors=[SystemPromptContributor(), TaskHintContributor()],
    context={"mode": "build"}
)
```

**Key Features**:
- Composable prompt contributors
- Task-aware prompt enhancement
- Mode-specific prompt customization
- Template-based prompt building

**SOLID Compliance**:
- **SRP**: Only builds prompts (not chat or execution)
- **OCP**: New contributors via BasePromptContributor

---

#### SessionCoordinator
**Responsibility**: Conversation session lifecycle management

```python
from victor.agent.coordinators import SessionCoordinator

coordinator = SessionCoordinator()

# Create new session
session = await coordinator.create_session(
    user_id="user123",
    metadata={"project": "my-app"}
)

# Manage session state
await coordinator.update_session(session_id, state=...)
```

**Key Features**:
- Session creation and initialization
- Session state persistence
- Session lifecycle management (create, update, close)
- Multi-user session support

**SOLID Compliance**:
- **SRP**: Only manages session lifecycle

---

#### ProviderCoordinator
**Responsibility**: Provider switching and management

```python
from victor.agent.coordinators import ProviderCoordinator

coordinator = ProviderCoordinator()

# Switch providers mid-conversation
await coordinator.switch_provider("openai", model="gpt-4")

# Get current provider
provider = coordinator.get_current_provider()
```

**Key Features**:
- Dynamic provider switching
- Provider capability checking
- Model selection guidance
- Provider fallback handling

**SOLID Compliance**:
- **SRP**: Only manages provider selection (not chat or tools)

---

#### ModeCoordinator
**Responsibility**: Agent modes (build, plan, explore)

```python
from victor.agent.coordinators import ModeCoordinator

coordinator = ModeCoordinator()

# Set agent mode
await coordinator.set_mode("plan")

# Get mode configuration
config = coordinator.get_mode_config()
```

**Key Features**:
- Mode switching (build, plan, explore)
- Mode-specific behavior configuration
- Tool budget adjustment per mode
- Exploration level management

**SOLID Compliance**:
- **SRP**: Only manages agent modes

---

#### ToolSelectionCoordinator
**Responsibility**: Semantic tool selection

```python
from victor.agent.coordinators import ToolSelectionCoordinator

coordinator = ToolSelectionCoordinator(
    strategy="hybrid",
    semantic_threshold=0.7
)

# Select tools based on context
tools = await coordinator.select_tools(
    query="analyze code quality",
    available_tools=all_tools
)
```

**Key Features**:
- Semantic similarity-based selection
- Keyword-based fallback
- Hybrid strategy combination
- Cached selection results

**SOLID Compliance**:
- **SRP**: Only handles tool selection (not execution)

---

### Design Principles

All application coordinators follow these principles:

1. **Single Responsibility**: Each coordinator has one reason to change
2. **Interface Segregation**: Focused protocols (e.g., IToolCoordinator)
3. **Dependency Inversion**: Depend on protocols, not concrete implementations
4. **Open/Closed**: Extensible through strategies and contributors

### Why Application Layer Exists

The application layer handles **Victor-specific logic**:
- How to manage conversation state
- When to switch providers or modes
- How to select and execute tools
- How to manage context windows
- How to collect session analytics

This logic is **specific to Victor's use case** and doesn't belong in a generic framework.

## Framework Layer: victor/framework/coordinators/

### Purpose

Framework-agnostic workflow infrastructure that can be reused across any vertical (Coding, DevOps, RAG, DataAnalysis, Research, or external verticals).

### Key Coordinators

#### YAMLWorkflowCoordinator
**Responsibility**: YAML workflow loading and execution

```python
from victor.framework.coordinators import YAMLWorkflowCoordinator

coordinator = YAMLWorkflowCoordinator()

# Execute YAML workflow
result = await coordinator.execute(
    "path/to/workflow.yaml",
    initial_state={"input": "data"}
)

# Stream execution events
async for event in coordinator.stream("workflow.yaml", state):
    print(f"{event.node_id}: {event.event_type}")
```

**Key Features**:
- Load workflows from YAML files
- Two-level caching (definition + node)
- Checkpointing for resumable workflows
- Condition and transform registries (escape hatches)
- Consistent execution via UnifiedWorkflowCompiler

**SOLID Compliance**:
- **SRP**: Only handles YAML workflow execution
- **DIP**: Depends on UnifiedWorkflowCompiler protocol

---

#### GraphExecutionCoordinator
**Responsibility**: StateGraph/CompiledGraph execution

```python
from victor.framework.coordinators import GraphExecutionCoordinator

coordinator = GraphExecutionCoordinator()

# Execute compiled graph
result = await coordinator.execute(
    compiled_graph,
    initial_state={"input": "data"}
)

# Stream execution events
async for event in coordinator.stream(compiled_graph, state):
    print(f"{event.node_id}: {event.event_type}")
```

**Key Features**:
- Direct CompiledGraph execution
- WorkflowGraph compilation and execution
- Streaming execution events
- LSP-compliant result handling
- Integration with NodeRunner registry

**SOLID Compliance**:
- **SRP**: Only handles graph execution
- **LSP**: Properly handles polymorphic return types

---

#### HITLCoordinator
**Responsibility**: Human-in-the-loop workflow integration

```python
from victor.framework.coordinators import HITLCoordinator

coordinator = HITLCoordinator()

# Execute workflow with human approval
result = await coordinator.execute_with_hitl(
    workflow,
    approval_handler=handler,
    state=initial_state
)
```

**Key Features**:
- Human approval workflow nodes
- Approval handler abstraction
- Automatic/rejection handling
- Integration with workflow execution

**SOLID Compliance**:
- **SRP**: Only handles HITL logic

---

#### CacheCoordinator
**Responsibility**: Workflow caching management

```python
from victor.framework.coordinators import CacheCoordinator

coordinator = CacheCoordinator()

# Enable workflow caching
coordinator.enable_caching(ttl=3600)

# Clear cache
coordinator.clear_cache()

# Get cache stats
stats = coordinator.get_stats()
```

**Key Features**:
- Definition-level caching
- Node-level caching
- TTL-based expiration
- Cache statistics and monitoring

**SOLID Compliance**:
- **SRP**: Only manages caching

---

### Design Principles

All framework coordinators follow these principles:

1. **Domain Agnostic**: No Victor-specific logic
2. **Vertical Independent**: Works with any vertical
3. **Protocol-Based**: Clean interfaces for all operations
4. **Reusable**: Can be used in external projects

### Why Framework Layer Exists

The framework layer provides **reusable infrastructure**:
- How to execute YAML workflows (any vertical)
- How to execute StateGraphs (any use case)
- How to handle human-in-the-loop (generic)
- How to cache workflow results (generic)

This infrastructure is **not Victor-specific** and can be used by any vertical or external project.

## How the Layers Interact

### Layer Separation

```
Application Layer (Victor-specific)
    - Uses Framework Layer for workflow execution
    - Adds Victor-specific logic on top
    - Manages conversation lifecycle

Framework Layer (Domain-agnostic)
    - Provides workflow execution primitives
    - No knowledge of Victor's business logic
    - Reusable across verticals
```

### Example: Workflow Execution in Chat

```python
# Application Layer (ChatCoordinator)
class ChatCoordinator:
    async def chat(self, messages, tools):
        # Victor-specific logic
        mode = self.mode_coordinator.get_mode()
        context = await self.context_coordinator.build_context(messages)

        # Use Framework Layer for workflow execution
        if self.workflow_enabled:
            result = await self.graph_coordinator.execute(
                self.chat_workflow,
                initial_state={
                    "messages": messages,
                    "tools": tools,
                    "mode": mode,
                    "context": context
                }
            )
        else:
            result = await self.provider.chat(messages, tools)

        # Victor-specific post-processing
        await self.analytics_coordinator.track_chat(result)
        return result
```

### Dependency Flow

```
Application Layer Components
    └── depend on ──→ Framework Layer Components
                          └── depend on ──→ Core Protocols
```

## Benefits of This Architecture

### 1. Single Responsibility Principle (SRP)

Each coordinator has **one reason to change**:

- **ChatCoordinator**: Changes when chat logic changes
- **ToolCoordinator**: Changes when tool execution changes
- **YAMLWorkflowCoordinator**: Changes when YAML workflow syntax changes
- **GraphExecutionCoordinator**: Changes when graph execution model changes

### 2. Layered Architecture

Clear separation between **application logic** and **framework infrastructure**:

```
┌─────────────────────────────────────┐
│   Application Logic (Victor)        │  Business rules, conversation mgmt
├─────────────────────────────────────┤
│   Framework Infrastructure          │  Workflow execution, caching
├─────────────────────────────────────┤
│   Core Protocols                    │  Base abstractions
└─────────────────────────────────────┘
```

### 3. Reusability

Framework coordinators are **reused across verticals**:

```python
# Coding vertical uses framework coordinators
from victor.framework.coordinators import YAMLWorkflowCoordinator

class CodingWorkflowProvider:
    def __init__(self):
        self.yaml_coordinator = YAMLWorkflowCoordinator()

# Research vertical also uses framework coordinators
class ResearchWorkflowProvider:
    def __init__(self):
        self.yaml_coordinator = YAMLWorkflowCoordinator()  # Same!

# External vertical can also use them
class SecurityVertical:
    def __init__(self):
        self.yaml_coordinator = YAMLWorkflowCoordinator()  # Reusable!
```

### 4. Testability

Coordinators can be **tested independently**:

```python
# Test ChatCoordinator in isolation
async def test_chat_coordinator():
    mock_orchestrator = MockOrchestrator()
    coordinator = ChatCoordinator(mock_orchestrator)

    result = await coordinator.chat(messages=[...])

    assert result.content == "expected"

# Test YAMLWorkflowCoordinator in isolation
async def test_yaml_workflow_coordinator():
    coordinator = YAMLWorkflowCoordinator()

    result = await coordinator.execute("workflow.yaml", state={...})

    assert result.status == "completed"
```

### 5. Separation of Concerns

Clear boundaries between different concerns:

| Concern | Owner | Location |
|---------|-------|----------|
| Conversation management | Application Layer | `victor/agent/coordinators/` |
| Tool execution | Application Layer | `victor/agent/coordinators/` |
| Workflow execution | Framework Layer | `victor/framework/coordinators/` |
| Caching strategy | Framework Layer | `victor/framework/coordinators/` |

### 6. Maintainability

Changes are **localized**:

- Changing chat logic → Only edit `ChatCoordinator`
- Changing tool selection → Only edit `ToolSelectionCoordinator`
- Changing YAML syntax → Only edit `YAMLWorkflowCoordinator`
- Changing caching strategy → Only edit `CacheCoordinator`

### 7. Extensibility

Easy to extend through **protocols and strategies**:

```python
# Add new compaction strategy
class CustomCompactionStrategy(BaseCompactionStrategy):
    async def compact(self, messages, max_tokens):
        # Custom logic
        return compacted_messages

# Add new tool selection strategy
class CustomSelectionStrategy(BaseToolSelectionStrategy):
    async def select_tools(self, query, available_tools):
        # Custom logic
        return selected_tools

# Add new analytics exporter
class CustomExporter(BaseAnalyticsExporter):
    async def export(self, analytics):
        # Custom logic
        pass
```

## Why This Is NOT Harmful Duplication

### What Review Said

> "Harmful duplication between `victor/agent/coordinators/` and `victor/framework/coordinators/`"

### Why This Is Wrong

The two directories serve **completely different purposes**:

| Aspect | Application Layer | Framework Layer |
|--------|-------------------|-----------------|
| **Purpose** | Victor-specific orchestration | Generic workflow infrastructure |
| **Domain** | AI agent conversation lifecycle | Workflow execution primitives |
| **Reusability** | Victor-only | All verticals + external |
| **Dependencies** | Depends on framework layer | No dependency on app layer |
| **Change Reason** | Business logic changes | Framework requirements change |
| **Example** | ChatCoordinator manages chat | YAMLWorkflowCoordinator executes YAML |

### Analogy

This is like comparing:

- **Application Layer**: A web application's controllers (handle HTTP requests, business logic)
- **Framework Layer**: A web framework's routing system (URL routing, middleware)

Both are "coordinators" but they operate at **different layers of abstraction**.

### Real-World Examples

This pattern is used in major frameworks:

1. **Django**:
   - Application layer: Django REST Framework ViewSets (business logic)
   - Framework layer: Django URL dispatcher (routing infrastructure)

2. **Spring Framework**:
   - Application layer: @Service classes (business logic)
   - Framework layer: DispatcherServlet (request routing)

3. **Express.js**:
   - Application layer: Route handlers (business logic)
   - Framework layer: Express router (routing infrastructure)

## SOLID Principles in Action

### Single Responsibility Principle

Each coordinator has **one reason to change**:

```python
# ✅ Good: Single responsibility
class ChatCoordinator:
    """Only handles chat operations"""

class ToolCoordinator:
    """Only handles tool operations"""

class YAMLWorkflowCoordinator:
    """Only handles YAML workflow execution"""

# ❌ Bad: Multiple responsibilities
class MonolithicCoordinator:
    """Handles chat, tools, workflows, config, analytics, ..."""
```

### Interface Segregation Principle

Coordinators depend on **focused protocols**:

```python
# ✅ Good: Focused protocols
class IToolCoordinator(Protocol):
    """Only tool-related methods"""
    async def select_tools(self, context) -> List[BaseTool]: ...
    async def execute_tool_calls(self, calls) -> List[ToolResult]: ...

class IChatCoordinator(Protocol):
    """Only chat-related methods"""
    async def chat(self, messages) -> CompletionResponse: ...
    async def stream_chat(self, messages) -> AsyncIterator[StreamChunk]: ...

# ❌ Bad: Fat interface
class ICoordinator(Protocol):
    """Everything mixed together"""
    async def select_tools(self): ...
    async def execute_tools(self): ...
    async def chat(self): ...
    async def manage_context(self): ...
    async def track_analytics(self): ...
    # ... 20 more methods
```

### Dependency Inversion Principle

Coordinators depend on **protocols, not implementations**:

```python
# ✅ Good: Depend on protocols
class ChatCoordinator:
    def __init__(self, orchestrator: IAgentOrchestrator):
        self.orchestrator = orchestrator  # Protocol

# ❌ Bad: Depend on concrete implementation
class ChatCoordinator:
    def __init__(self, orchestrator: AgentOrchestrator):
        self.orchestrator = orchestrator  # Concrete class
```

### Open/Closed Principle

Coordinators are **open for extension, closed for modification**:

```python
# ✅ Good: Extend through strategies
class ContextCoordinator:
    def __init__(self, strategy: BaseCompactionStrategy):
        self.strategy = strategy

# Extend without modifying
class SemanticCompactionStrategy(BaseCompactionStrategy):
    async def compact(self, messages, max_tokens):
        # Custom logic
        return compacted

# ❌ Bad: Modify for extension
class ContextCoordinator:
    async def compact_context(self, messages, max_tokens, strategy):
        if strategy == "truncation":
            # ...
        elif strategy == "semantic":
            # ...
        # Must modify this file to add new strategies
```

### Liskov Substitution Principle

Coordinators are **substitutable** through protocols:

```python
# ✅ Good: Substitutable implementations
def process_chat(coordinator: IChatCoordinator):
    # Works with any IChatCoordinator implementation
    return await coordinator.chat(messages)

# Can substitute any implementation
process_chat(BasicChatCoordinator(...))
process_chat(AdvancedChatCoordinator(...))
process_chat(MockChatCoordinator(...))
```

## Migration Path from Monolithic Orchestrator

### Before: Monolithic Orchestrator

```python
class AgentOrchestrator:
    """Monolithic orchestrator with 2000+ lines"""

    def __init__(self):
        self.config = self._load_config()
        self.prompt = self._build_prompt()
        self.context = self._manage_context()
        self.tools = self._manage_tools()
        self.chat = self._handle_chat()
        self.analytics = self._track_analytics()
        # ... 20+ more responsibilities

    def _load_config(self): ...
    def _build_prompt(self): ...
    def _manage_context(self): ...
    def _manage_tools(self): ...
    def _handle_chat(self): ...
    def _track_analytics(self): ...
    # ... 20+ more methods
```

### After: Coordinator-Based Architecture

```python
class AgentOrchestrator:
    """Facade coordinator - delegates to specialized coordinators"""

    def __init__(self):
        # Delegate to specialized coordinators
        self.config_coordinator = ConfigCoordinator()
        self.prompt_coordinator = PromptCoordinator()
        self.context_coordinator = ContextCoordinator()
        self.tool_coordinator = ToolCoordinator(...)
        self.chat_coordinator = ChatCoordinator(self)
        self.analytics_coordinator = AnalyticsCoordinator()

    # Simple delegation methods
    async def chat(self, messages):
        return await self.chat_coordinator.chat(messages)

    async def select_tools(self, context):
        return await self.tool_coordinator.select_tools(context)
```

### Benefits of Migration

- **Testability**: Each coordinator can be tested independently
- **Maintainability**: Changes are localized to specific coordinators
- **Reusability**: Framework coordinators used across verticals
- **Readability**: Clear separation of concerns
- **Performance**: Coordinators can be optimized independently

## Conclusion

The two-layer coordinator architecture is **intentional, well-designed, and beneficial**:

1. **Application Layer** (`victor/agent/coordinators/`): Victor-specific orchestration
2. **Framework Layer** (`victor/framework/coordinators/`): Reusable workflow infrastructure

This separation provides:
- Clear separation of concerns
- Independent testability
- Reusable framework components
- SOLID principle compliance
- Easy maintenance and extension

This is **not harmful duplication**—it's **good architecture** following industry best practices for layered systems.

## Further Reading

- [Coordinator-Based Architecture](coordinator_based_architecture.md)
- [Framework API Reference](../reference/internals/FRAMEWORK_API.md)
- [Development Guide](../contributing/index.md)

## Diagram

See [Coordinators Architecture Diagram](diagrams/coordinators.mmd) for a visual representation of this architecture.
