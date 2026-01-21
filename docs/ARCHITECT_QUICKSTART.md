<div align="center">

# Architect Quick Start Guide

**Understanding Victor AI architecture in 15 minutes**

[![Architecture](https://img.shields.io/badge/architecture-SOLID-blue)](./architecture/DESIGN_PATTERNS.md)
[![Documentation](https://img.shields.io/badge/docs-comprehensive-green)](./architecture/README.md)

</div>

---

## Welcome, Architects!

This guide provides a comprehensive overview of Victor AI's architecture, design patterns, and key architectural decisions. Perfect for system architects, technical leads, and anyone who needs to understand Victor's design at a deep level.

### What You'll Learn

- High-level architecture overview
- Core architectural patterns
- Key components and their responsibilities
- Design principles and SOLID compliance
- Technology stack and choices
- Where to find detailed architecture documentation

---

## Architecture Overview

### High-Level Architecture

Victor AI uses a **layered architecture** with **facade pattern** at its core:

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLIENTS                                       │
│  CLI/TUI  │  VS Code (HTTP)  │  MCP Server  │  API Server       │
└─────────────────────────┬───────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              ServiceContainer (DI Container)                     │
│  55+ registered services (singleton, scoped, transient)         │
└─────────────────────────┬───────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                 AGENT ORCHESTRATOR (Facade)                      │
│  Delegates to: ConversationController, ToolPipeline,            │
│  StreamingController, ProviderManager, ToolRegistrar            │
│  Coordinators: ToolCoordinator, StateCoordinator, etc.          │
└─────────────────────────┬───────────────────────────────────────┘
                          ▼
┌───────────┬─────────────┬───────────────┬───────────────────────┐
│ PROVIDERS │   TOOLS     │  WORKFLOWS    │  VERTICALS            │
│  21       │   55        │  StateGraph   │  Coding/DevOps/RAG/   │
│           │             │  + YAML       │  DataAnalysis/Research│
└───────────┴─────────────┴───────────────┴───────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Event Bus (Pub/Sub)                           │
│  Topics: tool.*, agent.*, workflow.*, error.*                   │
│  Backends: In-Memory, Kafka, SQS, RabbitMQ, Redis               │
└─────────────────────────────────────────────────────────────────┘
```

### Key Architectural Principles

1. **Provider Agnosticism** - Switch between 21 LLM providers seamlessly
2. **Vertical Architecture** - Self-contained domains with specialized tools
3. **Protocol-First Design** - 98 protocols for loose coupling
4. **Dependency Injection** - ServiceContainer manages 55+ services
5. **Event-Driven Architecture** - Pluggable event backends
6. **SOLID Compliance** - ISP, DIP, SRP across all components
7. **Facade Pattern** - AgentOrchestrator simplifies complex subsystems

---

## Core Architectural Patterns

### 1. Facade Pattern

**AgentOrchestrator** provides a simplified interface to complex subsystems:

```python
from victor.agent.orchestrator import AgentOrchestrator

# Single entry point for all agent operations
orchestrator = AgentOrchestrator(container)

# Facade handles complexity internally
response = await orchestrator.chat("Hello")
```

**Benefits**:
- Simplified API for clients
- Reduced coupling between subsystems
- Easier to test and maintain

### 2. Dependency Injection

**ServiceContainer** manages all dependencies:

```python
from victor.core.container import ServiceContainer, ServiceLifetime

container = ServiceContainer()

# Register services
container.register(
    ToolExecutorProtocol,
    lambda c: ToolExecutor(tool_registry=c.get(ToolRegistryProtocol)),
    ServiceLifetime.SINGLETON,
)

# Resolve dependencies
executor = container.get(ToolExecutorProtocol)
```

**Benefits**:
- Loose coupling between components
- Easy testing with mock dependencies
- Centralized dependency management
- Lifecycle management (singleton, scoped, transient)

### 3. Protocol-Based Design

**Protocols** define interfaces before implementation:

```python
from victor.protocols import ToolExecutorProtocol

# Define protocol
class ToolExecutorProtocol(Protocol):
    async def execute(self, tool: str, **kwargs) -> ToolResult:
        ...

# Implement protocol
class MyToolExecutor:
    async def execute(self, tool: str, **kwargs) -> ToolResult:
        ...

# Use protocol (type-safe)
executor: ToolExecutorProtocol = MyToolExecutor()
```

**Benefits**:
- Type safety with structural typing
- Multiple implementations possible
- Easy to mock for testing
- Clear contracts between components

### 4. Coordinator Pattern

**Coordinators** manage complex operations:

```python
from victor.agent.protocols import ToolCoordinatorProtocol

coordinator = container.get(ToolCoordinatorProtocol)
results = await coordinator.select_and_execute(
    query="Read Python files",
    context=AgentToolSelectionContext(max_tools=5),
)
```

**Benefits**:
- Single Responsibility Principle
- Specialized coordinators for complex tasks
- Easy to test and extend
- Clear separation of concerns

### 5. Event-Driven Architecture

**Event Bus** enables loose coupling:

```python
from victor.core.events import create_event_backend, MessagingEvent

backend = create_event_backend(BackendConfig.for_observability())
await backend.connect()

# Publish events
await backend.publish(
    MessagingEvent(
        topic="tool.complete",
        data={"tool": "read_file", "result": "..."},
    )
)

# Subscribe to events
await backend.subscribe("tool.*", my_handler)
```

**Benefits**:
- Asynchronous communication
- Scalable architecture
- Pluggable event backends
- Easy to add observers

---

## Key Components

### 1. AgentOrchestrator (Facade)

**Location**: `victor/agent/orchestrator.py`

**Responsibilities**:
- Single entry point for agent operations
- Delegates to specialized controllers
- Manages conversation flow
- Coordinates tool execution

**Key Methods**:
- `chat()` - Process user messages
- `stream_chat()` - Stream responses
- `switch_provider()` - Change LLM provider
- `get_context()` - Get conversation context

### 2. ServiceContainer (DI)

**Location**: `victor/core/container.py`

**Responsibilities**:
- Register and resolve services
- Manage service lifecycles
- Handle dependency injection
- Provide service locator

**Service Lifetimes**:
- **Singleton** - One instance for container lifetime
- **Scoped** - One instance per scope
- **Transient** - New instance each time

**Registered Services**: 55+ services

### 3. ToolPipeline

**Location**: `victor/agent/tool_pipeline.py`

**Responsibilities**:
- Manage tool selection and execution
- Handle tool composition
- Coordinate tool calls
- Manage tool budgets

**Key Features**:
- Semantic tool selection
- Tool composition patterns
- Budget-aware execution
- Error handling and retry

### 4. ConversationController

**Location**: `victor/agent/conversation_controller.py`

**Responsibilities**:
- Manage conversation state
- Track message history
- Handle context management
- Manage sessions

**Key Features**:
- Session persistence
- Context window management
- Message filtering
- State machine for conversation stages

### 5. ProviderManager

**Location**: `victor/agent/provider_manager.py`

**Responsibilities**:
- Manage provider instances
- Handle provider switching
- Validate provider capabilities
- Route requests to providers

**Key Features**:
- Provider pooling
- Circuit breaker for failures
- Rate limiting
- Capability negotiation

---

## Vertical Architecture

### Vertical Structure

Victor organizes functionality into **5 domain verticals**:

```
victor/
├── coding/          # Code analysis, review, testing
├── devops/          # CI/CD, infrastructure, deployment
├── rag/             # Document ingestion, vector search
├── dataanalysis/    # Data processing, visualization
└── research/        # Web search, citations, synthesis
```

### Vertical Base Class

All verticals inherit from `VerticalBase`:

```python
from victor.core.verticals.base import VerticalBase

class MyVertical(VerticalBase):
    name = "my_vertical"

    @classmethod
    def get_mode_config(cls, mode: str) -> ModeDefinition:
        return ModeConfigRegistry.get_instance().load_config(
            cls.name
        ).get_mode(mode)

    @classmethod
    def get_capability_provider(cls) -> CapabilitySet:
        return CapabilityLoader.from_vertical(cls.name)
```

### Benefits

- **Domain isolation** - Each vertical is self-contained
- **Specialized tools** - Vertical-specific tool sets
- **Mode configuration** - Vertical-specific agent modes
- **Capability providers** - Vertical-specific capabilities
- **Team formations** - Vertical-specific multi-agent teams

---

## Design Patterns

### SOLID Principles

**Single Responsibility Principle (SRP)**:
- Each class has one reason to change
- Coordinators handle specific complex operations
- Clear separation of concerns

**Open/Closed Principle (OCP)**:
- Open for extension, closed for modification
- Plugin architecture for providers and tools
- Vertical system for domain extensions

**Liskov Substitution Principle (LSP)**:
- Subtypes must be substitutable for base types
- All providers inherit from `BaseProvider`
- All tools inherit from `BaseTool`

**Interface Segregation Principle (ISP)**:
- Clients shouldn't depend on unused interfaces
- Lean protocols like `SubAgentContext`
- Specialized coordinator protocols

**Dependency Inversion Principle (DIP)**:
- Depend on abstractions, not concretions
- Protocol-based design
- Dependency injection container

### Other Patterns

- **Factory Pattern** - Provider and tool factories
- **Strategy Pattern** - Pluggable tool selection strategies
- **Observer Pattern** - Event bus for pub/sub
- **Template Method** - BaseYAMLWorkflowProvider
- **Builder Pattern** - AgentBuilder for agent composition
- **Registry Pattern** - UniversalRegistry for entity management

---

## Technology Stack

### Core Technologies

- **Python 3.10+** - Core language
- **AsyncIO** - Asynchronous programming
- **Pydantic** - Data validation and settings
- **Typer** - CLI framework
- **Rich** - Terminal UI

### LLM Integration

- **Anthropic** - Claude models
- **OpenAI** - GPT models
- **Google** - Gemini models
- **Azure** - Azure OpenAI
- **Local** - Ollama, LM Studio, vLLM

### Data & Storage

- **Tree-sitter** - AST parsing
- **Embeddings** - Vector search
- **SQLite** - Checkpoint persistence
- **YAML** - Configuration files

### Testing

- **Pytest** - Test framework
- **pytest-asyncio** - Async test support
- **pytest-mock** - Mocking support
- **Coverage** - Code coverage

### Code Quality

- **Black** - Code formatting
- **Ruff** - Fast linter
- **Mypy** - Type checking
- **Pylint** - Additional linting

---

## Key Architectural Decisions

### 1. Protocol-First Design

**Decision**: Define protocols before implementations

**Rationale**:
- Type safety without inheritance
- Multiple implementations possible
- Easy to mock for testing
- Clear contracts between components

**Impact**: 98 protocols defined across the codebase

### 2. Dependency Injection

**Decision**: Use ServiceContainer for all dependencies

**Rationale**:
- Loose coupling between components
- Easy testing with mock dependencies
- Centralized dependency management
- Lifecycle management

**Impact**: 55+ registered services

### 3. Event-Driven Architecture

**Decision**: Use event bus for pub/sub messaging

**Rationale**:
- Asynchronous communication
- Scalable architecture
- Pluggable backends (Kafka, SQS, etc.)
- Easy to add observers

**Impact**: Event-driven workflows and observability

### 4. Vertical Architecture

**Decision**: Organize into domain verticals

**Rationale**:
- Domain isolation
- Specialized tool sets
- Easy to extend
- Clear boundaries

**Impact**: 5 verticals with 55 specialized tools

### 5. YAML-First Workflows

**Decision**: Define workflows in YAML with Python escape hatches

**Rationale**:
- Declarative workflow definitions
- Easy to read and modify
- Python escape hatches for complex logic
- Two-level caching for performance

**Impact**: StateGraph DSL with YAML compilation

---

## Performance Architecture

### Caching Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                    Caching Layers                           │
├─────────────────────────────────────────────────────────────┤
│ 1. Tool Selection Cache (LRU, 1000 entries)                │
│ 2. Embedding Cache (TTL, 1 hour)                           │
│ 3. Workflow Cache (Definition + Execution)                 │
│ 4. Provider Pool (Connection reuse)                        │
│ 5. Universal Registry (Multiple strategies)                │
└─────────────────────────────────────────────────────────────┘
```

**Performance Improvements**:
- 24-37% latency reduction from tool selection caching
- Lazy loading for providers and tools
- Connection pooling for providers
- Async operations throughout

### Scalability

- **Horizontal Scaling** - Stateless design allows multiple instances
- **Vertical Scaling** - Efficient resource utilization
- **Provider Pooling** - Reuse connections across requests
- **Event Bus Scaling** - Pluggable backends (Kafka, SQS)

---

## Security Architecture

### Security Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    Security Layers                          │
├─────────────────────────────────────────────────────────────┤
│ 1. API Key Management (Environment variables)              │
│ 2. Secret Masking (Middleware)                             │
│ 3. Air-Gapped Mode (Local-only)                            │
│ 4. Tool Filtering (AirgappedFilter, SecurityFilter)        │
│ 5. Input Validation (Pydantic models)                      │
│ 6. Error Handling (No sensitive data in errors)            │
└─────────────────────────────────────────────────────────────┘
```

### Security Best Practices

- No hardcoded secrets
- Secret masking in logs
- Air-gapped mode for secure environments
- Tool filtering for security
- Input validation on all APIs
- Error handling without data leakage

---

## Where to Learn More

### Architecture Documentation

- [Architecture README](./architecture/README.md) - Architecture hub
- [Architecture Overview](./architecture/overview.md) - Detailed overview
- [Design Patterns](./architecture/DESIGN_PATTERNS.md) - All patterns
- [Component Reference](./architecture/COMPONENT_REFERENCE.md) - Component docs
- [Protocols Reference](./architecture/PROTOCOLS_REFERENCE.md) - Protocol definitions
- [Best Practices](./architecture/BEST_PRACTICES.md) - Best practices
- [Refactoring Overview](./architecture/REFACTORING_OVERVIEW.md) - Refactoring summary

### Migration Documentation

- [Migration Roadmap](./MIGRATION_ROADMAP.md) - All migrations
- [ISP Migration Guide](./ISP_MIGRATION_GUIDE.md) - Interface segregation
- [Refactoring Migration Guide](./architecture/REFACTORING_MIGRATION_GUIDE.md) - Refactoring

### Architecture Decision Records

- [ADR-001: Coordinator Architecture](./adr/ADR-001-coordinator-architecture.md)
- [ADR-002: YAML Vertical Config](./adr/ADR-002-yaml-vertical-config.md)
- [ADR-003: Distributed Caching](./adr/ADR-003-distributed-caching.md)
- [ADR-004: Protocol-Based Design](./adr/ADR-004-protocol-based-design.md)
- [ADR-005: Performance Optimization](./adr/ADR-005-performance-optimization.md)

### Specialized Topics

- [SOLID Refactoring Report](./SOLID_ARCHITECTURE_REFACTORING_REPORT.md) - SOLID compliance
- [Framework Analysis](./architecture/VICTOR_FRAMEWORK_ANALYSIS.md) - Framework analysis
- [Workflow Consolidation](./architecture/WORKFLOW_CONSOLIDATION.md) - Workflow system

---

## Quick Reference

### Architecture Metrics

- **Total Components**: 55+ services
- **Protocols**: 98 defined
- **Providers**: 21 LLM providers
- **Tools**: 55 specialized tools
- **Verticals**: 5 domain verticals
- **Coordinators**: 8 specialized coordinators

### Code Organization

```
victor/
├── agent/           # Orchestration and coordination
├── core/            # Bootstrap, container, events
├── providers/       # LLM provider implementations
├── tools/           # Tool implementations
├── frameworks/      # Reusable framework capabilities
├── workflows/       # Workflow engine and compiler
├── protocols/       # Protocol definitions
├── config/          # Configuration and settings
└── {vertical}/      # Domain-specific implementations
```

### Key Files to Understand

1. `victor/agent/orchestrator.py` - Facade pattern
2. `victor/core/container.py` - Dependency injection
3. `victor/agent/tool_pipeline.py` - Tool execution
4. `victor/providers/base.py` - Provider interface
5. `victor/tools/base.py` - Tool interface
6. `victor/framework/graph.py` - StateGraph DSL

---

## Next Steps

### For New Architects

1. ✅ Read this quick start
2. 📖 Read [Architecture Overview](./architecture/overview.md)
3. 🎯 Study [Design Patterns](./architecture/DESIGN_PATTERNS.md)
4. 🏗️ Review [Component Reference](./architecture/COMPONENT_REFERENCE.md)

### For System Design

1. 📋 Review [Architecture Decision Records](./adr/)
2. 🔍 Analyze [Best Practices](./architecture/BEST_PRACTICES.md)
3. 📊 Study [Performance Architecture](./performance/README.md)
4. 🔐 Review [Security Architecture](./SECURITY_BEST_PRACTICES.md)

### For Extensions

1. 🔧 Read [Extension Development](./extensions/README.md)
2. 📦 Study [Vertical Creation Guide](./development/vertical_creation_guide.md)
3. 🛠️ Review [Tool Composition](./TOOL_COMPOSITION_GUIDE.md)
4. 🎨 Explore [Framework Capabilities](./victor/framework/README.md)

---

## Need Help?

- [Architecture Documentation](./architecture/README.md) - Complete architecture docs
- [GitHub Discussions](https://github.com/vjsingh1984/victor/discussions) - Community discussions
- [GitHub Issues](https://github.com/vjsingh1984/victor/issues) - Architecture issues

---

<div align="center">

**Continue Your Architecture Journey**

[Read Architecture Overview](./architecture/overview.md) •
[Study Design Patterns](./architecture/DESIGN_PATTERNS.md) •
[Review Components](./architecture/COMPONENT_REFERENCE.md)

**[Back to Documentation Index](./INDEX.md)**

</div>
