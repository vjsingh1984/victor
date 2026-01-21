<div align="center">

# Victor AI Glossary

**Technical terms, acronyms, and concepts**

[![Glossary](https://img.shields.io/badge/glossary-200%2B%20terms-blue)](./INDEX.md)
[![Documentation](https://img.shields.io/badge/docs-complete-green)](./INDEX.md)

</div>

---

## A

### Agent

An AI-powered assistant that uses LLM providers to interact with users, execute tools, and manage workflows.

**See Also**: [AgentOrchestrator](#agentorchestrator), [SubAgent](#subagent)

### AgentOrchestrator

The main facade that coordinates all agent operations. Delegates to specialized controllers and coordinators.

**Location**: `victor/agent/orchestrator.py`

**See Also**: [Facade Pattern](#facade-pattern), [Coordinator](#coordinator)

### Air-Gapped Mode

An operational mode that restricts Victor to local-only providers (Ollama, LM Studio, vLLM) and local tools, with no internet access or web tools.

**Configuration**: Set `airgapped_mode: true` in config or use `--air-gapped` flag

### API Key

A secret token used to authenticate with LLM providers (e.g., Anthropic, OpenAI). Should be stored in environment variables or secure configuration.

### AST (Abstract Syntax Tree)

A tree representation of code structure, used by Victor for code analysis and understanding. Powered by Tree-sitter.

**See Also**: [Tree-sitter](#tree-sitter)

### AsyncIO

Python's asynchronous I/O library, used throughout Victor for concurrent operations.

---

## B

### BaseProvider

The abstract base class that all LLM providers inherit from. Defines the interface for chat, streaming, and tool support.

**Location**: `victor/providers/base.py`

### BaseTool

The abstract base class that all tools inherit from. Defines the interface for tool execution.

**Location**: `victor/tools/base.py`

### Benchmark

Performance measurement and comparison of Victor's components, particularly tool selection and caching.

**See**: [Performance Documentation](./performance/README.md)

### Budget

A limit on the number of tool executions or API calls allowed in a session or workflow.

---

## C

### Cache

Temporary storage of frequently used data to improve performance. Victor uses multiple caching strategies (LRU, TTL, Manual).

**Types**:
- Tool selection cache
- Embedding cache
- Workflow cache
- Provider connection pool

### Capability

A feature or functionality that a vertical or component provides. Capabilities are defined in YAML and loaded via CapabilityLoader.

**See**: [Capability System](#capability-system)

### Capability System

A YAML-first configuration system for defining capabilities (tools, workflows, middleware, validators, observers) by vertical.

**Location**: `victor/config/capabilities/`

### Chat

The primary interaction mode where users converse with the AI agent.

**Command**: `victor chat`

### CLI (Command-Line Interface)

The text-based interface for interacting with Victor. Can be used with `--no-tui` flag.

### Coordinator

A specialized component that manages a specific complex operation (e.g., tool selection, state management, workflow execution).

**See**: [Coordinator Pattern](#coordinator-pattern)

### Coordinator Pattern

A design pattern where specialized coordinators handle specific complex operations, following the Single Responsibility Principle.

**See**: [Design Patterns](./architecture/DESIGN_PATTERNS.md)

---

## D

### DIP (Dependency Inversion Principle)

A SOLID principle stating that components should depend on abstractions (protocols) rather than concretions.

**Implementation**: ServiceContainer with protocol-based dependencies

### Dependency Injection (DI)

A design pattern where dependencies are provided to a component rather than created internally. Victor uses ServiceContainer for DI.

**Location**: `victor/core/container.py`

### Docker

A containerization platform. Victor can be run in Docker for isolated environments.

**See**: [Docker Setup](./DOCKER_SETUP.md)

### DSR (Dependency Substitution Rule)

See [LSP](#lsp)

---

## E

### Embedding

A vector representation of text, used for semantic search and similarity matching.

**See**: [RAG](#rag)

### Event

A message published to the event bus to notify subscribers of actions or state changes.

**See**: [Event Bus](#event-bus)

### Event Bus

A pub/sub messaging system that enables loose coupling between components. Supports multiple backends (In-Memory, Kafka, SQS, RabbitMQ, Redis).

**Location**: `victor/core/events/`

### Extension

Additional functionality that can be added to Victor, such as custom providers, tools, or verticals.

---

## F

### Facade Pattern

A design pattern that provides a simplified interface to a complex subsystem. AgentOrchestrator is the main facade.

**See**: [AgentOrchestrator](#agentorchestrator)

### Feature Enhancement Proposal (FEP)

A proposal for adding new features to Victor.

**Location**: `docs/feps/`

### Framework

Reusable capabilities and components provided by Victor for building verticals and extensions.

**Location**: `victor/framework/`

---

## G

### Graph

See [StateGraph](#stategraph)

---

## H

### Handler

A function or class that processes workflow steps or handles specific events.

**See**: [Step Handler](#step-handler)

### HITL (Human-in-the-Loop)

A workflow pattern that requires human approval or input at certain points.

**Location**: `victor/framework/hitl/`

---

## I

### ISP (Interface Segregation Principle)

A SOLID principle stating that clients should not depend on interfaces they don't use. Victor implements ISP through 98 lean protocols.

**See**: [ISP Migration](./architecture/ISP_MIGRATION_GUIDE.md)

### Integration Test

A test that verifies multiple components work together correctly.

**Location**: `tests/integration/`

---

## L

### LLM (Large Language Model)

An AI model trained on large amounts of text that can understand and generate human-like text.

### LSP (Liskov Substitution Principle)

A SOLID principle stating that subtypes must be substitutable for their base types.

**Implementation**: All providers inherit from BaseProvider, all tools inherit from BaseTool

---

## M

### MCP (Model Context Protocol)

A protocol for integrating AI assistants with IDEs and other tools.

**See**: [MCP Integration](./user-guide/index.md)

### Message

A unit of conversation between user and agent, or between components in the event system.

### Middleware

Code that processes requests/responses before/after they reach their destination. Used for logging, authentication, etc.

**Location**: `victor/framework/middleware/`

### Mode

A configuration preset that adjusts agent behavior (e.g., build, plan, explore modes).

**See**: [Mode Configuration](./getting-started/configuration.md)

### Multi-Agent

A system where multiple AI agents work together, each with specialized roles.

**See**: [Agent Swarming](./AGENT_SWARMING_GUIDE.md)

---

## O

### OCP (Open/Closed Principle)

A SOLID principle stating that software should be open for extension but closed for modification.

**Implementation**: Plugin system for providers, tools, and verticals

### Orchestrator

See [AgentOrchestrator](#agentorchestrator)

---

## P

### Pipeline

A sequence of operations applied to data. ToolPipeline manages tool selection and execution.

**Location**: `victor/agent/tool_pipeline.py`

### Protocol

An interface definition using Python's typing.Protocol. Victor defines 98 protocols for type safety and loose coupling.

**See**: [Protocols Reference](./architecture/PROTOCOLS_REFERENCE.md)

### Provider

An LLM service integration (e.g., Anthropic, OpenAI, Ollama). Victor supports 21 providers.

**See**: [Providers](./user-guide/providers.md)

### Provider Pool

A collection of provider instances that can be reused across requests for improved performance.

---

## R

### RAG (Retrieval-Augmented Generation)

A technique that enhances LLM responses with retrieved information from a knowledge base.

**See**: [RAG Guide](./ADVANCED_RAG_GUIDE.md)

### Registry

A centralized store for managing entities (modes, teams, capabilities, workflows, etc.).

**See**: [Universal Registry](#universal-registry)

### Repository Pattern

A design pattern that mediates between the domain and data mapping layers. Used in Victor's configuration management.

### RL (Reinforcement Learning)

A machine learning approach where agents learn by receiving rewards/punishments. Victor has RL capabilities for tool selection.

**Location**: `victor/framework/rl/`

---

## S

### ServiceContainer

Victor's dependency injection container that manages 55+ services.

**Location**: `victor/core/container.py`

### Session

A persistent conversation context that can be resumed across multiple interactions.

**Command**: `victor chat --session my-session`

### SOLID

Five principles of object-oriented design:
- **SRP**: Single Responsibility Principle
- **OCP**: Open/Closed Principle
- **LSP**: Liskov Substitution Principle
- **ISP**: Interface Segregation Principle
- **DIP**: Dependency Inversion Principle

**See**: [Design Patterns](./architecture/DESIGN_PATTERNS.md)

### SRP (Single Responsibility Principle)

A SOLID principle stating that a class should have one reason to change. Victor implements SRP through specialized coordinators.

**See**: [Coordinator Pattern](#coordinator-pattern)

### StateGraph

A workflow definition using a graph of nodes and edges, with state management and checkpointing.

**Location**: `victor/framework/graph.py`

### Step Handler

A function or class that executes a specific step in a YAML workflow.

**See**: [Step Handler Guide](./extensions/step_handler_guide.md)

### Stream

Continuous output of data as it's generated, rather than waiting for completion.

**Command**: `victor chat --stream`

### SubAgent

A specialized agent that handles specific tasks, coordinated by a main agent. Uses lean SubAgentContext protocol (ISP compliance).

**Location**: `victor/agent/subagents/`

---

## T

### Tool

A reusable component that performs a specific function (e.g., read_file, search_web). Victor has 55 specialized tools.

**See**: [Tools Reference](./user-guide/tools.md)

### Tool Calling

The ability of LLMs to invoke tools/functions as part of their response.

**See**: [Tool Calling](./architecture/COMPONENT_REFERENCE.md)

### Tool Selection

The process of choosing which tools to use for a given task. Victor uses semantic, keyword, or hybrid strategies.

**See**: [Tool Selection](./performance/tool_selection_caching.md)

### Tree-sitter

A parser generator tool for building ASTs. Victor uses it for code analysis.

**See**: [AST](#ast)

### TUI (Terminal User Interface)

An interactive terminal interface for Victor, enabled by default.

**Command**: `victor chat --tui`

---

## U

### Universal Registry

A type-safe, thread-safe registry framework supporting multiple cache strategies (TTL, LRU, Manual, None).

**Location**: `victor/core/registries/universal_registry.py`

---

## V

### Validation

The process of verifying data or configuration meets requirements. Victor has a validation pipeline with built-in validators.

**Location**: `victor/framework/validation/`

### Vertical

A domain-specific module (e.g., coding, devops, rag) with specialized tools and capabilities.

**See**: [Verticals](./verticals/README.md)

### VerticalBase

The base class that all verticals inherit from, providing common functionality.

**Location**: `victor/core/verticals/base.py`

---

## W

### Workflow

A multi-step automation defined in YAML using StateGraph DSL or Python code.

**See**: [Workflows Guide](./user-guide/workflows.md)

### Workflow Compiler

Converts YAML workflow definitions into executable StateGraphs.

**Location**: `victor/workflows/unified_compiler.py`

---

## Y

### YAML

A human-readable data serialization language used for workflows, configurations, and capabilities in Victor.

---

## Acronyms

| Acronym | Full Term | Definition |
|---------|-----------|------------|
| ADR | Architecture Decision Record | Document describing architectural decisions |
| API | Application Programming Interface | Interface for software components |
| AST | Abstract Syntax Tree | Tree representation of code structure |
| CI | Continuous Integration | Automated integration of code changes |
| CLI | Command-Line Interface | Text-based interface |
| CD | Continuous Deployment | Automated deployment of code changes |
| DI | Dependency Injection | Pattern for providing dependencies |
| DSL | Domain-Specific Language | Specialized language for specific domain |
| DIP | Dependency Inversion Principle | SOLID principle |
| HITL | Human-in-the-Loop | Workflow pattern with human interaction |
| ISP | Interface Segregation Principle | SOLID principle |
| LLM | Large Language Model | AI model for text understanding/generation |
| LSP | Liskov Substitution Principle | SOLID principle |
| LRU | Least Recently Used | Cache eviction strategy |
| MCP | Model Context Protocol | IDE integration protocol |
| OCP | Open/Closed Principle | SOLID principle |
| RAG | Retrieval-Augmented Generation | LLM enhancement with retrieval |
| RL | Reinforcement Learning | Learning through rewards/punishments |
| SOLID | 5 OOP Principles | SRP, OCP, LSP, ISP, DIP |
| SRP | Single Responsibility Principle | SOLID principle |
| TUI | Terminal User Interface | Interactive terminal interface |
| TTL | Time To Live | Cache expiration time |

---

## Common Patterns

### Facade Pattern

Provides a simplified interface to a complex subsystem.

**Example**: AgentOrchestrator

### Strategy Pattern

Defines a family of algorithms and makes them interchangeable.

**Example**: Tool selection strategies (keyword, semantic, hybrid)

### Observer Pattern

Defines a subscription mechanism to notify multiple objects about events.

**Example**: Event bus with pub/sub

### Factory Pattern

Creates objects without specifying the exact class.

**Example**: ProviderFactory, ToolFactory

### Builder Pattern

Constructs complex objects step by step.

**Example**: AgentBuilder

### Template Method Pattern

Defines the skeleton of an algorithm in a base class, letting subclasses override specific steps.

**Example**: BaseYAMLWorkflowProvider

### Registry Pattern

Provides a centralized store for managing entities.

**Example**: UniversalRegistry

---

## Related Concepts

### Clean Architecture

Architectural style emphasizing separation of concerns and dependency inversion.

**See**: [Architecture Overview](./architecture/overview.md)

### Microservices

Architectural style where applications are composed of small, independent services.

**See**: [Deployment Guide](./DEPLOYMENT_GUIDE.md)

### Event-Driven Architecture

Architectural style where components communicate through events.

**See**: [Event System](./observability/event_system.md)

### Vertical Architecture

Architectural style organizing functionality into domain-specific verticals.

**See**: [Verticals](./verticals/README.md)

---

## Resources

### Documentation

- [Documentation Index](./INDEX.md) - Complete documentation
- [Architecture Documentation](./architecture/README.md) - Architecture docs
- [API Reference](./api/README.md) - API documentation

### Learning

- [Quick Start](./QUICKSTART.md) - Get started quickly
- [Developer Onboarding](./DEVELOPER_ONBOARDING.md) - Developer guide
- [Architect Quick Start](./ARCHITECT_QUICKSTART.md) - Architecture guide

---

<div align="center">

**Can't find a term?**

[Open an Issue](https://github.com/vjsingh1984/victor/issues/new) •
[Join Discussions](https://github.com/vjsingh1984/victor/discussions)

**[Back to Documentation Index](./INDEX.md)**

</div>
