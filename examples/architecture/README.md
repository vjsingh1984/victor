# Architecture Examples

Practical examples demonstrating the new architectural patterns from Victor's refactoring. These examples show how to use protocols, dependency injection, event-driven architecture, and coordinators in real-world scenarios.

## Overview

The refactoring introduced several key architectural patterns that improve code quality, maintainability, and testability:

1. **Protocol-Based Design** - Loose coupling through interfaces
2. **Dependency Injection Container** - Flexible service management
3. **Event-Driven Architecture** - Decoupled communication via events
4. **Coordinator Pattern** - Focused, composable coordinators

## Examples

### 1. Protocol Usage Example

**File:** `protocol_usage.py`

Demonstrates protocol-based design patterns for loose coupling.

**What it shows:**
- How to define custom protocols using `typing.Protocol`
- Implementing protocols in concrete classes
- Using protocols for dependency injection
- Testing with protocol mocks
- Before/after comparison of tight vs loose coupling

**Key concepts:**
- Interface Segregation Principle (ISP)
- Dependency Inversion Principle (DIP)
- Open/Closed Principle (OCP)
- Structural typing (no inheritance required)

**Run:**
```bash
python -m examples.architecture.protocol_usage
```

**Output:**
- Demonstrates swapping tool selection strategies without changing coordinator
- Shows mock-based testing without concrete dependencies
- Explains benefits of protocol-based design

### 2. DI Container Usage Example

**File:** `di_container_usage.py`

Demonstrates dependency injection for flexible service management.

**What it shows:**
- Registering services with different lifetimes (singleton, scoped, transient)
- Auto-resolution of constructor dependencies
- Using scoped containers for request isolation
- Overriding services for testing
- Lifecycle management with disposable services

**Key concepts:**
- Service lifetime management (SINGLETON, SCOPED, TRANSIENT)
- Constructor injection
- Service override for testing
- Resource cleanup with `Disposable` protocol

**Run:**
```bash
python -m examples.architecture.di_container_usage
```

**Output:**
- Shows automatic dependency resolution
- Demonstrates scoped service isolation
- Displays mock-based testing with service override
- Explains lifecycle management

### 3. Event-Driven Architecture Example

**File:** `event_usage.py`

Demonstrates event-driven communication for loose coupling.

**What it shows:**
- Defining domain events
- Publishing events to an event bus
- Subscribing to events with handlers
- Event filtering for selective handling
- Event correlation by session ID

**Key concepts:**
- Breaking circular dependencies
- Event-driven communication
- Event filtering and routing
- Correlation IDs for tracing
- Loose coupling between components

**Run:**
```bash
python -m examples.architecture.event_usage
```

**Output:**
- Shows evaluation events being published
- Demonstrates multiple independent listeners reacting
- Displays event filtering (quality alerts on low scores)
- Explains event correlation across sessions

### 4. Coordinator Pattern Example

**File:** `coordinator_usage.py`

Demonstrates the coordinator pattern for focused, composable components.

**What it shows:**
- Creating custom coordinators (Cache, Tool, Analytics)
- Using delegation to distribute responsibilities
- Integrating coordinators with orchestrator
- Composing coordinators for complex operations
- Testing with mock coordinators

**Key concepts:**
- Single Responsibility Principle (SRP)
- Delegation pattern
- Coordinator composition
- Separation of concerns
- Testing with mocks

**Run:**
```bash
python -m examples.architecture.coordinator_usage
```

**Output:**
- Shows orchestrator delegating to coordinators
- Demonstrates coordinator composition
- Displays status from multiple coordinators
- Explains testing with mock coordinators

## Architecture Patterns

### Protocol-Based Design

**Before (Tight Coupling):**
```python
class ToolSelector:
    def __init__(self, semantic_selector: SemanticToolSelector):
        self.semantic = semantic_selector  # Concrete dependency

# Hard to test, hard to swap implementations
```

**After (Protocol-Based):**
```python
class ToolSelector:
    def __init__(self, selector: IToolSelector):
        self.selector = selector  # Protocol dependency

# Easy to test with mocks, easy to swap implementations
```

**Benefits:**
- Loose coupling (DIP)
- Easy to extend (OCP)
- Simple to test with mocks
- Type-safe with runtime checking

### Dependency Injection Container

**Before (Manual DI):**
```python
class Orchestrator:
    def __init__(self):
        self.metrics = MetricsCollector()  # Hard dependency
        self.logger = DebugLogger()  # Hard dependency

# Hard to test, shared state issues
```

**After (DI Container):**
```python
container = ServiceContainer()
container.register(MetricsService, lambda c: MetricsCollector(), SINGLETON)
container.register(LoggerService, lambda c: DebugLogger(), SINGLETON)

orchestrator = container.get(Orchestrator)

# Easy to test, flexible configuration, proper lifecycle
```

**Benefits:**
- Centralized service configuration
- Consistent lifecycle management
- Easy testing via override_services
- Type-safe service resolution

### Event-Driven Architecture

**Before (Tight Coupling):**
```python
class Orchestrator:
    def complete_evaluation(self, result):
        evaluation_coordinator.process_result(result)  # Direct call
        rl_coordinator.update(result)  # Direct call
        analytics_coordinator.record(result)  # Direct call

# Tight coupling, circular dependencies
```

**After (Event-Driven):**
```python
class Orchestrator:
    def complete_evaluation(self, result):
        event = EvaluationCompletedEvent(...)
        bus.emit(event)  # Publish and forget

# Loose coupling, no circular dependencies
```

**Benefits:**
- Breaks circular dependencies
- Components can be added/removed without changes
- Event filtering enables selective handling
- Event correlation for distributed tracing

### Coordinator Pattern

**Before (God Object):**
```python
class AgentOrchestrator:
    def __init__(self):
        # 100+ lines of initialization
        self.tool_selection = ...
        self.budget_management = ...
        self.caching = ...
        # ... many more concerns

    def select_tools(self):
        # 200+ lines of tool selection logic

    # ... 50+ methods, 3000+ lines of code
```

**After (Coordinator Pattern):**
```python
class AgentOrchestrator:
    def __init__(self):
        self.tool_coordinator = ToolCoordinator(...)
        self.cache_coordinator = CacheCoordinator(...)

    def select_tools(self):
        return self.tool_coordinator.select_tools(...)  # Delegate

# Each coordinator is focused, testable, and reusable
```

**Benefits:**
- Single responsibility per coordinator
- Easier to test and maintain
- Coordinators can be reused
- Reduces orchestrator complexity

## Running the Examples

All examples are standalone and can be run directly:

```bash
# Run individual examples
python -m examples.architecture.protocol_usage
python -m examples.architecture.di_container_usage
python -m examples.architecture.event_usage
python -m examples.architecture.coordinator_usage

# Or run from the examples/architecture directory
cd examples/architecture
python protocol_usage.py
python di_container_usage.py
python event_usage.py
python coordinator_usage.py
```

## Testing the Examples

Each example includes testing demonstrations:

```bash
# Protocol usage - shows mock-based testing
python protocol_usage.py  # Includes test_coordinator_with_mock()

# DI container - shows service override for testing
python di_container_usage.py  # Includes demonstrate_testing_with_mocks()

# Event usage - shows event filtering and correlation
python event_usage.py  # Includes filtering and correlation demos

# Coordinator usage - shows mock coordinators
python coordinator_usage.py  # Includes demonstrate_testing_with_mocks()
```

## Real-World Usage in Victor

These patterns are used throughout Victor:

### Protocols
- `victor/protocols/` - 98+ protocol definitions
- Tool selection, quality assessment, grounding, search, etc.
- Used for dependency injection and testing

### DI Container
- `victor/core/container.py` - Service container implementation
- `victor/agent/service_provider.py` - Orchestrator service registration
- 55+ protocols registered for orchestrator dependencies

### Events
- `victor/observability/events.py` - Domain events
- `victor/core/events/` - Event bus implementation
- Used for breaking circular dependencies

### Coordinators
- `victor/agent/coordinators/` - Agent-level coordinators
- `victor/framework/coordinators/` - Framework-level coordinators
- Tool, Cache, Analytics, Session, and other coordinators

## Code Style

All examples follow Victor's code style guidelines:
- Type hints on all public APIs
- Google-style docstrings
- Line length: 100 chars (black enforced)
- Async/await for all I/O
- Protocol-based interfaces
- Dependency injection

## Further Reading

- **Main CLAUDE.md**: Project documentation and architecture
- **victor/protocols/**: Protocol definitions
- **victor/core/container.py**: DI container implementation
- **victor/observability/events.py**: Domain events
- **victor/agent/coordinators/**: Coordinator implementations

## Contributing

When adding new examples:
1. Make them runnable with `python -m examples.architecture.xxx`
2. Include docstrings explaining concepts
3. Show realistic use cases from Victor
4. Demonstrate both simple and advanced usage
5. Follow Victor code style guidelines
6. Update this README with the new example

## Summary

These examples demonstrate how Victor's refactoring improved code quality through:

1. **Loose Coupling** - Protocols and events break dependencies
2. **Testability** - DI container enables easy mocking
3. **Maintainability** - Coordinators separate concerns
4. **Flexibility** - Easy to swap implementations
5. **Scalability** - Event-driven architecture scales

Each example is self-contained and demonstrates practical patterns you can apply in your own code.
