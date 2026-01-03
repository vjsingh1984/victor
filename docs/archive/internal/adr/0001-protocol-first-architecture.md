# ADR 0001: Protocol-First Architecture

Date: 2025-12-16

## Status

Accepted

## Context

As Victor evolved from a simple CLI tool to an enterprise-ready AI coding assistant with multiple client interfaces (CLI, TUI, VS Code extension, MCP server), we faced several architectural challenges:

1. **Tight Coupling**: Components directly imported concrete implementations, creating circular dependencies and making testing difficult
2. **Testing Complexity**: Mocking deeply integrated components required extensive setup and brittle test fixtures
3. **Client Proliferation**: Adding new clients (VS Code, JetBrains, MCP) required duplicating orchestration logic
4. **Component Substitution**: Replacing implementations (e.g., swapping providers, tool executors) required changes throughout the codebase
5. **Circular Import Hell**: Chains like `orchestrator ↔ evaluation ↔ agent_adapter` blocked development

Traditional approaches considered:
- **Abstract Base Classes (ABCs)**: Requires inheritance, limiting flexibility and creating rigid hierarchies
- **Duck Typing**: No compile-time checking, difficult to discover capabilities
- **Direct Dependency Injection**: Still requires concrete type knowledge at injection points

## Decision

We adopt a **Protocol-First Architecture** using Python's structural typing (PEP 544) to define interfaces for all major components. This means:

1. **Protocols Define Contracts**: Every major component (providers, tools, orchestrator, conversation controller, etc.) is defined first as a `@runtime_checkable Protocol` in dedicated protocol modules
2. **Implementations Are Secondary**: Concrete classes implement protocols through structural subtyping (duck typing with type checking)
3. **Dependency Inversion**: High-level modules (orchestrator) depend on protocol abstractions, not concrete implementations
4. **Two Protocol Layers**:
   - **Core Protocols** (`victor/core/protocols.py`): Break circular import chains between major subsystems
   - **Component Protocols** (`victor/agent/protocols.py`): Define injectable service contracts for orchestrator

### Key Protocol Files

#### victor/core/protocols.py
Breaks circular import chains by defining high-level interfaces:
- `OrchestratorProtocol`: Allows evaluation/middleware to depend on orchestrator without importing it
- `TaskClassifierProtocol`: Enables task analysis without heavy ML dependencies
- `ToolCallingAdapterProtocol`: Decouples tool calling logic from provider implementations
- `ProviderProtocol`: Minimal provider interface for type checking

#### victor/agent/protocols.py
Defines 30+ protocols for orchestrator service injection:
- Provider protocols: `ProviderManagerProtocol`
- Tool protocols: `ToolRegistryProtocol`, `ToolSelectorProtocol`, `ToolPipelineProtocol`, `ToolExecutorProtocol`
- Conversation protocols: `ConversationControllerProtocol`, `ConversationStateMachineProtocol`, `MessageHistoryProtocol`
- Analysis protocols: `TaskAnalyzerProtocol`, `ComplexityClassifierProtocol`, `ActionAuthorizerProtocol`
- Utility protocols: `ArgumentNormalizerProtocol`, `ResponseSanitizerProtocol`, `ProjectContextProtocol`

### Design Principles

1. **Explicit Over Implicit**: No auto-wiring or magic. Dependencies are explicitly declared and injected
2. **Structural Typing**: Protocols use duck typing with `@runtime_checkable` for isinstance() checks
3. **Protocol Segregation**: Small, focused protocols rather than monolithic interfaces
4. **Runtime Validation**: Protocols can be validated at runtime with isinstance()
5. **Type Safety**: Full type checking support via mypy/pyright

## Consequences

### Positive Consequences

- **Testability**: Mock implementations satisfy protocols without complex inheritance
- **Flexibility**: Any class matching the protocol's signature can be substituted
- **Circular Import Resolution**: Protocol modules have minimal dependencies, breaking cycles
- **Client Decoupling**: New clients (VS Code, MCP) can be added without modifying core
- **Type Safety**: Full IDE autocomplete and type checking support
- **Documentation**: Protocols serve as executable documentation of component contracts
- **Backward Compatibility**: Existing code continues working; protocols are gradually adopted

### Negative Consequences

- **Learning Curve**: Team must understand protocol-based design patterns
- **Boilerplate**: Protocol definitions add lines of code (though less than ABCs)
- **Runtime Overhead**: `isinstance()` checks on protocols are slightly slower than concrete types
- **Protocol Duplication**: Some methods appear in multiple protocol files for different use cases

### Risks and Mitigations

- **Risk**: Protocols drift out of sync with implementations
  - **Mitigation**: Unit tests verify concrete classes satisfy protocols (`assert isinstance(obj, Protocol)`)

- **Risk**: New developers may bypass protocols and import concrete classes
  - **Mitigation**: Import validation in CI checks for circular dependencies and protocol usage

- **Risk**: Protocol methods without implementations could be called
  - **Mitigation**: Type checking catches missing implementations at development time

## Implementation Notes

### Protocol Naming Convention
- Protocol names end with `Protocol` suffix (e.g., `ToolRegistryProtocol`)
- Place in dedicated `protocols.py` modules
- Use `@runtime_checkable` decorator for isinstance() support

### Usage Pattern
```python
# GOOD: Depend on protocol
from victor.agent.protocols import ToolRegistryProtocol

def process_tools(registry: ToolRegistryProtocol) -> None:
    tools = registry.list_tools()
    ...

# BAD: Depend on concrete class
from victor.tools.base import ToolRegistry  # Creates coupling

def process_tools(registry: ToolRegistry) -> None:
    ...
```

### Testing Pattern
```python
from victor.agent.protocols import ProviderManagerProtocol

def test_orchestrator_with_mock():
    # Mock satisfies protocol through duck typing
    mock_provider = MagicMock(spec=ProviderManagerProtocol)
    mock_provider.model = "gpt-4"
    mock_provider.provider_name = "openai"

    orchestrator = AgentOrchestrator(provider_manager=mock_provider)
    # Test passes without complex fixture setup
```

### Key Files

**Protocol Definitions**:
- `victor/core/protocols.py` - Core system protocols (157 lines)
- `victor/agent/protocols.py` - Orchestrator service protocols (805 lines)
- `victor/verticals/protocols.py` - Vertical extension protocols (797 lines)

**Consumers**:
- `victor/agent/orchestrator.py` - Accepts all services as protocol types
- `victor/agent/service_provider.py` - Registers concrete implementations
- `victor/core/container.py` - DI container works with protocol types
- Tests throughout codebase use protocols for mocking

## References

- PEP 544 (Protocols: Structural subtyping): https://peps.python.org/pep-0544/
- Related: [ADR-0004: Phase 10 DI Migration](0004-phase-10-di-migration.md)
- Implementation PR: feat: Complete Victor Framework Evolution (Phases 1-9)
- CLAUDE.md: Search for "Protocol System" section
