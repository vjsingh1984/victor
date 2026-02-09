# Creational Design Patterns

**Version**: 0.5.0
**Last Updated**: January 31, 2026
**Audience**: Developers, Contributors
**Purpose**: Factory, Builder, and Singleton patterns in Victor AI

---

## Overview

Victor AI employs a comprehensive set of design patterns to achieve maintainability, testability,
  and extensibility. These patterns are applied consistently across the codebase,
  following SOLID principles and protocol-based design.

### Pattern Statistics

| Category | Patterns Used | Count |
|----------|---------------|-------|
| **Creational** | Factory, Builder | 2 |
| **Structural** | Facade, Adapter, Mixin, Decorator, Bridge | 5 |
| **Behavioral** | Strategy, Observer, Command, Chain of Responsibility, Template Method | 5 |
| **Architecture** | Protocol, DI Container, Event-Driven, Coordinator | 4 |
| **Total** | | **16** |

### Pattern Benefits

- **Maintainability**: 93% reduction in core complexity
- **Testability**: 81% test coverage achieved
- **Extensibility**: Plugin-based architecture
- **Flexibility**: 21 LLM providers, 55 tools
- **Scalability**: Event-driven architecture

---


## Creational Patterns

### Factory Pattern

**Intent**: Centralize object creation with proper dependency injection

**Problem**:
- Complex object initialization logic
- Need for consistent object creation
- Testability requires dependency injection
- Don't want to expose construction details

**Solution**: Factory classes encapsulate object creation logic

**Implementation**: `OrchestratorFactory`

**File Location**: `/Users/vijaysingh/code/codingagent/victor/agent/orchestrator_factory.py`

**Structure**:
```python
class OrchestratorFactory:
    """Factory for creating AgentOrchestrator instances."""

    def __init__(self, settings: Settings, provider: BaseProvider, model: str):
        self._settings = settings
        self._provider = provider
        self._model = model

    def create_orchestrator(self) -> AgentOrchestrator:
        """Create orchestrator with all dependencies."""
        # Create provider components
        provider_components = self._create_provider_components()

        # Create core services
        core_services = self._create_core_services()

        # Create conversation components
        conversation_components = self._create_conversation_components()

        # Create and return orchestrator
        return AgentOrchestrator(
            provider_components=provider_components,
            core_services=core_services,
            conversation_components=conversation_components,
        )

    def _create_provider_components(self) -> ProviderComponents:
        """Create provider-related components."""
        # Implementation

    def _create_core_services(self) -> CoreServices:
        """Create core service components."""
        # Implementation

    def _create_conversation_components(self) -> ConversationComponents:
        """Create conversation components."""
        # Implementation
```

**Usage Example**:
```python
# Production use
factory = OrchestratorFactory(settings, provider, model)
orchestrator = factory.create_orchestrator()

# Testing use
factory = OrchestratorFactory(settings, provider, model)
sanitizer = factory.create_sanitizer()
prompt_builder = factory.create_prompt_builder()
# Test individual components
```

**Benefits**:
1. **Centralized Creation**: All creation logic in one place
2. **Dependency Injection**: Proper DI with ServiceContainer
3. **Testability**: Easy to create test-specific components
4. **Consistency**: Ensures objects are created correctly
5. **Encapsulation**: Hides complex initialization

**Related Patterns**:
- Builder Pattern (for complex object construction)
- Dependency Injection (for managing dependencies)

---

### Builder Pattern

**Intent**: Separate construction of complex object from its representation

**Problem**:
- Complex objects with many parameters
- Need for step-by-step construction
- Want immutable objects
- Need fluent interface

**Solution**: Builder classes provide step-by-step construction

**Implementation**: `PromptBuilder`, `AgentBuilder`

**File Location**: `/Users/vijaysingh/code/codingagent/victor/framework/prompt_builder.py`

**Structure**:
```python
class PromptBuilder:
    """Builder for constructing prompts."""

    def __init__(self):
        self._sections = []

    def add_system_message(self, message: str) -> "PromptBuilder":
        """Add system message."""
        self._sections.append(PromptSection(
            type="system",
            content=message
        ))
        return self

    def add_context(self, context: Dict[str, Any]) -> "PromptBuilder":
        """Add context."""
        self._sections.append(PromptSection(
            type="context",
            content=context
        ))
        return self

    def add_tool_hints(self, tools: List[BaseTool]) -> "PromptBuilder":
        """Add tool hints."""
        self._sections.append(PromptSection(
            type="tools",
            content=tools
        ))
        return self

    def build(self) -> str:
        """Build final prompt."""
        return "\n\n".join(
            section.render() for section in self._sections
        )
```

**Usage Example**:
```python
# Build prompt step by step
prompt = (PromptBuilder()
    .add_system_message("You are a helpful assistant.")
    .add_context({"project": "victor", "language": "python"})
    .add_tool_hints([read_tool, write_tool, search_tool])
    .build())

# Result: Complete formatted prompt
print(prompt)
```

**Benefits**:
1. **Step-by-Step**: Clear construction flow
2. **Fluent Interface**: Method chaining
3. **Immutable**: Builder can be reused
4. **Readable**: Self-documenting code
5. **Flexible**: Easy to add new construction steps

**Related Patterns**:
- Factory Pattern (for object creation)
- Immutable Object (for built objects)

---

---

**Last Updated:** February 01, 2026
**Reading Time:** 2 min
