# ADR 0002: Vertical System Design

Date: 2025-12-16

## Status

Accepted

## Context

As Victor evolved to support multiple use cases beyond just coding (research, data analysis, DevOps), we needed a way to specialize the assistant's behavior for different domains without hardcoding domain-specific logic into the core framework.

### Problems with Monolithic Design

1. **Domain Logic Pollution**: Core orchestrator contained hardcoded coding-specific behavior (git safety checks, code review middleware)
2. **Extension Difficulty**: Adding new domains (research, DevOps) required modifying framework code
3. **Configuration Sprawl**: Mode configs, tool selections, prompt hints scattered across modules
4. **Testing Complexity**: Testing domain-specific behavior required loading entire framework
5. **Framework Rigidity**: No way to customize behavior per domain without forking code

### Alternative Approaches Considered

- **Configuration Files**: YAML/JSON configs for each domain
  - **Con**: Not code, can't encapsulate complex logic
  - **Con**: No type checking or IDE support

- **Plugin System**: Load domains as plugins
  - **Con**: Complex discovery mechanism
  - **Con**: Version compatibility challenges

- **Inheritance Hierarchy**: AgentOrchestrator → CodingOrchestrator → DataAnalysisOrchestrator
  - **Con**: Deep inheritance creates coupling
  - **Con**: Can't compose behaviors from multiple domains

## Decision

We implement a **Vertical System** that allows domain-specific specialization through protocol-based extension points. The framework remains domain-agnostic while verticals inject behavior through well-defined protocols.

### Architecture

#### Three-Layer Design

```
┌────────────────────────────────────────────────────────────┐
│  VERTICAL LAYER                                            │
│  (Domain-Specific: Coding, Research, DevOps)              │
├────────────────────────────────────────────────────────────┤
│  - CodingAssistant: 45+ tools, git safety, code hints     │
│  - ResearchAssistant: Web tools, citation tracking        │
│  - DevOpsAssistant: Docker, k8s, monitoring tools         │
│  - DataAnalysisAssistant: Pandas, visualization, SQL      │
└──────────────┬─────────────────────────────────────────────┘
               │ Implements Extension Protocols
               ▼
┌────────────────────────────────────────────────────────────┐
│  PROTOCOL LAYER                                            │
│  (Extension Points: victor/verticals/protocols.py)        │
├────────────────────────────────────────────────────────────┤
│  - MiddlewareProtocol: Pre/post tool execution            │
│  - SafetyExtensionProtocol: Domain danger patterns        │
│  - PromptContributorProtocol: System prompts & hints      │
│  - ModeConfigProviderProtocol: Mode configurations        │
│  - ToolDependencyProviderProtocol: Tool relationships     │
│  - WorkflowProviderProtocol: Domain workflows             │
│  - ServiceProviderProtocol: DI service registration       │
└──────────────┬─────────────────────────────────────────────┘
               │ Used by
               ▼
┌────────────────────────────────────────────────────────────┐
│  FRAMEWORK LAYER                                           │
│  (Domain-Agnostic: AgentOrchestrator, ToolPipeline)      │
├────────────────────────────────────────────────────────────┤
│  - Loads verticals via VerticalLoader                     │
│  - Calls vertical middleware before/after tool execution  │
│  - Merges vertical prompts into system prompt             │
│  - Uses vertical safety patterns for validation           │
│  - Applies vertical mode configs and tool dependencies    │
└────────────────────────────────────────────────────────────┘
```

### Extension Protocols

#### MiddlewareProtocol
Intercepts tool execution for validation/transformation:
```python
class CodeValidationMiddleware(MiddlewareProtocol):
    async def before_tool_call(self, tool_name, arguments):
        if tool_name == "write_file":
            # Validate syntax before writing
            ...
        return MiddlewareResult(proceed=True)
```

#### SafetyExtensionProtocol
Defines domain-specific danger patterns:
```python
class GitSafetyExtension(SafetyExtensionProtocol):
    def get_bash_patterns(self):
        return [
            SafetyPattern(
                pattern=r"git\s+push\s+.*--force",
                description="Force push (may lose commits)",
                risk_level="HIGH"
            )
        ]
```

#### PromptContributorProtocol
Adds domain-specific prompt sections and task hints:
```python
class CodingPromptContributor(PromptContributorProtocol):
    def get_task_type_hints(self):
        return {
            "edit": TaskTypeHint(
                task_type="edit",
                hint="[EDIT] Read target file first",
                tool_budget=5,
                priority_tools=["read_file", "edit_files"]
            )
        }
```

#### TieredToolConfig
Three-tier tool selection system:
1. **Mandatory**: Always included (e.g., "read", "ls", "grep")
2. **Vertical Core**: Always included for this vertical (e.g., "git" for coding)
3. **Semantic Pool**: Dynamically selected based on task

### Vertical Loader

`victor/verticals/vertical_loader.py` provides automatic discovery:
- Scans `victor/verticals/` for vertical modules
- Validates protocol implementations
- Merges extensions into framework
- Supports runtime vertical switching

### Key Design Principles

1. **Framework Agnostic**: Core contains ZERO domain logic
2. **Protocol-Driven**: Extensions through protocols, not inheritance
3. **Composable**: Can combine multiple vertical extensions
4. **Independent**: Verticals can function standalone for testing
5. **Discoverable**: Automatic loading via VerticalLoader

## Consequences

### Positive Consequences

- **Separation of Concerns**: Domain logic isolated in vertical modules
- **Extensibility**: New domains added without framework changes
- **Testability**: Verticals tested independently from framework
- **Maintainability**: Changes to coding behavior don't affect research/DevOps
- **Clarity**: Clear contracts through protocols
- **Reusability**: Vertical extensions can be shared across domains

### Negative Consequences

- **Indirection**: Extension points add conceptual overhead
- **Discovery Complexity**: Understanding behavior requires checking vertical code
- **Protocol Maintenance**: Changes to protocols require updating all verticals
- **Performance**: Middleware hooks add minimal latency to tool execution

### Risks and Mitigations

- **Risk**: Verticals could conflict (e.g., competing middleware)
  - **Mitigation**: Priority-based middleware execution order
  - **Mitigation**: Validation at registration time

- **Risk**: Breaking changes to protocols affect all verticals
  - **Mitigation**: Versioned protocols with deprecation cycle
  - **Mitigation**: Comprehensive test coverage

- **Risk**: Framework leakage (domain logic sneaking into core)
  - **Mitigation**: Linting rules to detect hardcoded domain logic
  - **Mitigation**: Code review guidelines

## Implementation Notes

### Vertical Structure

Each vertical lives in `victor/verticals/{domain}/`:
```
victor/verticals/coding/
├── __init__.py
├── assistant.py          # CodingAssistant (VerticalBase)
├── middleware.py         # CodeValidationMiddleware
├── safety.py            # GitSafetyExtension
├── prompts.py           # CodingPromptContributor
├── mode_config.py       # CodingModeProvider
├── tool_dependencies.py # CodingToolDependencies
├── service_provider.py  # CodingServiceProvider
└── workflows/           # Coding workflows
```

### Registration Pattern

```python
# In vertical's assistant.py
class CodingAssistant(VerticalBase):
    @classmethod
    def get_extensions(cls) -> VerticalExtensions:
        return VerticalExtensions(
            middleware=[CodeValidationMiddleware()],
            safety_extensions=[GitSafetyExtension()],
            prompt_contributors=[CodingPromptContributor()],
            mode_config_provider=CodingModeProvider(),
            tool_dependency_provider=CodingToolDependencies(),
            service_provider=CodingServiceProvider(),
        )
```

### Framework Integration

```python
# In orchestrator initialization
from victor.verticals.vertical_loader import load_vertical

vertical = load_vertical("coding")
extensions = vertical.get_extensions()

# Register middleware
for middleware in extensions.middleware:
    tool_pipeline.add_middleware(middleware)

# Merge safety patterns
safety_checker.add_patterns(extensions.get_all_safety_patterns())

# Augment system prompt
system_prompt += vertical.get_system_prompt()
```

### Key Files

**Protocol Definitions**:
- `victor/verticals/protocols.py` (797 lines)

**Vertical Implementations**:
- `victor/verticals/coding/` - Coding vertical (complete)
- `victor/verticals/research/` - Research vertical
- `victor/verticals/devops/` - DevOps vertical
- `victor/verticals/data_analysis/` - Data analysis vertical

**Framework Integration**:
- `victor/verticals/vertical_loader.py` - Discovery and loading
- `victor/verticals/base.py` - VerticalBase and VerticalConfig
- `victor/agent/orchestrator.py` - Uses vertical extensions

## References

- Related: [ADR-0001: Protocol-First Architecture](0001-protocol-first-architecture.md)
- Implementation: victor/verticals/ directory
- CLAUDE.md: Search for "Vertical System" section
