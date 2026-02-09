# Extension System Documentation

**Version**: 0.5.1
**Last Updated**: 2025-01-27

## Overview

Victor's extension system enables developers to extend the framework with custom verticals, tools, middleware,
  and workflows. The system follows SOLID principles (Open/Closed, Liskov Substitution, Interface Segregation,
  Dependency Inversion) to ensure maintainability and extensibility.

## Table of Contents

- [Extension Types](#extension-types)
- [Built-in Verticals](#built-in-verticals)
- [External Vertical Development](#external-vertical-development)
- [Tool Dependency Providers](#tool-dependency-providers)
- [Middleware Extensions](#middleware-extensions)
- [Prompt Builders](#prompt-builders)
- [Step Handlers](#step-handlers)
- [Registration Process](#registration-process)
- [Examples](#examples)

---

## Extension Types

Victor supports multiple extension mechanisms:

### 1. Verticals
Domain-specific AI assistants with specialized tools and workflows.

**Built-in Verticals:**
- `CodingAssistant` - Software development workflows
- `DevOpsAssistant` - Infrastructure and CI/CD workflows
- `ResearchAssistant` - Literature search and analysis
- `RAGAssistant` - Retrieval-augmented generation
- `DataAnalysisAssistant` - Data analysis and visualization

### 2. Tool Dependency Providers
Define tool execution patterns, sequences, and dependencies for intelligent tool selection.

### 3. Middleware
Request/response processing pipeline for safety, logging, and transformation.

### 4. Prompt Builders
Composable system prompt construction with section-based composition.

### 5. Step Handlers
Vertical integration pipeline handlers (SOLID-compliant, single responsibility).

---

## Built-in Verticals

### Directory Structure

```
victor/
├── coding/
│   ├── __init__.py                 # Vertical entry point
│   ├── config/
│   │   └── vertical.yaml            # Vertical configuration
│   ├── tool_dependencies.py         # Tool dependency provider
│   └── workflows/                   # Workflow definitions
├── devops/
│   └── ...
├── research/
│   └── ...
├── rag/
│   └── ...
└── dataanalysis/
    └── ...
```

### Vertical Configuration

Each vertical defines its configuration in `config/vertical.yaml`:

```yaml
metadata:
  name: coding
  version: 0.5.0
  description: Software development workflows

core:
  tools:
    list: [read, write, edit, shell, git, grep, test, ls, code_search]

  system_prompt:
    source: builder
    builder_factory: create_coding_prompt_builder
    sections:
      identity:
        override: "You are Victor, an expert software development assistant"

  stages:
    INITIAL: { name: INITIAL, ... }
    PLANNING: { name: PLANNING, ... }
    ...

extensions:
  # Use framework middleware profile
  middleware_profile: safety_first
  middleware_overrides:
    - class: victor.coding.middleware.CodeCorrectionMiddleware

  # Tool dependencies (auto-infer vertical name from module path)
  tool_dependencies:
    module: victor.core.tool_dependency_loader
    factory: create_vertical_tool_dependency_provider
```

---

## External Vertical Development

### Quick Start

Create a new external vertical in 3 steps:

#### Step 1: Create Package Structure

```bash
mkdir -p my_security_victim/tool_dependencies
cd my_security_victim
```

#### Step 2: Create Vertical Class

**File**: `my_security_victim/__init__.py`

```python
from victor.core.verticals.base import VerticalBase

class SecurityAnalysisAssistant(VerticalBase):
    """Security analysis vertical for Victor."""

    name: str = "security_analysis"
    version: str = "0.1.0"
    description: str = "Security vulnerability scanning and analysis"

    def get_tools(self) -> List[str]:
        return ["read", "grep", "shell", "web", "security_scan"]
```

#### Step 3: Create Tool Dependencies

**File**: `my_security_victim/tool_dependencies.py`

```python
from victor.core.tool_dependency_loader import create_vertical_tool_dependency_provider

# Auto-infers "security_analysis" from module path
SecurityAnalysisToolDependencyProvider = create_vertical_tool_dependency_provider()
```

### Register via Entry Point

**File**: `pyproject.toml`

```toml
[project.entry-points."victor.verticals"]
security_analysis = "my_security_victim:SecurityAnalysisAssistant"
```

### Advanced: Custom Configuration

Create `config/vertical.yaml` for complex configurations:

```yaml
metadata:
  name: security_analysis
  version: 0.1.0

core:
  tools:
    list: [read, grep, shell, security_scan, cve_check]

  system_prompt:
    source: inline
    text: |
      You are a security analysis assistant specialized in:
      - Vulnerability scanning
      - CVE database lookup
      - Security best practices

extensions:
  tool_dependencies:
    module: victor.core.tool_dependency_loader
    factory: create_vertical_tool_dependency_provider
    # Note: vertical_name auto-inferred from module path
```

---

## Tool Dependency Providers

### Auto-Inference

The `create_vertical_tool_dependency_provider()` factory automatically infers the vertical name from the calling module path:

```python
# Built-in verticals
# In victor/coding/tool_dependencies.py:
CodingToolDependencyProvider = create_vertical_tool_dependency_provider()
# → Infers "coding" from "victor.coding.tool_dependencies"

# External verticals with packages
# In my_company.security_analysis/tool_dependencies.py:
SecurityAnalysisToolDependencyProvider = create_vertical_tool_dependency_provider()
# → Infers "security_analysis" from "my_company.security_analysis.tool_dependencies"

# External verticals (flat structure)
# In security_analysis/tool_dependencies.py:
Provider = create_vertical_tool_dependency_provider()
# → Infers "security_analysis" from "security_analysis.tool_dependencies"
```

### Supported Patterns

| Module Path Pattern | Inferred Vertical | Example |
|-------------------|------------------|---------|
| `victor.<vertical>.tool_dependencies` | `<vertical>` | `victor.coding.tool_dependencies` → `coding` |
| `<company>.<vertical>.tool_dependencies` | `<vertical>` | `my_company.security_analysis.tool_dependencies` → `security_analysis` |
| `<vertical>.tool_dependencies` | `<vertical>` | `custom_security.tool_dependencies` → `custom_security` |

### Explicit Name (Always Works)

```python
# Always specify name explicitly if needed
Provider = create_vertical_tool_dependency_provider("my_custom_vertical")
```

### Tool Dependencies YAML

Create `tool_dependencies.yaml` in your vertical directory:

```yaml
vertical: security_analysis

# Tool dependency graph
dependencies:
  - tool: security_scan
    depends_on: [read]
    enables: [security_report]
    weight: 0.9

  - tool: cve_check
    depends_on: []
    enables: [security_report]
    weight: 0.7

# Transition probabilities
transitions:
  security_scan:
    - tool: security_report
      weight: 0.85
    - tool: cve_check
      weight: 0.15

# Common sequences
sequences:
  vulnerability_scan:
    - read
    - security_scan
    - cve_check
    - security_report

# Required tools
required_tools:
  - read
  - grep

# Optional tools
optional_tools:
  - shell
```

---

## Middleware Extensions

### Middleware Profiles

Framework provides pre-configured middleware profiles:

| Profile | Description | Use Case |
|---------|-------------|----------|
| `safety_first` | Git safety, secret masking | High-security environments |
| `production` | Complete validation, rate limiting | Production deployments |
| `development` | Lenient validation, debug logging | Development environments |
| `analysis` | Read-only operations | Analysis tasks |
| `ci_cd` | CI/CD specific checks | Continuous integration |
| `default` | Basic middleware | General purpose |

### Using Profiles

**In vertical YAML config:**

```yaml
extensions:
  middleware_profile: safety_first
  middleware_overrides:
    - class: my_company.middleware.CustomSecurityMiddleware
      enabled: true
      priority: high
```

### Custom Middleware

Create custom middleware by implementing the protocol:

```python
# my_company/middleware.py
from victor.protocols import MiddlewareProtocol

class CustomSecurityMiddleware(MiddlewareProtocol):
    """Custom security middleware."""

    async def process_request(
        self,
        request: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], bool]:
        """Process request before tool execution."""
        # Add custom security checks
        request["security_scan"] = True
        return request, True  # True = continue pipeline

    async def process_response(
        self,
        response: Any,
        context: Dict[str, Any]
    ) -> Any:
        """Process response after tool execution."""
        # Add security metadata
        return response
```

### Register in Config

```yaml
middleware_overrides:
  - module: my_company.middleware
    class: CustomSecurityMiddleware
    enabled: true
    priority: high
```

---

## Prompt Builders

### Framework Prompt Builders

Framework provides factory functions for each vertical:

- `create_coding_prompt_builder()`
- `create_devops_prompt_builder()`
- `create_research_prompt_builder()`
- `create_data_analysis_prompt_builder()`

### Using Prompt Builder

**In vertical YAML config:**

```yaml
core:
  system_prompt:
    source: builder
    builder_factory: create_coding_prompt_builder
    sections:
      identity:
        override: "You are an expert in {domain} development for {project_name}"
      guidelines:
        file: ./config/custom_guidelines.md
    extra_sections:
      - name: "project_context"
        content: "Working on {project_name}, a {domain} application"
```

### Custom Prompt Sections

Create custom sections in framework:

```python
# victor/framework/prompt_sections.py
CODING_CUSTOM_IDENTITY = """
You are Victor, specialized in {domain} development for {project_name}.

Current project: {project_name}
Tech stack: {tech_stack}
"""
```

### Section Override Priority

1. `override` - Replaces entire section content
2. `file` - Loads section from file
3. Default section from PromptBuilder

---

## Step Handlers

### What Are Step Handlers?

Step handlers implement the Single Responsibility Principle for vertical integration. Each handler handles one specific
  aspect of applying a vertical to the orchestrator.

### Built-in Handlers

| Handler | Order | Responsibility |
|---------|-------|----------------|
| `ToolStepHandler` | 10 | Apply tools filter |
| `PromptStepHandler` | 20 | Apply system prompt |
| `ConfigStepHandler` | 40 | Apply stages, mode config, tool deps |
| `ExtensionsStepHandler` | 45 | Apply extensions (middleware, safety, etc.) |
| `FrameworkStepHandler` | 60 | Apply workflows, RL configs, team specs |
| `ContextStepHandler` | 100 | Attach context |

### Creating Custom Handlers

```python
from victor.framework.step_handlers import BaseStepHandler

class CustomIntegrationHandler(BaseStepHandler):
    """Handler for custom vertical integration."""

    @property
    def name(self) -> str:
        return "custom_integration"

    @property
    def order(self) -> int:
        return 55  # Between Extensions (45) and Framework (60)

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Apply custom integration logic."""
        # Your custom logic here
        result.add_info("Custom integration applied")
```

### Register Handler

```python
from victor.framework.step_handlers import StepHandlerRegistry

registry = StepHandlerRegistry.default()
registry.add_handler(CustomIntegrationHandler())
```

---

## Registration Process

### 1. Entry Points (External Plugins)

**Add to `pyproject.toml`:**

```toml
[project.entry-points."victor.verticals"]
my_vertical = "my_package:MyVerticalAssistant"

[project.entry-points."victor.capabilities"]
my_capability = "my_package.capabilities:MyCapability"
```

### 2. Dynamic Registration

```python
from victor.core.verticals.base import VerticalBase

class CustomVertical(VerticalBase):
    name: str = "custom"

# Register at runtime
CustomVertical.register_extension(
    extension=MyCustomExtension(),
    capability="custom_feature"
)
```

### 3. Extension Registry

```python
from victor.core.verticals.base import ExtensionRegistry

registry = ExtensionRegistry.default()
registry.register_extension(
    extension_type="analytics",
    extension=AnalyticsExtension(name="analytics", api_key="key123")
)
```

---

## Examples

### Example 1: Minimal External Vertical

```python
# security_victim/__init__.py
from victor.core.verticals.base import VerticalBase

class SecurityAssistant(VerticalBase):
    name: str = "security"
    version: str = "0.1.0"
    description: str = "Security analysis"

    def get_tools(self) -> List[str]:
        return ["read", "grep", "shell"]
```

```python
# security_victim/tool_dependencies.py
from victor.core.tool_dependency_loader import create_vertical_tool_dependency_provider

SecurityToolDependencyProvider = create_vertical_tool_dependency_provider()
```

### Example 2: Advanced Vertical with Custom Middleware

```python
# company_x/vertical.py
from victor.core.verticals.base import VerticalBase

class CompanyXVertical(VerticalBase):
    name: str = "company_x"
    version: str = "1.0.0"
    description: str = "Company X specialized workflows"

    def get_tools(self) -> List[str]:
        return ["read", "write", "edit", "company_x_tool"]

    def get_system_prompt(self) -> str:
        return "You are a Company X specialized assistant..."
```

```python
# company_x/middleware.py
from victor.protocols import MiddlewareProtocol

class CompanyXMiddleware(MiddlewareProtocol):
    async def process_request(self, request, context):
        # Add company-specific logic
        request["company_x_context"] = True
        return request, True

    async def process_response(self, response, context):
        return response
```

```yaml
# company_x/config/vertical.yaml
extensions:
  middleware_overrides:
    - module: company_x.middleware
      class: CompanyXMiddleware
```

### Example 3: Custom Step Handler

```python
from victor.framework.step_handlers import BaseStepHandler
from victor.framework.vertical_integration import VerticalContext, IntegrationResult

class AnalyticsHandler(BaseStepHandler):
    """Handler for analytics integration."""

    @property
    def name(self) -> str:
        return "analytics"

    @property
    def order(self) -> int:
        return 105  # After ContextStepHandler

    def _do_apply(self, orchestrator, vertical, context, result):
        """Apply analytics tracking."""
        # Register analytics callback
        if hasattr(orchestrator, "register_analytics_callback"):
            orchestrator.register_analytics_callback(self._track_event)
        result.add_info("Analytics tracking enabled")

    def _track_event(self, event_name, data):
        """Track analytics event."""
        pass
```

---

## Best Practices

### DO ✓

1. **Use auto-inference** - Let the factory infer vertical names when possible
2. **Implement protocols** - Use `Protocol` types for loose coupling
3. **Single Responsibility** - Each extension/handler should have one purpose
4. **Document your extensions** - Provide clear docstrings
5. **Test thoroughly** - Test extensions in isolation and integration

### DON'T ✗

1. **Hardcode vertical names** - Use auto-inference instead
2. **Tight coupling** - Depend on abstractions, not concrete classes
3. **God objects** - Split complex extensions into focused handlers
4. **Skip validation** - Validate tool inputs and configurations
5. **Ignore caching** - Use appropriate caching for performance

---

## Troubleshooting

### Vertical Not Loading

```python
# Check if vertical is registered
from victor.core.verticals.base import VerticalBase
vertical = VerticalBase.get("my_vertical")
if vertical is None:
    print("Vertical not found - check entry point or import path")
```

### Tool Dependencies Not Working

```python
# Check provider loaded correctly
from my_vertical.tool_dependencies import MyToolDependencyProvider
provider = MyToolDependencyProvider
deps = provider.get_dependencies()
print(f"Dependencies: {deps}")
```

### Auto-Inference Failing

```python
# Check module path
import inspect
frame = inspect.currentframe()
module_name = frame.f_back.f_globals.get("__name__", "")
print(f"Calling module: {module_name}")
# Should end with ".tool_dependencies"
```

---

## API Reference

### Key Classes

- `VerticalBase` - Base class for all verticals
- `ToolDependencyProvider` - Tool dependency management
- `MiddlewareProtocol` - Middleware interface
- `PromptBuilder` - Prompt composition
- `StepHandlerRegistry` - Handler registration
- `ExtensionRegistry` - Dynamic extension registration

### Key Functions

- `create_vertical_tool_dependency_provider(vertical=None)` - Factory with auto-inference
- `VerticalBase.get(name)` - Get registered vertical
- `VerticalBase.list_verticals()` - List all verticals
- `StepHandlerRegistry.default()` - Get global registry

---

## Related Documentation

- [Architecture Overview](ARCHITECTURE.md)
- [Best Practices](BEST_PRACTICES.md)
- [Design Patterns](DESIGN_PATTERNS.md)
- [Component Reference](COMPONENT_REFERENCE.md)
- [Protocols Reference](PROTOCOLS_REFERENCE.md)
- [Migration Guides](MIGRATION_GUIDES.md)

---

## Support

For questions or issues:
1. Check the main documentation in `docs/`
2. Review examples in `victor/coding/`, `victor/devops/`, etc.
3. Run tests: `pytest tests/unit/core/verticals/`
4. Check protocol conformance: `make check-protocol`

---

**Last Updated:** February 01, 2026
**Reading Time:** 4 minutes
