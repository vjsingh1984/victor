# Victor SDK

Protocol definitions and abstractions for building Victor verticals without runtime dependencies.

## Overview

Victor SDK provides pure protocol/ABC definitions that external verticals can depend on without pulling in the entire Victor framework. This enables:

- **Zero-runtime-dependency verticals** - Define verticals with only `victor-sdk` as a dependency
- **Clean separation of concerns** - Protocols in SDK, implementations in `victor-ai`
- **Dependency Inversion** - Verticals depend on abstractions, not concrete implementations
- **Fast development** - No need to install the full framework to define a vertical

### Architecture

```
External Verticals → victor-sdk (protocols only)
                    ↓
                victor-ai (implements SDK protocols)
```

## Installation

### For Vertical Development (SDK Only)

```bash
pip install victor-sdk
```

This gives you access to all protocols and types with **ZERO runtime dependencies**.

### For Using Verticals (With Runtime)

```bash
pip install victor-ai
```

The victor-ai package depends on victor-sdk and includes all runtime implementations.

## Quick Start

### Creating a Minimal Vertical

```python
from victor_sdk.verticals.protocols.base import VerticalBase

class MyVertical(VerticalBase):
    """A zero-dependency vertical."""

    @classmethod
    def get_name(cls) -> str:
        return "my-vertical"

    @classmethod
    def get_description(cls) -> str:
        return "My custom vertical"

    @classmethod
    def get_tools(cls) -> list[str]:
        return ["read", "write", "search"]

    @classmethod
    def get_system_prompt(cls) -> str:
        return "You are a helpful assistant for my vertical."
```

### Package Structure

```
my-vertical/
├── pyproject.toml          # Only depends on victor-sdk
├── my_vertical/
│   └── __init__.py        # Vertical definition
└── README.md
```

### pyproject.toml

```toml
[project]
name = "my-vertical"
version = "1.0.0"
description = "My custom vertical"
dependencies = [
    "victor-sdk>=1.0.0",  # Only SDK - no runtime dependencies!
]

[project.entry-points."victor.verticals"]
my-vertical = "my_vertical:MyVertical"
```

## Documentation

- **[SDK Guide](SDK_GUIDE.md)** - Complete guide for using victor-sdk
- **[Vertical Development Guide](VERTICAL_DEVELOPMENT.md)** - How to develop verticals
- **[Migration Guide](MIGRATION_GUIDE.md)** - How to migrate existing verticals
- **[Implementation Summary](IMPLEMENTATION_SUMMARY.md)** - Architecture and design

## Features

### Protocol System

Victor SDK provides protocol definitions for:

- **ToolProvider**: Provide tool configurations
- **SafetyProvider**: Validate tool calls and prompts
- **PromptProvider**: Customize prompts
- **WorkflowProvider**: Define multi-stage workflows
- **TeamProvider**: Configure multi-agent teams
- **MiddlewareProvider**: Add execution middleware
- **ModeConfigProvider**: Define operation modes
- **RLProvider**: Configure reinforcement learning
- **EnrichmentProvider**: Add context enrichment

### Core Types

- **VerticalConfig**: Complete vertical configuration
- **StageDefinition**: Workflow stage configuration
- **TieredToolConfig**: Progressive tool tiers
- **ToolSet**: Tool collection with metadata
- **Tier**: Capability tier (BASIC, STANDARD, ADVANCED)

### Discovery System

Automatic discovery of verticals and protocols via entry points:

```python
from victor_sdk.discovery import get_global_registry, get_discovery_summary

registry = get_global_registry()

# Discover verticals
verticals = registry.get_verticals()

# Get statistics
print(get_discovery_summary())
```

## Entry Points

Victor SDK supports multiple entry point groups:

```toml
# Vertical registration (standard)
[project.entry-points."victor.verticals"]
my-vertical = "my_vertical:MyVertical"

# Protocol implementations (NEW - Phase 4)
[project.entry-points."victor.sdk.protocols"]
my-tools = "my_vertical.protocols:MyToolProvider"
my-safety = "my_vertical.protocols:MySafetyProvider"

# Capability providers (NEW - Phase 4)
[project.entry-points."victor.sdk.capabilities"]
my-search = "my_vertical.capabilities:MySearchCapability"

# Validators (NEW - Phase 4)
[project.entry-points."victor.sdk.validators"]
file-path = "my_vertical.validators:validate_file_path"
```

## Examples

See the `examples/` directory for complete examples:

- **[minimal_vertical](examples/minimal_vertical/)** - Minimal SDK-only vertical
- **[protocol_implementations](examples/minimal_vertical/protocols.py)** - Protocol examples
- **[capability_providers](examples/minimal_vertical/capabilities.py)** - Capability examples
- **[validators](examples/minimal_vertical/validators.py)** - Validator examples

## Testing

### Unit Tests

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
make test
# or
pytest tests/ -v
```

### Integration Tests

```bash
# Install with runtime
pip install -e ".[dev]" "victor-ai>=0.6.0"

# Run integration tests
pytest tests/integration/ -v
```

## Versioning

Victor SDK follows semantic versioning:

- **1.0.0a1**: Alpha release - Phase 1-3 complete (SDK + victor-ai integration)
- **1.0.0**: Stable release - All phases complete

### Current Status

✅ Phase 1: victor-sdk Package (Complete)
✅ Phase 2: victor-ai Integration (Complete)
✅ Phase 3: Backward Compatibility (Complete)
✅ Phase 4: Enhanced Entry Points (Complete)
🔄 Phase 5-6: External Vertical Migration (In Progress)
⏳ Phase 7-10: Testing, Documentation, Release (Pending)

## Contributing

Contributions are welcome! Please see our contributing guidelines.

## License

Apache-2.0

## Links

- **GitHub**: https://github.com/vjsingh1984/victor
- **Documentation**: https://docs.victor.dev/sdk
- **Issues**: https://github.com/vjsingh1984/victor/issues
