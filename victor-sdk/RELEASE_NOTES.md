# Victor SDK v1.0.0 Release Notes

## Release Summary

Victor SDK v1.0.0 is a **major milestone** that achieves complete dependency inversion between external verticals and the Victor framework. This release enables zero-runtime-dependency vertical development while maintaining 100% backward compatibility.

## What's New

### 🎯 Zero Runtime Dependencies

External verticals can now depend **only on victor-sdk** (~1MB) instead of the full victor-ai framework (50+ dependencies).

**Before:**
```toml
dependencies = ["victor-ai>=0.5.6"]  # Heavy dependency
```

**After:**
```toml
dependencies = ["victor-sdk>=1.0.0"]  # Zero runtime dependencies!
```

### 📦 Complete SDK Package

The victor-sdk package provides:

- **Protocol Definitions**: 12+ protocol interfaces for vertical extensibility
- **Core Types**: VerticalConfig, StageDefinition, TieredToolConfig, ToolSet
- **Discovery System**: Automatic protocol discovery via entry points
- **Exception Classes**: Structured error handling
- **Documentation**: Comprehensive guides and examples

### 🏗️ Three-Tier Architecture

```
External Verticals → victor-sdk (protocols only)
                    ↓
                victor-ai (implements SDK protocols)
```

### ✅ 100% Backward Compatible

All existing victor-ai verticals continue to work without any changes:

```python
# Still works!
from victor.core.verticals.base import VerticalBase

class MyVertical(VerticalBase):
    name = "my-vertical"
    # ... everything as before
```

## Key Features

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

### Discovery System

Enhanced entry point groups for protocol discovery:

```toml
# Vertical registration
[project.entry-points."victor.verticals"]
my-vertical = "my_vertical:MyVertical"

# Protocol implementations
[project.entry-points."victor.sdk.protocols"]
my-tools = "my_vertical.protocols:MyToolProvider"

# Capability providers
[project.entry-points."victor.sdk.capabilities"]
my-search = "my_vertical.capabilities:MySearchCapability"

# Validators
[project.entry-points."victor.sdk.validators"]
file-path = "my_vertical.validators:validate_file_path"
```

### Core Types

- **VerticalConfig**: Complete vertical configuration
- **StageDefinition**: Workflow stage configuration
- **TieredToolConfig**: Progressive tool tiers (BASIC, STANDARD, ADVANCED)
- **ToolSet**: Tool collection with metadata
- **Tier**: Capability tier enumeration

## Installation

### SDK Only (Development)

```bash
pip install victor-sdk
```

### With Runtime (Production)

```bash
pip install victor-ai
```

victor-ai v0.6.0+ automatically depends on victor-sdk.

## Migration Guide

### For New Verticals

Use SDK from the start:

```python
from victor_sdk.verticals.protocols.base import VerticalBase

class MyVertical(VerticalBase):
    @classmethod
    def get_name(cls) -> str:
        return "my-vertical"

    @classmethod
    def get_tools(cls) -> list[str]:
        return ["read", "write", "search"]
```

### For Existing Verticals

Two options:

**Option A: Keep victor-ai dependency (easiest)**
- No changes needed!
- Update to victor-ai v0.6.0+
- Everything works as before

**Option B: Migrate to SDK-only**
- Update imports to use victor_sdk
- Add abstract method implementations
- See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)

## Documentation

- **[SDK Guide](SDK_GUIDE.md)** - Complete usage guide
- **[Vertical Development Guide](VERTICAL_DEVELOPMENT.md)** - How to develop verticals
- **[Migration Guide](MIGRATION_GUIDE.md)** - How to migrate existing verticals
- **[Implementation Summary](IMPLEMENTATION_SUMMARY.md)** - Architecture details
- **[Examples](examples/)** - Complete working examples

## Testing

### Test Results

- **Unit Tests**: 51/51 passing
- **Integration Tests**: 11/11 passing
- **E2E Tests**: 9/9 passing (zero-dependency vertical verified)
- **Total**: 71/71 tests passing

### Running Tests

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests (requires victor-ai)
pytest tests/integration/ -v

# E2E tests (zero-dependency)
python tests/e2e/test_zero_dependency_vertical.py
```

## Breaking Changes

### None!

This release maintains **100% backward compatibility**. All existing victor-ai verticals work without changes.

## Deprecations

None in this release.

## Dependencies

### victor-sdk

- **Runtime**: `typing-extensions>=4.9` (only dependency!)
- **Dev**: pytest, pytest-asyncio, mypy, black, ruff

### victor-ai (Updated)

- Now depends on: `victor-sdk>=1.0.0`
- All other dependencies unchanged

## Performance

### Installation Size

| Package | Size | Dependencies |
|---------|------|-------------|
| victor-sdk | ~1MB | 1 |
| victor-ai | ~50MB | 50+ |

### Load Time

- victor-sdk: <100ms (only typing-extensions)
- victor-ai: ~2s (with all dependencies)

## Contributors

- Vijaykumar Singh - Lead Architect & Implementation

## Acknowledgments

This release is the result of the **Victor SDK: Vertical-Framework Decoupling Architecture** project, implementing complete dependency inversion through a three-tier architecture.

## Roadmap

### v1.1.0 (Planned)

- Additional protocol interfaces
- Enhanced capability providers
- Performance optimizations
- More examples and templates

### v2.0.0 (Future)

- Advanced composition APIs
- Dynamic protocol registration
- Plugin system extensions

## Support

- **Documentation**: https://docs.victor.dev/sdk
- **Issues**: https://github.com/vjsingh1984/victor/issues
- **Discussions**: https://github.com/vjsingh1984/victor/discussions

## License

Apache-2.0

---

**Thank you to all contributors and users of Victor!**

This release represents a significant step forward in modular, extensible AI agent development.
