# SDK Boundary Architecture

This document describes the boundary between `victor-sdk` (the thin contract layer) and `victor-ai` (the framework runtime). External verticals depend only on the SDK.

## Overview

```
External Vertical (victor-coding, etc.)
    │
    ▼
victor-sdk  ← Zero dependencies on victor-ai
    │           Provides: VerticalBase, PluginContext, VictorPlugin,
    │                     ExtensionManifest, ToolProvider, MockPluginContext
    ▼
victor-ai   ← Framework runtime
                Provides: AgentOrchestrator, ProviderRegistry, ToolExecutor,
                          CapabilityNegotiator, entry point loading
```

## SDK Contract (`victor-sdk/`)

The SDK has **zero dependencies** on `victor-ai`. Its only runtime dependency is `typing-extensions>=4.9`.

### Key Protocols

| Protocol | File | Purpose |
|----------|------|---------|
| `VerticalBase` | `victor_sdk/verticals/protocols/base.py` | Abstract base for all verticals |
| `PluginContext` | `victor_sdk/core/plugins.py` | DI interface for plugin registration |
| `VictorPlugin` | `victor_sdk/core/plugins.py` | Plugin lifecycle protocol |
| `ToolProvider` | `victor_sdk/verticals/protocols/tools.py` | Tool registration protocol |
| `ExtensionManifest` | `victor_sdk/verticals/manifest.py` | Capability declaration |

### Testing Utilities

| Utility | File | Purpose |
|---------|------|---------|
| `MockPluginContext` | `victor_sdk/testing/fixtures.py` | In-memory PluginContext for testing without victor-ai |
| `assert_valid_vertical()` | `victor_sdk/testing/__init__.py` | Validate vertical implements SDK contract |
| `assert_import_boundaries()` | `victor_sdk/testing/__init__.py` | Check vertical doesn't import victor internals |
| `validate_manifest()` | `victor_sdk/verticals/validation.py` | Static manifest completeness check |
| `audit_vertical_dependencies()` | `victor_sdk/verticals/validation.py` | Compare imports vs declared deps |

## Entry Points

Verticals register via `victor.plugins` entry point group in `pyproject.toml`:

```toml
[project.entry-points."victor.plugins"]
my_vertical = "my_package:plugin"
```

The framework discovers plugins via `victor/framework/entry_point_loader.py` using `importlib.metadata` (NOT `sys.path` scanning). Additional entry point groups:

- `victor.plugins` — VictorPlugin implementations
- `victor.skills` — Skill definitions
- `victor.safety_rules`, `victor.tool_dependencies`, `victor.rl_configs`, `victor.escape_hatches`, `victor.commands`, `victor.prompt_contributors`, `victor.mode_configs`, `victor.workflow_providers`, `victor.team_spec_providers`, `victor.capability_providers`, `victor.service_providers`

## Manifest Validation Lifecycle

```
1. Definition    @register_vertical(name="my-vert", version="1.0.0")
                 → Attaches ExtensionManifest to class._victor_manifest

2. Discovery     VerticalRegistry.discover_external_verticals()
                 → Scans victor.plugins entry point group

3. Negotiation   CapabilityNegotiator.negotiate(manifest)
                 → Validates: API version, required extensions, extension_dependencies
                 → File: victor/core/verticals/capability_negotiator.py

4. Activation    VictorPlugin.register(context: PluginContext)
                 → Plugin registers tools, verticals, commands via context

5. Runtime       Orchestrator resolves tools via registered vertical providers
```

### Extension Dependency Validation

Manifests can declare dependencies on other extensions:

```python
ExtensionManifest(
    name="my-vert",
    extension_dependencies=[
        ExtensionDependency(extension_name="chromadb", min_version=">=0.4"),
        ExtensionDependency(extension_name="victor-coding", optional=True),
    ],
)
```

`CapabilityNegotiator._validate_extension_dependencies()` checks:
- Required deps are installed (hard error if missing)
- Version constraints are satisfied
- Optional deps produce warnings only

## Import Boundaries

Layer rule enforced by `scripts/check_imports.py`:

```
config/ ← providers/ ← tools/ ← agent/ ← ui/
```

Verticals must import from `victor_sdk` only (not `victor.core`, `victor.agent`, etc.). The contrib directory (`victor/verticals/contrib/`) is a deprecated tombstone with regression guard tests at `tests/unit/sdk/test_contrib_import_boundaries.py`.

SDK boundary stability is enforced by contract shape tests at `tests/unit/sdk/test_sdk_contract_shapes.py`.

## External Vertical Development

1. Depend on `victor-sdk>=X.Y` (not `victor-ai`)
2. Use `@register_vertical()` decorator or set `_victor_manifest`
3. Implement `VictorPlugin.register(context)` in your entry point
4. Declare `extension_dependencies` in manifest for third-party deps
5. Test with `MockPluginContext` from `victor_sdk.testing`
6. Run `validate_manifest(MyVertical)` and `audit_vertical_dependencies(src_dir, manifest)` in CI
