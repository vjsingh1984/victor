# Extension Manifest Specification

## Overview

The Extension Manifest system provides a structured contract between verticals and the Victor framework. It enables capability negotiation during vertical activation, ensuring compatibility and graceful degradation.

## Components

### ExtensionManifest (`victor-sdk`)

A dataclass declaring what a vertical provides and requires:

```python
from victor_sdk import ExtensionManifest, ExtensionType

manifest = ExtensionManifest(
    api_version=2,
    name="coding",
    version="1.0.0",
    provides={ExtensionType.SAFETY, ExtensionType.TOOLS, ExtensionType.MIDDLEWARE},
    requires={ExtensionType.CAPABILITIES},
)
```

### ExtensionType Enum

| Type | Value | Description |
|------|-------|-------------|
| SAFETY | `safety` | Safety extension patterns |
| TOOLS | `tool_dependencies` | Tool dependency graphs |
| WORKFLOWS | `workflows` | Workflow definitions |
| TEAMS | `teams` | Team specifications |
| MIDDLEWARE | `middleware` | Middleware chain |
| MODE_CONFIG | `mode_config` | Mode configurations |
| RL_CONFIG | `rl_config` | RL learning config |
| ENRICHMENT | `enrichment` | Prompt enrichment |
| API_ROUTER | `api_router` | API routing |
| CAPABILITIES | `capabilities` | Dynamic capabilities |
| SERVICE_PROVIDER | `service_provider` | DI service registration |

### API Versioning

- `CURRENT_API_VERSION = 2` — current manifest schema version
- `MIN_SUPPORTED_API_VERSION = 1` — oldest supported version
- Bump `CURRENT_API_VERSION` for breaking manifest schema changes

### CapabilityNegotiator (`victor.core`)

Validates manifests during vertical activation:

1. **API version check** — must be within `[MIN_SUPPORTED, CURRENT]`
2. **Required capabilities** — all `requires` must be in framework capabilities
3. **Unknown types** — warns about `provides` types the framework doesn't recognize

```python
from victor.core.verticals.capability_negotiator import CapabilityNegotiator

negotiator = CapabilityNegotiator()
result = negotiator.negotiate(manifest)
# result.compatible: bool
# result.warnings: List[str]
# result.errors: List[str]
# result.degraded_features: Set[ExtensionType]
```

## Framework Capabilities

The framework declares these as supported at runtime (`FRAMEWORK_CAPABILITIES` in `capability_negotiator.py`):
`SAFETY`, `TOOLS`, `WORKFLOWS`, `TEAMS`, `MIDDLEWARE`, `MODE_CONFIG`, `RL_CONFIG`, `ENRICHMENT`, `CAPABILITIES`, `SERVICE_PROVIDER`

Note: `API_ROUTER` is defined in `ExtensionType` but not in `FRAMEWORK_CAPABILITIES`, so verticals providing it will receive a degradation warning.

## Auto-Generated Manifests

`VerticalBase.get_manifest()` automatically builds a manifest by inspecting which protocol hooks the subclass overrides. The detection maps method names to extension types:

| Method Override | Detected ExtensionType |
|----------------|----------------------|
| `get_middleware` | MIDDLEWARE |
| `get_safety_extension` | SAFETY |
| `get_tool_requirements` | TOOLS |
| `get_workflow_spec` | WORKFLOWS |
| `get_team_declarations` | TEAMS |
| `get_mode_config` | MODE_CONFIG |
| `get_rl_config` | RL_CONFIG |
| `get_enrichment_strategy` | ENRICHMENT |
| `get_capability_requirements` | CAPABILITIES |

`TOOLS` is always included (since `get_tools()` is abstract and always implemented). No manual manifest creation needed for basic verticals.

## Integration Point

The `VerticalLoader.load()` method calls `_negotiate_manifest()` after resolving the vertical class but before activation. Incompatible manifests raise `ValueError`. Negotiation failures are gracefully degraded if the manifest/negotiator is unavailable (e.g., during testing).
