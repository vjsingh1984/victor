# Victor Security Vertical - SDK-Only External Example

This example shows the supported external package model for Victor verticals:
author the vertical against `victor-sdk` only, publish it as a normal Python
package, and let `victor-ai` discover it at runtime through the
`victor.verticals` entry point.

## What This Example Demonstrates

- SDK-only package dependency for authoring
- canonical `ToolNames` and `CapabilityIds`
- manifest-first vertical definition via `get_definition()`
- entry-point based runtime discovery

## Package Structure

```text
examples/external_vertical/
├── pyproject.toml
├── README.md
└── src/
    └── victor_security/
        ├── __init__.py
        └── assistant.py
```

## Installation

### SDK-only authoring

```bash
cd examples/external_vertical
pip install -e .
```

This installs the package with only `victor-sdk` as a runtime dependency.

### With Victor runtime

```bash
cd examples/external_vertical
pip install -e ".[runtime]"
```

This also installs `victor-ai`, which can discover and run the vertical.

## Authoring Example

```python
from victor_security import SecurityAssistant

definition = SecurityAssistant.get_definition()

print(definition.name)
print(definition.tools)
print(definition.capability_requirements)
print(definition.workflow_metadata.workflow_spec)
```

The preferred contract is `get_definition()`. `get_config()` remains available
as a compatibility bridge for current runtime integrations.

## Runtime Usage

After installing the runtime extra, Victor can discover the package through the
entry point declared in `pyproject.toml`:

```toml
[project.entry-points."victor.verticals"]
security = "victor_security:SecurityAssistant"
```

Examples:

```bash
victor --vertical security
victor --list-verticals
```

## Key Contract Choices In This Example

- `victor_sdk.VerticalBase` is the only base class used by the package
- tools are declared with `ToolRequirement` and `ToolNames`
- runtime needs are declared with `CapabilityRequirement` and `CapabilityIds`
- prompt templates, task hints, stages, and workflow metadata are all expressed
  through SDK hooks
- no `victor.core` or `victor.framework` imports are required to author the package

## Next Steps

- Compare this example with [victor-sdk/README.md](../../victor-sdk/README.md)
- See the broader SDK guide in [victor-sdk/VERTICAL_DEVELOPMENT.md](../../victor-sdk/VERTICAL_DEVELOPMENT.md)
