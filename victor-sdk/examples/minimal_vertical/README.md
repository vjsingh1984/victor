# Victor Minimal Vertical

An example vertical that demonstrates **zero runtime dependencies** using only `victor-sdk`.

## Features

- ✅ **Zero Runtime Dependencies**: Only depends on `victor-sdk` (~1MB)
- ✅ **Protocol-Based**: Implements SDK protocols for extensibility
- ✅ **Entry Points**: Registers protocols, capabilities, and validators
- ✅ **Production Ready**: Can be installed and used independently

## Installation

### SDK Only (Zero Dependencies)

```bash
pip install victor-minimal-vertical
```

This installs the vertical definition with **no runtime dependencies**.

### With Runtime (To Actually Use)

```bash
pip install victor-minimal-vertical[runtime]
```

This also installs `victor-ai` so you can actually run the vertical.

## Usage

### As a SDK-Only Vertical

```python
from victor_minimal_vertical import MinimalVertical

# Use the vertical (requires victor-ai runtime)
vertical = MinimalVertical()

# Get configuration
config = vertical.get_config()

# Get tools
tools = vertical.get_tools()
print(f"Available tools: {tools}")
```

### Protocol Discovery

```python
from victor_sdk.discovery import get_global_registry

registry = get_global_registry()

# Get the vertical
vertical = registry.get_vertical("minimal")

# Get protocol implementations
tool_provider = registry.get_capability_provider("minimal-tools")
safety_provider = registry.get_capability_provider("minimal-safety")

# Get capabilities
search_capability = registry.get_capability_provider("minimal-search")
validation_capability = registry.get_capability_provider("minimal-validation")

# Get validators
validators = registry.get_validators()
```

## Entry Points

This vertical demonstrates the current entry point groups:

### victor.plugins (Standard)
```
minimal = "victor_minimal_vertical:plugin"
```

### victor.sdk.protocols (NEW)
```
minimal-tools = "victor_minimal_vertical.protocols:MinimalToolProvider"
minimal-safety = "victor_minimal_vertical.protocols:MinimalSafetyProvider"
```

### victor.sdk.capabilities (NEW)
```
minimal-search = "victor_minimal_vertical.capabilities:MinimalSearchCapability"
minimal-validation = "victor_minimal_vertical.capabilities:MinimalValidationCapability"
```

### victor.sdk.validators (NEW)
```
file-path = "victor_minimal_vertical.validators:validate_file_path"
code-content = "victor_minimal_vertical.validators:validate_code_content"
```

## Architecture

```
victor-minimal-vertical (depends on victor-sdk only)
        ↓
    victor-sdk (protocols only)
        ↓
    victor-ai (implements SDK protocols) [optional runtime]
```

## Benefits

1. **Fast Installation**: No need to install 50+ dependencies just to define a vertical
2. **Clear Contracts**: SDK provides explicit protocol definitions
3. **Easy Testing**: Test verticals without full framework runtime
4. **Modular Development**: Verticals can be developed independently

## Development

To use this vertical in development with the victor-ai runtime:

```bash
# Install in development mode
pip install -e .

# Or with runtime dependencies
pip install -e ".[runtime]"
```

## License

Apache-2.0
