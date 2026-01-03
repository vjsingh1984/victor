# Victor Security Vertical - External Plugin Example

This directory contains a complete example of how to create an external Victor vertical that can be installed as a standalone Python package. Third-party developers can use this as a template for creating their own verticals.

## What is a Victor Vertical?

A vertical is a domain-specific configuration for Victor that provides:
- **Tools**: Which tools are available for the domain (read, write, shell, etc.)
- **System Prompt**: Expert persona and domain guidance
- **Stages**: Workflow stages for the domain (e.g., reconnaissance, analysis, reporting)
- **Safety Patterns**: Domain-specific dangerous operation detection
- **Task Hints**: Guidance for different types of tasks in the domain

Victor comes with built-in verticals for Coding, Research, DevOps, Data Analysis, and RAG. This example shows how to create a custom Security vertical as an external package.

## Package Structure

```
examples/external_vertical/
├── pyproject.toml              # Package configuration with entry_points
├── README.md                   # This documentation
└── src/
    └── victor_security/
        ├── __init__.py         # Exports SecurityAssistant
        ├── assistant.py        # SecurityAssistant (VerticalBase subclass)
        ├── safety.py           # Security-specific safety patterns
        └── prompts.py          # Security task hints and prompt contributions
```

## How Plugin Discovery Works

Victor uses Python's entry point mechanism for plugin discovery. When you define an entry point in your `pyproject.toml`:

```toml
[project.entry-points."victor.verticals"]
security = "victor_security:SecurityAssistant"
```

This tells Python's packaging system:
1. The entry point group is `victor.verticals`
2. The entry point name is `security`
3. The target is `SecurityAssistant` from the `victor_security` module

When Victor starts, it scans for packages that define entry points in the `victor.verticals` group and automatically registers them as available verticals.

## Installation

### Development Installation

For development and testing, install the package in editable mode:

```bash
# Navigate to the external_vertical directory
cd examples/external_vertical

# Install in editable mode
pip install -e .

# Verify installation
pip show victor-security
```

### Production Installation

For distribution, you would publish to PyPI and users install with:

```bash
pip install victor-security
```

## Usage

After installation, the Security vertical is automatically available in Victor:

### Command Line

```bash
# Use the security vertical directly
victor --vertical security

# Or with the short flag
victor -v security

# The vertical appears in the list of available verticals
victor --list-verticals
```

### Programmatic Usage

```python
from victor.core.verticals import VerticalRegistry

# Discovery happens automatically on startup, but you can trigger it manually
VerticalRegistry.discover_external_verticals()

# Get the security vertical
security_vertical = VerticalRegistry.get("security")

# Get its configuration
config = security_vertical.get_config()
print(f"Tools: {security_vertical.get_tools()}")
print(f"Stages: {list(security_vertical.get_stages().keys())}")

# Get extensions for framework integration
extensions = security_vertical.get_extensions()

# Create an agent with the security vertical
agent = await security_vertical.create_agent(
    provider="anthropic",
    model="claude-sonnet-4-5-20250514",
)
```

### Direct Import

```python
from victor_security import SecurityAssistant

# Use the vertical directly
system_prompt = SecurityAssistant.get_system_prompt()
tools = SecurityAssistant.get_tools()
```

## Creating Your Own Vertical

To create your own vertical, follow these steps:

### 1. Set Up Package Structure

```
my_vertical_package/
├── pyproject.toml
├── README.md
└── src/
    └── my_vertical/
        ├── __init__.py
        ├── assistant.py
        ├── safety.py        # Optional
        └── prompts.py       # Optional
```

### 2. Implement VerticalBase Subclass

```python
# src/my_vertical/assistant.py
from typing import List
from victor.core.verticals import VerticalBase

class MyAssistant(VerticalBase):
    # Required class attributes
    name = "my_vertical"
    description = "Description of my vertical"
    version = "1.0.0"

    @classmethod
    def get_tools(cls) -> List[str]:
        """Required: Return list of tool names."""
        return ["read", "write", "code_search", "shell"]

    @classmethod
    def get_system_prompt(cls) -> str:
        """Required: Return system prompt with domain expertise."""
        return "You are an expert in..."

    # Optional: Override other methods for enhanced functionality
    # - get_stages()
    # - get_tiered_tool_config()
    # - get_safety_extension()
    # - get_prompt_contributor()
    # - get_provider_hints()
    # - get_evaluation_criteria()
```

### 3. Configure Entry Point

```toml
# pyproject.toml
[project.entry-points."victor.verticals"]
my_vertical = "my_vertical:MyAssistant"
```

### 4. Export from Package

```python
# src/my_vertical/__init__.py
from my_vertical.assistant import MyAssistant

__all__ = ["MyAssistant"]
```

## Key Extension Points

### SafetyExtensionProtocol

Implement this to add domain-specific dangerous operation detection:

```python
from victor.core.verticals.protocols import SafetyExtensionProtocol, SafetyPattern

class MySafetyExtension(SafetyExtensionProtocol):
    def get_bash_patterns(self) -> List[SafetyPattern]:
        return [
            SafetyPattern(
                pattern=r"dangerous_command",
                description="Why this is dangerous",
                risk_level="HIGH",
                category="my_category",
            ),
        ]
```

### PromptContributorProtocol

Implement this to add task hints and prompt sections:

```python
from victor.core.verticals.protocols import PromptContributorProtocol
from victor.core.vertical_types import TaskTypeHint

class MyPromptContributor(PromptContributorProtocol):
    def get_task_type_hints(self) -> Dict[str, TaskTypeHint]:
        return {
            "my_task": TaskTypeHint(
                task_type="my_task",
                hint="[MY TASK] Do this, then that...",
                tool_budget=10,
                priority_tools=["read", "code_search"],
            ),
        }
```

### TieredToolConfig

Configure tool tiers for intelligent selection:

```python
from victor.core.verticals.protocols import TieredToolConfig

@classmethod
def get_tiered_tool_config(cls) -> TieredToolConfig:
    return TieredToolConfig(
        mandatory={"read", "ls", "code_search"},      # Always available
        vertical_core={"my_tool", "shell"},           # Core for this vertical
        readonly_only_for_analysis=True,       # Hide write tools during analysis
    )
```

## Testing Your Vertical

```python
# tests/test_vertical.py
import pytest
from my_vertical import MyAssistant

def test_vertical_configuration():
    """Test that vertical is properly configured."""
    assert MyAssistant.name == "my_vertical"
    assert len(MyAssistant.get_tools()) > 0
    assert len(MyAssistant.get_system_prompt()) > 0

def test_vertical_registration():
    """Test that vertical can be discovered."""
    from victor.core.verticals import VerticalRegistry

    # Clear and rediscover
    VerticalRegistry.clear()
    VerticalRegistry.discover_external_verticals()

    vertical = VerticalRegistry.get("my_vertical")
    assert vertical is not None
    assert vertical.name == "my_vertical"
```

## Best Practices

1. **Name Uniquely**: Choose a vertical name that won't conflict with built-in or other external verticals.

2. **Version Appropriately**: Use semantic versioning and update when you make breaking changes.

3. **Document Well**: Include clear documentation of what your vertical does and how to use it.

4. **Test Thoroughly**: Test both the vertical implementation and its integration with Victor.

5. **Handle Errors Gracefully**: Your vertical methods should handle errors without crashing Victor.

6. **Stay Compatible**: Test with multiple Victor versions if possible.

7. **Minimize Dependencies**: Only depend on what you truly need to keep installation simple.

## Support

- Victor Documentation: https://victor.dev
- GitHub Issues: https://github.com/victor-ai/victor/issues
- Community Discord: https://discord.gg/victor-ai

## License

This example is provided under the Apache 2.0 License. You may use this as a template for your own verticals under any license you choose.
