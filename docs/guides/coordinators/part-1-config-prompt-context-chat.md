# Coordinator Usage Guide - Part 1

**Part 1 of 2:** Config, Prompt, Context, Chat, and Tool Coordinators

---

## Navigation

- **[Part 1: Core Coordinators](#)** (Current)
- [Part 2: Provider, Session, Additional Coordinators](part-2-provider-session-additional.md)
- [**Complete Guide**](../coordinators.md)

---

**Version**: 0.5.0
**Last Updated**: January 31, 2026
**Audience**: Developers, Contributors
**Purpose**: How to use Victor AI coordinators

This guide covers the 20 specialized coordinators in Victor AI's two-layer architecture.

---

## Coordinators

Coordinators are specialized components that encapsulate complex operations. Each coordinator has a single, well-defined responsibility.

### ConfigCoordinator

**Purpose**: Configuration loading, validation, and management

**File Location**: `/victor/agent/coordinators/config_coordinator.py`

**Dependencies**: Settings

**Key Methods**:

```python
class ConfigCoordinator:
    """Configuration management coordinator."""

    def get_config(
        self,
    ) -> OrchestratorConfig:
        """
        Get current configuration.

        Returns:
            OrchestratorConfig with all settings

        Example:
            >>> config = config_coord.get_config()
            >>> print(config.model)
            'gpt-4'
        """
```

### PromptCoordinator

**Purpose**: Build prompts from multiple contributors

**File Location**: `/victor/agent/coordinators/prompt_coordinator.py`

**Dependencies**: ConfigCoordinator, ModeCoordinator

### ContextCoordinator

**Purpose**: Manage conversation context and lifecycle

**File Location**: `/victor/agent/coordinators/context_coordinator.py`

**Dependencies**: ConfigCoordinator

### ChatCoordinator

**Purpose**: Orchestrate chat interactions with LLMs

**File Location**: `/victor/agent/coordinators/chat_coordinator.py`

**Dependencies**: PromptCoordinator, ContextCoordinator, ProviderCoordinator

### ToolCoordinator

**Purpose**: Tool selection and execution

**File Location**: `/victor/agent/coordinators/tool_coordinator.py`

**Dependencies**: ContextCoordinator

[Content continues through Tool Coordinator...]

---

**Continue to [Part 2: Provider, Session, Additional Coordinators](part-2-provider-session-additional.md)**
