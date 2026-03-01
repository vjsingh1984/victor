# Victor Contrib Packages

Victor Contrib provides shared optional packages for verticals. These packages are not part of the core framework but provide ready-made functionality that verticals can consume to avoid code duplication.

## Architecture

```
Core (victor/):         Essential framework (Agent, StateGraph, Tools, Events)
Contrib (victor/contrib/):  Shared optional packages for verticals
Verticals (victor-xyz/):     Domain-specific implementations
```

## Packages

### 1. Safety (`victor.contrib.safety`)

Enhanced safety extensions for all verticals using the Template Method pattern.

**Classes:**
- `BaseSafetyExtension`: Base class for vertical-specific safety rules
- `SafetyContext`: Tracks safety operations and statistics
- `VerticalSafetyMixin`: Helper methods for creating common safety patterns

**Usage:**
```python
from victor.contrib.safety import BaseSafetyExtension, VerticalSafetyMixin

class MySafetyExtension(BaseSafetyExtension, VerticalSafetyMixin):
    def get_vertical_name(self) -> str:
        return "myvertical"

    def get_vertical_rules(self) -> List[SafetyRule]:
        return [
            self.create_dangerous_command_rule(
                "my_rule", r"dangerous-cmd", "Dangerous operation"
            ),
        ]
```

**Helper Methods:**
- `create_dangerous_command_rule()`: Shell command safety
- `create_file_deletion_rule()`: File deletion safety
- `create_blocked_operation_rule()`: Blocked operations
- `create_git_force_push_rule()`: Git force push safety
- `create_docker_container_deletion_rule()`: Docker safety
- `create_system_write_rule()`: System directory protection

### 2. Conversation (`victor.contrib.conversation`)

Enhanced conversation management for verticals.

**Classes:**
- `BaseConversationManager`: Base class for vertical-specific conversation management
- `VerticalConversationContext`: Tracks vertical-specific conversation context

**Usage:**
```python
from victor.contrib.conversation import BaseConversationManager

class MyConversationManager(BaseConversationManager):
    def get_vertical_name(self) -> str:
        return "myvertical"

    def get_system_prompt(self) -> str:
        return "You are a specialized assistant for myvertical."
```

### 3. Mode Config (`victor.contrib.mode_config`)

Mode configuration provider base classes building on `ModeConfigRegistry`.

**Classes:**
- `BaseModeConfigProvider`: Base class for vertical mode configuration
- `ModeHelperMixin`: Helper methods for creating common modes

**Usage:**
```python
from victor.contrib.mode_config import BaseModeConfigProvider, ModeHelperMixin

class MyModeConfig(BaseModeConfigProvider, ModeHelperMixin):
    def get_vertical_name(self) -> str:
        return "myvertical"

    def get_vertical_modes(self) -> Dict[str, ModeDefinition]:
        return {
            **self.create_quick_modes(),
            **self.create_standard_modes(),
            **self.create_domain_specific_modes(),
        }
```

**Helper Methods:**
- `create_quick_mode()`: Quick operation mode
- `create_standard_mode()`: Balanced mode
- `create_thorough_mode()`: Deep analysis mode
- `create_exploration_mode()`: Exploration mode
- `create_custom_mode()`: Custom mode configuration

### 4. Workflows (`victor.contrib.workflows`)

YAML workflow escape hatch base classes.

**Classes:**
- `BaseWorkflowProvider`: Base class for YAML workflow discovery and loading
- `WorkflowLoaderMixin`: Utilities for validating and creating workflows

**Usage:**
```python
from victor.contrib.workflows import BaseWorkflowProvider

class MyWorkflowProvider(BaseWorkflowProvider):
    def get_vertical_name(self) -> str:
        return "myvertical"

    def get_workflow_directories(self) -> List[str]:
        return [
            "/usr/local/lib/victor-workflows/common",
            "~/.victor/workflows/myvertical",
        ]
```

### 5. Testing (`victor.contrib.testing`)

Common testing utilities for verticals.

**Classes:**
- `VerticalTestCase`: Base test case with common test utilities
- `MockProviderMixin`: Mock provider fixtures
- `TestAssistantMixin`: Test assistant creation utilities
- `MockToolMixin`: Mock tool fixtures

**Usage:**
```python
from victor.contrib.testing import VerticalTestCase, MockProviderMixin

class TestMyVertical(VerticalTestCase, MockProviderMixin):
    vertical_name = "myvertical"

    def test_with_mock_provider(self):
        provider = self.create_mock_provider("Test response")
        # Use provider in tests
```

## Design Patterns

### Template Method Pattern
All base classes use the Template Method pattern:
1. Base class provides common infrastructure
2. Verticals override abstract methods for domain-specific customization
3. Verticals can optionally override template methods for extended customization

### Mixin Pattern
Utility mixins provide helper methods for common operations:
- `VerticalSafetyMixin`: Safety rule creation
- `ModeHelperMixin`: Mode configuration
- `WorkflowLoaderMixin`: Workflow utilities
- `MockProviderMixin`: Test fixtures

## Benefits

1. **Code Reuse**: Eliminates ~2,300 lines of duplicated code across verticals
2. **Consistency**: All verticals use the same patterns for common operations
3. **Maintainability**: Bug fixes and improvements benefit all verticals
4. **Extensibility**: Verticals can customize behavior through overrides

## Migration

Verticals can migrate incrementally:

1. **Safety**: Replace duplicated safety code with `BaseSafetyExtension`
2. **Conversation**: Use `BaseConversationManager` for conversation management
3. **Mode Config**: Use `BaseModeConfigProvider` for mode configuration
4. **Workflows**: Use `BaseWorkflowProvider` for YAML workflow support
5. **Testing**: Use `VerticalTestCase` for test infrastructure

## Testing

All contrib packages have comprehensive tests:

```bash
pytest tests/unit/contrib/ -v
```

Current test coverage: 17 tests, all passing
