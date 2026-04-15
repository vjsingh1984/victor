# Victor SDK Migration Guide

## Migrating from `victor.framework.*` to `victor_sdk`

External verticals should import types from `victor_sdk` whenever possible, reserving `victor.framework.*` imports for runtime-only features that require `victor-ai` to be installed.

### Types Available in SDK (import from `victor_sdk`)

#### Core Protocols
```python
from victor_sdk import VerticalBase, VictorPlugin, PluginContext
from victor_sdk import ExtensionManifest, ExtensionType
from victor_sdk import SkillDefinition, SkillProvider
```

#### Vertical Protocols (26 protocol definitions)
```python
from victor_sdk.verticals.protocols import (
    ToolProvider, SafetyProvider, PromptProvider,
    WorkflowProvider, TeamProvider, MiddlewareProvider,
    ModeConfigProvider, RLProvider, EnrichmentProvider,
    ServiceProvider, HandlerProvider, CapabilityProvider,
)
```

#### Team & Multi-Agent (NEW in v0.7.0)
```python
from victor_sdk import TeamFormation, TeamMemberSpec
# Previously: from victor.framework.teams import TeamFormation, TeamMemberSpec
```

#### Safety (NEW in v0.7.0)
```python
from victor_sdk import SafetyLevel
from victor_sdk.safety import SafetyAction, SafetyCategory, SafetyRule, SafetyCoordinator
# Previously: from victor.framework.config import SafetyLevel
```

#### RL / Learning (NEW in v0.7.0)
```python
from victor_sdk import RLOutcome, RLRecommendation, LearnerType, BaseRLConfig
# Previously: from victor.framework.rl.base import RLOutcome, RLRecommendation
```

#### Capabilities
```python
from victor_sdk import (
    BaseCapabilityProvider, CapabilityEntry, CapabilityMetadata,
    CapabilityType, OrchestratorCapability,
)
```

#### Workflows
```python
from victor_sdk import BaseYAMLWorkflowProvider
```

#### Tools
```python
from victor_sdk import ToolNames
from victor_sdk.verticals.protocols import ToolFactory, ToolPluginHelper
```

### Types That Stay in `victor.framework.*`

These require the full `victor-ai` runtime and cannot be promoted to the zero-dependency SDK:

| Type | Module | Reason |
|------|--------|--------|
| `Agent` | `victor.framework.agent` | Core framework entry point with full DI |
| `CapabilityLoader` | `victor.framework.capability_loader` | Runtime service with container access |
| `SafetyEnforcer` | `victor.framework.config` | Runtime class with coordinator logic |
| `BaseLearner` | `victor.framework.rl.base` | ABC with framework method contracts |
| `get_team_registry` | `victor.framework.team_registry` | Global singleton accessor |
| `get_rl_coordinator` | `victor.framework.rl.coordinator` | Runtime RL system |

**Best practice**: Import these inside method bodies (lazy), not at module level:
```python
class MyVertical(VerticalBase):
    def get_extensions(self):
        # Lazy import — only when victor-ai is installed
        from victor.framework.config import SafetyEnforcer
        return VerticalExtensions(safety=SafetyEnforcer(...))
```

### Dependency Configuration

```toml
# pyproject.toml
[project]
dependencies = [
    "victor-sdk>=0.7.0,<1.0",  # Base dependency (zero framework deps)
]

[project.optional-dependencies]
runtime = [
    "victor-ai>=0.7.0,<1.0",   # Full framework (optional)
]
```

### Testing Without Framework

```python
# test_sdk_boundary.py
def test_vertical_imports_sdk_only():
    """Verify vertical definition works without victor-ai."""
    from victor_sdk.testing import MockPluginContext, assert_valid_vertical
    from my_vertical import MyVertical

    assert_valid_vertical(MyVertical)
```
