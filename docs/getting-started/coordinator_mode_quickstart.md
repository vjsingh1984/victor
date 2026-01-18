# Quick Start: Coordinator Orchestrator Mode

## What You'll Learn

In this guide, you'll learn how to:
- Enable coordinator orchestrator mode
- Understand the differences between legacy and coordinator modes
- Use the new coordinator-based architecture
- Extend functionality with custom coordinators

## Prerequisites

- Victor installed: `pip install victor-ai`
- Basic understanding of Victor architecture
- (Optional) Familiarity with design patterns (facade, coordinator)

## Step 1: Enable Coordinator Mode

**Method 1: Environment Variable (Recommended)**

```bash
# Set environment variable
export VICTOR_USE_COORDINATOR_ORCHESTRATOR=true

# Run Victor
victor chat
```

**Method 2: .env File**

```bash
# Add to .env file
echo "VICTOR_USE_COORDINATOR_ORCHESTRATOR=true" >> .env
```

**Method 3: Settings File**

```yaml
# victor.yaml
use_coordinator_orchestrator: true
```

**Method 4: Python Code**

```python
from victor.config.settings import Settings

settings = Settings(use_coordinator_orchestrator=True)
```

## Step 2: Verify Coordinator Mode

Create `verify_mode.py`:

```python
import asyncio
from victor import Agent
from victor.config.settings import Settings

async def main():
    # Enable coordinator mode
    settings = Settings(use_coordinator_orchestrator=True)

    # Create agent
    agent = await Agent.create(settings=settings)

    # Check orchestrator type
    orchestrator = agent.orchestrator
    print(f"Orchestrator type: {type(orchestrator).__name__}")

    # Check for coordinators
    if hasattr(orchestrator, 'config_coordinator'):
        print("✅ ConfigCoordinator available")
    if hasattr(orchestrator, 'prompt_coordinator'):
        print("✅ PromptCoordinator available")
    if hasattr(orchestrator, 'context_coordinator'):
        print("✅ ContextCoordinator available")
    if hasattr(orchestrator, 'analytics_coordinator'):
        print("✅ AnalyticsCoordinator available")

    await agent.close()

asyncio.run(main())
```

Run it:

```bash
python verify_mode.py
```

Expected output:

```
Orchestrator type: AgentOrchestratorRefactored
✅ ConfigCoordinator available
✅ PromptCoordinator available
✅ ContextCoordinator available
✅ AnalyticsCoordinator available
```

## Step 3: Use Coordinator Mode

Create `coordinator_example.py`:

```python
import asyncio
from victor import Agent

async def main():
    # Enable coordinator mode
    import os
    os.environ["VICTOR_USE_COORDINATOR_ORCHESTRATOR"] = "true"

    # Create agent (uses coordinator orchestrator)
    agent = await Agent.create()

    # Use agent normally - interface is the same!
    result = await agent.run("Say 'Hello from Coordinator Mode!'")
    print(f"Response: {result.content}")

    await agent.close()

asyncio.run(main())
```

## Step 4: Understand the Architecture

**Legacy Mode:**

```
AgentOrchestrator (monolithic)
├── chat()
├── stream_chat()
├── _handle_tool_calls()
├── _process_response()
├── _manage_context()
└── ... 50+ more methods
```

**Coordinator Mode:**

```
AgentOrchestratorRefactored (facade)
├── ConfigCoordinator (configuration management)
├── PromptCoordinator (prompt generation)
├── ContextCoordinator (context compaction)
└── AnalyticsCoordinator (analytics export)
```

## Step 5: Add Custom Coordinators

Create `custom_coordinator.py`:

```python
from victor.agent.protocols import IConfigCoordinator
from typing import Any

class CustomConfigCoordinator(IConfigCoordinator):
    """Custom configuration coordinator."""

    def __init__(self):
        self.custom_configs = {
            "custom_setting": "custom_value",
            "tool_budget": 100,
        }

    def get_config(self, key: str) -> Any:
        """Get configuration value."""
        return self.custom_configs.get(key)

    def set_config(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.custom_configs[key] = value

# Usage
async def use_custom_coordinator():
    import os
    os.environ["VICTOR_USE_COORDINATOR_ORCHESTRATOR"] = "true"

    from victor import Agent

    agent = await Agent.create()

    # Replace config coordinator
    if hasattr(agent.orchestrator, 'config_coordinator'):
        agent.orchestrator.config_coordinator = CustomConfigCoordinator()

        # Use custom config
        value = agent.orchestrator.config_coordinator.get_config("custom_setting")
        print(f"Custom config value: {value}")

    await agent.close()

asyncio.run(use_custom_coordinator())
```

## Step 6: Compare Modes

Create `comparison.py`:

```python
import asyncio
from victor import Agent
from victor.config.settings import Settings

async def test_mode(use_coordinator: bool):
    """Test a specific mode."""
    mode_name = "Coordinator" if use_coordinator else "Legacy"
    print(f"\nTesting {mode_name} Mode...")

    settings = Settings(use_coordinator_orchestrator=use_coordinator)
    agent = await Agent.create(settings=settings)

    print(f"  Orchestrator: {type(agent.orchestrator).__name__}")
    print(f"  Has coordinators: {hasattr(agent.orchestrator, 'config_coordinator')}")

    # Test basic functionality
    result = await agent.run("Say 'Hello!'")
    print(f"  Response: {result.content[:50]}...")

    await agent.close()

async def main():
    await test_mode(use_coordinator=False)  # Legacy
    await test_mode(use_coordinator=True)   # Coordinator

asyncio.run(main())
```

## Common Use Cases

### Use Case 1: Development

```bash
# Use coordinator mode for new development
export VICTOR_USE_COORDINATOR_ORCHESTRATOR=true

# Run your application
python my_app.py
```

### Use Case 2: Testing

```python
import pytest

@pytest.mark.parametrize("use_coordinator", [True, False])
async def test_workflow(use_coordinator):
    """Test workflow in both modes."""
    settings = Settings(use_coordinator_orchestrator=use_coordinator)
    agent = await Agent.create(settings=settings)
    result = await agent.run("Test")
    assert result is not None
```

### Use Case 3: Migration

```python
# Gradually migrate by testing both modes
async def migrate_to_coordinator():
    # Test legacy mode first
    settings = Settings(use_coordinator_orchestrator=False)
    agent = await Agent.create(settings=settings)
    result1 = await agent.run("Test task")
    await agent.close()

    # Test coordinator mode
    settings = Settings(use_coordinator_orchestrator=True)
    agent = await Agent.create(settings=settings)
    result2 = await agent.run("Test task")
    await agent.close()

    # Compare results
    assert result1.content == result2.content
    print("✅ Migration successful!")
```

## Troubleshooting

### Issue: Coordinator not available

```python
# Check if coordinator mode is enabled
from victor.config.settings import Settings

settings = Settings()
print(f"Coordinator mode: {settings.use_coordinator_orchestrator}")

# Enable if needed
settings.use_coordinator_orchestrator = True
```

### Issue: Import errors

```bash
# Ensure latest version is installed
pip install --upgrade victor-ai

# Check installation
python -c "from victor.agent.orchestrator_refactored import AgentOrchestratorRefactored"
```

### Issue: Different behavior

```python
# Compare both modes to identify differences
async def compare_modes():
    # Test both modes with same input
    for use_coordinator in [False, True]:
        settings = Settings(use_coordinator_orchestrator=use_coordinator)
        agent = await Agent.create(settings=settings)
        result = await agent.run("Test")
        print(f"Mode: {use_coordinator}, Result: {result.content[:50]}")
        await agent.close()
```

## Performance Considerations

**Initialization:**
- Legacy: ~500ms
- Coordinator: ~600ms (~100ms overhead, negligible)

**Memory:**
- Legacy: ~50MB
- Coordinator: ~55MB (~5MB overhead, negligible)

**Execution Speed:**
- Both modes: Fast (coordinator delegation adds ~1-2ms per call)

**Verdict:** The overhead is minimal compared to the benefits.

## Benefits of Coordinator Mode

### 1. Modularity

```python
# Easy to add new functionality
class NewFeatureCoordinator:
    def handle_new_feature(self):
        pass

# Register with orchestrator
orchestrator.new_feature_coordinator = NewFeatureCoordinator()
```

### 2. Testability

```python
# Test coordinators in isolation
def test_config_coordinator():
    coordinator = ConfigCoordinator(providers=[mock_provider])
    config = coordinator.get_config("key")
    assert config == "expected_value"
```

### 3. Extensibility

```python
# Extend without modifying core code
custom_coordinator = CustomConfigCoordinator()
orchestrator.config_coordinator = custom_coordinator
```

### 4. SOLID Principles

- **Single Responsibility**: Each coordinator has one job
- **Open/Closed**: Add coordinators without modifying existing code
- **Liskov Substitution**: Coordinators implement protocols
- **Interface Segregation**: Focused interfaces
- **Dependency Inversion**: Depend on protocols, not implementations

## Migration Path

### Phase 1: Test (Current)

```bash
# Enable in development only
export VICTOR_USE_COORDINATOR_ORCHESTRATOR=true

# Run tests
pytest tests/

# Run workflows
victor chat --no-tui
```

### Phase 2: Adopt (Future)

```python
# Default to coordinator mode for new code
settings = Settings(use_coordinator_orchestrator=True)
```

### Phase 3: Migrate (Future)

```python
# Update custom code to use coordinator pattern
# Replace direct AgentOrchestrator usage with coordinators
```

## Next Steps

1. **Read full documentation**: `docs/features/orchestrator_modes.md`
2. **Run complete example**: `examples/feature_flag_demo.py`
3. **Explore coordinators**: Check out coordinator-specific tutorials
4. **Report issues**: Provide feedback on coordinator mode

## Summary

✅ You learned:
- How to enable coordinator orchestrator mode
- Differences between legacy and coordinator modes
- How to use and extend coordinators
- Common use cases and patterns
- Migration path and best practices

You're now ready to use the coordinator orchestrator mode!
