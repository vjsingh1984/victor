# Orchestrator Thinking Parameter Usage

The orchestrator in this codebase supports an extended "thinking" mode that is specifically designed for Anthropic models to enable extended reasoning capabilities.

## Key Features

1. **Parameter Support**: The orchestrator accepts a `thinking` boolean parameter in its `__init__` method and `from_settings` class method.

2. **Provider Integration**: When `thinking=True` is set, the orchestrator passes a specific "thinking" parameter to the provider:
   ```python
   if self.thinking:
       provider_kwargs["thinking"] = {"type": "enabled", "budget_tokens": 10000}
   ```

3. **Anthropic Specific**: This feature is specifically designed for Anthropic models (like Claude) that support extended thinking mode, which allows the model to use more tokens for reasoning.

## Usage Examples

### In `__init__` method:
```python
orchestrator = AgentOrchestrator(
    settings=settings,
    provider=provider,
    model="claude-3-opus-20240229",
    thinking=True  # Enable extended thinking mode
)
```

### In `from_settings` class method:
```python
orchestrator = await AgentOrchestrator.from_settings(
    settings=settings,
    profile_name="default",
    thinking=True  # Enable extended thinking mode
)
```

## Implementation Details

- The `thinking` parameter is stored as `self.thinking = thinking` during initialization
- It's only passed to providers that support tool calls (`self.provider.supports_tools()`)
- The parameter is formatted specifically for Anthropic's extended thinking mode with a 10,000 token budget
- This is a model-specific feature and will be ignored by other providers

## Files Related to This Feature

- `victor/agent/orchestrator.py` - Main orchestrator implementation with thinking parameter support
- `tests/test_thinking_mode.py` - Tests for thinking mode functionality