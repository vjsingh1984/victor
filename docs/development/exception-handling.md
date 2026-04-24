# Exception Handling

Victor uses the consolidated error hierarchy in `victor.core.errors`. Runtime
code should import concrete error classes from that module rather than defining
local exception hierarchies or importing deprecated framework shims.

## Error Hierarchy

Use `VictorError` as the base for project errors. The common public categories
are:

- `ProviderError` for LLM provider failures.
- `ToolError` for tool execution failures.
- `ConfigurationError` for invalid or missing configuration.
- `ValidationError` for invalid user or API input.
- `FileError` and `NetworkError` for resource failures.

Specialized subclasses such as `ProviderConnectionError`,
`ProviderRateLimitError`, `ToolExecutionError`, and `ToolValidationError`
should be preferred when they describe the failure precisely.

## Rules

1. Catch the most specific Victor error type available.
2. Preserve exception context with `raise ... from e` when wrapping lower-level
   exceptions.
3. Log with `exc_info` and structured context before re-raising.
4. Avoid bare `except:` except for cleanup blocks that immediately re-raise.
5. Do not return `None` or silently swallow failures that callers need to
   handle.
6. Put new reusable error classes in `victor.core.errors`.

## Provider Pattern

```python
import logging

from victor.core.errors import ProviderAuthError, ProviderConnectionError, ProviderError

logger = logging.getLogger(__name__)


async def call_provider(provider, messages):
    try:
        return await provider.chat(messages)
    except ProviderAuthError:
        logger.error("Provider authentication failed", exc_info=True, extra={"provider": provider.name})
        raise
    except ProviderConnectionError:
        logger.error("Provider connection failed", exc_info=True, extra={"provider": provider.name})
        raise
    except ProviderError:
        logger.error("Provider call failed", exc_info=True, extra={"provider": provider.name})
        raise
```

## Tool Pattern

```python
import logging

from victor.core.errors import ToolExecutionError, ToolValidationError

logger = logging.getLogger(__name__)


async def run_tool(tool, args):
    try:
        return await tool.execute(**args)
    except ToolValidationError:
        logger.warning("Tool arguments are invalid", exc_info=True, extra={"tool": tool.name})
        raise
    except ToolExecutionError:
        logger.error("Tool execution failed", exc_info=True, extra={"tool": tool.name, "args": args})
        raise
```

## Wrapping External Exceptions

```python
from victor.core.errors import ProviderConnectionError


async def connect(provider_name, client):
    try:
        return await client.connect()
    except OSError as e:
        raise ProviderConnectionError(
            "Failed to connect to provider",
            provider=provider_name,
        ) from e
```

## Anti-Patterns

Avoid generic catch-and-suppress logic:

```python
try:
    result = await provider.chat(messages)
except Exception as e:
    logger.error(f"Error: {e}")
    return None
```

Avoid losing the original cause:

```python
try:
    result = await provider.chat(messages)
except Exception as e:
    raise Exception(str(e))
```

Prefer the canonical hierarchy:

```python
from victor.core.errors import ProviderError

try:
    result = await provider.chat(messages)
except ProviderError:
    logger.error("Provider call failed", exc_info=True)
    raise
```
