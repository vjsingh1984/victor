# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Standard Exception Handling Patterns for Victor.

This module documents the official exception handling patterns to be used
throughout the Victor codebase. All exception handling should follow these
patterns to ensure consistency, debuggability, and proper error propagation.

## Error Hierarchy

Use the consolidated error hierarchy from victor.core.errors:

- VictorError (base)
  - ProviderError (LLM provider failures)
    - ProviderConnectionError (connection issues)
    - ProviderAuthError (authentication failures)
    - ProviderRateLimitError (rate limiting)
    - ProviderTimeoutError (timeouts)
    - ProviderNotFoundError (provider not in registry)
    - ProviderInvalidResponseError (malformed responses)
  - ToolError (tool execution failures)
    - ToolNotFoundError (tool not in registry)
    - ToolExecutionError (execution failures)
    - ToolValidationError (argument validation)
    - ToolTimeoutError (execution timeouts)
  - ConfigurationError (configuration/validation issues)
  - ValidationError (input validation)
  - FileError (file operation failures)
    - FileNotFoundError (file doesn't exist)
  - NetworkError (network-related failures)

## Rules

### RULE 1: Always catch specific exceptions when possible

✓ GOOD: Catches specific error type, provides context
```python
from victor.core.errors import ProviderError

try:
    result = provider.chat(messages)
except ProviderError as e:
    logger.error(
        "Provider call failed",
        exc_info=e,
        extra={"provider": provider.name, "model": model}
    )
    raise
```

✗ BAD: Generic exception loses type information
```python
try:
    result = provider.chat(messages)
except Exception as e:
    logger.error(f"Error: {e}")
    return None
```

### RULE 2: Use consolidated error hierarchy from victor.core.errors

✓ GOOD: Uses standard hierarchy
```python
from victor.core.errors import ProviderError, ToolError

try:
    result = provider.chat(messages)
except (ProviderError, ToolError) as e:
    logger.error("Operation failed", exc_info=e)
    raise
```

✗ BAD: Custom exception not in hierarchy
```python
try:
    result = provider.chat(messages)
except SomeRandomError as e:  # Not in victor.core.errors
    pass
```

### RULE 3: Always log exceptions with context

✓ GOOD: Logs with exc_info and extra context
```python
from victor.core.errors import ToolError

try:
    tool.execute(args)
except ToolError as e:
    logger.error(
        "Tool execution failed",
        exc_info=e,
        extra={"tool": tool.name, "args": args}
    )
    raise
```

✗ BAD: Just prints, no logging, no context
```python
try:
    tool.execute(args)
except Exception as e:
    print(e)
```

### RULE 4: Re-raise exceptions with context when appropriate

✓ GOOD: Adds context while preserving stack trace
```python
from victor.core.errors import ProviderError

try:
    connect_to_api(endpoint)
except requests.RequestException as e:
    raise ProviderConnectionError(
        f"Failed to connect to {endpoint}",
        provider=provider_name
    ) from e
```

✗ BAD: Loses original exception context
```python
try:
    connect_to_api(endpoint)
except Exception as e:
    raise Exception(str(e))  # Stack trace lost
```

### RULE 5: Use bare except: only for cleanup, then re-raise

✓ GOOD: Cleanup with re-raise
```python
try:
    resource.use()
except:
    resource.cleanup()
    raise  # Re-raise original exception
```

✗ BAD: Swallows all exceptions
```python
try:
    do_something()
except:
    pass  # Silently fails
```

### RULE 6: Log with exc_info for full stack traces

✓ GOOD: Full traceback captured
```python
from victor.core.errors import ToolExecutionError

try:
    result = tool.execute(**args)
except ToolExecutionError as e:
    logger.error(
        "Tool execution failed",
        exc_info=e,
        extra={"tool": tool.name}
    )
    raise
```

✗ BAD: Only logs string message
```python
try:
    result = tool.execute(**args)
except Exception as e:
    logger.error(f"Error: {e}")  # No traceback
```

## Common Patterns

### Pattern: Provider API Call
```python
import logging
from victor.core.errors import (
    ProviderError,
    ProviderConnectionError,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)

logger = logging.getLogger(__name__)

try:
    response = await provider.chat(messages, tools)
except ProviderConnectionError as e:
    logger.error(
        "Provider connection failed",
        exc_info=e,
        extra={"provider": provider.name, "endpoint": endpoint}
    )
    raise
except ProviderAuthError as e:
    logger.error(
        "Provider authentication failed",
        exc_info=e,
        extra={"provider": provider.name}
    )
    raise
except ProviderRateLimitError as e:
    logger.warning(
        "Provider rate limit exceeded",
        exc_info=e,
        extra={"provider": provider.name, "retry_after": e.retry_after}
    )
    raise
except ProviderTimeoutError as e:
    logger.error(
        "Provider request timeout",
        exc_info=e,
        extra={"provider": provider.name, "timeout": e.timeout}
    )
    raise
except ProviderError as e:
    logger.error(
        "Provider API call failed",
        exc_info=e,
        extra={"provider": provider.name, "model": model}
    )
    raise
```

### Pattern: Tool Execution
```python
import logging
from victor.core.errors import (
    ToolError,
    ToolNotFoundError,
    ToolExecutionError,
    ToolValidationError,
    ToolTimeoutError,
)

logger = logging.getLogger(__name__)

try:
    result = await tool.execute(**arguments)
except ToolNotFoundError as e:
    logger.error(
        "Tool not found",
        exc_info=e,
        extra={"tool_name": e.tool_name}
    )
    return {"error": str(e), "tool": e.tool_name}
except ToolValidationError as e:
    logger.error(
        "Tool validation failed",
        exc_info=e,
        extra={"tool": e.tool_name, "invalid_args": e.invalid_args}
    )
    return {"error": str(e), "tool": e.tool_name}
except ToolTimeoutError as e:
    logger.warning(
        "Tool execution timeout",
        exc_info=e,
        extra={"tool": e.tool_name, "timeout": e.timeout}
    )
    return {"error": str(e), "tool": e.tool_name}
except ToolExecutionError as e:
    logger.error(
        "Tool execution failed",
        exc_info=e,
        extra={"tool": e.tool_name}
    )
    return {"error": str(e), "tool": e.tool_name}
except ToolError as e:
    logger.error(
        "Tool error",
        exc_info=e,
        extra={"tool": e.tool_name}
    )
    return {"error": str(e), "tool": e.tool_name}
```

### Pattern: Configuration Validation
```python
import logging
from pydantic import ValidationError
from victor.core.errors import ConfigurationError

logger = logging.getLogger(__name__)

try:
    settings = VictorSettings.from_sources()
except ValidationError as e:
    logger.error(
        "Configuration validation failed",
        exc_info=e,
        extra={"errors": e.errors()}
    )
    raise ConfigurationError(f"Invalid configuration: {e}") from e
except Exception as e:
    logger.error(
        "Configuration loading failed",
        exc_info=e
    )
    raise ConfigurationError(f"Failed to load configuration: {e}") from e
```

### Pattern: File Operations
```python
import logging
from victor.core.errors import FileError, FileNotFoundError

logger = logging.getLogger(__name__)

try:
    with open(file_path, 'r') as f:
        content = f.read()
except FileNotFoundError as e:
    logger.error(
        "File not found",
        exc_info=e,
        extra={"path": file_path}
    )
    raise FileNotFoundError(path=file_path) from e
except PermissionError as e:
    logger.error(
        "File permission denied",
        exc_info=e,
        extra={"path": file_path}
    )
    raise FileError(
        f"Permission denied for {file_path}",
        path=file_path
    ) from e
except IOError as e:
    logger.error(
        "File I/O error",
        exc_info=e,
        extra={"path": file_path}
    )
    raise FileError(
        f"I/O error reading {file_path}",
        path=file_path
    ) from e
```

### Pattern: Network Operations
```python
import logging
import httpx
from victor.core.errors import NetworkError

logger = logging.getLogger(__name__)

try:
    response = await client.get(url)
except httpx.ConnectError as e:
    logger.error(
        "Network connection failed",
        exc_info=e,
        extra={"url": url}
    )
    raise NetworkError(
        f"Failed to connect to {url}",
        url=url
    ) from e
except httpx.TimeoutException as e:
    logger.error(
        "Network request timeout",
        exc_info=e,
        extra={"url": url}
    )
    raise NetworkError(
        f"Request timeout for {url}",
        url=url
    ) from e
except httpx.HTTPError as e:
    logger.error(
        "HTTP error",
        exc_info=e,
        extra={"url": url, "status": getattr(e, 'response', {}).get('status_code')}
    )
    raise NetworkError(
        f"HTTP error for {url}",
        url=url
    ) from e
```

### Pattern: Cleanup with Re-raise
```python
import logging

logger = logging.getLogger(__name__)

resource = None
try:
    resource = acquire_resource()
    await resource.process()
except:
    # Cleanup code that must run even on error
    if resource:
        try:
            await resource.cleanup()
        except Exception as cleanup_error:
            logger.error(
                "Cleanup failed",
                exc_info=cleanup_error
            )
    raise  # Re-raise original exception
finally:
    # Final cleanup
    if resource:
        resource.release()
```

### Pattern: Converting Standard Exceptions
```python
import logging
from victor.core.errors import ProviderError, ToolError

logger = logging.getLogger(__name__)

# Convert standard exceptions to Victor exceptions
try:
    result = external_api_call()
except ValueError as e:
    logger.error(
        "Invalid value from provider",
        exc_info=e
    )
    raise ProviderError(
        f"Provider returned invalid value: {e}",
        provider=provider_name
    ) from e
except KeyError as e:
    logger.error(
        "Missing required field in response",
        exc_info=e
    )
    raise ProviderError(
        f"Provider response missing required field: {e}",
        provider=provider_name
    ) from e
```

## Testing Exception Handling

Always test that:
1. Correct exception type is raised
2. Exception message is informative
3. Exception is logged properly
4. Cleanup occurs on exception

```python
import pytest
import logging
from victor.core.errors import ProviderError, ToolError

def test_provider_error_handling(caplog):
    \"\"\"Verify provider errors are handled correctly.\"\"\"
    caplog.set_level(logging.ERROR)

    with pytest.raises(ProviderError) as exc_info:
        provider.chat(invalid_messages)

    # Verify exception type and message
    assert "Provider call failed" in str(exc_info.value)
    assert exc_info.value.provider == "test_provider"

    # Verify logging occurred
    assert "Provider call failed" in caplog.text
    assert "test_provider" in caplog.text

def test_tool_error_handling(caplog):
    \"\"\"Verify tool errors are handled correctly.\"\"\"
    caplog.set_level(logging.ERROR)

    with pytest.raises(ToolError) as exc_info:
        tool.execute(invalid_args)

    # Verify exception details
    assert "Tool execution failed" in str(exc_info.value)
    assert exc_info.value.tool_name == "test_tool"

    # Verify logging
    assert "Tool execution failed" in caplog.text

def test_cleanup_on_error():
    \"\"\"Verify cleanup occurs even on exception.\"\"\"
    resource = MockResource()

    with pytest.raises(Exception):
        with resource:
            raise Exception("Test error")

    # Verify cleanup was called
    assert resource.cleaned_up
```

## Migration Checklist

When migrating exception handling code:

1. [ ] Identify the operation type (provider, tool, config, file, network)
2. [ ] Import appropriate exception types from victor.core.errors
3. [ ] Replace generic `except Exception:` with specific exception types
4. [ ] Add logging with `exc_info=e` and `extra={}` context
5. [ ] Either re-raise or convert to Victor exception with `from e`
6. [ ] Add tests verifying exception type and logging
7. [ ] Remove bare `except:` unless needed for cleanup (then re-raise)
8. [ ] Verify all exception messages are informative

## Examples of Migration

### Before: Generic Exception
```python
try:
    result = provider.chat(messages)
except Exception as e:
    logger.error(f"Error: {e}")
    return None
```

### After: Specific Exception
```python
from victor.core.errors import ProviderError

try:
    result = provider.chat(messages)
except ProviderError as e:
    logger.error(
        "Provider call failed",
        exc_info=e,
        extra={"provider": provider.name, "model": self.model}
    )
    raise
```

### Before: Bare Except
```python
try:
    do_something()
except:
    pass
```

### After: Specific with Cleanup
```python
from victor.core.errors import ToolError

try:
    do_something()
except (ProviderError, ToolError) as e:
    logger.error("Operation failed", exc_info=e)
    cleanup()
    raise
```

### Before: No Logging Context
```python
try:
    tool.execute(args)
except Exception as e:
    print(e)
```

### After: Full Context
```python
from victor.core.errors import ToolExecutionError

try:
    tool.execute(args)
except ToolExecutionError as e:
    logger.error(
        "Tool execution failed",
        exc_info=e,
        extra={"tool": tool.name, "args": args}
    )
    raise
```

## Summary

Follow these patterns to ensure:
- **Consistency**: All code handles errors the same way
- **Debuggability**: Full stack traces and context in logs
- **Maintainability**: Clear error types and recovery hints
- **Reliability**: Proper cleanup and error propagation
"""
