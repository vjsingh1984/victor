# Victor Error Handling Patterns

This document describes the standardized error handling patterns used across the Victor codebase.

## Table of Contents

1. [Overview](#overview)
2. [Standard Error Patterns](#standard-error-patterns)
3. [Pattern by Module](#pattern-by-module)
4. [Best Practices](#best-practices)
5. [Examples](#examples)

---

## Overview

Victor uses **exception-based error handling** as the primary pattern. All errors inherit from `VictorError` base class and provide:

- Structured error information (category, severity, correlation ID)
- User-friendly messages with recovery hints
- Contextual details for debugging
- Consistent logging and tracking

### Error Hierarchy

```
VictorError (base)
├── ProviderError
│   ├── ProviderConnectionError
│   ├── ProviderAuthError
│   ├── ProviderRateLimitError
│   ├── ProviderTimeoutError
│   ├── ProviderNotFoundError
│   ├── ProviderInitializationError
│   └── ProviderInvalidResponseError
├── ToolError
│   ├── ToolNotFoundError
│   ├── ToolExecutionError
│   ├── ToolValidationError
│   └── ToolTimeoutError
├── ConfigurationError
│   └── ConfigurationValidationError
├── ValidationError
├── FileError
│   └── FileNotFoundError
├── NetworkError
├── SearchError
├── WorkflowExecutionError
└── ExtensionLoadError
```

---

## Standard Error Patterns

### Pattern 1: Raise Specialized Exception (Recommended)

**When to use:** All error conditions that should propagate to the caller.

**Implementation:**
```python
from victor.core.errors import ToolExecutionError, ErrorCategory

def execute_tool(tool_name: str, **kwargs):
    if not tool_exists(tool_name):
        raise ToolNotFoundError(
            tool_name=tool_name,
        )
    # Tool execution logic
```

**Benefits:**
- Type-safe error handling
- Structured error information
- Automatic correlation ID generation
- Built-in recovery hints
- Error tracking integration

### Pattern 2: Return Result Objects (Specific Cases Only)

**When to use:** Tool execution results via `ToolResult` class.

**Implementation:**
```python
from victor.tools.base import ToolResult

async def execute(self, _exec_ctx: Dict[str, Any], **kwargs) -> ToolResult:
    try:
        result = perform_operation(**kwargs)
        return ToolResult(success=True, output=result)
    except Exception as e:
        return ToolResult(success=False, output=None, error=str(e))
```

**Note:** This pattern is ONLY for `ToolResult` returns from tool `execute()` methods. All other errors should use exceptions.

### Pattern 3: Return None on Failure (Deprecated)

**When to use:** Nowhere - this pattern is deprecated.

**Migration path:**
```python
# OLD (deprecated):
def get_tool(name: str) -> Optional[BaseTool]:
    return registry.get(name)  # Returns None if not found

# NEW (recommended):
def get_tool(name: str) -> BaseTool:
    tool = registry.get(name)
    if not tool:
        raise ToolNotFoundError(tool_name=name)
    return tool
```

---

## Pattern by Module

### Tool Executor (`victor/agent/tool_executor.py`)

**Pattern:** Raise `ToolExecutionError` with context

```python
from victor.core.errors import ToolExecutionError, ToolTimeoutError

async def execute(self, tool_name: str, arguments: Dict[str, Any]) -> ToolExecutionResult:
    # Tool not found
    tool = self.tools.get(tool_name)
    if not tool:
        raise ToolNotFoundError(tool_name=tool_name)

    # Tool disabled
    if not self.tools.is_tool_enabled(tool_name):
        raise ToolExecutionError(
            message=f"Tool '{tool_name}' is disabled",
            tool_name=tool_name,
        )

    # Validation failed
    if not validation.valid:
        raise ToolValidationError(
            message="Invalid arguments",
            tool_name=tool_name,
            invalid_args=list(validation.invalid_params.keys()),
        )

    # Timeout
    try:
        async with timeout(seconds):
            result = await tool.execute(_exec_ctx=context, **arguments)
    except asyncio.TimeoutError:
        raise ToolTimeoutError(tool_name=tool_name, timeout=seconds)

    # Execution failure
    if isinstance(result, ToolResult) and not result.success:
        raise ToolExecutionError(
            message=result.error or "Tool execution failed",
            tool_name=tool_name,
            arguments=arguments,
        )
```

**Key patterns:**
1. Always include `tool_name` in error context
2. Use specific error types (not generic `ToolError`)
3. Include `arguments` in `ToolExecutionError` for debugging
4. Log via `ErrorHandler` for structured tracking

### Provider Base (`victor/providers/base.py`)

**Pattern:** Raise `ProviderError` subclasses with provider context

```python
from victor.core.errors import (
    ProviderConnectionError,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)

async def chat(self, messages, *, model, **kwargs):
    try:
        response = await self._make_request(messages, model, **kwargs)
        return CompletionResponse(**response)
    except ConnectionError as e:
        raise ProviderConnectionError(
            message=f"Failed to connect to {self.name}",
            provider=self.name,
        ) from e
    except AuthenticationError as e:
        raise ProviderAuthError(
            message=f"Authentication failed for {self.name}",
            provider=self.name,
        ) from e
    except HTTPStatusError as e:
        if e.response.status_code == 429:
            retry_after = e.response.headers.get("Retry-After")
            raise ProviderRateLimitError(
                message=f"Rate limit exceeded for {self.name}",
                provider=self.name,
                retry_after=int(retry_after) if retry_after else None,
            ) from e
        elif e.response.status_code >= 500:
            raise ProviderConnectionError(
                message=f"Server error from {self.name}",
                provider=self.name,
            ) from e
```

**Key patterns:**
1. Always include `provider` (or `provider_name`) in error context
2. Use specific `ProviderError` subclasses
3. Include `model` parameter when relevant
4. Chain exceptions with `from e` for traceback
5. Map HTTP status codes to appropriate error types

### Tool Base (`victor/tools/base.py`)

**Pattern:** Return `ToolResult` from execute(), raise for validation errors

```python
from victor.tools.base import BaseTool, ToolResult, ToolValidationResult

class MyTool(BaseTool):
    @property
    def name(self) -> str:
        return "my_tool"

    @property
    def description(self) -> str:
        return "Does something useful"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "Input data"},
            },
            "required": ["input"],
        }

    async def execute(self, _exec_ctx: Dict[str, Any], **kwargs) -> ToolResult:
        # Validate parameters (uses tool's JSON schema)
        validation = self.validate_parameters_detailed(**kwargs)
        if not validation.valid:
            # For tools: return ToolResult, don't raise
            return ToolResult(
                success=False,
                output=None,
                error=f"Validation failed: {', '.join(validation.errors)}",
            )

        try:
            result = self._do_work(**kwargs)
            return ToolResult(success=True, output=result)
        except Exception as e:
            # Return error in ToolResult format
            return ToolResult(
                success=False,
                output=None,
                error=f"Execution failed: {str(e)}",
            )
```

**Key patterns:**
1. `execute()` returns `ToolResult`, never raises
2. Use `ToolResult.success` to indicate outcome
3. Include error message in `ToolResult.error`
4. Return actual output in `ToolResult.output`

---

## Best Practices

### 1. Always Use Specific Error Types

```python
# BAD:
raise Exception("Tool not found")

# GOOD:
raise ToolNotFoundError(tool_name="my_tool")
```

### 2. Include Contextual Information

```python
# BAD:
raise ToolExecutionError("Failed")

# GOOD:
raise ToolExecutionError(
    message="Failed to write file",
    tool_name="write_file",
    arguments={"path": "/tmp/file.txt", "content": "..."},
)
```

### 3. Provide Recovery Hints

Most error types include default recovery hints. Override when specific guidance helps:

```python
raise ToolExecutionError(
    message="Docker daemon not running",
    tool_name="docker",
    recovery_hint="Start Docker with: sudo systemctl start docker (Linux) or open Docker Desktop (Mac/Windows)",
)
```

### 4. Chain Exceptions

```python
# BAD:
try:
    risky_operation()
except ValueError:
    raise ToolExecutionError("Failed")

# GOOD:
try:
    risky_operation()
except ValueError as e:
    raise ToolExecutionError("Failed", tool_name="my_tool") from e
```

### 5. Use ErrorHandler for Logging

```python
from victor.core.errors import get_error_handler

error_handler = get_error_handler()

try:
    operation()
except Exception as e:
    error_info = error_handler.handle(
        e,
        context={"tool": tool_name, "attempt": attempt},
    )
    # error_info contains correlation_id, category, etc.
    logger.error("Operation failed: %s", error_info.correlation_id)
```

---

## Examples

### Example 1: Tool with Validation

```python
from victor.tools.base import BaseTool, ToolResult

class ReadFileTool(BaseTool):
    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read file contents"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path to read",
                },
                "encoding": {
                    "type": "string",
                    "description": "File encoding",
                    "default": "utf-8",
                },
            },
            "required": ["path"],
        }

    async def execute(self, _exec_ctx: Dict[str, Any], **kwargs) -> ToolResult:
        # Validate
        validation = self.validate_parameters_detailed(**kwargs)
        if not validation.valid:
            return ToolResult(
                success=False,
                output=None,
                error=f"Invalid arguments: {validation.errors[0]}",
            )

        path = kwargs["path"]
        encoding = kwargs.get("encoding", "utf-8")

        try:
            with open(path, "r", encoding=encoding) as f:
                content = f.read()
            return ToolResult(success=True, output=content)
        except FileNotFoundError:
            return ToolResult(
                success=False,
                output=None,
                error=f"File not found: {path}",
            )
        except PermissionError:
            return ToolResult(
                success=False,
                output=None,
                error=f"Permission denied: {path}",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Failed to read file: {str(e)}",
            )
```

### Example 2: Provider with Error Mapping

```python
from victor.providers.base import BaseProvider
from victor.core.errors import (
    ProviderConnectionError,
    ProviderAuthError,
    ProviderRateLimitError,
)

class MyProvider(BaseProvider):
    @property
    def name(self) -> str:
        return "myprovider"

    async def chat(self, messages, *, model, **kwargs):
        try:
            response = await self._client.chat(
                messages=messages,
                model=model,
                **kwargs
            )
            return CompletionResponse(**response)

        except httpx.HTTPStatusError as e:
            status = e.response.status_code

            if status == 401:
                raise ProviderAuthError(
                    message="Invalid API key",
                    provider=self.name,
                ) from e

            elif status == 429:
                retry_after = e.response.headers.get("Retry-After")
                raise ProviderRateLimitError(
                    message="Rate limit exceeded",
                    provider=self.name,
                    retry_after=int(retry_after) if retry_after else None,
                ) from e

            elif status >= 500:
                raise ProviderConnectionError(
                    message=f"Server error: {status}",
                    provider=self.name,
                ) from e

            else:
                raise ProviderConnectionError(
                    message=f"HTTP error: {status}",
                    provider=self.name,
                ) from e

        except httpx.ConnectError as e:
            raise ProviderConnectionError(
                message="Failed to connect",
                provider=self.name,
            ) from e

        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(
                message="Request timed out",
                provider=self.name,
                timeout=self.timeout,
            ) from e
```

### Example 3: Tool Executor with Comprehensive Error Handling

```python
from victor.core.errors import (
    ToolNotFoundError,
    ToolExecutionError,
    ToolTimeoutError,
    get_error_handler,
)

class ToolExecutor:
    async def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> ToolExecutionResult:
        error_handler = get_error_handler()

        # Get tool
        tool = self.tools.get(tool_name)
        if not tool:
            error = ToolNotFoundError(tool_name=tool_name)
            error_info = error_handler.handle(error)
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=str(error),
                error_info=error_info,
            )

        # Execute with timeout
        try:
            result = await asyncio.wait_for(
                tool.execute(_exec_ctx=self.context, **arguments),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError as e:
            error = ToolTimeoutError(
                tool_name=tool_name,
                timeout=self.timeout,
            )
            error_info = error_handler.handle(
                e,
                context={"tool": tool_name, "timeout": self.timeout},
            )
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=str(error),
                error_info=error_info,
            )

        # Handle result
        if isinstance(result, ToolResult):
            if result.success:
                return ToolExecutionResult(
                    tool_name=tool_name,
                    success=True,
                    result=result.output,
                )
            else:
                error = ToolExecutionError(
                    message=result.error or "Tool failed",
                    tool_name=tool_name,
                    arguments=arguments,
                )
                error_info = error_handler.handle(error)
                return ToolExecutionResult(
                    tool_name=tool_name,
                    success=False,
                    result=None,
                    error=str(error),
                    error_info=error_info,
                )
```

---

## Error Categories

| Category | Description | Example Errors |
|----------|-------------|----------------|
| `PROVIDER_CONNECTION` | Network/connection failures | `ProviderConnectionError` |
| `PROVIDER_AUTH` | Authentication failures | `ProviderAuthError` |
| `PROVIDER_RATE_LIMIT` | Rate limit exceeded | `ProviderRateLimitError` |
| `TOOL_NOT_FOUND` | Tool not in registry | `ToolNotFoundError` |
| `TOOL_EXECUTION` | Tool execution failed | `ToolExecutionError` |
| `TOOL_VALIDATION` | Invalid tool arguments | `ToolValidationError` |
| `TOOL_TIMEOUT` | Tool execution timeout | `ToolTimeoutError` |
| `CONFIG_INVALID` | Invalid configuration | `ConfigurationError` |
| `FILE_NOT_FOUND` | File doesn't exist | `FileNotFoundError` |
| `NETWORK_ERROR` | Network failures | `NetworkError` |

---

## Testing Error Handling

```python
import pytest
from victor.core.errors import ToolNotFoundError, ToolExecutionError

def test_tool_not_found():
    executor = ToolExecutor(tool_registry=empty_registry())

    with pytest.raises(ToolNotFoundError) as exc_info:
        executor.get_tool("nonexistent")

    assert exc_info.value.tool_name == "nonexistent"
    assert exc_info.value.category == ErrorCategory.TOOL_NOT_FOUND
    assert exc_info.value.correlation_id is not None

def test_tool_execution_error():
    tool = FailingTool()
    executor = ToolExecutor(tool_registry=registry_with(tool))

    result = await executor.execute("failing_tool", {})

    assert result.success is False
    assert result.error_info is not None
    assert result.error_info.category == ErrorCategory.TOOL_EXECUTION
```

---

## Migration Guide

### From Return-None to Raise

```python
# BEFORE:
def get_provider(name: str) -> Optional[BaseProvider]:
    return registry.get(name)

# Usage requires None check:
provider = get_provider("anthropic")
if provider is None:
    logger.error("Provider not found")
    return

# AFTER:
def get_provider(name: str) -> BaseProvider:
    provider = registry.get(name)
    if not provider:
        raise ProviderNotFoundError(provider=name)
    return provider

# Usage is cleaner:
try:
    provider = get_provider("anthropic")
except ProviderNotFoundError as e:
    logger.error(str(e))
    return
```

### From Generic Exception to Specific Error

```python
# BEFORE:
raise ValueError(f"Tool {name} not found")

# AFTER:
raise ToolNotFoundError(tool_name=name)
```

---

## Summary

| Pattern | Use When | Example |
|---------|----------|---------|
| **Raise exception** | All error conditions | `raise ToolNotFoundError(tool_name="x")` |
| **Return ToolResult** | Tool `execute()` methods | `return ToolResult(success=True, output=data)` |
| **Return None** | Nowhere (deprecated) | N/A |

**Key principles:**
1. Use specific error types from `victor.core.errors`
2. Always include context (tool_name, provider, etc.)
3. Chain exceptions with `from e`
4. Use `ErrorHandler` for structured logging
5. Return `ToolResult` only from tool `execute()` methods

---

**Last Updated:** 2025-01-14
**Version:** 0.5.1
