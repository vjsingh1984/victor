# Protocol Best Practices

**Version**: 0.5.0
**Last Updated**: January 31, 2026
**Audience**: Developers, Contributors
**Purpose**: Best practices for using protocols in Victor AI

---

## Using Protocols

### Define Protocols First

When creating new components, define the protocol before implementation:

```python
# Good: Protocol first
@runtime_checkable
class MyServiceProtocol(Protocol):
    """Protocol for my service."""

    async def process(self, data: str) -> dict:
        """Process input data."""
        ...

class MyService:
    """Implementation of my service."""

    async def process(self, data: str) -> dict:
        # Implementation
        return {"result": data.upper()}
```

**Why**:
- Clear interface contract
- Easy to create mocks
- Enables multiple implementations

### Use Protocol Composition

Combine multiple protocols for complex interfaces:

```python
@runtime_checkable
class CacheProtocol(Protocol):
    """Caching operations."""

    async def get(self, key: str) -> Optional[Any]:
        ...

    async def set(self, key: str, value: Any) -> None:
        ...

@runtime_checkable
class MetricsProtocol(Protocol):
    """Metrics operations."""

    def track_operation(self, operation: str) -> None:
        ...

@runtime_checkable
class SmartCacheProtocol(CacheProtocol, MetricsProtocol, Protocol):
    """Cache with metrics tracking."""
    # Inherits both CacheProtocol and MetricsProtocol methods
    ...
```

### Prefer Protocols Over Abstract Base Classes

Use protocols for structural subtyping (duck typing):

```python
# Good: Protocol
@runtime_checkable
class WriterProtocol(Protocol):
    def write(self, data: str) -> None:
        ...

# Any class with write() method matches, no inheritance needed
class FileWriter:
    def write(self, data: str) -> None:
        with open("file.txt", "w") as f:
            f.write(data)

class ConsoleWriter:
    def write(self, data: str) -> None:
        print(data)

# Both match WriterProtocol
writer: WriterProtocol = FileWriter()  # OK
writer = ConsoleWriter()  # Also OK
```

### Use Runtime Checkable for Protocols

Make protocols runtime-checkable for isinstance() checks:

```python
# Good: Runtime checkable
@runtime_checkable
class MyProtocol(Protocol):
    def method(self) -> None:
        ...

class Implementation:
    def method(self) -> None:
        pass

impl = Implementation()
assert isinstance(impl, MyProtocol)  # Works
```

### Document Protocol Contracts Clearly

Use detailed docstrings for protocols:

```python
@runtime_checkable
class ToolExecutorProtocol(Protocol):
    """Protocol for tool execution.

    Responsibilities:
    - Execute tools with proper error handling
    - Track tool execution metrics
    - Manage tool timeouts

    Thread Safety:
    - Implementations must be thread-safe

    Error Handling:
    - Must raise ToolExecutionError on tool failure
    - Must validate arguments before execution
    """

    async def execute_tool(
        self,
        tool: BaseTool,
        arguments: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> ToolCallResult:
        """Execute a tool with the given arguments.

        Args:
            tool: Tool instance to execute
            arguments: Tool arguments (must be validated)
            timeout: Optional timeout in seconds

        Returns:
            ToolCallResult with execution result

        Raises:
            ToolExecutionError: If tool execution fails
            TimeoutError: If execution times out
            ValidationError: If arguments are invalid
        """
        ...
```

---

