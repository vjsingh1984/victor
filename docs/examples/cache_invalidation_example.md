# Cache Invalidation Integration Examples

This document provides practical examples of using the newly integrated cache invalidation components.

## Table of Contents

1. [FileWatcher Integration](#filewatcher-integration)
2. [DependencyExtractor Integration](#dependencyextractor-integration)
3. [Middleware Integration](#middleware-integration)
4. [Complete Example](#complete-example)

---

## FileWatcher Integration

### Basic Usage

```python
import asyncio
from victor.agent.cache.tool_cache_manager import ToolCacheManager
from victor.storage.cache.tiered_cache import TieredCache

async def main():
    # Create cache manager with file watching enabled
    manager = ToolCacheManager(
        backend=TieredCache(),
        enable_file_watching=True,  # Enable file watching
        default_ttl=3600,
    )

    # Start watching directories
    await manager.start_file_watching([
        "/Users/vijaysingh/code/myproject/src",
        "/Users/vijaysingh/code/myproject/tests",
    ])

    try:
        # Use the cache manager
        # File changes will automatically invalidate dependent caches

        # Cache a result that depends on /src/main.py
        await manager.set_tool_result(
            tool_name="read_file",
            args={"path": "/src/main.py"},
            result={"content": "print('hello')"},
            namespace=CacheNamespace.SESSION,
            file_dependencies={"/src/main.py"},  # Tracked dependency
        )

        # When /src/main.py is modified, the cache is automatically invalidated
        print("Watching files... (Press Ctrl+C to stop)")

        # Keep the program running to process file changes
        while True:
            await asyncio.sleep(1)

    finally:
        # Stop watching when done
        await manager.stop_file_watching()

if __name__ == "__main__":
    asyncio.run(main())
```

### With ToolPipeline

```python
from victor.agent.cache.tool_cache_manager import ToolCacheManager
from victor.storage.cache.tiered_cache import TieredCache
from victor.agent.tool_pipeline import ToolPipeline

# Create cache manager with file watching
cache_manager = ToolCacheManager(
    backend=TieredCache(),
    enable_file_watching=True,
)

# Start file watching
await cache_manager.start_file_watching(["/src"])

# Create pipeline with cache manager
pipeline = ToolPipeline(
    tool_registry=registry,
    tool_executor=executor,
    tool_cache=cache_manager,
)

# Now when tools execute, their dependencies are tracked
# And file changes automatically invalidate caches
```

---

## DependencyExtractor Integration

### Automatic Dependency Tracking

The DependencyExtractor is automatically integrated with ToolPipeline. No additional setup required:

```python
from victor.agent.cache.dependency_extractor import DependencyExtractor

# Dependencies are extracted automatically from tool arguments
extractor = DependencyExtractor()

# Example 1: Single file path
deps = extractor.extract_file_dependencies(
    "read_file",
    {"path": "/src/main.py", "encoding": "utf-8"}
)
print(deps)  # Output: {"/src/main.py"}

# Example 2: Multiple files
deps = extractor.extract_file_dependencies(
    "code_search",
    {
        "query": "authentication",
        "files": ["/src/auth.py", "/src/login.py", "/src/user.py"]
    }
)
print(deps)  # Output: {"/src/auth.py", "/src/login.py", "/src/user.py"}

# Example 3: Directory paths
deps = extractor.extract_file_dependencies(
    "list_directory",
    {"directory": "/src", "recursive": True}
)
print(deps)  # Output: {"/src"}
```

### Custom Extraction Patterns

```python
from victor.agent.cache.dependency_extractor import DependencyExtractor

class CustomDependencyExtractor(DependencyExtractor):
    """Custom extractor with additional patterns."""

    def __init__(self):
        super().__init__()

        # Add custom patterns for your tools
        self._file_patterns.extend([
            "config_file",  # For custom tools
            "template_path",
        ])

# Use custom extractor in pipeline
pipeline = ToolPipeline(
    tool_registry=registry,
    tool_executor=executor,
    dependency_extractor=CustomDependencyExtractor(),
)
```

---

## Middleware Integration

### YAML Configuration (Recommended)

Create a YAML configuration file for your vertical:

```yaml
# victor/myvertical/config/vertical.yaml

extensions:
  middleware:
    # Validation middleware - validates tool arguments
    - class: victor.framework.middleware_implementations.ValidationMiddleware
      enabled: true
      priority: high
      applicable_tools: [write_file, edit_file, create_directory]
      config:
        strict_mode: true
        validators:
          write_file:
            - lambda args: os.path.exists(os.path.dirname(args.get("path", "")))

    # Safety middleware - blocks dangerous operations
    - class: victor.framework.middleware_implementations.SafetyCheckMiddleware
      enabled: true
      priority: critical
      applicable_tools: [execute_bash, shell]
      config:
        check_path_traversal: true
        check_command_injection: true
        dangerous_patterns:
          - "rm -rf /"
          - "dd if=/dev/zero"

    # Cache middleware - caches tool results
    - class: victor.framework.middleware_implementations.CacheMiddleware
      enabled: true
      priority: high
      config:
        ttl_seconds: 300
        cacheable_tools: [read_file, list_directory, grep]

    # Metrics middleware - collects execution metrics
    - class: victor.framework.middleware_implementations.MetricsMiddleware
      enabled: true
      priority: low
      config:
        export_prometheus: false
        log_interval_seconds: 60

metadata:
  name: myvertical
  version: "1.0.0"
  description: "My custom vertical with middleware"

core:
  tools:
    list: [read_file, write_file, execute_bash]

  system_prompt:
    source: inline
    text: "You are an expert assistant..."
```

### Programmatic Configuration

```python
from victor.core.verticals.base import VerticalBase
from victor.framework.middleware_implementations import (
    ValidationMiddleware,
    SafetyCheckMiddleware,
    CacheMiddleware,
    MetricsMiddleware,
)

class MyVertical(VerticalBase):
    """Custom vertical with middleware."""

    name = "my_vertical"
    description = "Custom vertical with middleware support"

    @classmethod
    def get_tools(cls):
        return ["read_file", "write_file", "execute_bash", "grep"]

    @classmethod
    def get_system_prompt(cls):
        return "You are an expert assistant..."

    @classmethod
    def get_middleware(cls):
        """Provide middleware for this vertical."""
        return [
            # High priority validation
            ValidationMiddleware(
                enabled=True,
                applicable_tools={"write_file", "edit_file"},
                priority=MiddlewarePriority.HIGH,
            ),

            # Critical safety checks
            SafetyCheckMiddleware(
                enabled=True,
                applicable_tools={"execute_bash", "shell"},
                priority=MiddlewarePriority.CRITICAL,
            ),

            # Cache for read operations
            CacheMiddleware(
                enabled=True,
                ttl_seconds=300,
                cacheable_tools={"read_file", "grep", "list_directory"},
                priority=MiddlewarePriority.HIGH,
            ),

            # Low priority metrics
            MetricsMiddleware(
                enabled=True,
                priority=MiddlewarePriority.LOW,
            ),
        ]
```

### Custom Middleware

```python
from victor.framework.middleware_base import BaseMiddleware
from victor.framework.middleware_protocols import (
    MiddlewareResult,
    MiddlewarePriority,
    MiddlewarePhase,
)
import logging

class LoggingMiddleware(BaseMiddleware):
    """Custom middleware that logs all tool calls."""

    def __init__(self, log_level: str = "INFO"):
        super().__init__(
            enabled=True,
            priority=MiddlewarePriority.NORMAL,
            phase=MiddlewarePhase.AROUND,
        )
        self._logger = logging.getLogger(f"victor.middleware.logging")
        self._logger.setLevel(getattr(logging, log_level))

    async def before_tool_call(
        self,
        tool_name: str,
        arguments: dict,
    ) -> MiddlewareResult:
        """Log before tool execution."""
        self._logger.info(f"→ Calling {tool_name} with args: {arguments}")
        return MiddlewareResult(proceed=True)

    async def after_tool_call(
        self,
        tool_name: str,
        arguments: dict,
        result: any,
        success: bool,
    ) -> None:
        """Log after tool execution."""
        if success:
            self._logger.info(f"← {tool_name} succeeded")
        else:
            self._logger.warning(f"← {tool_name} failed")

# Use in vertical
class MyVertical(VerticalBase):
    @classmethod
    def get_middleware(cls):
        return [LoggingMiddleware(log_level="DEBUG")]
```

---

## Complete Example

### Putting It All Together

```python
import asyncio
from pathlib import Path

from victor.agent.cache.tool_cache_manager import ToolCacheManager, CacheNamespace
from victor.storage.cache.tiered_cache import TieredCache
from victor.agent.tool_pipeline import ToolPipeline, ToolPipelineConfig
from victor.agent.tool_executor import ToolExecutor
from victor.tools.registry import ToolRegistry
from victor.framework.middleware_implementations import (
    ValidationMiddleware,
    SafetyCheckMiddleware,
)
from victor.framework.middleware_protocols import MiddlewarePriority

async def main():
    # 1. Setup cache manager with file watching
    cache_manager = ToolCacheManager(
        backend=TieredCache(),
        enable_file_watching=True,
        default_ttl=3600,
    )

    # Start watching project files
    project_root = Path("/Users/vijaysingh/code/myproject")
    await cache_manager.start_file_watching([
        str(project_root / "src"),
        str(project_root / "tests"),
    ])

    # 2. Setup tool registry and executor
    registry = ToolRegistry.get_instance()
    executor = ToolExecutor(registry=registry)

    # 3. Create pipeline with caching and middleware
    pipeline = ToolPipeline(
        tool_registry=registry,
        tool_executor=executor,
        tool_cache=cache_manager,
        config=ToolPipelineConfig(
            tool_budget=50,
            enable_caching=True,
            enable_idempotent_caching=True,
        ),
    )

    # 4. Add middleware programmatically (or use YAML config)
    # middleware_chain = MiddlewareChain()
    # middleware_chain.add_middleware(ValidationMiddleware())
    # middleware_chain.add_middleware(SafetyCheckMiddleware())
    # pipeline.middleware_chain = middleware_chain

    try:
        # 5. Execute tools
        # Dependencies are automatically tracked
        result = await pipeline.execute_tool_calls([
            {
                "name": "read_file",
                "arguments": {"path": str(project_root / "src" / "main.py")}
            }
        ])

        print(f"Result: {result.results[0].result}")

        # 6. File changes automatically invalidate caches
        print("Watching for file changes...")

        # Simulate some work
        await asyncio.sleep(10)

    finally:
        # Cleanup
        await cache_manager.stop_file_watching()

if __name__ == "__main__":
    asyncio.run(main())
```

### Using Vertical with YAML Configuration

```python
from victor.core.verticals.base import VerticalBase

# Create a vertical with YAML config
class MyVertical(VerticalBase):
    name = "my_vertical"

# Config is automatically loaded from:
# victor/my_vertical/config/vertical.yaml

# Middleware is loaded from YAML:
# extensions:
#   middleware:
#     - class: ...

# Get the config (includes middleware)
config = MyVertical.get_config()

# Use the vertical
agent = await MyVertical.create_agent(
    provider="anthropic",
    model="claude-sonnet-4-5"
)
```

---

## Best Practices

### 1. File Watching

- Watch only necessary directories to avoid overhead
- Use `enable_file_watching=True` only when needed
- Always call `stop_file_watching()` in cleanup

### 2. Dependency Tracking

- Dependencies are tracked automatically - no manual work needed
- Use `file_dependencies` parameter for explicit dependencies
- Check logs for dependency tracking issues

### 3. Middleware

- Use YAML configuration for production (easier to maintain)
- Use programmatic configuration for testing (more flexibility)
- Order middleware by priority (CRITICAL > HIGH > NORMAL > LOW)
- Test middleware interactions carefully

### 4. Performance

- Enable caching for read-heavy workloads
- Use file watching only for long-running sessions
- Monitor cache hit rates to optimize TTL

---

## Troubleshooting

### File Changes Not Detected

```python
# Check if file watching is enabled
assert cache_manager._enable_file_watching

# Check if watcher is running
assert cache_manager._file_watcher is not None
assert cache_manager._file_watcher.is_running()
```

### Dependencies Not Tracked

```python
# Check if dependency extractor is configured
assert pipeline._dependency_extractor is not None

# Manually extract dependencies to debug
deps = pipeline._dependency_extractor.extract_file_dependencies(
    "read_file",
    {"path": "/src/main.py"}
)
print(f"Dependencies: {deps}")
```

### Middleware Not Executing

```python
# Check if middleware is enabled
for mw in middleware_chain._middleware:
    assert mw.enabled, f"{mw.__class__.__name__} is disabled"

# Check if middleware applies to the tool
tool_name = "write_file"
for mw in middleware_chain._middleware:
    if mw.applies_to_tool(tool_name):
        print(f"{mw.__class__.__name__} applies to {tool_name}")
```

---

## Additional Resources

- [FileWatcher Documentation](../agent/cache/file_watcher.py)
- [DependencyExtractor Documentation](../agent/cache/dependency_extractor.py)
- [Middleware Protocols](../framework/middleware_protocols.py)
- [Middleware Implementations](../framework/middleware_implementations.py)
- [VerticalBase Documentation](../core/verticals/base.py)
