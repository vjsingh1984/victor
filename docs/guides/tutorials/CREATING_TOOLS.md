# Creating Custom Tools for Victor AI

This comprehensive guide teaches you how to create custom tools for Victor AI.

## Table of Contents

1. [Introduction](#introduction)
2. [Tool Architecture](#tool-architecture)
3. [Basic Tool Creation](#basic-tool-creation)
4. [Advanced Tool Features](#advanced-tool-features)
5. [Tool Registration](#tool-registration)
6. [Testing Tools](#testing-tools)
7. [Best Practices](#best-practices)
8. [Examples](#examples)

## Introduction

### What are Tools?

Tools are reusable components that extend Victor AI's capabilities. They can:
- Execute system commands
- Query databases
- Make API calls
- Process files
- Perform domain-specific operations

### Why Create Custom Tools?

- **Extend functionality**: Add capabilities specific to your needs
- **Integrate services**: Connect to external APIs and services
- **Automate workflows**: Automate repetitive tasks
- **Custom logic**: Implement business logic specific to your domain

## Tool Architecture

### Base Tool Class

All tools inherit from `BaseTool`:

```python
from victor.tools.base import BaseTool, CostTier
from typing import Dict, Any, Optional
import json

class MyCustomTool(BaseTool):
    """Description of what your tool does."""

    name = "my_custom_tool"  # Unique tool identifier
    description = "Detailed description for LLM understanding"
    cost_tier = CostTier.LOW  # FREE, LOW, MEDIUM, HIGH
    category = "custom"  # Tool category for grouping

    parameters = {
        "type": "object",
        "properties": {
            "param1": {
                "type": "string",
                "description": "Description of parameter 1"
            },
            "param2": {
                "type": "integer",
                "description": "Description of parameter 2"
            }
        },
        "required": ["param1"]
    }

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool logic."""
        # Your implementation here
        return {
            "success": True,
            "result": "Tool execution result"
        }
```

### Cost Tiers

Tools are categorized by cost:

| Tier | Description | Examples |
|------|-------------|----------|
| `FREE` | No cost | File operations, local computations |
| `LOW` | Minimal cost | Simple API calls |
| `MEDIUM` | Moderate cost | Complex operations |
| `HIGH` | Expensive operations | Large-scale computations |

## Basic Tool Creation

### Step 1: Define Your Tool

Let's create a simple tool that calculates file statistics:

```python
# victor/tools/file_stats.py

from pathlib import Path
from typing import Dict, Any
from victor.tools.base import BaseTool, CostTier

class FileStatsTool(BaseTool):
    """Calculate statistics for a file or directory."""

    name = "file_stats"
    description = "Calculate statistics including line count, size, and file type for files or directories"
    cost_tier = CostTier.FREE
    category = "filesystem"

    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to file or directory"
            },
            "recursive": {
                "type": "boolean",
                "description": "Recursively analyze directories",
                "default": False
            }
        },
        "required": ["path"]
    }

    async def execute(
        self,
        path: str,
        recursive: bool = False
    ) -> Dict[str, Any]:
        """Execute file statistics calculation."""
        try:
            path_obj = Path(path)

            if not path_obj.exists():
                return {
                    "success": False,
                    "error": f"Path not found: {path}"
                }

            if path_obj.is_file():
                return self._analyze_file(path_obj)
            else:
                return self._analyze_directory(path_obj, recursive)

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single file."""
        stats = file_path.stat()

        # Count lines if text file
        line_count = 0
        if self._is_text_file(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    line_count = sum(1 for _ in f)
            except Exception:
                pass

        return {
            "success": True,
            "type": "file",
            "path": str(file_path),
            "size_bytes": stats.st_size,
            "size_human": self._format_size(stats.st_size),
            "line_count": line_count,
            "extension": file_path.suffix,
            "is_text": self._is_text_file(file_path)
        }

    def _analyze_directory(
        self,
        dir_path: Path,
        recursive: bool
    ) -> Dict[str, Any]:
        """Analyze a directory."""
        files = list(dir_path.rglob('*')) if recursive else list(dir_path.iterdir())

        total_size = 0
        file_count = 0
        dir_count = 0
        line_count = 0
        file_types = {}

        for item in files:
            if item.is_file():
                file_count += 1
                stats = item.stat()
                total_size += stats.st_size

                # Count file types
                ext = item.suffix or "no_extension"
                file_types[ext] = file_types.get(ext, 0) + 1

                # Count lines for text files
                if self._is_text_file(item):
                    try:
                        with open(item, 'r', encoding='utf-8') as f:
                            line_count += sum(1 for _ in f)
                    except Exception:
                        pass
            elif item.is_dir():
                dir_count += 1

        return {
            "success": True,
            "type": "directory",
            "path": str(dir_path),
            "file_count": file_count,
            "directory_count": dir_count,
            "total_size_bytes": total_size,
            "total_size_human": self._format_size(total_size),
            "total_line_count": line_count,
            "file_types": file_types,
            "recursive": recursive
        }

    @staticmethod
    def _is_text_file(file_path: Path) -> bool:
        """Check if file is likely a text file."""
        text_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp',
            '.h', '.hpp', '.cs', '.go', '.rs', '.rb', '.php', '.swift',
            '.kt', '.scala', '.md', '.txt', '.json', '.yaml', '.yml',
            '.xml', '.html', '.css', '.scss', '.sh', '.bash', '.zsh',
            '.toml', '.ini', '.cfg', '.conf', '.sql', '.graphql'
        }
        return file_path.suffix.lower() in text_extensions

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"
```

### Step 2: Register Your Tool

```python
# victor/tools/__init__.py

from .file_stats import FileStatsTool

# Register tool
tool_registry.register(FileStatsTool())
```

Or use entry points for external tools:

```toml
# pyproject.toml

[project.entry-points."victor.tools"]
file_stats = "victor.tools.file_stats:FileStatsTool"
```

### Step 3: Use Your Tool

```bash
# In Victor CLI
victor chat "Calculate statistics for src/ directory"

# In Python code
from victor import Agent

agent = await Agent.create()
result = await agent.execute_tool(
    "file_stats",
    path="src/",
    recursive=True
)
print(result)
```

## Advanced Tool Features

### Tool with Configuration

```python
from victor.tools.base import BaseTool
from victor.core.capabilities import ConfigurationCapabilityProvider

class ConfigurableTool(BaseTool):
    """Tool with configuration support."""

    name = "configurable_tool"
    description = "Tool that uses configuration"
    cost_tier = CostTier.LOW

    def __init__(self):
        super().__init__()
        # Load configuration
        self.config = ConfigurationCapabilityProvider()
        self.api_endpoint = self.config.get("api_endpoint")
        self.timeout = self.config.get("timeout", 30)

    async def execute(self, **kwargs):
        # Use configuration
        timeout = kwargs.get("timeout", self.timeout)
        # Implementation...
        pass
```

### Tool with Dependencies

```python
from victor.tools.base import BaseTool
from victor.protocols import ISemanticSearch

class SearchAwareTool(BaseTool):
    """Tool that uses semantic search."""

    name = "search_aware_tool"
    description = "Tool using semantic search"

    def __init__(self, semantic_search: ISemanticSearch):
        super().__init__()
        self.search = semantic_search

    async def execute(self, query: str, **kwargs):
        # Use semantic search
        results = await self.search.search(query, limit=5)
        return {"results": results}
```

### Tool with Error Handling

```python
from victor.tools.base import BaseTool, ToolExecutionError

class RobustTool(BaseTool):
    """Tool with comprehensive error handling."""

    name = "robust_tool"
    description = "Tool with error handling"

    async def execute(self, **kwargs):
        try:
            # Validate input
            if not self._validate_input(kwargs):
                raise ToolExecutionError("Invalid input")

            # Execute logic
            result = await self._execute_logic(kwargs)

            # Validate output
            if not self._validate_output(result):
                raise ToolExecutionError("Invalid output")

            return {"success": True, "data": result}

        except ToolExecutionError as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": "ToolExecutionError"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "error_type": type(e).__name__
            }

    def _validate_input(self, params: Dict) -> bool:
        """Validate input parameters."""
        # Validation logic
        return True

    def _validate_output(self, result: Any) -> bool:
        """Validate output."""
        return result is not None

    async def _execute_logic(self, params: Dict) -> Any:
        """Main tool logic."""
        # Implementation
        pass
```

### Tool with Retry Logic

```python
from victor.tools.base import BaseTool
from victor.framework.resilience import with_retry, ExponentialBackoffStrategy

class ResilientTool(BaseTool):
    """Tool with automatic retry logic."""

    name = "resilient_tool"
    description = "Tool that retries on failure"

    @with_retry(
        max_attempts=3,
        backoff_strategy=ExponentialBackoffStrategy(
            initial_delay=1.0,
            max_delay=10.0,
            exponent=2.0
        ),
        retry_on=(ConnectionError, TimeoutError)
    )
    async def execute(self, **kwargs):
        # This will automatically retry on specific errors
        result = await self._call_external_api(kwargs)
        return {"success": True, "data": result}

    async def _call_external_api(self, params: Dict) -> Any:
        # External API call that might fail
        pass
```

### Async Tool with Parallel Processing

```python
from victor.tools.base import BaseTool
import asyncio

class ParallelProcessingTool(BaseTool):
    """Tool that processes multiple items in parallel."""

    name = "parallel_tool"
    description = "Process multiple items in parallel"

    async def execute(self, items: list, **kwargs):
        # Process items in parallel
        tasks = [self._process_item(item) for item in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Separate successful and failed results
        successful = [r for r in results if not isinstance(r, Exception)]
        failed = [r for r in results if isinstance(r, Exception)]

        return {
            "success": True,
            "results": successful,
            "failed": len(failed),
            "total": len(results)
        }

    async def _process_item(self, item: Any) -> Any:
        """Process single item."""
        # Processing logic
        await asyncio.sleep(0.1)  # Simulate async work
        return f"processed_{item}"
```

## Tool Registration

### Method 1: Direct Registration

```python
# In your tool module
from victor.tools.base import BaseTool
from victor.core.tools import SharedToolRegistry

class MyTool(BaseTool):
    name = "my_tool"
    # ... implementation

# Register
registry = SharedToolRegistry.get_instance()
registry.register(MyTool())
```

### Method 2: Entry Points (Recommended for External Tools)

```toml
# pyproject.toml

[project]
name = "victor-custom-tools"
version = "0.5.0"

[project.entry-points."victor.tools"]
my_tool = "my_package.tools:MyTool"
another_tool = "my_package.tools:AnotherTool"
```

### Method 3: Dynamic Discovery

```python
# victor/tools/discovery.py

from importlib import import_module
from pathlib import Path

def discover_tools(package_path: str):
    """Discover and register tools in a package."""
    tools_dir = Path(package_path)

    for tool_file in tools_dir.glob("*_tool.py"):
        module = import_module(f"victor.tools.{tool_file.stem}")

        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, BaseTool):
                if attr != BaseTool:
                    registry.register(attr())
```

## Testing Tools

### Unit Test Example

```python
# tests/unit/tools/test_file_stats.py

import pytest
from pathlib import Path
from victor.tools.file_stats import FileStatsTool

@pytest.fixture
def file_stats_tool():
    """Create tool instance."""
    return FileStatsTool()

@pytest.fixture
def temp_file(tmp_path):
    """Create temporary test file."""
    test_file = tmp_path / "test.py"
    test_file.write_text("""
def hello():
    print("Hello, World!")

def add(a, b):
    return a + b
""")
    return test_file

@pytest.mark.asyncio
async def test_file_stats_on_file(file_stats_tool, temp_file):
    """Test file statistics on a single file."""
    result = await file_stats_tool.execute(path=str(temp_file))

    assert result["success"] is True
    assert result["type"] == "file"
    assert result["line_count"] == 7
    assert result["extension"] == ".py"
    assert result["is_text"] is True

@pytest.mark.asyncio
async def test_file_stats_on_nonexistent(file_stats_tool):
    """Test file stats on nonexistent path."""
    result = await file_stats_tool.execute(path="/nonexistent/file")

    assert result["success"] is False
    assert "not found" in result["error"].lower()

@pytest.mark.asyncio
async def test_file_stats_directory(file_stats_tool, tmp_path):
    """Test file statistics on directory."""
    # Create test files
    (tmp_path / "file1.py").write_text("print('hello')")
    (tmp_path / "file2.js").write_text("console.log('world')")

    result = await file_stats_tool.execute(
        path=str(tmp_path),
        recursive=False
    )

    assert result["success"] is True
    assert result["type"] == "directory"
    assert result["file_count"] == 2
    assert result["total_line_count"] == 2
```

### Integration Test Example

```python
# tests/integration/tools/test_tool_integration.py

import pytest
from victor import Agent

@pytest.mark.asyncio
async def test_tool_in_agent_context():
    """Test tool within agent execution."""
    agent = await Agent.create(provider="ollama")

    response = await agent.run(
        "Calculate statistics for the src/ directory"
    )

    assert "file_stats" in response.metadata.get("tools_used", [])
    assert "statistics" in response.content.lower()
```

## Best Practices

### 1. Parameter Validation

```python
from pydantic import BaseModel, validator

class ToolParameters(BaseModel):
    """Typed and validated parameters."""
    path: str
    max_depth: int = 10

    @validator('max_depth')
    def validate_depth(cls, v):
        if v < 1 or v > 100:
            raise ValueError('max_depth must be between 1 and 100')
        return v

class ValidatedTool(BaseTool):
    async def execute(self, **kwargs):
        # Validate parameters
        params = ToolParameters(**kwargs)
        # Use validated parameters
        pass
```

### 2. Comprehensive Error Messages

```python
async def execute(self, **kwargs):
    try:
        result = self._do_work(kwargs)
        return {"success": True, "result": result}
    except FileNotFoundError as e:
        return {
            "success": False,
            "error": f"File not found: {e.filename}",
            "suggestion": "Check the file path and try again"
        }
    except PermissionError:
        return {
            "success": False,
            "error": "Permission denied",
            "suggestion": "Check file permissions"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {type(e).__name__}: {str(e)}",
            "suggestion": "Contact support if issue persists"
        }
```

### 3. Progress Reporting

```python
from victor.protocols import IEventObserver

class ProgressReportingTool(BaseTool):
    """Tool that reports progress."""

    def __init__(self, event_observer: IEventObserver = None):
        super().__init__()
        self.event_observer = event_observer

    async def execute(self, items: list, **kwargs):
        total = len(items)
        for i, item in enumerate(items):
            # Process item
            result = await self._process_item(item)

            # Report progress
            if self.event_observer:
                await self.event_observer.on_event(
                    "tool.progress",
                    {
                        "tool": self.name,
                        "progress": (i + 1) / total,
                        "current": i + 1,
                        "total": total
                    }
                )

        return {"success": True, "processed": total}
```

### 4. Caching

```python
from functools import lru_cache
from hashlib import md5
import json

class CachedTool(BaseTool):
    """Tool with result caching."""

    def __init__(self):
        super().__init__()
        self._cache = {}

    async def execute(self, **kwargs):
        # Generate cache key
        cache_key = self._generate_cache_key(kwargs)

        # Check cache
        if cache_key in self._cache:
            return {
                "success": True,
                "result": self._cache[cache_key],
                "cached": True
            }

        # Execute and cache
        result = await self._execute_impl(kwargs)
        self._cache[cache_key] = result

        return {
            "success": True,
            "result": result,
            "cached": False
        }

    def _generate_cache_key(self, params: Dict) -> str:
        """Generate cache key from parameters."""
        params_str = json.dumps(params, sort_keys=True)
        return md5(params_str.encode()).hexdigest()
```

### 5. Logging

```python
import logging

class LoggingTool(BaseTool):
    """Tool with comprehensive logging."""

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

    async def execute(self, **kwargs):
        self.logger.info(f"Executing {self.name} with params: {kwargs}")

        try:
            result = await self._execute_impl(kwargs)
            self.logger.info(f"{self.name} completed successfully")
            return {"success": True, "result": result}

        except Exception as e:
            self.logger.error(f"{self.name} failed: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}
```

## Examples

### Example 1: API Client Tool

```python
import aiohttp
from victor.tools.base import BaseTool, CostTier

class APIClientTool(BaseTool):
    """Tool for making HTTP API requests."""

    name = "api_client"
    description = "Make HTTP requests to external APIs"
    cost_tier = CostTier.LOW

    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "API endpoint URL"},
            "method": {
                "type": "string",
                "enum": ["GET", "POST", "PUT", "DELETE"],
                "default": "GET"
            },
            "headers": {"type": "object", "description": "Request headers"},
            "body": {"type": "object", "description": "Request body"}
        },
        "required": ["url"]
    }

    async def execute(
        self,
        url: str,
        method: str = "GET",
        headers: dict = None,
        body: dict = None
    ):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method,
                    url,
                    headers=headers,
                    json=body
                ) as response:
                    return {
                        "success": True,
                        "status": response.status,
                        "data": await response.json()
                    }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
```

### Example 2: Database Query Tool

```python
import asyncpg
from victor.tools.base import BaseTool, CostTier

class DatabaseQueryTool(BaseTool):
    """Tool for executing database queries."""

    name = "database_query"
    description = "Execute SQL queries on PostgreSQL database"
    cost_tier = CostTier.LOW

    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "SQL query"},
            "params": {"type": "array", "description": "Query parameters"}
        },
        "required": ["query"]
    }

    def __init__(self, connection_string: str):
        super().__init__()
        self.connection_string = connection_string
        self._pool = None

    async def _get_pool(self):
        """Get or create connection pool."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(self.connection_string)
        return self._pool

    async def execute(self, query: str, params: list = None):
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            try:
                if query.strip().upper().startswith('SELECT'):
                    rows = await conn.fetch(query, *params)
                    return {
                        "success": True,
                        "rows": [dict(row) for row in rows],
                        "count": len(rows)
                    }
                else:
                    result = await conn.execute(query, *params)
                    return {
                        "success": True,
                        "result": result
                    }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e)
                }
```

## Conclusion

Creating custom tools for Victor AI allows you to extend its capabilities to fit your specific needs. Follow these patterns and best practices to create robust, reusable tools.

For more examples, see:
- `victor/tools/` - Built-in tools
- `examples/custom_plugin.py` - Example custom tool

Happy tool building! 🛠️
