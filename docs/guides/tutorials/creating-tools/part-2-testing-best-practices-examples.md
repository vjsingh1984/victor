# Creating Custom Tools for Victor AI - Part 2

**Part 2 of 2:** Testing Tools, Best Practices, Examples

---

## Navigation

- [Part 1: Tool Creation & Registration](part-1-introduction-advanced-features-registration.md)
- **[Part 2: Testing, Best Practices, Examples](#)** (Current)
- [**Complete Guide**](../CREATING_TOOLS.md)

---

## Testing Tools

### Unit Testing

Test your tool in isolation:

```python
import pytest
from victor.tools.base import ToolExecutionResult

def test_my_tool():
    """Test basic tool functionality."""
    tool = MyCustomTool()

    result = tool.execute(param1="value1", param2="value2")

    assert isinstance(result, ToolExecutionResult)
    assert result.success is True
    assert "output" in result.data
```

### Integration Testing

Test tool with real dependencies:

```python
@pytest.mark.integration
def test_my_tool_integration():
    """Test tool with real API calls."""
    tool = MyCustomTool()

    result = tool.execute(real_data=True)

    assert result.success is True
```

### Mock Testing

Mock external dependencies:

```python
from unittest.mock import Mock, patch

def test_my_tool_with_mock():
    """Test tool with mocked dependencies."""
    tool = MyCustomTool()

    with patch('my_tool.external_api') as mock_api:
        mock_api.return_value = {"data": "test"}
        result = tool.execute()

        assert result.success is True
        mock_api.assert_called_once()
```

---

## Best Practices

### 1. Clear Descriptions

Write clear, actionable descriptions:

```python
# Good
description = "Execute SQL query on PostgreSQL database and return results as JSON"

# Bad
description = "Run query"
```

### 2. Proper Error Handling

Handle errors gracefully:

```python
def execute(self, **kwargs) -> ToolExecutionResult:
    try:
        result = self._do_work(**kwargs)
        return ToolExecutionResult(
            success=True,
            data=result
        )
    except Exception as e:
        return ToolExecutionResult(
            success=False,
            error=str(e)
        )
```

### 3. Parameter Validation

Validate input parameters:

```python
def execute(self, url: str, **kwargs) -> ToolExecutionResult:
    if not url or not url.startswith(('http://', 'https://')):
        return ToolExecutionResult(
            success=False,
            error="Invalid URL format"
        )
    # Continue execution...
```

### 4. Cost Tiers

Assign appropriate cost tiers:

```python
# CostTier.FREE - No cost (local operations)
# CostTier.LOW - Minimal cost (simple API calls)
# CostTier.MEDIUM - Moderate cost (complex operations)
# CostTier.HIGH - High cost (expensive operations)

cost_tier = CostTier.LOW
```

---

## Examples

### Example 1: HTTP Request Tool

```python
import requests
from victor.tools.base import BaseTool, ToolExecutionResult, CostTier

class HttpRequestTool(BaseTool):
    """Make HTTP requests to external APIs."""

    name = "http_request"
    description = "Make HTTP GET/POST requests to external APIs"
    cost_tier = CostTier.LOW

    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to request"
            },
            "method": {
                "type": "string",
                "enum": ["GET", "POST", "PUT", "DELETE"],
                "description": "HTTP method"
            }
        },
        "required": ["url"]
    }

    def execute(self, url: str, method: str = "GET", **kwargs) -> ToolExecutionResult:
        try:
            response = requests.request(method, url, **kwargs)
            return ToolExecutionResult(
                success=response.status_code < 400,
                data={
                    "status_code": response.status_code,
                    "content": response.text
                }
            )
        except Exception as e:
            return ToolExecutionResult(
                success=False,
                error=str(e)
            )
```

### Example 2: Database Query Tool

```python
import sqlite3
from victor.tools.base import BaseTool, ToolExecutionResult, CostTier

class DatabaseQueryTool(BaseTool):
    """Execute SQL queries on SQLite database."""

    name = "database_query"
    description = "Execute SQL queries and return results"
    cost_tier = CostTier.MEDIUM

    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "SQL query to execute"
            },
            "database": {
                "type": "string",
                "description": "Path to SQLite database"
            }
        },
        "required": ["query", "database"]
    }

    def execute(self, query: str, database: str, **kwargs) -> ToolExecutionResult:
        try:
            conn = sqlite3.connect(database)
            cursor = conn.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            conn.close()

            return ToolExecutionResult(
                success=True,
                data={"results": results}
            )
        except Exception as e:
            return ToolExecutionResult(
                success=False,
                error=str(e)
            )
```

### Example 3: File Processing Tool

```python
from pathlib import Path
from victor.tools.base import BaseTool, ToolExecutionResult, CostTier

class FileProcessTool(BaseTool):
    """Process and analyze files."""

    name = "file_process"
    description = "Read, analyze, and process files"
    cost_tier = CostTier.FREE

    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to file"
            },
            "operation": {
                "type": "string",
                "enum": ["read", "analyze", "transform"],
                "description": "Operation to perform"
            }
        },
        "required": ["path", "operation"]
    }

    def execute(self, path: str, operation: str, **kwargs) -> ToolExecutionResult:
        try:
            file_path = Path(path)

            if operation == "read":
                content = file_path.read_text()
                return ToolExecutionResult(
                    success=True,
                    data={"content": content}
                )
            elif operation == "analyze":
                stats = {
                    "lines": len(file_path.read_text().splitlines()),
                    "size": file_path.stat().st_size
                }
                return ToolExecutionResult(
                    success=True,
                    data=stats
                )
        except Exception as e:
            return ToolExecutionResult(
                success=False,
                error=str(e)
            )
```

---

## Conclusion

Creating custom tools for Victor AI is straightforward. Follow the patterns in this guide to build powerful tools that extend Victor's capabilities.

**Next Steps:**
- Explore [Tool Examples](../../examples/tools/)
- Read [Tools API Reference](../../reference/internals/tools-api.md)
- Learn [Advanced Patterns](../ADVANCED_TOOL_PATTERNS.md)

---

**Last Updated:** February 01, 2026
