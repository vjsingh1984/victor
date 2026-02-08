# Victor Tools API Reference - Part 2

**Part 2 of 2:** Advanced Features and Complete Example

---

## Navigation

- [Part 1: Core API](part-1-base-registration-parameters.md)
- **[Part 2: Advanced & Examples](#)** (Current)
- [**Complete Reference**](../tools-api.md)

---

## Advanced Features

### Tool Dependencies

Tools can declare dependencies on other tools:

```python
class DependentTool(BaseTool):
    @property
    def dependencies(self) -> List[str]:
        """List of tools this tool depends on."""
        return ["read", "write"]

    @property
    def name(self) -> str:
        return "dependent_tool"
```

### Async Tool Execution

Tools support async execution:

```python
class AsyncTool(BaseTool):
    async def execute(
        self,
        **kwargs
    ) -> Any:
        """Async tool execution."""
        result = await some_async_operation()
        return result
```

### Streaming Results

Tools can stream results:

```python
class StreamingTool(BaseTool):
    def execute(
        self,
        **kwargs
    ) -> Iterator[Any]:
        """Stream results incrementally."""
        for item in large_dataset:
            yield process(item)
```

---

## Example: Complete Tool Implementation

```python
from victor.tools.base import BaseTool
from typing import Any, Dict, List

class CodeAnalysisTool(BaseTool):
    """Analyze Python code for quality issues."""

    @property
    def name(self) -> str:
        return "analyze_code"

    @property
    def description(self) -> str:
        return "Analyze Python code for quality and security issues"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to analyze"
                },
                "severity": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "Minimum severity level"
                }
            },
            "required": ["code"]
        }

    @property
    def cost_tier(self) -> int:
        return 2  # Medium computational cost

    @property
    def access_mode(self) -> str:
        return "read"  # Read-only access

    def execute(
        self,
        code: str,
        severity: str = "medium",
        **kwargs
    ) -> Dict[str, Any]:
        """Execute code analysis."""
        issues = []

        # Analyze code
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            if 'TODO' in line and severity != 'low':
                issues.append({
                    'line': i,
                    'severity': 'low',
                    'message': 'TODO comment found'
                })

        return {
            'issues': issues,
            'total_issues': len(issues)
        }
```

---

## See Also

- [Creating Tools Guide](../../../guides/tutorials/CREATING_TOOLS.md)
- [Tool Registry](../../tools/README.md)
- [Examples](../../../examples/tools/)

---

**Last Updated:** February 01, 2026
