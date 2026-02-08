# Building Custom Tools in Victor - Part 1

**Part 1 of 2:** Tool Anatomy, Step-by-Step Implementation, Parameter Validation, Error Handling, Registration, and Testing

---

## Navigation

- **[Part 1: Building & Testing](#)** (Current)
- [Part 2: Complete Code, Advanced Topics](part-2-complete-code-advanced.md)
- [**Complete Tutorial](../build-custom-tool.md)**

---

Learn how to create custom tools that extend Victor's capabilities. This tutorial walks you through building a weather fetching tool from scratch.

## What You'll Build

A `weather` tool that fetches current weather conditions for any location. This example demonstrates:
- Tool structure and required attributes
- Parameter validation with JSON Schema
- Async execution
- Error handling
- Testing patterns

## Prerequisites

- Python 3.10+
- Victor installed (`pip install -e ".[dev]"`)
- Basic understanding of async/await in Python
- Familiarity with JSON Schema

## Time Estimate

30 minutes

---

## 1. Tool Anatomy

Victor provides two approaches for creating tools:

1. **Decorator-based** (`@tool`) - Recommended for most cases
2. **Class-based** (`BaseTool`) - For complex tools needing fine-grained control

### The @tool Decorator

The decorator automatically extracts metadata from your function's signature and docstring:

```python
from victor.tools.decorators import tool
from victor.tools.base import CostTier

@tool(cost_tier=CostTier.MEDIUM)
async def my_tool(param1: str, param2: int = 5) -> Dict[str, Any]:
    """Short description of what the tool does.

    Detailed description goes here. This becomes part of the
    tool's documentation shown to the LLM.

    Args:
        param1: Description of the first parameter.
```

[Content continues through Testing the Tool...]

---

**Continue to [Part 2: Complete Code, Advanced Topics](part-2-complete-code-advanced.md)**
