# Building Custom Tools in Victor

Tutorial for building custom tools that extend Victor's capabilities.

---

## Quick Summary

Learn how to create custom tools that extend Victor's capabilities. This tutorial walks you through building a weather
  fetching tool from scratch.

**What You'll Build:**
- A `weather` tool that fetches current weather conditions
- Demonstrates tool structure, validation, async execution, error handling, and testing

**Time Estimate:** 30 minutes

**Prerequisites:**
- Python 3.10+
- Victor installed (`pip install -e ".[dev]"`)
- Basic async/await understanding
- Familiarity with JSON Schema

---

## Tutorial Parts

### [Part 1: Building & Testing](part-1-building-testing.md)
- Tool Anatomy (Decorator vs Class-based)
- Step-by-Step Implementation
- Parameter Validation
- Error Handling
- Registration
- Testing the Tool

### [Part 2: Complete Code, Advanced Topics](part-2-complete-code-advanced.md)
- Complete Code Listing
- Advanced Topics (Composition, Chaining, Batch Operations)
- Summary
- Next Steps

---

## Quick Start

```python
from victor.tools.decorators import tool
from victor.tools.base import CostTier

@tool(
    name="weather",
    description="Fetch current weather conditions",
    cost_tier=CostTier.LOW
)
async def weather(location: str, units: str = "metric") -> Dict[str, Any]:
    """Get current weather for a location.

    Args:
        location: City name or ZIP code
        units: Temperature units ("metric" or "imperial")

    Returns:
        Weather data including temperature, conditions, humidity
    """
    # Implementation...
    pass
```

---

## Related Documentation

- [Creating Tools Guide](../CREATING_TOOLS.md)
- [Tools API Reference](../../../reference/internals/tools-api.md)
- [Advanced Tool Patterns](../ADVANCED_TOOL_PATTERNS.md)

---

**Last Updated:** February 01, 2026
**Reading Time:** 30 min (all parts)
