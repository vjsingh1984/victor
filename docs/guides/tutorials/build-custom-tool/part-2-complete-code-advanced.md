# Building Custom Tools in Victor - Part 2

**Part 2 of 2:** Complete Code Listing, Advanced Topics, Summary, Next Steps

---

## Navigation

- [Part 1: Building & Testing](part-1-building-testing.md)
- **[Part 2: Complete Code, Advanced Topics](#)** (Current)
- [**Complete Tutorial](../build-custom-tool.md)**

---

## 7. Complete Code Listing

### Weather Tool (Final Version)

```python
from victor.tools.decorators import tool
from victor.tools.base import CostTier
import httpx
from typing import Dict, Any, Optional
from datetime import datetime

@tool(
    name="weather",
    description="Fetch current weather conditions for any location",
    cost_tier=CostTier.LOW,
    timeout=10.0
)
async def weather(
    location: str,
    units: str = "metric",
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """Get current weather for a location.

    Args:
        location: City name or ZIP code (e.g., "London" or "90210")
        units: Temperature units ("metric" or "imperial")
        api_key: OpenWeatherMap API key (optional, uses env var by default)

    Returns:
        Weather data including temperature, conditions, humidity, wind speed

    Example:
        >>> await weather(location="London", units="metric")
        {
            "location": "London",
            "temperature": 15.5,
            "conditions": "partly cloudy",
            "humidity": 72,
            "wind_speed": 5.2
        }
    """
    # Implementation...
    pass
```

---

## 8. Advanced Topics

### Tool Composition

Combine multiple tools:

```python
@tool
async def weather_forecast(location: str, days: int = 5):
    """Get weather forecast."""
    forecast = await fetch_forecast(location, days)
    return forecast

@tool
async def weather_alerts(location: str):
    """Get weather alerts."""
    alerts = await fetch_alerts(location)
    return alerts

# Compose in a workflow
async def get_weather_summary(location: str):
    """Get complete weather summary."""
    current = await weather(location)
    forecast = await weather_forecast(location)
    alerts = await weather_alerts(location)

    return {
        "current": current,
        "forecast": forecast,
        "alerts": alerts
    }
```

### Tool Chaining

Chain tools together:

```python
from victor.tools import chain_tools

@tool
async def process_weather_data(location: str):
    """Process and format weather data."""
    # Step 1: Get raw data
    raw = await fetch_raw_weather(location)

    # Step 2: Parse data
    parsed = parse_weather_data(raw)

    # Step 3: Format output
    formatted = format_weather(parsed)

    return formatted
```

### Batch Operations

Execute tools in batch:

```python
from victor.tools import batch_execute

async def get_weather_for_locations(locations: List[str]):
    """Get weather for multiple locations."""
    tools = [
        ("weather", {"location": loc})
        for loc in locations
    ]

    results = await batch_execute(tools)
    return results
```

---

## Summary

In this tutorial, you learned:

1. ✅ Tool structure and anatomy
2. ✅ Step-by-step implementation
3. ✅ Parameter validation
4. ✅ Error handling
5. ✅ Registration and discovery
6. ✅ Testing strategies
7. ✅ Advanced patterns

**Key Takeaways:**
- Use `@tool` decorator for simple tools
- Inherit `BaseTool` for complex tools
- Always validate parameters
- Handle errors gracefully
- Write comprehensive tests
- Use type hints for auto-documentation

---

## Next Steps

1. **Explore More Tools**: Check out [existing tools](../../../reference/tools/README.md)
2. **Build Complex Tools**: Learn [advanced patterns](../ADVANCED_TOOL_PATTERNS.md)
3. **Integrate with Verticals**: See [vertical integration](../../reference/extensions/step_handler_examples.md)
4. **Contribute Your Tools**: Share tools with the community

---

**Reading Time:** 2 min
**Last Updated:** February 01, 2026
**Time to Complete:** 30 minutes
