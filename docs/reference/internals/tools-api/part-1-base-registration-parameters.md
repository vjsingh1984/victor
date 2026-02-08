# Victor Tools API Reference - Part 1

**Part 1 of 2:** BaseTool Interface, Tool Registration, Parameters, Cost Tiers, Access Modes, Implementation Patterns, and Decorators

---

## Navigation

- **[Part 1: Core API](#)** (Current)
- [Part 2: Advanced Features & Examples](part-2-advanced-features-examples.md)
- [**Complete Reference**](../tools-api.md)

---

This document provides comprehensive API documentation for creating and registering tools in Victor.

## Table of Contents

- [BaseTool Interface](#basetool-interface)
- [Tool Registration](#tool-registration)
- [Parameter Schema](#parameter-schema)
- [Cost Tiers](#cost-tiers)
- [Access Modes](#access-modes)
- [Tool Implementation Patterns](#tool-implementation-patterns)
- [Tool Decorator](#tool-decorator)
- [Advanced Features](#advanced-features) *(in Part 2)*

---

## BaseTool Interface

All Victor tools inherit from the `BaseTool` abstract base class defined in `victor/tools/base.py`.

### Required Attributes

```python
from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseTool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name - unique identifier used for tool calls."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description - explains what the tool does."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """JSON Schema for tool parameters."""
        pass
```

### Optional Attributes

[Content continues through Tool Decorator...]

---

**Continue to [Part 2: Advanced Features & Examples](part-2-advanced-features-examples.md)**
