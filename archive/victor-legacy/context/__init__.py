"""Context management for Victor.

Handles:
- Token counting
- Context window budgeting
- Automatic context pruning
- Smart file selection
- Message prioritization
"""

from victor.context.manager import ContextManager, ContextWindow, PruningStrategy

__all__ = ["ContextManager", "ContextWindow", "PruningStrategy"]
