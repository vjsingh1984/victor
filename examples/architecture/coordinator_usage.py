#!/usr/bin/env python
"""Coordinator Pattern Usage Example.

This example shows how to:
1. Create custom coordinators using the coordinator pattern
2. Use delegation to distribute responsibilities
3. Integrate coordinators with the orchestrator
4. Compose coordinators for complex operations

Before Refactoring (God Object):
    class AgentOrchestrator:
        def __init__(self):
            # 100+ lines of initialization
            self.tool_selection = ...
            self.budget_management = ...
            self.caching = ...
            self.analytics = ...
            # ... many more concerns

        def select_tools(self):
            # 200+ lines of tool selection logic

        def manage_budget(self):
            # 150+ lines of budget management

        # ... 50+ methods, 3000+ lines of code

After Refactoring (Coordinator Pattern):
    class AgentOrchestrator:
        def __init__(self):
            self.tool_coordinator = ToolCoordinator(...)
            self.cache_coordinator = CacheCoordinator(...)

        def select_tools(self):
            return self.tool_coordinator.select_tools(...)  # Delegate

    # Each coordinator is focused, testable, and reusable
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# PART 1: Define Coordinator Protocols
# =============================================================================


class ICoordinator(ABC):
    """Base protocol for all coordinators.

    Coordinators are responsible for coordinating a specific aspect
    of the system (tools, caching, analytics, etc.).

    Design Principles:
        - Single Responsibility: Each coordinator handles one concern
        - Delegation: Orchestrator delegates to coordinators
        - Composition: Coordinators can be composed together
    """

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the coordinator."""
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get coordinator status."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset coordinator state."""
        pass


class ICacheCoordinator(ICoordinator):
    """Protocol for cache coordination."""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        pass

    @abstractmethod
    def invalidate(self, key: str) -> None:
        """Invalidate cache entry."""
        pass

    @abstractmethod
    def clear_all(self) -> None:
        """Clear all cache entries."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass


class IToolCoordinator(ICoordinator):
    """Protocol for tool coordination."""

    @abstractmethod
    def select_tools(self, task: str, limit: int) -> List[str]:
        """Select tools for a task."""
        pass

    @abstractmethod
    def check_budget(self) -> bool:
        """Check if budget allows more tool calls."""
        pass

    @abstractmethod
    def consume_budget(self, amount: int = 1) -> None:
        """Consume tool call budget."""
        pass

    @abstractmethod
    def get_remaining_budget(self) -> int:
        """Get remaining budget."""
        pass


class IAnalyticsCoordinator(ICoordinator):
    """Protocol for analytics coordination."""

    @abstractmethod
    def record_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Record an analytics event."""
        pass

    @abstractmethod
    def flush(self) -> None:
        """Flush analytics to storage."""
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get analytics metrics."""
        pass


# =============================================================================
# PART 2: Implement Concrete Coordinators
# =============================================================================


class CacheCoordinator(ICacheCoordinator):
    """Coordinates caching operations.

    This coordinator handles all cache-related concerns:
    - Cache storage (in-memory, Redis, etc.)
    - TTL management
    - Cache invalidation
    - Statistics and monitoring
    """

    def __init__(self, default_ttl: int = 300):
        self.default_ttl = default_ttl
        self._cache: Dict[str, tuple[Any, float]] = {}  # key -> (value, expiry)
        self._hits = 0
        self._misses = 0
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the cache coordinator."""
        logger.info("Initializing CacheCoordinator...")
        self._initialized = True
        logger.info(f"CacheCoordinator ready (TTL={self.default_ttl}s)")

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self._initialized:
            logger.warning("CacheCoordinator not initialized")
            return None

        if key not in self._cache:
            self._misses += 1
            return None

        value, expiry = self._cache[key]

        # Check if expired
        if time.time() > expiry:
            del self._cache[key]
            self._misses += 1
            return None

        self._hits += 1
        logger.debug(f"Cache hit: {key}")
        return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        if not self._initialized:
            logger.warning("CacheCoordinator not initialized")
            return

        ttl = ttl or self.default_ttl
        expiry = time.time() + ttl
        self._cache[key] = (value, expiry)
        logger.debug(f"Cache set: {key} (TTL={ttl}s)")

    def invalidate(self, key: str) -> None:
        """Invalidate cache entry."""
        if key in self._cache:
            del self._cache[key]
            logger.debug(f"Cache invalidated: {key}")

    def clear_all(self) -> None:
        """Clear all cache entries."""
        count = len(self._cache)
        self._cache.clear()
        logger.info(f"Cleared {count} cache entries")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0

        return {
            "initialized": self._initialized,
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "default_ttl": self.default_ttl,
        }

    def get_status(self) -> Dict[str, Any]:
        """Get coordinator status."""
        return self.get_stats()

    def reset(self) -> None:
        """Reset coordinator state."""
        self.clear_all()
        self._hits = 0
        self._misses = 0
        logger.info("CacheCoordinator reset")


class ToolCoordinator(IToolCoordinator):
    """Coordinates tool selection and budgeting.

    This coordinator handles:
    - Tool selection logic
    - Budget management
    - Tool access control
    """

    def __init__(self, initial_budget: int = 25):
        self.initial_budget = initial_budget
        self._budget = initial_budget
        self._tools = {
            "code_search": "Search codebase with queries",
            "file_edit": "Edit files",
            "test_gen": "Generate tests",
            "refactor": "Refactor code",
            "debug": "Debug errors",
        }
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the tool coordinator."""
        logger.info("Initializing ToolCoordinator...")
        self._initialized = True
        logger.info(f"ToolCoordinator ready (budget={self._budget})")

    def select_tools(self, task: str, limit: int = 10) -> List[str]:
        """Select tools for a task."""
        if not self._initialized:
            logger.warning("ToolCoordinator not initialized")
            return []

        # Simple keyword matching
        task_lower = task.lower()
        tool_scores = []

        for tool_name, description in self._tools.items():
            score = self._compute_score(task_lower, description)
            tool_scores.append((tool_name, score))

        # Sort by score and return top N
        tool_scores.sort(key=lambda x: x[1], reverse=True)
        selected = [name for name, _ in tool_scores[:limit]]

        logger.info(f"Selected {len(selected)} tools for task")
        return selected

    def _compute_score(self, task: str, description: str) -> float:
        """Compute relevance score."""
        task_words = set(task.split())
        desc_words = set(description.lower().split())
        overlap = len(task_words & desc_words)
        return min(overlap / max(len(task_words), 1), 1.0)

    def check_budget(self) -> bool:
        """Check if budget allows more tool calls."""
        return self._budget > 0

    def consume_budget(self, amount: int = 1) -> None:
        """Consume tool call budget."""
        self._budget = max(0, self._budget - amount)
        logger.debug(f"Budget consumed: {amount}, remaining: {self._budget}")

    def get_remaining_budget(self) -> int:
        """Get remaining budget."""
        return self._budget

    def get_status(self) -> Dict[str, Any]:
        """Get coordinator status."""
        return {
            "initialized": self._initialized,
            "budget": self._budget,
            "initial_budget": self.initial_budget,
            "available_tools": len(self._tools),
        }

    def reset(self) -> None:
        """Reset coordinator state."""
        self._budget = self.initial_budget
        logger.info("ToolCoordinator reset")


class AnalyticsCoordinator(IAnalyticsCoordinator):
    """Coordinates analytics collection and export.

    This coordinator handles:
    - Event collection
    - Metrics aggregation
    - Export to storage/backends
    """

    def __init__(self, flush_interval: int = 60):
        self.flush_interval = flush_interval
        self._events: List[Dict[str, Any]] = []
        self._metrics: Dict[str, List[float]] = {}
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the analytics coordinator."""
        logger.info("Initializing AnalyticsCoordinator...")
        self._initialized = True
        logger.info(f"AnalyticsCoordinator ready (flush_interval={self.flush_interval}s)")

    def record_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Record an analytics event."""
        if not self._initialized:
            logger.warning("AnalyticsCoordinator not initialized")
            return

        event = {
            "type": event_type,
            "data": data,
            "timestamp": time.time(),
        }
        self._events.append(event)
        logger.debug(f"Recorded event: {event_type}")

    def flush(self) -> None:
        """Flush analytics to storage."""
        if not self._events:
            logger.info("No events to flush")
            return

        count = len(self._events)
        logger.info(f"Flushing {count} events to storage...")

        # Simulate flush
        time.sleep(0.1)

        self._events.clear()
        logger.info("Flush complete")

    def record_metric(self, name: str, value: float) -> None:
        """Record a metric value."""
        if name not in self._metrics:
            self._metrics[name] = []
        self._metrics[name].append(value)

    def get_metrics(self) -> Dict[str, Any]:
        """Get analytics metrics."""
        summary = {}
        for name, values in self._metrics.items():
            summary[name] = {
                "count": len(values),
                "avg": sum(values) / len(values) if values else 0.0,
                "min": min(values) if values else 0.0,
                "max": max(values) if values else 0.0,
            }
        return summary

    def get_status(self) -> Dict[str, Any]:
        """Get coordinator status."""
        return {
            "initialized": self._initialized,
            "pending_events": len(self._events),
            "metrics_count": len(self._metrics),
            "flush_interval": self.flush_interval,
        }

    def reset(self) -> None:
        """Reset coordinator state."""
        self._events.clear()
        self._metrics.clear()
        logger.info("AnalyticsCoordinator reset")


# =============================================================================
# PART 3: Orchestrator using Coordinators (Delegation Pattern)
# =============================================================================


class SimpleOrchestrator:
    """Simplified orchestrator demonstrating coordinator delegation.

    Instead of handling all concerns directly, the orchestrator delegates
    to specialized coordinators. This makes the orchestrator simpler and
    each coordinator more focused and testable.
    """

    def __init__(
        self,
        cache_coordinator: ICacheCoordinator,
        tool_coordinator: IToolCoordinator,
        analytics_coordinator: IAnalyticsCoordinator,
    ):
        """Initialize with coordinators injected via constructor.

        Args:
            cache_coordinator: Handles caching concerns
            tool_coordinator: Handles tool selection and budgeting
            analytics_coordinator: Handles analytics and metrics
        """
        self.cache = cache_coordinator
        self.tools = tool_coordinator
        self.analytics = analytics_coordinator

    def initialize(self) -> None:
        """Initialize all coordinators."""
        logger.info("Initializing orchestrator...")
        self.cache.initialize()
        self.tools.initialize()
        self.analytics.initialize()
        logger.info("Orchestrator ready")

    def execute_task(self, task: str) -> Dict[str, Any]:
        """Execute a task using coordinators.

        This method demonstrates how the orchestrator delegates to
        coordinators instead of implementing logic directly.
        """
        logger.info(f"\nExecuting task: {task}")

        # Check cache
        cache_key = f"task:{hash(task)}"
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            logger.info("Returning cached result")
            return cached_result

        # Check tool budget
        if not self.tools.check_budget():
            logger.warning("Tool budget exhausted")
            return {"error": "Budget exhausted"}

        # Select tools
        start_time = time.time()
        selected_tools = self.tools.select_tools(task, limit=3)
        self.tools.consume_budget(len(selected_tools))

        # Simulate task execution
        logger.info(f"Executing with tools: {selected_tools}")
        time.sleep(0.1)  # Simulate work

        result = {
            "task": task,
            "tools_used": selected_tools,
            "status": "success",
            "execution_time": time.time() - start_time,
        }

        # Cache result
        self.cache.set(cache_key, result, ttl=60)

        # Record analytics
        self.analytics.record_event(
            "task_completed",
            {
                "task": task,
                "tools": selected_tools,
                "duration": result["execution_time"],
            },
        )
        self.analytics.record_metric("task.duration", result["execution_time"])

        return result

    def get_status(self) -> Dict[str, Any]:
        """Get status of all coordinators."""
        return {
            "cache": self.cache.get_status(),
            "tools": self.tools.get_status(),
            "analytics": self.analytics.get_status(),
        }

    def shutdown(self) -> None:
        """Shutdown all coordinators."""
        logger.info("Shutting down orchestrator...")
        self.analytics.flush()
        self.cache.reset()
        self.tools.reset()
        logger.info("Orchestrator shutdown complete")


# =============================================================================
# PART 4: Composite Coordinator (Composition Pattern)
# =============================================================================


class CompositeCoordinator(ICoordinator):
    """Composes multiple coordinators together.

    This demonstrates how coordinators can be composed to create
    more complex coordinators while maintaining single responsibility.
    """

    def __init__(self, coordinators: List[ICoordinator]):
        self._coordinators = coordinators

    def initialize(self) -> None:
        """Initialize all child coordinators."""
        for coordinator in self._coordinators:
            coordinator.initialize()

    def get_status(self) -> Dict[str, Any]:
        """Get status of all child coordinators."""
        status = {}
        for i, coordinator in enumerate(self._coordinators):
            status[f"coordinator_{i}"] = coordinator.get_status()
        return status

    def reset(self) -> None:
        """Reset all child coordinators."""
        for coordinator in self._coordinators:
            coordinator.reset()


# =============================================================================
# PART 5: Demonstration Functions
# =============================================================================


def demonstrate_basic_delegation():
    """Demonstrate basic coordinator delegation."""
    logger.info("\n" + "=" * 70)
    logger.info("DEMONSTRATION: Basic Delegation")
    logger.info("=" * 70)

    # Create coordinators
    cache = CacheCoordinator(default_ttl=60)
    tools = ToolCoordinator(initial_budget=10)
    analytics = AnalyticsCoordinator(flush_interval=30)

    # Create orchestrator
    orchestrator = SimpleOrchestrator(cache, tools, analytics)
    orchestrator.initialize()

    # Execute tasks
    logger.info("\n--- Executing tasks ---")
    result1 = orchestrator.execute_task("search and refactor code")
    logger.info(f"Result: {result1}")

    result2 = orchestrator.execute_task("search and refactor code")  # Should hit cache
    logger.info(f"Result (cached): {result2}")

    # Show status
    logger.info("\n--- Coordinator Status ---")
    status = orchestrator.get_status()
    for name, stats in status.items():
        logger.info(f"{name.capitalize()}:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

    orchestrator.shutdown()


def demonstrate_coordinator_composition():
    """Demonstrate coordinator composition."""
    logger.info("\n" + "=" * 70)
    logger.info("DEMONSTRATION: Coordinator Composition")
    logger.info("=" * 70)

    # Create coordinators
    cache = CacheCoordinator(default_ttl=60)
    tools = ToolCoordinator(initial_budget=10)
    analytics = AnalyticsCoordinator(flush_interval=30)

    # Compose coordinators
    composite = CompositeCoordinator([cache, tools, analytics])
    composite.initialize()

    logger.info("\n--- Composite Coordinator Status ---")
    status = composite.get_status()
    for key, value in status.items():
        logger.info(f"{key}: {value}")

    composite.reset()


def demonstrate_testing_with_mocks():
    """Demonstrate testing with mock coordinators."""
    logger.info("\n" + "=" * 70)
    logger.info("DEMONSTRATION: Testing with Mock Coordinators")
    logger.info("=" * 70)

    # Mock cache coordinator
    class MockCacheCoordinator(ICacheCoordinator):
        def __init__(self):
            self._cache = {}

        def initialize(self) -> None:
            logger.info("MockCache initialized")

        def get(self, key: str) -> Optional[Any]:
            return self._cache.get(key)

        def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
            self._cache[key] = value

        def invalidate(self, key: str) -> None:
            if key in self._cache:
                del self._cache[key]

        def clear_all(self) -> None:
            self._cache.clear()

        def get_stats(self) -> Dict[str, Any]:
            return {"size": len(self._cache)}

        def get_status(self) -> Dict[str, Any]:
            return self.get_stats()

        def reset(self) -> None:
            self._cache.clear()

    # Create orchestrator with mock cache
    cache = MockCacheCoordinator()
    tools = ToolCoordinator(initial_budget=5)
    analytics = AnalyticsCoordinator()

    orchestrator = SimpleOrchestrator(cache, tools, analytics)
    orchestrator.initialize()

    # Execute task
    result = orchestrator.execute_task("test task")
    logger.info(f"\n✓ Task executed with mock cache")
    logger.info(f"✓ No I/O operations performed")
    logger.info(f"✓ Fast and isolated test")


def main():
    """Run all examples."""
    logger.info("\n" + "=" * 70)
    logger.info("COORDINATOR PATTERN USAGE")
    logger.info("=" * 70)

    # Basic delegation
    demonstrate_basic_delegation()

    # Coordinator composition
    demonstrate_coordinator_composition()

    # Testing with mocks
    demonstrate_testing_with_mocks()

    logger.info("\n" + "=" * 70)
    logger.info("KEY TAKEAWAYS")
    logger.info("=" * 70)
    logger.info("1. Coordinators encapsulate specific concerns")
    logger.info("2. Orchestrator delegates to coordinators")
    logger.info("3. Each coordinator is focused and testable")
    logger.info("4. Coordinators can be composed together")
    logger.info("5. Easy to mock coordinators for testing")
    logger.info("6. Reduces orchestrator complexity")
    logger.info("\nRun with: python -m examples.architecture.coordinator_usage")


if __name__ == "__main__":
    main()
