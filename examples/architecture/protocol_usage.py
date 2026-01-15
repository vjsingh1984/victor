#!/usr/bin/env python
"""Protocol Usage Example - Demonstrates protocol-based design patterns.

This example shows how to:
1. Define custom protocols for loose coupling
2. Implement protocols in concrete classes
3. Use protocols for dependency injection
4. Test with protocol mocks

Before Refactoring (Tight Coupling):
    class ToolSelector:
        def __init__(self, semantic_selector: SemanticToolSelector):
            self.semantic = semantic_selector  # Concrete dependency

    # Hard to test, hard to swap implementations

After Refactoring (Protocol-Based Design):
    class ToolSelector:
        def __init__(self, selector: IToolSelector):
            self.selector = selector  # Protocol dependency

    # Easy to test with mocks, easy to swap implementations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# PART 1: Define Protocol
# =============================================================================


@runtime_checkable
class IToolSelector(Protocol):
    """Protocol for tool selection implementations.

    This protocol defines the minimal interface that all tool selectors
    must implement. Following ISP (Interface Segregation Principle) by
    defining only the methods needed by consumers.

    Benefits:
        - DIP: Consumers depend on abstraction, not concretions
        - OCP: New selectors can be added without modifying consumers
        - Testability: Easy to create test doubles
    """

    def select_tools(self, task: str, limit: int = 10) -> List[str]:
        """Select relevant tools for a task.

        Args:
            task: Task description to match tools against
            limit: Maximum number of tools to return

        Returns:
            List of tool names ordered by relevance
        """
        ...

    def get_tool_score(self, tool_name: str, task: str) -> float:
        """Get relevance score for a specific tool.

        Args:
            tool_name: Name of the tool to score
            task: Task description to score against

        Returns:
            Relevance score from 0.0 (not relevant) to 1.0 (highly relevant)
        """
        ...

    @property
    def strategy_name(self) -> str:
        """Get the name of the selection strategy."""
        ...


# =============================================================================
# PART 2: Implement Protocol
# =============================================================================


class SemanticToolSelector:
    """Semantic-based tool selector using embeddings."""

    def __init__(self, embedding_model: str = "default"):
        self.embedding_model = embedding_model
        self._tools = {
            "code_search": "Search codebase with semantic queries",
            "file_edit": "Edit files with precision",
            "test_gen": "Generate unit tests",
            "refactor": "Refactor code for quality",
            "debug": "Debug and fix errors",
        }

    def select_tools(self, task: str, limit: int = 10) -> List[str]:
        """Select tools using semantic similarity."""
        logger.info(f"SemanticToolSelector selecting tools for: {task}")

        # Simulate semantic matching
        task_lower = task.lower()
        scores = {
            name: self._compute_similarity(task_lower, desc)
            for name, desc in self._tools.items()
        }

        # Sort by score and return top N
        sorted_tools = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [name for name, _ in sorted_tools[:limit]]

    def get_tool_score(self, tool_name: str, task: str) -> float:
        """Get semantic similarity score."""
        if tool_name not in self._tools:
            return 0.0
        return self._compute_similarity(task.lower(), self._tools[tool_name])

    def _compute_similarity(self, task: str, description: str) -> float:
        """Simulate semantic similarity computation."""
        # Simple keyword overlap as proxy for embeddings
        task_words = set(task.split())
        desc_words = set(description.lower().split())
        overlap = len(task_words & desc_words)
        return min(overlap / max(len(task_words), 1), 1.0)

    @property
    def strategy_name(self) -> str:
        return "semantic"


class KeywordToolSelector:
    """Keyword-based tool selector."""

    def __init__(self):
        self._tool_keywords = {
            "code_search": ["search", "find", "locate", "query"],
            "file_edit": ["edit", "modify", "change", "update"],
            "test_gen": ["test", "unit", "coverage"],
            "refactor": ["refactor", "clean", "improve"],
            "debug": ["debug", "fix", "error", "bug"],
        }

    def select_tools(self, task: str, limit: int = 10) -> List[str]:
        """Select tools using keyword matching."""
        logger.info(f"KeywordToolSelector selecting tools for: {task}")

        task_lower = task.lower()
        scores = {
            name: self._keyword_score(task_lower, keywords)
            for name, keywords in self._tool_keywords.items()
        }

        sorted_tools = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [name for name, _ in sorted_tools[:limit]]

    def get_tool_score(self, tool_name: str, task: str) -> float:
        """Get keyword match score."""
        if tool_name not in self._tool_keywords:
            return 0.0
        return self._keyword_score(
            task.lower(), self._tool_keywords[tool_name]
        )

    def _keyword_score(self, task: str, keywords: List[str]) -> float:
        """Compute keyword match score."""
        task_words = set(task.split())
        keyword_set = set(keywords)
        matches = len(task_words & keyword_set)
        return matches / len(keyword_set) if keyword_set else 0.0

    @property
    def strategy_name(self) -> str:
        return "keyword"


class HybridToolSelector:
    """Hybrid selector combining semantic and keyword approaches."""

    def __init__(
        self,
        semantic_selector: SemanticToolSelector,
        keyword_selector: KeywordToolSelector,
        semantic_weight: float = 0.7,
    ):
        self.semantic = semantic_selector
        self.keyword = keyword_selector
        self.semantic_weight = semantic_weight

    def select_tools(self, task: str, limit: int = 10) -> List[str]:
        """Select tools using hybrid approach."""
        logger.info(f"HybridToolSelector selecting tools for: {task}")

        # Get scores from both selectors
        semantic_tools = self.semantic.select_tools(task, limit=10)
        keyword_tools = self.keyword.select_tools(task, limit=10)

        # Combine scores
        all_tools = set(semantic_tools + keyword_tools)
        combined_scores = {}

        for tool in all_tools:
            semantic_score = self.semantic.get_tool_score(tool, task)
            keyword_score = self.keyword.get_tool_score(tool, task)
            combined_scores[tool] = (
                self.semantic_weight * semantic_score
                + (1 - self.semantic_weight) * keyword_score
            )

        sorted_tools = sorted(
            combined_scores.items(), key=lambda x: x[1], reverse=True
        )
        return [name for name, _ in sorted_tools[:limit]]

    def get_tool_score(self, tool_name: str, task: str) -> float:
        """Get combined relevance score."""
        semantic_score = self.semantic.get_tool_score(tool_name, task)
        keyword_score = self.keyword.get_tool_score(tool_name, task)
        return (
            self.semantic_weight * semantic_score
            + (1 - self.semantic_weight) * keyword_score
        )

    @property
    def strategy_name(self) -> str:
        return "hybrid"


# =============================================================================
# PART 3: Use Protocol for Dependency Injection
# =============================================================================


@dataclass
class ToolCoordinator:
    """Coordinates tool selection and execution.

    This class depends on the IToolSelector protocol, not any concrete
    implementation. This makes it flexible and testable.
    """

    selector: IToolSelector
    max_tools: int = 10

    def select_and_execute(self, task: str) -> Dict[str, Any]:
        """Select tools for task and simulate execution.

        Args:
            task: Task description

        Returns:
            Dictionary with selected tools and metadata
        """
        logger.info(f"ToolCoordinator using strategy: {self.selector.strategy_name}")

        # Select tools using the protocol interface
        selected_tools = self.selector.select_tools(task, limit=self.max_tools)

        # Get scores for selected tools
        tool_scores = {
            tool: self.selector.get_tool_score(tool, task)
            for tool in selected_tools
        }

        return {
            "task": task,
            "selected_tools": selected_tools,
            "scores": tool_scores,
            "strategy": self.selector.strategy_name,
        }


# =============================================================================
# PART 4: Test with Protocol Mocks
# =============================================================================


class MockToolSelector:
    """Mock selector for testing.

    Demonstrates how protocols enable easy testing without
    depending on real implementations.
    """

    def __init__(self, fixed_tools: List[str]):
        self._fixed_tools = fixed_tools

    def select_tools(self, task: str, limit: int = 10) -> List[str]:
        """Return fixed list of tools."""
        return self._fixed_tools[:limit]

    def get_tool_score(self, tool_name: str, task: str) -> float:
        """Return fixed score."""
        return 1.0 if tool_name in self._fixed_tools else 0.0

    @property
    def strategy_name(self) -> str:
        return "mock"


def test_coordinator_with_mock():
    """Test ToolCoordinator with mock selector."""
    logger.info("Testing ToolCoordinator with mock selector...")

    # Create mock selector
    mock_selector = MockToolSelector(
        fixed_tools=["code_search", "file_edit", "test_gen"]
    )

    # Create coordinator with mock
    coordinator = ToolCoordinator(selector=mock_selector, max_tools=5)

    # Test
    result = coordinator.select_and_execute("test task")

    logger.info(f"Mock test result: {result}")
    assert result["selected_tools"] == ["code_search", "file_edit", "test_gen"]
    assert result["strategy"] == "mock"

    logger.info("Mock test passed!")


# =============================================================================
# PART 5: Demonstrate Flexibility
# =============================================================================


def demonstrate_strategy_swapping():
    """Demonstrate swapping strategies without changing coordinator."""
    logger.info("\n" + "=" * 70)
    logger.info("DEMONSTRATION: Strategy Swapping")
    logger.info("=" * 70)

    # Create different selectors
    semantic = SemanticToolSelector()
    keyword = KeywordToolSelector()
    hybrid = HybridToolSelector(semantic, keyword, semantic_weight=0.7)

    # Single coordinator works with all selectors
    coordinator = ToolCoordinator(selector=semantic, max_tools=3)

    task = "search and refactor code for better testing"

    # Test with semantic
    logger.info("\n--- Using Semantic Selector ---")
    coordinator.selector = semantic  # Swap strategy
    result_semantic = coordinator.select_and_execute(task)
    logger.info(f"Selected: {result_semantic['selected_tools']}")

    # Test with keyword
    logger.info("\n--- Using Keyword Selector ---")
    coordinator.selector = keyword  # Swap strategy
    result_keyword = coordinator.select_and_execute(task)
    logger.info(f"Selected: {result_keyword['selected_tools']}")

    # Test with hybrid
    logger.info("\n--- Using Hybrid Selector ---")
    coordinator.selector = hybrid  # Swap strategy
    result_hybrid = coordinator.select_and_execute(task)
    logger.info(f"Selected: {result_hybrid['selected_tools']}")

    logger.info("\n✓ All strategies work with same coordinator!")
    logger.info("✓ No coordinator code changes needed!")


def main():
    """Run all examples."""
    logger.info("\n" + "=" * 70)
    logger.info("PROTOCOL USAGE EXAMPLE")
    logger.info("=" * 70)

    # Test with mock
    test_coordinator_with_mock()

    # Demonstrate flexibility
    demonstrate_strategy_swapping()

    # Real-world usage
    logger.info("\n" + "=" * 70)
    logger.info("REAL-WORLD USAGE")
    logger.info("=" * 70)

    # Create selectors
    semantic = SemanticToolSelector()
    keyword = KeywordToolSelector()
    hybrid = HybridToolSelector(semantic, keyword)

    # Use in coordinator
    coordinator = ToolCoordinator(selector=hybrid, max_tools=5)

    result = coordinator.select_and_execute(
        "I need to debug and refactor the authentication code"
    )

    logger.info(f"\nFinal Result:")
    logger.info(f"  Task: {result['task']}")
    logger.info(f"  Strategy: {result['strategy']}")
    logger.info(f"  Selected Tools: {result['selected_tools']}")
    logger.info(f"  Scores: {result['scores']}")

    logger.info("\n" + "=" * 70)
    logger.info("KEY TAKEAWAYS")
    logger.info("=" * 70)
    logger.info("1. Protocols enable loose coupling (DIP)")
    logger.info("2. Easy to swap implementations (OCP)")
    logger.info("3. Simple to test with mocks")
    logger.info("4. Type-safe with runtime_checkable")
    logger.info("5. No inheritance required - structural typing")
    logger.info("\nRun with: python -m examples.architecture.protocol_usage")


if __name__ == "__main__":
    main()
