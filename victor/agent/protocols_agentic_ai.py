# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Agentic AI component protocols for Phase 3 integration.

This module defines protocol interfaces for advanced agentic AI capabilities:
- Hierarchical Planning: Task decomposition and planning
- Episodic Memory: Agent experience storage and retrieval
- Semantic Memory: Factual knowledge storage and querying
- Skill Discovery: Dynamic tool discovery and composition
- Skill Chaining: Multi-step skill chain execution
- Proficiency Tracking: Performance tracking and improvement

These protocols enable loose coupling and testability for agentic AI components.
All components depend on protocols, not concrete implementations.

Design Pattern: Protocol-Based Design
- All agentic components use protocol interfaces
- Enables dependency injection and mocking
- Supports multiple implementations
- Facilitates testing and extensibility
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from enum import Enum


# =============================================================================
# Hierarchical Planning Protocols
# =============================================================================


class TaskStatus(Enum):
    """Status of a task in the decomposition graph."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@runtime_checkable
class HierarchicalPlannerProtocol(Protocol):
    """Protocol for hierarchical task decomposition and planning.

    The hierarchical planner breaks down complex tasks into smaller subtasks
    with explicit dependencies, enabling systematic execution and re-planning.

    Key Methods:
        decompose_task: Break down a complex task into subtasks
        suggest_next_tasks: Get executable tasks based on dependencies
        update_plan: Update plan after task execution
        validate_plan: Validate plan for correctness

    Example:
        planner: HierarchicalPlannerProtocol = container.get(HierarchicalPlannerProtocol)

        # Decompose task
        graph = await planner.decompose_task("Implement user authentication")

        # Get next tasks
        tasks = await planner.suggest_next_tasks(graph)

        # Execute and update
        result = await planner.update_plan(graph, completed_tasks=["task_1"])
    """

    async def decompose_task(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        max_depth: int = 5,
    ) -> Any:
        """Decompose a complex task into hierarchical subtasks.

        Args:
            goal: High-level goal to decompose
            context: Additional context for decomposition
            max_depth: Maximum decomposition depth

        Returns:
            TaskGraph with hierarchical task decomposition
        """
        ...

    async def suggest_next_tasks(
        self,
        graph: Any,
        max_tasks: int = 5,
    ) -> List[Any]:
        """Suggest next executable tasks based on dependencies.

        Args:
            graph: TaskGraph to analyze
            max_tasks: Maximum tasks to return

        Returns:
            List of executable Task objects
        """
        ...

    async def update_plan(
        self,
        graph: Any,
        completed_tasks: List[str],
        failed_tasks: Optional[List[str]] = None,
    ) -> Any:
        """Update plan after task execution.

        Args:
            graph: TaskGraph to update
            completed_tasks: List of completed task IDs
            failed_tasks: List of failed task IDs (optional)

        Returns:
            Updated UpdatedPlan object
        """
        ...

    def validate_plan(self, graph: Any) -> Any:
        """Validate plan for correctness.

        Args:
            graph: TaskGraph to validate

        Returns:
            ValidationResult with validation status and errors
        """
        ...


# =============================================================================
# Memory Protocols
# =============================================================================


@runtime_checkable
class EpisodicMemoryProtocol(Protocol):
    """Protocol for episodic memory (agent experiences).

    Episodic memory stores and retrieves agent experiences with context,
    actions, and outcomes for learning and retrieval.

    Key Methods:
        store_episode: Store an episode in memory
        recall_relevant: Recall episodes relevant to query
        recall_recent: Recall recent episodes
        recall_by_outcome: Recall episodes by outcome pattern
        get_memory_statistics: Get memory statistics

    Example:
        memory: EpisodicMemoryProtocol = container.get(EpisodicMemoryProtocol)

        # Store episode
        episode_id = await memory.store_episode(episode)

        # Recall relevant episodes
        relevant = await memory.recall_relevant("fix authentication bug", k=5)
        """

    async def store_episode(
        self,
        inputs: Dict[str, Any],
        actions: List[Any],
        outcomes: Dict[str, Any],
        rewards: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store an episode in episodic memory.

        Args:
            inputs: Initial inputs/context
            actions: Actions taken during episode
            outcomes: Results and outcomes
            rewards: Optional reward signal
            context: Additional context

        Returns:
            Episode ID (UUID)
        """
        ...

    async def recall_relevant(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.3,
    ) -> List[Any]:
        """Recall episodes relevant to query.

        Args:
            query: Query text for similarity search
            k: Maximum episodes to return
            threshold: Minimum similarity threshold

        Returns:
            List of relevant Episode objects
        """
        ...

    async def recall_recent(
        self,
        n: int = 10,
    ) -> List[Any]:
        """Recall recent episodes.

        Args:
            n: Maximum episodes to return

        Returns:
            List of recent Episode objects
        """
        ...

    async def recall_by_outcome(
        self,
        outcome_key: str,
        outcome_value: Any,
        n: int = 10,
    ) -> List[Any]:
        """Recall episodes by outcome pattern.

        Args:
            outcome_key: Key in outcomes dict
            outcome_value: Value to match
            n: Maximum episodes to return

        Returns:
            List of matching Episode objects
        """
        ...

    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory statistics.

        Returns:
            Dictionary with memory stats (count, size, etc.)
        """
        ...


@runtime_checkable
class SemanticMemoryProtocol(Protocol):
    """Protocol for semantic memory (factual knowledge).

    Semantic memory stores and queries factual knowledge with
    vector similarity search and knowledge graph linking.

    Key Methods:
        store_knowledge: Store a fact in semantic memory
        query_knowledge: Query facts by similarity
        link_facts: Create links between related facts
        get_knowledge_graph: Get knowledge graph structure

    Example:
        memory: SemanticMemoryProtocol = container.get(SemanticMemoryProtocol)

        # Store knowledge
        fact_id = await memory.store_knowledge("Python uses asyncio for concurrency")

        # Query knowledge
        facts = await memory.query_knowledge("concurrency in Python")
        """

    async def store_knowledge(
        self,
        fact: str,
        metadata: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0,
    ) -> str:
        """Store a fact in semantic memory.

        Args:
            fact: Factual knowledge to store
            metadata: Optional metadata
            confidence: Confidence score (0-1)

        Returns:
            Fact ID (UUID)
        """
        ...

    async def query_knowledge(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.25,
    ) -> List[Any]:
        """Query facts by similarity.

        Args:
            query: Query text
            k: Maximum facts to return
            threshold: Minimum similarity threshold

        Returns:
            List of Knowledge objects
        """
        ...

    async def link_facts(
        self,
        fact_id_1: str,
        fact_id_2: str,
        link_type: str = "related",
        strength: float = 1.0,
    ) -> bool:
        """Create a link between two facts.

        Args:
            fact_id_1: First fact ID
            fact_id_2: Second fact ID
            link_type: Type of link (related, depends_on, etc.)
            strength: Link strength (0-1)

        Returns:
            True if link created successfully
        """
        ...

    def get_knowledge_graph(self) -> Any:
        """Get knowledge graph structure.

        Returns:
            KnowledgeGraph object
        """
        ...


# =============================================================================
# Skill Discovery Protocols
# =============================================================================


@runtime_checkable
class SkillDiscoveryProtocol(Protocol):
    """Protocol for dynamic skill discovery and composition.

    Skill discovery enables agents to discover tools, compose skills,
    and adapt to new requirements dynamically.

    Key Methods:
        discover_tools: Discover available tools for a context
        compose_skill: Compose multiple tools into a skill
        analyze_compatibility: Analyze tool compatibility

    Example:
        discovery: SkillDiscoveryProtocol = container.get(SkillDiscoveryProtocol)

        # Discover tools
        tools = await discovery.discover_tools(context="code analysis")

        # Compose skill
        skill = await discovery.compose_skill("analyzer", tools, "Analyzes code")
        """

    async def discover_tools(
        self,
        context: str,
        max_tools: int = 20,
    ) -> List[Any]:
        """Discover tools relevant to context.

        Args:
            context: Context description (query, task, etc.)
            max_tools: Maximum tools to return

        Returns:
            List of AvailableTool objects
        """
        ...

    async def compose_skill(
        self,
        skill_name: str,
        tools: List[Any],
        description: str,
    ) -> Any:
        """Compose multiple tools into a skill.

        Args:
            skill_name: Name for the composed skill
            tools: List of tools to compose
            description: Skill description

        Returns:
            Skill object
        """
        ...

    async def analyze_compatibility(
        self,
        tool_1: Any,
        tool_2: Any,
    ) -> float:
        """Analyze compatibility between two tools.

        Args:
            tool_1: First tool
            tool_2: Second tool

        Returns:
            Compatibility score (0-1)
        """
        ...


# =============================================================================
# Skill Chaining Protocols
# =============================================================================


@runtime_checkable
class SkillChainerProtocol(Protocol):
    """Protocol for skill chaining and execution.

    Skill chaining enables agents to plan and execute multi-step
    workflows composed of dynamic skills.

    Key Methods:
        plan_chain: Plan a skill chain for a goal
        execute_chain: Execute a skill chain
        validate_chain: Validate a chain before execution

    Example:
        chainer: SkillChainerProtocol = container.get(SkillChainerProtocol)

        # Plan chain
        chain = await chainer.plan_chain("Fix bugs", skills)

        # Execute chain
        result = await chainer.execute_chain(chain)
        """

    async def plan_chain(
        self,
        goal: str,
        skills: List[Any],
        max_length: int = 10,
    ) -> Any:
        """Plan a skill chain for a goal.

        Args:
            goal: Goal to achieve
            skills: Available skills
            max_length: Maximum chain length

        Returns:
            SkillChain object
        """
        ...

    async def execute_chain(
        self,
        chain: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Execute a skill chain.

        Args:
            chain: SkillChain to execute
            context: Optional execution context

        Returns:
            ChainResult object
        """
        ...

    def validate_chain(self, chain: Any) -> Any:
        """Validate a skill chain.

        Args:
            chain: SkillChain to validate

        Returns:
            ValidationResult object
        """
        ...


# =============================================================================
# Self-Improvement Protocols
# =============================================================================


@runtime_checkable
class ProficiencyTrackerProtocol(Protocol):
    """Protocol for proficiency tracking and self-improvement.

    Proficiency tracking enables agents to monitor performance,
    identify improvement opportunities, and optimize decisions.

    Key Methods:
        record_outcome: Record task/tool outcome for learning
        get_proficiency: Get proficiency score for a tool/task
        get_suggestions: Get improvement suggestions
        get_metrics: Get proficiency metrics

    Example:
        tracker: ProficiencyTrackerProtocol = container.get(ProficiencyTrackerProtocol)

        # Record outcome
        await tracker.record_outcome(
            task="code_review",
            tool="ast_analyzer",
            outcome=TaskOutcome(success=True, duration=1.5, cost=0.001)
        )

        # Get proficiency
        score = tracker.get_proficiency(tool="ast_analyzer")

        # Get suggestions
        suggestions = tracker.get_suggestions()
        """

    async def record_outcome(
        self,
        task: str,
        tool: str,
        success: bool,
        duration: float,
        cost: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record task/tool outcome for learning.

        Args:
            task: Task type
            tool: Tool used
            success: Success status
            duration: Execution duration (seconds)
            cost: Execution cost
            metadata: Optional metadata
        """
        ...

    def get_proficiency(
        self,
        tool: str,
        task: Optional[str] = None,
    ) -> Any:
        """Get proficiency score for a tool/task.

        Args:
            tool: Tool name
            task: Optional task type

        Returns:
            ProficiencyScore object
        """
        ...

    def get_suggestions(
        self,
        n: int = 5,
    ) -> List[Any]:
        """Get improvement suggestions.

        Args:
            n: Maximum suggestions to return

        Returns:
            List of Suggestion objects
        """
        ...

    def get_metrics(self) -> Dict[str, Any]:
        """Get proficiency metrics.

        Returns:
            Dictionary with metrics (averages, trends, etc.)
        """
        ...


# =============================================================================
# RL Coordinator Protocol
# =============================================================================


@runtime_checkable
class RLCoordinatorProtocol(Protocol):
    """Protocol for reinforcement learning coordination.

    RL coordinator enables agents to learn optimal policies
    through reward-based optimization.

    Key Methods:
        select_action: Select action using current policy
        update_policy: Update policy based on reward
        get_policy: Get current policy state

    Example:
        rl: RLCoordinatorProtocol = container.get(RLCoordinatorProtocol)

        # Select action
        action = await rl.select_action(state)

        # Update policy
        await rl.update_policy(state, action, reward, next_state)
        """

    async def select_action(
        self,
        state: Dict[str, Any],
        actions: List[Any],
        explore: bool = False,
    ) -> Any:
        """Select action using current policy.

        Args:
            state: Current state
            actions: Available actions
            explore: Whether to explore (epsilon-greedy)

        Returns:
            Selected Action object
        """
        ...

    async def update_policy(
        self,
        state: Dict[str, Any],
        action: Any,
        reward: float,
        next_state: Dict[str, Any],
        done: bool = False,
    ) -> None:
        """Update policy based on reward.

        Args:
            state: Previous state
            action: Action taken
            reward: Reward received
            next_state: New state
            done: Whether episode is complete
        """
        ...

    def get_policy(self) -> Dict[str, Any]:
        """Get current policy state.

        Returns:
            Dictionary with policy information
        """
        ...


# =============================================================================
# Protocol Exports
# =============================================================================


__all__ = [
    # Hierarchical Planning
    "HierarchicalPlannerProtocol",
    "TaskStatus",
    # Episodic Memory
    "EpisodicMemoryProtocol",
    # Semantic Memory
    "SemanticMemoryProtocol",
    # Skill Discovery
    "SkillDiscoveryProtocol",
    # Skill Chaining
    "SkillChainerProtocol",
    # Self-Improvement
    "ProficiencyTrackerProtocol",
    # RL Coordinator
    "RLCoordinatorProtocol",
]
