"""
Task Decomposition Module for Victor AI.

This module provides intelligent task decomposition and execution planning capabilities,
including dependency management, critical path analysis, and parallel execution identification.

Key Features:
- NetworkX-based DAG for dependency management
- Topological sorting for execution order
- Cycle detection in dependencies
- Critical path analysis for optimization
- Parallel execution identification
- Task status tracking with comprehensive state management

Example:
    from victor.agent.planning.task_decomposition import TaskDecomposition
    from victor.framework import Task

    decomposition = TaskDecomposition()

    # Add tasks with dependencies
    task1 = Task(id="t1", description="First task")
    task2 = Task(id="t2", description="Second task")
    decomposition.add_task(task1, dependencies=[])
    decomposition.add_task(task2, dependencies=["t1"])

    # Get execution order and ready tasks
    execution_order = decomposition.get_execution_order()
    ready_tasks = decomposition.get_ready_tasks()

    # Analyze critical path
    critical_path = decomposition.get_critical_path()
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx


@dataclass
class SimpleTask:
    """
    Simple task representation for the TaskDecomposition system.

    This is a lightweight task class used by the new NetworkX-based
    TaskDecomposition implementation. For backward compatibility,
    the legacy Task class is also provided.

    Attributes:
        id: Unique task identifier
        description: Human-readable task description
        context: Additional context data
    """

    id: str
    description: str
    context: Dict[str, Any] = field(default_factory=dict)


class TaskStatus(enum.Enum):
    """Status of a task in the decomposition graph.

    Attributes:
        PENDING: Task is waiting to be executed
        IN_PROGRESS: Task is currently being executed
        COMPLETED: Task has completed successfully
        FAILED: Task has failed
        SKIPPED: Task was skipped (e.g., due to failed dependencies)
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class DependencyType(enum.Enum):
    """Type of dependency between tasks.

    Attributes:
        STRONG: Must complete before dependent starts (hard dependency)
        WEAK: Prefer to complete, but not required (soft dependency)
        CONFLICT: Must not run in parallel (resource conflict)
    """

    STRONG = "strong"
    WEAK = "weak"
    CONFLICT = "conflict"


@dataclass
class TaskNode:
    """
    A node in the task decomposition graph.

    Wraps a SimpleTask with additional metadata for execution planning and analysis.

    Attributes:
        task: The underlying SimpleTask object
        status: Current status of the task
        complexity: Estimated complexity score (1-10)
        estimated_time: Estimated execution time in seconds
        resource_requirements: Resources needed for execution (e.g., {"cpu": 2, "memory": "4GB"})
        metadata: Additional task metadata for custom use cases
    """

    task: SimpleTask
    status: TaskStatus = TaskStatus.PENDING
    complexity: int = 5
    estimated_time: float = 0.0
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        """Hash based on task ID for use in sets and as dict keys."""
        return hash(self.task.id)

    def __eq__(self, other: object) -> bool:
        """Equality based on task ID."""
        if not isinstance(other, TaskNode):
            return False
        return self.task.id == other.task.id

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task.id,
            "description": self.task.description,
            "status": self.status.value,
            "complexity": self.complexity,
            "estimated_time": self.estimated_time,
            "resource_requirements": self.resource_requirements,
            "metadata": self.metadata,
        }


@dataclass
class DependencyEdge:
    """
    An edge in the task decomposition graph representing a dependency.

    Attributes:
        dependency_type: Type of dependency relationship
        strength: Strength of dependency (0.0-1.0), where 1.0 is mandatory
        metadata: Additional edge metadata for custom use cases
    """

    dependency_type: DependencyType = DependencyType.STRONG
    strength: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "dependency_type": self.dependency_type.value,
            "strength": self.strength,
            "metadata": self.metadata,
        }


class TaskDecomposition:
    """
    Intelligent task decomposition and execution planning system.

    This class manages a directed acyclic graph (DAG) of tasks with dependencies,
    providing capabilities for execution planning, critical path analysis, and
    parallel execution identification using NetworkX for efficient graph algorithms.

    Features:
    - Dependency graph management with NetworkX DiGraph
    - Topological sorting for valid execution order
    - Cycle detection in dependencies
    - Critical path analysis for optimization
    - Parallel execution identification
    - Task status tracking and updates
    - Resource conflict detection
    - Execution level grouping
    - Comprehensive statistics and visualization

    Example:
        >>> decomposition = TaskDecomposition()
        >>> task1 = Task(id="t1", description="First task")
        >>> task2 = Task(id="t2", description="Second task")
        >>> decomposition.add_task(task1, dependencies=[])
        >>> decomposition.add_task(task2, dependencies=["t1"])
        >>> execution_order = decomposition.get_execution_order()
        >>> ready_tasks = decomposition.get_ready_tasks()
        >>> critical_path = decomposition.get_critical_path()
    """

    def __init__(self) -> None:
        """Initialize an empty task decomposition graph."""
        self._graph: nx.DiGraph = nx.DiGraph()
        self._task_nodes: Dict[str, TaskNode] = {}
        self._dependency_edges: Dict[Tuple[str, str], DependencyEdge] = {}

    def add_task(
        self,
        task: SimpleTask,
        dependencies: List[str],
        complexity: int = 5,
        estimated_time: float = 0.0,
        resource_requirements: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a task to the decomposition graph.

        Args:
            task: The SimpleTask object to add
            dependencies: List of task IDs this task depends on
            complexity: Estimated complexity (1-10, default: 5)
            estimated_time: Estimated execution time in seconds (default: 0.0)
            resource_requirements: Required resources for execution
            metadata: Additional task metadata

        Raises:
            ValueError: If task ID already exists or dependencies contain unknown tasks

        Example:
            >>> task = SimpleTask(id="task1", description="Implement feature")
            >>> decomposition.add_task(
            ...     task,
            ...     dependencies=[],
            ...     complexity=7,
            ...     estimated_time=300.0,
            ...     resource_requirements={"cpu": 2, "memory": "4GB"}
            ... )
        """
        if task.id in self._task_nodes:
            raise ValueError(f"Task with ID '{task.id}' already exists")

        # Validate dependencies exist
        for dep_id in dependencies:
            if dep_id not in self._task_nodes:
                raise ValueError(f"Dependency task '{dep_id}' not found for task '{task.id}'")

        # Create task node
        task_node = TaskNode(
            task=task,
            status=TaskStatus.PENDING,
            complexity=max(1, min(10, complexity)),
            estimated_time=estimated_time,
            resource_requirements=resource_requirements or {},
            metadata=metadata or {},
        )

        # Add to graph
        self._task_nodes[task.id] = task_node
        self._graph.add_node(task.id, node=task_node)

        # Add dependency edges
        for dep_id in dependencies:
            self._add_dependency_edge(dep_id, task.id)

    def _add_dependency_edge(
        self,
        from_task: str,
        to_task: str,
        dependency_type: DependencyType = DependencyType.STRONG,
        strength: float = 1.0,
    ) -> None:
        """
        Add a dependency edge between two tasks.

        Args:
            from_task: Source task ID (dependency)
            to_task: Target task ID (dependent)
            dependency_type: Type of dependency (default: STRONG)
            strength: Strength of dependency 0.0-1.0 (default: 1.0)
        """
        edge = DependencyEdge(
            dependency_type=dependency_type,
            strength=max(0.0, min(1.0, strength)),
        )

        self._dependency_edges[(from_task, to_task)] = edge
        self._graph.add_edge(
            from_task,
            to_task,
            dependency_type=dependency_type.value,
            strength=strength,
        )

    def to_networkx_graph(self) -> nx.DiGraph:
        """
        Convert the task decomposition to a NetworkX DiGraph.

        Returns a copy of the internal dependency graph. Nodes are task IDs,
        edges represent dependencies (from dependency to dependent).

        Returns:
            A directed graph representing task dependencies.
            Can be used with NetworkX algorithms for custom analysis.

        Example:
            >>> graph = decomposition.to_networkx_graph()
            >>> print(f"Nodes: {graph.number_of_nodes()}")
            >>> print(f"Edges: {graph.number_of_edges()}")
        """
        return self._graph.copy()

    # Note: get_ready_tasks() and validate_plan() methods are now below
    # with legacy API compatibility. The old methods are removed to avoid conflicts.

    def get_task_dependencies(self, task_id: str) -> List[str]:
        """
        Get the list of task IDs that a task depends on.

        Args:
            task_id: The task to query

        Returns:
            List of task IDs that are direct dependencies

        Raises:
            KeyError: If task_id not found in the graph

        Example:
            >>> deps = decomposition.get_task_dependencies("task2")
            >>> print(f"task2 depends on: {deps}")
        """
        if task_id not in self._task_nodes:
            raise KeyError(f"Task '{task_id}' not found")

        return list(self._graph.predecessors(task_id))

    def get_task_dependents(self, task_id: str) -> List[str]:
        """
        Get the list of task IDs that depend on a task.

        Args:
            task_id: The task to query

        Returns:
            List of task IDs that directly depend on this task

        Raises:
            KeyError: If task_id not found in the graph

        Example:
            >>> dependents = decomposition.get_task_dependents("task1")
            >>> print(f"Tasks waiting for task1: {dependents}")
        """
        if task_id not in self._task_nodes:
            raise KeyError(f"Task '{task_id}' not found")

        return list(self._graph.successors(task_id))

    def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update the status of a task.

        Args:
            task_id: The task to update
            status: New status (PENDING, IN_PROGRESS, COMPLETED, FAILED, SKIPPED)
            metadata: Optional metadata to attach to status update

        Raises:
            KeyError: If task_id not found

        Example:
            >>> decomposition.update_task_status("task1", TaskStatus.IN_PROGRESS)
            >>> # ... execute task ...
            >>> decomposition.update_task_status(
            ...     "task1",
            ...     TaskStatus.COMPLETED,
            ...     metadata={"execution_time": 250.0}
            ... )
        """
        if task_id not in self._task_nodes:
            raise KeyError(f"Task '{task_id}' not found")

        self._task_nodes[task_id].status = status

        if metadata:
            self._task_nodes[task_id].metadata.update(metadata)

    def get_execution_order(self) -> List[str]:
        """
        Get a valid topological execution order for all tasks.

        Uses Kahn's algorithm to compute a topological sort of the DAG.
        Tasks are ordered such that all dependencies come before the tasks
        that depend on them.

        Returns:
            List of task IDs in valid execution order

        Raises:
            ValueError: If the graph contains cycles

        Example:
            >>> order = decomposition.get_execution_order()
            >>> print("Execution order:", " -> ".join(order))
        """
        try:
            return list(nx.topological_sort(self._graph))
        except nx.NetworkXUnfeasible:
            cycles = self.detect_cycles()
            cycle_str = ", ".join([" -> ".join(cycle) for cycle in cycles[:3]])
            raise ValueError(
                f"Cannot compute execution order: graph contains cycles. "
                f"Detected cycles: {cycle_str}"
            )

    def detect_cycles(self) -> List[List[str]]:
        """
        Detect cycles in the task dependency graph.

        Uses NetworkX's simple_cycles algorithm to find all cycles.
        Useful for debugging invalid dependency structures.

        Returns:
            List of cycles, where each cycle is a list of task IDs

        Example:
            >>> cycles = decomposition.detect_cycles()
            >>> if cycles:
            ...     print(f"Found {len(cycles)} cycles:")
            ...     for cycle in cycles:
            ...         print(f"  {' -> '.join(cycle)}")
        """
        cycles = []

        try:
            # simple_cycles returns a generator of cycles
            for cycle in nx.simple_cycles(self._graph):
                if len(cycle) > 0:  # Filter out empty cycles
                    cycles.append(cycle)
        except Exception:
            # Fallback to basic cycle detection
            pass

        return cycles

    def get_parallel_execution_groups(self) -> List[List[str]]:
        """
        Identify groups of tasks that can be executed in parallel.

        Tasks are in the same group if:
        1. They are at the same execution level (depth) in the graph
        2. There are no dependencies between them
        3. No resource conflicts exist

        Returns:
            List of groups, where each group is a list of task IDs.
            Groups are in execution order (group 0 first, then group 1, etc.).

        Example:
            >>> groups = decomposition.get_parallel_execution_groups()
            >>> for i, group in enumerate(groups):
            ...     print(f"Group {i}: {len(group)} tasks can run in parallel")
        """
        execution_order = self.get_execution_order()
        groups: List[List[str]] = []
        assigned: Set[str] = set()

        for task_id in execution_order:
            if task_id in assigned:
                continue

            # Find tasks that can run in parallel with this one
            parallel_group = [task_id]
            assigned.add(task_id)

            for other_id in execution_order:
                if other_id in assigned:
                    continue

                # Check if tasks can run in parallel
                if self._can_run_in_parallel(task_id, other_id):
                    parallel_group.append(other_id)
                    assigned.add(other_id)

            groups.append(parallel_group)

        return groups

    def _can_run_in_parallel(self, task_id1: str, task_id2: str) -> bool:
        """
        Check if two tasks can run in parallel.

        Tasks can run in parallel if:
        1. Neither depends on the other (directly or transitively)
        2. No shared exclusive resources (e.g., both requiring the same file lock)

        Args:
            task_id1: First task ID
            task_id2: Second task ID

        Returns:
            True if tasks can run in parallel, False otherwise
        """
        # Check direct dependencies
        if self._graph.has_edge(task_id1, task_id2) or self._graph.has_edge(task_id2, task_id1):
            return False

        # Check if one is ancestor of the other (transitive dependency)
        try:
            if nx.has_path(self._graph, task_id1, task_id2) or nx.has_path(
                self._graph, task_id2, task_id1
            ):
                return False
        except nx.NetworkXError:
            # Path finding may fail if graph is inconsistent
            return False

        # Check resource conflicts
        node1 = self._task_nodes[task_id1]
        node2 = self._task_nodes[task_id2]

        resources1 = node1.resource_requirements
        resources2 = node2.resource_requirements

        # Check for exclusive resource conflicts (e.g., file locks, GPU)
        for resource, amount1 in resources1.items():
            if resource.endswith("_exclusive"):
                if resource in resources2:
                    return False

        return True

    def get_critical_path(self) -> List[str]:
        """
        Calculate the critical path through the task graph.

        The critical path is the longest path (by estimated time) from any
        source task to any sink task. Tasks on the critical path determine
        the minimum total execution time - delaying any task on the critical
        path delays the entire execution.

        Uses the estimated_time attribute of TaskNode for path weight.
        Tasks with no time estimate are treated as zero duration.

        Returns:
            List of task IDs in critical path order (from source to sink)

        Raises:
            ValueError: If the graph contains cycles

        Example:
            >>> critical = decomposition.get_critical_path()
            >>> print(f"Critical path: {' -> '.join(critical)}")
            >>> print(f"Total time: {decomposition.get_critical_path_length()}s")
        """
        # Ensure graph is a DAG
        if self.detect_cycles():
            raise ValueError("Cannot compute critical path: graph contains cycles")

        # Build weighted graph for longest path calculation
        weighted_graph = self._graph.copy()

        # Set edge weights to successor's estimated time
        for u, v in weighted_graph.edges():
            successor_node = self._task_nodes[v]
            weighted_graph.edges[u, v]["weight"] = successor_node.estimated_time

        # Find all source and sink nodes
        sources = [n for n in weighted_graph.nodes() if weighted_graph.in_degree(n) == 0]
        sinks = [n for n in weighted_graph.nodes() if weighted_graph.out_degree(n) == 0]

        if not sources or not sinks:
            return []

        # Find longest path from any source to any sink
        longest_path: List[str] = []
        max_length = float("-inf")

        for source in sources:
            for sink in sinks:
                try:
                    # Use shortest_path with negative weights for longest path
                    path = nx.shortest_path(
                        weighted_graph, source=source, target=sink, weight="weight"
                    )

                    # Calculate total path weight (sum of task durations)
                    path_length = sum(self._task_nodes[task_id].estimated_time for task_id in path)

                    if path_length > max_length:
                        max_length = path_length
                        longest_path = path

                except nx.NetworkXNoPath:
                    continue

        return longest_path

    def get_critical_path_length(self) -> float:
        """
        Calculate the total estimated time of the critical path.

        Returns:
            Total estimated execution time in seconds for the critical path.

        Example:
            >>> length = decomposition.get_critical_path_length()
            >>> print(f"Minimum execution time: {length:.2f}s")
        """
        critical_path = self.get_critical_path()
        return sum(self._task_nodes[task_id].estimated_time for task_id in critical_path)

    def get_task_depth(self, task_id: str) -> int:
        """
        Get the depth of a task in the dependency graph.

        Depth is the length of the longest path from any source node to this task.
        Source tasks (no dependencies) have depth 0.

        Args:
            task_id: The task to query

        Returns:
            Depth of the task (0 for source tasks)

        Raises:
            KeyError: If task_id not found

        Example:
            >>> depth = decomposition.get_task_depth("task3")
            >>> print(f"task3 is at depth {depth}")
        """
        if task_id not in self._task_nodes:
            raise KeyError(f"Task '{task_id}' not found")

        # Find longest path from any source to this task
        sources = [n for n in self._graph.nodes() if self._graph.in_degree(n) == 0]

        max_depth = 0
        for source in sources:
            try:
                path = nx.shortest_path(self._graph, source=source, target=task_id)
                depth = len(path) - 1
                max_depth = max(max_depth, depth)
            except nx.NetworkXNoPath:
                continue

        return max_depth

    def get_execution_levels(self) -> List[List[str]]:
        """
        Group tasks by their depth in the dependency graph.

        All tasks at the same level can potentially execute in parallel.
        This is useful for understanding the parallel structure of the task graph.

        Returns:
            List of levels, where each level is a list of task IDs.
            Levels are ordered from 0 (source tasks) to N (sink tasks).

        Example:
            >>> levels = decomposition.get_execution_levels()
            >>> for i, level in enumerate(levels):
            ...     print(f"Level {i}: {len(level)} tasks")
        """
        task_depths: Dict[int, List[str]] = {}

        for task_id in self._task_nodes:
            depth = self.get_task_depth(task_id)
            if depth not in task_depths:
                task_depths[depth] = []
            task_depths[depth].append(task_id)

        # Return levels in order
        sorted_depths = sorted(task_depths.keys())
        return [task_depths[depth] for depth in sorted_depths]

    def get_completion_percentage(self) -> float:
        """
        Calculate the overall completion percentage of all tasks.

        Returns:
            Percentage (0.0-100.0) of completed tasks

        Example:
            >>> pct = decomposition.get_completion_percentage()
            >>> print(f"Progress: {pct:.1f}%")
        """
        if not self._task_nodes:
            return 0.0

        completed = sum(
            1 for node in self._task_nodes.values() if node.status == TaskStatus.COMPLETED
        )

        return (completed / len(self._task_nodes)) * 100.0

    def get_failed_tasks(self) -> List[SimpleTask]:
        """
        Get all tasks that have failed.

        Returns:
            List of failed SimpleTask objects

        Example:
            >>> failed = decomposition.get_failed_tasks()
            >>> for task in failed:
            ...     print(f"Failed: {task.id} - {task.description}")
        """
        return [node.task for node in self._task_nodes.values() if node.status == TaskStatus.FAILED]

    def get_blocked_tasks(self) -> List[SimpleTask]:
        """
        Get tasks that are blocked by failed dependencies.

        A task is blocked if:
        - It's in PENDING status
        - At least one STRONG dependency has FAILED

        Returns:
            List of SimpleTask objects that cannot proceed due to failed dependencies

        Example:
            >>> blocked = decomposition.get_blocked_tasks()
            >>> if blocked:
            ...     print(f"{len(blocked)} tasks are blocked by failures")
        """
        blocked_tasks: List[SimpleTask] = []

        for task_id, node in self._task_nodes.items():
            if node.status != TaskStatus.PENDING:
                continue

            dependencies = list(self._graph.predecessors(task_id))
            has_failed_deps = False

            for dep_id in dependencies:
                dep_node = self._task_nodes[dep_id]
                edge = self._dependency_edges[(dep_id, task_id)]

                # Only check STRONG dependencies
                if (
                    edge.dependency_type == DependencyType.STRONG
                    and dep_node.status == TaskStatus.FAILED
                ):
                    has_failed_deps = True
                    break

            if has_failed_deps:
                blocked_tasks.append(node.task)

        return blocked_tasks

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the task decomposition.

        Returns:
            Dictionary containing various statistics:
            - total_tasks: Total number of tasks
            - completed_tasks: Number of completed tasks
            - in_progress_tasks: Number of in-progress tasks
            - failed_tasks: Number of failed tasks
            - pending_tasks: Number of pending tasks
            - total_dependencies: Total number of dependency edges
            - average_complexity: Average complexity score (1-10)
            - total_estimated_time: Sum of all task estimates
            - critical_path_length: Number of tasks on critical path
            - critical_path_time: Total time for critical path
            - completion_percentage: Overall completion percentage

        Example:
            >>> stats = decomposition.get_statistics()
            >>> print(f"Completion: {stats['completion_percentage']}%")
            >>> print(f"Critical path time: {stats['critical_path_time']}s")
        """
        total_tasks = len(self._task_nodes)
        completed = sum(
            1 for node in self._task_nodes.values() if node.status == TaskStatus.COMPLETED
        )
        in_progress = sum(
            1 for node in self._task_nodes.values() if node.status == TaskStatus.IN_PROGRESS
        )
        failed = sum(1 for node in self._task_nodes.values() if node.status == TaskStatus.FAILED)

        # Calculate total edges
        total_edges = self._graph.number_of_edges()

        # Calculate average complexity
        avg_complexity = (
            sum(node.complexity for node in self._task_nodes.values()) / total_tasks
            if total_tasks > 0
            else 0.0
        )

        # Calculate total estimated time
        total_estimated_time = sum(node.estimated_time for node in self._task_nodes.values())

        # Get critical path info
        try:
            critical_path = self.get_critical_path()
            critical_path_length = len(critical_path)
            critical_path_time = self.get_critical_path_length()
        except ValueError:
            critical_path_length = 0
            critical_path_time = 0.0

        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed,
            "in_progress_tasks": in_progress,
            "failed_tasks": failed,
            "pending_tasks": total_tasks - completed - in_progress - failed,
            "total_dependencies": total_edges,
            "average_complexity": round(avg_complexity, 2),
            "total_estimated_time": round(total_estimated_time, 2),
            "critical_path_length": critical_path_length,
            "critical_path_time": round(critical_path_time, 2),
            "completion_percentage": round(self.get_completion_percentage(), 2),
        }

    def visualize(
        self, output_path: Optional[str] = None, include_metadata: bool = False
    ) -> Optional[str]:
        """
        Generate a visual representation of the task decomposition graph.

        Creates a matplotlib visualization showing tasks as nodes colored by status,
        with arrows indicating dependencies.

        Args:
            output_path: Optional path to save the visualization (PNG format)
            include_metadata: Whether to include task metadata in labels

        Returns:
            DOT format string if output_path is None, otherwise None (saves to file)

        Raises:
            ImportError: If matplotlib is not installed

        Example:
            >>> # Display graph (returns DOT format)
            >>> dot = decomposition.visualize()
            >>>
            >>> # Save to file
            >>> decomposition.visualize(output_path="task_graph.png")
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for visualization. "
                "Install it with: pip install matplotlib"
            )

        # Create figure
        plt.figure(figsize=(14, 10))

        # Use graphviz layout if available, otherwise spring layout
        try:
            import pygraphviz

            pos = nx.nx_agraph.graphviz_layout(self._graph, prog="dot")
        except (ImportError, AttributeError):
            pos = nx.spring_layout(self._graph, k=2, iterations=50, seed=42)

        # Color nodes by status
        color_map = []
        node_sizes = []

        for task_id in self._graph.nodes():
            node = self._task_nodes[task_id]
            if node.status == TaskStatus.COMPLETED:
                color_map.append("#90EE90")  # Light green
            elif node.status == TaskStatus.IN_PROGRESS:
                color_map.append("#87CEEB")  # Light blue
            elif node.status == TaskStatus.FAILED:
                color_map.append("#FFB6C1")  # Light coral
            elif node.status == TaskStatus.SKIPPED:
                color_map.append("#D3D3D3")  # Light gray
            else:
                color_map.append("#FFD700")  # Gold for pending

            # Node size based on complexity
            node_sizes.append(800 + node.complexity * 150)

        # Create labels
        if include_metadata:
            labels = {
                task_id: f"{task_id}\n{node.task.description}\n"
                f"{node.status.value}\n{node.estimated_time}s"
                for task_id, node in self._task_nodes.items()
            }
        else:
            labels = {task_id: task_id for task_id in self._task_nodes}

        # Draw graph
        nx.draw(
            self._graph,
            pos,
            with_labels=True,
            labels=labels,
            node_color=color_map,
            node_size=node_sizes,
            font_size=9,
            font_weight="bold",
            arrows=True,
            arrowstyle="->",
            arrowsize=20,
            edge_color="gray",
            width=1.5,
            alpha=0.9,
        )

        # Add legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#90EE90", label="Completed"),
            Patch(facecolor="#87CEEB", label="In Progress"),
            Patch(facecolor="#FFB6C1", label="Failed"),
            Patch(facecolor="#D3D3D3", label="Skipped"),
            Patch(facecolor="#FFD700", label="Pending"),
        ]
        plt.legend(handles=legend_elements, loc="upper right", fontsize=10)

        # Add title with statistics
        stats = self.get_statistics()
        title = (
            f"Task Decomposition Graph\n"
            f"Tasks: {stats['total_tasks']} | "
            f"Completed: {stats['completed_tasks']} | "
            f"Failed: {stats['failed_tasks']} | "
            f"Progress: {stats['completion_percentage']:.1f}%"
        )
        plt.title(title, fontsize=12, fontweight="bold")
        plt.axis("off")

        if output_path:
            plt.savefig(output_path, format="png", dpi=300, bbox_inches="tight")
            plt.close()
            return None
        else:
            # Return DOT format representation if pygraphviz is available
            try:
                import pygraphviz

                dot_str = nx.nx_agraph.to_agraph(self._graph).to_string()
                plt.close()
                return dot_str
            except ImportError:
                plt.close()

    # ========================================================================
    # Legacy API Compatibility Methods for HierarchicalPlanner Integration
    # ========================================================================

    def to_execution_graph(self, tasks: Optional[List["TaskLegacy"]] = None) -> "TaskGraph":
        """Convert tasks to legacy TaskGraph for backward compatibility.

        This method provides compatibility with the legacy API used by HierarchicalPlanner.
        If tasks are provided, it creates a new TaskGraph from them. Otherwise, it returns
        the internal NetworkX graph converted to a legacy TaskGraph.

        Args:
            tasks: Optional list of legacy Task objects to convert

        Returns:
            Legacy TaskGraph with tasks and dependencies

        Example:
            >>> # New API: convert internal state
            >>> graph = decomposition.to_execution_graph()
            >>>
            >>> # Legacy API: convert list of tasks
            >>> graph = decomposition.to_execution_graph(tasks)
        """
        # If tasks provided, create a new TaskGraph from them (legacy API)
        if tasks is not None:
            legacy_graph = TaskGraph()
            for task in tasks:
                legacy_graph.add_node(task)
                # Add edges for dependencies
                for dep_id in task.depends_on:
                    legacy_graph.add_edge(dep_id, task.id)
            return legacy_graph

        # Otherwise, convert internal state to legacy TaskGraph (new API)
        legacy_graph = TaskGraph()
        for task_id, node in self._task_nodes.items():
            # Create legacy task from node
            legacy_task = TaskLegacy(
                id=task_id,
                description=node.task.description,
                depends_on=list(self._graph.predecessors(task_id)),
                estimated_complexity=node.complexity,
                context=node.metadata.copy(),
            )
            legacy_graph.add_node(legacy_task)

        # Add all edges
        for from_id, to_id in self._graph.edges():
            legacy_graph.add_edge(from_id, to_id)

        return legacy_graph

    def get_ready_tasks(self, task_graph: Optional["TaskGraph"] = None) -> List:
        """Get tasks that are ready to execute.

        This method provides compatibility with both legacy and new APIs.
        If a legacy TaskGraph is provided, it extracts ready tasks from it.
        Otherwise, it uses the internal state.

        Args:
            task_graph: Optional legacy TaskGraph to query

        Returns:
            List of Task objects ready to execute (SimpleTask for new API, TaskLegacy for legacy)

        Example:
            >>> # New API: get from internal state
            >>> ready = decomposition.get_ready_tasks()
            >>>
            >>> # Legacy API: get from legacy graph
            >>> ready = decomposition.get_ready_tasks(task_graph)
        """
        if task_graph is not None:
            # Legacy API: extract ready tasks from TaskGraph
            ready_tasks = []
            for task_id, task in task_graph.nodes.items():
                if task.status != "pending":
                    continue

                # Check if all dependencies satisfied
                deps = task_graph.get_dependencies(task_id)
                if all(
                    task_graph.nodes.get(
                        dep, TaskLegacy(id=dep, description="", status="pending")
                    ).status
                    == "completed"
                    for dep in deps
                ):
                    ready_tasks.append(task)
            # Sort by complexity descending (highest complexity first)
            ready_tasks.sort(key=lambda t: t.estimated_complexity, reverse=True)
            return ready_tasks

        # New API: use internal state
        ready_tasks: List[SimpleTask] = []

        for task_id, node in self._task_nodes.items():
            if node.status != TaskStatus.PENDING:
                continue

            # Check if all strong dependencies are satisfied
            dependencies = list(self._graph.predecessors(task_id))

            all_deps_completed = True
            for dep_id in dependencies:
                dep_node = self._task_nodes[dep_id]
                edge = self._dependency_edges[(dep_id, task_id)]

                # Only enforce STRONG dependencies
                if edge.dependency_type == DependencyType.STRONG:
                    if dep_node.status != TaskStatus.COMPLETED:
                        all_deps_completed = False
                        break

            if all_deps_completed:
                ready_tasks.append(node.task)

        return ready_tasks

    def validate_plan(self, task_graph: Optional["TaskGraph"] = None) -> "ValidationResult":
        """Validate a task graph for correctness.

        This method provides compatibility with both legacy and new APIs.
        If a legacy TaskGraph is provided, it validates that. Otherwise, it validates
        the internal state.

        Args:
            task_graph: Optional legacy TaskGraph to validate

        Returns:
            ValidationResult with errors and warnings

        Example:
            >>> # New API: validate internal state
            >>> result = decomposition.validate_plan()
            >>>
            >>> # Legacy API: validate provided graph
            >>> result = decomposition.validate_plan(task_graph)
        """
        if task_graph is not None:
            # Legacy API: validate TaskGraph
            errors = []
            warnings = []
            has_cycles = False

            # Check for cycles
            try:
                visited = set()
                rec_stack = set()

                def _detect_cycles(node_id, visited, rec_stack):
                    visited.add(node_id)
                    rec_stack.add(node_id)

                    for neighbor in task_graph.edges.get(node_id, []):
                        if neighbor not in visited:
                            if _detect_cycles(neighbor, visited, rec_stack):
                                return True
                        elif neighbor in rec_stack:
                            return True

                    rec_stack.remove(node_id)
                    return False

                for node_id in task_graph.nodes:
                    if node_id not in visited:
                        if _detect_cycles(node_id, visited, rec_stack):
                            has_cycles = True
                            break
            except Exception:
                pass

            # Check for missing dependencies
            for task_id, task in task_graph.nodes.items():
                for dep_id in task.depends_on:
                    if dep_id not in task_graph.nodes:
                        errors.append(f"Missing dependency: {dep_id} for task {task_id}")

            # Check for orphaned tasks (no path from root)
            if task_graph.root_task_id is not None:
                for task_id in task_graph.nodes:
                    if task_id == task_graph.root_task_id:
                        continue

                    # Check if task is reachable from root using BFS
                    visited = set()
                    queue = [task_graph.root_task_id]
                    visited.add(task_graph.root_task_id)

                    found = False
                    while queue and not found:
                        current = queue.pop(0)
                        if current == task_id:
                            found = True
                            break

                        for neighbor in task_graph.edges.get(current, []):
                            if neighbor not in visited:
                                visited.add(neighbor)
                                queue.append(neighbor)

                    if not found:
                        warnings.append(f"No path from root to task: {task_id}")

            is_valid = len(errors) == 0 and not has_cycles

            if has_cycles:
                errors.append("Graph contains circular dependencies")

            return ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                has_cycles=has_cycles,
            )

        # New API: validate internal state (use existing implementation)
        cycles = self.detect_cycles()

        errors = []
        warnings = []

        if cycles:
            has_cycles = True
            cycle_strs = [" -> ".join(cycle) for cycle in cycles[:3]]
            errors.append(f"Circular dependencies detected: {', '.join(cycle_strs)}")
        else:
            has_cycles = False

        # Check for orphaned tasks (no dependencies and no dependents)
        for task_id in self._task_nodes:
            has_deps = self._graph.in_degree(task_id) > 0
            has_dependents = self._graph.out_degree(task_id) > 0

            if not has_deps and not has_dependents and len(self._task_nodes) > 1:
                warnings.append(f"Orphaned task: {task_id}")

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            has_cycles=has_cycles,
        )


# ============================================================================
# Legacy classes for backward compatibility with hierarchical_planner
# ============================================================================


@dataclass
class TaskLegacy:
    """Legacy Task class for backward compatibility.

    Deprecated: Use SimpleTask instead.
    """

    id: str
    description: str
    depends_on: List[str] = field(default_factory=list)
    estimated_complexity: int = 5
    context: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending, in_progress, completed, failed

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "description": self.description,
            "depends_on": self.depends_on,
            "estimated_complexity": self.estimated_complexity,
            "context": self.context,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskLegacy":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            description=data["description"],
            depends_on=data.get("depends_on", []),
            estimated_complexity=data.get("estimated_complexity", 5),
            context=data.get("context", {}),
            status=data.get("status", "pending"),
        )


@dataclass
class TaskGraph:
    """Legacy TaskGraph for backward compatibility.

    Deprecated: Use TaskDecomposition with NetworkX instead.
    """

    nodes: Dict[str, TaskLegacy] = field(default_factory=dict)
    edges: Dict[str, List[str]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    root_task_id: Optional[str] = None

    def add_node(self, task: TaskLegacy) -> None:
        """Add a task node to the graph."""
        self.nodes[task.id] = task
        if task.id not in self.edges:
            self.edges[task.id] = []

    def add_edge(self, from_id: str, to_id: str) -> None:
        """Add a dependency edge between tasks."""
        if from_id not in self.edges:
            self.edges[from_id] = []
        if to_id not in self.edges:
            self.edges[to_id] = []

        if to_id not in self.edges[from_id]:
            self.edges[from_id].append(to_id)

    def get_dependencies(self, task_id: str) -> List[str]:
        """Get dependencies for a task (tasks that must complete before this one)."""
        # Return tasks that this task depends on (predecessors)
        # With edges[dep] = [task], we need to find all dep where task_id in edges[dep]
        dependencies = []
        for from_id, to_ids in self.edges.items():
            if task_id in to_ids:
                dependencies.append(from_id)
        return dependencies

    def get_dependents(self, task_id: str) -> List[str]:
        """Get tasks that depend on this task (tasks that require this task to complete)."""
        # Return tasks that depend on this task (successors)
        # With edges[task_id] = [successors], we can return edges[task_id] directly
        return self.edges.get(task_id, [])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "nodes": {tid: task.to_dict() for tid, task in self.nodes.items()},
            "edges": self.edges.copy(),
            "metadata": self.metadata,
            "root_task_id": self.root_task_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskGraph":
        """Create from dictionary."""
        graph = cls()
        graph.nodes = {
            tid: TaskLegacy.from_dict(task_data) for tid, task_data in data.get("nodes", {}).items()
        }
        graph.edges = data.get("edges", {})
        graph.metadata = data.get("metadata", {})
        graph.root_task_id = data.get("root_task_id")
        return graph


@dataclass
class ComplexityScore:
    """Complexity score for a task or plan."""

    score: float
    confidence: float
    factors: List[str] = field(default_factory=list)
    estimated_steps: int = 5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "score": self.score,
            "confidence": self.confidence,
            "factors": self.factors,
            "estimated_steps": self.estimated_steps,
        }


@dataclass
class ValidationResult:
    """Result of validating a task graph."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    has_cycles: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "has_cycles": self.has_cycles,
        }


@dataclass
class UpdatedPlan:
    """Result of updating a plan after task execution."""

    graph: TaskGraph
    new_ready_tasks: List[TaskLegacy] = field(default_factory=list)
    completed_tasks: List[str] = field(default_factory=list)
    failed_tasks: List[str] = field(default_factory=list)
    can_proceed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "graph": self.graph.to_dict(),
            "new_ready_tasks": [t.to_dict() for t in self.new_ready_tasks],
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "can_proceed": self.can_proceed,
        }


# Type alias for backward compatibility
Task = TaskLegacy


__all__ = [
    # New NetworkX-based implementation
    "SimpleTask",
    "TaskStatus",
    "DependencyType",
    "TaskNode",
    "DependencyEdge",
    "TaskDecomposition",
    # Legacy classes for backward compatibility
    "Task",
    "TaskGraph",
    "ComplexityScore",
    "ValidationResult",
    "UpdatedPlan",
]
