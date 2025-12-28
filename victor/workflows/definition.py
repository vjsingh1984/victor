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

"""Workflow definition DSL.

Provides a LangGraph-like fluent API for defining multi-agent workflows
as directed acyclic graphs (DAGs).

Example:
    @workflow("code_review", "Comprehensive code review")
    def code_review_workflow():
        return (
            WorkflowBuilder("code_review")
            .add_agent("analyze", "researcher", "Find code patterns")
            .add_condition("decide", lambda ctx: "fix" if ctx.get("issues") else "done",
                          {"fix": "fixer", "done": "report"})
            .add_agent("fixer", "executor", "Fix issues", next_nodes=["report"])
            .add_agent("report", "planner", "Summarize findings")
            .build()
        )
"""

from __future__ import annotations

import functools
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.agent.subagents import SubAgentRole

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Type of workflow node."""

    AGENT = "agent"  # Spawns an agent to execute
    CONDITION = "condition"  # Branch based on condition
    PARALLEL = "parallel"  # Execute multiple nodes in parallel
    TRANSFORM = "transform"  # Transform context data
    HITL = "hitl"  # Human-in-the-loop interrupt
    START = "start"  # Entry point
    END = "end"  # Terminal node


@dataclass
class WorkflowNode(ABC):
    """Base class for workflow nodes.

    Attributes:
        id: Unique identifier for this node
        name: Human-readable name
        next_nodes: IDs of nodes to execute after this one
    """

    id: str
    name: str
    next_nodes: List[str] = field(default_factory=list)

    @property
    @abstractmethod
    def node_type(self) -> NodeType:
        """The type of this node."""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Serialize node to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.node_type.value,
            "next_nodes": self.next_nodes,
        }


@dataclass
class AgentNode(WorkflowNode):
    """Node that spawns an agent to execute a task.

    Attributes:
        role: Agent role (researcher, executor, etc.)
        goal: Task description for the agent
        tool_budget: Maximum tool calls allowed
        allowed_tools: Specific tools to allow (None = role defaults)
        input_mapping: Map context keys to agent inputs
        output_key: Key to store agent output in context
    """

    role: str = "executor"
    goal: str = ""
    tool_budget: int = 15
    allowed_tools: Optional[List[str]] = None
    input_mapping: Dict[str, str] = field(default_factory=dict)
    output_key: Optional[str] = None

    @property
    def node_type(self) -> NodeType:
        return NodeType.AGENT

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "role": self.role,
            "goal": self.goal,
            "tool_budget": self.tool_budget,
            "allowed_tools": self.allowed_tools,
            "input_mapping": self.input_mapping,
            "output_key": self.output_key,
        })
        return d


@dataclass
class ConditionNode(WorkflowNode):
    """Node that branches based on a condition.

    Attributes:
        condition: Function that takes context and returns branch name
        branches: Mapping from branch names to node IDs
    """

    condition: Callable[[Dict[str, Any]], str] = field(default=lambda ctx: "default")
    branches: Dict[str, str] = field(default_factory=dict)

    @property
    def node_type(self) -> NodeType:
        return NodeType.CONDITION

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "branches": self.branches,
            # Note: condition function is not serialized
        })
        return d

    def evaluate(self, context: Dict[str, Any]) -> Optional[str]:
        """Evaluate condition and return next node ID."""
        branch = self.condition(context)
        return self.branches.get(branch)


@dataclass
class ParallelNode(WorkflowNode):
    """Node that executes multiple nodes in parallel.

    Attributes:
        parallel_nodes: IDs of nodes to execute in parallel
        join_strategy: How to combine results (all, any, merge)
    """

    parallel_nodes: List[str] = field(default_factory=list)
    join_strategy: str = "all"  # all, any, merge

    @property
    def node_type(self) -> NodeType:
        return NodeType.PARALLEL

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "parallel_nodes": self.parallel_nodes,
            "join_strategy": self.join_strategy,
        })
        return d


@dataclass
class TransformNode(WorkflowNode):
    """Node that transforms context data.

    Attributes:
        transform: Function that transforms context
    """

    transform: Callable[[Dict[str, Any]], Dict[str, Any]] = field(
        default=lambda ctx: ctx
    )

    @property
    def node_type(self) -> NodeType:
        return NodeType.TRANSFORM


@dataclass
class WorkflowDefinition:
    """A complete workflow definition.

    Represents a DAG of nodes that define a multi-agent workflow.

    Attributes:
        name: Workflow name (used as identifier)
        description: Human-readable description
        nodes: All nodes in the workflow
        start_node: ID of the entry point node
        metadata: Additional workflow metadata
    """

    name: str
    description: str = ""
    nodes: Dict[str, WorkflowNode] = field(default_factory=dict)
    start_node: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate workflow structure."""
        if self.nodes and not self.start_node:
            # Find first node added as start
            self.start_node = next(iter(self.nodes.keys()))

    def get_node(self, node_id: str) -> Optional[WorkflowNode]:
        """Get node by ID."""
        return self.nodes.get(node_id)

    def get_next_nodes(self, node_id: str) -> List[WorkflowNode]:
        """Get nodes that follow the given node."""
        node = self.nodes.get(node_id)
        if not node:
            return []
        return [self.nodes[nid] for nid in node.next_nodes if nid in self.nodes]

    def validate(self) -> List[str]:
        """Validate workflow and return list of errors."""
        errors = []

        if not self.name:
            errors.append("Workflow must have a name")

        if not self.nodes:
            errors.append("Workflow must have at least one node")
            return errors

        if not self.start_node:
            errors.append("Workflow must have a start node")
        elif self.start_node not in self.nodes:
            errors.append(f"Start node '{self.start_node}' not found")

        # Check for unreachable nodes
        reachable = self._find_reachable_nodes()
        for node_id in self.nodes:
            if node_id not in reachable and node_id != self.start_node:
                errors.append(f"Node '{node_id}' is unreachable")

        # Check for broken references
        for node in self.nodes.values():
            for next_id in node.next_nodes:
                if next_id not in self.nodes:
                    errors.append(
                        f"Node '{node.id}' references non-existent node '{next_id}'"
                    )

            if isinstance(node, ConditionNode):
                for branch, target in node.branches.items():
                    if target not in self.nodes:
                        errors.append(
                            f"Condition '{node.id}' branch '{branch}' "
                            f"references non-existent node '{target}'"
                        )

            if isinstance(node, ParallelNode):
                for parallel_id in node.parallel_nodes:
                    if parallel_id not in self.nodes:
                        errors.append(
                            f"Parallel '{node.id}' references non-existent "
                            f"node '{parallel_id}'"
                        )

        return errors

    def _find_reachable_nodes(self) -> Set[str]:
        """Find all nodes reachable from start."""
        if not self.start_node:
            return set()

        reachable = set()
        to_visit = [self.start_node]

        while to_visit:
            node_id = to_visit.pop()
            if node_id in reachable:
                continue
            reachable.add(node_id)

            node = self.nodes.get(node_id)
            if not node:
                continue

            to_visit.extend(node.next_nodes)

            if isinstance(node, ConditionNode):
                to_visit.extend(node.branches.values())

            if isinstance(node, ParallelNode):
                to_visit.extend(node.parallel_nodes)

        return reachable

    def to_dict(self) -> Dict[str, Any]:
        """Serialize workflow to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "start_node": self.start_node,
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "metadata": self.metadata,
        }

    def get_agent_count(self) -> int:
        """Count agent nodes in workflow."""
        return sum(
            1 for node in self.nodes.values() if isinstance(node, AgentNode)
        )

    def get_total_budget(self) -> int:
        """Sum of all agent tool budgets."""
        return sum(
            node.tool_budget
            for node in self.nodes.values()
            if isinstance(node, AgentNode)
        )


class WorkflowBuilder:
    """Fluent builder for creating workflow definitions.

    Provides a chainable API for constructing workflows:

    Example:
        workflow = (
            WorkflowBuilder("review")
            .add_agent("analyze", "researcher", "Find issues")
            .add_condition("decide", decide_func, {"fix": "fixer", "done": "end"})
            .add_agent("fixer", "executor", "Fix issues", next_nodes=["end"])
            .build()
        )
    """

    def __init__(self, name: str, description: str = ""):
        """Initialize builder.

        Args:
            name: Workflow name
            description: Optional description
        """
        self.name = name
        self.description = description
        self._nodes: Dict[str, WorkflowNode] = {}
        self._first_node: Optional[str] = None
        self._last_node: Optional[str] = None
        self._metadata: Dict[str, Any] = {}

    def add_agent(
        self,
        node_id: str,
        role: str,
        goal: str,
        *,
        name: Optional[str] = None,
        tool_budget: int = 15,
        allowed_tools: Optional[List[str]] = None,
        input_mapping: Optional[Dict[str, str]] = None,
        output_key: Optional[str] = None,
        next_nodes: Optional[List[str]] = None,
    ) -> "WorkflowBuilder":
        """Add an agent node.

        Args:
            node_id: Unique identifier
            role: Agent role (researcher, executor, etc.)
            goal: Task description
            name: Optional display name
            tool_budget: Max tool calls
            allowed_tools: Specific tools to allow
            input_mapping: Context key mappings
            output_key: Key for storing output
            next_nodes: Nodes to execute after

        Returns:
            Self for chaining
        """
        node = AgentNode(
            id=node_id,
            name=name or node_id,
            role=role,
            goal=goal,
            tool_budget=tool_budget,
            allowed_tools=allowed_tools,
            input_mapping=input_mapping or {},
            output_key=output_key or node_id,
            next_nodes=next_nodes or [],
        )
        return self._add_node(node)

    def add_condition(
        self,
        node_id: str,
        condition: Callable[[Dict[str, Any]], str],
        branches: Dict[str, str],
        *,
        name: Optional[str] = None,
    ) -> "WorkflowBuilder":
        """Add a condition node.

        Args:
            node_id: Unique identifier
            condition: Function taking context, returning branch name
            branches: Map from branch names to node IDs
            name: Optional display name

        Returns:
            Self for chaining
        """
        node = ConditionNode(
            id=node_id,
            name=name or node_id,
            condition=condition,
            branches=branches,
        )
        return self._add_node(node)

    def add_parallel(
        self,
        node_id: str,
        parallel_nodes: List[str],
        *,
        name: Optional[str] = None,
        join_strategy: str = "all",
        next_nodes: Optional[List[str]] = None,
    ) -> "WorkflowBuilder":
        """Add a parallel execution node.

        Args:
            node_id: Unique identifier
            parallel_nodes: Nodes to execute in parallel
            name: Optional display name
            join_strategy: How to combine results (all, any, merge)
            next_nodes: Nodes to execute after

        Returns:
            Self for chaining
        """
        node = ParallelNode(
            id=node_id,
            name=name or node_id,
            parallel_nodes=parallel_nodes,
            join_strategy=join_strategy,
            next_nodes=next_nodes or [],
        )
        return self._add_node(node)

    def add_transform(
        self,
        node_id: str,
        transform: Callable[[Dict[str, Any]], Dict[str, Any]],
        *,
        name: Optional[str] = None,
        next_nodes: Optional[List[str]] = None,
    ) -> "WorkflowBuilder":
        """Add a transform node.

        Args:
            node_id: Unique identifier
            transform: Function to transform context
            name: Optional display name
            next_nodes: Nodes to execute after

        Returns:
            Self for chaining
        """
        node = TransformNode(
            id=node_id,
            name=name or node_id,
            transform=transform,
            next_nodes=next_nodes or [],
        )
        return self._add_node(node)

    def add_hitl_approval(
        self,
        node_id: str,
        prompt: str,
        *,
        name: Optional[str] = None,
        context_keys: Optional[List[str]] = None,
        timeout: float = 300.0,
        fallback: str = "abort",
        next_nodes: Optional[List[str]] = None,
    ) -> "WorkflowBuilder":
        """Add a human approval gate.

        Pauses workflow execution until a human approves or rejects.

        Args:
            node_id: Unique identifier
            prompt: Message to display for approval
            name: Optional display name
            context_keys: Keys from context to include in display
            timeout: Timeout in seconds (default: 5 minutes)
            fallback: Behavior on timeout (abort, continue, skip)
            next_nodes: Nodes to execute after approval

        Returns:
            Self for chaining

        Example:
            workflow.add_hitl_approval(
                "approve_changes",
                prompt="Proceed with the following changes?",
                context_keys=["files_to_modify"],
                timeout=300.0,
            )
        """
        from victor.workflows.hitl import HITLNode, HITLNodeType, HITLFallback

        fallback_enum = HITLFallback(fallback) if fallback else HITLFallback.ABORT

        node = HITLNode(
            id=node_id,
            name=name or node_id,
            hitl_type=HITLNodeType.APPROVAL,
            prompt=prompt,
            context_keys=context_keys or [],
            timeout=timeout,
            fallback=fallback_enum,
            next_nodes=next_nodes or [],
        )
        return self._add_node(node)

    def add_hitl_choice(
        self,
        node_id: str,
        prompt: str,
        choices: List[str],
        *,
        name: Optional[str] = None,
        context_keys: Optional[List[str]] = None,
        default_value: Optional[str] = None,
        timeout: float = 300.0,
        fallback: str = "continue",
        next_nodes: Optional[List[str]] = None,
    ) -> "WorkflowBuilder":
        """Add a human choice selection node.

        Pauses workflow until human selects from options.

        Args:
            node_id: Unique identifier
            prompt: Message to display
            choices: Available options
            name: Optional display name
            context_keys: Keys from context to include in display
            default_value: Default choice if timeout
            timeout: Timeout in seconds
            fallback: Behavior on timeout
            next_nodes: Nodes to execute after

        Returns:
            Self for chaining
        """
        from victor.workflows.hitl import HITLNode, HITLNodeType, HITLFallback

        fallback_enum = HITLFallback(fallback) if fallback else HITLFallback.CONTINUE

        node = HITLNode(
            id=node_id,
            name=name or node_id,
            hitl_type=HITLNodeType.CHOICE,
            prompt=prompt,
            context_keys=context_keys or [],
            choices=choices,
            default_value=default_value,
            timeout=timeout,
            fallback=fallback_enum,
            next_nodes=next_nodes or [],
        )
        return self._add_node(node)

    def add_hitl_review(
        self,
        node_id: str,
        prompt: str,
        *,
        name: Optional[str] = None,
        context_keys: Optional[List[str]] = None,
        timeout: float = 600.0,
        fallback: str = "abort",
        next_nodes: Optional[List[str]] = None,
    ) -> "WorkflowBuilder":
        """Add a human review/modification node.

        Pauses workflow until human reviews and optionally modifies context.

        Args:
            node_id: Unique identifier
            prompt: Message to display
            name: Optional display name
            context_keys: Keys from context to include in review
            timeout: Timeout in seconds (default: 10 minutes)
            fallback: Behavior on timeout
            next_nodes: Nodes to execute after

        Returns:
            Self for chaining
        """
        from victor.workflows.hitl import HITLNode, HITLNodeType, HITLFallback

        fallback_enum = HITLFallback(fallback) if fallback else HITLFallback.ABORT

        node = HITLNode(
            id=node_id,
            name=name or node_id,
            hitl_type=HITLNodeType.REVIEW,
            prompt=prompt,
            context_keys=context_keys or [],
            timeout=timeout,
            fallback=fallback_enum,
            next_nodes=next_nodes or [],
        )
        return self._add_node(node)

    def add_hitl_confirmation(
        self,
        node_id: str,
        prompt: str = "Press Enter to continue",
        *,
        name: Optional[str] = None,
        timeout: float = 60.0,
        fallback: str = "continue",
        next_nodes: Optional[List[str]] = None,
    ) -> "WorkflowBuilder":
        """Add a simple confirmation gate.

        Pauses briefly for human to acknowledge before continuing.

        Args:
            node_id: Unique identifier
            prompt: Message to display
            name: Optional display name
            timeout: Timeout in seconds (default: 1 minute)
            fallback: Behavior on timeout (default: continue)
            next_nodes: Nodes to execute after

        Returns:
            Self for chaining
        """
        from victor.workflows.hitl import HITLNode, HITLNodeType, HITLFallback

        fallback_enum = HITLFallback(fallback) if fallback else HITLFallback.CONTINUE

        node = HITLNode(
            id=node_id,
            name=name or node_id,
            hitl_type=HITLNodeType.CONFIRMATION,
            prompt=prompt,
            timeout=timeout,
            fallback=fallback_enum,
            next_nodes=next_nodes or [],
        )
        return self._add_node(node)

    def set_metadata(self, key: str, value: Any) -> "WorkflowBuilder":
        """Set workflow metadata.

        Args:
            key: Metadata key
            value: Metadata value

        Returns:
            Self for chaining
        """
        self._metadata[key] = value
        return self

    def _add_node(self, node: WorkflowNode) -> "WorkflowBuilder":
        """Add a node to the workflow."""
        if node.id in self._nodes:
            raise ValueError(f"Node '{node.id}' already exists")

        # Auto-chain: connect previous node to this one
        if self._last_node and not self._nodes[self._last_node].next_nodes:
            # Only auto-chain if previous node has no explicit next_nodes
            # and is not a condition node (conditions have explicit branches)
            prev = self._nodes[self._last_node]
            if not isinstance(prev, ConditionNode):
                prev.next_nodes.append(node.id)

        self._nodes[node.id] = node
        if self._first_node is None:
            self._first_node = node.id
        self._last_node = node.id

        return self

    def chain(self, from_node: str, to_node: str) -> "WorkflowBuilder":
        """Explicitly chain one node to another.

        Args:
            from_node: Source node ID
            to_node: Target node ID

        Returns:
            Self for chaining
        """
        if from_node not in self._nodes:
            raise ValueError(f"Source node '{from_node}' not found")
        if to_node not in self._nodes:
            raise ValueError(f"Target node '{to_node}' not found")

        if to_node not in self._nodes[from_node].next_nodes:
            self._nodes[from_node].next_nodes.append(to_node)

        return self

    def build(self) -> WorkflowDefinition:
        """Build the workflow definition.

        Returns:
            Complete WorkflowDefinition

        Raises:
            ValueError: If workflow validation fails
        """
        workflow = WorkflowDefinition(
            name=self.name,
            description=self.description,
            nodes=self._nodes.copy(),
            start_node=self._first_node,
            metadata=self._metadata.copy(),
        )

        errors = workflow.validate()
        if errors:
            raise ValueError(f"Workflow validation failed: {'; '.join(errors)}")

        return workflow


# Registry for decorator-registered workflows
_workflow_registry: Dict[str, Callable[[], WorkflowDefinition]] = {}


def workflow(
    name: str,
    description: str = "",
) -> Callable[[Callable[[], WorkflowDefinition]], Callable[[], WorkflowDefinition]]:
    """Decorator to register a workflow factory function.

    The decorated function should return a WorkflowDefinition.

    Example:
        @workflow("code_review", "Review code quality")
        def code_review_workflow():
            return (
                WorkflowBuilder("code_review")
                .add_agent("analyze", "researcher", "Find issues")
                .build()
            )

    Args:
        name: Workflow name
        description: Optional description

    Returns:
        Decorator function
    """

    def decorator(
        func: Callable[[], WorkflowDefinition]
    ) -> Callable[[], WorkflowDefinition]:
        @functools.wraps(func)
        def wrapper() -> WorkflowDefinition:
            defn = func()
            # Override name/description from decorator
            defn.name = name
            if description:
                defn.description = description
            return defn

        _workflow_registry[name] = wrapper
        logger.debug(f"Registered workflow: {name}")
        return wrapper

    return decorator


def get_registered_workflows() -> Dict[str, Callable[[], WorkflowDefinition]]:
    """Get all registered workflow factories."""
    return _workflow_registry.copy()


__all__ = [
    "NodeType",
    "WorkflowNode",
    "AgentNode",
    "ConditionNode",
    "ParallelNode",
    "TransformNode",
    "WorkflowDefinition",
    "WorkflowBuilder",
    "workflow",
    "get_registered_workflows",
]
