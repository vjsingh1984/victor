from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from victor.tools.base import CostTier


@dataclass
class ToolSpec:
    name: str
    inputs: Set[str] = field(default_factory=set)
    outputs: Set[str] = field(default_factory=set)
    cost_tier: CostTier = CostTier.FREE


class ToolDependencyGraph:
    """Dependency graph for planning and learning multi-step tool execution.

    Provides two capabilities:
    1. Cost-aware planning: resolve execution order for a set of goals.
    2. Trajectory learning: record tool→tool transitions and predict next tools.
    """

    def __init__(self, cost_aware: bool = True) -> None:
        self.tools: Dict[str, ToolSpec] = {}
        self.output_index: Dict[str, Set[str]] = {}
        self.cost_aware = cost_aware
        # Explicit dependency edges (for protocol compatibility)
        self._dependencies: Dict[str, List[str]] = {}
        # Transition counter: (from_tool, task_type) → {to_tool: count}
        self._transitions: Dict[Tuple[str, str], Counter] = {}

    # ------------------------------------------------------------------
    # ToolDependencyGraphProtocol compatibility
    # ------------------------------------------------------------------

    def add_dependency(self, tool: str, depends_on: str) -> None:
        """Register an explicit dependency: `tool` must run after `depends_on`."""
        self._dependencies.setdefault(tool, [])
        if depends_on not in self._dependencies[tool]:
            self._dependencies[tool].append(depends_on)

    def get_dependencies(self, tool: str) -> List[str]:
        """Return direct dependencies for `tool`."""
        return list(self._dependencies.get(tool, []))

    def get_execution_order(self, tools: List[str]) -> List[str]:
        """Return `tools` sorted so each tool's dependencies precede it.

        Uses topological sort over the explicit dependency graph.  Tools
        with no registered dependencies keep their original relative order.
        """
        if len(tools) <= 1:
            return list(tools)

        tool_set = set(tools)
        visited: Set[str] = set()
        order: List[str] = []

        def visit(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            for dep in self._dependencies.get(name, []):
                if dep in tool_set:
                    visit(dep)
            order.append(name)

        for t in tools:
            visit(t)

        return order

    # ------------------------------------------------------------------
    # Cost-aware planning (original capability)
    # ------------------------------------------------------------------

    def add_tool(
        self,
        name: str,
        inputs: List[str],
        outputs: List[str],
        cost_tier: Optional[CostTier] = None,
    ) -> None:
        spec = ToolSpec(
            name=name,
            inputs=set(inputs),
            outputs=set(outputs),
            cost_tier=cost_tier or CostTier.FREE,
        )
        self.tools[name] = spec
        for out in spec.outputs:
            self.output_index.setdefault(out, set()).add(name)

    def _select_provider(self, providers: Set[str]) -> str:
        if not self.cost_aware or len(providers) == 1:
            return sorted(providers)[0]
        ranked = sorted(
            providers,
            key=lambda t: (
                self.tools[t].cost_tier.weight if t in self.tools else 0,
                t,
            ),
        )
        return ranked[0]

    def plan(self, goals: List[str], available: List[str]) -> List[str]:
        """Return tool execution order to satisfy goals, or [] if impossible."""
        plan: List[str] = []
        satisfied = set(available)
        visiting: Set[str] = set()
        memo: Dict[str, bool] = {}

        def resolve(goal: str) -> bool:
            if goal in satisfied:
                return True
            providers = self.output_index.get(goal, set())
            if not providers:
                return False
            tool_name = self._select_provider(providers)
            if memo.get(tool_name):
                return True
            if tool_name in visiting:
                return False  # cycle
            visiting.add(tool_name)
            spec = self.tools[tool_name]
            for req in spec.inputs:
                if not resolve(req):
                    visiting.remove(tool_name)
                    return False
            if tool_name not in plan:
                plan.append(tool_name)
            satisfied.update(spec.outputs)
            visiting.remove(tool_name)
            memo[tool_name] = True
            return True

        for goal in goals:
            if not resolve(goal):
                return []

        return plan

    def get_total_plan_cost(self, plan: List[str]) -> float:
        total = 0.0
        for tool_name in plan:
            if tool_name in self.tools:
                total += self.tools[tool_name].cost_tier.weight
        return total

    # ------------------------------------------------------------------
    # Trajectory learning
    # ------------------------------------------------------------------

    def record_transition(self, from_tool: str, to_tool: str, task_type: str) -> None:
        """Record a tool→tool transition for trajectory learning.

        Called after each tool execution to accumulate statistics used by
        `predict_next()` for pre-warming/prefetching.
        """
        key = (from_tool, task_type)
        if key not in self._transitions:
            self._transitions[key] = Counter()
        self._transitions[key][to_tool] += 1

    def predict_next(
        self, current_tool: str, task_type: str, top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """Predict the most likely next tools given the current tool and task type.

        Returns a list of (tool_name, probability) tuples sorted by probability
        descending, or an empty list if no transitions have been recorded yet.
        """
        key = (current_tool, task_type)
        counts = self._transitions.get(key)
        if not counts:
            return []
        total = sum(counts.values())
        return [(t, n / total) for t, n in counts.most_common(top_k)]

    def get_preready_tools(
        self, current_tool: str, task_type: str, min_probability: float = 0.3
    ) -> List[str]:
        """Return tools likely to be needed next (probability >= min_probability)."""
        return [t for t, p in self.predict_next(current_tool, task_type) if p >= min_probability]
