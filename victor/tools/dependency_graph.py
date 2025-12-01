from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from victor.tools.base import CostTier


@dataclass
class ToolSpec:
    name: str
    inputs: Set[str] = field(default_factory=set)
    outputs: Set[str] = field(default_factory=set)
    cost_tier: CostTier = CostTier.FREE


class ToolDependencyGraph:
    """Minimal dependency graph for planning multi-step tool execution.

    Supports cost-aware planning to prefer lower-cost tools when multiple
    tools can satisfy the same goal.
    """

    def __init__(self, cost_aware: bool = True) -> None:
        self.tools: Dict[str, ToolSpec] = {}
        self.output_index: Dict[str, Set[str]] = {}
        self.cost_aware = cost_aware

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
        """Select the best provider tool, preferring lower-cost options.

        Args:
            providers: Set of tool names that can provide the needed output

        Returns:
            The selected tool name
        """
        if not self.cost_aware or len(providers) == 1:
            return sorted(providers)[0]

        # Sort by cost tier (ascending), then by name for determinism
        ranked = sorted(
            providers,
            key=lambda t: (
                self.tools[t].cost_tier.weight if t in self.tools else 0,
                t,
            ),
        )
        return ranked[0]

    def plan(self, goals: List[str], available: List[str]) -> List[str]:
        """Return a tool execution order to satisfy goals or [] if impossible.

        Uses cost-aware selection to prefer lower-cost tools when multiple
        tools can satisfy the same goal.

        Args:
            goals: List of output goals to satisfy
            available: List of already-available inputs

        Returns:
            Ordered list of tool names to execute, or [] if impossible
        """
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
            # Use cost-aware provider selection
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
        """Calculate total cost weight for a plan.

        Args:
            plan: List of tool names in execution order

        Returns:
            Total cost weight of the plan
        """
        total = 0.0
        for tool_name in plan:
            if tool_name in self.tools:
                total += self.tools[tool_name].cost_tier.weight
        return total
