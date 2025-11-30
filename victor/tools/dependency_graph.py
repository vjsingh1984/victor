from dataclasses import dataclass, field
from typing import Dict, List, Set


@dataclass
class ToolSpec:
    name: str
    inputs: Set[str] = field(default_factory=set)
    outputs: Set[str] = field(default_factory=set)


class ToolDependencyGraph:
    """Minimal dependency graph for planning multi-step tool execution."""

    def __init__(self) -> None:
        self.tools: Dict[str, ToolSpec] = {}
        self.output_index: Dict[str, Set[str]] = {}

    def add_tool(self, name: str, inputs: List[str], outputs: List[str]) -> None:
        spec = ToolSpec(name=name, inputs=set(inputs), outputs=set(outputs))
        self.tools[name] = spec
        for out in spec.outputs:
            self.output_index.setdefault(out, set()).add(name)

    def plan(self, goals: List[str], available: List[str]) -> List[str]:
        """Return a tool execution order to satisfy goals or [] if impossible."""
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
            tool_name = sorted(providers)[0]
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
