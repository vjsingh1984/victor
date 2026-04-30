"""Programmatic StateGraph example using a team coordinator directly.

This example shows the preferred framework layering:

- ``StateGraph`` owns workflow order and control flow.
- ``UnifiedTeamCoordinator`` owns intra-team collaboration.
- ``select_formation`` is plugged in as a synchronous strategy.

Use ``TeamStep`` only when a declarative workflow surface needs an adapter.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional, TypedDict

from victor.framework.agentic_graph.team_selector import select_formation
from victor.framework.graph import END, StateGraph
from victor.teams import AgentMessage, StateGraphNodeConfig, TeamFormation, UnifiedTeamCoordinator


class ResearchState(TypedDict, total=False):
    task: str
    context: Dict[str, Any]
    result: str
    team_output: Dict[str, Any]


class DemoMember:
    """Minimal team member implementation for runnable examples."""

    def __init__(self, member_id: str, output: str) -> None:
        self._id = member_id
        self._output = output
        self._role = None

    @property
    def id(self) -> str:
        return self._id

    @property
    def role(self) -> Any:
        return self._role

    async def execute_task(self, task: str, context: Dict[str, Any]) -> str:
        scope = context.get("scope", "general")
        return f"{self._output} for '{task}' ({scope})"

    async def receive_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        return None


async def main() -> None:
    coordinator = UnifiedTeamCoordinator(enable_observability=False, enable_rl=False)
    coordinator.add_member(DemoMember("researcher", "Collected findings"))
    coordinator.add_member(DemoMember("summarizer", "Summarized findings"))
    coordinator.set_formation(TeamFormation.SEQUENTIAL)
    coordinator.with_state_graph_config(
        StateGraphNodeConfig(formation_strategy=select_formation)
    )

    graph = StateGraph(ResearchState)
    graph.add_node("research_team", coordinator)
    graph.add_edge("research_team", END)
    graph.set_entry_point("research_team")

    compiled = graph.compile()
    result = await compiled.invoke(
        {
            "task": "Map the authentication flow",
            "context": {
                "task_type": "research",
                "team_size": 2,
                "scope": "auth",
            },
        }
    )
    final_state = result.state

    print(json.dumps(final_state["team_output"], indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
