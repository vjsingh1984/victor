from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from victor.agent.tool_selection import ToolSelectionStats


@dataclass
class ToolSelectionRecorder:
    stats: ToolSelectionStats
    on_selection_recorded: Callable[[str, int], None] | None = None

    def record(self, method: str, num_tools: int) -> None:
        self.stats.record_selection(method, num_tools)
        if self.on_selection_recorded is not None:
            self.on_selection_recorded(method, num_tools)

    def record_result(self, *, is_fallback: bool, num_tools: int) -> None:
        self.record("fallback" if is_fallback else "semantic", num_tools)
