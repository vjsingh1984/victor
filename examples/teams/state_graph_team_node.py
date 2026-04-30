"""Compatibility wrapper for the renamed coordinator example.

Prefer running ``examples/teams/state_graph_team_coordinator.py``.
"""

from __future__ import annotations

import asyncio

from state_graph_team_coordinator import main


if __name__ == "__main__":
    asyncio.run(main())
