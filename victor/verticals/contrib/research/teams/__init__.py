"""Compatibility shim for Research runtime team helpers."""

from victor.verticals.contrib.research.runtime import teams as _runtime_teams
from victor.verticals.contrib.research.runtime.teams import *  # noqa: F401,F403

__all__ = list(_runtime_teams.__all__)
