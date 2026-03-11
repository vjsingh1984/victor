"""Compatibility shim for Data Analysis runtime team helpers."""

from victor.verticals.contrib.dataanalysis.runtime import teams as _runtime_teams
from victor.verticals.contrib.dataanalysis.runtime.teams import *  # noqa: F401,F403

__all__ = list(_runtime_teams.__all__)
