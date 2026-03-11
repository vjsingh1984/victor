"""Compatibility shim for Data Analysis runtime persona helpers."""

from victor.verticals.contrib.dataanalysis.runtime import team_personas as _runtime_team_personas
from victor.verticals.contrib.dataanalysis.runtime.team_personas import *  # noqa: F401,F403

__all__ = list(_runtime_team_personas.__all__)
