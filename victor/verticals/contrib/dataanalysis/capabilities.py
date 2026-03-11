"""Compatibility shim for Data Analysis runtime capability helpers."""

from victor.verticals.contrib.dataanalysis.runtime import capabilities as _runtime_capabilities
from victor.verticals.contrib.dataanalysis.runtime.capabilities import *  # noqa: F401,F403

__all__ = list(_runtime_capabilities.__all__)
