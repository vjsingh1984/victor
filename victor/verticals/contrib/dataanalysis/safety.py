"""Compatibility shim for Data Analysis runtime safety helpers."""

from victor.verticals.contrib.dataanalysis.runtime import safety as _runtime_safety
from victor.verticals.contrib.dataanalysis.runtime.safety import *  # noqa: F401,F403

__all__ = list(_runtime_safety.__all__)
