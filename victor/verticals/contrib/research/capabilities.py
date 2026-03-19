"""Compatibility shim for Research runtime capability helpers."""

from victor.verticals.contrib.research.runtime import capabilities as _runtime_capabilities
from victor.verticals.contrib.research.runtime.capabilities import *  # noqa: F401,F403

__all__ = list(_runtime_capabilities.__all__)
