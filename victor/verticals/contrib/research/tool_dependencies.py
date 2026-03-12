"""Compatibility shim for Research runtime tool-dependency helpers."""

from victor.verticals.contrib.research.runtime import tool_dependencies as _runtime_tool_dependencies
from victor.verticals.contrib.research.runtime.tool_dependencies import *  # noqa: F401,F403

__all__ = list(_runtime_tool_dependencies.__all__)
