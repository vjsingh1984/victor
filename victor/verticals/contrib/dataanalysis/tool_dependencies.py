"""Compatibility shim for Data Analysis runtime tool-dependency helpers."""

from victor.verticals.contrib.dataanalysis.runtime import tool_dependencies as _runtime_tool_dependencies
from victor.verticals.contrib.dataanalysis.runtime.tool_dependencies import *  # noqa: F401,F403

__all__ = list(_runtime_tool_dependencies.__all__)
