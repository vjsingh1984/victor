"""Plugin system configuration."""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class PluginSettings(BaseModel):
    """Plugin system configuration."""

    plugin_enabled: bool = True
    plugin_packages: List[str] = Field(default_factory=list)
    plugin_disabled: List[str] = Field(default_factory=list)
    plugin_config: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
