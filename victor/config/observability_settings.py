"""Logging, observability, and analytics configuration."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class ObservabilitySettings(BaseModel):
    """Logging, observability, and analytics configuration."""

    log_level: str = "INFO"
    log_file: Optional[str] = None
    enable_observability_logging: bool = False
    observability_log_path: Optional[str] = None
    analytics_enabled: bool = True
