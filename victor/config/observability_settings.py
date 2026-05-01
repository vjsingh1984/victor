"""Logging, observability, and analytics configuration."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ObservabilitySettings(BaseModel):
    """Logging, observability, and analytics configuration."""

    log_level: str = "INFO"
    log_file: Optional[str] = None
    enable_observability_logging: bool = False
    observability_log_path: Optional[str] = None
    analytics_enabled: bool = True

    # Native dispatch observability
    # When enabled, adds timing and backend tracking to flat dispatch functions
    # in victor.processing.native modules. Minimal overhead (<1% when enabled).
    native_observability_enabled: bool = Field(
        default=True,
        description="Enable observability (timing, backend tracking) for native dispatch functions",
    )
