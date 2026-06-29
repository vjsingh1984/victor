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

    # Adaptive accelerator dispatch (closes the measurement→dispatch loop).
    # When enabled, get_preferred_backend() picks the empirically faster backend
    # from observed per-backend EWMA latency once enough samples accumulate,
    # falling back to the static benchmark prior on cold start.
    native_adaptive_dispatch_enabled: bool = Field(
        default=True,
        description="Pick accelerator backend from observed latency once enough samples exist",
    )
    native_adaptive_min_samples: int = Field(
        default=20,
        ge=1,
        description="Per-backend EWMA samples required before adaptive dispatch overrides the static benchmark",
    )
    native_adaptive_ewma_alpha: float = Field(
        default=0.3,
        ge=0.01,
        le=1.0,
        description="EWMA smoothing factor for adaptive dispatch (higher = weight recent samples more)",
    )
