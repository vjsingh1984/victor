"""Event system backend and configuration."""

from __future__ import annotations

from typing import Dict

from pydantic import BaseModel, Field


class EventSettings(BaseModel):
    """Event system backend and configuration."""

    event_backend_type: str = "in_memory"
    event_backend_lazy_init: bool = True
    event_delivery_guarantee: str = "at_most_once"
    event_max_batch_size: int = 100
    event_flush_interval_ms: float = 1000.0
    event_queue_maxsize: int = 10000
    event_queue_overflow_policy: str = "drop_newest"
    event_queue_overflow_block_timeout_ms: float = 50.0
    event_queue_overflow_topic_policies: Dict[str, str] = Field(
        default_factory=lambda: {
            "lifecycle.session.*": "block_with_timeout",
            "vertical.applied": "block_with_timeout",
            "error.*": "block_with_timeout",
            "core.events.emit_sync.metrics": "drop_oldest",
            "vertical.extensions.loader.metrics": "drop_oldest",
        }
    )
    event_queue_overflow_topic_block_timeout_ms: Dict[str, float] = Field(
        default_factory=lambda: {
            "lifecycle.session.*": 150.0,
            "vertical.applied": 120.0,
            "error.*": 200.0,
        }
    )
    event_emit_sync_metrics_enabled: bool = False
    event_emit_sync_metrics_interval_seconds: float = 60.0
    event_emit_sync_metrics_reset_after_emit: bool = False
    event_emit_sync_metrics_topic: str = "core.events.emit_sync.metrics"
    extension_loader_warn_queue_threshold: int = 24
    extension_loader_error_queue_threshold: int = 32
    extension_loader_warn_in_flight_threshold: int = 6
    extension_loader_error_in_flight_threshold: int = 8
    extension_loader_pressure_cooldown_seconds: float = 5.0
    extension_loader_emit_pressure_events: bool = False
    extension_loader_metrics_reporter_enabled: bool = False
    extension_loader_metrics_reporter_interval_seconds: float = 60.0
    extension_loader_metrics_reporter_reset_after_emit: bool = False
    extension_loader_metrics_reporter_topic: str = "vertical.extensions.loader.metrics"
