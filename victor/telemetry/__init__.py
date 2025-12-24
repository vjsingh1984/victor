# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Telemetry module for Victor.

This module provides:
- OpenTelemetry integration for tracing and metrics
- Distributed tracing across agent operations
- Performance metrics collection
"""

from victor.telemetry.opentelemetry import (
    setup_opentelemetry,
    get_tracer,
    get_meter,
    is_telemetry_enabled,
)

__all__ = [
    "setup_opentelemetry",
    "get_tracer",
    "get_meter",
    "is_telemetry_enabled",
]
