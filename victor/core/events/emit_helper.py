# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper utilities for emitting events from synchronous code.

This module provides utilities to safely call async event emission
functions from synchronous code without RuntimeWarnings.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def emit_event_sync(
    event_bus: Any,
    topic: str,
    data: Dict[str, Any],
    *,
    source: str = "victor",
    correlation_id: Optional[str] = None,
) -> None:
    """Safely emit an event from synchronous code.

    This function handles the async event emission by creating a task
    on the running event loop and adding a done callback to handle errors.
    If no event loop is running or the backend is not connected, it silently
    skips the emission.

    Args:
        event_bus: The ObservabilityBus instance
        topic: Event topic (e.g., "tool.start", "metric.latency")
        data: Event payload
        source: Event source identifier
        correlation_id: Optional correlation ID

    Example:
        >>> from victor.core.events.emit_helper import emit_event_sync
        >>> emit_event_sync(bus, "tool.start", {"tool": "read_file"})
    """

    async def _emit_and_handle_errors():
        """Emit event and handle any errors."""
        try:
            await event_bus.emit(
                topic=topic,
                data=data,
                source=source,
                correlation_id=correlation_id,
            )
        except Exception as e:
            # Silently handle any errors (backend not connected, etc.)
            logger.debug(f"Event emission failed for {topic}: {e}")

    try:
        # Check if backend is connected before attempting to emit
        backend = event_bus.backend if hasattr(event_bus, "backend") else None
        if backend and hasattr(backend, "_is_connected") and not backend._is_connected:
            logger.debug(f"Backend not connected, skipping event emission: {topic}")
            return

        loop = asyncio.get_running_loop()
        task = loop.create_task(_emit_and_handle_errors())
        # Add done callback to handle any unhandled exceptions
        task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
    except RuntimeError:
        # No running loop, skip event emission
        logger.debug(f"No event loop running, skipping event emission: {topic}")
