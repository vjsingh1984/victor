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

"""Event exporters implementing the Strategy pattern.

Exporters handle persisting or transmitting events to external systems.
Each exporter implements a specific strategy for event handling.

Available exporters:
- JsonLineExporter: Writes events to JSONL file
- CallbackExporter: Calls user-defined callback
- CompositeExporter: Combines multiple exporters
- FilteringExporter: Filters events before delegation

Example:
    from victor.observability import EventBus, JsonLineExporter

    bus = EventBus.get_instance()
    bus.add_exporter(JsonLineExporter("events.jsonl"))
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

from victor.observability.event_bus import EventCategory, VictorEvent

logger = logging.getLogger(__name__)


class BaseExporter(ABC):
    """Abstract base class for event exporters.

    Implements the Strategy pattern - each exporter defines its own
    strategy for handling events.

    Subclasses must implement:
        - export(): Synchronous event handling
        - close(): Cleanup resources

    Optional:
        - export_async(): Asynchronous event handling
    """

    @abstractmethod
    def export(self, event: VictorEvent) -> None:
        """Export an event synchronously.

        Args:
            event: Event to export.
        """
        pass

    async def export_async(self, event: VictorEvent) -> None:
        """Export an event asynchronously.

        Default implementation calls sync export.

        Args:
            event: Event to export.
        """
        self.export(event)

    @abstractmethod
    def close(self) -> None:
        """Close the exporter and release resources."""
        pass

    def __enter__(self) -> "BaseExporter":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class JsonLineExporter(BaseExporter):
    """Exports events to a JSONL (JSON Lines) file.

    Each event is written as a single line of JSON, making it
    easy to process with standard tools (jq, grep, etc.).

    Attributes:
        path: Path to output file.
        buffer_size: Number of events to buffer before flushing.
        include_categories: Optional set of categories to include.
        exclude_categories: Optional set of categories to exclude.

    Example:
        exporter = JsonLineExporter("events.jsonl")
        exporter.export(event)
        exporter.close()
    """

    def __init__(
        self,
        path: Union[str, Path],
        *,
        buffer_size: int = 10,
        include_categories: Optional[Set[EventCategory]] = None,
        exclude_categories: Optional[Set[EventCategory]] = None,
        append: bool = True,
    ) -> None:
        """Initialize the JSONL exporter.

        Args:
            path: Path to output file.
            buffer_size: Events to buffer before flush.
            include_categories: Categories to include (None = all).
            exclude_categories: Categories to exclude.
            append: Whether to append to existing file.
        """
        self.path = Path(path)
        self.buffer_size = buffer_size
        self.include_categories = include_categories
        self.exclude_categories = exclude_categories or set()

        # Ensure parent directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Open file
        mode = "a" if append else "w"
        self._file = open(self.path, mode, encoding="utf-8")
        self._buffer: List[str] = []
        self._event_count = 0

    def export(self, event: VictorEvent) -> None:
        """Export event to JSONL file.

        Args:
            event: Event to export.
        """
        # Filter by category
        if self.include_categories and event.category not in self.include_categories:
            return
        if event.category in self.exclude_categories:
            return

        # Serialize and buffer
        line = json.dumps(event.to_dict()) + "\n"
        self._buffer.append(line)
        self._event_count += 1

        # Flush if buffer is full
        if len(self._buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        """Flush buffered events to file."""
        if self._buffer:
            self._file.writelines(self._buffer)
            self._file.flush()
            self._buffer.clear()

    def close(self) -> None:
        """Close the file, flushing any remaining events."""
        self.flush()
        self._file.close()

    @property
    def event_count(self) -> int:
        """Get total number of events exported."""
        return self._event_count


class CallbackExporter(BaseExporter):
    """Exports events by calling a user-defined callback.

    Useful for integrating with existing logging systems,
    metrics collectors, or custom handlers.

    Example:
        def my_handler(event):
            print(f"Event: {event.name}")

        exporter = CallbackExporter(my_handler)
    """

    def __init__(
        self,
        callback: Callable[[VictorEvent], None],
        *,
        async_callback: Optional[Callable[[VictorEvent], Any]] = None,
        error_handler: Optional[Callable[[Exception, VictorEvent], None]] = None,
    ) -> None:
        """Initialize callback exporter.

        Args:
            callback: Sync callback function.
            async_callback: Optional async callback.
            error_handler: Optional error handler.
        """
        self._callback = callback
        self._async_callback = async_callback
        self._error_handler = error_handler

    def export(self, event: VictorEvent) -> None:
        """Export event via callback.

        Args:
            event: Event to export.
        """
        try:
            self._callback(event)
        except Exception as e:
            if self._error_handler:
                self._error_handler(e, event)
            else:
                logger.warning(f"Callback error: {e}")

    async def export_async(self, event: VictorEvent) -> None:
        """Export event via async callback.

        Args:
            event: Event to export.
        """
        if self._async_callback:
            try:
                await self._async_callback(event)
            except Exception as e:
                if self._error_handler:
                    self._error_handler(e, event)
                else:
                    logger.warning(f"Async callback error: {e}")
        else:
            self.export(event)

    def close(self) -> None:
        """No-op for callback exporter."""
        pass


class CompositeExporter(BaseExporter):
    """Combines multiple exporters into one.

    Implements the Composite pattern - events are sent to all
    child exporters.

    Example:
        composite = CompositeExporter([
            JsonLineExporter("events.jsonl"),
            CallbackExporter(my_handler),
        ])
    """

    def __init__(self, exporters: List[BaseExporter]) -> None:
        """Initialize composite exporter.

        Args:
            exporters: List of exporters to delegate to.
        """
        self._exporters = list(exporters)

    def add(self, exporter: BaseExporter) -> None:
        """Add an exporter.

        Args:
            exporter: Exporter to add.
        """
        self._exporters.append(exporter)

    def remove(self, exporter: BaseExporter) -> None:
        """Remove an exporter.

        Args:
            exporter: Exporter to remove.
        """
        if exporter in self._exporters:
            self._exporters.remove(exporter)

    def export(self, event: VictorEvent) -> None:
        """Export event to all child exporters.

        Args:
            event: Event to export.
        """
        for exporter in self._exporters:
            try:
                exporter.export(event)
            except Exception as e:
                logger.warning(f"Exporter {type(exporter).__name__} error: {e}")

    async def export_async(self, event: VictorEvent) -> None:
        """Export event to all child exporters asynchronously.

        Args:
            event: Event to export.
        """
        tasks = []
        for exporter in self._exporters:
            if hasattr(exporter, "export_async"):
                tasks.append(exporter.export_async(event))
            else:
                exporter.export(event)

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Async exporter error: {result}")

    def close(self) -> None:
        """Close all child exporters."""
        for exporter in self._exporters:
            try:
                exporter.close()
            except Exception as e:
                logger.warning(f"Exporter close error: {e}")


class FilteringExporter(BaseExporter):
    """Wraps another exporter with filtering logic.

    Implements the Decorator pattern - adds filtering capability
    to any existing exporter.

    Example:
        # Only export TOOL events
        filtered = FilteringExporter(
            JsonLineExporter("tools.jsonl"),
            categories={EventCategory.TOOL},
        )
    """

    def __init__(
        self,
        exporter: BaseExporter,
        *,
        categories: Optional[Set[EventCategory]] = None,
        names: Optional[Set[str]] = None,
        predicate: Optional[Callable[[VictorEvent], bool]] = None,
    ) -> None:
        """Initialize filtering exporter.

        Args:
            exporter: Underlying exporter.
            categories: Set of categories to include.
            names: Set of event names to include.
            predicate: Custom filter function.
        """
        self._exporter = exporter
        self._categories = categories
        self._names = names
        self._predicate = predicate

    def _should_export(self, event: VictorEvent) -> bool:
        """Check if event should be exported.

        Args:
            event: Event to check.

        Returns:
            True if event should be exported.
        """
        if self._categories and event.category not in self._categories:
            return False
        if self._names and event.name not in self._names:
            return False
        if self._predicate and not self._predicate(event):
            return False
        return True

    def export(self, event: VictorEvent) -> None:
        """Export event if it passes filters.

        Args:
            event: Event to export.
        """
        if self._should_export(event):
            self._exporter.export(event)

    async def export_async(self, event: VictorEvent) -> None:
        """Export event asynchronously if it passes filters.

        Args:
            event: Event to export.
        """
        if self._should_export(event):
            await self._exporter.export_async(event)

    def close(self) -> None:
        """Close underlying exporter."""
        self._exporter.close()


class BufferedExporter(BaseExporter):
    """Buffers events and exports in batches.

    Useful for reducing I/O overhead when exporting many events.

    Example:
        buffered = BufferedExporter(
            JsonLineExporter("events.jsonl"),
            batch_size=100,
            flush_interval=5.0,
        )
    """

    def __init__(
        self,
        exporter: BaseExporter,
        *,
        batch_size: int = 100,
        flush_interval: float = 5.0,
    ) -> None:
        """Initialize buffered exporter.

        Args:
            exporter: Underlying exporter.
            batch_size: Events to buffer before flush.
            flush_interval: Seconds between auto-flushes.
        """
        self._exporter = exporter
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._buffer: List[VictorEvent] = []
        self._last_flush = datetime.now(timezone.utc)

    def export(self, event: VictorEvent) -> None:
        """Buffer event for later export.

        Args:
            event: Event to buffer.
        """
        self._buffer.append(event)

        # Check if we should flush
        should_flush = (
            len(self._buffer) >= self._batch_size
            or (datetime.now(timezone.utc) - self._last_flush).total_seconds() >= self._flush_interval
        )

        if should_flush:
            self.flush()

    def flush(self) -> None:
        """Flush all buffered events."""
        for event in self._buffer:
            self._exporter.export(event)
        self._buffer.clear()
        self._last_flush = datetime.now(timezone.utc)

    def close(self) -> None:
        """Flush remaining events and close."""
        self.flush()
        self._exporter.close()
