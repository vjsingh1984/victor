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

"""Event Bus Monitor - Tool to monitor events flowing through the Victor system.

This tool provides:
- Subscribe to all events and print them
- Filter by event type or source
- Show event timing and frequency
- Export event streams to files

Usage:
    python -m victor.devtools.event_monitor
    python -m victor.devtools.event_monitor --filter tool
    python -m victor.devtools.event_monitor --duration 30
    python -m victor.devtools.event_monitor --export events.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, cast
from collections.abc import Callable


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class EventRecord:
    """Record of an event that flowed through the system."""

    timestamp: float
    datetime_str: str
    event_type: str
    category: str
    source: Optional[str]
    data: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "datetime": self.datetime_str,
            "event_type": self.event_type,
            "category": self.category,
            "source": self.source,
            "data": self._serialize_data(self.data),
            "metadata": self.metadata,
        }

    @staticmethod
    def _serialize_data(data: Any) -> Any:
        """Serialize data for JSON."""
        if isinstance(data, (str, int, float, bool, type(None))):
            return data
        elif isinstance(data, dict):
            return {k: EventRecord._serialize_data(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [EventRecord._serialize_data(item) for item in data]
        else:
            return str(data)


@dataclass
class EventStatistics:
    """Statistics about events."""

    total_events: int = 0
    events_by_type: dict[str, int] = field(default_factory=dict)
    events_by_category: dict[str, int] = field(default_factory=dict)
    events_by_source: dict[str, int] = field(default_factory=dict)
    first_event_time: Optional[float] = None
    last_event_time: Optional[float] = None

    def record_event(self, record: EventRecord) -> None:
        """Record an event."""
        self.total_events += 1
        self.events_by_type[record.event_type] = self.events_by_type.get(record.event_type, 0) + 1
        self.events_by_category[record.category] = (
            self.events_by_category.get(record.category, 0) + 1
        )

        if record.source:
            self.events_by_source[record.source] = self.events_by_source.get(record.source, 0) + 1

        if self.first_event_time is None:
            self.first_event_time = record.timestamp
        self.last_event_time = record.timestamp

    def get_summary(self) -> str:
        """Get summary string."""
        duration = 0.0
        if self.first_event_time and self.last_event_time:
            duration = self.last_event_time - self.first_event_time

        lines = [
            f"\n{'=' * 60}",
            "Event Statistics",
            f"{'=' * 60}",
            f"Total Events: {self.total_events}",
            f"Duration: {duration:.2f} seconds",
            f"Event Rate: {self.total_events / duration:.2f} events/sec" if duration > 0 else "",
            "\nTop Event Types:",
        ]

        # Top 10 event types
        sorted_types = sorted(self.events_by_type.items(), key=lambda x: x[1], reverse=True)[:10]
        for event_type, count in sorted_types:
            lines.append(f"  {event_type}: {count}")

        lines.append("\nBy Category:")
        for category, count in sorted(
            self.events_by_category.items(), key=lambda x: x[1], reverse=True
        ):
            lines.append(f"  {category}: {count}")

        if self.events_by_source:
            lines.append("\nBy Source:")
            for source, count in sorted(
                self.events_by_source.items(), key=lambda x: x[1], reverse=True
            )[:10]:
                lines.append(f"  {source}: {count}")

        return "\n".join(lines)


class EventMonitor:
    """Monitor for Victor's event system."""

    def __init__(
        self,
        filter_category: Optional[str] = None,
        filter_event_type: Optional[str] = None,
        verbose: bool = False,
    ):
        """Initialize the monitor.

        Args:
            filter_category: Only show events from this category
            filter_event_type: Only show events of this type
            verbose: Show detailed event data
        """
        self.filter_category = filter_category
        self.filter_event_type = filter_event_type
        self.verbose = verbose
        self.events: list[EventRecord] = []
        self.stats = EventStatistics()
        self._running = False
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start monitoring events."""
        self._running = True

        # Try to hook into Victor's event system
        try:
            self._hook_into_observability_bus()
            logger.info("Successfully hooked into ObservabilityBus")
        except Exception as e:
            logger.warning(f"Could not hook into ObservabilityBus: {e}")

        try:
            self._hook_into_event_registry()
            logger.info("Successfully hooked into EventRegistry")
        except Exception as e:
            logger.warning(f"Could not hook into EventRegistry: {e}")

    def _hook_into_observability_bus(self) -> None:
        """Hook into the ObservabilityBus if available."""
        try:
            from victor.core.events import ObservabilityBus, get_observability_bus

            bus = get_observability_bus()

            # Subscribe to all events
            import asyncio
            from typing import Any
            from collections.abc import Awaitable
            from victor.core.events.protocols import MessagingEvent

            async def _subscribe_wrapper() -> None:
                await bus.subscribe(
                    "*",
                    cast(Callable[[MessagingEvent], Awaitable[None]], self._on_observability_event),
                )

            asyncio.create_task(_subscribe_wrapper())
            logger.info("Subscribed to ObservabilityBus")
        except ImportError:
            raise

    def _hook_into_event_registry(self) -> None:
        """Hook into EventRegistry if available."""
        try:
            from victor.observability.event_registry import EventCategoryRegistry

            # Just check that we can access the registry
            registry = EventCategoryRegistry.get_instance()
            logger.debug(f"Event registry has {registry.count()} custom categories")
        except ImportError:
            raise

    def _on_observability_event(self, event: Any) -> None:
        """Handle event from ObservabilityBus."""
        if not self._running:
            return

        try:
            # Extract event information
            timestamp = time.time()
            datetime_str = datetime.fromtimestamp(timestamp, timezone.utc).isoformat()

            # Get event type
            event_type = type(event).__name__

            # Get category
            category = getattr(event, "category", "unknown")

            # Get source
            source = getattr(event, "source", None)

            # Get data
            data = {}
            if hasattr(event, "__dict__"):
                for key, value in event.__dict__.items():
                    if not key.startswith("_"):
                        data[key] = value

            # Create record
            record = EventRecord(
                timestamp=timestamp,
                datetime_str=datetime_str,
                event_type=event_type,
                category=category,
                source=source,
                data=data,
            )

            # Apply filters
            if self.filter_category and category != self.filter_category:
                return

            if self.filter_event_type and event_type != self.filter_event_type:
                return

            # Store and print
            with self._lock:
                self.events.append(record)
                self.stats.record_event(record)

            # Print event
            self._print_event(record)

        except Exception as e:
            logger.error(f"Error handling event: {e}")

    def _print_event(self, record: EventRecord) -> None:
        """Print event to console."""
        category_color = self._get_category_color(record.category)

        if self.verbose:
            print(
                f"\n[{record.datetime_str}] "
                f"{category_color}{record.category}\033[0m - "
                f"{record.event_type}"
            )

            if record.source:
                print(f"  Source: {record.source}")

            if record.data:
                print("  Data:")
                for key, value in record.data.items():
                    value_str = str(value)
                    if len(value_str) > 100:
                        value_str = value_str[:100] + "..."
                    print(f"    {key}: {value_str}")
        else:
            print(
                f"[{record.datetime_str}] "
                f"{category_color}{record.category}\033[0m - "
                f"{record.event_type}"
            )

    def _get_category_color(self, category: str) -> str:
        """Get ANSI color code for category."""
        colors = {
            "tool": "\033[92m",  # Green
            "state": "\033[94m",  # Blue
            "model": "\033[93m",  # Yellow
            "error": "\033[91m",  # Red
            "audit": "\033[95m",  # Magenta
            "metric": "\033[96m",  # Cyan
            "lifecycle": "\033[97m",  # White
        }
        return colors.get(category, "\033[0m")  # Default: no color

    def stop(self) -> None:
        """Stop monitoring events."""
        self._running = False

    def get_statistics(self) -> EventStatistics:
        """Get event statistics."""
        return self.stats

    def export_events(self, output_path: Path) -> None:
        """Export events to JSON.

        Args:
            output_path: Path to output JSON file
        """
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "events": [record.to_dict() for record in self.events],
                    "statistics": {
                        "total": self.stats.total_events,
                        "by_type": self.stats.events_by_type,
                        "by_category": self.stats.events_by_category,
                        "by_source": self.stats.events_by_source,
                    },
                },
                f,
                indent=2,
            )

        print(f"\nExported {len(self.events)} events to {output_path}")

    def print_summary(self) -> None:
        """Print event summary."""
        print(self.stats.get_summary())


async def monitor_async(
    monitor: EventMonitor,
    duration: Optional[int] = None,
    event_count: Optional[int] = None,
) -> None:
    """Run monitor asynchronously.

    Args:
        monitor: Event monitor instance
        duration: Duration in seconds (None for infinite)
        event_count: Stop after N events (None for infinite)
    """
    monitor.start()

    start_time = time.time()

    try:
        while monitor._running:
            await asyncio.sleep(0.1)

            # Check duration
            if duration and (time.time() - start_time) >= duration:
                logger.info(f"Duration limit reached: {duration}s")
                break

            # Check event count
            if event_count and len(monitor.events) >= event_count:
                logger.info(f"Event count limit reached: {event_count}")
                break

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        monitor.stop()
        monitor.print_summary()


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Monitor Victor's event system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Monitor all events
  python -m victor.devtools.event_monitor

  # Filter by category
  python -m victor.devtools.event_monitor --filter tool

  # Filter by event type
  python -m victor.devtools.event_monitor --event-type ToolExecutionEvent

  # Monitor for specific duration
  python -m victor.devtools.event_monitor --duration 30

  # Monitor until N events received
  python -m victor.devtools.event_monitor --event-count 100

  # Export events to JSON
  python -m victor.devtools.event_monitor --export events.json

  # Verbose mode with full event data
  python -m victor.devtools.event_monitor --verbose
        """,
    )

    parser.add_argument(
        "-f",
        "--filter",
        type=str,
        metavar="CATEGORY",
        help="Filter events by category (tool, state, model, error, etc.)",
    )

    parser.add_argument(
        "-e",
        "--event-type",
        type=str,
        metavar="TYPE",
        help="Filter events by type",
    )

    parser.add_argument(
        "-d",
        "--duration",
        type=int,
        metavar="SECONDS",
        help="Monitor for specific duration (default: infinite)",
    )

    parser.add_argument(
        "-c",
        "--event-count",
        type=int,
        metavar="N",
        help="Stop after N events (default: infinite)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed event data",
    )

    parser.add_argument(
        "--export",
        type=str,
        metavar="PATH",
        help="Export events to JSON file",
    )

    args = parser.parse_args()

    # Create monitor
    monitor = EventMonitor(
        filter_category=args.filter,
        filter_event_type=args.event_type,
        verbose=args.verbose,
    )

    print("Starting Event Monitor...")
    print(f"Filter: category={args.filter}, event_type={args.event_type}")
    print(f"Duration: {args.duration if args.duration else 'infinite'}")
    print(f"Event count: {args.event_count if args.event_count else 'infinite'}")
    print(f"Verbose: {args.verbose}")
    print("\nListening for events...\n")

    # Run monitor
    try:
        asyncio.run(
            monitor_async(
                monitor,
                duration=args.duration,
                event_count=args.event_count,
            )
        )
    except Exception as e:
        logger.error(f"Error running monitor: {e}")
        return 1

    # Export if requested
    if args.export:
        monitor.export_events(Path(args.export))

    return 0


if __name__ == "__main__":
    sys.exit(main())
