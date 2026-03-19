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

"""Background indexer service for periodic incremental reindexing."""

import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class BackgroundIndexerService:
    """Background service for periodic incremental reindexing.

    Uses mtime-based change detection to efficiently update the index
    without blocking user operations. Runs as a daemon thread.

    Features:
    - Periodic polling (configurable interval, default 60s)
    - mtime-based change detection (no unnecessary work)
    - Graceful shutdown on session end
    - Thread-safe singleton pattern
    """

    _instance: Optional["BackgroundIndexerService"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        root: Path,
        interval_seconds: float = 60.0,
        auto_start: bool = False,
    ):
        self.root = root
        self.interval_seconds = interval_seconds
        self._indexer = None  # Lazy import to avoid circular
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        self._last_run: Optional[float] = None
        self._stats: Dict[str, Any] = {"runs": 0, "files_updated": 0, "errors": 0}

        if auto_start:
            self.start()

    @classmethod
    def get_instance(
        cls,
        root: Optional[Path] = None,
        interval_seconds: float = 60.0,
    ) -> "BackgroundIndexerService":
        """Get or create the singleton instance."""
        with cls._lock:
            if cls._instance is None:
                if root is None:
                    root = Path.cwd()
                cls._instance = cls(root, interval_seconds)
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton for testing."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.stop()
                cls._instance = None

    def start(self) -> None:
        """Start the background indexer thread."""
        if self._running:
            logger.debug("Background indexer already running")
            return

        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            name="BackgroundIndexer",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            f"Background indexer started (interval={self.interval_seconds}s, root={self.root})"
        )

    def stop(self) -> None:
        """Stop the background indexer gracefully."""
        if not self._running:
            return

        self._stop_event.set()
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

        logger.info("Background indexer stopped")

    def _run_loop(self) -> None:
        """Main loop for periodic reindexing."""
        while not self._stop_event.is_set():
            try:
                self._run_incremental_reindex()
            except Exception as e:
                logger.warning(f"Background indexer error: {e}")
                self._stats["errors"] += 1

            # Wait for next interval or stop event
            self._stop_event.wait(self.interval_seconds)

    def _run_incremental_reindex(self) -> None:
        """Perform incremental reindex using mtime detection."""
        if self._indexer is None:
            from victor.verticals.contrib.coding.codebase.indexer.core import CodebaseIndex

            self._indexer = CodebaseIndex(self.root)

        # Only reindex if there are changes (mtime-based detection)
        if not self._indexer._is_stale and self._indexer._is_indexed:
            logger.debug("Background indexer: no changes detected, skipping")
            return

        # Run incremental reindex in asyncio
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(self._indexer.incremental_reindex())
            self._stats["runs"] += 1
            self._stats["files_updated"] += len(result.get("updated", []))
            self._stats["files_updated"] += len(result.get("added", []))
            self._last_run = time.time()

            total_changes = (
                len(result.get("updated", []))
                + len(result.get("added", []))
                + len(result.get("removed", []))
            )
            if total_changes > 0:
                logger.info(
                    f"Background reindex: {len(result.get('updated', []))} updated, "
                    f"{len(result.get('added', []))} added, "
                    f"{len(result.get('removed', []))} removed"
                )
        finally:
            loop.close()

    def get_stats(self) -> Dict[str, Any]:
        """Get background indexer statistics."""
        return {
            **self._stats,
            "running": self._running,
            "interval_seconds": self.interval_seconds,
            "last_run": self._last_run,
        }


def start_background_indexer(
    root: Optional[Path] = None,
    interval_seconds: float = 60.0,
) -> BackgroundIndexerService:
    """Start the background indexer service (convenience function).

    Args:
        root: Project root directory (defaults to cwd)
        interval_seconds: Polling interval in seconds (default 60)

    Returns:
        BackgroundIndexerService instance
    """
    service = BackgroundIndexerService.get_instance(root, interval_seconds)
    service.start()
    return service
