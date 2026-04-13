import gzip
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from victor.observability.analytics.sampling_filter import SemanticSamplingFilter


class UsageLogger:
    """Logs agent usage events to a JSONL file for analytics.

    Includes automatic log rotation: when the file exceeds MAX_FILE_SIZE,
    it is compressed and rotated (usage.jsonl → usage.1.jsonl.gz → ...).
    Consumers already expect this naming: prompt_optimizer.py and benchmark.py
    read ``usage.*.jsonl.gz`` via glob.
    """

    # Class-level dedup: prevents duplicate session_start events when
    # multiple UsageLogger instances share the same logical session.
    _started_sessions: set = set()

    # Rotation settings
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10 MB
    MAX_ROTATED_FILES: int = 5

    def __init__(
        self,
        log_file: Path,
        enabled: bool = True,
        sampling_filter: Optional["SemanticSamplingFilter"] = None,
    ):
        """
        Initializes the UsageLogger.

        Args:
            log_file: Path to the log file.
            enabled: Whether logging is enabled.
            sampling_filter: Optional filter to reduce noise events before disk I/O.
        """
        self._enabled = enabled
        self._log_file = log_file
        self.session_id = str(uuid.uuid4())
        self._logger = logging.getLogger(__name__)
        self._sampling_filter = sampling_filter

        if self._enabled:
            self._prepare_log_file()

    def _prepare_log_file(self) -> None:
        """Ensures the log directory and file exist."""
        try:
            self._log_file.parent.mkdir(parents=True, exist_ok=True)
            self._log_file.touch(exist_ok=True)
        except Exception as e:
            self._logger.error(f"Failed to create log directory or file {self._log_file}: {e}")
            self._enabled = False

    def is_enabled(self) -> bool:
        """Returns True if logging is enabled."""
        return self._enabled

    def _maybe_rotate(self) -> None:
        """Rotate log file if it exceeds MAX_FILE_SIZE.

        Rotation scheme: usage.jsonl → usage.1.jsonl.gz → usage.2.jsonl.gz → ...
        Oldest file (usage.{MAX_ROTATED_FILES}.jsonl.gz) is deleted.
        """
        try:
            if not self._log_file.exists():
                return
            if self._log_file.stat().st_size < self.MAX_FILE_SIZE:
                return
        except OSError:
            return

        try:
            parent = self._log_file.parent
            # Shift existing rotated files (5 → delete, 4 → 5, 3 → 4, ...)
            for i in range(self.MAX_ROTATED_FILES, 0, -1):
                src = parent / f"usage.{i}.jsonl.gz"
                if not src.exists():
                    continue
                if i == self.MAX_ROTATED_FILES:
                    src.unlink()
                else:
                    dst = parent / f"usage.{i + 1}.jsonl.gz"
                    src.rename(dst)

            # Compress current file → usage.1.jsonl.gz
            gz_path = parent / "usage.1.jsonl.gz"
            with open(self._log_file, "rb") as f_in:
                with gzip.open(gz_path, "wb") as f_out:
                    f_out.write(f_in.read())

            # Truncate current file
            self._log_file.unlink()
            self._log_file.touch()

            self._logger.info(
                "Rotated usage log: %s → %s", self._log_file.name, gz_path.name
            )
        except Exception as e:
            self._logger.debug("Log rotation failed (non-critical): %s", e)

    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Logs a usage event.

        Args:
            event_type: The type of event (e.g., 'tool_call', 'user_prompt').
            data: A dictionary of event-specific data.
        """
        if not self._enabled:
            return

        # Semantic sampling: drop noise events before disk I/O
        if self._sampling_filter and not self._sampling_filter.should_emit(event_type, data):
            return

        self._maybe_rotate()

        # Deduplicate session_start — emit only once per session_id
        if event_type == "session_start":
            if self.session_id in UsageLogger._started_sessions:
                return
            UsageLogger._started_sessions.add(self.session_id)

        log_entry = {
            "session_id": self.session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data,
        }

        try:
            log_line = json.dumps(log_entry)
            with open(self._log_file, "a") as f:
                f.write(log_line + "\n")
        except TypeError as e:
            self._logger.error(
                f"Failed to serialize log entry for event '{event_type}': {e}. Data: {data}"
            )
        except Exception as e:
            self._logger.error(f"Failed to write to log file {self._log_file}: {e}")
