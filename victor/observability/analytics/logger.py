import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


class UsageLogger:
    """Logs agent usage events to a JSONL file for analytics."""

    def __init__(self, log_file: Path, enabled: bool = True):
        """
        Initializes the UsageLogger.

        Args:
            log_file: Path to the log file.
            enabled: Whether logging is enabled.
        """
        self._enabled = enabled
        self._log_file = log_file
        self.session_id = str(uuid.uuid4())
        self._logger = logging.getLogger(__name__)

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

    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Logs a usage event.

        Args:
            event_type: The type of event (e.g., 'tool_call', 'user_prompt').
            data: A dictionary of event-specific data.
        """
        if not self._enabled:
            return

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
