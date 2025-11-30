import json
import os
import uuid
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from victor.analytics.logger import UsageLogger


@pytest.fixture
def log_file(tmp_path: Path) -> Path:
    return tmp_path / "usage_log.jsonl"


def test_usage_logger_initialization(log_file: Path):
    """Tests that the UsageLogger initializes correctly."""
    logger = UsageLogger(log_file=log_file, enabled=True)
    assert logger.is_enabled()
    assert os.path.exists(log_file)
    assert logger.session_id is not None


def test_usage_logger_disabled(log_file: Path):
    """Tests that the logger does nothing when disabled."""
    logger = UsageLogger(log_file=log_file, enabled=False)
    assert not logger.is_enabled()
    logger.log_event("test_event", {"data": "value"})
    assert not os.path.exists(log_file)


def test_log_event_writes_valid_json(log_file: Path):
    """Tests that log_event writes a valid, structured JSON line to the file."""
    session_id = uuid.uuid4()
    with patch("uuid.uuid4", return_value=session_id):
        logger = UsageLogger(log_file=log_file, enabled=True)

    event_type = "test_event"
    event_data = {"key": "value", "number": 123}
    logger.log_event(event_type, event_data)

    with open(log_file, "r") as f:
        log_line = f.readline()
        log_entry = json.loads(log_line)

    assert log_entry["session_id"] == str(session_id)
    assert log_entry["event_type"] == event_type
    assert log_entry["data"] == event_data
    assert "timestamp" in log_entry


def test_log_multiple_events(log_file: Path):
    """Tests that multiple events are logged correctly."""
    logger = UsageLogger(log_file=log_file, enabled=True)

    logger.log_event("event1", {"data": 1})
    logger.log_event("event2", {"data": 2})

    with open(log_file, "r") as f:
        lines = f.readlines()

    assert len(lines) == 2
    entry1 = json.loads(lines[0])
    entry2 = json.loads(lines[1])

    assert entry1["data"]["data"] == 1
    assert entry2["data"]["data"] == 2
    assert entry1["session_id"] == entry2["session_id"]


def test_handles_non_serializable_data(log_file: Path):
    """Tests that the logger handles non-serializable data gracefully."""
    logger = UsageLogger(log_file=log_file, enabled=True)

    non_serializable_data = {"a_mock": MagicMock()}

    with patch.object(logger._logger, "error") as mock_error:
        logger.log_event("bad_event", non_serializable_data)
        mock_error.assert_called_once()
        assert "Failed to serialize log entry" in mock_error.call_args[0][0]

    # The file should be empty as the write failed
    assert os.stat(log_file).st_size == 0
