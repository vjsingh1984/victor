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

"""Storage backend for workflow execution recordings.

This module provides storage backends for persisting workflow execution recordings,
with support for:
- File-based storage (JSON with optional compression)
- Optional database storage (PostgreSQL, MongoDB)
- Metadata indexing and search
- Retention policy management

Example:
    from victor.workflows.recording_storage import FileRecordingStorage

    storage = FileRecordingStorage(base_path="/recordings")
    await storage.save(recorder)
    recordings = await storage.list(workflow_name="my_workflow")
    recording = await storage.load(recording_id="...")
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    cast,
)
import builtins

if TYPE_CHECKING:
    from victor.workflows.execution_recorder import ExecutionReplayer

logger = logging.getLogger(__name__)


class StorageBackend(Enum):
    """Available storage backends."""

    FILE = "file"  # File-based storage (default)
    POSTGRESQL = "postgresql"  # PostgreSQL database
    MONGODB = "mongodb"  # MongoDB database


@dataclass
class RecordingQuery:
    """Query parameters for searching recordings.

    Attributes:
        workflow_name: Filter by workflow name
        recording_id: Specific recording ID
        start_date: Filter by start date (inclusive)
        end_date: Filter by end date (inclusive)
        success: Filter by success status
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds
        tags: Filter by tags (all must match)
        limit: Maximum number of results
        offset: Offset for pagination
        sort_by: Field to sort by
        sort_order: Sort order (asc or desc)
    """

    workflow_name: Optional[str] = None
    recording_id: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    success: Optional[bool] = None
    min_duration: Optional[float] = None
    max_duration: Optional[float] = None
    tags: Optional[list[str]] = None
    limit: Optional[int] = None
    offset: int = 0
    sort_by: str = "started_at"
    sort_order: str = "desc"

    def matches(self, metadata: dict[str, Any]) -> bool:
        """Check if metadata matches this query.

        Args:
            metadata: Recording metadata dictionary

        Returns:
            True if metadata matches all query criteria
        """
        if self.workflow_name and metadata.get("workflow_name") != self.workflow_name:
            return False

        if self.recording_id and metadata.get("recording_id") != self.recording_id:
            return False

        if self.start_date:
            started_at = metadata.get("started_at", 0)
            if datetime.fromtimestamp(started_at) < self.start_date:
                return False

        if self.end_date:
            started_at = metadata.get("started_at", 0)
            if datetime.fromtimestamp(started_at) > self.end_date:
                return False

        if self.success is not None and metadata.get("success") != self.success:
            return False

        if self.min_duration:
            duration = metadata.get("duration_seconds", 0)
            if duration < self.min_duration:
                return False

        if self.max_duration:
            duration = metadata.get("duration_seconds", 0)
            if duration > self.max_duration:
                return False

        if self.tags:
            metadata_tags = set(metadata.get("tags", []))
            query_tags = set(self.tags)
            if not query_tags.issubset(metadata_tags):
                return False

        return True


@dataclass
class RetentionPolicy:
    """Retention policy for managing recordings.

    Attributes:
        max_age_days: Maximum age in days (None = no limit)
        max_count: Maximum number of recordings (None = no limit)
        max_size_gb: Maximum total size in GB (None = no limit)
        keep_failed: Whether to keep failed recordings
        tags_to_keep: Tags for recordings to always keep
    """

    max_age_days: Optional[int] = None
    max_count: Optional[int] = None
    max_size_gb: Optional[float] = None
    keep_failed: bool = True
    tags_to_keep: list[str] = field(default_factory=list)

    def should_keep(self, metadata: dict[str, Any]) -> bool:
        """Check if a recording should be kept.

        Args:
            metadata: Recording metadata

        Returns:
            True if recording should be kept
        """
        # Always keep recordings with special tags
        if self.tags_to_keep:
            metadata_tags = set(metadata.get("tags", []))
            if any(tag in metadata_tags for tag in self.tags_to_keep):
                return True

        # Keep failed recordings if configured
        if not metadata.get("success", True) and self.keep_failed:
            return True

        # Check age
        if self.max_age_days:
            started_at = metadata.get("started_at", 0)
            age_days = (datetime.now().timestamp() - started_at) / 86400
            if age_days > self.max_age_days:
                return False

        return True


class RecordingStorage(ABC):
    """Abstract base class for recording storage backends."""

    @abstractmethod
    async def save(
        self,
        recorder: Any,
        filepath: Optional[Path] = None,
    ) -> str:
        """Save a recording.

        Args:
            recorder: ExecutionRecorder instance
            filepath: Optional specific filepath

        Returns:
            Recording ID
        """
        pass

    @abstractmethod
    async def load(self, recording_id: str) -> Optional[ExecutionReplayer]:
        """Load a recording.

        Args:
            recording_id: Recording identifier

        Returns:
            ExecutionReplayer instance or None if not found
        """
        pass

    @abstractmethod
    async def delete(self, recording_id: str) -> bool:
        """Delete a recording.

        Args:
            recording_id: Recording identifier

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def list(self, query: Optional[RecordingQuery] = None) -> builtins.list[dict[str, Any]]:
        """List recordings matching query.

        Args:
            query: Optional query filters

        Returns:
            List of recording metadata
        """
        pass

    @abstractmethod
    async def get_metadata(self, recording_id: str) -> Optional[dict[str, Any]]:
        """Get recording metadata.

        Args:
            recording_id: Recording identifier

        Returns:
            Metadata dictionary or None if not found
        """
        pass

    async def apply_retention_policy(
        self,
        policy: RetentionPolicy,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Apply retention policy to recordings.

        Args:
            policy: RetentionPolicy to apply
            dry_run: If True, don't actually delete

        Returns:
            Dictionary with deletion results
        """
        recordings = await self.list()

        to_delete = []
        total_size = 0

        for metadata in recordings:
            if not policy.should_keep(metadata):
                to_delete.append(metadata["recording_id"])
                total_size += metadata.get("file_size_bytes", 0)

        if not dry_run:
            for recording_id in to_delete:
                await self.delete(recording_id)

        return {
            "total_recordings": len(recordings),
            "to_delete": len(to_delete),
            "total_size_bytes": total_size,
            "deleted_ids": to_delete if not dry_run else [],
        }


class FileRecordingStorage(RecordingStorage):
    """File-based storage for recordings.

    Stores recordings as JSON files (optionally compressed) with a
    metadata index for efficient searching.

    Directory structure:
        base_path/
        ├── index.json
        └── recordings/
            ├── {recording_id}.json
            └── {recording_id}.json.gz

    Attributes:
        base_path: Base directory for storage
        compress: Whether to compress recordings
        auto_create_dir: Whether to create directories automatically
    """

    def __init__(
        self,
        base_path: str | Path = "./recordings",
        compress: bool = True,
        auto_create_dir: bool = True,
    ):
        """Initialize file-based storage.

        Args:
            base_path: Base directory for storage (will contain recordings/ subdirectory)
            compress: Whether to compress recordings
            auto_create_dir: Whether to create directories automatically
        """
        self.base_path = Path(base_path)
        self.compress = compress
        self.auto_create_dir = auto_create_dir

        # recordings_dir is where actual recording files are stored
        # If base_path already ends with "recordings", use it directly
        if self.base_path.name == "recordings":
            self.recordings_dir = self.base_path
            self.index_file = self.base_path.parent / "index.json"
        else:
            self.recordings_dir = self.base_path / "recordings"
            self.index_file = self.base_path / "index.json"

        if self.auto_create_dir:
            self.base_path.mkdir(parents=True, exist_ok=True)
            self.recordings_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Initialized FileRecordingStorage: {self.base_path}")

    def _get_recording_path(self, recording_id: str) -> Path:
        """Get filepath for a recording.

        Args:
            recording_id: Recording identifier

        Returns:
            Path to recording file
        """
        extension = ".json.gz" if self.compress else ".json"
        return self.recordings_dir / f"{recording_id}{extension}"

    async def _load_index(self) -> dict[str, Any]:
        """Load the metadata index.

        Returns:
            Index dictionary
        """
        if not self.index_file.exists():
            return {"recordings": {}, "last_updated": None}

        with open(self.index_file, "r") as f:
            data = json.load(f)
            return cast(dict[str, Any], data)

    async def _save_index(self, index: dict[str, Any]) -> None:
        """Save the metadata index.

        Args:
            index: Index dictionary
        """
        index["last_updated"] = datetime.now().isoformat()
        with open(self.index_file, "w") as f:
            json.dump(index, f, indent=2)

    async def save(
        self,
        recorder: Any,
        filepath: Optional[Path] = None,
    ) -> str:
        """Save a recording.

        Args:
            recorder: ExecutionRecorder instance
            filepath: Optional specific filepath

        Returns:
            Recording ID
        """

        # Determine filepath
        if filepath:
            recording_path = filepath
        else:
            metadata = recorder.finalize()
            # Get path with extension from _get_recording_path
            recording_path = self._get_recording_path(metadata.recording_id)

        # Save recording
        # Note: recorder.save() will add .gz if compress=True in recorder config,
        # so we need to pass the path without extension to avoid double extension
        if self.compress and str(recording_path).endswith(".json.gz"):
            # Remove .gz extension since recorder.save() will add it
            recording_path_no_gz = recording_path.with_suffix("")
            await recorder.save(recording_path_no_gz)
        else:
            await recorder.save(recording_path)

        # Update index
        index = await self._load_index()
        index["recordings"][metadata.recording_id] = {
            "recording_id": metadata.recording_id,
            "workflow_name": metadata.workflow_name,
            "started_at": metadata.started_at,
            "completed_at": metadata.completed_at,
            "duration_seconds": metadata.duration_seconds,
            "success": metadata.success,
            "error": metadata.error,
            "node_count": metadata.node_count,
            "team_count": metadata.team_count,
            "recursion_max_depth": metadata.recursion_max_depth,
            "event_count": metadata.event_count,
            "file_size_bytes": metadata.file_size_bytes,
            "checksum": metadata.checksum,
            "tags": metadata.tags,
            "filepath": str(recording_path),
        }
        await self._save_index(index)

        logger.info(f"Saved recording: {metadata.recording_id}")

        return str(metadata.recording_id)

    async def load(self, recording_id: str) -> Optional[ExecutionReplayer]:
        """Load a recording.

        Args:
            recording_id: Recording identifier

        Returns:
            ExecutionReplayer instance or None if not found

        Raises:
            FileNotFoundError: If recording not found
        """
        from victor.workflows.execution_recorder import ExecutionReplayer

        # Get filepath from index
        index = await self._load_index()
        if recording_id not in index["recordings"]:
            raise FileNotFoundError(f"Recording not found: {recording_id}")

        filepath = Path(index["recordings"][recording_id]["filepath"])

        if not filepath.exists():
            # Fallback to default path
            filepath = self._get_recording_path(recording_id)

        return ExecutionReplayer.load(filepath)

    async def delete(self, recording_id: str) -> bool:
        """Delete a recording.

        Args:
            recording_id: Recording identifier

        Returns:
            True if deleted, False if not found
        """
        index = await self._load_index()

        if recording_id not in index["recordings"]:
            return False

        # Delete file
        filepath = Path(index["recordings"][recording_id]["filepath"])
        if filepath.exists():
            filepath.unlink()

        # Update index
        del index["recordings"][recording_id]
        await self._save_index(index)

        logger.info(f"Deleted recording: {recording_id}")

        return True

    async def list(self, query: Optional[RecordingQuery] = None) -> builtins.list[dict[str, Any]]:
        """List recordings matching query.

        Args:
            query: Optional query filters

        Returns:
            List of recording metadata
        """
        index = await self._load_index()
        recordings = list(index["recordings"].values())

        # Filter by query
        if query:
            recordings = [r for r in recordings if query.matches(r)]

            # Sort
            reverse = query.sort_order == "desc"
            recordings.sort(key=lambda r: r.get(query.sort_by, 0), reverse=reverse)

            # Pagination
            if query.offset:
                recordings = recordings[query.offset :]
            if query.limit:
                recordings = recordings[: query.limit]

        return recordings

    async def get_metadata(self, recording_id: str) -> Optional[dict[str, Any]]:
        """Get recording metadata.

        Args:
            recording_id: Recording identifier

        Returns:
            Metadata dictionary or None if not found
        """
        index = await self._load_index()
        result = index["recordings"].get(recording_id)
        return cast(Optional[dict[str, Any]], result)

    async def cleanup_empty_files(self) -> int:
        """Remove empty or corrupted recording files.

        Returns:
            Number of files removed
        """
        removed = 0
        index = await self._load_index()

        for recording_id, metadata in list(index["recordings"].items()):
            filepath = Path(metadata["filepath"])

            if not filepath.exists():
                del index["recordings"][recording_id]
                removed += 1
                continue

            if filepath.stat().st_size == 0:
                filepath.unlink()
                del index["recordings"][recording_id]
                removed += 1

        if removed > 0:
            await self._save_index(index)
            logger.info(f"Cleaned up {removed} empty/corrupted recordings")

        return removed

    async def get_storage_stats(self) -> dict[str, Any]:
        """Get storage statistics.

        Returns:
            Statistics dictionary
        """
        index = await self._load_index()
        recordings = list(index["recordings"].values())

        total_size = sum(r.get("file_size_bytes", 0) for r in recordings)
        total_duration = sum(r.get("duration_seconds", 0) for r in recordings)

        success_count = sum(1 for r in recordings if r.get("success", False))
        failed_count = len(recordings) - success_count

        # Workflow breakdown
        workflow_counts: dict[str, int] = {}
        for r in recordings:
            workflow = r.get("workflow_name", "unknown")
            workflow_counts[workflow] = workflow_counts.get(workflow, 0) + 1

        return {
            "total_recordings": len(recordings),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "total_duration_seconds": total_duration,
            "success_count": success_count,
            "failed_count": failed_count,
            "workflow_counts": workflow_counts,
            "oldest_recording": min(
                (r.get("started_at", float("inf")) for r in recordings), default=None
            ),
            "newest_recording": max((r.get("started_at", 0) for r in recordings), default=None),
        }


class InMemoryRecordingStorage(RecordingStorage):
    """In-memory storage for recordings (useful for testing)."""

    def __init__(self) -> None:
        """Initialize in-memory storage."""
        self._recordings: dict[str, tuple[bytes, dict[str, Any]]] = {}
        logger.debug("Initialized InMemoryRecordingStorage")

    async def save(
        self,
        recorder: Any,
        filepath: Optional[Path] = None,
    ) -> str:
        """Save a recording.

        Args:
            recorder: ExecutionRecorder instance
            filepath: Ignored (for compatibility)

        Returns:
            Recording ID
        """
        import io
        import gzip

        metadata = recorder.finalize()

        # Serialize recording

        recording_data = {
            "metadata": metadata.to_dict(),
            "events": [event.to_dict() for event in recorder.events],
            "snapshots": [snapshot.to_dict() for snapshot in recorder.snapshots],
        }

        json_data = json.dumps(recording_data)

        # Compress if needed
        if recorder.config.get("compress", True):
            buf = io.BytesIO()
            with gzip.GzipFile(fileobj=buf, mode="wb") as f:
                f.write(json_data.encode())
            data = buf.getvalue()
        else:
            data = json_data.encode()

        self._recordings[metadata.recording_id] = (data, metadata.to_dict())

        logger.info(f"Saved recording to memory: {metadata.recording_id}")

        return str(metadata.recording_id)

    async def load(self, recording_id: str) -> Optional[ExecutionReplayer]:
        """Load a recording.

        Args:
            recording_id: Recording identifier

        Returns:
            ExecutionReplayer instance or None if not found

        Raises:
            FileNotFoundError: If recording not found
        """
        from victor.workflows.execution_recorder import (
            ExecutionReplayer,
            RecordingEvent,
            RecordingMetadata,
            StateSnapshot,
            RecordingEventType,
        )
        import io
        import gzip

        if recording_id not in self._recordings:
            raise FileNotFoundError(f"Recording not found: {recording_id}")

        data, metadata_dict = self._recordings[recording_id]

        # Decompress if needed
        try:
            buf = io.BytesIO(data)
            with gzip.GzipFile(fileobj=buf, mode="rb") as f:
                json_data = f.read().decode()
        except Exception:
            json_data = data.decode()

        recording_data = json.loads(json_data)

        # Parse
        metadata = RecordingMetadata.from_dict(recording_data["metadata"])

        events = []
        for event_data in recording_data.get("events", []):
            event_data["event_type"] = RecordingEventType(event_data["event_type"])
            events.append(RecordingEvent(**event_data))

        snapshots = []
        for snapshot_data in recording_data.get("snapshots", []):
            snapshots.append(StateSnapshot(**snapshot_data))

        return ExecutionReplayer(
            recording_path=Path(f"<memory:{recording_id}>"),
            metadata=metadata,
            events=events,
            snapshots=snapshots,
        )

    async def delete(self, recording_id: str) -> bool:
        """Delete a recording.

        Args:
            recording_id: Recording identifier

        Returns:
            True if deleted, False if not found
        """
        if recording_id in self._recordings:
            del self._recordings[recording_id]
            logger.info(f"Deleted recording from memory: {recording_id}")
            return True
        return False

    async def list(self, query: Optional[RecordingQuery] = None) -> builtins.list[dict[str, Any]]:
        """List recordings matching query.

        Args:
            query: Optional query filters

        Returns:
            List of recording metadata
        """
        recordings = [metadata for _, metadata in self._recordings.values()]

        if query:
            recordings = [r for r in recordings if query.matches(r)]

            # Sort
            reverse = query.sort_order == "desc"
            recordings.sort(key=lambda r: r.get(query.sort_by, 0), reverse=reverse)

            # Pagination
            if query.offset:
                recordings = recordings[query.offset :]
            if query.limit:
                recordings = recordings[: query.limit]

        return recordings

    async def get_metadata(self, recording_id: str) -> Optional[dict[str, Any]]:
        """Get recording metadata.

        Args:
            recording_id: Recording identifier

        Returns:
            Metadata dictionary or None if not found
        """
        if recording_id in self._recordings:
            return self._recordings[recording_id][1]
        return None

    def clear(self) -> None:
        """Clear all recordings."""
        self._recordings.clear()


__all__ = [
    "StorageBackend",
    "RecordingQuery",
    "RetentionPolicy",
    "RecordingStorage",
    "FileRecordingStorage",
    "InMemoryRecordingStorage",
]
