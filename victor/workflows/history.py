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

"""Workflow Execution History - Record, Replay, and Audit.

This module provides comprehensive execution history capabilities:

Features:
- Record all workflow executions
- Store execution metadata and results
- Replay executions with same inputs
- Compare execution results
- Generate audit trails
- Export history for analysis
- Search execution history

Usage:
    history = WorkflowHistory()

    # Record execution
    history.record_execution(
        execution_id="exec_123",
        workflow_name="my_workflow",
        inputs={"task": "fix bug"},
        outputs={"result": "success"},
        duration_seconds=10.5,
    )

    # Query history
    executions = history.list_executions(workflow_name="my_workflow")

    # Replay execution
    result = history.replay_execution("exec_123")
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
)
from uuid import uuid4

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ExecutionRecord:
    """Record of a workflow execution.

    Attributes:
        execution_id: Unique execution identifier
        workflow_name: Workflow name
        timestamp: Execution timestamp
        inputs: Workflow inputs
        outputs: Workflow outputs
        success: Whether execution succeeded
        duration_seconds: Execution duration
        error: Error if failed
        nodes_executed: List of node IDs executed
        metadata: Additional metadata
    """

    execution_id: str
    workflow_name: str
    timestamp: float
    inputs: Dict[str, Any]
    outputs: Optional[Dict[str, Any]]
    success: bool
    duration_seconds: float
    error: Optional[str] = None
    nodes_executed: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionComparison:
    """Comparison of two executions.

    Attributes:
        execution_1_id: First execution ID
        execution_2_id: Second execution ID
        output_diff: Differences in outputs
        performance_diff: Performance comparison
        node_diff: Differences in node execution
    """

    execution_1_id: str
    execution_2_id: str
    output_diff: Dict[str, Any]
    performance_diff: Dict[str, Any]
    node_diff: Dict[str, Any]


# =============================================================================
# Workflow Execution History
# =============================================================================


class WorkflowExecutionHistory:
    """Workflow execution history manager.

    Records, stores, and manages workflow execution history with
    capabilities for replay, comparison, and audit.
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        max_records: int = 1000,
        auto_save: bool = True,
    ):
        """Initialize execution history.

        Args:
            storage_path: Path to history storage directory
            max_records: Maximum number of records to keep
            auto_save: Whether to auto-save after each record
        """
        self.storage_path = storage_path or Path.home() / ".victor" / "workflow_history"
        self.max_records = max_records
        self.auto_save = auto_save

        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory records
        self._records: Dict[str, ExecutionRecord] = {}
        self._record_order: List[str] = []

        # Load existing records
        self._load_records()

    # =========================================================================
    # Recording
    # =========================================================================

    def record_execution(
        self,
        execution_id: str,
        workflow_name: str,
        inputs: Dict[str, Any],
        outputs: Optional[Dict[str, Any]],
        success: bool,
        duration_seconds: float,
        error: Optional[str] = None,
        nodes_executed: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExecutionRecord:
        """Record a workflow execution.

        Args:
            execution_id: Execution identifier
            workflow_name: Workflow name
            inputs: Workflow inputs
            outputs: Workflow outputs
            success: Whether execution succeeded
            duration_seconds: Execution duration
            error: Error if failed
            nodes_executed: List of node IDs executed
            metadata: Additional metadata

        Returns:
            Execution record
        """
        record = ExecutionRecord(
            execution_id=execution_id,
            workflow_name=workflow_name,
            timestamp=time.time(),
            inputs=inputs,
            outputs=outputs,
            success=success,
            duration_seconds=duration_seconds,
            error=error,
            nodes_executed=nodes_executed or [],
            metadata=metadata or {},
        )

        # Add to storage
        self._records[execution_id] = record
        self._record_order.append(execution_id)

        # Enforce max records
        while len(self._record_order) > self.max_records:
            old_id = self._record_order.pop(0)
            del self._records[old_id]

        # Auto-save if enabled
        if self.auto_save:
            self._save_record(record)

        logger.info(f"Recorded execution '{execution_id}' for workflow '{workflow_name}'")

        return record

    def get_execution(self, execution_id: str) -> Optional[ExecutionRecord]:
        """Get an execution record by ID.

        Args:
            execution_id: Execution identifier

        Returns:
            Execution record or None
        """
        return self._records.get(execution_id)

    def list_executions(
        self,
        workflow_name: Optional[str] = None,
        limit: int = 10,
        success_only: bool = False,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[ExecutionRecord]:
        """List execution records.

        Args:
            workflow_name: Optional workflow name filter
            limit: Maximum number of records
            success_only: Only successful executions
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            List of execution records
        """
        records: List[ExecutionRecord] = []

        for exec_id in reversed(self._record_order):
            if len(records) >= limit:
                break

            record = self._records[exec_id]

            # Apply filters
            if workflow_name and record.workflow_name != workflow_name:
                continue

            if success_only and not record.success:
                continue

            if start_time and record.timestamp < start_time:
                continue

            if end_time and record.timestamp > end_time:
                continue

            records.append(record)

        return records

    # =========================================================================
    # Replay
    # =========================================================================

    def replay_execution(
        self,
        execution_id: str,
        executor: Optional[Callable[..., Any]] = None,
        override_inputs: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Replay a previous execution.

        Args:
            execution_id: Execution identifier
            executor: Optional executor function
            override_inputs: Optional input overrides

        Returns:
            Execution result

        Raises:
            ValueError: If execution not found or no executor provided
        """
        record = self.get_execution(execution_id)

        if not record:
            raise ValueError(f"Execution '{execution_id}' not found in history")

        # Use inputs from record (with overrides)
        inputs = record.inputs.copy()
        if override_inputs:
            inputs.update(override_inputs)

        logger.info(f"Replaying execution '{execution_id}'")

        # If no executor provided, return inputs for manual replay
        if executor is None:
            logger.warning("No executor provided, returning inputs for manual replay")
            return {
                "execution_id": execution_id,
                "inputs": inputs,
                "workflow_name": record.workflow_name,
            }

        # Execute with replayed inputs
        try:
            result = executor(inputs)

            # Compare with original
            comparison = self.compare_executions(execution_id, "current")

            logger.info(f"Replay completed: {comparison}")

            return result

        except Exception as e:
            logger.error(f"Replay failed: {e}")
            raise

    # =========================================================================
    # Comparison
    # =========================================================================

    def compare_executions(
        self,
        execution_id_1: str,
        execution_id_2: str,
    ) -> ExecutionComparison:
        """Compare two executions.

        Args:
            execution_id_1: First execution ID
            execution_id_2: Second execution ID

        Returns:
            Execution comparison
        """
        record_1 = self.get_execution(execution_id_1)
        record_2 = self.get_execution(execution_id_2)

        if not record_1 or not record_2:
            raise ValueError("One or both executions not found")

        # Compare outputs
        output_diff = self._compare_outputs(
            record_1.outputs or {},
            record_2.outputs or {},
        )

        # Compare performance
        performance_diff = {
            "duration_delta": record_2.duration_seconds - record_1.duration_seconds,
            "duration_ratio": (
                record_2.duration_seconds / record_1.duration_seconds
                if record_1.duration_seconds > 0
                else float("inf")
            ),
        }

        # Compare node execution
        node_diff = {
            "nodes_only_in_1": list(set(record_1.nodes_executed) - set(record_2.nodes_executed)),
            "nodes_only_in_2": list(set(record_2.nodes_executed) - set(record_1.nodes_executed)),
            "common_nodes": list(set(record_1.nodes_executed) & set(record_2.nodes_executed)),
        }

        return ExecutionComparison(
            execution_1_id=execution_id_1,
            execution_2_id=execution_id_2,
            output_diff=output_diff,
            performance_diff=performance_diff,
            node_diff=node_diff,
        )

    def _compare_outputs(
        self,
        outputs_1: Dict[str, Any],
        outputs_2: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compare two output dictionaries.

        Args:
            outputs_1: First outputs
            outputs_2: Second outputs

        Returns:
            Comparison dictionary
        """
        all_keys = set(outputs_1.keys()) | set(outputs_2.keys())

        return {
            "added_keys": list(all_keys - set(outputs_1.keys())),
            "removed_keys": list(all_keys - set(outputs_2.keys())),
            "changed_keys": [
                k
                for k in all_keys
                if k in outputs_1 and k in outputs_2 and outputs_1[k] != outputs_2[k]
            ],
            "unchanged_keys": [
                k
                for k in all_keys
                if k in outputs_1 and k in outputs_2 and outputs_1[k] == outputs_2[k]
            ],
        }

    # =========================================================================
    # Analytics
    # =========================================================================

    def get_workflow_stats(
        self,
        workflow_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get execution statistics for a workflow.

        Args:
            workflow_name: Workflow name (all workflows if None)

        Returns:
            Statistics dictionary
        """
        records = self.list_executions(
            workflow_name=workflow_name,
            limit=self.max_records,
        )

        if not records:
            return {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "average_duration": 0.0,
                "success_rate": 0.0,
            }

        successful = sum(1 for r in records if r.success)
        failed = sum(1 for r in records if not r.success)
        total_duration = sum(r.duration_seconds for r in records)

        return {
            "workflow_name": workflow_name or "all",
            "total_executions": len(records),
            "successful_executions": successful,
            "failed_executions": failed,
            "average_duration": total_duration / len(records),
            "success_rate": successful / len(records) if records else 0.0,
            "min_duration": min(r.duration_seconds for r in records),
            "max_duration": max(r.duration_seconds for r in records),
        }

    def get_execution_trends(
        self,
        workflow_name: str,
        window_size: int = 10,
    ) -> Dict[str, Any]:
        """Get execution trends over time.

        Args:
            workflow_name: Workflow name
            window_size: Window size for trend analysis

        Returns:
            Trends dictionary
        """
        records = self.list_executions(
            workflow_name=workflow_name,
            limit=window_size,
        )

        if len(records) < 2:
            return {"trend": "insufficient_data"}

        # Calculate trends
        durations = [r.duration_seconds for r in records]
        success_rate = sum(1 for r in records if r.success) / len(records)

        # Duration trend
        if durations[-1] > durations[0]:
            duration_trend = "increasing"
            duration_change = (durations[-1] - durations[0]) / durations[0] * 100
        else:
            duration_trend = "decreasing"
            duration_change = (durations[0] - durations[-1]) / durations[0] * 100

        return {
            "workflow_name": workflow_name,
            "window_size": len(records),
            "duration_trend": duration_trend,
            "duration_change_percent": duration_change,
            "average_duration": sum(durations) / len(durations),
            "success_rate": success_rate,
            "recent_executions": [
                {
                    "execution_id": r.execution_id,
                    "timestamp": r.timestamp,
                    "duration": r.duration_seconds,
                    "success": r.success,
                }
                for r in records[-5:]
            ],
        }

    # =========================================================================
    # Audit Trail
    # =========================================================================

    def generate_audit_trail(
        self,
        workflow_name: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Generate audit trail of executions.

        Args:
            workflow_name: Optional workflow name filter
            start_time: Optional start time
            end_time: Optional end time

        Returns:
            List of audit entries
        """
        records = self.list_executions(
            workflow_name=workflow_name,
            limit=self.max_records,
            start_time=start_time,
            end_time=end_time,
        )

        audit_trail = []

        for record in records:
            entry = {
                "execution_id": record.execution_id,
                "workflow_name": record.workflow_name,
                "timestamp": datetime.fromtimestamp(record.timestamp).isoformat(),
                "success": record.success,
                "duration_seconds": record.duration_seconds,
                "nodes_executed": len(record.nodes_executed),
                "error": record.error,
            }

            audit_trail.append(entry)

        return audit_trail

    def export_audit_trail(
        self,
        output_path: Path,
        workflow_name: Optional[str] = None,
        format: str = "json",
    ) -> None:
        """Export audit trail to file.

        Args:
            output_path: Output file path
            workflow_name: Optional workflow name filter
            format: Export format (json, csv)
        """
        audit_trail = self.generate_audit_trail(workflow_name=workflow_name)

        if format == "json":
            output_path.write_text(json.dumps(audit_trail, indent=2, default=str))

        elif format == "csv":
            import csv

            with open(output_path, "w", newline="") as f:
                if audit_trail:
                    writer = csv.DictWriter(f, fieldnames=audit_trail[0].keys())
                    writer.writeheader()
                    writer.writerows(audit_trail)

        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Audit trail exported to {output_path}")

    # =========================================================================
    # Storage
    # =========================================================================

    def _load_records(self) -> None:
        """Load records from storage."""
        if not self.storage_path.exists():
            return

        for file_path in self.storage_path.glob("*.json"):
            try:
                data = json.loads(file_path.read_text())

                record = ExecutionRecord(
                    execution_id=data["execution_id"],
                    workflow_name=data["workflow_name"],
                    timestamp=data["timestamp"],
                    inputs=data["inputs"],
                    outputs=data["outputs"],
                    success=data["success"],
                    duration_seconds=data["duration_seconds"],
                    error=data.get("error"),
                    nodes_executed=data.get("nodes_executed", []),
                    metadata=data.get("metadata", {}),
                )

                self._records[record.execution_id] = record
                self._record_order.append(record.execution_id)

            except Exception as e:
                logger.warning(f"Failed to load record from {file_path}: {e}")

        logger.info(f"Loaded {len(self._records)} execution records from storage")

    def _save_record(self, record: ExecutionRecord) -> None:
        """Save record to storage.

        Args:
            record: Record to save
        """
        file_path = self.storage_path / f"{record.execution_id}.json"

        data = {
            "execution_id": record.execution_id,
            "workflow_name": record.workflow_name,
            "timestamp": record.timestamp,
            "inputs": record.inputs,
            "outputs": record.outputs,
            "success": record.success,
            "duration_seconds": record.duration_seconds,
            "error": record.error,
            "nodes_executed": record.nodes_executed,
            "metadata": record.metadata,
        }

        file_path.write_text(json.dumps(data, indent=2, default=str))

    def save_all(self) -> None:
        """Save all records to storage."""
        for record_id in self._record_order:
            record = self._records[record_id]
            self._save_record(record)

        logger.info(f"Saved {len(self._records)} records to storage")

    def clear_history(self, workflow_name: Optional[str] = None) -> int:
        """Clear execution history.

        Args:
            workflow_name: Optional workflow name filter

        Returns:
            Number of records cleared
        """
        if workflow_name:
            to_remove = [
                exec_id
                for exec_id in self._record_order
                if self._records[exec_id].workflow_name == workflow_name
            ]
        else:
            to_remove = self._record_order.copy()

        for exec_id in to_remove:
            del self._records[exec_id]
            self._record_order.remove(exec_id)

            # Remove file
            file_path = self.storage_path / f"{exec_id}.json"
            if file_path.exists():
                file_path.unlink()

        logger.info(f"Cleared {len(to_remove)} execution records")

        return len(to_remove)


# =============================================================================
# Convenience Functions
# =============================================================================


def create_history(
    storage_path: Optional[Path] = None,
    max_records: int = 1000,
) -> WorkflowExecutionHistory:
    """Create a workflow execution history manager.

    Args:
        storage_path: Path to storage directory
        max_records: Maximum records to keep

    Returns:
        WorkflowExecutionHistory instance
    """
    return WorkflowExecutionHistory(
        storage_path=storage_path,
        max_records=max_records,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Data classes
    "ExecutionRecord",
    "ExecutionComparison",
    # Main class
    "WorkflowExecutionHistory",
    # Functions
    "create_history",
]
