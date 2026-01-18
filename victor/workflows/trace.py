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

"""Workflow Execution Tracing.

This module provides comprehensive execution tracing capabilities:

Features:
- Trace all node executions with inputs and outputs
- Log execution time and resource usage
- Record tool calls and LLM invocations
- Export traces in multiple formats (JSON, CSV, HTML)
- Visualize execution flow
- Analyze performance bottlenecks
- Generate trace reports

Usage:
    tracer = WorkflowTracer()
    tracer.start_trace("workflow_1", {"input": "data"})

    tracer.trace_node("node_1", {"key": "value"}, {"result": "output"})
    tracer.trace_tool_call("read_file", {"path": "file.py"}, {"content": "..."})

    tracer.end_trace()
    trace_data = tracer.get_trace()
    tracer.export_trace("output.json")
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
    Dict,
    List,
    Optional,
    Union,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ToolCallRecord:
    """Record of a tool call.

    Attributes:
        tool_name: Name of tool called
        inputs: Tool inputs
        outputs: Tool outputs
        duration_seconds: Execution time
        success: Whether call succeeded
        error: Error if failed
        timestamp: Call timestamp
    """

    tool_name: str
    inputs: Dict[str, Any]
    outputs: Optional[Dict[str, Any]]
    duration_seconds: float
    success: bool
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class NodeTraceRecord:
    """Trace record for a node execution.

    Attributes:
        node_id: Node ID
        node_type: Type of node
        inputs: Node inputs
        outputs: Node outputs
        duration_seconds: Execution time
        tool_calls: Tool calls made
        success: Whether execution succeeded
        error: Error if failed
        timestamp: Execution timestamp
        metadata: Additional metadata
    """

    node_id: str
    node_type: str
    inputs: Dict[str, Any]
    outputs: Optional[Dict[str, Any]]
    duration_seconds: float
    tool_calls: List[ToolCallRecord]
    success: bool
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowTrace:
    """Complete workflow execution trace.

    Attributes:
        trace_id: Unique trace identifier
        workflow_name: Workflow name
        start_time: Start timestamp
        end_time: End timestamp
        inputs: Workflow inputs
        outputs: Workflow outputs
        nodes: Node trace records
        metadata: Trace metadata
    """

    trace_id: str
    workflow_name: str
    start_time: float
    end_time: Optional[float]
    inputs: Dict[str, Any]
    outputs: Optional[Dict[str, Any]]
    nodes: List[NodeTraceRecord]
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Workflow Tracer
# =============================================================================


class WorkflowTracer:
    """Workflow execution tracer.

    Records detailed traces of workflow execution including node
    executions, tool calls, and performance metrics.
    """

    def __init__(self, auto_tool_calls: bool = True):
        """Initialize workflow tracer.

        Args:
            auto_tool_calls: Whether to automatically trace tool calls
        """
        self.auto_tool_calls = auto_tool_calls

        # Current trace
        self._current_trace: Optional[WorkflowTrace] = None
        self._current_node: Optional[NodeTraceRecord] = None

        # Trace history
        self._traces: List[WorkflowTrace] = []

    def start_trace(
        self,
        workflow_name: str,
        inputs: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Start a new trace.

        Args:
            workflow_name: Workflow name
            inputs: Workflow inputs
            metadata: Optional metadata

        Returns:
            Trace ID
        """
        import uuid

        trace_id = str(uuid.uuid4())

        self._current_trace = WorkflowTrace(
            trace_id=trace_id,
            workflow_name=workflow_name,
            start_time=time.time(),
            end_time=None,
            inputs=inputs,
            outputs=None,
            nodes=[],
            metadata=metadata or {},
        )

        logger.info(f"Started trace '{trace_id}' for workflow '{workflow_name}'")
        return trace_id

    def end_trace(
        self,
        outputs: Optional[Dict[str, Any]] = None,
    ) -> Optional[WorkflowTrace]:
        """End current trace.

        Args:
            outputs: Workflow outputs

        Returns:
            Completed trace or None if no active trace
        """
        if self._current_trace is None:
            logger.warning("No active trace to end")
            return None

        # End current node if active
        if self._current_node:
            self.end_node()

        # Finalize trace
        self._current_trace.end_time = time.time()
        self._current_trace.outputs = outputs

        # Add to history
        self._traces.append(self._current_trace)

        logger.info(
            f"Ended trace '{self._current_trace.trace_id}' "
            f"({self._current_trace.end_time - self._current_trace.start_time:.3f}s)"
        )

        trace = self._current_trace
        self._current_trace = None

        return trace

    def start_node(
        self,
        node_id: str,
        node_type: str,
        inputs: Dict[str, Any],
    ) -> None:
        """Start tracing a node execution.

        Args:
            node_id: Node ID
            node_type: Node type
            inputs: Node inputs
        """
        if self._current_trace is None:
            logger.warning("No active trace, cannot start node")
            return

        # End current node if active
        if self._current_node:
            self.end_node()

        # Create new node record
        self._current_node = NodeTraceRecord(
            node_id=node_id,
            node_type=node_type,
            inputs=inputs,
            outputs=None,
            duration_seconds=0.0,
            tool_calls=[],
            success=True,
            timestamp=time.time(),
        )

        logger.debug(f"Started node trace: {node_id} ({node_type})")

    def end_node(
        self,
        outputs: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """End tracing current node.

        Args:
            outputs: Node outputs
            success: Whether execution succeeded
            error: Error if failed
        """
        if self._current_node is None:
            logger.warning("No active node to end")
            return

        # Finalize node record
        self._current_node.outputs = outputs
        self._current_node.duration_seconds = time.time() - self._current_node.timestamp
        self._current_node.success = success
        self._current_node.error = error

        # Add to trace
        if self._current_trace:
            self._current_trace.nodes.append(self._current_node)

        logger.debug(
            f"Ended node trace: {self._current_node.node_id} "
            f"({self._current_node.duration_seconds:.3f}s)"
        )

        self._current_node = None

    def trace_node(
        self,
        node_id: str,
        node_type: str,
        inputs: Dict[str, Any],
        outputs: Optional[Dict[str, Any]] = None,
        duration_seconds: float = 0.0,
        tool_calls: Optional[List[ToolCallRecord]] = None,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """Trace a node execution (single call).

        Args:
            node_id: Node ID
            node_type: Node type
            inputs: Node inputs
            outputs: Node outputs
            duration_seconds: Execution time
            tool_calls: Tool calls made
            success: Whether execution succeeded
            error: Error if failed
        """
        if self._current_trace is None:
            logger.warning("No active trace, cannot trace node")
            return

        record = NodeTraceRecord(
            node_id=node_id,
            node_type=node_type,
            inputs=inputs,
            outputs=outputs,
            duration_seconds=duration_seconds,
            tool_calls=tool_calls or [],
            success=success,
            error=error,
            timestamp=time.time(),
        )

        self._current_trace.nodes.append(record)

        logger.debug(
            f"Traced node: {node_id} ({node_type}) "
            f"success={success} duration={duration_seconds:.3f}s"
        )

    def trace_tool_call(
        self,
        tool_name: str,
        inputs: Dict[str, Any],
        outputs: Optional[Dict[str, Any]],
        duration_seconds: float,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """Trace a tool call.

        Args:
            tool_name: Tool name
            inputs: Tool inputs
            outputs: Tool outputs
            duration_seconds: Execution time
            success: Whether call succeeded
            error: Error if failed
        """
        record = ToolCallRecord(
            tool_name=tool_name,
            inputs=inputs,
            outputs=outputs,
            duration_seconds=duration_seconds,
            success=success,
            error=error,
        )

        # Add to current node if active
        if self._current_node:
            self._current_node.tool_calls.append(record)

        logger.debug(
            f"Traced tool call: {tool_name} " f"success={success} duration={duration_seconds:.3f}s"
        )

    def get_trace(self) -> Optional[WorkflowTrace]:
        """Get current trace.

        Returns:
            Current trace or None
        """
        return self._current_trace

    def get_trace_history(self) -> List[WorkflowTrace]:
        """Get trace history.

        Returns:
            List of completed traces
        """
        return self._traces.copy()

    def get_trace_by_id(self, trace_id: str) -> Optional[WorkflowTrace]:
        """Get trace by ID.

        Args:
            trace_id: Trace ID

        Returns:
            Trace or None
        """
        for trace in self._traces:
            if trace.trace_id == trace_id:
                return trace
        return None

    # =========================================================================
    # Analysis and Reporting
    # =========================================================================

    def analyze_trace(self, trace: Optional[WorkflowTrace] = None) -> Dict[str, Any]:
        """Analyze trace for insights.

        Args:
            trace: Trace to analyze (uses current if None)

        Returns:
            Analysis results
        """
        trace = trace or self._current_trace
        if not trace:
            return {}

        # Calculate metrics
        total_duration = (trace.end_time or time.time()) - trace.start_time

        node_durations = {}
        tool_usage = {}
        error_count = 0

        for node in trace.nodes:
            # Node durations
            if node.node_id not in node_durations:
                node_durations[node.node_id] = 0.0
            node_durations[node.node_id] += node.duration_seconds

            # Tool usage
            for tool_call in node.tool_calls:
                if tool_call.tool_name not in tool_usage:
                    tool_usage[tool_call.tool_name] = 0
                tool_usage[tool_call.tool_name] += 1

            # Errors
            if not node.success:
                error_count += 1

        # Find slowest nodes
        slowest_nodes = sorted(
            node_durations.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        # Find most used tools
        most_used_tools = sorted(
            tool_usage.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        return {
            "trace_id": trace.trace_id,
            "workflow_name": trace.workflow_name,
            "total_duration_seconds": total_duration,
            "total_nodes": len(trace.nodes),
            "successful_nodes": sum(1 for n in trace.nodes if n.success),
            "failed_nodes": sum(1 for n in trace.nodes if not n.success),
            "total_tool_calls": sum(len(n.tool_calls) for n in trace.nodes),
            "slowest_nodes": [{"node_id": nid, "duration": dur} for nid, dur in slowest_nodes],
            "most_used_tools": [{"tool": tool, "calls": count} for tool, count in most_used_tools],
            "error_count": error_count,
        }

    # =========================================================================
    # Export
    # =========================================================================

    def export_trace(
        self,
        output_path: Path,
        format: str = "json",
        trace: Optional[WorkflowTrace] = None,
    ) -> None:
        """Export trace to file.

        Args:
            output_path: Output file path
            format: Export format (json, csv, html)
            trace: Trace to export (uses current if None)
        """
        trace = trace or self._current_trace
        if not trace:
            raise ValueError("No trace to export")

        if format == "json":
            self._export_json(trace, output_path)
        elif format == "csv":
            self._export_csv(trace, output_path)
        elif format == "html":
            self._export_html(trace, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")

        logger.info(f"Trace exported to {output_path}")

    def _export_json(self, trace: WorkflowTrace, output_path: Path) -> None:
        """Export trace as JSON.

        Args:
            trace: Trace to export
            output_path: Output file path
        """
        data = {
            "trace_id": trace.trace_id,
            "workflow_name": trace.workflow_name,
            "start_time": trace.start_time,
            "end_time": trace.end_time,
            "duration_seconds": (trace.end_time or time.time()) - trace.start_time,
            "inputs": trace.inputs,
            "outputs": trace.outputs,
            "metadata": trace.metadata,
            "nodes": [
                {
                    "node_id": node.node_id,
                    "node_type": node.node_type,
                    "timestamp": node.timestamp,
                    "duration_seconds": node.duration_seconds,
                    "success": node.success,
                    "error": node.error,
                    "inputs": node.inputs,
                    "outputs": node.outputs,
                    "tool_calls": [
                        {
                            "tool_name": tc.tool_name,
                            "duration_seconds": tc.duration_seconds,
                            "success": tc.success,
                            "error": tc.error,
                        }
                        for tc in node.tool_calls
                    ],
                }
                for node in trace.nodes
            ],
        }

        output_path.write_text(json.dumps(data, indent=2, default=str))

    def _export_csv(self, trace: WorkflowTrace, output_path: Path) -> None:
        """Export trace as CSV.

        Args:
            trace: Trace to export
            output_path: Output file path
        """
        import csv

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(
                [
                    "node_id",
                    "node_type",
                    "timestamp",
                    "duration_seconds",
                    "success",
                    "error",
                    "tool_calls_count",
                ]
            )

            # Rows
            for node in trace.nodes:
                writer.writerow(
                    [
                        node.node_id,
                        node.node_type,
                        datetime.fromtimestamp(node.timestamp).isoformat(),
                        f"{node.duration_seconds:.3f}",
                        node.success,
                        node.error or "",
                        len(node.tool_calls),
                    ]
                )

    def _export_html(self, trace: WorkflowTrace, output_path: Path) -> None:
        """Export trace as HTML report.

        Args:
            trace: Trace to export
            output_path: Output file path
        """
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Workflow Trace: {trace.workflow_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .success {{ color: green; }}
        .error {{ color: red; }}
        .summary {{ margin: 20px 0; padding: 15px; background: #f9f9f9; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>Workflow Trace: {trace.workflow_name}</h1>

    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Trace ID:</strong> {trace.trace_id}</p>
        <p><strong>Duration:</strong> {(trace.end_time or time.time()) - trace.start_time:.3f}s</p>
        <p><strong>Nodes Executed:</strong> {len(trace.nodes)}</p>
        <p><strong>Successful:</strong> {sum(1 for n in trace.nodes if n.success)}</p>
        <p><strong>Failed:</strong> {sum(1 for n in trace.nodes if not n.success)}</p>
    </div>

    <h2>Node Execution</h2>
    <table>
        <tr>
            <th>Node ID</th>
            <th>Type</th>
            <th>Duration</th>
            <th>Status</th>
            <th>Tool Calls</th>
        </tr>
"""

        for node in trace.nodes:
            status_class = "success" if node.success else "error"
            status_text = "Success" if node.success else "Failed"

            html += f"""
        <tr>
            <td>{node.node_id}</td>
            <td>{node.node_type}</td>
            <td>{node.duration_seconds:.3f}s</td>
            <td class="{status_class}">{status_text}</td>
            <td>{len(node.tool_calls)}</td>
        </tr>
"""

        html += """
    </table>
</body>
</html>
"""

        output_path.write_text(html)


# =============================================================================
# Convenience Functions
# =============================================================================


def trace_workflow(
    workflow_name: str,
    inputs: Dict[str, Any],
) -> WorkflowTracer:
    """Create and start a workflow trace.

    Args:
        workflow_name: Workflow name
        inputs: Workflow inputs

    Returns:
        WorkflowTracer with started trace
    """
    tracer = WorkflowTracer()
    tracer.start_trace(workflow_name, inputs)
    return tracer


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Data classes
    "ToolCallRecord",
    "NodeTraceRecord",
    "WorkflowTrace",
    # Main class
    "WorkflowTracer",
    # Functions
    "trace_workflow",
]
