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

"""Tool Observability Handler - Extracted from ToolCoordinator.

Handles tool completion events, preview building, statistics,
and observability integration for tool execution.

Extracted as part of E1 M3 (ToolCoordinator size reduction).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
)

if TYPE_CHECKING:
    from victor.agent.coordinators.tool_coordinator import ToolCoordinator
from victor.observability.request_correlation import get_request_correlation_id

logger = logging.getLogger(__name__)


class ToolObservabilityHandler:
    """Handles tool completion tracking, preview building, and statistics.

    This class encapsulates observability concerns that were previously
    embedded in ToolCoordinator, including:
    - Tool completion event emission
    - File read tracking and nudge generation
    - Preview building for UI consumption
    - Execution and selection statistics

    Design: Takes a coordinator reference at init to access shared state
    (selection history, execution counts, budget, etc.).
    """

    def __init__(self, coordinator: "ToolCoordinator") -> None:
        """Initialize with a reference to the parent ToolCoordinator.

        Args:
            coordinator: Parent ToolCoordinator for state access
        """
        self._coordinator = coordinator

    # =====================================================================
    # Tool Completion Tracking
    # =====================================================================

    def on_tool_complete(
        self,
        result: Any,
        metrics_collector: Any,
        read_files_session: Optional[Set[str]] = None,
        required_files: Optional[List[str]] = None,
        required_outputs: Optional[List[str]] = None,
        nudge_sent_flag: Optional[List[bool]] = None,
        add_message: Optional[Callable[[str, str], None]] = None,
        observability: Optional[Any] = None,
        pipeline_calls_used: int = 0,
    ) -> None:
        """Handle tool execution completion with event emission and file tracking.

        Args:
            result: ToolCallResult from execution
            metrics_collector: MetricsCollector to record completion
            read_files_session: Mutable set of read file paths for tracking
            required_files: List of files required for the task
            required_outputs: List of required output descriptions
            nudge_sent_flag: Single-element list used as mutable bool flag for nudge tracking
            add_message: Callback to inject messages into conversation
            observability: Optional observability handler
            pipeline_calls_used: Current pipeline call count for observability
        """
        metrics_collector.on_tool_complete(result)
        follow_up_suggestions = self._extract_follow_up_suggestions(result.result)
        tool_id = (
            getattr(result, "tool_id", None)
            or f"tool-{max(pipeline_calls_used - 1, 0)}"
        )

        # Emit tool complete event
        from victor.core.events import get_observability_bus

        bus = get_observability_bus()
        correlation_id = get_request_correlation_id()
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                bus.emit(
                    topic="tool.complete",
                    data={
                        "tool_id": tool_id,
                        "tool_name": result.tool_name,
                        "success": result.success,
                        "result_length": (
                            len(str(result.result or "")) if result.result else 0
                        ),
                        "error": str(result.error) if result.error else None,
                        "category": "tool",
                        "arguments": self._sanitize_arguments(result.arguments or {}),
                        "execution_time_ms": getattr(result, "execution_time_ms", None),
                        "result_excerpt": self._build_result_excerpt(result.result),
                        "preview": self._build_tool_preview(result),
                        "follow_up_suggestions": follow_up_suggestions,
                    },
                    correlation_id=correlation_id,
                )
            )
        except RuntimeError:
            # No event loop running
            pass
        except Exception:
            # Ignore errors during event emission
            pass

        # Track read files for task completion detection
        if (
            result.success
            and result.tool_name in ("read", "Read", "read_file")
            and read_files_session is not None
        ):
            if result.arguments:
                file_path = result.arguments.get("path") or result.arguments.get(
                    "file_path"
                )
                if file_path:
                    read_files_session.add(file_path)
                    logger.debug(f"Tracked read file: {file_path}")

                    # Check if all required files have been read - nudge to produce output
                    if (
                        required_files
                        and read_files_session.issuperset(set(required_files))
                        and nudge_sent_flag is not None
                        and not nudge_sent_flag[0]
                    ):
                        nudge_sent_flag[0] = True
                        logger.info(
                            f"All {len(required_files)} required files have been read. "
                            "Agent should now produce the required output."
                        )

                        # Emit nudge event
                        event_bus = get_observability_bus()
                        event_bus.emit(
                            topic="state.task.all_files_read_nudge",
                            data={
                                "required_files": list(required_files),
                                "read_files": list(read_files_session),
                                "required_outputs": required_outputs,
                                "action": "nudge_output_production",
                                "category": "state",
                            },
                        )

                        # Inject nudge message to encourage output production
                        if required_outputs and add_message:
                            outputs_str = ", ".join(required_outputs)
                            add_message(
                                "system",
                                f"[REMINDER] All required files have been read. "
                                f"Please now produce the required output: {outputs_str}. "
                                f"Avoid further exploration - focus on synthesizing findings.",
                            )

        # Emit observability event for tool completion
        if observability:
            observability.on_tool_end(
                tool_name=result.tool_name,
                result=result.result,
                success=result.success,
                tool_id=tool_id,
                error=result.error,
            )

    # =====================================================================
    # Statistics
    # =====================================================================

    def get_selection_stats(self) -> Dict[str, Any]:
        """Get tool selection statistics.

        Returns:
            Dict with selection method distribution and counts
        """
        method_counts: Dict[str, int] = {}
        total_selected = 0

        for method, count in self._coordinator._selection_history:
            method_counts[method] = method_counts.get(method, 0) + 1
            total_selected += count

        return {
            "total_selections": len(self._coordinator._selection_history),
            "total_tools_selected": total_selected,
            "method_distribution": method_counts,
            "avg_tools_per_selection": (
                total_selected / len(self._coordinator._selection_history)
                if self._coordinator._selection_history
                else 0
            ),
        }

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get tool execution statistics.

        Returns:
            Dict with execution counts and budget usage
        """
        return {
            "total_executions": self._coordinator._execution_count,
            "budget_used": self._coordinator._budget_used,
            "budget_total": self._coordinator._total_budget,
            "budget_remaining": self._coordinator.get_remaining_budget(),
            "budget_utilization": (
                self._coordinator._budget_used / self._coordinator._total_budget
                if self._coordinator._total_budget > 0
                else 0
            ),
            "executed_tools": list(self._coordinator._executed_tools),
            "failed_signatures_count": len(self._coordinator._failed_tool_signatures),
        }

    def get_tool_usage_stats(self) -> Dict[str, Any]:
        """Get comprehensive tool usage statistics.

        Returns:
            Dictionary with usage analytics including:
            - Selection stats (semantic/keyword/fallback counts)
            - Per-tool execution stats (calls, success rate, timing)
            - Cost tracking (by tier and total)
            - Overall metrics
        """
        return {
            "selection": self.get_selection_stats(),
            "execution": self.get_execution_stats(),
            "budget": {
                "total": self._coordinator._total_budget,
                "used": self._coordinator._budget_used,
                "remaining": self._coordinator.get_remaining_budget(),
            },
        }

    def clear_selection_history(self) -> None:
        """Clear the selection history."""
        self._coordinator._selection_history.clear()

    def clear_failed_signatures(self) -> None:
        """Clear the failed tool signatures cache."""
        self._coordinator._failed_tool_signatures.clear()

    def _extract_follow_up_suggestions(
        self, result_value: Any
    ) -> Optional[List[Dict[str, Any]]]:
        """Extract normalized follow-up suggestions from a tool result payload."""
        if not isinstance(result_value, dict):
            return None
        metadata = result_value.get("metadata")
        if not isinstance(metadata, dict):
            return None
        suggestions = metadata.get("follow_up_suggestions")
        if not isinstance(suggestions, list):
            return None
        normalized: List[Dict[str, Any]] = []
        for suggestion in suggestions:
            if not isinstance(suggestion, dict):
                continue
            command = suggestion.get("command")
            if not isinstance(command, str) or not command.strip():
                continue
            normalized.append(suggestion)
        return normalized or None

    # =====================================================================
    # Preview helpers (UI/observability integration)
    # =====================================================================

    def _sanitize_arguments(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Return a shallow copy of arguments with large values truncated."""

        def _clean(value: Any, depth: int = 0) -> Any:
            if depth > 2:
                return "…"
            if isinstance(value, str):
                text = value.strip()
                return text if len(text) <= 200 else text[:200] + "…"
            if isinstance(value, dict):
                return {k: _clean(v, depth + 1) for k, v in list(value.items())[:5]}
            if isinstance(value, list):
                return [_clean(v, depth + 1) for v in value[:5]]
            return value

        try:
            return {k: _clean(v) for k, v in (arguments or {}).items()}
        except Exception:
            return {}

    def _truncate_text(self, text: str, limit: int = 600) -> str:
        if not text:
            return ""
        return text if len(text) <= limit else text[:limit] + "…"

    def _build_result_excerpt(self, result: Any) -> Optional[str]:
        if result is None:
            return None
        if isinstance(result, str):
            text = result.strip()
        elif isinstance(result, (dict, list)):
            try:
                text = json.dumps(result, ensure_ascii=False)
            except Exception:
                text = str(result)
        else:
            text = str(result)
        text = text.strip()
        if not text:
            return None
        return self._truncate_text(text, 400)

    def _extract_text_content(self, payload: Any) -> Optional[str]:
        if isinstance(payload, str):
            return payload
        if isinstance(payload, dict):
            for key in ("content", "text", "body", "value", "preview"):
                value = payload.get(key)
                if isinstance(value, str):
                    return value
        return None

    def _looks_like_diff(self, text: str) -> bool:
        if not text or len(text) < 10:
            return False
        lines = text.splitlines()[:40]
        plus = any(line.startswith("+") for line in lines)
        minus = any(line.startswith("-") for line in lines)
        return text.lstrip().startswith("---") or "@@" in text or (plus and minus)

    def _build_tool_preview(self, result: Any) -> Optional[Dict[str, Any]]:
        """Build a structured preview payload for UI consumption."""
        tool_name = (result.tool_name or "").lower()
        arguments = result.arguments or {}
        content = result.result

        if tool_name in {"read", "read_file"}:
            text = self._extract_text_content(content)
            if text:
                return {
                    "type": "file_read",
                    "tool_name": result.tool_name,
                    "path": arguments.get("path") or arguments.get("file_path"),
                    "snippet": self._truncate_text(text),
                    "content": text,
                }

        diff_text = None
        metadata: Dict[str, Any] = {}
        if isinstance(content, dict):
            if isinstance(content.get("diff"), str):
                diff_text = content["diff"]
                metadata = {
                    "additions": content.get("additions"),
                    "deletions": content.get("deletions"),
                }
            elif isinstance(content.get("preview"), str) and self._looks_like_diff(
                content["preview"]
            ):
                diff_text = content["preview"]
        elif isinstance(content, str) and self._looks_like_diff(content):
            diff_text = content

        if diff_text:
            return {
                "type": "diff",
                "tool_name": result.tool_name,
                "path": arguments.get("path") or arguments.get("file_path"),
                "snippet": self._truncate_text(diff_text),
                "diff": diff_text,
                "metadata": metadata,
            }

        return None


__all__ = [
    "ToolObservabilityHandler",
]
