"""SDK-owned workflow handler contracts for extracted verticals."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, runtime_checkable


class ExecutorNodeStatus(str, Enum):
    """Execution status for workflow compute handlers."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class NodeResult:
    """Result from executing a workflow node."""

    node_id: str
    status: ExecutorNodeStatus
    output: Optional[Any] = None
    error: Optional[str] = None
    duration_seconds: float = 0.0
    tool_calls_used: int = 0

    @property
    def success(self) -> bool:
        return self.status == ExecutorNodeStatus.COMPLETED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "duration_seconds": self.duration_seconds,
            "tool_calls_used": self.tool_calls_used,
        }


@runtime_checkable
class ComputeNodeProtocol(Protocol):
    """Structural protocol for compute nodes consumed by extracted handlers."""

    id: str
    input_mapping: Dict[str, Any]
    output_key: Optional[str]


@runtime_checkable
class WorkflowContextProtocol(Protocol):
    """Structural protocol for workflow context consumed by extracted handlers."""

    def get(self, key: str, default: Any = None) -> Any: ...

    def set(self, key: str, value: Any) -> None: ...


@runtime_checkable
class ComputeHandlerProtocol(Protocol):
    """Async handler protocol for compute-node execution."""

    async def __call__(
        self,
        node: ComputeNodeProtocol,
        context: WorkflowContextProtocol,
        tool_registry: Any,
    ) -> NodeResult: ...


ComputeHandlerRegistrar = Callable[[str, ComputeHandlerProtocol], None]


def register_compute_handlers(
    registrar: Optional[ComputeHandlerRegistrar],
    handlers: Mapping[str, ComputeHandlerProtocol],
) -> Mapping[str, ComputeHandlerProtocol]:
    """Register handlers through an explicit host-side registrar when provided."""

    if registrar is not None:
        for name, handler in handlers.items():
            registrar(name, handler)
    return handlers


__all__ = [
    "ComputeHandlerProtocol",
    "ComputeHandlerRegistrar",
    "ComputeNodeProtocol",
    "ExecutorNodeStatus",
    "NodeResult",
    "WorkflowContextProtocol",
    "register_compute_handlers",
]
