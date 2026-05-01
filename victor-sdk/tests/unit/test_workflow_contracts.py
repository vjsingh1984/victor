"""Tests for SDK-owned workflow handler contracts."""

from victor_sdk.workflows import (
    ExecutorNodeStatus,
    NodeResult,
    register_compute_handlers,
)


def test_node_result_success_tracks_completed_status() -> None:
    result = NodeResult(node_id="demo", status=ExecutorNodeStatus.COMPLETED, output={"ok": True})

    assert result.success is True
    assert result.to_dict()["status"] == "completed"


def test_register_compute_handlers_registers_all_entries() -> None:
    calls = []

    def registrar(name, handler):
        calls.append((name, handler))

    handlers = {"alpha": object(), "beta": object()}

    returned_handlers = register_compute_handlers(registrar, handlers)

    assert returned_handlers is handlers
    assert [name for name, _ in calls] == ["alpha", "beta"]
