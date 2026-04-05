from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import victor.security.audit.manager as audit_manager_module
from victor.security.audit.protocol import AuditEventType, Severity


def _make_manager() -> audit_manager_module.AuditManager:
    manager = object.__new__(audit_manager_module.AuditManager)
    manager._config = SimpleNamespace(enabled=True)
    manager._logger = SimpleNamespace(log_event=Mock())
    manager._session_id = "session-123"
    return manager


class TestAuditManagerSyncBridge:
    def test_log_event_uses_shared_sync_bridge_without_running_loop(self) -> None:
        manager = _make_manager()
        event = object()
        coro = object()
        manager._logger.log_event.return_value = coro

        with (
            patch.object(
                audit_manager_module, "create_event", return_value=event
            ) as mock_create_event,
            patch.object(
                audit_manager_module.asyncio, "get_running_loop", side_effect=RuntimeError
            ),
            patch.object(audit_manager_module, "run_sync", return_value=None) as mock_run_sync,
        ):
            manager.log_event(
                AuditEventType.SECURITY_SCAN,
                "scanned project",
                details={"path": "."},
                resource="repo",
                severity=Severity.WARNING,
            )

        mock_create_event.assert_called_once_with(
            event_type=AuditEventType.SECURITY_SCAN,
            action="scanned project",
            severity=Severity.WARNING,
            resource="repo",
            metadata={"path": "."},
            session_id="session-123",
        )
        manager._logger.log_event.assert_called_once_with(event)
        mock_run_sync.assert_called_once_with(coro)

    def test_log_event_schedules_task_when_loop_running(self) -> None:
        manager = _make_manager()
        event = object()
        coro = object()
        manager._logger.log_event.return_value = coro
        loop = Mock()

        with (
            patch.object(audit_manager_module, "create_event", return_value=event),
            patch.object(audit_manager_module.asyncio, "get_running_loop", return_value=loop),
            patch.object(audit_manager_module, "run_sync") as mock_run_sync,
        ):
            manager.log_event(AuditEventType.TOOL_EXECUTION, "ran tool")

        manager._logger.log_event.assert_called_once_with(event)
        loop.create_task.assert_called_once_with(coro)
        mock_run_sync.assert_not_called()
