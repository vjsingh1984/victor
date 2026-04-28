"""Tests for canonical chat message persistence and shim delegation."""

from unittest.mock import MagicMock, patch

import pytest

from victor.agent.services.chat_compat import ChatCoordinator
from victor.agent.services.chat_service import ChatService


class TestPersistMessage:
    def test_persists_to_memory_manager(self):
        """ChatService.persist_message calls memory_manager.add_message."""
        mm = MagicMock()
        logger = MagicMock(spec=["log_event"])

        # Patch get_running_loop to simulate no event loop (sync path)
        with patch("asyncio.get_running_loop", side_effect=RuntimeError):
            ChatService.persist_message(
                role="user",
                content="hello",
                memory_manager=mm,
                memory_session_id="sess-1",
                usage_logger=logger,
            )

        mm.add_message.assert_called_once()
        call_args = mm.add_message.call_args
        assert call_args.kwargs["session_id"] == "sess-1"
        assert call_args.kwargs["content"] == "hello"

    def test_persists_metadata_to_memory_manager(self):
        mm = MagicMock()
        logger = MagicMock(spec=["log_event"])
        metadata = {"interactive_history": False, "internal_prompt_kind": "prompt_tool_call"}

        with patch("asyncio.get_running_loop", side_effect=RuntimeError):
            ChatService.persist_message(
                role="user",
                content="Continue. Use appropriate tools if needed.",
                memory_manager=mm,
                memory_session_id="sess-1",
                usage_logger=logger,
                metadata=metadata,
            )

        assert mm.add_message.call_args.kwargs["metadata"] == metadata

    def test_logs_user_prompt_event(self):
        logger = MagicMock(spec=["log_event"])

        ChatService.persist_message(
            role="user",
            content="test query",
            memory_manager=None,
            memory_session_id=None,
            usage_logger=logger,
        )

        logger.log_event.assert_called_once_with("user_prompt", {"content": "test query"})

    def test_logs_assistant_response_event(self):
        logger = MagicMock(spec=["log_event", "set_reasoning_context"])

        ChatService.persist_message(
            role="assistant",
            content="here's the answer",
            memory_manager=None,
            memory_session_id=None,
            usage_logger=logger,
        )

        logger.log_event.assert_called_once_with(
            "assistant_response", {"content": "here's the answer"}
        )
        logger.set_reasoning_context.assert_called_once_with("here's the answer")

    def test_system_role_no_event(self):
        logger = MagicMock(spec=["log_event"])

        ChatService.persist_message(
            role="system",
            content="sys prompt",
            memory_manager=None,
            memory_session_id=None,
            usage_logger=logger,
        )

        logger.log_event.assert_not_called()

    def test_no_memory_manager_skips_persistence(self):
        logger = MagicMock(spec=["log_event"])

        # Should not raise
        ChatService.persist_message(
            role="user",
            content="hello",
            memory_manager=None,
            memory_session_id=None,
            usage_logger=logger,
        )

    def test_memory_error_is_swallowed(self):
        mm = MagicMock()
        mm.add_message.side_effect = RuntimeError("db locked")
        logger = MagicMock(spec=["log_event"])

        # Should not raise
        ChatService.persist_message(
            role="user",
            content="hello",
            memory_manager=mm,
            memory_session_id="sess-1",
            usage_logger=logger,
        )

        # Usage event still logged despite persistence failure
        logger.log_event.assert_called_once()

    def test_chat_coordinator_persist_message_warns_and_delegates(self):
        with patch("victor.agent.services.chat_service.ChatService.persist_message") as persist:
            with pytest.warns(
                DeprecationWarning,
                match="ChatCoordinator.persist_message\\(\\) is deprecated compatibility surface",
            ):
                ChatCoordinator.persist_message(
                    role="user",
                    content="hello",
                    memory_manager=None,
                    memory_session_id=None,
                    usage_logger=None,
                )

        persist.assert_called_once_with(
            role="user",
            content="hello",
            memory_manager=None,
            memory_session_id=None,
            usage_logger=None,
            tool_name=None,
            tool_call_id=None,
            tool_calls=None,
            metadata=None,
        )
