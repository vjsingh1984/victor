from unittest.mock import MagicMock, patch
import pytest
from victor.agent.orchestrator import AgentOrchestrator
from victor.analytics.logger import UsageLogger
from victor.config.settings import Settings
from victor.providers.base import Message


@pytest.fixture
def mock_usage_logger():
    """Fixture to provide a mocked UsageLogger."""
    return MagicMock(spec=UsageLogger)


@pytest.fixture
def orchestrator(mock_usage_logger: MagicMock) -> AgentOrchestrator:
    """Fixture to create an AgentOrchestrator with a mocked logger."""
    settings = Settings(analytics_enabled=True)
    mock_provider = MagicMock()

    # Patch the logger inside the orchestrator's init
    with patch("victor.analytics.logger.UsageLogger", return_value=mock_usage_logger):
        orc = AgentOrchestrator(
            settings=settings,
            provider=mock_provider,
            model="test-model",
        )
    return orc


def test_orchestrator_initializes_logger(mock_usage_logger: MagicMock):
    """Tests that the orchestrator initializes and uses the UsageLogger."""
    settings = Settings(analytics_enabled=True)
    mock_provider = MagicMock()

    with patch("victor.analytics.logger.UsageLogger", return_value=mock_usage_logger) as MockLogger:
        AgentOrchestrator(
            settings=settings,
            provider=mock_provider,
            model="test-model",
        )
        MockLogger.assert_called_once()


def test_add_message_logs_user_prompt(
    orchestrator: AgentOrchestrator, mock_usage_logger: MagicMock
):
    """Tests that add_message logs a 'user_prompt' event."""
    orchestrator.add_message(role="user", content="Hello, world!")
    mock_usage_logger.log_event.assert_called_with("user_prompt", {"content": "Hello, world!"})


def test_add_message_logs_assistant_response(
    orchestrator: AgentOrchestrator, mock_usage_logger: MagicMock
):
    """Tests that add_message logs an 'assistant_response' event."""
    orchestrator.add_message(role="assistant", content="Hi there!")
    mock_usage_logger.log_event.assert_called_with("assistant_response", {"content": "Hi there!"})


@pytest.mark.asyncio
async def test_handle_tool_calls_logs_events(
    orchestrator: AgentOrchestrator, mock_usage_logger: MagicMock
):
    """Tests that _handle_tool_calls logs tool_call and tool_result events."""
    tool_calls = [{"name": "test_tool", "arguments": {"arg1": "val1"}}]

    # Mock the tool execution
    mock_result = MagicMock()
    mock_result.success = True
    mock_result.output = "Success"

    with patch.object(orchestrator.tools, "execute", return_value=mock_result) as mock_execute:
        await orchestrator._handle_tool_calls(tool_calls)

    # Check for tool_call log
    mock_usage_logger.log_event.assert_any_call(
        "tool_call", {"tool_name": "test_tool", "tool_args": {"arg1": "val1"}}
    )

    # Check for tool_result log
    mock_usage_logger.log_event.assert_any_call(
        "tool_result",
        {
            "tool_name": "test_tool",
            "success": True,
            "result": "Success",
            "error": None,
        },
    )


def test_record_tool_selection_logs_event(
    orchestrator: AgentOrchestrator, mock_usage_logger: MagicMock
):
    """Tests that _record_tool_selection logs a 'tool_selection' event."""
    orchestrator._record_tool_selection("semantic", 5)
    mock_usage_logger.log_event.assert_called_with(
        "tool_selection", {"method": "semantic", "tool_count": 5}
    )
