from unittest.mock import MagicMock, patch
import pytest
from victor.agent.orchestrator import AgentOrchestrator
from victor.analytics.logger import UsageLogger
from victor.config.settings import Settings


@pytest.fixture
def mock_usage_logger():
    """Fixture to provide a mocked UsageLogger."""
    return MagicMock(spec=UsageLogger)


@pytest.fixture
def orchestrator(mock_usage_logger: MagicMock) -> AgentOrchestrator:
    """Fixture to create an AgentOrchestrator with a mocked logger."""
    settings = Settings(analytics_enabled=True)
    mock_provider = MagicMock()

    # Patch at the import location in orchestrator module
    with patch("victor.agent.orchestrator.UsageLogger", return_value=mock_usage_logger):
        orc = AgentOrchestrator(
            settings=settings,
            provider=mock_provider,
            model="test-model",
        )
    # Replace the logger with our mock after creation
    orc.usage_logger = mock_usage_logger
    return orc


def test_orchestrator_initializes_logger(mock_usage_logger: MagicMock):
    """Tests that the orchestrator initializes and uses the UsageLogger."""
    settings = Settings(analytics_enabled=True)
    mock_provider = MagicMock()

    with patch(
        "victor.agent.orchestrator.UsageLogger", return_value=mock_usage_logger
    ) as MockLogger:
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
    from victor.agent.tool_executor import ToolExecutionResult

    # Use an existing registered tool
    tool_calls = [{"name": "read_file", "arguments": {"path": "/tmp/test.txt"}}]

    # Mock the tool_executor.execute to return a ToolExecutionResult
    mock_exec_result = ToolExecutionResult(
        tool_name="read_file",
        success=True,
        result="File contents",
        error=None,
    )

    with patch.object(orchestrator.tool_executor, "execute", return_value=mock_exec_result):
        await orchestrator._handle_tool_calls(tool_calls)

    # Check for tool_call log
    mock_usage_logger.log_event.assert_any_call(
        "tool_call", {"tool_name": "read_file", "tool_args": {"path": "/tmp/test.txt"}}
    )

    # Check for tool_result log
    mock_usage_logger.log_event.assert_any_call(
        "tool_result",
        {
            "tool_name": "read_file",
            "success": True,
            "result": "File contents",
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
