from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, List, Optional

from victor.agent.streaming.intent_classification import IntentClassificationResult
from victor.providers.base import StreamChunk


class DummyTaskType:
    def __init__(self, value: str) -> None:
        self.value = value


class DummyStreamContext:
    def __init__(self) -> None:
        self.max_total_iterations = 1
        self.max_exploration_iterations = 1
        self.is_analysis_task = False
        self.unified_task_type = DummyTaskType("direct")
        self.coarse_task_type = "coding"
        self.is_action_task = False
        self.needs_execution = False
        self.context_msg = "ctx"
        self.total_iterations = 0
        self.last_quality_score = 1.0
        self.pending_grounding_feedback: Optional[str] = None
        self.total_tokens = 0
        self.total_accumulated_chars = 0
        self.force_completion = False

    def accumulate_content(self, content: str) -> None:
        self.total_accumulated_chars = len(content)

    def update_quality_score(self, score: float) -> None:
        self.last_quality_score = score


class OrchestratorStub:
    def __init__(self) -> None:
        self._tool_planner = SimpleNamespace(infer_goals_from_message=lambda _: [])
        self.tool_budget = 0
        self.debug_logger = SimpleNamespace(
            reset=lambda: None,
            log_iteration_start=lambda *_, **__: None,
            log_limits=lambda *_, **__: None,
        )
        self.thinking = None
        self._required_files: List[str] = []
        self._required_outputs: List[str] = []
        self._read_files_session = set()
        self._all_files_read_nudge_sent = False
        self._task_completion_detector = None
        self.sanitizer = SimpleNamespace(sanitize=lambda text: text, strip_markup=lambda text: text)
        self.add_message = lambda role, content: None
        self._recovery_coordinator = SimpleNamespace(
            check_natural_completion=lambda *_, **__: None,
            get_recovery_fallback_message=lambda *_, **__: "fallback",
            handle_empty_response=lambda *_, **__: (None, False),
            check_force_action=lambda *_, **__: (False, None),
        )
        self._recovery_integration = SimpleNamespace(record_outcome=lambda **__: None)
        self._chunk_generator = SimpleNamespace(
            generate_content_chunk=lambda content, is_final=False: StreamChunk(
                content=content, is_final=is_final
            )
        )
        self.unified_tracker = SimpleNamespace(
            record_tool_call=lambda *_, **__: None,
            record_iteration=lambda *_, **__: None,
            check_loop_warning=lambda: None,
            get_metrics=lambda: {},
            unique_resources=set(),
        )
        self._streaming_handler = SimpleNamespace(handle_loop_warning=lambda *_, **__: None)
        self.tool_calls_used = 0
        self.observed_files: List[str] = []
        self._continuation_prompts = 0
        self._asking_input_prompts = 0
        self._consecutive_blocked_attempts = 0
        self._cumulative_prompt_interventions = 0
        self._record_intelligent_outcome = lambda **__: None
        self._force_finalize = False
        self._container = None

    # Methods moved from DummyCoordinator to OrchestratorStub (pipeline now calls orch directly)
    def _apply_intent_guard(self, message: str):
        return None

    def _apply_task_guidance(self, *args, **kwargs):
        return None

    def _get_max_context_chars(self):
        return 1000

    async def _handle_context_and_iteration_limits(self, *args, **kwargs):
        return (True, None)

    async def _select_tools_for_turn(self, *args, **kwargs):
        return []

    def _parse_and_validate_tool_calls(self, tool_calls, full_content):
        return tool_calls, full_content

    async def _handle_recovery_with_integration(self, *args, **kwargs):
        return SimpleNamespace(action="continue")

    def _apply_recovery_action(self, *args, **kwargs):
        return None

    def _create_recovery_context(self, *_):
        return object()

    async def _validate_intelligent_response(self, *args, **kwargs):
        return None


class DummyCoordinator:
    """Minimal coordinator façade for StreamingChatPipeline tests."""

    def __init__(self, pre_chunks=None, limit_result=None) -> None:
        self._stream_ctx = DummyStreamContext()
        self._pre_chunks = pre_chunks or []
        self._create_stream_calls: List[str] = []
        self._provider_response = ("", None, None, False)
        self._empty_recovery = (False, None, None)
        self._intent_classification_handler = None
        self._continuation_handler = None
        self._tool_execution_handler = None
        self._orchestrator = OrchestratorStub()

        # Override orchestrator defaults if custom limit_result provided
        if limit_result is not None:
            self._orchestrator._handle_context_and_iteration_limits = self._make_limit_handler(
                limit_result
            )

    @staticmethod
    def _make_limit_handler(result):
        async def handler(*args, **kwargs):
            return result

        return handler

    async def _create_stream_context(self, user_message: str):
        self._create_stream_calls.append(user_message)
        return self._stream_ctx

    async def _run_iteration_pre_checks(self, *args, **kwargs):
        for chunk in self._pre_chunks:
            yield chunk

    async def _stream_provider_response(self, *args, **kwargs):
        return self._provider_response

    async def _handle_empty_response_recovery(self, *args, **kwargs):
        return self._empty_recovery


class StubIntentHandler:
    def __init__(self, result: IntentClassificationResult) -> None:
        self.result = result
        self.calls = []

    def classify_and_determine_action(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


@dataclass
class StubContinuationResult:
    chunks: List[StreamChunk]
    state_updates: dict
    should_return: bool


class StubContinuationHandler:
    def __init__(self, result: StubContinuationResult) -> None:
        self.result = result
        self.calls = []

    async def handle_action(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


@dataclass
class StubToolExecutionResult:
    chunks: List[StreamChunk]
    tool_calls_executed: int
    should_return: bool


class StubToolExecutionHandler:
    def __init__(self, result: StubToolExecutionResult) -> None:
        self.result = result
        self.updated_files = None
        self.calls = []

    def update_observed_files(self, files):
        self.updated_files = files

    async def execute_tools(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


__all__ = [
    "DummyCoordinator",
    "DummyStreamContext",
    "DummyTaskType",
    "OrchestratorStub",
    "StubIntentHandler",
    "StubContinuationHandler",
    "StubContinuationResult",
    "StubToolExecutionHandler",
    "StubToolExecutionResult",
]
