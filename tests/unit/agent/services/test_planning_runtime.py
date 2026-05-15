import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.agent.planning.base import PlanResult, StepResult
from victor.agent.planning.readable_schema import ReadableTaskPlan, TaskComplexity
from victor.agent.services.planning_runtime import (
    PlanningConfig,
    PlanningMode,
    PlanningRuntimeService,
)
from victor.framework.execution_checkpoint import ApprovalState
from victor.providers.base import CompletionResponse


@pytest.mark.asyncio
async def test_execute_plan_routes_complex_plan_through_team_adapter():
    orchestrator = SimpleNamespace(active_session_id="session_root")
    service = PlanningRuntimeService(orchestrator)
    plan = ReadableTaskPlan(
        name="Rust Arc Review",
        complexity=TaskComplexity.COMPLEX,
        desc="Review Rust Arc usage by workspace",
        steps=[
            ["1", "research", "Map Rust workspaces", "ls,read"],
            ["2", "review", "Review Arc usage", "grep,read", ["1"]],
        ],
    )
    adapter = MagicMock()
    adapter.should_use_team.return_value = True
    adapter.execute_step = AsyncMock(
        side_effect=[
            StepResult(
                success=True,
                output="Read Cargo.toml and clients/rust/Cargo.toml; found 2 Rust workspaces.",
                tool_calls_used=2,
            ),
            StepResult(
                success=True,
                output="Reviewed src/lib.rs:10 and clients/rust/src/lib.rs:4 for Arc usage.",
                tool_calls_used=3,
            ),
        ]
    )

    with (
        patch(
            "victor.agent.planning.team_execution.PlanningTeamExecutionAdapter",
            return_value=adapter,
        ),
        patch("victor.agent.planning.autonomous.AutonomousPlanner") as planner_cls,
    ):
        result = await service._execute_plan(plan, user_approved=True)

    planner_cls.assert_not_called()
    assert adapter.execute_step.await_count == 2
    first_call = adapter.execute_step.await_args_list[0].kwargs
    assert first_call["root_session_id"] == "session_root"
    assert first_call["step"].id == "1"
    assert result.success is True
    assert result.steps_completed == 2
    assert result.total_tool_calls == 5
    assert "Cargo.toml" in result.final_output
    execution_state = result.metadata["plan_execution_state"]
    assert execution_state["plan_id"] == result.plan_id
    assert execution_state["execution_mode"] == "team_adapter"
    assert execution_state["success"] is True
    assert execution_state["step_statuses"] == {"1": "completed", "2": "completed"}
    assert execution_state["completed_step_ids"] == ["1", "2"]
    assert execution_state["failed_step_ids"] == []


@pytest.mark.asyncio
async def test_planning_compaction_prefers_context_service_before_legacy_compactor():
    context_service = SimpleNamespace(
        get_compaction_recommendation=MagicMock(return_value={"should_compact": True}),
        compact_context=AsyncMock(return_value=2),
    )
    legacy_compactor = MagicMock()
    orchestrator = SimpleNamespace(
        _context_service=context_service,
        settings=SimpleNamespace(context_compaction_strategy="semantic"),
        tool_calls_used=0,
        conversation=SimpleNamespace(get_latest_user_message=MagicMock(return_value="review plan")),
        has_capability=MagicMock(return_value=True),
        get_capability_value=MagicMock(return_value=legacy_compactor),
    )
    service = PlanningRuntimeService(orchestrator)

    await service._compact_context_if_needed()

    context_service.get_compaction_recommendation.assert_called_once()
    context_service.compact_context.assert_awaited_once_with(
        strategy="semantic",
        min_messages=6,
    )
    legacy_compactor.check_and_compact.assert_not_called()


@pytest.mark.asyncio
async def test_chat_with_planning_attaches_execution_state_to_response_metadata():
    service = PlanningRuntimeService(
        SimpleNamespace(),
        config=PlanningConfig(show_plan_before_execution=False),
    )
    plan = ReadableTaskPlan(
        name="Graph Native Plan",
        complexity=TaskComplexity.COMPLEX,
        desc="Exercise plan execution state metadata",
        steps=[["1", "research", "Map files", "read"]],
    )
    execution_state = {
        "plan_id": "plan-1",
        "execution_mode": "team_adapter",
        "success": True,
        "step_statuses": {"1": "completed"},
    }
    result = PlanResult(
        plan_id="plan-1",
        success=True,
        total_steps=1,
        steps_completed=1,
        metadata={"plan_execution_state": execution_state},
    )
    response = CompletionResponse(
        content="summary",
        role="assistant",
        metadata={"provider": "test-provider"},
    )
    service._compact_context_if_needed = AsyncMock()
    service._generate_plan = AsyncMock(return_value=plan)
    service._execute_plan = AsyncMock(return_value=result)
    service._generate_final_response = AsyncMock(return_value=response)

    returned = await service.chat_with_planning("plan this", mode=PlanningMode.ALWAYS)

    assert returned is response
    assert returned.metadata["provider"] == "test-provider"
    assert returned.metadata["plan_execution_state"] == execution_state
    assert returned.metadata["planning"]["mode"] == "planned"
    assert returned.metadata["planning"]["plan_name"] == "Graph Native Plan"
    assert returned.metadata["planning"]["steps_completed"] == 1
    assert returned.metadata["planning"]["steps_total"] == 1


@pytest.mark.asyncio
async def test_team_plan_execution_bounds_independent_step_concurrency():
    orchestrator = SimpleNamespace(active_session_id="session_root")
    service = PlanningRuntimeService(
        orchestrator,
        config=PlanningConfig(max_parallel_steps=2),
    )
    plan = ReadableTaskPlan(
        name="Large Review",
        complexity=TaskComplexity.COMPLEX,
        desc="Review many independent areas",
        steps=[
            ["1", "research", "Map workspace A", "read"],
            ["2", "research", "Map workspace B", "read"],
            ["3", "research", "Map workspace C", "read"],
            ["4", "research", "Map workspace D", "read"],
        ],
    )
    adapter = MagicMock()
    active = 0
    max_active = 0

    async def execute_step(**kwargs):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return StepResult(
            success=True,
            output=f"Read workspace_{kwargs['step'].id}/Cargo.toml and found 3 files.",
            tool_calls_used=1,
        )

    adapter.execute_step = AsyncMock(side_effect=execute_step)

    result = await service._execute_plan_via_team_adapter(plan, adapter)

    assert result.success is True
    assert result.steps_completed == 4
    assert adapter.execute_step.await_count == 4
    assert max_active == 2


@pytest.mark.asyncio
async def test_team_plan_stops_when_read_heavy_step_lacks_evidence():
    orchestrator = SimpleNamespace(active_session_id="session_root")
    service = PlanningRuntimeService(orchestrator)
    plan = ReadableTaskPlan(
        name="Rust Review",
        complexity=TaskComplexity.COMPLEX,
        desc="Review Rust source after inventory",
        steps=[
            ["1", "analyze", "Enumerate all Rust source files", "shell,read"],
            ["2", "review", "Review Arc usage in inventoried files", "grep,read", [1]],
        ],
    )
    adapter = MagicMock()
    adapter.execute_step = AsyncMock(
        side_effect=[
            StepResult(success=True, output="inventory complete", tool_calls_used=1),
            StepResult(
                success=True,
                output="Reviewed src/lib.rs:12 and src/storage/mod.rs:44 for Arc usage.",
                tool_calls_used=2,
            ),
        ]
    )

    result = await service._execute_plan_via_team_adapter(plan, adapter)

    assert result.success is False
    assert result.steps_completed == 0
    assert result.steps_failed == 1
    assert adapter.execute_step.await_count == 1
    assert [call.kwargs["step"].id for call in adapter.execute_step.await_args_list] == [
        "1",
    ]
    assert "Insufficient execution evidence" in result.error_message
    execution_state = result.metadata["plan_execution_state"]
    assert execution_state["execution_mode"] == "team_adapter"
    assert execution_state["success"] is False
    assert execution_state["step_statuses"]["1"] == "failed"
    assert execution_state["step_statuses"]["2"] == "skipped"
    assert execution_state["failed_step_ids"] == ["1"]
    assert execution_state["skipped_step_ids"] == ["2"]


@pytest.mark.asyncio
async def test_team_plan_accepts_concrete_read_heavy_evidence():
    orchestrator = SimpleNamespace(active_session_id="session_root")
    service = PlanningRuntimeService(orchestrator)
    plan = ReadableTaskPlan(
        name="Rust Review",
        complexity=TaskComplexity.COMPLEX,
        desc="Review Rust source after inventory",
        steps=[
            ["1", "analyze", "Enumerate all Rust source files", "shell,read"],
            ["2", "review", "Review Arc usage in inventoried files", "grep,read", [1]],
        ],
    )
    adapter = MagicMock()
    adapter.execute_step = AsyncMock(
        side_effect=[
            StepResult(
                success=True,
                output="Enumerated 192 Rust files, including src/lib.rs and src/storage/mod.rs.",
                tool_calls_used=1,
            ),
            StepResult(
                success=True,
                output="Reviewed src/lib.rs:12 and src/storage/mod.rs:44 for Arc usage.",
                tool_calls_used=2,
            ),
        ]
    )

    result = await service._execute_plan_via_team_adapter(plan, adapter)

    assert result.success is True
    assert result.steps_completed == 2
    assert result.steps_failed == 0
    assert adapter.execute_step.await_count == 2
    assert "192 Rust files" in result.final_output


@pytest.mark.asyncio
async def test_team_plan_accepts_concrete_file_target_from_step_description():
    orchestrator = SimpleNamespace(active_session_id="session_root")
    service = PlanningRuntimeService(orchestrator)
    plan = ReadableTaskPlan(
        name="Rust Review",
        complexity=TaskComplexity.COMPLEX,
        desc="Review Rust workspace",
        steps=[
            ["1", "analyze", "Read rust/Cargo.toml to map workspace members", "read"],
            ["2", "review", "Review Arc usage in mapped crates", "grep,read", [1]],
        ],
    )
    adapter = MagicMock()
    adapter.execute_step = AsyncMock(
        side_effect=[
            StepResult(success=True, output="Workspace manifest inspected.", tool_calls_used=1),
            StepResult(
                success=True,
                output="Reviewed rust/crates/core/src/lib.rs:12 for Arc usage.",
                tool_calls_used=2,
            ),
        ]
    )

    result = await service._execute_plan_via_team_adapter(plan, adapter)

    assert result.success is True
    assert result.steps_completed == 2
    validation = result.step_results["1"].metadata["evidence_validation"]
    assert validation["passed"] is True
    assert validation["has_file_reference"] is True


@pytest.mark.asyncio
async def test_team_plan_accepts_directory_tree_mapping_scope():
    orchestrator = SimpleNamespace(active_session_id="session_root")
    service = PlanningRuntimeService(orchestrator)
    plan = ReadableTaskPlan(
        name="Rust Review",
        complexity=TaskComplexity.COMPLEX,
        desc="Review Rust workspace",
        steps=[
            [
                "1",
                "research",
                "Map the full directory tree of rust/ to identify src/, benches/, tests/",
                "shell,read",
            ],
        ],
    )
    adapter = MagicMock()
    adapter.execute_step = AsyncMock(
        return_value=StepResult(
            success=True,
            output="Directory tree mapped.",
            tool_calls_used=1,
        )
    )

    result = await service._execute_plan_via_team_adapter(plan, adapter)

    assert result.success is True
    validation = result.step_results["1"].metadata["evidence_validation"]
    assert validation["passed"] is True
    assert validation["has_directory_scope"] is True


def test_read_only_plan_does_not_require_execution_approval():
    service = PlanningRuntimeService(SimpleNamespace())
    plan = ReadableTaskPlan(
        name="Read-only Review",
        complexity=TaskComplexity.COMPLEX,
        desc="Review Rust code without changing files",
        steps=[
            ["1", "analyze", "Read Cargo.toml", "read"],
            ["2", "review", "Search Arc usages", "grep,code_search,overview"],
            ["3", "doc", "Summarize findings in chat", "read"],
        ],
    )

    assert service._plan_requires_execution_approval(plan) is False


def test_checklist_planning_step_does_not_require_read_evidence_contract():
    service = PlanningRuntimeService(SimpleNamespace())
    plan = ReadableTaskPlan(
        name="Rust Checklist",
        complexity=TaskComplexity.COMPLEX,
        desc="Create Rust review checklist",
        steps=[
            [
                "1",
                "doc",
                "Build master Rust best practices checklist for Arc and immutability",
                "write",
            ],
            ["2", "review", "Present checklist to user for approval", "read", [1]],
        ],
    )
    execution_plan = plan.to_execution_plan()

    assert service._step_requires_evidence_contract(execution_plan.steps[0]) is False
    assert service._step_requires_evidence_contract(execution_plan.steps[1]) is False


def test_summary_prompt_surfaces_evidence_validation_failure():
    service = PlanningRuntimeService(SimpleNamespace())
    plan = ReadableTaskPlan(
        name="Rust Evidence Audit",
        complexity=TaskComplexity.COMPLEX,
        desc="Audit Rust evidence quality",
        steps=[
            ["1", "analyze", "Enumerate Rust files", "shell,read"],
            ["2", "review", "Review Arc usage", "grep,read", [1]],
        ],
    )
    result = SimpleNamespace(
        steps_completed=0,
        total_steps=2,
        success=False,
        final_output="",
        error_message="Insufficient execution evidence for step 1",
        step_results={
            "1": StepResult(
                success=False,
                output="inventory complete",
                error=(
                    "Insufficient execution evidence for step 1: "
                    "step output is only a generic completion marker"
                ),
                tool_calls_used=1,
                metadata={
                    "evidence_validation": {
                        "passed": False,
                        "reason": "step output is only a generic completion marker",
                        "tool_calls_used": 1,
                        "has_file_reference": False,
                        "has_counted_scope": False,
                        "has_artifacts": False,
                        "source_count": 0,
                    }
                },
            )
        },
    )

    prompt = service._build_summary_prompt(plan, result)

    assert "Evidence validation: failed" in prompt
    assert "step output is only a generic completion marker" in prompt
    assert "tool_calls=1" in prompt
    assert "file_ref=False" in prompt
    assert "Do not report failed evidence-validation steps as completed" in prompt


def test_summary_prompt_reports_aggregate_evidence_validation_counts():
    service = PlanningRuntimeService(SimpleNamespace())
    plan = ReadableTaskPlan(
        name="Rust Evidence Coverage",
        complexity=TaskComplexity.COMPLEX,
        desc="Audit which completed steps have evidence backing",
        steps=[
            ["1", "analyze", "Map Rust workspaces", "read"],
            ["2", "review", "Review Arc usage", "grep,read", [1]],
            ["3", "doc", "Summarize findings", "read", [2]],
            ["4", "review", "Review allocation patterns", "grep,read", [1]],
        ],
    )
    result = SimpleNamespace(
        steps_completed=3,
        total_steps=4,
        success=False,
        final_output="",
        error_message="",
        step_results={
            "1": StepResult(
                success=True,
                output="Read Cargo.toml and clients/rust/Cargo.toml.",
                tool_calls_used=2,
                metadata={
                    "evidence_validation": {
                        "passed": True,
                        "reason": "concrete file or scope evidence found",
                    }
                },
            ),
            "2": StepResult(
                success=True,
                output="Reviewed Arc usage.",
                tool_calls_used=2,
            ),
            "3": StepResult(
                success=False,
                output="report complete",
                error="Insufficient execution evidence",
                tool_calls_used=1,
                metadata={
                    "evidence_validation": {
                        "passed": False,
                        "reason": "step output is only a generic completion marker",
                    }
                },
            ),
        },
    )

    prompt = service._build_summary_prompt(plan, result)

    assert ("Evidence validation summary: passed=1; failed=1; " "missing=1; not_run=1") in prompt
    assert "Treat steps missing evidence validation as unverified" in prompt


def test_summary_prompt_surfaces_provider_retry_diagnostics():
    service = PlanningRuntimeService(SimpleNamespace())
    plan = ReadableTaskPlan(
        name="Rust Retry Audit",
        complexity=TaskComplexity.COMPLEX,
        desc="Audit Rust execution evidence and provider stability",
        steps=[
            ["1", "analyze", "Map Rust workspaces", "read"],
            ["2", "review", "Review Arc usage", "grep,read", [1]],
        ],
    )
    result = SimpleNamespace(
        steps_completed=1,
        total_steps=2,
        success=False,
        final_output="Read root Cargo.toml and clients/rust/Cargo.toml.",
        error_message="Provider recovered after transient disconnects",
        metadata={
            "provider_retry_diagnostics": [
                {
                    "provider": "zai",
                    "model": "glm-5.1",
                    "retry_count": 3,
                    "last_error": "Server disconnected without sending a response.",
                }
            ]
        },
        step_results={
            "1": StepResult(
                success=True,
                output="Read root Cargo.toml and clients/rust/Cargo.toml.",
                tool_calls_used=2,
                metadata={
                    "provider_retry_diagnostics": {
                        "provider": "zai",
                        "model": "glm-5.1",
                        "retry_count": 1,
                        "last_error": "Server disconnected without sending a response.",
                    }
                },
            )
        },
    )

    prompt = service._build_summary_prompt(plan, result)

    assert "Provider retry diagnostics:" in prompt
    assert "provider=zai" in prompt
    assert "model=glm-5.1" in prompt
    assert "retry_count=3" in prompt
    assert "retry_count=1" in prompt
    assert "Server disconnected without sending a response" in prompt
    assert "Mention provider retry diagnostics separately from repository findings" in prompt


def test_shell_plan_requires_execution_approval():
    service = PlanningRuntimeService(SimpleNamespace())
    plan = ReadableTaskPlan(
        name="Inventory",
        complexity=TaskComplexity.COMPLEX,
        desc="Inventory Rust files using a command",
        steps=[
            ["1", "analyze", "Enumerate Rust files", "shell"],
        ],
    )

    assert service._plan_requires_execution_approval(plan) is True


def test_write_plan_requires_execution_approval():
    service = PlanningRuntimeService(SimpleNamespace())
    plan = ReadableTaskPlan(
        name="Write Report",
        complexity=TaskComplexity.MODERATE,
        desc="Write findings to disk",
        steps=[
            ["1", "doc", "Write findings report", "write"],
        ],
    )

    assert service._plan_requires_execution_approval(plan) is True


@pytest.mark.asyncio
async def test_read_only_plan_decision_uses_not_required_approval_state():
    service = PlanningRuntimeService(SimpleNamespace())
    plan = ReadableTaskPlan(
        name="Read-only Review",
        complexity=TaskComplexity.COMPLEX,
        desc="Review without changing files",
        steps=[["1", "review", "Read docs", "read"]],
    )

    with patch.object(service, "_save_plan_to_disk"):
        decision = await service._show_plan_with_console(plan)

    assert decision.proceed is True
    assert decision.user_approved_execution is False
    assert decision.approval_state is ApprovalState.NOT_REQUIRED


@pytest.mark.asyncio
async def test_effectful_plan_decision_uses_approved_approval_state():
    service = PlanningRuntimeService(SimpleNamespace())
    plan = ReadableTaskPlan(
        name="Effectful Plan",
        complexity=TaskComplexity.MODERATE,
        desc="Run tests",
        steps=[["1", "testing", "Run tests", "shell"]],
    )

    with (
        patch.object(service, "_save_plan_to_disk"),
        patch("sys.stdin.isatty", return_value=True),
        patch("asyncio.to_thread", new=AsyncMock(return_value=True)),
    ):
        decision = await service._show_plan_with_console(plan)

    assert decision.proceed is True
    assert decision.user_approved_execution is True
    assert decision.approval_state is ApprovalState.APPROVED


@pytest.mark.asyncio
async def test_effectful_plan_decision_uses_rejected_approval_state():
    service = PlanningRuntimeService(SimpleNamespace())
    plan = ReadableTaskPlan(
        name="Rejected Plan",
        complexity=TaskComplexity.MODERATE,
        desc="Run tests",
        steps=[["1", "testing", "Run tests", "shell"]],
    )

    with (
        patch.object(service, "_save_plan_to_disk"),
        patch("sys.stdin.isatty", return_value=True),
        patch("asyncio.to_thread", new=AsyncMock(return_value=False)),
    ):
        decision = await service._show_plan_with_console(plan)

    assert decision.proceed is False
    assert decision.user_approved_execution is False
    assert decision.approval_state is ApprovalState.REJECTED


@pytest.mark.asyncio
async def test_plan_approval_prompt_explains_enter_and_reject_options():
    service = PlanningRuntimeService(SimpleNamespace())
    plan = ReadableTaskPlan(
        name="Effectful Plan",
        complexity=TaskComplexity.MODERATE,
        desc="Run tests",
        steps=[["1", "testing", "Run tests", "shell"]],
    )
    console = MagicMock()

    with (
        patch("sys.stdin.isatty", return_value=True),
        patch("asyncio.to_thread", new=AsyncMock(return_value=True)),
    ):
        approved = await service._request_plan_approval(plan, console)

    assert approved is True
    rendered = "\n".join(str(call.args[0]) for call in console.print.call_args_list if call.args)
    assert "Press Enter to execute" in rendered
    assert "type y then Enter" in rendered
    assert "Type n then Enter to reject" in rendered


# ---------------------------------------------------------------------------
# Plan-state: _extract_list_from_output
# ---------------------------------------------------------------------------


class TestExtractListFromOutput:
    svc = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))

    def test_bullet_list(self) -> None:
        out = "- core\n- util\n- cli"
        items = self.svc._extract_list_from_output(out)
        assert items == ["core", "util", "cli"]

    def test_numbered_list(self) -> None:
        out = "1. core\n2. util\n3. cli"
        items = self.svc._extract_list_from_output(out)
        assert items == ["core", "util", "cli"]

    def test_plain_newlines(self) -> None:
        out = "core\nutil\ncli"
        items = self.svc._extract_list_from_output(out)
        assert items == ["core", "util", "cli"]

    def test_empty_string_returns_empty(self) -> None:
        assert self.svc._extract_list_from_output("") == []

    def test_single_line_falls_back_to_whole_string(self) -> None:
        out = "Just one item without newlines"
        items = self.svc._extract_list_from_output(out)
        assert items == ["Just one item without newlines"]

    def test_filters_long_lines(self) -> None:
        long_line = "x" * 201
        out = f"core\n{long_line}\nutil"
        items = self.svc._extract_list_from_output(out)
        assert "core" in items
        assert "util" in items
        assert long_line not in items

    def test_asterisk_bullets(self) -> None:
        out = "* alpha\n* beta"
        items = self.svc._extract_list_from_output(out)
        assert items == ["alpha", "beta"]

    def test_mixed_bullet_styles(self) -> None:
        out = "• one\n- two\n* three"
        items = self.svc._extract_list_from_output(out)
        assert items == ["one", "two", "three"]


# ---------------------------------------------------------------------------
# Plan-state: _skip_specific_steps
# ---------------------------------------------------------------------------


class TestSkipSpecificSteps:
    from victor.agent.planning.base import ExecutionPlan, PlanStep, StepStatus, StepType

    @staticmethod
    def _make_exec_plan(*step_ids: str) -> "ExecutionPlan":
        from victor.agent.planning.base import ExecutionPlan, PlanStep, StepStatus, StepType

        steps = [
            PlanStep(
                id=sid,
                description=f"step {sid}",
                step_type=StepType.RESEARCH,
                status=StepStatus.PENDING,
            )
            for sid in step_ids
        ]
        return ExecutionPlan(id="plan-1", goal="test", steps=steps)

    def test_marks_pending_steps_as_skipped(self) -> None:
        from victor.agent.planning.base import StepStatus

        plan = self._make_exec_plan("1", "2", "3")
        PlanningRuntimeService._skip_specific_steps(plan, ["2", "3"])
        statuses = {s.id: s.status for s in plan.steps}
        assert statuses["1"] == StepStatus.PENDING
        assert statuses["2"] == StepStatus.SKIPPED
        assert statuses["3"] == StepStatus.SKIPPED

    def test_does_not_skip_non_pending_steps(self) -> None:
        from victor.agent.planning.base import StepStatus

        plan = self._make_exec_plan("1", "2")
        plan.steps[1].status = StepStatus.COMPLETED
        PlanningRuntimeService._skip_specific_steps(plan, ["2"])
        assert plan.steps[1].status == StepStatus.COMPLETED

    def test_unknown_step_id_is_ignored(self) -> None:
        plan = self._make_exec_plan("1", "2")
        PlanningRuntimeService._skip_specific_steps(plan, ["99"])
        from victor.agent.planning.base import StepStatus
        assert all(s.status == StepStatus.PENDING for s in plan.steps)

    def test_empty_skip_list_is_noop(self) -> None:
        plan = self._make_exec_plan("1", "2")
        PlanningRuntimeService._skip_specific_steps(plan, [])
        from victor.agent.planning.base import StepStatus
        assert all(s.status == StepStatus.PENDING for s in plan.steps)

    def test_step_ids_coerced_from_strings(self) -> None:
        from victor.agent.planning.base import StepStatus

        plan = self._make_exec_plan("3", "4")
        PlanningRuntimeService._skip_specific_steps(plan, ["3"])
        assert plan.steps[0].status == StepStatus.SKIPPED


# ---------------------------------------------------------------------------
# Plan-state integration: produces + skip_step_ids in execution loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_state_produces_key_is_stored_after_step():
    """tool/compute steps with 'produces' write their output list to plan_state."""
    from victor.agent.planning.base import StepResult
    from victor.agent.planning.team_execution import PlanningTeamExecutionAdapter

    svc = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))

    captured_plan_state: dict = {}

    async def _fake_execute_step(**kwargs) -> StepResult:
        # Capture the plan_state as received by step 2 (after step 1 produced it)
        if kwargs.get("plan_state"):
            captured_plan_state.update(kwargs["plan_state"])
        step = kwargs["step"]
        if step.id == "1":
            return StepResult(
                success=True,
                output="- core\n- util\n- cli",
                tool_calls_used=1,
                metadata={},
            )
        return StepResult(success=True, output="done", tool_calls_used=0, metadata={})

    plan = ReadableTaskPlan(
        name="State test",
        complexity=TaskComplexity.COMPLEX,
        desc="Test produces propagation",
        steps=[
            {
                "id": "1",
                "type": "analyze",
                "desc": "Discover members",
                "exec": "tool",
                "produces": "workspace_members",
            },
            {
                "id": "2",
                "type": "feature",
                "desc": "Loop over members",
                "exec": "loop",
                "loop_over": "workspace_members",
                "deps": ["1"],
            },
        ],
    )

    mock_adapter = MagicMock(spec=PlanningTeamExecutionAdapter)
    mock_adapter.execute_step = AsyncMock(side_effect=_fake_execute_step)

    # Bypass evidence contract so the test focuses on plan_state flow only.
    with patch.object(svc, "_apply_step_evidence_contract", side_effect=lambda step, r: r):
        await svc._execute_plan_via_team_adapter(plan, mock_adapter)

    # Step 2 should have received workspace_members extracted from step 1's output
    assert "workspace_members" in captured_plan_state
    assert captured_plan_state["workspace_members"] == ["core", "util", "cli"]


@pytest.mark.asyncio
async def test_plan_state_skip_step_ids_marks_branch_skipped():
    """When a conditional step returns skip_step_ids, downstream steps are marked SKIPPED."""
    from victor.agent.planning.base import StepResult, StepStatus
    from victor.agent.planning.team_execution import PlanningTeamExecutionAdapter

    svc = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))

    async def _fake_execute_step(**kwargs) -> StepResult:
        step = kwargs["step"]
        if step.id == "cond":
            return StepResult(
                success=True,
                output="condition evaluated",
                tool_calls_used=0,
                metadata={"skip_step_ids": ["3b"]},
            )
        return StepResult(success=True, output="done", tool_calls_used=0, metadata={})

    plan = ReadableTaskPlan(
        name="Conditional test",
        complexity=TaskComplexity.COMPLEX,
        desc="Branch routing",
        steps=[
            {"id": "cond", "type": "analyze", "desc": "Check workspace", "exec": "conditional"},
            {"id": "3a", "type": "feature", "desc": "Loop path", "exec": "loop", "deps": ["cond"]},
            {"id": "3b", "type": "feature", "desc": "Single path", "exec": "agent", "deps": ["cond"]},
        ],
    )

    mock_adapter = MagicMock(spec=PlanningTeamExecutionAdapter)
    mock_adapter.execute_step = AsyncMock(side_effect=_fake_execute_step)

    # Bypass evidence contract to focus on branch routing logic.
    with patch.object(svc, "_apply_step_evidence_contract", side_effect=lambda step, r: r):
        result = await svc._execute_plan_via_team_adapter(plan, mock_adapter)

    # cond and 3a completed; 3b was SKIPPED (not failed) so steps_completed == 2.
    assert result.steps_completed == 2
    assert result.steps_failed == 0
