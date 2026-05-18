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


def test_unpack_step_handles_list_format():
    svc = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))
    step = ["1", "analyze", "Discover workspace", "read,grep"]
    sid, stype, sdesc, stools = svc._unpack_step(step)
    assert sid == "1"
    assert stype == "analyze"
    assert sdesc == "Discover workspace"
    assert stools == "read,grep"


def test_unpack_step_handles_dict_format():
    svc = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))
    step = {"id": "2", "type": "feature", "desc": "Create checklist", "tools": ["write", "read"]}
    sid, stype, sdesc, stools = svc._unpack_step(step)
    assert sid == "2"
    assert stype == "feature"
    assert sdesc == "Create checklist"
    assert stools == ["write", "read"]


def test_unpack_step_dict_uses_description_fallback():
    svc = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))
    step = {"id": "3", "type": "compute", "description": "Run check"}
    _, _, sdesc, _ = svc._unpack_step(step)
    assert sdesc == "Run check"


def test_plan_requires_execution_approval_works_with_dict_steps():
    """Rich dict steps must not crash _plan_requires_execution_approval."""
    svc = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))
    plan = ReadableTaskPlan(
        name="Mixed plan",
        complexity=TaskComplexity.COMPLEX,
        desc="Test dict step approval check",
        steps=[
            {"id": "1", "type": "analyze", "desc": "Read workspace", "tools": ["read"]},
            {"id": "2", "type": "deployment", "desc": "Deploy", "tools": ["shell"]},
        ],
    )
    result = svc._plan_requires_execution_approval(plan)
    assert result is True


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

    assert (
        "Evidence validation summary: passed=1; failed=1; exempt=0; missing=1; not_run=1"
    ) in prompt
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

    def test_none_sentinel_returns_empty(self) -> None:
        """Regression: '(none)' sentinel must yield [] not ['(none)'].

        Bug: agents instructed to output '(none)' when they found nothing would
        produce a one-item list ['(none)'], which the prose-guard sometimes let
        through, causing conditional nodes to see a non-empty list and take the
        wrong branch.
        Fix: _extract_list_from_output now checks for the exact sentinel before
        any other extraction so it always returns [].
        """
        assert self.svc._extract_list_from_output("(none)") == []

    def test_none_sentinel_with_surrounding_whitespace_returns_empty(self) -> None:
        """(none) sentinel with leading/trailing whitespace is still recognised."""
        assert self.svc._extract_list_from_output("  (none)  ") == []

    def test_prose_with_embedded_paths_returns_tokens(self) -> None:
        """Prose narration that embeds slash-path tokens should yield those paths.

        GLM-5.x models sometimes output 'Now let me examine crates/state and
        crates/tools...' instead of a plain list. The secondary embedded-token scan
        should rescue the actual crate paths.
        """
        out = "Now let me examine crates/state and crates/tools to review the workspace."
        items = self.svc._extract_list_from_output(out)
        assert "crates/state" in items
        assert "crates/tools" in items

    def test_prose_without_embedded_paths_returns_empty(self) -> None:
        """Pure narration prose with no path-like tokens returns []."""
        out = "Now let me begin reading the files in the workspace to understand the structure."
        items = self.svc._extract_list_from_output(out)
        assert items == []

    def test_prose_with_embedded_snake_case_returns_tokens(self) -> None:
        """Snake_case identifiers embedded in prose are extracted as fallback items."""
        out = "I will now examine workspace_members and edge_runtime to understand the setup."
        items = self.svc._extract_list_from_output(out)
        assert "workspace_members" in items
        assert "edge_runtime" in items

    def test_xml_thinking_blocks_are_stripped(self) -> None:
        """<thinking>...</thinking> blocks from reasoning models must not appear as items.

        GLM-5.x emits <thinking> XML blocks in content.  Without stripping,
        '</thinking>' and '<tool_call >' surface as list items.
        """
        out = (
            "<thinking>\nLet me read the source files.\n</thinking>\n"
            "rust/crates/state\n"
            "rust/crates/tools\n"
            "<tool_call >\n<tool_name >read</tool_name >\n</tool_call >"
        )
        items = self.svc._extract_list_from_output(out)
        assert "rust/crates/state" in items
        assert "rust/crates/tools" in items
        # XML tags must not appear as items
        assert not any("<" in item or ">" in item for item in items)
        assert not any("thinking" in item for item in items)

    def test_deterministic_summary_prose_does_not_include_hyphenated_noise(self) -> None:
        """'Deterministic read-only execution...' prose must not yield 'read-only' as a token.

        When a step's only surviving line is the deterministic execution summary,
        the secondary path-token scan should return slash-paths extracted from the
        full output — not hyphenated adjectives like 'read-only'.
        """
        # Simulate: the sole line after extraction is the execution summary;
        # the full output also contains actual crate paths in thinking content.
        out = (
            "Deterministic read-only execution completed for shell: 1 succeeded, 0 failed.\n"
            "rust/crates/edge-runtime/src/agent\n"
            "rust/crates/edge-runtime/src/lib"
        )
        items = self.svc._extract_list_from_output(out)
        # Path tokens must be present
        assert any("/" in item for item in items)
        # 'read-only' must not appear (no "/" so filtered out)
        assert "read-only" not in items

    def test_xml_only_output_returns_empty(self) -> None:
        """Output consisting entirely of XML tags after stripping returns []."""
        out = "<thinking>\nsome internal reasoning\n</thinking>"
        items = self.svc._extract_list_from_output(out)
        assert items == []

    def test_truncation_message_filtered_from_list(self) -> None:
        """Lines containing file-truncation narration must not appear as list items.

        When sub-agents read large files, the tool emits 'The file was truncated.'
        and the model echoes 'Let me read the middle portion...'.  These lines
        contaminated per_crate_findings with prose instead of crate paths.
        """
        out = (
            "[rust/crates/protocol]\n"
            "The file was truncated. Let me read the middle portion...\n"
            "[rust/crates/state]\n"
            "Let me read the next section to get more context.\n"
            "[rust/crates/tools]\n"
        )
        items = self.svc._extract_list_from_output(out)
        # Real crate references must be present
        assert "rust/crates/protocol" in items or any("protocol" in i for i in items)
        assert "rust/crates/state" in items or any("state" in i for i in items)
        assert "rust/crates/tools" in items or any("tools" in i for i in items)
        # Truncation narration must be absent
        assert not any("truncated" in i.lower() for i in items), f"truncation phrase in items: {items}"
        assert not any("let me read" in i.lower() for i in items), f"continuation phrase in items: {items}"

    def test_firstresponder_halted_line_filtered(self) -> None:
        """'FirstResponderTool halted, entering general response mode.' must be filtered."""
        out = "FirstResponderTool halted, entering general response mode.\ncore\nutil"
        items = self.svc._extract_list_from_output(out)
        assert "core" in items
        assert "util" in items
        assert not any("halted" in i.lower() for i in items), f"halted phrase in items: {items}"

    def test_let_me_examine_filtered(self) -> None:
        """'Let me examine...' continuation lines must not appear as list items."""
        out = "rust/crates/protocol\nLet me examine the remaining files.\nrust/crates/state"
        items = self.svc._extract_list_from_output(out)
        assert not any("let me examine" in i.lower() for i in items)

    def test_clean_list_unaffected_by_filter(self) -> None:
        """A clean crate list without any continuation lines must pass through unchanged."""
        out = "rust/crates/protocol\nrust/crates/state\nrust/crates/tools"
        items = self.svc._extract_list_from_output(out)
        assert items == ["rust/crates/protocol", "rust/crates/state", "rust/crates/tools"]

    def test_delta_symbol_items_stripped_from_output(self) -> None:
        """Items containing the ∂ character (U+2202) must be stripped from the extracted list.

        Regression: ZAI/GLM-5.1 uses ∂ as tool-call field separators in its output.
        Lines like 'read∂', '/path∂rust/crates/edge-runtime/src/agent.rs∂',
        '<_Tool_Dependency>∂' were leaking into per_crate_findings and cross_crate_findings,
        polluting plan_state with model-internal formatting artifacts.
        """
        # Simulate the output format GLM-5.1 produces when it interleaves tool calls
        out = (
            "rust/crates/protocol\n"
            "rust/crates/state\n"
            "Let me explore the edge-runtime submodules. ∂\n"  # ∂ suffix
            "∂\n"  # bare ∂ line
            "<_Tool_Dependency>∂\n"
            "read∂\n"
            "/path∂rust/crates/edge-runtime/src/agent.rs∂\n"
            "rust/crates/tools\n"
        )
        items = self.svc._extract_list_from_output(out)
        # Real crate paths must survive
        assert "rust/crates/protocol" in items
        assert "rust/crates/state" in items
        assert "rust/crates/tools" in items
        # ∂-bearing items must be removed
        assert not any("∂" in i for i in items), (
            f"Items containing ∂ must be filtered; got: {items}"
        )

    def test_actually_let_me_reread_continuation_filtered(self) -> None:
        """'Actually, let me re-read...' continuation lines must not appear as list items.

        Regression: the per_crate_findings list for step 7a contained
        'Actually, let me re-read specifying lines to get the truncated portion:'
        because _CONTINUATION_RE matched 'let me read' but not 'let me re-read'
        or sentences starting with 'Actually, let me'.
        """
        out = (
            "rust/crates/protocol\n"
            "Actually, let me re-read specifying lines to get the truncated portion:\n"
            "rust/crates/state\n"
            "Actually, let me look at this more carefully.\n"
            "rust/crates/tools\n"
        )
        items = self.svc._extract_list_from_output(out)
        # Real crate paths must survive
        assert "rust/crates/protocol" in items
        assert "rust/crates/state" in items
        assert "rust/crates/tools" in items
        # Continuation lines must be removed
        assert not any("actually" in i.lower() for i in items), (
            f"'Actually...' continuation lines must be filtered; got: {items}"
        )
        assert not any("re-read" in i.lower() for i in items), (
            f"'let me re-read' lines must be filtered; got: {items}"
        )


# ---------------------------------------------------------------------------
# Evidence contract: _is_directory_listing_only + _assess_step_evidence
# ---------------------------------------------------------------------------


class TestIsDirectoryListingOnly:
    svc = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))

    def test_pure_ls_output_is_listing(self) -> None:
        ls_out = "src/main.rs\nsrc/lib.rs\nCargo.toml\nREADME.md"
        assert self.svc._is_directory_listing_only(ls_out) is True

    def test_source_file_content_is_not_listing(self) -> None:
        code = 'fn main() {\n    println!("hello");\n}\n'
        assert self.svc._is_directory_listing_only(code) is False

    def test_empty_string_is_not_listing(self) -> None:
        assert self.svc._is_directory_listing_only("") is False

    def test_mixed_paths_and_code_is_not_listing(self) -> None:
        # 30% code lines → below 70% threshold
        text = "src/main.rs\n" * 3 + "fn foo() {\n    return 1;\n}\n"
        assert self.svc._is_directory_listing_only(text) is False

    def test_paths_without_extension_below_threshold(self) -> None:
        # Lines without '/' and without extension → below 70% → not a listing
        text = "hello world\nno path here\njust text"
        assert self.svc._is_directory_listing_only(text) is False


class TestAssessStepEvidence:
    svc = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))

    @staticmethod
    def _make_step(desc: str = "Analyze code") -> SimpleNamespace:
        from victor.agent.planning.base import StepType

        return SimpleNamespace(
            id="step1",
            description=desc,
            step_type=StepType.RESEARCH,
            artifacts=[],
            context={},
        )

    @staticmethod
    def _make_result(output: str, tool_calls: int = 1, artifacts=None) -> StepResult:
        return StepResult(
            success=True,
            output=output,
            tool_calls_used=tool_calls,
            artifacts=artifacts or [],
        )

    def test_ls_only_output_fails_evidence_contract(self) -> None:
        step = self._make_step()
        # ls output: extension matches regex but no content
        result = self._make_result("src/main.rs\nsrc/lib.rs\nCargo.toml", tool_calls=1)
        passed, reason, evidence = self.svc._assess_step_evidence(step, result, {})
        assert not passed
        assert evidence["is_directory_listing_only"] is True
        assert evidence["has_file_reference"] is False

    def test_file_read_output_passes_evidence_contract(self) -> None:
        step = self._make_step()
        code = (
            "fn parse_args() -> Args {\n"
            '    let matches = App::new("tool").get_matches();\n'
            '    Args { verbose: matches.is_present("verbose") }\n'
            "}"
        )
        result = self._make_result(code, tool_calls=2)
        passed, reason, evidence = self.svc._assess_step_evidence(step, result, {})
        assert passed
        assert evidence["has_content_tool"] is True

    def test_explicit_content_tool_in_metadata_passes(self) -> None:
        step = self._make_step()
        result = self._make_result("Some output without obvious code patterns", tool_calls=1)
        metadata = {"tool_names_used": ["read", "grep"]}
        passed, reason, evidence = self.svc._assess_step_evidence(step, result, metadata)
        assert evidence["has_content_tool"] is True

    def test_zero_tool_calls_always_fails(self) -> None:
        step = self._make_step()
        result = self._make_result("fn foo() { return 42; }", tool_calls=0)
        passed, reason, evidence = self.svc._assess_step_evidence(step, result, {})
        assert not passed
        assert "no tool-backed" in reason

    def test_write_step_passes_with_short_output(self) -> None:
        """Regression: synthesis/write steps produce short summaries, not file content.

        Bug: step 9 (type=doc, exec=agent, tools=[write]) used all 10 tool budget
        re-reading prior results and produced only 95 chars output — too short for the
        multi-tool analysis threshold (240 chars). Evidence contract failed despite the
        step having completed the write.
        Fix: write-tool steps pass evidence when tool_calls >= 1 and output >= 20 chars.
        """
        from victor.agent.planning.base import StepType

        step = SimpleNamespace(
            id="9",
            description="Synthesize all findings into a prioritized report",
            step_type=StepType.RESEARCH,  # doc maps to RESEARCH
            artifacts=[],
            context={},
            allowed_tools=["write"],
        )
        # 95 chars — realistic short confirmation from a write step
        result = self._make_result(
            "Report written to rust_best_practices_report.md (2847 words).",
            tool_calls=10,
        )
        passed, reason, evidence = self.svc._assess_step_evidence(step, result, {})
        assert passed, f"write step should pass evidence contract; reason={reason}"
        assert evidence["is_write_step"] is True
        assert "write tool" in reason

    def test_non_write_step_still_requires_substantive_output(self) -> None:
        """Non-write steps still need 240+ chars or concrete file/scope evidence."""
        step = self._make_step("Review Arc usage in codebase")
        # 95 chars, no file refs, no code patterns
        result = self._make_result("Analysis complete. Found some patterns.", tool_calls=10)
        passed, reason, evidence = self.svc._assess_step_evidence(step, result, {})
        assert not passed
        assert evidence["is_write_step"] is False

    def test_intent_phrase_with_5_plus_tool_calls_passes_evidence(self) -> None:
        """Regression: ZAI/GLM sub-agent final message is an intent phrase after running
        8 tool calls. The evidence contract must not reject the step in this case
        — 5+ actual tool executions are sufficient evidence of substantive work even when
        the spawn summary is accidentally captured as a planning statement.

        Production failure: step 8 (cross-crate analysis), tools=8, chars=62,
        output="Now let me read the source files to find cross-crate patterns:"
        """
        step = self._make_step("Cross-crate analysis: shared Arc patterns, redundant clones")
        result = self._make_result(
            "Now let me read the source files to find cross-crate patterns:",
            tool_calls=8,
        )
        passed, reason, evidence = self.svc._assess_step_evidence(step, result, {})
        assert passed, (
            f"Step with 8 tool calls must pass even when spawn summary is an intent phrase; "
            f"reason={reason}"
        )

    def test_intent_phrase_with_4_tool_calls_still_fails(self) -> None:
        """With fewer than 5 tool calls, the intent-phrase check applies normally.
        4 tools is below the 'substantive work' threshold.
        """
        step = self._make_step("Cross-crate analysis: shared Arc patterns")
        result = self._make_result(
            "Now let me read the source files to find cross-crate patterns:",
            tool_calls=4,
        )
        passed, reason, evidence = self.svc._assess_step_evidence(step, result, {})
        assert not passed, "4 tool calls with an intent-phrase output should still fail"
        assert "intent" in reason

    def test_intent_phrase_with_few_tools_still_rejected(self) -> None:
        """The existing intent-phrase rejection must fire for 1-2 tool calls
        — this prevents hollow steps from slipping through."""
        step = self._make_step("Analyze Arc usage")
        result = self._make_result("Let me use grep to search for Arc patterns.", tool_calls=2)
        passed, reason, evidence = self.svc._assess_step_evidence(step, result, {})
        assert not passed
        assert "intent" in reason


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
    with patch.object(svc, "_apply_step_evidence_contract", side_effect=lambda step, r, *a, **kw: r):
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
            {
                "id": "3b",
                "type": "feature",
                "desc": "Single path",
                "exec": "agent",
                "deps": ["cond"],
            },
        ],
    )

    mock_adapter = MagicMock(spec=PlanningTeamExecutionAdapter)
    mock_adapter.execute_step = AsyncMock(side_effect=_fake_execute_step)

    # Bypass evidence contract to focus on branch routing logic.
    with patch.object(svc, "_apply_step_evidence_contract", side_effect=lambda step, r, *a, **kw: r):
        result = await svc._execute_plan_via_team_adapter(plan, mock_adapter)

    # cond and 3a completed; 3b was SKIPPED (not failed) so steps_completed == 2.
    assert result.steps_completed == 2
    assert result.steps_failed == 0


def test_toml_content_passes_evidence_contract():
    """Cargo.toml / TOML manifest reads must pass evidence even without code keywords."""
    from victor.agent.planning.base import PlanStep, StepResult, StepType

    svc = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))
    step = PlanStep(
        id="dep_audit",
        step_type=StepType.RESEARCH,
        description="Audit Cargo.toml dependencies for version pinning and feature-flag bloat",
        allowed_tools=["read", "grep"],
    )
    # Output that a real sub-agent would produce from reading Cargo.toml files:
    toml_output = (
        "rust/crates/python-bindings/Cargo.toml\n"
        "[dependencies]\n"
        "serde = { version = \"1.0\", features = [\"derive\"] }\n"
        "once_cell = \"1.19\"  # redundant with std::sync::LazyLock (Rust 1.80+)\n"
        "serde_yaml = \"0.9\"  # deprecated crate — migrate to serde_yml\n"
        "[dev-dependencies]\n"
        "criterion = \"0.5\"\n"
        "edition = \"2021\"\n"
    )
    step_result = StepResult(success=True, output=toml_output, tool_calls_used=2)
    result = svc._apply_step_evidence_contract(step, step_result)
    assert result.success, f"TOML dependency audit should pass evidence: {result.error}"


def test_synthesis_step_not_skipped_when_upstream_fails():
    """Synthesis steps must survive upstream evidence failures and run with partial data."""
    from victor.agent.planning.base import ExecutionPlan, PlanStep, StepStatus, StepType
    from victor.agent.services.planning_runtime import PlanningRuntimeService

    svc = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))

    analysis = PlanStep(
        id="dep_audit",
        step_type=StepType.RESEARCH,
        description="Audit dependencies",
        allowed_tools=["read", "grep"],
    )
    analysis.status = StepStatus.FAILED

    synthesis = PlanStep(
        id="report",
        step_type=StepType.IMPLEMENTATION,
        description="Synthesize all findings into a report",
        allowed_tools=["write"],
        depends_on=["dep_audit"],
    )
    synthesis.status = StepStatus.PENDING

    plan = ExecutionPlan(id="test-plan", goal="audit rust", steps=[analysis, synthesis])
    svc._skip_team_plan_dependents(plan, ["dep_audit"])

    # Synthesis step must NOT be skipped — it should survive and report partial results.
    assert synthesis.status == StepStatus.PENDING, (
        "Synthesis step should not be cascaded to SKIPPED when upstream fails"
    )


@pytest.mark.asyncio
async def test_clarification_fp_rescue_upgrades_failed_step_to_completed():
    """A step FAILED with 'Clarification required' but that produced valid output must be rescued.

    Regression: victor chat -p zai-coding step 5 (Create comprehensive Rust best practices
    checklist) — the sub-agent generated 2830 chars of valid checklist content, but the
    agentic loop decided EvaluationDecision.FAIL because PerceptionIntegration's
    underspecified_target check fired.  The planning engine then skipped all downstream
    steps even though the output WAS produced and the produces key had 4 items.

    Fix: when produces key has items AND output >= 100 chars AND error contains
    'Clarification required', rescue the step as COMPLETED.
    """
    from victor.agent.planning.base import PlanStep, StepResult, StepStatus, StepType
    from victor.agent.planning.readable_schema import ReadableTaskPlan, TaskComplexity
    from victor.agent.planning.team_execution import PlanningTeamExecutionAdapter

    svc = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))

    checklist_output = (
        "Arc vs Rc: Use Rc for single-threaded scenarios; use Arc only when data is shared across threads\n"
        "Immutable bindings: Use let instead of let mut wherever possible\n"
        "Concurrency safety: Ensure shared state is protected by Mutex/RwLock or is Arc-wrapped\n"
        "Cow usage: Use Cow<str> or Cow<[T]> when a function may return either borrowed or owned data\n"
    )

    plan = ReadableTaskPlan(
        name="Rescue test",
        complexity=TaskComplexity.COMPLEX,
        desc="Rust best practices rescue",
        steps=[
            {"id": "1", "type": "analyze", "desc": "Map workspace", "produces": "workspace_members"},
            {
                "id": "2",
                "type": "doc",
                "desc": "Create comprehensive Rust best practices checklist",
                "deps": ["1"],
                "produces": "best_practices_checklist",
                "exec": "compute",
            },
            {
                "id": "3",
                "type": "analyze",
                "desc": "Review workspace using checklist",
                "deps": ["2"],
                "inputs": ["workspace_members", "best_practices_checklist"],
                "produces": "review_findings",
                "tools": ["read", "grep"],
            },
        ],
    )

    call_counts: dict[str, int] = {"step1": 0, "step2": 0, "step3": 0}

    async def fake_execute(step, **kwargs):
        if step.id == "1":
            call_counts["step1"] += 1
            return StepResult(
                success=True, output="rust/crates/protocol\nrust/crates/state\nrust/crates/tools",
                tool_calls_used=1,
            )
        if step.id == "2":
            call_counts["step2"] += 1
            # Simulate what the real sub-agent did: agentic loop FAILED but content was produced.
            return StepResult(
                success=False,
                output=checklist_output,
                tool_calls_used=0,
                error="Clarification required: target artifact or scope is underspecified",
            )
        if step.id == "3":
            call_counts["step3"] += 1
            return StepResult(success=True, output="review done", tool_calls_used=5)
        return StepResult(success=True, output="ok", tool_calls_used=0)

    mock_adapter = MagicMock(spec=PlanningTeamExecutionAdapter)
    mock_adapter.execute_step = AsyncMock(side_effect=fake_execute)

    with patch.object(svc, "_apply_step_evidence_contract", side_effect=lambda step, r, *a, **kw: r):
        result = await svc._execute_plan_via_team_adapter(plan, mock_adapter)

    # Step 2 should be rescued and step 3 should run (not skipped).
    assert call_counts["step2"] == 1, "Step 2 must have been attempted"
    assert call_counts["step3"] == 1, (
        "Step 3 must run — step 2 should have been rescued, not treated as failed"
    )
    assert result.steps_completed == 3, f"All 3 steps should complete; got {result.steps_completed}"
    assert result.steps_failed == 0, f"No failures expected after rescue; got {result.steps_failed}"


@pytest.mark.asyncio
async def test_insufficient_evidence_rescue_upgrades_synthesis_step_to_completed():
    """A step FAILED with 'Insufficient execution evidence' but produced valid output must be rescued.

    Regression: victor chat -p zai-coding step 8 (Cross-crate analysis) — the evidence contract
    rejected the step because 180-char synthesis text lacked file refs, but the step ran 6 tools
    and populated cross_crate_findings with 6 items.  The rescue must catch 'Insufficient
    execution evidence' to allow downstream steps (report synthesis) to proceed.
    """
    from victor.agent.planning.base import PlanStep, StepResult, StepStatus, StepType
    from victor.agent.planning.readable_schema import ReadableTaskPlan, TaskComplexity
    from victor.agent.planning.team_execution import PlanningTeamExecutionAdapter

    svc = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))

    synthesis_output = (
        "Cross-crate analysis complete.\n"
        "Shared Arc patterns found across protocol and state crates.\n"
        "Redundant clone() calls identified at crate boundaries.\n"
        "Dependency coupling issues noted between edge-runtime and python-bindings.\n"
        "See per_crate_findings for crate-level detail.\n"
    )

    plan = ReadableTaskPlan(
        name="Synthesis rescue test",
        complexity=TaskComplexity.COMPLEX,
        desc="Rust cross-crate synthesis rescue",
        steps=[
            {"id": "1", "type": "analyze", "desc": "Per-crate review", "produces": "per_crate_findings"},
            {
                "id": "2",
                "type": "analyze",
                "desc": "Cross-crate analysis: identify shared Arc patterns and redundant clones",
                "deps": ["1"],
                "inputs": ["per_crate_findings"],
                "produces": "cross_crate_findings",
                "tools": ["read", "grep"],
            },
            {
                "id": "3",
                "type": "doc",
                "desc": "Write consolidated report",
                "deps": ["2"],
                "inputs": ["cross_crate_findings"],
                "tools": ["write"],
            },
        ],
    )

    call_counts: dict[str, int] = {"step1": 0, "step2": 0, "step3": 0}

    async def fake_execute(step, **kwargs):
        if step.id == "1":
            call_counts["step1"] += 1
            return StepResult(
                success=True, output="[rust/crates/protocol]\n[rust/crates/state]\n[rust/crates/tools]",
                tool_calls_used=6,
            )
        if step.id == "2":
            call_counts["step2"] += 1
            # Agentic loop said COMPLETE, but evidence contract will override to FAILED
            # because 180-char output lacks file refs and counted scope.
            return StepResult(
                success=False,
                output=synthesis_output,
                tool_calls_used=6,
                error=(
                    "Insufficient execution evidence for step 2: missing concrete file "
                    "references, counts, artifacts, or scoped findings"
                ),
            )
        if step.id == "3":
            call_counts["step3"] += 1
            return StepResult(success=True, output="report.md written", tool_calls_used=1)
        return StepResult(success=True, output="ok", tool_calls_used=0)

    mock_adapter = MagicMock(spec=PlanningTeamExecutionAdapter)
    mock_adapter.execute_step = AsyncMock(side_effect=fake_execute)

    with patch.object(svc, "_apply_step_evidence_contract", side_effect=lambda step, r, *a, **kw: r):
        result = await svc._execute_plan_via_team_adapter(plan, mock_adapter)

    assert call_counts["step2"] == 1, "Step 2 must have been attempted"
    assert call_counts["step3"] == 1, (
        "Step 3 must run — step 2 should have been rescued, not treated as failed"
    )
    assert result.steps_completed == 3, f"All 3 steps should complete; got {result.steps_completed}"
    assert result.steps_failed == 0, f"No failures expected after rescue; got {result.steps_failed}"


def test_evidence_contract_exempts_synthesis_step_when_inputs_in_plan_state():
    """Evidence contract must be skipped for synthesis steps whose inputs are all in plan_state.

    Regression: step 8 cross-crate analysis declares inputs=['per_crate_findings'] which
    is already populated by step 7.  The evidence contract should not fire for such steps —
    they synthesize from collected data, and requiring file-ref patterns in output text is
    a false positive.
    """
    from victor.agent.planning.base import PlanStep, StepResult, StepType

    svc = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))

    step = PlanStep(
        id="8",
        step_type=StepType.RESEARCH,
        description="Cross-crate analysis: identify shared Arc patterns",
        inputs=["per_crate_findings"],
        context={"produces": "cross_crate_findings"},
    )

    step_result = StepResult(
        success=True,
        output="Cross-crate analysis: protocol and state share Arc<Config> pattern.",
        tool_calls_used=6,
    )

    # plan_state has per_crate_findings → synthesis step → contract exempt
    plan_state = {"per_crate_findings": ["protocol findings", "state findings"]}
    result = svc._apply_step_evidence_contract(step, step_result, plan_state)
    assert result.success is True, (
        f"Synthesis step with inputs in plan_state should be exempt from evidence contract; "
        f"got success={result.success}, error={getattr(result, 'error', None)}"
    )

    # Without plan_state the contract fires normally (not exempt)
    result_no_state = svc._apply_step_evidence_contract(step, step_result, plan_state=None)
    # With only 6 tools and 57 chars, evidence contract should fail (no file refs, low chars)
    # This verifies the contract IS active when plan_state is absent.
    assert result_no_state.success is False, (
        f"Without plan_state, evidence contract should reject thin 57-char research output; "
        f"got success={result_no_state.success}"
    )


def test_evidence_contract_not_exempt_for_research_step_with_gathering_tools():
    """Research step with data-gathering tools (read/grep/code_search) must NOT be exempt
    from evidence contract even when its declared inputs are present in plan_state.

    Regression: step 9 (cross-crate analysis, allowed_tools=['read','grep','code_search'])
    was being exempted because per_crate_findings was in plan_state.  This allowed the step
    to pass despite producing only an intent statement ("Let me use grep...") with 2 tools
    and 65 chars — yielding cross_crate_findings=[] and a hollow synthesis step downstream.
    """
    from victor.agent.planning.base import PlanStep, StepResult, StepType

    svc = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))

    step = PlanStep(
        id="9",
        step_type=StepType.RESEARCH,
        description="Cross-crate analysis: identify shared Arc patterns",
        inputs=["per_crate_findings"],
        allowed_tools=["read", "grep", "code_search"],
        context={"produces": "cross_crate_findings"},
    )
    step_result = StepResult(
        success=True,
        output="Let me use grep and file reading to analyze cross-crate patterns.",
        tool_calls_used=2,
        metadata={},
    )

    # plan_state has per_crate_findings — but step has gathering tools → NOT exempt
    plan_state = {"per_crate_findings": ["protocol findings", "state findings"]}
    result = svc._apply_step_evidence_contract(step, step_result, plan_state)
    assert result.success is False, (
        "Research step with data-gathering tools must not be exempt from evidence contract "
        "even when inputs are in plan_state; "
        f"got success={result.success}, error={getattr(result, 'error', None)}"
    )
    assert "Insufficient execution evidence" in (result.error or ""), (
        f"Expected evidence contract failure message; got error={result.error}"
    )


@pytest.mark.asyncio
async def test_failed_step_does_not_store_produces_key_in_plan_state():
    """A step that fails (success=False) must not write its produces_key to plan_state.

    Regression: even when a step's result has success=False (e.g. after evidence contract
    rejects it), the produces storage code used `if produces_key and step_result.output`
    which is truthy for any non-empty string output.  This silently stored empty findings
    (cross_crate_findings=[]) in plan_state, allowing downstream synthesis steps to proceed
    with no real data rather than being BLOCKED.
    """
    from victor.agent.planning.base import StepResult
    from victor.agent.planning.team_execution import PlanningTeamExecutionAdapter

    svc = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))
    captured_plan_state_step2: dict = {}

    async def _fake_execute(**kw) -> StepResult:
        step = kw["step"]
        ps = dict(kw.get("plan_state") or {})
        if step.id == "1":
            # step 1 fails with a truthy (non-empty) output — the classic regression case
            return StepResult(
                success=False,
                output="Let me analyze the crates.",
                error="Insufficient execution evidence",
                tool_calls_used=2,
                metadata={},
            )
        # step 2 — capture what's in plan_state after step 1 ran
        captured_plan_state_step2.update(ps)
        return StepResult(success=True, output="done", tool_calls_used=0, metadata={})

    plan = ReadableTaskPlan(
        name="Produces fail test",
        complexity=TaskComplexity.COMPLEX,
        desc="Failed step produces should not pollute plan_state",
        steps=[
            {
                "id": "1",
                "type": "analyze",
                "desc": "Gather crate findings",
                "exec": "agent",
                "produces": "crate_findings",
            },
            {
                "id": "2",
                "type": "doc",
                "desc": "Step that runs after step 1",
                "exec": "agent",
                # no dep on step 1 — so it runs and we can observe plan_state
            },
        ],
    )

    mock_adapter = MagicMock(spec=PlanningTeamExecutionAdapter)
    mock_adapter.execute_step = AsyncMock(side_effect=_fake_execute)

    # Bypass evidence contract and force sequential execution (1 at a time) so step 2
    # runs after step 1 and receives the live (post-step-1) plan_state.
    with (
        patch.object(svc, "_apply_step_evidence_contract", side_effect=lambda s, r, *a, **kw: r),
        patch.object(svc, "_effective_team_plan_concurrency", return_value=1),
    ):
        await svc._execute_plan_via_team_adapter(plan, mock_adapter)

    assert "crate_findings" not in captured_plan_state_step2, (
        "Failed step must not store its produces_key in plan_state; "
        f"plan_state seen by step 2: {list(captured_plan_state_step2.keys())}"
    )


def test_exempt_step_has_evidence_validation_metadata():
    """Steps exempt from evidence contract must have evidence_validation set in metadata
    (with exempt=True) so they appear as 'exempt' rather than 'missing' in the summary.

    Regression: execution-type-exempt steps (conditional, approval, compute) returned
    the original step_result unchanged — evidence_validation was never set in metadata,
    causing _format_evidence_validation_summary_for_summary to count them as 'missing'.
    """
    from victor.agent.planning.base import PlanStep, StepResult, StepType

    svc = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))

    step = PlanStep(
        id="7",
        step_type=StepType.RESEARCH,
        description="Route: multi-crate vs single crate",
        context={"execution": "conditional"},
    )
    step_result = StepResult(
        success=True, output="multi-crate", tool_calls_used=0, metadata={}
    )

    result = svc._apply_step_evidence_contract(step, step_result)

    ev = (result.metadata or {}).get("evidence_validation")
    assert ev is not None, (
        "Exempt step must have evidence_validation in metadata; "
        f"metadata={result.metadata}"
    )
    assert ev.get("exempt") is True, (
        f"Exempt step must have exempt=True in evidence_validation; got: {ev}"
    )
    assert ev.get("passed") is True, (
        f"Exempt step must be marked passed=True in evidence_validation; got: {ev}"
    )


def test_evidence_summary_counts_exempt_separately_from_missing():
    """Evidence validation summary must show 'exempt=N' for legitimately bypassed steps
    rather than lumping them into 'missing=N'.

    'missing' should only reflect steps where evidence_validation was never set due to a
    code bug — not steps where the evidence contract was intentionally skipped.
    """
    from victor.agent.planning.base import PlanResult, PlanStep, StepResult, StepType

    svc = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))

    plan = ReadableTaskPlan(
        name="Summary test",
        complexity=TaskComplexity.COMPLEX,
        desc="test",
        steps=[
            {
                "id": "1",
                "type": "analyze",
                "desc": "Conditional route",
                "exec": "conditional",
            },
        ],
    )

    execution_plan = plan.to_execution_plan()
    step = execution_plan.steps[0]
    step_result = StepResult(
        success=True, output="multi-crate", tool_calls_used=0, metadata={}
    )
    result = svc._apply_step_evidence_contract(step, step_result)

    plan_result = PlanResult(
        plan_id="test",
        success=True,
        total_steps=1,
        step_results={"1": result},
    )

    summary = svc._format_evidence_validation_summary_for_summary(plan, plan_result)
    assert summary, "Summary should not be empty"
    assert "missing=0" in summary, (
        f"Exempt steps must not count as missing; summary: {summary}"
    )
    assert "exempt=1" in summary, (
        f"Exempt steps must appear as exempt=1 in summary; summary: {summary}"
    )
