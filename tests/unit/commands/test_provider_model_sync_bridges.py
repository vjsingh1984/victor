from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from pathlib import Path

import pytest
import typer
import click

import victor.ui.commands.ab_testing as ab_testing_cmd
import victor.ui.commands.benchmark as benchmark_cmd
import victor.ui.commands.chat as chat_cmd
import victor.ui.commands.config as config_cmd
import victor.ui.commands.dashboard as dashboard_cmd
import victor.ui.commands.index as index_cmd
import victor.ui.commands.init as init_cmd
import victor.ui.commands.mcp as mcp_cmd
import victor.ui.commands.models as models_cmd
import victor.ui.commands.optimization as optimization_cmd
import victor.ui.commands.providers as providers_cmd
import victor.ui.commands.serve as serve_cmd
import victor.ui.commands.test_provider as test_provider_cmd
import victor.ui.commands.tools as tools_cmd
import victor.ui.commands.workflow as workflow_cmd


def _call_chat_command(**overrides: object) -> None:
    ctx = overrides.pop("ctx", MagicMock(invoked_subcommand=None))
    chat_cmd.chat(
        ctx,
        message=overrides.pop("message", None),
        message_opt=overrides.pop("message_opt", None),
        profile=overrides.pop("profile", "default"),
        stream=overrides.pop("stream", True),
        log_level=overrides.pop("log_level", None),
        thinking=overrides.pop("thinking", False),
        json_output=overrides.pop("json_output", False),
        plain=overrides.pop("plain", False),
        code_only=overrides.pop("code_only", False),
        stdin=overrides.pop("stdin", False),
        quiet=overrides.pop("quiet", False),
        renderer=overrides.pop("renderer", "auto"),
        mode=overrides.pop("mode", None),
        tool_budget=overrides.pop("tool_budget", None),
        max_iterations=overrides.pop("max_iterations", None),
        provider=overrides.pop("provider", None),
        model=overrides.pop("model", None),
        endpoint=overrides.pop("endpoint", None),
        input_file=overrides.pop("input_file", None),
        preindex=overrides.pop("preindex", False),
        vertical=overrides.pop("vertical", None),
        workflow=overrides.pop("workflow", None),
        validate_workflow=overrides.pop("validate_workflow", False),
        render_format=overrides.pop("render_format", None),
        render_output=overrides.pop("render_output", None),
        auth_mode=overrides.pop("auth_mode", None),
        coding_plan=overrides.pop("coding_plan", False),
        legacy_mode=overrides.pop("legacy_mode", False),
        enable_observability=overrides.pop("enable_observability", True),
        log_events=overrides.pop("log_events", False),
        show_reasoning=overrides.pop("show_reasoning", False),
        enable_planning=overrides.pop("enable_planning", None),
        planning_model=overrides.pop("planning_model", None),
        list_sessions=overrides.pop("list_sessions", False),
        session_id=overrides.pop("session_id", None),
        tui=overrides.pop("tui", False),
    )
    assert not overrides


class TestModelsSyncBridge:
    def test_list_models_uses_shared_sync_bridge(self) -> None:
        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(models_cmd, "list_models_async", mock_async),
            patch.object(models_cmd, "run_sync", return_value=None) as mock_run_sync,
        ):
            models_cmd.list_models(provider="ollama", endpoint="http://localhost:11434")

        mock_async.assert_called_once_with("ollama", "http://localhost:11434")
        mock_run_sync.assert_called_once_with(coro)


class TestProvidersSyncBridge:
    def test_check_provider_uses_shared_sync_bridge(self) -> None:
        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(providers_cmd, "_check_provider_async", mock_async),
            patch.object(providers_cmd, "run_sync", return_value=None) as mock_run_sync,
        ):
            providers_cmd.check_provider(
                provider="deepseek",
                model="deepseek-chat",
                connectivity=True,
                timeout=7.5,
                json_output=True,
            )

        mock_async.assert_called_once_with(
            provider="deepseek",
            model="deepseek-chat",
            connectivity=True,
            timeout=7.5,
            json_output=True,
        )
        mock_run_sync.assert_called_once_with(coro)

    def test_verify_provider_uses_shared_sync_bridge(self) -> None:
        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(providers_cmd, "_verify_provider_async", mock_async),
            patch.object(providers_cmd, "run_sync", return_value=None) as mock_run_sync,
        ):
            providers_cmd.verify_provider(
                provider="anthropic",
                model="claude-3-5-haiku",
                api_key="secret",
            )

        mock_async.assert_called_once_with(
            provider="anthropic",
            model="claude-3-5-haiku",
            api_key="secret",
        )
        mock_run_sync.assert_called_once_with(coro)

    def test_auth_login_uses_shared_sync_bridge_and_normalizes_provider(self) -> None:
        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(providers_cmd, "_auth_login_async", mock_async),
            patch.object(providers_cmd, "run_sync", return_value=None) as mock_run_sync,
        ):
            providers_cmd.auth_login(provider="OpenAI", force=True)

        mock_async.assert_called_once_with(provider="openai", force=True)
        mock_run_sync.assert_called_once_with(coro)

    def test_auth_login_rejects_unsupported_provider_before_bridge(self) -> None:
        with (
            patch.object(providers_cmd, "_auth_login_async") as mock_async,
            patch.object(providers_cmd, "run_sync") as mock_run_sync,
        ):
            with pytest.raises(typer.Exit):
                providers_cmd.auth_login(provider="unsupported", force=False)

        mock_async.assert_not_called()
        mock_run_sync.assert_not_called()


class TestAdditionalCommandSyncBridges:
    def test_list_tools_uses_shared_sync_bridge(self) -> None:
        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(tools_cmd, "_list_tools_async", mock_async),
            patch.object(tools_cmd, "run_sync", return_value=None) as mock_run_sync,
        ):
            tools_cmd.list_tools(profile="default", lightweight=False)

        mock_async.assert_called_once_with("default")
        mock_run_sync.assert_called_once_with(coro)

    def test_mcp_uses_shared_sync_bridge_for_stdio(self) -> None:
        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(mcp_cmd, "_run_mcp_server", mock_async),
            patch.object(mcp_cmd, "run_sync", return_value=None) as mock_run_sync,
            patch.object(mcp_cmd, "setup_logging"),
        ):
            mcp_cmd._mcp(stdio=True, log_level="info")

        mock_async.assert_called_once_with()
        mock_run_sync.assert_called_once_with(coro)

    def test_test_provider_uses_shared_sync_bridge(self) -> None:
        coro = object()
        mock_async = Mock(return_value=coro)
        ctx = MagicMock(invoked_subcommand=None)

        with (
            patch.object(test_provider_cmd, "test_provider_async", mock_async),
            patch.object(test_provider_cmd, "run_sync", return_value=None) as mock_run_sync,
            patch.object(test_provider_cmd.console, "print"),
        ):
            test_provider_cmd.test_provider(
                ctx,
                provider="openai",
                auth_mode="oauth",
                coding_plan=True,
                endpoint="coding",
            )

        mock_async.assert_called_once_with(
            "openai",
            auth_mode="oauth",
            coding_plan=True,
            endpoint="coding",
        )
        mock_run_sync.assert_called_once_with(coro)

    def test_index_uses_shared_sync_bridge(self) -> None:
        coro = object()
        mock_async = Mock(return_value=coro)
        ctx = MagicMock(invoked_subcommand=None)

        with (
            patch.object(index_cmd, "load_settings", return_value=object()),
            patch.object(index_cmd.os.path, "isdir", return_value=True),
            patch.object(index_cmd, "_build_index_async", mock_async),
            patch.object(index_cmd, "run_sync", return_value=True) as mock_run_sync,
            patch.object(index_cmd.console, "print"),
        ):
            index_cmd.index(ctx, force=True, path=".")

        mock_async.assert_called_once()
        call_args = mock_async.call_args
        assert call_args.args[0] == index_cmd.os.path.abspath(".")
        assert call_args.args[2] is True
        mock_run_sync.assert_called_once_with(coro)


class TestABTestingSyncBridge:
    @pytest.mark.parametrize(
        ("command", "helper_name", "args"),
        [
            (
                "create_experiment",
                "_create_experiment_async",
                (Path("experiment.yaml"),),
            ),
            ("start_experiment", "_start_experiment_async", ("exp_123",)),
            ("stop_experiment", "_stop_experiment_async", ("exp_123",)),
            ("show_status", "_show_status_async", ("exp_123",)),
            ("show_results", "_show_results_async", ("exp_123", True)),
        ],
    )
    def test_ab_command_uses_shared_sync_bridge(
        self,
        command: str,
        helper_name: str,
        args: tuple[object, ...],
    ) -> None:
        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(ab_testing_cmd, helper_name, mock_async),
            patch.object(ab_testing_cmd, "run_sync", return_value=None) as mock_run_sync,
        ):
            getattr(ab_testing_cmd, command)(*args)

        mock_async.assert_called_once_with(*args)
        mock_run_sync.assert_called_once_with(coro)

    def test_list_experiments_uses_sync_helper(self) -> None:
        with patch.object(ab_testing_cmd, "_list_experiments") as mock_list:
            ab_testing_cmd.list_experiments(status_filter="running")

        mock_list.assert_called_once_with("running")


class TestOptimizationSyncBridge:
    def test_optimization_group_exposes_prompt_rollout_command(self) -> None:
        assert "prompt-rollout" in optimization_cmd.opt.commands
        assert "prompt-rollouts" in optimization_cmd.opt.commands
        assert "prompt-rollout-status" in optimization_cmd.opt.commands
        assert "prompt-rollout-results" in optimization_cmd.opt.commands
        assert "prompt-rollout-apply" in optimization_cmd.opt.commands
        assert "prompt-rollout-auto-apply" in optimization_cmd.opt.commands

    def test_profile_uses_shared_sync_bridge(self) -> None:
        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(optimization_cmd, "_profile_async", mock_async),
            patch.object(optimization_cmd, "run_sync", return_value=None) as mock_run_sync,
        ):
            optimization_cmd.profile.callback("workflow-a", 5, "profile.json")

        mock_async.assert_called_once_with("workflow-a", 5, "profile.json")
        mock_run_sync.assert_called_once_with(coro)

    def test_suggest_uses_shared_sync_bridge(self) -> None:
        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(optimization_cmd, "_suggest_async", mock_async),
            patch.object(optimization_cmd, "run_sync", return_value=None) as mock_run_sync,
        ):
            optimization_cmd.suggest.callback("workflow-a", 4, 8, 0.7, 0.2, "suggestions.json")

        mock_async.assert_called_once_with(
            workflow_id="workflow-a",
            min_executions=4,
            max_suggestions=8,
            min_confidence=0.7,
            min_improvement=0.2,
            output="suggestions.json",
        )
        mock_run_sync.assert_called_once_with(coro)

    def test_optimize_uses_shared_sync_bridge(self) -> None:
        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(optimization_cmd, "_optimize_async", mock_async),
            patch.object(optimization_cmd, "run_sync", return_value=None) as mock_run_sync,
        ):
            optimization_cmd.optimize.callback(
                "workflow-a",
                "config.json",
                "hill_climbing",
                25,
                True,
                "optimized.json",
            )

        mock_async.assert_called_once_with(
            workflow_id="workflow-a",
            config_file="config.json",
            algorithm="hill_climbing",
            max_iterations=25,
            validate=True,
            output="optimized.json",
        )
        mock_run_sync.assert_called_once_with(coro)

    def test_validate_uses_shared_sync_bridge(self) -> None:
        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(optimization_cmd, "_validate_async", mock_async),
            patch.object(optimization_cmd, "run_sync", return_value=None) as mock_run_sync,
        ):
            optimization_cmd.validate.callback("variant.json", "workflow-a", "tests.json")

        mock_async.assert_called_once_with("variant.json", "workflow-a", "tests.json")
        mock_run_sync.assert_called_once_with(coro)

    def test_prompt_rollout_uses_shared_sync_bridge(self) -> None:
        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(optimization_cmd, "_prompt_rollout_async", mock_async),
            patch.object(optimization_cmd, "run_sync", return_value=None) as mock_run_sync,
        ):
            optimization_cmd.prompt_rollout.callback(
                "GROUNDING_RULES",
                "anthropic",
                "candidate123",
                "control456",
                0.2,
                25,
            )

        mock_async.assert_called_once_with(
            section_name="GROUNDING_RULES",
            provider="anthropic",
            treatment_hash="candidate123",
            control_hash="control456",
            traffic_split=0.2,
            min_samples_per_variant=25,
        )
        mock_run_sync.assert_called_once_with(coro)

    def test_prompt_rollouts_uses_listing_helper(self) -> None:
        with patch.object(optimization_cmd, "_list_prompt_rollouts") as mock_list:
            optimization_cmd.prompt_rollouts.callback(status_filter="running")

        mock_list.assert_called_once_with("running")

    def test_prompt_rollout_status_uses_status_helper(self) -> None:
        with patch.object(optimization_cmd, "_show_prompt_rollout_status") as mock_status:
            optimization_cmd.prompt_rollout_status.callback(
                "prompt_optimizer_grounding_rules_anthropic_candidate123"
            )

        mock_status.assert_called_once_with(
            "prompt_optimizer_grounding_rules_anthropic_candidate123"
        )

    def test_prompt_rollout_results_uses_results_helper(self) -> None:
        with patch.object(optimization_cmd, "_show_prompt_rollout_results") as mock_results:
            optimization_cmd.prompt_rollout_results.callback(
                "prompt_optimizer_grounding_rules_anthropic_candidate123"
            )

        mock_results.assert_called_once_with(
            "prompt_optimizer_grounding_rules_anthropic_candidate123"
        )

    def test_prompt_rollout_apply_uses_apply_helper(self) -> None:
        with patch.object(optimization_cmd, "_apply_prompt_rollout_decision") as mock_apply:
            optimization_cmd.prompt_rollout_apply.callback(
                "prompt_optimizer_grounding_rules_anthropic_candidate123",
                "rollout",
            )

        mock_apply.assert_called_once_with(
            "prompt_optimizer_grounding_rules_anthropic_candidate123",
            "rollout",
        )

    def test_prompt_rollout_auto_apply_uses_auto_apply_helper(self) -> None:
        with patch.object(optimization_cmd, "_auto_apply_prompt_rollout_decision") as mock_apply:
            optimization_cmd.prompt_rollout_auto_apply.callback(
                "prompt_optimizer_grounding_rules_anthropic_candidate123"
            )

        mock_apply.assert_called_once_with(
            "prompt_optimizer_grounding_rules_anthropic_candidate123"
        )

    @pytest.mark.asyncio
    async def test_prompt_rollout_async_reports_started_experiment(self) -> None:
        with (
            patch.object(
                optimization_cmd,
                "create_prompt_rollout_experiment_async",
                AsyncMock(return_value="prompt_exp_123"),
            ) as mock_create,
            patch.object(optimization_cmd.click, "echo") as mock_echo,
        ):
            await optimization_cmd._prompt_rollout_async(
                section_name="GROUNDING_RULES",
                provider="anthropic",
                treatment_hash="candidate123",
                control_hash=None,
                traffic_split=0.1,
                min_samples_per_variant=50,
            )

        mock_create.assert_awaited_once_with(
            section_name="GROUNDING_RULES",
            provider="anthropic",
            treatment_hash="candidate123",
            control_hash=None,
            traffic_split=0.1,
            min_samples_per_variant=50,
        )
        mock_echo.assert_any_call("Starting prompt rollout experiment:")
        mock_echo.assert_any_call("  Section: GROUNDING_RULES")
        mock_echo.assert_any_call("  Provider: anthropic")
        mock_echo.assert_any_call("  Treatment hash: candidate123")
        mock_echo.assert_any_call("  Traffic split: 10.0%")
        mock_echo.assert_any_call("  Min samples per variant: 50")
        mock_echo.assert_any_call("Prompt rollout experiment started: prompt_exp_123")

    @pytest.mark.asyncio
    async def test_prompt_rollout_async_reports_creation_failure(self) -> None:
        with (
            patch.object(
                optimization_cmd,
                "create_prompt_rollout_experiment_async",
                AsyncMock(return_value=None),
            ),
            patch.object(optimization_cmd.click, "echo") as mock_echo,
        ):
            await optimization_cmd._prompt_rollout_async(
                section_name="GROUNDING_RULES",
                provider="anthropic",
                treatment_hash="candidate123",
                control_hash=None,
                traffic_split=0.1,
                min_samples_per_variant=50,
            )

        mock_echo.assert_any_call(
            "Unable to start prompt rollout experiment for section: GROUNDING_RULES",
            err=True,
        )

    @pytest.mark.asyncio
    async def test_prompt_rollout_async_reports_validation_error(self) -> None:
        with (
            patch.object(
                optimization_cmd,
                "create_prompt_rollout_experiment_async",
                AsyncMock(side_effect=ValueError("benchmark gating required")),
            ),
            patch.object(optimization_cmd.click, "echo") as mock_echo,
        ):
            await optimization_cmd._prompt_rollout_async(
                section_name="GROUNDING_RULES",
                provider="anthropic",
                treatment_hash="candidate123",
                control_hash=None,
                traffic_split=0.1,
                min_samples_per_variant=50,
            )

        mock_echo.assert_any_call("Cannot start prompt rollout: benchmark gating required", err=True)

    def test_list_prompt_rollouts_shows_filtered_prompt_experiments(self) -> None:
        coordinator = MagicMock()
        coordinator.list_experiments.return_value = [
            {
                "experiment_id": "prompt_optimizer_grounding_rules_anthropic_candidate123",
                "name": "Prompt rollout for GROUNDING_RULES",
                "status": "running",
                "traffic_split": 0.1,
                "control": {
                    "name": "control456",
                    "samples": 12,
                    "success_rate": 0.75,
                },
                "treatment": {
                    "name": "candidate123",
                    "samples": 9,
                    "success_rate": 0.89,
                },
            },
            {
                "experiment_id": "other_experiment",
                "name": "Other experiment",
                "status": "running",
                "traffic_split": 0.5,
                "control": {"name": "a", "samples": 1, "success_rate": 1.0},
                "treatment": {"name": "b", "samples": 1, "success_rate": 1.0},
            },
        ]

        with (
            patch.object(
                optimization_cmd,
                "get_experiment_coordinator",
                return_value=coordinator,
            ),
            patch.object(optimization_cmd.click, "echo") as mock_echo,
        ):
            optimization_cmd._list_prompt_rollouts("running")

        coordinator.list_experiments.assert_called_once_with()
        mock_echo.assert_any_call("Prompt rollout experiments:")
        mock_echo.assert_any_call(
            "  prompt_optimizer_grounding_rules_anthropic_candidate123"
        )
        mock_echo.assert_any_call("    Status: running")
        mock_echo.assert_any_call("    Traffic split: 10.0%")
        mock_echo.assert_any_call("    Control: control456 samples=12 success_rate=75.0%")
        mock_echo.assert_any_call("    Treatment: candidate123 samples=9 success_rate=89.0%")

    def test_list_prompt_rollouts_reports_none_found(self) -> None:
        coordinator = MagicMock()
        coordinator.list_experiments.return_value = [
            {
                "experiment_id": "other_experiment",
                "name": "Other experiment",
                "status": "running",
                "traffic_split": 0.5,
                "control": {"name": "a", "samples": 1, "success_rate": 1.0},
                "treatment": {"name": "b", "samples": 1, "success_rate": 1.0},
            }
        ]

        with (
            patch.object(
                optimization_cmd,
                "get_experiment_coordinator",
                return_value=coordinator,
            ),
            patch.object(optimization_cmd.click, "echo") as mock_echo,
        ):
            optimization_cmd._list_prompt_rollouts("paused")

        mock_echo.assert_called_once_with("No prompt rollout experiments found.")

    def test_show_prompt_rollout_status_displays_experiment_summary(self) -> None:
        coordinator = MagicMock()
        coordinator.get_experiment_status.return_value = {
            "experiment_id": "prompt_optimizer_grounding_rules_anthropic_candidate123",
            "name": "Prompt rollout for GROUNDING_RULES",
            "status": "running",
            "traffic_split": 0.1,
            "control": {
                "name": "control456",
                "samples": 12,
                "success_rate": 0.75,
            },
            "treatment": {
                "name": "candidate123",
                "samples": 9,
                "success_rate": 0.89,
            },
        }

        with (
            patch.object(
                optimization_cmd,
                "get_experiment_coordinator",
                return_value=coordinator,
            ),
            patch.object(optimization_cmd.click, "echo") as mock_echo,
        ):
            optimization_cmd._show_prompt_rollout_status(
                "prompt_optimizer_grounding_rules_anthropic_candidate123"
            )

        coordinator.get_experiment_status.assert_called_once_with(
            "prompt_optimizer_grounding_rules_anthropic_candidate123"
        )
        mock_echo.assert_any_call(
            "Prompt rollout experiment: "
            "prompt_optimizer_grounding_rules_anthropic_candidate123"
        )
        mock_echo.assert_any_call("  Name: Prompt rollout for GROUNDING_RULES")
        mock_echo.assert_any_call("  Status: running")
        mock_echo.assert_any_call("  Traffic split: 10.0%")
        mock_echo.assert_any_call("  Control: control456 samples=12 success_rate=75.0%")
        mock_echo.assert_any_call("  Treatment: candidate123 samples=9 success_rate=89.0%")

    def test_show_prompt_rollout_status_rejects_non_prompt_rollout_id(self) -> None:
        with patch.object(optimization_cmd.click, "echo") as mock_echo:
            optimization_cmd._show_prompt_rollout_status("other_experiment")

        mock_echo.assert_called_once_with(
            "Experiment is not a prompt rollout: other_experiment",
            err=True,
        )

    def test_show_prompt_rollout_status_reports_missing_experiment(self) -> None:
        coordinator = MagicMock()
        coordinator.get_experiment_status.return_value = None

        with (
            patch.object(
                optimization_cmd,
                "get_experiment_coordinator",
                return_value=coordinator,
            ),
            patch.object(optimization_cmd.click, "echo") as mock_echo,
        ):
            optimization_cmd._show_prompt_rollout_status(
                "prompt_optimizer_grounding_rules_anthropic_candidate123"
            )

        mock_echo.assert_called_once_with(
            "Prompt rollout experiment not found: "
            "prompt_optimizer_grounding_rules_anthropic_candidate123",
            err=True,
        )

    def test_show_prompt_rollout_results_displays_analysis(self) -> None:
        coordinator = MagicMock()
        coordinator.analyze_experiment.return_value = SimpleNamespace(
            experiment_id="prompt_optimizer_grounding_rules_anthropic_candidate123",
            is_significant=True,
            treatment_better=True,
            effect_size=0.12,
            p_value=0.03,
            confidence_interval=(0.02, 0.18),
            recommendation="Roll out treatment - significant improvement detected",
            details={
                "control": {
                    "samples": 24,
                    "success_rate": 0.75,
                    "avg_quality": 0.81,
                },
                "treatment": {
                    "samples": 25,
                    "success_rate": 0.84,
                    "avg_quality": 0.88,
                },
                "z_score": 2.1,
            },
        )

        with (
            patch.object(
                optimization_cmd,
                "get_experiment_coordinator",
                return_value=coordinator,
            ),
            patch.object(optimization_cmd.click, "echo") as mock_echo,
        ):
            optimization_cmd._show_prompt_rollout_results(
                "prompt_optimizer_grounding_rules_anthropic_candidate123"
            )

        coordinator.analyze_experiment.assert_called_once_with(
            "prompt_optimizer_grounding_rules_anthropic_candidate123"
        )
        mock_echo.assert_any_call(
            "Prompt rollout analysis: "
            "prompt_optimizer_grounding_rules_anthropic_candidate123"
        )
        mock_echo.assert_any_call("  Significant: yes")
        mock_echo.assert_any_call("  Treatment better: yes")
        mock_echo.assert_any_call("  Effect size: 12.0%")
        mock_echo.assert_any_call("  P-value: 0.0300")
        mock_echo.assert_any_call("  Confidence interval: [0.0200, 0.1800]")
        mock_echo.assert_any_call(
            "  Recommendation: Roll out treatment - significant improvement detected"
        )
        mock_echo.assert_any_call("  Auto-apply action: rollout")
        mock_echo.assert_any_call(
            "  Control: samples=24 success_rate=75.0% avg_quality=0.810"
        )
        mock_echo.assert_any_call(
            "  Treatment: samples=25 success_rate=84.0% avg_quality=0.880"
        )

    def test_show_prompt_rollout_results_rejects_non_prompt_rollout_id(self) -> None:
        with patch.object(optimization_cmd.click, "echo") as mock_echo:
            optimization_cmd._show_prompt_rollout_results("other_experiment")

        mock_echo.assert_called_once_with(
            "Experiment is not a prompt rollout: other_experiment",
            err=True,
        )

    def test_show_prompt_rollout_results_reports_missing_analysis(self) -> None:
        coordinator = MagicMock()
        coordinator.analyze_experiment.return_value = None

        with (
            patch.object(
                optimization_cmd,
                "get_experiment_coordinator",
                return_value=coordinator,
            ),
            patch.object(optimization_cmd.click, "echo") as mock_echo,
        ):
            optimization_cmd._show_prompt_rollout_results(
                "prompt_optimizer_grounding_rules_anthropic_candidate123"
            )

        mock_echo.assert_called_once_with(
            "Prompt rollout analysis unavailable: "
            "prompt_optimizer_grounding_rules_anthropic_candidate123",
            err=True,
        )

    def test_apply_prompt_rollout_decision_rolls_out_treatment(self) -> None:
        coordinator = MagicMock()
        coordinator.rollout_treatment.return_value = True

        with (
            patch.object(
                optimization_cmd,
                "get_experiment_coordinator",
                return_value=coordinator,
            ),
            patch.object(optimization_cmd.click, "echo") as mock_echo,
        ):
            optimization_cmd._apply_prompt_rollout_decision(
                "prompt_optimizer_grounding_rules_anthropic_candidate123",
                "rollout",
            )

        coordinator.rollout_treatment.assert_called_once_with(
            "prompt_optimizer_grounding_rules_anthropic_candidate123"
        )
        mock_echo.assert_called_once_with(
            "Prompt rollout marked as rolled out: "
            "prompt_optimizer_grounding_rules_anthropic_candidate123"
        )

    def test_apply_prompt_rollout_decision_rolls_back_experiment(self) -> None:
        coordinator = MagicMock()
        coordinator.rollback_experiment.return_value = True

        with (
            patch.object(
                optimization_cmd,
                "get_experiment_coordinator",
                return_value=coordinator,
            ),
            patch.object(optimization_cmd.click, "echo") as mock_echo,
        ):
            optimization_cmd._apply_prompt_rollout_decision(
                "prompt_optimizer_grounding_rules_anthropic_candidate123",
                "rollback",
            )

        coordinator.rollback_experiment.assert_called_once_with(
            "prompt_optimizer_grounding_rules_anthropic_candidate123"
        )
        mock_echo.assert_called_once_with(
            "Prompt rollout marked as rolled back: "
            "prompt_optimizer_grounding_rules_anthropic_candidate123"
        )

    def test_apply_prompt_rollout_decision_rejects_non_prompt_rollout_id(self) -> None:
        with patch.object(optimization_cmd.click, "echo") as mock_echo:
            optimization_cmd._apply_prompt_rollout_decision("other_experiment", "rollout")

        mock_echo.assert_called_once_with(
            "Experiment is not a prompt rollout: other_experiment",
            err=True,
        )

    def test_apply_prompt_rollout_decision_reports_failed_transition(self) -> None:
        coordinator = MagicMock()
        coordinator.rollout_treatment.return_value = False

        with (
            patch.object(
                optimization_cmd,
                "get_experiment_coordinator",
                return_value=coordinator,
            ),
            patch.object(optimization_cmd.click, "echo") as mock_echo,
        ):
            optimization_cmd._apply_prompt_rollout_decision(
                "prompt_optimizer_grounding_rules_anthropic_candidate123",
                "rollout",
            )

        mock_echo.assert_called_once_with(
            "Unable to apply rollout decision for prompt rollout: "
            "prompt_optimizer_grounding_rules_anthropic_candidate123",
            err=True,
        )

    def test_auto_apply_prompt_rollout_decision_rolls_out_treatment(self) -> None:
        coordinator = MagicMock()
        coordinator.analyze_experiment.return_value = SimpleNamespace(
            experiment_id="prompt_optimizer_grounding_rules_anthropic_candidate123",
            is_significant=True,
            treatment_better=True,
            effect_size=0.12,
            p_value=0.03,
            confidence_interval=(0.02, 0.18),
            recommendation="Roll out treatment - significant improvement detected",
            details={},
        )
        coordinator.rollout_treatment.return_value = True

        with (
            patch.object(
                optimization_cmd,
                "get_experiment_coordinator",
                return_value=coordinator,
            ),
            patch.object(optimization_cmd.click, "echo") as mock_echo,
        ):
            optimization_cmd._auto_apply_prompt_rollout_decision(
                "prompt_optimizer_grounding_rules_anthropic_candidate123"
            )

        coordinator.analyze_experiment.assert_called_once_with(
            "prompt_optimizer_grounding_rules_anthropic_candidate123"
        )
        coordinator.rollout_treatment.assert_called_once_with(
            "prompt_optimizer_grounding_rules_anthropic_candidate123"
        )
        mock_echo.assert_called_once_with(
            "Prompt rollout auto-applied: rolled out treatment for "
            "prompt_optimizer_grounding_rules_anthropic_candidate123"
        )

    def test_auto_apply_prompt_rollout_decision_rolls_back_experiment(self) -> None:
        coordinator = MagicMock()
        coordinator.analyze_experiment.return_value = SimpleNamespace(
            experiment_id="prompt_optimizer_grounding_rules_anthropic_candidate123",
            is_significant=True,
            treatment_better=False,
            effect_size=-0.12,
            p_value=0.03,
            confidence_interval=(-0.18, -0.02),
            recommendation="Keep control - treatment performed worse",
            details={},
        )
        coordinator.rollback_experiment.return_value = True

        with (
            patch.object(
                optimization_cmd,
                "get_experiment_coordinator",
                return_value=coordinator,
            ),
            patch.object(optimization_cmd.click, "echo") as mock_echo,
        ):
            optimization_cmd._auto_apply_prompt_rollout_decision(
                "prompt_optimizer_grounding_rules_anthropic_candidate123"
            )

        coordinator.analyze_experiment.assert_called_once_with(
            "prompt_optimizer_grounding_rules_anthropic_candidate123"
        )
        coordinator.rollback_experiment.assert_called_once_with(
            "prompt_optimizer_grounding_rules_anthropic_candidate123"
        )
        mock_echo.assert_called_once_with(
            "Prompt rollout auto-applied: kept control for "
            "prompt_optimizer_grounding_rules_anthropic_candidate123"
        )

    def test_auto_apply_prompt_rollout_decision_skips_non_actionable_recommendation(self) -> None:
        coordinator = MagicMock()
        coordinator.analyze_experiment.return_value = SimpleNamespace(
            experiment_id="prompt_optimizer_grounding_rules_anthropic_candidate123",
            is_significant=False,
            treatment_better=True,
            effect_size=0.12,
            p_value=0.18,
            confidence_interval=(-0.01, 0.18),
            recommendation="Continue experiment - results not yet significant",
            details={},
        )

        with (
            patch.object(
                optimization_cmd,
                "get_experiment_coordinator",
                return_value=coordinator,
            ),
            patch.object(optimization_cmd.click, "echo") as mock_echo,
        ):
            optimization_cmd._auto_apply_prompt_rollout_decision(
                "prompt_optimizer_grounding_rules_anthropic_candidate123"
            )

        coordinator.analyze_experiment.assert_called_once_with(
            "prompt_optimizer_grounding_rules_anthropic_candidate123"
        )
        coordinator.rollout_treatment.assert_not_called()
        coordinator.rollback_experiment.assert_not_called()
        mock_echo.assert_called_once_with(
            "Prompt rollout auto-apply skipped for "
            "prompt_optimizer_grounding_rules_anthropic_candidate123: "
            "Continue experiment - results not yet significant"
        )

    def test_auto_apply_prompt_rollout_decision_rejects_non_prompt_rollout_id(self) -> None:
        with patch.object(optimization_cmd.click, "echo") as mock_echo:
            optimization_cmd._auto_apply_prompt_rollout_decision("other_experiment")

        mock_echo.assert_called_once_with(
            "Experiment is not a prompt rollout: other_experiment",
            err=True,
        )

    def test_auto_apply_prompt_rollout_decision_reports_missing_analysis(self) -> None:
        coordinator = MagicMock()
        coordinator.analyze_experiment.return_value = None

        with (
            patch.object(
                optimization_cmd,
                "get_experiment_coordinator",
                return_value=coordinator,
            ),
            patch.object(optimization_cmd.click, "echo") as mock_echo,
        ):
            optimization_cmd._auto_apply_prompt_rollout_decision(
                "prompt_optimizer_grounding_rules_anthropic_candidate123"
            )

        mock_echo.assert_called_once_with(
            "Prompt rollout analysis unavailable: "
            "prompt_optimizer_grounding_rules_anthropic_candidate123",
            err=True,
        )

    def test_auto_apply_prompt_rollout_decision_reports_failed_transition(self) -> None:
        coordinator = MagicMock()
        coordinator.analyze_experiment.return_value = SimpleNamespace(
            experiment_id="prompt_optimizer_grounding_rules_anthropic_candidate123",
            is_significant=True,
            treatment_better=True,
            effect_size=0.12,
            p_value=0.03,
            confidence_interval=(0.02, 0.18),
            recommendation="Roll out treatment - significant improvement detected",
            details={},
        )
        coordinator.rollout_treatment.return_value = False

        with (
            patch.object(
                optimization_cmd,
                "get_experiment_coordinator",
                return_value=coordinator,
            ),
            patch.object(optimization_cmd.click, "echo") as mock_echo,
        ):
            optimization_cmd._auto_apply_prompt_rollout_decision(
                "prompt_optimizer_grounding_rules_anthropic_candidate123"
            )

        mock_echo.assert_called_once_with(
            "Unable to auto-apply prompt rollout decision for "
            "prompt_optimizer_grounding_rules_anthropic_candidate123",
            err=True,
        )

    def test_get_prompt_rollout_auto_action_returns_rollout(self) -> None:
        result = SimpleNamespace(
            is_significant=True,
            treatment_better=True,
            recommendation="Roll out treatment - significant improvement detected",
        )

        assert optimization_cmd._get_prompt_rollout_auto_action(result) == "rollout"

    def test_get_prompt_rollout_auto_action_returns_rollback(self) -> None:
        result = SimpleNamespace(
            is_significant=True,
            treatment_better=False,
            recommendation="Keep control - treatment performed worse",
        )

        assert optimization_cmd._get_prompt_rollout_auto_action(result) == "rollback"

    def test_get_prompt_rollout_auto_action_returns_none_for_non_actionable_result(self) -> None:
        result = SimpleNamespace(
            is_significant=False,
            treatment_better=True,
            recommendation="Continue experiment - results not yet significant",
        )

        assert optimization_cmd._get_prompt_rollout_auto_action(result) is None

    def test_prompt_rollout_rejects_invalid_traffic_split_before_bridge(self) -> None:
        with (
            patch.object(optimization_cmd, "_prompt_rollout_async") as mock_async,
            patch.object(optimization_cmd, "run_sync") as mock_run_sync,
        ):
            with pytest.raises(click.BadParameter, match="traffic_split must be between 0 and 1"):
                optimization_cmd.prompt_rollout.callback(
                    "GROUNDING_RULES",
                    "anthropic",
                    "candidate123",
                    None,
                    1.0,
                    25,
                )

        mock_async.assert_not_called()
        mock_run_sync.assert_not_called()

    def test_prompt_rollout_rejects_non_positive_min_samples_before_bridge(self) -> None:
        with (
            patch.object(optimization_cmd, "_prompt_rollout_async") as mock_async,
            patch.object(optimization_cmd, "run_sync") as mock_run_sync,
        ):
            with pytest.raises(
                click.BadParameter,
                match="min_samples_per_variant must be at least 1",
            ):
                optimization_cmd.prompt_rollout.callback(
                    "GROUNDING_RULES",
                    "anthropic",
                    "candidate123",
                    None,
                    0.2,
                    0,
                )

        mock_async.assert_not_called()
        mock_run_sync.assert_not_called()


class TestConfigSyncBridge:
    def test_config_validate_uses_shared_sync_bridge_for_connectivity_checks(
        self,
        tmp_path: Path,
    ) -> None:
        config_dir = tmp_path / ".victor"
        config_dir.mkdir(exist_ok=True)
        (config_dir / "profiles.yaml").write_text("profiles:\n  default:\n    provider: ollama\n")

        settings = MagicMock()
        settings.get_config_dir.return_value = config_dir
        settings.load_profiles.return_value = {
            "default": MagicMock(
                provider="ollama",
                model="qwen",
                temperature=0.7,
                max_tokens=1024,
            )
        }
        settings.get_provider_settings.return_value = {}

        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(config_cmd, "load_settings", return_value=settings),
            patch.object(
                config_cmd,
                "validate_configuration",
                return_value=MagicMock(is_valid=True),
            ),
            patch.object(config_cmd, "format_validation_result", return_value="ok"),
            patch(
                "victor.providers.registry.ProviderRegistry.list_providers",
                return_value=["ollama"],
            ),
            patch.object(config_cmd, "_check_connectivity", mock_async),
            patch.object(config_cmd, "run_sync", return_value=None) as mock_run_sync,
            patch("builtins.print"),
            patch.object(config_cmd.Console, "print"),
        ):
            config_cmd.config_validate(verbose=True, check_connectivity=True, fix=False)

        mock_async.assert_called_once_with(settings, settings.load_profiles.return_value, True)
        mock_run_sync.assert_called_once_with(coro)


class TestBenchmarkSyncBridge:
    def test_setup_benchmark_uses_shared_sync_bridge(self) -> None:
        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(benchmark_cmd, "_configure_log_level"),
            patch.object(benchmark_cmd, "_setup_benchmark_async", mock_async),
            patch.object(benchmark_cmd, "run_sync", return_value=None) as mock_run_sync,
        ):
            benchmark_cmd.setup_benchmark(
                benchmark="swe-bench",
                max_tasks=3,
                force_reindex=True,
                log_level="INFO",
            )

        mock_async.assert_called_once_with(
            benchmark="swe-bench",
            benchmark_lower="swe-bench",
            max_tasks=3,
            force_reindex=True,
        )
        mock_run_sync.assert_called_once_with(coro)

    def test_run_benchmark_uses_shared_sync_bridge(self) -> None:
        runner = object()
        config = SimpleNamespace(model="model-x")
        coro = object()
        mock_async = Mock(return_value=coro)
        result = MagicMock()
        result.get_metrics.return_value = {
            "total_tasks": 1,
            "passed": 1,
            "failed": 0,
            "errors": 0,
            "pass_rate": 1.0,
            "duration_seconds": 1.0,
            "total_tokens": 42,
            "total_tool_calls": 5,
            "total_code_search_calls": 0,
            "total_graph_calls": 0,
            "tasks_using_code_intelligence": 0,
            "code_intelligence_task_coverage": 0.0,
        }
        result.task_results = []

        with (
            patch.object(benchmark_cmd, "_configure_log_level"),
            patch(
                "victor.evaluation.benchmarks.HumanEvalRunner",
                Mock(return_value=runner),
            ),
            patch("victor.evaluation.protocol.EvaluationConfig", return_value=config),
            patch.object(benchmark_cmd, "_run_benchmark_async", mock_async),
            patch.object(benchmark_cmd, "run_sync", return_value=result) as mock_run_sync,
            patch.object(benchmark_cmd.console, "print"),
        ):
            benchmark_cmd.run_benchmark(
                benchmark="humaneval",
                max_tasks=2,
                start_task=0,
                model="model-x",
                profile="default",
                output=None,
                timeout=30,
                max_turns=4,
                parallel=1,
                resume=False,
                provider=None,
                log_level=None,
                debug_modules=None,
                no_edge_model=False,
                account=None,
            )

        mock_async.assert_called_once_with(
            runner=runner,
            config=config,
            profile="default",
            model="model-x",
            timeout=30,
            max_turns=4,
            resume=False,
            provider_override=None,
            start_task=0,
            resolved_account=None,
        )
        mock_run_sync.assert_called_once_with(coro)


class TestDashboardSyncBridge:
    def test_dashboard_uses_shared_sync_bridge(self) -> None:
        coro = object()
        ctx = MagicMock(invoked_subcommand=None)
        mock_async = Mock(return_value=coro)

        with (
            patch("victor.ui.commands.utils.setup_logging"),
            patch.object(dashboard_cmd, "run_dashboard", mock_async),
            patch.object(dashboard_cmd, "run_sync", return_value=None) as mock_run_sync,
        ):
            dashboard_cmd.dashboard(
                ctx,
                log_file="events.jsonl",
                live=False,
                demo=True,
                log_level="DEBUG",
            )

        mock_async.assert_called_once_with(log_file="events.jsonl", live=False, demo=True)
        mock_run_sync.assert_called_once_with(coro)


class TestInitSyncBridge:
    def test_generate_init_content_enhanced_uses_shared_sync_bridge(self) -> None:
        coro = object()
        mock_async = Mock(return_value=coro)
        on_progress = Mock()

        with (
            patch.object(init_cmd, "_generate_init_content_async", mock_async),
            patch.object(init_cmd, "run_sync", return_value="content") as mock_run_sync,
        ):
            result = init_cmd._generate_init_content(
                mode="enhanced",
                use_llm=True,
                include_conversations=False,
                on_progress=on_progress,
                force=True,
                include_dirs=["src"],
                exclude_dirs=["tests"],
            )

        assert result == "content"
        mock_async.assert_called_once_with(
            mode="enhanced",
            use_llm=True,
            include_conversations=False,
            on_progress=on_progress,
            force=True,
            include_dirs=["src"],
            exclude_dirs=["tests"],
            provider=None,
            model=None,
        )
        mock_run_sync.assert_called_once_with(coro)

    def test_generate_init_content_index_uses_shared_sync_bridge(self) -> None:
        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(init_cmd, "_generate_init_content_async", mock_async),
            patch.object(init_cmd, "run_sync", return_value="content") as mock_run_sync,
        ):
            result = init_cmd._generate_init_content(
                mode="index",
                force=False,
                include_dirs=["src"],
                exclude_dirs=["tests"],
            )

        assert result == "content"
        mock_async.assert_called_once_with(
            mode="index",
            use_llm=False,
            include_conversations=False,
            on_progress=None,
            force=False,
            include_dirs=["src"],
            exclude_dirs=["tests"],
            provider=None,
            model=None,
        )
        mock_run_sync.assert_called_once_with(coro)

    def test_generate_init_content_quick_bypasses_run_sync(self) -> None:
        """Quick mode is sync — should not call run_sync."""
        mock_smart = Mock(return_value="quick content")

        with (
            patch.object(init_cmd, "load_codebase_analyzer_attr", return_value=mock_smart),
            patch.object(init_cmd, "run_sync") as mock_run_sync,
        ):
            result = init_cmd._generate_init_content(
                mode="quick",
                include_dirs=["src"],
                exclude_dirs=["tests"],
            )

        assert result == "quick content"
        mock_run_sync.assert_not_called()


class TestServeSyncBridge:
    def test_serve_uses_shared_sync_bridge(self) -> None:
        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(serve_cmd, "setup_logging"),
            patch.object(serve_cmd.console, "print"),
            patch.object(serve_cmd, "_run_fastapi_server", mock_async),
            patch.object(serve_cmd, "run_sync", return_value=None) as mock_run_sync,
        ):
            serve_cmd._serve(
                host="127.0.0.1",
                port=8765,
                log_level="info",
                profile="default",
                enable_hitl=True,
                hitl_auth_token="token",
            )

        mock_async.assert_called_once_with("127.0.0.1", 8765, "default", True, "token")
        mock_run_sync.assert_called_once_with(coro)

    def test_serve_hitl_uses_shared_sync_bridge(self) -> None:
        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(serve_cmd, "setup_logging"),
            patch.object(serve_cmd.console, "print"),
            patch.object(serve_cmd, "_run_hitl_server", mock_async),
            patch.object(serve_cmd, "run_sync", return_value=None) as mock_run_sync,
        ):
            serve_cmd.serve_hitl(
                host="0.0.0.0",
                port=8080,
                auth_token="secret",
                require_auth=True,
                persistent=False,
                db_path=None,
                log_level="debug",
            )

        mock_async.assert_called_once_with("0.0.0.0", 8080, True, "secret", False, None)
        mock_run_sync.assert_called_once_with(coro)


class TestWorkflowSyncBridge:
    def test_run_workflow_uses_shared_sync_bridge(self) -> None:
        workflow = SimpleNamespace(
            name="demo-workflow",
            description="demo",
            start_node="start",
            nodes={},
            metadata={},
        )
        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(
                workflow_cmd,
                "_load_workflow_file",
                return_value={"demo-workflow": workflow},
            ),
            patch.object(workflow_cmd, "_display_workflow_info"),
            patch.object(workflow_cmd.console, "print"),
            patch.object(workflow_cmd, "_execute_workflow_async", mock_async),
            patch.object(workflow_cmd, "run_sync", return_value=None) as mock_run_sync,
        ):
            workflow_cmd.run_workflow(
                "workflow.yaml",
                context=None,
                context_file=None,
                workflow_name=None,
                profile=None,
                dry_run=False,
                log_level=None,
            )

        mock_async.assert_called_once_with(workflow, {}, None)
        mock_run_sync.assert_called_once_with(coro)

    def test_generate_workflow_uses_shared_sync_bridge(self) -> None:
        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(workflow_cmd.console, "print"),
            patch.object(workflow_cmd, "_generate_workflow_async", mock_async),
            patch.object(workflow_cmd, "run_sync", return_value=None) as mock_run_sync,
        ):
            workflow_cmd.generate_workflow(
                "Analyze code and fix issues",
                output="generated.yaml",
                vertical="coding",
                profile="default",
                strategy="multi_stage",
                interactive=True,
                validate=False,
                max_retries=5,
                dry_run=False,
            )

        mock_async.assert_called_once()
        kwargs = mock_async.call_args.kwargs
        assert kwargs["description"] == "Analyze code and fix issues"
        assert kwargs["output"] == "generated.yaml"
        assert kwargs["vertical"] == "coding"
        assert kwargs["profile"] == "default"
        assert kwargs["strategy"] == "multi_stage"
        assert kwargs["interactive"] is True
        assert kwargs["validate"] is False
        assert kwargs["max_retries"] == 5
        assert kwargs["dry_run"] is False
        assert kwargs["gen_strategy"].name == "LLM_MULTI_STAGE"
        mock_run_sync.assert_called_once_with(coro)


class TestChatSyncBridge:
    def test_chat_workflow_mode_uses_shared_sync_bridge(self) -> None:
        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(chat_cmd.console, "print"),
            patch.object(chat_cmd, "run_workflow_mode", mock_async),
            patch.object(chat_cmd, "run_sync", return_value=None) as mock_run_sync,
        ):
            _call_chat_command(workflow="workflow.yaml", vertical="coding")

        mock_async.assert_called_once_with(
            workflow_path="workflow.yaml",
            validate_only=False,
            render_format=None,
            render_output=None,
            profile="default",
            vertical="coding",
            log_level=None,
        )
        mock_run_sync.assert_called_once_with(coro)

    def test_chat_oneshot_uses_shared_sync_bridge(self) -> None:
        settings = MagicMock()
        formatter = MagicMock()
        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(chat_cmd, "setup_logging"),
            patch.object(chat_cmd, "setup_safety_confirmation"),
            patch.object(chat_cmd, "create_formatter", return_value=formatter),
            patch.object(chat_cmd.InputReader, "read_message", return_value="hello"),
            patch.object(chat_cmd, "load_settings", return_value=settings),
            patch(
                "victor.config.validation.validate_configuration",
                return_value=SimpleNamespace(is_valid=True),
            ),
            patch.object(chat_cmd, "run_oneshot", mock_async),
            patch.object(chat_cmd, "run_sync", return_value=None) as mock_run_sync,
            patch.object(chat_cmd.console, "print"),
        ):
            _call_chat_command(message_opt="hello")

        mock_async.assert_called_once_with(
            "hello",
            settings,
            "default",
            True,
            False,
            formatter=formatter,
            preindex=False,
            renderer_choice="auto",
            mode=None,
            tool_budget=None,
            max_iterations=None,
            vertical=None,
            enable_observability=True,
            enable_planning=None,
            planning_model=None,
            legacy_mode=False,
            show_reasoning=False,
        )
        mock_run_sync.assert_called_once_with(coro)

    def test_chat_interactive_uses_shared_sync_bridge(self) -> None:
        settings = MagicMock()
        formatter = MagicMock()
        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(chat_cmd, "setup_logging"),
            patch.object(chat_cmd, "setup_safety_confirmation"),
            patch.object(chat_cmd, "create_formatter", return_value=formatter),
            patch.object(chat_cmd.InputReader, "read_message", return_value=None),
            patch.object(chat_cmd, "load_settings", return_value=settings),
            patch(
                "victor.config.validation.validate_configuration",
                return_value=SimpleNamespace(is_valid=True),
            ),
            patch.object(chat_cmd, "run_interactive", mock_async),
            patch.object(chat_cmd, "run_sync", return_value=None) as mock_run_sync,
            patch.object(chat_cmd.console, "print"),
        ):
            _call_chat_command(stream=False, thinking=True, session_id="session-1")

        mock_async.assert_called_once_with(
            settings,
            "default",
            False,
            True,
            preindex=False,
            renderer_choice="auto",
            mode=None,
            tool_budget=None,
            max_iterations=None,
            vertical=None,
            enable_observability=True,
            enable_planning=None,
            planning_model=None,
            legacy_mode=False,
            use_tui=False,
            resume_session_id="session-1",
            show_reasoning=False,
        )
        mock_run_sync.assert_called_once_with(coro)

    def test_default_interactive_uses_shared_sync_bridge(self) -> None:
        settings = MagicMock()
        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(chat_cmd, "setup_logging"),
            patch.object(chat_cmd, "load_settings", return_value=settings),
            patch.object(chat_cmd, "setup_safety_confirmation"),
            patch.object(chat_cmd, "run_interactive", mock_async),
            patch.object(chat_cmd, "run_sync", return_value=None) as mock_run_sync,
        ):
            chat_cmd._run_default_interactive()

        mock_async.assert_called_once_with(settings, "default", True, False, use_tui=False)
        mock_run_sync.assert_called_once_with(coro)
