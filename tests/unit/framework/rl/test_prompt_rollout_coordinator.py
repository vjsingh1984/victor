"""Tests for prompt rollout helpers on RLCoordinator."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch


class TestPromptRolloutCoordinator:
    def test_method_exists(self):
        from victor.framework.rl.coordinator import RLCoordinator

        assert hasattr(RLCoordinator, "create_prompt_rollout_experiment")
        assert hasattr(RLCoordinator, "analyze_prompt_rollout_experiment")
        assert hasattr(RLCoordinator, "apply_prompt_rollout_recommendation")
        assert hasattr(RLCoordinator, "process_prompt_candidate_evaluation_suite")

    def test_get_prompt_rollout_experiment_id_matches_convention(self):
        from victor.framework.rl.coordinator import RLCoordinator

        experiment_id = RLCoordinator.get_prompt_rollout_experiment_id(
            section_name="GROUNDING_RULES",
            provider="anthropic",
            treatment_hash="candidate123",
        )

        assert experiment_id == "prompt_optimizer_grounding_rules_anthropic_candidate123"

    def test_create_prompt_rollout_experiment_delegates_to_prompt_optimizer(self):
        from victor.framework.rl.coordinator import RLCoordinator

        coord = MagicMock(spec=RLCoordinator)
        learner = MagicMock()
        learner.create_rollout_experiment.return_value = "exp_123"
        coord.get_learner.return_value = learner
        coord.db = object()

        fake_exp_coordinator = MagicMock()

        with patch(
            "victor.framework.rl.coordinator.get_experiment_coordinator",
            return_value=fake_exp_coordinator,
        ):
            experiment_id = RLCoordinator.create_prompt_rollout_experiment(
                coord,
                section_name="GROUNDING_RULES",
                provider="ollama",
                treatment_hash="candidate123",
                traffic_split=0.2,
                min_samples_per_variant=25,
            )

        assert experiment_id == "exp_123"
        coord.get_learner.assert_called_once_with("prompt_optimizer")
        learner.create_rollout_experiment.assert_called_once_with(
            fake_exp_coordinator,
            section_name="GROUNDING_RULES",
            provider="ollama",
            treatment_hash="candidate123",
            control_hash=None,
            traffic_split=0.2,
            min_samples_per_variant=25,
        )

    def test_create_prompt_rollout_experiment_returns_none_without_prompt_optimizer(self):
        from victor.framework.rl.coordinator import RLCoordinator

        coord = MagicMock(spec=RLCoordinator)
        coord.get_learner.return_value = None

        result = RLCoordinator.create_prompt_rollout_experiment(
            coord,
            section_name="GROUNDING_RULES",
            provider="ollama",
            treatment_hash="candidate123",
        )

        assert result is None

    def test_create_prompt_rollout_experiment_returns_none_on_failure(self):
        from victor.framework.rl.coordinator import RLCoordinator

        coord = MagicMock(spec=RLCoordinator)
        learner = MagicMock()
        learner.create_rollout_experiment.side_effect = ValueError("benchmark gating")
        coord.get_learner.return_value = learner
        coord.db = object()

        with patch(
            "victor.framework.rl.coordinator.get_experiment_coordinator",
            return_value=MagicMock(),
        ):
            result = RLCoordinator.create_prompt_rollout_experiment(
                coord,
                section_name="GROUNDING_RULES",
                provider="ollama",
                treatment_hash="candidate123",
            )

        assert result is None

    def test_analyze_prompt_rollout_experiment_returns_structured_report(self):
        from victor.framework.rl.coordinator import RLCoordinator

        coord = MagicMock(spec=RLCoordinator)
        coord.db = object()
        fake_exp_coordinator = MagicMock()
        fake_exp_coordinator.get_experiment_status.return_value = {
            "experiment_id": "prompt_optimizer_grounding_rules_anthropic_candidate123",
            "status": "running",
            "section_name": "GROUNDING_RULES",
            "provider": "anthropic",
        }
        fake_exp_coordinator.analyze_experiment.return_value = SimpleNamespace(
            recommendation="Roll out treatment - significant improvement detected",
            is_significant=True,
            treatment_better=True,
            effect_size=0.2,
            p_value=0.01,
            confidence_interval=(0.05, 0.3),
            details={
                "control": {"samples": 120, "success_rate": 0.55},
                "treatment": {"samples": 120, "success_rate": 0.66},
            },
        )

        with patch(
            "victor.framework.rl.coordinator.get_experiment_coordinator",
            return_value=fake_exp_coordinator,
        ):
            report = RLCoordinator.analyze_prompt_rollout_experiment(
                coord,
                section_name="GROUNDING_RULES",
                provider="anthropic",
                treatment_hash="candidate123",
            )

        assert report is not None
        assert report["experiment_id"] == "prompt_optimizer_grounding_rules_anthropic_candidate123"
        assert report["analysis_available"] is True
        assert report["auto_action"] == "rollout"
        assert report["recommendation"].startswith("Roll out treatment")

    def test_apply_prompt_rollout_recommendation_rolls_out_treatment(self):
        from victor.framework.rl.coordinator import RLCoordinator

        coord = MagicMock(spec=RLCoordinator)
        coord.db = object()
        fake_exp_coordinator = MagicMock()
        fake_exp_coordinator.get_experiment_status.return_value = {
            "experiment_id": "prompt_optimizer_grounding_rules_anthropic_candidate123",
            "status": "running",
            "section_name": "GROUNDING_RULES",
            "provider": "anthropic",
        }
        fake_exp_coordinator.analyze_experiment.return_value = SimpleNamespace(
            recommendation="Roll out treatment - significant improvement detected",
            is_significant=True,
            treatment_better=True,
            effect_size=0.2,
            p_value=0.01,
            confidence_interval=(0.05, 0.3),
            details={},
        )
        fake_exp_coordinator.rollout_treatment.return_value = True

        with patch(
            "victor.framework.rl.coordinator.get_experiment_coordinator",
            return_value=fake_exp_coordinator,
        ):
            decision = RLCoordinator.apply_prompt_rollout_recommendation(
                coord,
                section_name="GROUNDING_RULES",
                provider="anthropic",
                treatment_hash="candidate123",
            )

        assert decision is not None
        assert decision["action"] == "rollout"
        assert decision["applied"] is True
        fake_exp_coordinator.rollout_treatment.assert_called_once_with(
            "prompt_optimizer_grounding_rules_anthropic_candidate123"
        )

    def test_apply_prompt_rollout_recommendation_supports_dry_run(self):
        from victor.framework.rl.coordinator import RLCoordinator

        coord = MagicMock(spec=RLCoordinator)
        coord.db = object()
        fake_exp_coordinator = MagicMock()
        fake_exp_coordinator.get_experiment_status.return_value = {
            "experiment_id": "prompt_optimizer_grounding_rules_anthropic_candidate123",
            "status": "running",
            "section_name": "GROUNDING_RULES",
            "provider": "anthropic",
        }
        fake_exp_coordinator.analyze_experiment.return_value = SimpleNamespace(
            recommendation="Keep control - treatment performed worse",
            is_significant=True,
            treatment_better=False,
            effect_size=-0.1,
            p_value=0.02,
            confidence_interval=(-0.2, -0.01),
            details={},
        )

        with patch(
            "victor.framework.rl.coordinator.get_experiment_coordinator",
            return_value=fake_exp_coordinator,
        ):
            decision = RLCoordinator.apply_prompt_rollout_recommendation(
                coord,
                section_name="GROUNDING_RULES",
                provider="anthropic",
                treatment_hash="candidate123",
                dry_run=True,
            )

        assert decision is not None
        assert decision["action"] == "rollback"
        assert decision["applied"] is False
        assert decision["dry_run"] is True
        fake_exp_coordinator.rollback_experiment.assert_not_called()

    def test_process_prompt_candidate_evaluation_suite_creates_rollout(self):
        from victor.framework.rl.coordinator import RLCoordinator

        coord = MagicMock(spec=RLCoordinator)
        coord.db = object()
        suite = SimpleNamespace(runs=[])
        learner = MagicMock()
        learner.create_rollout_experiment.return_value = "prompt_exp_123"
        sync_result = SimpleNamespace(
            approved_prompt_candidate_hash="candidate123",
            decisions=[
                SimpleNamespace(
                    prompt_candidate_hash="candidate123",
                    section_name="GROUNDING_RULES",
                    provider="anthropic",
                )
            ],
        )
        learner.sync_evaluation_suite.return_value = sync_result
        coord.get_learner.return_value = learner

        fake_exp_coordinator = MagicMock()
        with patch(
            "victor.framework.rl.coordinator.get_experiment_coordinator",
            return_value=fake_exp_coordinator,
        ):
            workflow = RLCoordinator.process_prompt_candidate_evaluation_suite(
                coord,
                suite,
                min_pass_rate=0.6,
                create_rollout=True,
                rollout_traffic_split=0.2,
                rollout_min_samples_per_variant=25,
            )

        learner.sync_evaluation_suite.assert_called_once_with(
            suite,
            min_pass_rate=0.6,
            promote_best=False,
        )
        assert workflow.prompt_optimizer_sync is sync_result
        assert workflow.prompt_rollout == {
            "created": True,
            "experiment_id": "prompt_exp_123",
        }

    def test_process_prompt_candidate_evaluation_suite_analyzes_and_applies_decision(self):
        from victor.framework.rl.coordinator import RLCoordinator

        coord = MagicMock(spec=RLCoordinator)
        coord.db = object()
        suite = SimpleNamespace(runs=[])
        learner = MagicMock()
        sync_result = SimpleNamespace(
            approved_prompt_candidate_hash="candidate123",
            decisions=[
                SimpleNamespace(
                    prompt_candidate_hash="candidate123",
                    section_name="GROUNDING_RULES",
                    provider="anthropic",
                )
            ],
        )
        learner.sync_evaluation_suite.return_value = sync_result
        coord.get_learner.return_value = learner

        fake_exp_coordinator = MagicMock()
        fake_exp_coordinator.get_experiment_status.return_value = {
            "experiment_id": "prompt_optimizer_grounding_rules_anthropic_candidate123",
            "status": "running",
            "section_name": "GROUNDING_RULES",
            "provider": "anthropic",
        }
        fake_exp_coordinator.analyze_experiment.return_value = SimpleNamespace(
            recommendation="Roll out treatment - significant improvement detected",
            is_significant=True,
            treatment_better=True,
            effect_size=0.2,
            p_value=0.01,
            confidence_interval=(0.05, 0.3),
            details={},
        )
        fake_exp_coordinator.rollout_treatment.return_value = True

        with patch(
            "victor.framework.rl.coordinator.get_experiment_coordinator",
            return_value=fake_exp_coordinator,
        ):
            workflow = RLCoordinator.process_prompt_candidate_evaluation_suite(
                coord,
                suite,
                analyze_rollout=True,
                apply_rollout_decision=True,
            )

        assert workflow.prompt_optimizer_sync is sync_result
        assert workflow.prompt_rollout_analysis is not None
        assert workflow.prompt_rollout_analysis["auto_action"] == "rollout"
        assert workflow.prompt_rollout_decision == {
            "experiment_id": "prompt_optimizer_grounding_rules_anthropic_candidate123",
            "action": "rollout",
            "applied": True,
            "dry_run": False,
        }
        fake_exp_coordinator.rollout_treatment.assert_called_once_with(
            "prompt_optimizer_grounding_rules_anthropic_candidate123"
        )
