"""Tests for prompt rollout helpers on RLCoordinator."""

from unittest.mock import MagicMock, patch


class TestPromptRolloutCoordinator:
    def test_method_exists(self):
        from victor.framework.rl.coordinator import RLCoordinator

        assert hasattr(RLCoordinator, "create_prompt_rollout_experiment")

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
