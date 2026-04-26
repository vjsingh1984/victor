"""Tests for RLManager and module-level prompt rollout helpers."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from victor.framework.rl import (
    RLManager,
    create_prompt_rollout_experiment,
    create_prompt_rollout_experiment_async,
)


class TestRLManagerPromptRollouts:
    def test_create_prompt_rollout_experiment_delegates_to_coordinator(self) -> None:
        coordinator = MagicMock()
        coordinator.create_prompt_rollout_experiment.return_value = "prompt_exp_123"
        manager = RLManager(coordinator=coordinator)

        experiment_id = manager.create_prompt_rollout_experiment(
            section_name="GROUNDING_RULES",
            provider="ollama",
            treatment_hash="candidate123",
            traffic_split=0.2,
            min_samples_per_variant=25,
        )

        assert experiment_id == "prompt_exp_123"
        coordinator.create_prompt_rollout_experiment.assert_called_once_with(
            section_name="GROUNDING_RULES",
            provider="ollama",
            treatment_hash="candidate123",
            control_hash=None,
            traffic_split=0.2,
            min_samples_per_variant=25,
        )

    @pytest.mark.asyncio
    async def test_create_prompt_rollout_experiment_async_delegates_to_coordinator(self) -> None:
        coordinator = MagicMock()
        coordinator.create_prompt_rollout_experiment_async = AsyncMock(
            return_value="prompt_exp_123"
        )
        manager = RLManager(coordinator=coordinator)

        experiment_id = await manager.create_prompt_rollout_experiment_async(
            section_name="GROUNDING_RULES",
            provider="ollama",
            treatment_hash="candidate123",
            traffic_split=0.2,
            min_samples_per_variant=25,
        )

        assert experiment_id == "prompt_exp_123"
        coordinator.create_prompt_rollout_experiment_async.assert_called_once_with(
            section_name="GROUNDING_RULES",
            provider="ollama",
            treatment_hash="candidate123",
            control_hash=None,
            traffic_split=0.2,
            min_samples_per_variant=25,
        )


class TestRLModulePromptRollouts:
    def test_create_prompt_rollout_experiment_uses_global_coordinator(self) -> None:
        coordinator = MagicMock()
        coordinator.create_prompt_rollout_experiment.return_value = "prompt_exp_456"

        with patch("victor.framework.rl.get_rl_coordinator", return_value=coordinator):
            experiment_id = create_prompt_rollout_experiment(
                section_name="GROUNDING_RULES",
                provider="ollama",
                treatment_hash="candidate456",
                traffic_split=0.15,
                min_samples_per_variant=40,
            )

        assert experiment_id == "prompt_exp_456"
        coordinator.create_prompt_rollout_experiment.assert_called_once_with(
            section_name="GROUNDING_RULES",
            provider="ollama",
            treatment_hash="candidate456",
            control_hash=None,
            traffic_split=0.15,
            min_samples_per_variant=40,
        )

    @pytest.mark.asyncio
    async def test_create_prompt_rollout_experiment_async_uses_global_coordinator(self) -> None:
        coordinator = MagicMock()
        coordinator.create_prompt_rollout_experiment_async = AsyncMock(
            return_value="prompt_exp_456"
        )

        with patch(
            "victor.framework.rl.get_rl_coordinator_async",
            new=AsyncMock(return_value=coordinator),
        ):
            experiment_id = await create_prompt_rollout_experiment_async(
                section_name="GROUNDING_RULES",
                provider="ollama",
                treatment_hash="candidate456",
                traffic_split=0.15,
                min_samples_per_variant=40,
            )

        assert experiment_id == "prompt_exp_456"
        coordinator.create_prompt_rollout_experiment_async.assert_called_once_with(
            section_name="GROUNDING_RULES",
            provider="ollama",
            treatment_hash="candidate456",
            control_hash=None,
            traffic_split=0.15,
            min_samples_per_variant=40,
        )
