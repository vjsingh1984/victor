"""Reinforcement Learning-related protocol definitions.

These protocols define how verticals provide RL configurations.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable, Dict, Any, List, Optional


@runtime_checkable
class RLProvider(Protocol):
    """Protocol for providing reinforcement learning configurations.

    RL providers enable verticals to learn from feedback and improve
    over time.
    """

    def get_rl_config(self) -> Dict[str, Any]:
        """Return RL configuration.

        Returns:
            Dictionary with RL parameters
        """
        ...

    def get_reward_function(self) -> Optional[Any]:
        """Return the reward function for training.

        Returns:
            Reward function or None if not configured
        """
        ...

    def get_training_data_config(self) -> Optional[Dict[str, Any]]:
        """Return training data configuration.

        Returns:
            Training data config or None
        """
        ...

    def get_evaluation_metrics(self) -> List[str]:
        """Return evaluation metrics for RL.

        Returns:
            List of metric names
        """
        ...
