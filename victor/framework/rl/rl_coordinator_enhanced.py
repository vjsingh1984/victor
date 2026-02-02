# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Enhanced RL Coordinator with Advanced Learning Algorithms.

This module extends the RL coordinator with state-of-the-art reinforcement learning
algorithms including Q-learning with experience replay, policy gradient methods,
reward shaping, and multiple exploration strategies.

Architecture:
┌────────────────────────────────────────────────────────────────────┐
│                 Enhanced RLCoordinator                             │
│  ├─ Q-Learning with Experience Replay                             │
│  ├─ Policy Gradient (REINFORCE)                                   │
│  ├─ Reward Shaping                                                │
│  ├─ Exploration Strategies (epsilon-greedy, UCB, Thompson)        │
│  └─ Policy Persistence (save/load from YAML)                      │
└────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│                 Learning Components                               │
│  ├─ ExperienceReplayBuffer: Sample-efficient learning             │
│  ├─ TargetNetwork: Stable Q-learning updates                     │
│  ├─ RewardShaper: Accelerate learning with shaped rewards        │
│  ├─ ExplorationStrategy: Adaptive exploration (epsilon/UCB/Thompson)│
│  └─ PolicySerializer: Save/load policies to YAML                 │
└────────────────────────────────────────────────────────────────────┘

Key Features:
- Experience replay for sample efficiency
- Target networks for stable learning
- Reward shaping to accelerate convergence
- Adaptive exploration strategies
- Policy persistence across sessions

Usage:
    coordinator = EnhancedRLCoordinator()
    coordinator.update_policy(reward=0.8, state=state, action=action)
    action = coordinator.select_action(state, available_actions)
    stats = coordinator.get_policy_statistics()
    coordinator.save_policy("/path/to/policy.yaml")
"""

from __future__ import annotations

import logging
import math
import random
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Generic,
    Optional,
    TypeVar,
    cast,
)
from collections.abc import Callable

import numpy as np
import yaml

from victor.framework.task import TaskResult

logger = logging.getLogger(__name__)

# Type variables for generics
S = TypeVar("S")  # State type
A = TypeVar("A")  # Action type


class ExplorationStrategy(str, Enum):
    """Exploration strategies for action selection.

    Strategies:
        EPSILON_GREEDY: Explore with probability epsilon, otherwise exploit
        UCB: Upper Confidence Bound - optimistically initialize uncertain actions
        THOMPSON_SAMPLING: Sample from posterior distribution for each action
        BOLTZMANN: Softmax selection based on action values
        ENTROPY_BONUS: Add entropy regularization for exploration
    """

    EPSILON_GREEDY = "epsilon_greedy"
    UCB = "ucb"
    THOMPSON_SAMPLING = "thompson_sampling"
    BOLTZMANN = "boltzmann"
    ENTROPY_BONUS = "entropy_bonus"


class LearningAlgorithm(str, Enum):
    """RL algorithms supported by the coordinator.

    Algorithms:
        Q_LEARNING: Classic Q-learning with experience replay
        REINFORCE: Policy gradient with Monte Carlo updates
        ACTOR_CRITIC: Actor-critic architecture (future)
        DQN: Deep Q-Network (future)
        PPO: Proximal Policy Optimization (future)
    """

    Q_LEARNING = "q_learning"
    REINFORCE = "reinforce"
    ACTOR_CRITIC = "actor_critic"
    DQN = "dqn"
    PPO = "ppo"


@dataclass
class PolicyStats:
    """Statistics about the learned policy.

    Attributes:
        algorithm: Current learning algorithm
        total_updates: Total number of policy updates
        average_reward: Rolling average of recent rewards
        exploration_rate: Current exploration rate (epsilon)
        state_count: Number of unique states visited
        action_count: Total actions taken
        q_table_size: Size of Q-table (for Q-learning)
        policy_entropy: Entropy of policy (for policy gradients)
        convergence_rate: Rate of convergence
        last_update_time: Timestamp of last update
    """

    algorithm: LearningAlgorithm
    total_updates: int = 0
    average_reward: float = 0.0
    exploration_rate: float = 0.1
    state_count: int = 0
    action_count: int = 0
    q_table_size: int = 0
    policy_entropy: float = 0.0
    convergence_rate: float = 0.0
    last_update_time: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "algorithm": self.algorithm.value,
            "total_updates": self.total_updates,
            "average_reward": self.average_reward,
            "exploration_rate": self.exploration_rate,
            "state_count": self.state_count,
            "action_count": self.action_count,
            "q_table_size": self.q_table_size,
            "policy_entropy": self.policy_entropy,
            "convergence_rate": self.convergence_rate,
            "last_update_time": self.last_update_time,
        }


@dataclass
class Experience(Generic[S, A]):
    """A single experience tuple for replay buffer.

    Attributes:
        state: State before action
        action: Action taken
        reward: Reward received
        next_state: State after action
        done: Whether episode terminated
        timestamp: When this experience occurred
    """

    state: S
    action: A
    reward: float
    next_state: Optional[S]
    done: bool
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    def to_tuple(self) -> tuple[Any, Any, float, Any, bool]:
        """Convert to tuple for efficient storage."""
        return (self.state, self.action, self.reward, self.next_state, self.done)


class ExperienceReplayBuffer(Generic[S, A]):
    """Experience replay buffer for sample-efficient learning.

    Implements uniform sampling and prioritized experience replay (PER).
    Stores experiences and samples mini-batches for training.

    Attributes:
        capacity: Maximum number of experiences to store
        buffer: Deque of experiences
        priorities: Priority values for PER (None if not using PER)
        alpha: Priority exponent for PER (0 = uniform, 1 = full priority)
        beta: Importance sampling exponent for PER

    Example:
        buffer = ExperienceReplayBuffer(capacity=10000)
        buffer.add(Experience(state, action, reward, next_state, done))
        batch = buffer.sample(batch_size=32)
    """

    def __init__(
        self,
        capacity: int = 10000,
        use_prioritization: bool = False,
        alpha: float = 0.6,
        beta: float = 0.4,
    ):
        """Initialize experience replay buffer.

        Args:
            capacity: Maximum number of experiences to store
            use_prioritization: Whether to use prioritized experience replay
            alpha: Priority exponent (0=uniform, 1=full priority)
            beta: Importance sampling exponent
        """
        self.capacity = capacity
        self.buffer: deque["Experience[S, A]"] = deque(maxlen=capacity)
        self.use_prioritization = use_prioritization
        self.priorities: Optional[np.ndarray] = None
        self.alpha = alpha
        self.beta = beta
        self.max_priority = 1.0

        if use_prioritization:
            self.priorities = np.zeros(capacity, dtype=np.float32)

    def add(self, experience: "Experience[S, A]") -> None:
        """Add experience to buffer.

        Args:
            experience: Experience tuple to add
        """
        self.buffer.append(experience)

        if self.use_prioritization and self.priorities is not None:
            # New experiences get max priority
            idx = len(self.buffer) - 1
            self.priorities[idx] = self.max_priority

    def sample(self, batch_size: int) -> list["Experience[S, A]"]:
        """Sample a batch of experiences.

        Uses uniform sampling if prioritization is disabled,
        otherwise uses proportional prioritization.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            List of sampled experiences
        """
        if not self.buffer:
            return []

        batch_size = min(batch_size, len(self.buffer))

        if not self.use_prioritization or self.priorities is None:
            # Uniform sampling
            return random.sample(list(self.buffer), batch_size)
        else:
            # Prioritized sampling
            priorities = self.priorities[: len(self.buffer)]
            probabilities = priorities**self.alpha
            probabilities /= probabilities.sum()

            indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
            return [self.buffer[idx] for idx in indices]

    def update_priorities(self, indices: list[int], priorities: list[float]) -> None:
        """Update priorities for sampled experiences.

        Args:
            indices: Indices of experiences to update
            priorities: New priority values (typically TD-error)
        """
        if not self.use_prioritization or self.priorities is None:
            return

        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = abs(priority) ** self.alpha
            self.max_priority = max(self.max_priority, self.priorities[idx])

    def __len__(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)

    def clear(self) -> None:
        """Clear all experiences from buffer."""
        self.buffer.clear()
        if self.priorities is not None:
            self.priorities.fill(0.0)
        self.max_priority = 1.0


class TargetNetwork:
    """Target network for stable Q-learning.

    Maintains a separate target network that is updated periodically
    from the main network to provide stable learning targets.

    Attributes:
        update_frequency: Steps between target network updates
        update_tau: Soft update coefficient (0 = hard update, 1 = no update)
        last_update_step: Step counter for last update

    Example:
        target_net = TargetNetwork(update_frequency=1000)
        if target_net.should_update(step):
            target_net.update(main_q_table)
    """

    def __init__(self, update_frequency: int = 1000, tau: float = 0.005):
        """Initialize target network.

        Args:
            update_frequency: Steps between updates (hard update)
            tau: Soft update coefficient (0 = hard, 1 = no update)
        """
        self.update_frequency = update_frequency
        self.tau = tau
        self.last_update_step = 0
        self.target_q_table: dict[Any, dict[Any, float]] = {}

    def should_update(self, step: int) -> bool:
        """Check if target network should be updated.

        Args:
            step: Current training step

        Returns:
            True if update is due
        """
        return step - self.last_update_step >= self.update_frequency

    def update(self, main_q_table: dict[Any, dict[Any, float]], step: int) -> None:
        """Update target network from main network.

        Supports both hard and soft updates.

        Args:
            main_q_table: Main Q-table to copy from
            step: Current training step
        """
        if not self.should_update(step):
            return

        if self.tau == 0.0:
            # Hard update
            self.target_q_table = {state: actions.copy() for state, actions in main_q_table.items()}
        else:
            # Soft update
            for state, actions in main_q_table.items():
                if state not in self.target_q_table:
                    self.target_q_table[state] = {}

                for action, q_value in actions.items():
                    old_value = self.target_q_table[state].get(action, 0.0)
                    self.target_q_table[state][action] = (
                        self.tau * q_value + (1 - self.tau) * old_value
                    )

        self.last_update_step = step
        logger.debug(f"Target network updated at step {step}")

    def get_q_value(self, state: Any, action: Any) -> float:
        """Get Q-value from target network.

        Args:
            state: State key
            action: Action key

        Returns:
            Q-value from target network
        """
        if state not in self.target_q_table:
            return 0.0
        return self.target_q_table[state].get(action, 0.0)


class RewardShaper:
    """Reward shaping to accelerate learning.

    Transforms raw rewards using potential-based reward shaping
    to maintain optimal policy while accelerating convergence.

    Attributes:
        gamma: Discount factor for potential computation
        potential_function: Function to compute state potential

    Example:
        def potential(state):
            return -distance_to_goal(state)

        shaper = RewardShaper(gamma=0.99, potential_function=potential)
        shaped_reward = shaper.shape_reward(state, action, reward, next_state)
    """

    def __init__(
        self,
        gamma: float = 0.99,
        potential_function: Optional[Callable[[Any], float]] = None,
    ):
        """Initialize reward shaper.

        Args:
            gamma: Discount factor
            potential_function: Optional function to compute state potential
        """
        self.gamma = gamma
        self.potential_function = potential_function or self._default_potential

    def _default_potential(self, state: Any) -> float:
        """Default potential function.

        Computes potential based on task completion and quality.

        Args:
            state: State to compute potential for

        Returns:
            Potential value (default: 0.0)
        """
        # Can be overridden with domain-specific logic
        return 0.0

    def shape_reward(self, state: Any, action: Any, reward: float, next_state: Any) -> float:
        """Apply potential-based reward shaping.

        F(s, a) = reward + gamma * Phi(next_state) - Phi(state)

        Args:
            state: Current state
            action: Action taken
            reward: Original reward
            next_state: Next state

        Returns:
            Shaped reward
        """
        phi_s = self.potential_function(state)
        phi_s_prime = self.potential_function(next_state)

        shaping_bonus = self.gamma * phi_s_prime - phi_s
        shaped_reward = reward + shaping_bonus

        logger.debug(
            f"Reward shaped: {reward:.3f} -> {shaped_reward:.3f} " f"(bonus: {shaping_bonus:.3f})"
        )

        return shaped_reward


class ExplorationStrategyImpl:
    """Exploration strategy implementation.

    Supports multiple exploration strategies with adaptive rate decay.

    Attributes:
        strategy: Exploration strategy type
        initial_epsilon: Initial exploration rate
        min_epsilon: Minimum exploration rate
        decay_rate: Exponential decay rate
        ucb_c: UCB exploration constant
        temperature: Boltzmann temperature

    Example:
        strategy = ExplorationStrategyImpl(
            strategy=ExplorationStrategy.EPSILON_GREEDY,
            initial_epsilon=1.0,
            decay_rate=0.995
        )
        action = strategy.select_action(q_values, available_actions)
    """

    def __init__(
        self,
        strategy: ExplorationStrategy = ExplorationStrategy.EPSILON_GREEDY,
        initial_epsilon: float = 1.0,
        min_epsilon: float = 0.01,
        decay_rate: float = 0.995,
        ucb_c: float = 2.0,
        temperature: float = 1.0,
    ):
        """Initialize exploration strategy.

        Args:
            strategy: Exploration strategy type
            initial_epsilon: Initial exploration rate
            min_epsilon: Minimum exploration rate
            decay_rate: Decay rate for epsilon
            ucb_c: UCB exploration constant
            temperature: Temperature for Boltzmann exploration
        """
        self.strategy = strategy
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.ucb_c = ucb_c
        self.temperature = temperature

        # Track action counts for UCB and Thompson sampling
        self.action_counts: dict[Any, int] = {}
        self.action_values: dict[Any, tuple[float, float]] = {}  # (mean, variance) for Thompson

    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)

    def select_action(
        self,
        q_values: dict[Any, float],
        available_actions: list[Any],
        step: int = 0,
    ) -> Any:
        """Select action using configured exploration strategy.

        Args:
            q_values: Q-values for each action
            available_actions: List of available actions
            step: Current training step

        Returns:
            Selected action
        """
        if not available_actions:
            raise ValueError("No available actions")

        if self.strategy == ExplorationStrategy.EPSILON_GREEDY:
            return self._epsilon_greedy(q_values, available_actions)
        elif self.strategy == ExplorationStrategy.UCB:
            return self._ucb(q_values, available_actions, step)
        elif self.strategy == ExplorationStrategy.THOMPSON_SAMPLING:
            return self._thompson_sampling(q_values, available_actions)
        elif self.strategy == ExplorationStrategy.BOLTZMANN:
            return self._boltzmann(q_values, available_actions)
        elif self.strategy == ExplorationStrategy.ENTROPY_BONUS:
            return self._entropy_bonus(q_values, available_actions)
        else:
            # Fallback for unknown strategies (should never happen with Enum)
            # This is kept for defensive programming and future extensibility
            return self._epsilon_greedy(q_values, available_actions)

    def _epsilon_greedy(self, q_values: dict[Any, float], available_actions: list[Any]) -> Any:
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            # Explore: random action
            action = random.choice(available_actions)
            logger.debug(f"Explore: selected random action {action}")
        else:
            # Exploit: best action
            valid_actions = [a for a in available_actions if a in q_values]
            if valid_actions:
                action = max(valid_actions, key=lambda a: q_values[a])
                logger.debug(f"Exploit: selected best action {action} (Q={q_values[action]:.3f})")
            else:
                action = random.choice(available_actions)
                logger.debug(f"Explore: no Q-values, selected random action {action}")

        return action

    def _ucb(self, q_values: dict[Any, float], available_actions: list[Any], step: int) -> Any:
        """Upper Confidence Bound action selection."""
        best_action = None
        best_value = float("-inf")

        for action in available_actions:
            # Update action count
            self.action_counts[action] = self.action_counts.get(action, 0) + 1
            count = self.action_counts[action]

            # Compute UCB value
            q_value = q_values.get(action, 0.0)
            exploration_bonus = self.ucb_c * math.sqrt(math.log(step + 1) / count)
            ucb_value = q_value + exploration_bonus

            if ucb_value > best_value:
                best_value = ucb_value
                best_action = action

        logger.debug(f"UCB: selected action {best_action} (UCB={best_value:.3f})")
        return best_action

    def _thompson_sampling(self, q_values: dict[Any, float], available_actions: list[Any]) -> Any:
        """Thompson sampling action selection."""
        # Sample from Beta distribution for each action
        # Using Q-value as mean and building posterior
        best_action = None
        best_sample = float("-inf")

        for action in available_actions:
            q_value = q_values.get(action, 0.0)

            # Simplified Thompson sampling using normal approximation
            # In practice, you'd maintain full posterior distributions
            mean, var = self.action_values.get(action, (q_value, 1.0))
            sample = random.gauss(mean, math.sqrt(var))

            # Update posterior (simplified)
            self.action_values[action] = (mean, var * 0.99)  # Decay variance

            if sample > best_sample:
                best_sample = sample
                best_action = action

        logger.debug(f"Thompson: selected action {best_action} (sample={best_sample:.3f})")
        return best_action

    def _boltzmann(self, q_values: dict[Any, float], available_actions: list[Any]) -> Any:
        """Boltzmann (softmax) action selection."""
        # Compute probabilities
        valid_actions = [a for a in available_actions if a in q_values]
        if not valid_actions:
            return random.choice(available_actions)

        # Get max Q for numerical stability
        max_q = max(q_values[a] for a in valid_actions)

        # Compute softmax probabilities
        exp_values = [math.exp((q_values[a] - max_q) / self.temperature) for a in valid_actions]
        total = sum(exp_values)
        probs = [e / total for e in exp_values]

        # Sample action
        action = random.choices(valid_actions, weights=probs, k=1)[0]
        logger.debug(f"Boltzmann: selected action {action} from distribution")
        return action

    def _entropy_bonus(self, q_values: dict[Any, float], available_actions: list[Any]) -> Any:
        """Entropy-regularized action selection."""
        # Add entropy bonus to Q-values
        valid_actions = [a for a in available_actions if a in q_values]
        if not valid_actions:
            return random.choice(available_actions)

        # Compute policy distribution
        max_q = max(q_values[a] for a in valid_actions)
        exp_values = [math.exp(q_values[a] - max_q) for a in valid_actions]
        total = sum(exp_values)

        # Select action with entropy bonus
        best_action = None
        best_value = float("-inf")

        for i, action in enumerate(valid_actions):
            policy_prob = exp_values[i] / total
            entropy_bonus = -policy_prob * math.log(policy_prob + 1e-10)
            value = q_values[action] + 0.1 * entropy_bonus

            if value > best_value:
                best_value = value
                best_action = action

        logger.debug(f"Entropy bonus: selected action {best_action}")
        return best_action


class PolicySerializer:
    """Serialize and deserialize policies to/from YAML.

    Handles saving and loading of learned policies, Q-tables,
    and training metadata.

    Attributes:
        policy_dir: Directory to save policies

    Example:
        serializer = PolicySerializer(policy_dir="~/.victor/policies")
        serializer.save_policy(q_table, "tool_selector")
        loaded_table = serializer.load_policy("tool_selector")
    """

    def __init__(self, policy_dir: Optional[Path] = None):
        """Initialize policy serializer.

        Args:
            policy_dir: Directory to save policies (default: ~/.victor/config/rl/)
        """
        if policy_dir is None:
            policy_dir = Path.home() / ".victor" / "config" / "rl"

        self.policy_dir = Path(policy_dir)
        self.policy_dir.mkdir(parents=True, exist_ok=True)

    def save_policy(
        self,
        q_table: dict[Any, dict[Any, float]],
        policy_name: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Path:
        """Save policy to YAML file.

        Args:
            q_table: Q-table to save
            policy_name: Name of the policy
            metadata: Optional metadata to include

        Returns:
            Path to saved policy file
        """
        policy_path = self.policy_dir / f"{policy_name}.yaml"

        # Convert to serializable format
        serializable_table = self._make_serializable(q_table)

        policy_data = {
            "policy_name": policy_name,
            "saved_at": datetime.now().isoformat(),
            "q_table": serializable_table,
            "metadata": metadata or {},
        }

        with open(policy_path, "w") as f:
            yaml.safe_dump(policy_data, f, default_flow_style=False)

        logger.info(f"Policy saved to {policy_path}")
        return policy_path

    def load_policy(self, policy_name: str) -> Optional[dict[Any, dict[Any, float]]]:
        """Load policy from YAML file.

        Args:
            policy_name: Name of the policy to load

        Returns:
            Loaded Q-table, or None if not found
        """
        policy_path = self.policy_dir / f"{policy_name}.yaml"

        if not policy_path.exists():
            logger.warning(f"Policy file not found: {policy_path}")
            return None

        try:
            with open(policy_path, "r") as f:
                policy_data = yaml.safe_load(f)

            q_table = policy_data.get("q_table", {})
            logger.info(f"Policy loaded from {policy_path}")

            # Convert back from strings to original types
            return self._deserialize_q_table(q_table)

        except Exception as e:
            logger.error(f"Failed to load policy from {policy_path}: {e}")
            return None

    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to YAML-serializable format.

        Args:
            obj: Object to convert

        Returns:
            Serializable version of object
        """
        if isinstance(obj, dict):
            # Convert keys to strings
            return {str(k): self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def _deserialize_q_table(
        self, q_table: dict[str, dict[str, float]]
    ) -> dict[Any, dict[Any, float]]:
        """Deserialize Q-table, attempting to restore original types.

        Args:
            q_table: Serialized Q-table

        Returns:
            Deserialized Q-table
        """
        # For now, return as-is with string keys
        # In practice, you might want to try to parse back to original types
        return q_table

    def list_policies(self) -> list[str]:
        """List all saved policies.

        Returns:
            List of policy names
        """
        policies = []
        for path in self.policy_dir.glob("*.yaml"):
            policies.append(path.stem)
        return sorted(policies)


class EnhancedRLCoordinator(Generic[S, A]):
    """Enhanced RL coordinator with advanced learning algorithms.

    Integrates Q-learning, policy gradients, experience replay, target networks,
    reward shaping, and multiple exploration strategies.

    Attributes:
        algorithm: Current learning algorithm
        learning_rate: Learning rate for updates
        gamma: Discount factor
        exploration_strategy: Exploration strategy
        replay_buffer: Experience replay buffer
        target_network: Target network for stable learning
        reward_shaper: Reward shaping function
        policy_serializer: Policy persistence manager

    Example:
        coordinator = EnhancedRLCoordinator(
            algorithm=LearningAlgorithm.Q_LEARNING,
            exploration_strategy=ExplorationStrategy.EPSILON_GREEDY
        )

        # Update policy
        coordinator.update_policy(
            reward=0.8,
            state=state,
            action=action,
            next_state=next_state
        )

        # Select action
        action = coordinator.select_action(state, available_actions)

        # Save policy
        coordinator.save_policy("my_policy")
    """

    def __init__(
        self,
        algorithm: LearningAlgorithm = LearningAlgorithm.Q_LEARNING,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        exploration_strategy: ExplorationStrategy = ExplorationStrategy.EPSILON_GREEDY,
        replay_buffer_size: int = 10000,
        target_network_update_freq: int = 1000,
        use_reward_shaping: bool = False,
        policy_dir: Optional[Path] = None,
    ):
        """Initialize enhanced RL coordinator.

        Args:
            algorithm: Learning algorithm to use
            learning_rate: Learning rate for updates
            gamma: Discount factor
            exploration_strategy: Exploration strategy
            replay_buffer_size: Size of experience replay buffer
            target_network_update_freq: Target network update frequency
            use_reward_shaping: Whether to use reward shaping
            policy_dir: Directory to save policies
        """
        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.exploration_strategy = exploration_strategy

        # Q-table for Q-learning
        self.q_table: dict[Any, dict[Any, float]] = {}

        # Policy for REINFORCE (policy gradient)
        self.policy: dict[Any, dict[Any, float]] = {}
        self.policy_grad_returns: list[float] = []  # For Monte Carlo returns

        # Experience replay buffer
        self.replay_buffer: ExperienceReplayBuffer[S, A] = ExperienceReplayBuffer(
            capacity=replay_buffer_size
        )

        # Target network for stable learning
        self.target_network = TargetNetwork(update_frequency=target_network_update_freq)

        # Reward shaping
        self.reward_shaper = RewardShaper(gamma=gamma) if use_reward_shaping else None

        # Exploration strategy
        self.exploration = ExplorationStrategyImpl(strategy=exploration_strategy)

        # Policy persistence
        self.policy_serializer = PolicySerializer(policy_dir=policy_dir)

        # Statistics
        self.stats = PolicyStats(algorithm=algorithm)
        self.total_steps = 0
        self.episode_rewards: list[float] = []

        logger.info(
            f"Enhanced RL Coordinator initialized with {algorithm.value} "
            f"and {exploration_strategy.value} exploration"
        )

    def update_policy(
        self,
        reward: float,
        state: S,
        action: A,
        next_state: Optional[S] = None,
        done: bool = False,
    ) -> None:
        """Update policy using reward signal.

        Performs policy update using the configured algorithm.

        Args:
            reward: Reward received for the action
            state: State before action
            action: Action taken
            next_state: State after action (optional for REINFORCE)
            done: Whether episode terminated

        Raises:
            ValueError: If algorithm is not supported
        """
        # Apply reward shaping if enabled
        if self.reward_shaper and next_state is not None:
            reward = self.reward_shaper.shape_reward(state, action, reward, next_state)

        # Add experience to replay buffer
        experience = Experience(
            state=state, action=action, reward=reward, next_state=next_state, done=done
        )
        self.replay_buffer.add(experience)

        # Update policy based on algorithm
        if self.algorithm == LearningAlgorithm.Q_LEARNING:
            self._update_q_learning(reward, state, action, next_state, done)
        elif self.algorithm == LearningAlgorithm.REINFORCE:
            self._update_reinforce(reward, state, action, done)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

        # Update statistics
        self.stats.total_updates += 1
        self.stats.last_update_time = datetime.now().isoformat()
        self.episode_rewards.append(reward)
        self.total_steps += 1

        # Decay exploration rate
        self.exploration.decay_epsilon()
        self.stats.exploration_rate = self.exploration.epsilon

        logger.debug(
            f"Policy updated: state={state}, action={action}, "
            f"reward={reward:.3f}, step={self.total_steps}"
        )

    def _update_q_learning(
        self, reward: float, state: S, action: A, next_state: Optional[S], done: bool
    ) -> None:
        """Update Q-table using Q-learning with experience replay.

        Q(s,a) = Q(s,a) + alpha * (reward + gamma * max_a' Q(s',a') - Q(s,a))

        Args:
            reward: Reward received
            state: Current state
            action: Action taken
            next_state: Next state
            done: Episode termination flag
        """
        # Initialize Q-values if needed
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0

        # Sample mini-batch from replay buffer
        if len(self.replay_buffer) >= 32:  # Minimum batch size
            batch: list["Experience[S, A]"] = self.replay_buffer.sample(batch_size=32)

            for exp in batch:
                s, a, r, ns, d = exp.to_tuple()

                if s not in self.q_table:
                    self.q_table[s] = {}
                if a not in self.q_table[s]:
                    self.q_table[s][a] = 0.0

                # Compute target Q-value
                if d or ns is None:
                    target = r
                else:
                    if ns not in self.q_table:
                        self.q_table[ns] = {}
                    max_next_q = max(self.q_table[ns].values()) if self.q_table[ns] else 0.0
                    target = r + self.gamma * max_next_q

                # Update Q-value
                self.q_table[s][a] += self.learning_rate * (target - self.q_table[s][a])

        # Also update current experience
        if next_state is not None and not done:
            if next_state not in self.q_table:
                self.q_table[next_state] = {}
            max_next_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0.0
            target = reward + self.gamma * max_next_q
        else:
            target = reward

        self.q_table[state][action] += self.learning_rate * (target - self.q_table[state][action])

        # Update target network periodically
        if self.target_network.should_update(self.total_steps):
            self.target_network.update(self.q_table, self.total_steps)

    def _update_reinforce(self, reward: float, state: S, action: A, done: bool) -> None:
        """Update policy using REINFORCE (Monte Carlo policy gradient).

        ∇θ J(θ) = E[∇θ log πθ(a|s) * G]

        Args:
            reward: Reward received
            state: Current state
            action: Action taken
            done: Episode termination flag
        """
        # Initialize policy if needed
        if state not in self.policy:
            self.policy[state] = {}

        # For REINFORCE, we accumulate rewards and update at end of episode
        self.policy_grad_returns.append(reward)

        if done:
            # Compute return (discounted cumulative reward)
            G = 0.0
            returns = []
            for r in reversed(self.policy_grad_returns):
                G = r + self.gamma * G
                returns.append(G)
            returns.reverse()

            # Update policy using policy gradient
            for i, (s, a) in enumerate(zip([state] * len(returns), [action] * len(returns))):
                if s not in self.policy:
                    self.policy[s] = {}
                if a not in self.policy[s]:
                    self.policy[s][a] = 0.0

                # Policy gradient update
                self.policy[s][a] += self.learning_rate * returns[i]

                # Normalize to maintain valid probability distribution
                total = sum(self.policy[s].values())
                if total > 0:
                    for act in self.policy[s]:
                        self.policy[s][act] /= total

            # Clear returns for next episode
            self.policy_grad_returns.clear()

    def select_action(self, state: S, available_actions: list[A]) -> A:
        """Select action using current policy.

        Args:
            state: Current state
            available_actions: List of available actions

        Returns:
            Selected action

        Raises:
            ValueError: If no actions available
        """
        if not available_actions:
            raise ValueError("No available actions")

        # Get Q-values or policy values
        if self.algorithm == LearningAlgorithm.Q_LEARNING:
            if state not in self.q_table:
                self.q_table[state] = {}
            q_values = self.q_table[state]
        elif self.algorithm == LearningAlgorithm.REINFORCE:
            if state not in self.policy:
                self.policy[state] = {}
            q_values = self.policy[state]
        else:
            q_values = {}

        # Select action using exploration strategy
        action = self.exploration.select_action(q_values, available_actions, self.total_steps)

        # Validate action is in available_actions
        if action not in available_actions:
            # Fallback to first available action
            logger.warning(f"Selected action {action} not in available_actions, using first")
            action = available_actions[0]

        # Update statistics
        self.stats.action_count += 1
        if state not in self.q_table and state not in self.policy:
            self.stats.state_count += 1

        return cast(A, action)

    def compute_reward(self, outcome: TaskResult) -> float:
        """Compute reward signal from task outcome.

        Uses ProficiencyTracker metrics for reward computation.

        Args:
            outcome: Task result with quality metrics

        Returns:
            Reward value (typically -1.0 to 1.0)
        """
        # Base reward from success/failure
        if hasattr(outcome, "success") and outcome.success:
            base_reward = 1.0
        else:
            base_reward = -0.5

        # Quality score modifier
        if hasattr(outcome, "quality_score") and outcome.quality_score is not None:
            quality_modifier = outcome.quality_score
        else:
            quality_modifier = 0.5

        # Tool usage efficiency
        efficiency_modifier = 0.0
        if hasattr(outcome, "metadata") and outcome.metadata:
            tools_used = outcome.metadata.get("tools_used", 0)
            if tools_used > 0:
                efficiency_modifier = min(0.2, 10.0 / tools_used - 0.1)

        # Time penalty (encourage efficiency)
        time_penalty = 0.0
        if hasattr(outcome, "duration_seconds") and outcome.duration_seconds:
            time_penalty = -min(0.2, outcome.duration_seconds / 300.0)  # 5 min max penalty

        # Combine components
        reward = (
            float(base_reward) * 0.5
            + float(quality_modifier) * 0.3
            + float(efficiency_modifier)
            + float(time_penalty)
        )

        logger.debug(f"Computed reward: {reward:.3f} from outcome")
        return reward

    def get_policy_statistics(self) -> PolicyStats:
        """Get statistics about the learned policy.

        Returns:
            PolicyStats with current policy metrics
        """
        # Update statistics
        self.stats.q_table_size = len(self.q_table)
        self.stats.state_count = len(self.q_table) + len(self.policy)

        # Compute average reward
        if self.episode_rewards:
            self.stats.average_reward = sum(self.episode_rewards) / len(self.episode_rewards)

        # Compute policy entropy (for policy gradient methods)
        if self.algorithm == LearningAlgorithm.REINFORCE and self.policy:
            entropies = []
            for state, action_probs in self.policy.items():
                if action_probs:
                    probs = list(action_probs.values())
                    entropy = -sum(p * math.log(p + 1e-10) for p in probs if p > 0)
                    entropies.append(entropy)
            self.stats.policy_entropy = sum(entropies) / len(entropies) if entropies else 0.0

        # Compute convergence rate (change in Q-values)
        if len(self.episode_rewards) >= 100:
            recent = self.episode_rewards[-100:]
            older = self.episode_rewards[-200:-100]
            if older:
                self.stats.convergence_rate = sum(recent) / len(recent) - sum(older) / len(older)

        return self.stats

    def save_policy(self, policy_name: str, metadata: Optional[dict[str, Any]] = None) -> Path:
        """Save policy to YAML file.

        Args:
            policy_name: Name of the policy
            metadata: Optional metadata to include

        Returns:
            Path to saved policy file
        """
        # Decide what to save based on algorithm
        if self.algorithm == LearningAlgorithm.Q_LEARNING:
            policy_to_save = self.q_table
        elif self.algorithm == LearningAlgorithm.REINFORCE:
            policy_to_save = self.policy
        else:
            policy_to_save = self.q_table

        # Include statistics in metadata
        full_metadata = {
            "algorithm": self.algorithm.value,
            "exploration_strategy": self.exploration.strategy.value,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "total_steps": self.total_steps,
            "stats": self.get_policy_statistics().to_dict(),
            **(metadata or {}),
        }

        return self.policy_serializer.save_policy(policy_to_save, policy_name, full_metadata)

    def load_policy(self, policy_name: str) -> bool:
        """Load policy from YAML file.

        Args:
            policy_name: Name of the policy to load

        Returns:
            True if policy loaded successfully
        """
        loaded_policy = self.policy_serializer.load_policy(policy_name)

        if loaded_policy is None:
            return False

        # Load into appropriate structure
        if self.algorithm == LearningAlgorithm.Q_LEARNING:
            self.q_table = loaded_policy
        elif self.algorithm == LearningAlgorithm.REINFORCE:
            self.policy = loaded_policy
        else:
            self.q_table = loaded_policy

        logger.info(f"Policy '{policy_name}' loaded successfully")
        return True

    def get_q_table(self) -> dict[Any, dict[Any, float]]:
        """Get current Q-table.

        Returns:
            Copy of current Q-table
        """
        return {state: actions.copy() for state, actions in self.q_table.items()}

    def get_policy(self) -> dict[Any, dict[Any, float]]:
        """Get current policy (for policy gradient methods).

        Returns:
            Copy of current policy
        """
        return {state: actions.copy() for state, actions in self.policy.items()}

    def reset(self) -> None:
        """Reset all learning state.

        Clears Q-table, policy, replay buffer, and statistics.
        """
        self.q_table.clear()
        self.policy.clear()
        self.replay_buffer.clear()
        self.policy_grad_returns.clear()
        self.episode_rewards.clear()
        self.total_steps = 0
        self.stats = PolicyStats(algorithm=self.algorithm)
        logger.info("Enhanced RL Coordinator reset")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"EnhancedRLCoordinator("
            f"algorithm={self.algorithm.value}, "
            f"steps={self.total_steps}, "
            f"states={self.stats.state_count})"
        )


__all__ = [
    # Main coordinator
    "EnhancedRLCoordinator",
    # Strategies and algorithms
    "ExplorationStrategy",
    "LearningAlgorithm",
    # Supporting classes
    "PolicyStats",
    "Experience",
    "ExperienceReplayBuffer",
    "TargetNetwork",
    "RewardShaper",
    "ExplorationStrategyImpl",
    "PolicySerializer",
]
