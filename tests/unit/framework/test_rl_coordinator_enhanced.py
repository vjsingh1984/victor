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

"""Tests for Enhanced RL Coordinator."""

import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from victor.framework.rl.rl_coordinator_enhanced import (
    EnhancedRLCoordinator,
    Experience,
    ExperienceReplayBuffer,
    ExplorationStrategy,
    ExplorationStrategyImpl,
    LearningAlgorithm,
    PolicySerializer,
    PolicyStats,
    RewardShaper,
    TargetNetwork,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_policy_dir():
    """Create temporary directory for policy storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_states():
    """Sample states for testing."""
    return ["state_1", "state_2", "state_3"]


@pytest.fixture
def sample_actions():
    """Sample actions for testing."""
    return ["action_a", "action_b", "action_c"]


@pytest.fixture
def sample_q_table(sample_states, sample_actions):
    """Sample Q-table for testing."""
    return {
        "state_1": {"action_a": 0.5, "action_b": 0.3, "action_c": 0.2},
        "state_2": {"action_a": 0.2, "action_b": 0.8, "action_c": 0.1},
        "state_3": {"action_a": 0.1, "action_b": 0.2, "action_c": 0.9},
    }


# =============================================================================
# Experience Tests
# =============================================================================


class TestExperience:
    """Tests for Experience dataclass."""

    def test_experience_creation(self, sample_states, sample_actions):
        """Test creating an experience."""
        exp = Experience(
            state="state_1",
            action="action_a",
            reward=1.0,
            next_state="state_2",
            done=False,
        )

        assert exp.state == "state_1"
        assert exp.action == "action_a"
        assert exp.reward == 1.0
        assert exp.next_state == "state_2"
        assert exp.done is False
        assert exp.timestamp > 0

    def test_experience_to_tuple(self, sample_states, sample_actions):
        """Test converting experience to tuple."""
        exp = Experience(
            state="state_1",
            action="action_a",
            reward=1.0,
            next_state="state_2",
            done=False,
        )

        tup = exp.to_tuple()
        assert tup == ("state_1", "action_a", 1.0, "state_2", False)


# =============================================================================
# Experience Replay Buffer Tests
# =============================================================================


class TestExperienceReplayBuffer:
    """Tests for ExperienceReplayBuffer."""

    def test_buffer_initialization(self):
        """Test buffer initialization."""
        buffer = ExperienceReplayBuffer(capacity=1000)
        assert len(buffer) == 0
        assert buffer.capacity == 1000
        assert buffer.use_prioritization is False

    def test_add_experience(self, sample_states, sample_actions):
        """Test adding experiences to buffer."""
        buffer = ExperienceReplayBuffer(capacity=10)

        for i in range(5):
            exp = Experience(
                state=f"state_{i}",
                action=f"action_{i}",
                reward=float(i),
                next_state=f"state_{i+1}",
                done=False,
            )
            buffer.add(exp)

        assert len(buffer) == 5

    def test_buffer_capacity_limit(self):
        """Test buffer respects capacity limit."""
        buffer = ExperienceReplayBuffer(capacity=5)

        # Add more than capacity
        for i in range(10):
            exp = Experience(state=f"s{i}", action=f"a{i}", reward=0.0, next_state=None, done=True)
            buffer.add(exp)

        # Should only keep capacity items
        assert len(buffer) == 5

    def test_sample_batch(self, sample_states, sample_actions):
        """Test sampling batch from buffer."""
        buffer = ExperienceReplayBuffer(capacity=100)

        # Add experiences
        for i in range(50):
            exp = Experience(
                state=f"state_{i}",
                action=f"action_{i}",
                reward=float(i),
                next_state=f"state_{i+1}",
                done=False,
            )
            buffer.add(exp)

        # Sample batch
        batch = buffer.sample(batch_size=10)
        assert len(batch) == 10
        assert all(isinstance(exp, Experience) for exp in batch)

    def test_sample_from_empty_buffer(self):
        """Test sampling from empty buffer."""
        buffer = ExperienceReplayBuffer(capacity=100)
        batch = buffer.sample(batch_size=10)
        assert batch == []

    def test_clear_buffer(self):
        """Test clearing buffer."""
        buffer = ExperienceReplayBuffer(capacity=100)

        for i in range(10):
            exp = Experience(state="s", action="a", reward=0.0, next_state=None, done=True)
            buffer.add(exp)

        assert len(buffer) == 10
        buffer.clear()
        assert len(buffer) == 0


# =============================================================================
# Target Network Tests
# =============================================================================


class TestTargetNetwork:
    """Tests for TargetNetwork."""

    def test_initialization(self):
        """Test target network initialization."""
        target_net = TargetNetwork(update_frequency=1000)
        assert target_net.update_frequency == 1000
        assert target_net.tau == 0.005
        assert target_net.last_update_step == 0
        assert target_net.target_q_table == {}

    def test_should_update(self):
        """Test update scheduling."""
        target_net = TargetNetwork(update_frequency=100)

        # At step 0, since last_update_step is also 0, difference is 0, which is NOT >= 100
        assert target_net.should_update(0) is False  # No update needed yet

        # Force an initial update
        target_net.last_update_step = -100
        assert target_net.should_update(0) is True
        target_net.update({}, step=0)

        # Should not update before 100 steps
        assert target_net.should_update(50) is False
        assert target_net.should_update(99) is False

        # Should update at 100 steps
        assert target_net.should_update(100) is True

    def test_hard_update(self, sample_q_table):
        """Test hard update (tau=0)."""
        target_net = TargetNetwork(update_frequency=10, tau=0.0)

        # Force update by setting last_update_step back
        target_net.last_update_step = -100

        target_net.update(sample_q_table, step=0)

        assert target_net.target_q_table == sample_q_table
        assert target_net.last_update_step == 0

    def test_soft_update(self, sample_q_table):
        """Test soft update (tau>0)."""
        target_net = TargetNetwork(update_frequency=10, tau=0.5)

        # Force update by setting last_update_step
        target_net.last_update_step = -100

        # Initialize with some values
        target_net.target_q_table = {"state_1": {"action_a": 0.2, "action_b": 0.2, "action_c": 0.2}}

        target_net.update(sample_q_table, step=0)

        # Should blend old and new values
        assert "state_1" in target_net.target_q_table
        # With tau=0.5: new_value = 0.5 * 0.5 + 0.5 * 0.2 = 0.35
        expected = 0.5 * 0.5 + 0.5 * 0.2  # 0.35
        actual = target_net.target_q_table["state_1"]["action_a"]
        assert abs(actual - expected) < 0.01  # Allow floating point tolerance

    def test_get_q_value(self):
        """Test getting Q-value from target network."""
        target_net = TargetNetwork()
        target_net.target_q_table = {"state_1": {"action_a": 0.5}}

        assert target_net.get_q_value("state_1", "action_a") == 0.5
        assert target_net.get_q_value("unknown_state", "unknown_action") == 0.0


# =============================================================================
# Reward Shaper Tests
# =============================================================================


class TestRewardShaper:
    """Tests for RewardShaper."""

    def test_initialization(self):
        """Test reward shaper initialization."""
        shaper = RewardShaper(gamma=0.99)
        assert shaper.gamma == 0.99
        assert shaper.potential_function is not None

    def test_custom_potential_function(self):
        """Test custom potential function."""

        def custom_potential(state):
            return -abs(state)  # Negative distance to zero

        shaper = RewardShaper(gamma=0.99, potential_function=custom_potential)
        assert shaper.potential_function == custom_potential

    def test_shape_reward(self):
        """Test reward shaping."""

        def distance_potential(state):
            return -abs(state)  # Negative distance to zero

        shaper = RewardShaper(gamma=0.99, potential_function=distance_potential)

        # State gets closer to goal (potential increases from -10 to -5)
        shaped = shaper.shape_reward(state=-10, action="move", reward=0.0, next_state=-5)

        # Should add positive bonus (Phi(next) - Phi(current) = -5 - (-10) = 5, scaled by gamma)
        expected_bonus = 0.99 * (-5) - (-10)  # = -4.95 + 10 = 5.05
        assert shaped > 0.0
        assert abs(shaped - expected_bonus) < 0.01

    def test_potential_based_preserves_optimality(self):
        """Test that potential-based shaping preserves optimal policy."""

        def potential(state):
            return -abs(state)

        shaper = RewardShaper(gamma=0.99, potential_function=potential)

        # Shaping term: gamma * Phi(s') - Phi(s)
        reward = 1.0
        shaped = shaper.shape_reward(state=-10, action="move", reward=reward, next_state=-5)

        # The difference should be the shaping bonus
        phi_s = potential(-10)
        phi_s_prime = potential(-5)
        expected_bonus = 0.99 * phi_s_prime - phi_s
        assert abs(shaped - (reward + expected_bonus)) < 1e-6


# =============================================================================
# Exploration Strategy Tests
# =============================================================================


class TestExplorationStrategyImpl:
    """Tests for ExplorationStrategyImpl."""

    def test_epsilon_greedy_initialization(self):
        """Test epsilon-greedy initialization."""
        strategy = ExplorationStrategyImpl(
            strategy=ExplorationStrategy.EPSILON_GREEDY,
            initial_epsilon=1.0,
            min_epsilon=0.01,
            decay_rate=0.995,
        )

        assert strategy.strategy == ExplorationStrategy.EPSILON_GREEDY
        assert strategy.epsilon == 1.0
        assert strategy.min_epsilon == 0.01
        assert strategy.decay_rate == 0.995

    def test_epsilon_decay(self):
        """Test epsilon decay."""
        strategy = ExplorationStrategyImpl(
            strategy=ExplorationStrategy.EPSILON_GREEDY,
            initial_epsilon=1.0,
            decay_rate=0.9,
        )

        initial_epsilon = strategy.epsilon
        strategy.decay_epsilon()
        assert strategy.epsilon == initial_epsilon * 0.9

        # Decay until minimum
        for _ in range(100):
            strategy.decay_epsilon()
        assert strategy.epsilon >= strategy.min_epsilon

    def test_epsilon_greedy_selection(self):
        """Test epsilon-greedy action selection."""
        strategy = ExplorationStrategyImpl(
            strategy=ExplorationStrategy.EPSILON_GREEDY, initial_epsilon=0.0
        )  # Pure greedy

        q_values = {"action_a": 0.8, "action_b": 0.5, "action_c": 0.3}
        available = ["action_a", "action_b", "action_c"]

        # Should select best action
        action = strategy.select_action(q_values, available)
        assert action == "action_a"

    def test_epsilon_greedy_explore(self):
        """Test epsilon-greedy exploration."""
        strategy = ExplorationStrategyImpl(
            strategy=ExplorationStrategy.EPSILON_GREEDY, initial_epsilon=1.0
        )  # Pure explore

        q_values = {"action_a": 0.8, "action_b": 0.5}
        available = ["action_a", "action_b"]

        # Should select random action
        action = strategy.select_action(q_values, available)
        assert action in available

    def test_ucb_selection(self):
        """Test UCB action selection."""
        strategy = ExplorationStrategyImpl(strategy=ExplorationStrategy.UCB, ucb_c=2.0)

        q_values = {"action_a": 0.5, "action_b": 0.5}
        available = ["action_a", "action_b"]

        # First selection should favor least-visited action
        action1 = strategy.select_action(q_values, available, step=10)
        action2 = strategy.select_action(q_values, available, step=10)

        # Both should be selected (different counts)
        assert action1 in available
        assert action2 in available

    def test_boltzmann_selection(self):
        """Test Boltzmann (softmax) action selection."""
        strategy = ExplorationStrategyImpl(strategy=ExplorationStrategy.BOLTZMANN, temperature=1.0)

        q_values = {"action_a": 10.0, "action_b": 0.0, "action_c": 0.0}
        available = ["action_a", "action_b", "action_c"]

        # High Q-value should get higher probability
        # but randomness means any action could be selected
        action = strategy.select_action(q_values, available)
        assert action in available

    def test_entropy_bonus_selection(self):
        """Test entropy bonus action selection."""
        strategy = ExplorationStrategyImpl(strategy=ExplorationStrategy.ENTROPY_BONUS)

        q_values = {"action_a": 0.5, "action_b": 0.5}
        available = ["action_a", "action_b"]

        action = strategy.select_action(q_values, available)
        assert action in available


# =============================================================================
# Policy Serializer Tests
# =============================================================================


class TestPolicySerializer:
    """Tests for PolicySerializer."""

    def test_initialization(self, temp_policy_dir):
        """Test serializer initialization."""
        serializer = PolicySerializer(policy_dir=temp_policy_dir)
        assert serializer.policy_dir == temp_policy_dir
        assert serializer.policy_dir.exists()

    def test_save_policy(self, temp_policy_dir, sample_q_table):
        """Test saving policy to YAML."""
        serializer = PolicySerializer(policy_dir=temp_policy_dir)

        path = serializer.save_policy(sample_q_table, "test_policy")

        assert path.exists()
        assert path.name == "test_policy.yaml"

    def test_save_and_load_policy(self, temp_policy_dir, sample_q_table):
        """Test saving and loading policy."""
        serializer = PolicySerializer(policy_dir=temp_policy_dir)

        # Save
        serializer.save_policy(sample_q_table, "test_policy")

        # Load
        loaded = serializer.load_policy("test_policy")

        assert loaded is not None
        assert isinstance(loaded, dict)
        # Keys are converted to strings
        assert "state_1" in loaded

    def test_load_nonexistent_policy(self, temp_policy_dir):
        """Test loading non-existent policy."""
        serializer = PolicySerializer(policy_dir=temp_policy_dir)
        loaded = serializer.load_policy("nonexistent")
        assert loaded is None

    def test_list_policies(self, temp_policy_dir, sample_q_table):
        """Test listing policies."""
        serializer = PolicySerializer(policy_dir=temp_policy_dir)

        serializer.save_policy(sample_q_table, "policy_1")
        serializer.save_policy(sample_q_table, "policy_2")

        policies = serializer.list_policies()
        assert "policy_1" in policies
        assert "policy_2" in policies
        assert len(policies) == 2


# =============================================================================
# Enhanced RL Coordinator Tests
# =============================================================================


class TestEnhancedRLCoordinator:
    """Tests for EnhancedRLCoordinator."""

    def test_initialization_q_learning(self):
        """Test coordinator initialization with Q-learning."""
        coord = EnhancedRLCoordinator(
            algorithm=LearningAlgorithm.Q_LEARNING,
            learning_rate=0.1,
            gamma=0.99,
        )

        assert coord.algorithm == LearningAlgorithm.Q_LEARNING
        assert coord.learning_rate == 0.1
        assert coord.gamma == 0.99
        assert coord.q_table == {}
        assert coord.policy == {}

    def test_initialization_reinforce(self):
        """Test coordinator initialization with REINFORCE."""
        coord = EnhancedRLCoordinator(
            algorithm=LearningAlgorithm.REINFORCE,
            learning_rate=0.01,
        )

        assert coord.algorithm == LearningAlgorithm.REINFORCE
        assert coord.policy == {}

    def test_select_action_q_learning(self):
        """Test action selection with Q-learning."""
        coord = EnhancedRLCoordinator(
            algorithm=LearningAlgorithm.Q_LEARNING,
            exploration_strategy=ExplorationStrategy.EPSILON_GREEDY,
        )

        # Set initial Q-values
        coord.q_table = {"state_1": {"action_a": 0.8, "action_b": 0.2}}

        action = coord.select_action("state_1", ["action_a", "action_b"])
        assert action in ["action_a", "action_b"]

    def test_select_action_unknown_state(self):
        """Test action selection for unknown state."""
        coord = EnhancedRLCoordinator(algorithm=LearningAlgorithm.Q_LEARNING)

        action = coord.select_action("unknown_state", ["action_a", "action_b"])
        assert action in ["action_a", "action_b"]

    def test_select_action_no_actions(self):
        """Test action selection with no available actions."""
        coord = EnhancedRLCoordinator(algorithm=LearningAlgorithm.Q_LEARNING)

        with pytest.raises(ValueError, match="No available actions"):
            coord.select_action("state_1", [])

    def test_update_policy_q_learning(self):
        """Test policy update with Q-learning."""
        coord = EnhancedRLCoordinator(
            algorithm=LearningAlgorithm.Q_LEARNING,
            learning_rate=0.1,
            gamma=0.99,
        )

        coord.update_policy(
            reward=1.0,
            state="state_1",
            action="action_a",
            next_state="state_2",
            done=False,
        )

        assert "state_1" in coord.q_table
        assert "action_a" in coord.q_table["state_1"]
        assert coord.q_table["state_1"]["action_a"] > 0.0

    def test_update_policy_reinforce(self):
        """Test policy update with REINFORCE."""
        coord = EnhancedRLCoordinator(
            algorithm=LearningAlgorithm.REINFORCE,
            learning_rate=0.1,
        )

        # Add multiple rewards
        coord.update_policy(0.5, "state_1", "action_a")
        coord.update_policy(0.7, "state_1", "action_a")
        coord.update_policy(1.0, "state_1", "action_a", done=True)

        assert "state_1" in coord.policy

    def test_compute_reward_from_success(self):
        """Test reward computation from successful outcome."""
        coord = EnhancedRLCoordinator(algorithm=LearningAlgorithm.Q_LEARNING)

        outcome = MagicMock()
        outcome.success = True
        outcome.quality_score = 0.9
        outcome.metadata = {"tools_used": 5}
        outcome.duration_seconds = 30

        reward = coord.compute_reward(outcome)
        assert reward > 0.0  # Should be positive for success

    def test_compute_reward_from_failure(self):
        """Test reward computation from failed outcome."""
        coord = EnhancedRLCoordinator(algorithm=LearningAlgorithm.Q_LEARNING)

        outcome = MagicMock()
        outcome.success = False
        outcome.quality_score = 0.2
        outcome.metadata = {"tools_used": 20}
        outcome.duration_seconds = 300

        reward = coord.compute_reward(outcome)
        assert reward < 0.5  # Should be low for failure

    def test_get_policy_statistics(self):
        """Test getting policy statistics."""
        coord = EnhancedRLCoordinator(algorithm=LearningAlgorithm.Q_LEARNING)

        # Perform some updates and actions
        for i in range(10):
            state = f"state_{i % 3}"
            action = coord.select_action(state, [f"action_{i % 2}", f"action_{(i+1) % 2}"])
            coord.update_policy(
                reward=1.0,
                state=state,
                action=action,
                next_state=f"state_{(i+1) % 3}",
            )

        stats = coord.get_policy_statistics()
        assert isinstance(stats, PolicyStats)
        assert stats.total_updates == 10
        assert stats.state_count > 0
        assert stats.action_count == 10

    def test_save_and_load_policy(self, temp_policy_dir):
        """Test saving and loading policy."""
        coord = EnhancedRLCoordinator(
            algorithm=LearningAlgorithm.Q_LEARNING,
            policy_dir=temp_policy_dir,
        )

        # Learn something
        coord.update_policy(1.0, "state_1", "action_a", "state_2")

        # Save
        path = coord.save_policy("test_policy")
        assert path.exists()

        # Create new coordinator and load
        coord2 = EnhancedRLCoordinator(
            algorithm=LearningAlgorithm.Q_LEARNING,
            policy_dir=temp_policy_dir,
        )
        success = coord2.load_policy("test_policy")

        assert success is True
        assert "state_1" in coord2.q_table

    def test_reset(self):
        """Test resetting coordinator."""
        coord = EnhancedRLCoordinator(algorithm=LearningAlgorithm.Q_LEARNING)

        # Add some data
        coord.update_policy(1.0, "state_1", "action_a", "state_2")
        coord.update_policy(0.5, "state_2", "action_b", "state_3")

        assert len(coord.q_table) > 0
        assert coord.total_steps > 0

        # Reset
        coord.reset()

        assert len(coord.q_table) == 0
        assert coord.total_steps == 0
        assert len(coord.episode_rewards) == 0

    def test_get_q_table(self):
        """Test getting Q-table."""
        coord = EnhancedRLCoordinator(algorithm=LearningAlgorithm.Q_LEARNING)

        coord.q_table = {"state_1": {"action_a": 0.5}}
        q_table = coord.get_q_table()

        assert q_table == coord.q_table
        # Should be a copy
        q_table["state_2"] = {}
        assert "state_2" not in coord.q_table

    def test_get_policy(self):
        """Test getting policy (REINFORCE)."""
        coord = EnhancedRLCoordinator(algorithm=LearningAlgorithm.REINFORCE)

        coord.policy = {"state_1": {"action_a": 0.7}}
        policy = coord.get_policy()

        assert policy == coord.policy
        # Should be a copy
        policy["state_2"] = {}
        assert "state_2" not in coord.policy

    def test_experience_replay_integration(self):
        """Test experience replay integration."""
        coord = EnhancedRLCoordinator(
            algorithm=LearningAlgorithm.Q_LEARNING,
            replay_buffer_size=100,
        )

        # Add many experiences
        for i in range(50):
            coord.update_policy(
                reward=0.1,
                state=f"state_{i}",
                action=f"action_{i % 3}",
                next_state=f"state_{i+1}",
            )

        assert len(coord.replay_buffer) == 50

    def test_target_network_integration(self):
        """Test target network integration."""
        coord = EnhancedRLCoordinator(
            algorithm=LearningAlgorithm.Q_LEARNING,
            target_network_update_freq=10,
        )

        # Update 10 times
        for i in range(10):
            coord.update_policy(0.5, "state_1", "action_a", "state_2")

        # Target network should have been updated
        assert coord.target_network.last_update_step >= 0

    def test_reward_shaping_integration(self):
        """Test reward shaping integration."""
        coord = EnhancedRLCoordinator(
            algorithm=LearningAlgorithm.Q_LEARNING,
            use_reward_shaping=True,
        )

        # Reward shaper should be configured
        assert coord.reward_shaper is not None

    def test_exploration_decay(self):
        """Test exploration rate decay."""
        coord = EnhancedRLCoordinator(
            algorithm=LearningAlgorithm.Q_LEARNING,
            exploration_strategy=ExplorationStrategy.EPSILON_GREEDY,
        )

        # Set initial epsilon
        coord.exploration.epsilon = 1.0
        coord.exploration.decay_rate = 0.9
        initial_epsilon = coord.exploration.epsilon

        # Perform updates (each calls decay_epsilon)
        for _ in range(5):
            coord.update_policy(0.5, "state_1", "action_a", "state_2")

        # Epsilon should have decayed
        assert coord.exploration.epsilon < initial_epsilon

    def test_repr(self):
        """Test string representation."""
        coord = EnhancedRLCoordinator(algorithm=LearningAlgorithm.Q_LEARNING)

        repr_str = repr(coord)
        assert "EnhancedRLCoordinator" in repr_str
        assert "q_learning" in repr_str


# =============================================================================
# Integration Tests
# =============================================================================


class TestRLIntegration:
    """Integration tests for RL components."""

    def test_full_learning_loop(self):
        """Test complete learning loop."""
        coord = EnhancedRLCoordinator(
            algorithm=LearningAlgorithm.Q_LEARNING,
            learning_rate=0.1,
            gamma=0.99,
            exploration_strategy=ExplorationStrategy.EPSILON_GREEDY,
        )

        # Simulate episodes
        for episode in range(10):
            state = "initial"
            total_reward = 0

            for step in range(5):
                # Select action
                action = coord.select_action(state, ["action_a", "action_b"])

                # Simulate environment
                next_state = f"state_{episode}_{step}"
                reward = 1.0 if action == "action_a" else -0.5

                # Update policy
                coord.update_policy(reward, state, action, next_state, done=(step == 4))

                total_reward += reward
                state = next_state

            # Episode completed
            assert total_reward is not None

        # Check learning occurred
        stats = coord.get_policy_statistics()
        assert stats.total_updates == 50  # 10 episodes * 5 steps

    def test_multi_state_learning(self):
        """Test learning across multiple states."""
        coord = EnhancedRLCoordinator(
            algorithm=LearningAlgorithm.Q_LEARNING,
            learning_rate=0.2,
        )

        states = ["state_1", "state_2", "state_3"]
        actions = ["action_a", "action_b", "action_c"]

        # Train
        for state in states:
            for action in actions:
                reward = 1.0 if action == "action_a" else 0.0
                coord.update_policy(reward, state, action, f"{state}_next")

        # Check learned policy
        for state in states:
            q_values = coord.q_table.get(state, {})
            if q_values:
                best_action = max(q_values, key=q_values.get)
                assert best_action == "action_a"  # Should learn best action

    def test_policy_persistence_roundtrip(self, temp_policy_dir):
        """Test full policy save/load roundtrip."""
        # Train coordinator
        coord1 = EnhancedRLCoordinator(
            algorithm=LearningAlgorithm.Q_LEARNING,
            policy_dir=temp_policy_dir,
        )

        for i in range(20):
            coord1.update_policy(1.0, f"state_{i}", f"action_{i % 3}", f"state_{i+1}")

        # Save policy
        path = coord1.save_policy("saved_policy", metadata={"epochs": 20})
        assert path.exists()

        # Load in new coordinator
        coord2 = EnhancedRLCoordinator(
            algorithm=LearningAlgorithm.Q_LEARNING,
            policy_dir=temp_policy_dir,
        )
        coord2.load_policy("saved_policy")

        # Verify Q-table transferred
        assert len(coord2.q_table) > 0
        assert coord2.q_table == coord1.q_table

    def test_exploration_exploitation_tradeoff(self):
        """Test exploration-exploitation tradeoff."""
        coord = EnhancedRLCoordinator(
            algorithm=LearningAlgorithm.Q_LEARNING,
            exploration_strategy=ExplorationStrategy.EPSILON_GREEDY,
        )

        # Set up exploration parameters manually
        coord.exploration.epsilon = 1.0  # Start with pure exploration
        coord.exploration.decay_rate = 0.95

        # Set Q-values
        coord.q_table = {"state_1": {"action_a": 1.0, "action_b": 0.0}}

        actions_taken = []
        for _ in range(100):
            action = coord.select_action("state_1", ["action_a", "action_b"])
            actions_taken.append(action)
            coord.update_policy(1.0, "state_1", action, "state_2")

        # Should see mix of exploration and exploitation
        # As epsilon decays, should exploit more
        unique_actions = set(actions_taken)
        assert len(unique_actions) == 2  # Both actions tried

        # Later iterations should prefer action_a
        later_actions = actions_taken[-20:]
        assert later_actions.count("action_a") > 15  # Mostly exploiting
