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

"""Agent self-improvement module.

This module provides self-improvement capabilities for Victor agents,
including proficiency tracking and reinforcement learning.

Key Components:
- ProficiencyTracker: Track tool and task performance over time
- RLCoordinator: Enhanced RL coordinator with reward shaping and policy optimization

Usage:
    from victor.agent.improvement import ProficiencyTracker

    tracker = ProficiencyTracker()
    tracker.record_outcome(
        task="code_review",
        tool="ast_analyzer",
        outcome=TaskOutcome(success=True, duration=1.5, cost=0.001)
    )
"""

from __future__ import annotations

from victor.agent.improvement.proficiency_tracker import (
    ImprovementTrajectory,
    MovingAverageMetrics,
    ProficiencyMetrics,
    ProficiencyScore,
    ProficiencyTracker,
    Suggestion,
    TaskOutcome,
    TrendDirection,
)
from victor.agent.improvement.rl_coordinator import (
    Action,
    EnhancedRLCoordinator,
    Hyperparameters,
    Policy,
    Reward,
    RewardShapingStrategy,
)

__all__ = [
    # Proficiency tracking
    "ProficiencyTracker",
    "ProficiencyScore",
    "TaskOutcome",
    "Suggestion",
    "ProficiencyMetrics",
    "TrendDirection",
    "ImprovementTrajectory",
    "MovingAverageMetrics",
    # Enhanced RL coordinator
    "EnhancedRLCoordinator",
    "Reward",
    "RewardShapingStrategy",
    "Policy",
    "Hyperparameters",
    "Action",
]
