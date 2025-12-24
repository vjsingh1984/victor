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

"""Framework-level reinforcement learning infrastructure.

This module provides centralized RL coordination, unified SQLite storage,
and cross-vertical learning capabilities.

Key Components:
- RLCoordinator: Central coordinator for all learners
- BaseLearner: Abstract base class for all RL learners
- Unified SQLite database: ~/.victor/rl_data/rl.db

Architecture:
┌──────────────────────────────────────────────┐
│          RLCoordinator (Singleton)           │
│  ├─ Learner registry                         │
│  ├─ Unified SQLite storage                   │
│  ├─ Telemetry collection                     │
│  └─ Cross-vertical learning                  │
└────────────────┬─────────────────────────────┘
                 │ manages
                 ▼
┌──────────────────────────────────────────────┐
│          Specialized Learners                 │
│  ├─ ContinuationPatienceLearner              │
│  ├─ ContinuationPromptLearner                │
│  ├─ SemanticThresholdLearner                 │
│  ├─ ModelSelectorLearner                     │
│  ├─ QualityThresholdLearner                  │
│  └─ GroundingStrictnessLearner               │
└──────────────────────────────────────────────┘

Usage:
    from victor.agent.rl.coordinator import get_rl_coordinator

    coordinator = get_rl_coordinator()

    # Record outcome
    coordinator.record_outcome(
        learner_name="continuation_patience",
        outcome=RLOutcome(...),
        vertical="coding",
    )

    # Get recommendation
    recommendation = coordinator.get_recommendation(
        learner_name="continuation_patience",
        provider="deepseek",
        model="deepseek-chat",
        task_type="analysis",
    )
"""

from victor.agent.rl.base import BaseLearner, RLOutcome, RLRecommendation
from victor.agent.rl.coordinator import RLCoordinator, get_rl_coordinator

__all__ = [
    "BaseLearner",
    "RLOutcome",
    "RLRecommendation",
    "RLCoordinator",
    "get_rl_coordinator",
]
