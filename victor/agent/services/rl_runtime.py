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

"""Service-owned host for RL runtime access.

The RL implementation remains a first-class public framework API in
``victor.framework.rl.coordinator``. This module provides the canonical
service-first import surface for agent runtime code so the live orchestration
path does not depend on framework module paths directly.
"""

from typing import Optional

from victor.framework.rl.coordinator import (
    AsyncWriterQueue,
    RLCoordinator,
    get_rl_coordinator,
    get_rl_coordinator_async,
    reset_rl_coordinator,
)


def create_prompt_rollout_experiment(
    *,
    section_name: str,
    provider: str,
    treatment_hash: str,
    control_hash: Optional[str] = None,
    traffic_split: float = 0.1,
    min_samples_per_variant: int = 50,
) -> Optional[str]:
    """Create a prompt rollout experiment via the global RL coordinator."""
    coordinator = get_rl_coordinator()
    return coordinator.create_prompt_rollout_experiment(
        section_name=section_name,
        provider=provider,
        treatment_hash=treatment_hash,
        control_hash=control_hash,
        traffic_split=traffic_split,
        min_samples_per_variant=min_samples_per_variant,
    )


async def create_prompt_rollout_experiment_async(
    *,
    section_name: str,
    provider: str,
    treatment_hash: str,
    control_hash: Optional[str] = None,
    traffic_split: float = 0.1,
    min_samples_per_variant: int = 50,
) -> Optional[str]:
    """Create a prompt rollout experiment asynchronously via the global RL coordinator."""
    coordinator = await get_rl_coordinator_async()
    return await coordinator.create_prompt_rollout_experiment_async(
        section_name=section_name,
        provider=provider,
        treatment_hash=treatment_hash,
        control_hash=control_hash,
        traffic_split=traffic_split,
        min_samples_per_variant=min_samples_per_variant,
    )


__all__ = [
    "AsyncWriterQueue",
    "RLCoordinator",
    "create_prompt_rollout_experiment",
    "create_prompt_rollout_experiment_async",
    "get_rl_coordinator",
    "get_rl_coordinator_async",
    "reset_rl_coordinator",
]
