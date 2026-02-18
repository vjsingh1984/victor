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

"""Framework defaults — one-stop import for external verticals.

Collects generic stage definitions, safety patterns, RL config, and mode
config so that external verticals can bootstrap with::

    from victor.framework.defaults import (
        get_default_stages,
        DefaultSafetyExtension,
        BaseRLConfig,
    )
"""

from __future__ import annotations

# Stages
from victor.framework.defaults.stages import get_default_stages

# Safety
from victor.framework.defaults.safety import DefaultSafetyExtension

# RL (re-export from existing)
from victor.framework.rl.config import (
    BaseRLConfig,
    DEFAULT_ACTIVE_LEARNERS,
    DEFAULT_PATIENCE_MAP,
)

# Mode config (re-export from existing)
from victor.core.mode_config import (
    DEFAULT_MODES,
    DEFAULT_TASK_BUDGETS,
    ModeConfigRegistry,
    ModeDefinition,
    RegistryBasedModeConfigProvider,
)

# Stage definition type (re-export from existing)
from victor.framework.stage_manager import StageDefinition

__all__ = [
    # Stages
    "get_default_stages",
    "StageDefinition",
    # Safety
    "DefaultSafetyExtension",
    # RL
    "BaseRLConfig",
    "DEFAULT_ACTIVE_LEARNERS",
    "DEFAULT_PATIENCE_MAP",
    # Mode config
    "ModeConfigRegistry",
    "ModeDefinition",
    "RegistryBasedModeConfigProvider",
    "DEFAULT_MODES",
    "DEFAULT_TASK_BUDGETS",
]
