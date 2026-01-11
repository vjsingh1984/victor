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

"""Data-Driven Configuration System for Victor Framework.

This package provides centralized, YAML-based configuration management
for modes, RL hooks, capabilities, and teams across all verticals.

Exports:
    ModeConfigRegistry: Registry for mode configurations
    AgentMode: Mode configuration dataclass
    ExplorationLevel: Exploration intensity enum
    EditPermission: Edit permission enum
    RLConfigRegistry: Registry for RL configurations
    RLConfig: RL configuration dataclass

Example:
    from victor.core.config import ModeConfigRegistry, RLConfigRegistry

    # Mode configuration
    mode_registry = ModeConfigRegistry.get_instance()
    mode = mode_registry.get_mode("coding", "plan")
    print(mode.exploration)  # ExplorationLevel.THOROUGH

    # RL configuration
    rl_registry = RLConfigRegistry.get_instance()
    rl_config = rl_registry.get_rl_config("coding")
    print(rl_config.task_type_mappings)
"""

from victor.core.config.mode_config import (
    AgentMode,
    EditPermission,
    ExplorationLevel,
    ModeConfigRegistry,
    VerticalModeConfig,
    create_mode_config_registry,
)
from victor.core.config.rl_config import (
    RLConfig,
    RLConfigRegistry,
)

__all__ = [
    # Mode configuration
    "ModeConfigRegistry",
    "AgentMode",
    "ExplorationLevel",
    "EditPermission",
    "VerticalModeConfig",
    "create_mode_config_registry",
    # RL configuration
    "RLConfigRegistry",
    "RLConfig",
]
