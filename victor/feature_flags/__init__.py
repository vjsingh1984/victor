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

"""Feature flag system for gradual rollout of new capabilities.

This module provides a comprehensive feature flag system for controlling
the availability of new Victor 0.5.0 capabilities, enabling gradual rollout,
A/B testing, and instant rollback without deployment.

Key Features:
- Environment-based flag resolution (VICTOR_FEATURE_* env vars)
- Settings file integration (settings.feature_flags.*)
- Runtime updates (hot-reload without restart)
- Flag validation and dependency checking
- Audit logging for all flag changes
- Integration with observability infrastructure

Example:
    from victor.feature_flags import FeatureFlagManager, get_feature_flag_manager

    # Get singleton manager
    manager = get_feature_flag_manager()

    # Check if feature is enabled
    if manager.is_enabled("hierarchical_planning_enabled"):
        # Use hierarchical planning
        pass

    # Set flag at runtime
    manager.set_flag("enhanced_memory_enabled", True)

    # Get all flags
    flags = manager.get_all_flags()
"""

from __future__ import annotations

from victor.feature_flags.flags import FEATURE_FLAGS
from victor.feature_flags.manager import FeatureFlagManager, get_feature_flag_manager
from victor.feature_flags.resolvers import (
    FlagResolver,
    EnvironmentFlagResolver,
    SettingsFlagResolver,
    RuntimeFlagResolver,
    ChainedFlagResolver,
)

__all__ = [
    # Flag definitions
    "FEATURE_FLAGS",
    # Manager
    "FeatureFlagManager",
    "get_feature_flag_manager",
    # Resolvers
    "FlagResolver",
    "EnvironmentFlagResolver",
    "SettingsFlagResolver",
    "RuntimeFlagResolver",
    "ChainedFlagResolver",
]
