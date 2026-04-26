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

"""Deprecated shim for provider switch coordination.

The canonical implementation now lives under ``victor.agent.provider``.
This module remains as a public compatibility surface for legacy imports.
"""

from __future__ import annotations

import warnings

from victor.agent.provider.switch_coordinator import (
    HookPriority,
    PostSwitchHook,
    ProviderSwitchCoordinator,
    RegisteredHook,
    SwitchContext,
    create_provider_switch_coordinator,
)

warnings.warn(
    "victor.agent.provider_switch_coordinator is deprecated. "
    "Import provider switch APIs from victor.agent.provider.switch_coordinator instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "ProviderSwitchCoordinator",
    "create_provider_switch_coordinator",
    "PostSwitchHook",
    "SwitchContext",
    "HookPriority",
    "RegisteredHook",
]
