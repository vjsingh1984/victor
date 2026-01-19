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

"""Framework-level mode configuration for verticals.

This module provides the BaseVerticalModeProvider which eliminates ~400 lines
of duplication across verticals by leveraging the centralized VerticalModeDefaults
from victor.core.mode_config.

Verticals should inherit from BaseVerticalModeProvider and only override if they
need custom behavior beyond the defaults.

Example:
    from victor.framework.modes import BaseVerticalModeProvider

    class CodingModeProvider(BaseVerticalModeProvider):
        def __init__(self):
            super().__init__("coding")
            # That's it! All modes are auto-registered from VerticalModeDefaults
"""

from __future__ import annotations

from victor.framework.modes.base_vertical_mode_provider import BaseVerticalModeProvider

__all__ = [
    "BaseVerticalModeProvider",
]
