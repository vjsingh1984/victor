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

"""Stable public API surface for vertical base classes.

External verticals should import from this module (or ``victor.framework.*``)
rather than from ``victor.core.verticals.base`` directly.  This provides an
indirection layer that insulates external packages from internal refactoring.

Usage::

    from victor.framework.vertical_base import (
        VerticalBase,
        VerticalConfig,
        StageDefinition,
        StageBuilder,
        register_vertical,
        ExtensionDependency,
    )
"""

from victor.core.verticals.base import VerticalBase, VerticalConfig
from victor.core.vertical_types import StageDefinition, StageBuilder
from victor.core.verticals.registration import register_vertical, ExtensionDependency

__all__ = [
    "VerticalBase",
    "VerticalConfig",
    "StageDefinition",
    "StageBuilder",
    "register_vertical",
    "ExtensionDependency",
]
