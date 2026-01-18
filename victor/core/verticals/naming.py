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

"""Vertical name normalization and module mapping helpers.

This module centralizes alias handling for vertical names so code can
accept legacy names while keeping canonical identifiers consistent.
"""

from __future__ import annotations

from typing import Dict


_VERTICAL_NAME_ALIASES: Dict[str, str] = {
    "dataanalysis": "data_analysis",
}

_VERTICAL_MODULE_OVERRIDES: Dict[str, str] = {
    "data_analysis": "dataanalysis",
}


def normalize_vertical_name(name: str) -> str:
    """Normalize a vertical name to its canonical form."""
    if not name:
        return name
    normalized = name.strip().lower()
    return _VERTICAL_NAME_ALIASES.get(normalized, normalized)


def get_vertical_module_name(name: str) -> str:
    """Map a canonical vertical name to its module package name."""
    normalized = normalize_vertical_name(name)
    return _VERTICAL_MODULE_OVERRIDES.get(normalized, normalized)
