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

"""Vertical configuration system for YAML-based vertical definitions.

This package provides YAML-based configuration for Victor verticals,
replacing multiple get_* methods with declarative YAML configuration.

Components:
- VerticalConfigLoader: Load and parse YAML vertical configs
- vertical_schema.yaml: YAML schema documentation
"""

from victor.core.verticals.config.vertical_config_loader import VerticalConfigLoader

__all__ = [
    "VerticalConfigLoader",
]
