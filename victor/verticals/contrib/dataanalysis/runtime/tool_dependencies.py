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

"""Data-analysis tool dependencies provider.

This module exposes a single provider factory backed by YAML configuration.
Legacy compatibility constants and wrapper provider classes were removed as part
of roadmap deprecation cleanup.
"""

from pathlib import Path

from victor.core.tool_dependency_loader import YAMLToolDependencyProvider

_YAML_PATH = Path(__file__).resolve().parent.parent / "tool_dependencies.yaml"


def get_provider() -> YAMLToolDependencyProvider:
    """Entry-point provider factory for the dataanalysis vertical."""
    return YAMLToolDependencyProvider(
        yaml_path=_YAML_PATH,
        canonicalize=False,
    )


__all__ = ["get_provider"]
