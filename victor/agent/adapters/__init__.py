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

"""Adapter classes for orchestrator integration.

This package provides adapter classes that bridge the orchestrator and
various components, extracting adapter logic and reducing orchestrator complexity.

Adapters:
- IntelligentPipelineAdapter: Intelligent pipeline integration
- CoordinatorAdapter: Coordinator integration (state, evaluation, checkpoint)
- ResultConverters: Result type conversion utilities

Design Patterns:
- Adapter Pattern: Converts between incompatible interfaces
- Single Responsibility: Each adapter handles one integration
- Dependency Inversion: Adapters depend on protocols, not concrete implementations
"""

from victor.agent.adapters.intelligent_pipeline_adapter import IntelligentPipelineAdapter
from victor.agent.adapters.coordinator_adapter import CoordinatorAdapter
from victor.agent.adapters.result_converters import ResultConverters

__all__ = [
    "IntelligentPipelineAdapter",
    "CoordinatorAdapter",
    "ResultConverters",
]
