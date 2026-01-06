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

"""Formation strategy pattern for multi-agent coordination.

This module implements the Strategy pattern for different team
formation patterns, allowing easy extension without modifying
core coordinator logic (Open/Closed Principle).
"""

from victor.coordination.formations.base import BaseFormationStrategy
from victor.coordination.formations.consensus import ConsensusFormation
from victor.coordination.formations.hierarchical import HierarchicalFormation
from victor.coordination.formations.parallel import ParallelFormation
from victor.coordination.formations.pipeline import PipelineFormation
from victor.coordination.formations.sequential import SequentialFormation

__all__ = [
    "BaseFormationStrategy",
    "SequentialFormation",
    "ParallelFormation",
    "HierarchicalFormation",
    "PipelineFormation",
    "ConsensusFormation",
]
