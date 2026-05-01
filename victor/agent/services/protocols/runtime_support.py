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

"""Service-owned aliases for migrated agent runtime protocol surfaces.

These aliases keep protocol identity stable while moving the canonical host for
active agent runtime imports under ``victor.agent.services.protocols``.

Legacy imports from ``victor.agent.protocols`` remain supported for backward
compatibility, but new service-first runtime code should prefer these names.
"""

from victor.agent.protocols.coordination_protocols import (
    CoordinationAdvisorRuntimeProtocol,
    PromptCoordinatorProtocol as PromptRuntimeProtocol,
    StateCoordinatorProtocol as StateRuntimeProtocol,
    TaskCoordinatorProtocol as TaskRuntimeProtocol,
    ToolPlannerProtocol as ToolPlanningRuntimeProtocol,
)
from victor.agent.protocols.infrastructure_protocols import (
    RLCoordinatorProtocol as RLLearningRuntimeProtocol,
)
from victor.agent.protocols.streaming_protocols import (
    ChunkGeneratorProtocol as ChunkRuntimeProtocol,
    StreamingRecoveryCoordinatorProtocol as StreamingRecoveryRuntimeProtocol,
)

__all__ = [
    "ChunkRuntimeProtocol",
    "CoordinationAdvisorRuntimeProtocol",
    "PromptRuntimeProtocol",
    "RLLearningRuntimeProtocol",
    "StateRuntimeProtocol",
    "StreamingRecoveryRuntimeProtocol",
    "TaskRuntimeProtocol",
    "ToolPlanningRuntimeProtocol",
]
