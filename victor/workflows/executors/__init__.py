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

"""Node executors package.

Provides specialized executor classes for each workflow node type.
Each executor handles ONE node type (SRP compliance).

Executors:
- AgentNodeExecutor: Handles agent nodes (LLM + tools)
- ComputeNodeExecutor: Handles compute nodes (direct function calls)
- TransformNodeExecutor: Handles transform nodes (state transformations)
- ParallelNodeExecutor: Handles parallel nodes (concurrent execution)
- ConditionNodeExecutor: Handles condition nodes (branching logic)
"""

from victor.workflows.executors.factory import NodeExecutorFactory

__all__ = [
    "NodeExecutorFactory",
]
