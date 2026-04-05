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

"""Backward-compatible concrete compiler facade for DI registration."""

from __future__ import annotations

from typing import Any

from victor.workflows.compiler.unified_compiler import WorkflowCompiler


class WorkflowCompilerImpl(WorkflowCompiler):
    """Concrete compatibility wrapper around the shared compiler facade."""

    def __init__(
        self,
        yaml_loader: Any,
        validator: Any,
        node_factory: Any,
    ) -> None:
        """Initialize the backward-compatible compiler wrapper.

        Args:
            yaml_loader: YAMLWorkflowLoader instance
            validator: WorkflowValidator instance
            node_factory: NodeExecutorFactoryProtocol instance
        """
        super().__init__(
            yaml_loader=yaml_loader,
            validator=validator,
            node_executor_factory=node_factory,
        )


__all__ = [
    "WorkflowCompilerImpl",
]
