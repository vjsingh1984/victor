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

"""Adapter layer — REMOVED (E5 M3).

The following classes have been removed:
- UnifiedWorkflowCompilerAdapter
- CompiledGraphAdapter
- ExecutorResultAdapter

Migration: use the protocol-based compiler API directly::

    from victor.workflows.compiler_protocols import WorkflowCompilerProtocol
    from victor.workflows.services import configure_workflow_services
    from victor.core.container import ServiceContainer

    container = ServiceContainer()
    configure_workflow_services(container, settings)
    compiler = container.get(WorkflowCompilerProtocol)
    result = compiler.compile("workflow.yaml")
"""

_REMOVED_NAMES = {
    "UnifiedWorkflowCompilerAdapter",
    "CompiledGraphAdapter",
    "ExecutorResultAdapter",
    "DeprecationAdapter",
}


def __getattr__(name: str):
    if name in _REMOVED_NAMES:
        raise ImportError(
            f"{name} was removed in E5 M3. "
            "Migrate to victor.workflows.compiler_protocols.WorkflowCompilerProtocol. "
            "See victor/workflows/adapter.py docstring for migration details."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__: list = []
