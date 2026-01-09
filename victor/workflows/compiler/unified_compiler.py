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

"""Pure workflow compiler - NO execution logic.

Compiles YAML workflows into executable StateGraph instances.
This is a stub that delegates to legacy implementation during migration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from victor.workflows.definition import WorkflowDefinition
    from victor.workflows.compiler_protocols import CompiledGraphProtocol, NodeExecutorFactoryProtocol

logger = logging.getLogger(__name__)


class WorkflowCompiler:
    """Pure compiler for YAML workflows.

    Responsibility (SRP):
    - Load YAML from file/string
    - Validate workflow definition
    - Build StateGraph from definition
    - Return CompiledGraphProtocol

    Non-responsibility:
    - Execution (handled by WorkflowExecutor)
    - Caching (handled by decorator/wrapper)
    - Observability (handled by decorator/wrapper)
    - Node execution logic (handled by node executors)

    Design:
    - SRP compliance: ONLY compiles, doesn't execute
    - DIP compliance: Depends on NodeExecutorFactoryProtocol
    - OCP compliance: Extensible via factory.register_executor_type()

    Attributes:
        _yaml_loader: Loads and parses YAML
        _validator: Validates workflow structure
        _node_executor_factory: Creates executor functions

    Example:
        compiler = WorkflowCompiler(
            yaml_loader=yaml_loader,
            validator=validator,
            node_executor_factory=factory,
        )

        compiled = compiler.compile("workflow.yaml")
        result = await compiled.invoke({"input": "data"})
    """

    def __init__(
        self,
        yaml_loader: Any,
        validator: Any,
        node_executor_factory: "NodeExecutorFactoryProtocol",
    ):
        """Initialize the compiler.

        Args:
            yaml_loader: YAMLWorkflowLoader instance
            validator: WorkflowValidator instance
            node_executor_factory: NodeExecutorFactoryProtocol instance
        """
        self._yaml_loader = yaml_loader
        self._validator = validator
        self._node_executor_factory = node_executor_factory

    def compile(
        self,
        source: str,
        *,
        workflow_name: Optional[str] = None,
        validate: bool = True,
    ) -> "CompiledGraphProtocol":
        """Compile a workflow from YAML source.

        Args:
            source: YAML file path or YAML string content
            workflow_name: Name of workflow to compile (for multi-workflow files)
            validate: Whether to validate workflow definition before compilation

        Returns:
            CompiledGraphProtocol: Executable compiled graph

        Raises:
            ValueError: If source is invalid or validation fails
            FileNotFoundError: If YAML file doesn't exist

        Example:
            compiler = WorkflowCompiler(...)
            compiled = compiler.compile("workflow.yaml")
            result = await compiled.invoke({"input": "data"})
        """
        logger.info(f"Compiling workflow: {source[:100]}...")

        # TODO: Implement proper compilation
        # For now, delegate to legacy implementation
        return self._compile_legacy(source, workflow_name=workflow_name, validate=validate)

    def _compile_legacy(
        self,
        source: str,
        *,
        workflow_name: Optional[str] = None,
        validate: bool = True,
    ) -> "CompiledGraphProtocol":
        """Delegate to legacy implementation (temporary stub)."""
        from victor.workflows.yaml_to_graph_compiler import YAMLToStateGraphCompiler

        # Create legacy compiler
        legacy_compiler = YAMLToStateGraphCompiler(
            orchestrator=None,  # Will be set by execution context
            orchestrators=None,  # Will be set by execution context
        )

        # Load YAML
        yaml_def = self._yaml_loader.load(source)

        # Compile using legacy implementation
        return legacy_compiler.compile(
            yaml_def,
            workflow_name=workflow_name,
        )


__all__ = [
    "WorkflowCompiler",
]
