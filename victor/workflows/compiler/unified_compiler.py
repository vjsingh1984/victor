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

"""Workflow compiler facade with explicit parse/validate/compile stages."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from victor.workflows.compiler.boundary import (
    NativeWorkflowGraphCompiler,
    WorkflowCompilationRequest,
    WorkflowDefinitionValidator,
    WorkflowParser,
)

if TYPE_CHECKING:
    from victor.workflows.compiler_protocols import (
        CompiledGraphProtocol,
        NodeExecutorFactoryProtocol,
    )

logger = logging.getLogger(__name__)


class WorkflowCompiler:
    """DI-facing compiler with explicit parsing and validation boundaries.

    Responsibility (SRP):
    - Normalize source requests into parsed workflow definitions
    - Validate workflow definitions through a dedicated validation stage
    - Build executable graphs from validated definitions
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
        _graph_compiler: Compiles validated definitions into executable graphs

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
        workflow_parser: Optional[WorkflowParser] = None,
        workflow_definition_validator: Optional[WorkflowDefinitionValidator] = None,
        graph_compiler: Optional[Any] = None,
    ):
        """Initialize the compiler.

        Args:
            yaml_loader: YAMLWorkflowLoader instance
            validator: WorkflowValidator instance
            node_executor_factory: NodeExecutorFactoryProtocol instance
            workflow_parser: Optional parser override for testing/injection
            workflow_definition_validator: Optional validator stage override
            graph_compiler: Optional compiler backend override
        """
        self._yaml_loader = yaml_loader
        self._validator = validator
        self._node_executor_factory = node_executor_factory
        self._workflow_parser = workflow_parser or WorkflowParser(yaml_loader)
        self._workflow_definition_validator = (
            workflow_definition_validator or WorkflowDefinitionValidator(validator)
        )
        self._graph_compiler = graph_compiler or NativeWorkflowGraphCompiler(
            node_executor_factory=node_executor_factory
        )

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
        request = WorkflowCompilationRequest(
            source=source,
            workflow_name=workflow_name,
            validate=validate,
        )
        parsed = self._workflow_parser.parse(request)

        if validate:
            parsed = self._workflow_definition_validator.validate(parsed)

        return self._graph_compiler.compile(parsed)


__all__ = [
    "WorkflowCompiler",
]
