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

Compiles YAML workflows into executable CompiledGraph instances.
This is a pure compiler following SRP - ONLY compiles, NEVER executes.

Architecture:
    YAML File → YAMLWorkflowLoader → WorkflowDefinition → WorkflowDefinitionCompiler → CompiledGraph

The CompiledGraph can then be executed by StateGraphExecutor.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from victor.workflows.definition import WorkflowDefinition
    from victor.workflows.compiler_protocols import (
        CompiledGraphProtocol,
        NodeExecutorFactoryProtocol,
    )

logger = logging.getLogger(__name__)


class WorkflowCompiler:
    """Pure compiler for YAML workflows.

    Responsibility (SRP):
    - Load YAML from file/string
    - Validate workflow definition
    - Build CompiledGraph from definition
    - Return CompiledGraphProtocol

    Non-responsibility:
    - Execution (handled by StateGraphExecutor)
    - Caching (handled by decorator/wrapper)
    - Observability (handled by decorator/wrapper)
    - Node execution logic (handled by node executors)

    Design:
    - SRP compliance: ONLY compiles, doesn't execute
    - DIP compliance: Depends on NodeExecutorFactoryProtocol
    - OCP compliance: Extensible via factory.register_executor_type()

    Attributes:
        _yaml_loader: Loads and parses YAML
        _node_executor_factory: Creates executor functions for nodes

    Example:
        from victor.workflows.yaml_loader import YAMLWorkflowLoader
        from victor.workflows.compiler import WorkflowCompiler

        loader = YAMLWorkflowLoader()
        factory = NodeExecutorFactory(container=container)
        compiler = WorkflowCompiler(
            yaml_loader=loader,
            node_executor_factory=factory,
        )

        compiled = compiler.compile("workflow.yaml")
        # Use executor to run:
        # executor = StateGraphExecutor(compiled)
        # result = await executor.invoke({"input": "data"})
    """

    def __init__(
        self,
        yaml_loader: Any,
        node_executor_factory: "NodeExecutorFactoryProtocol",
        yaml_config: Optional[Any] = None,
    ):
        """Initialize the compiler.

        Args:
            yaml_loader: YAMLWorkflowLoader instance for loading YAML
            node_executor_factory: NodeExecutorFactoryProtocol for creating node executors
            yaml_config: Optional YAMLWorkflowConfig for loading configuration
        """
        self._yaml_loader = yaml_loader
        self._node_executor_factory = node_executor_factory
        self._yaml_config = yaml_config

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
            compiler = WorkflowCompiler(loader, factory)
            compiled = compiler.compile("workflow.yaml")
            # Execute with:
            # executor = StateGraphExecutor(compiled)
            # result = await executor.invoke({"input": "data"})
        """
        logger.info(f"Compiling workflow: {source[:100]}...")

        # Load YAML to WorkflowDefinition
        workflow_def = self._load_workflow(source, workflow_name)

        # Validate if requested
        if validate:
            errors = workflow_def.validate()
            if errors:
                raise ValueError(f"Workflow validation failed: {'; '.join(errors)}")

        # Compile to CompiledGraph using WorkflowDefinitionCompiler
        return self._compile_to_graph(workflow_def)

    def _load_workflow(
        self,
        source: str,
        workflow_name: Optional[str] = None,
    ) -> "WorkflowDefinition":
        """Load YAML source into WorkflowDefinition.

        Args:
            source: YAML file path or YAML string content
            workflow_name: Name of workflow to load

        Returns:
            WorkflowDefinition object

        Raises:
            FileNotFoundError: If file path doesn't exist
        """
        # Check if source is a file path
        if isinstance(source, str) and len(source) < 500 and not source.startswith((" ", "\t", "\n", "{", "[")):
            path = Path(source)
            if path.exists():
                # Load from file
                logger.debug(f"Loading workflow from file: {source}")
                from victor.workflows.yaml_loader import load_workflow_from_file

                result = load_workflow_from_file(
                    str(source),
                    workflow_name,
                    config=self._yaml_config,
                )
            else:
                # Treat as YAML content string
                logger.debug(f"Loading workflow from YAML string (path doesn't exist): {source[:50]}...")
                from victor.workflows.yaml_loader import load_workflow_from_yaml

                result = load_workflow_from_yaml(
                    source,
                    workflow_name,
                    config=self._yaml_config,
                )
        else:
            # Treat as YAML content string
            logger.debug(f"Loading workflow from YAML string: {source[:50]}...")
            from victor.workflows.yaml_loader import load_workflow_from_yaml

            result = load_workflow_from_yaml(
                source,
                workflow_name,
                config=self._yaml_config,
            )

        # Handle dict or single workflow result
        if isinstance(result, dict):
            if not result:
                raise ValueError(f"No workflows found in source")
            # Return the requested workflow or first one
            if workflow_name and workflow_name in result:
                return result[workflow_name]
            return next(iter(result.values()))
        else:
            return result

    def _compile_to_graph(
        self,
        workflow_def: "WorkflowDefinition",
    ) -> "CompiledGraphProtocol":
        """Compile WorkflowDefinition to CompiledGraph.

        Uses WorkflowDefinitionCompiler to convert the workflow definition
        into an executable CompiledGraph.

        Args:
            workflow_def: The workflow definition to compile

        Returns:
            CompiledGraph ready for execution
        """
        from victor.workflows.graph_compiler import WorkflowDefinitionCompiler

        # Create compiler
        compiler = WorkflowDefinitionCompiler(
            runner_registry=getattr(self._node_executor_factory, "_runner_registry", None),
        )

        # Compile to CompiledGraph
        compiled = compiler.compile(workflow_def)

        logger.info(
            f"Compiled workflow '{workflow_def.name}' with "
            f"{len(workflow_def.nodes)} nodes"
        )

        return compiled


__all__ = [
    "WorkflowCompiler",
]
