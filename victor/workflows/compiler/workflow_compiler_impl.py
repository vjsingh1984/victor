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

"""Concrete workflow compiler implementation.

This module provides concrete implementations that can be registered
in the DI container, bridging the gap between protocols and
the legacy implementation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from victor.workflows.compiler_protocols import CompiledGraphProtocol

logger = logging.getLogger(__name__)


class WorkflowCompilerImpl:
    """Concrete implementation of workflow compiler.

    This class wraps the legacy YAMLToStateGraphCompiler and provides
    a protocol-compliant interface that can be registered in the DI
    container.

    During migration, this delegates to the legacy implementation.
    In Phase 8, this will be replaced with pure SOLID implementation.

    Attributes:
        _yaml_loader: Loads and parses YAML workflow definitions
        _validator: Validates workflow structure
        _node_factory: Creates executor functions for nodes

    Example:
        compiler = WorkflowCompilerImpl(yaml_loader, validator, node_factory)
        compiled = compiler.compile("workflow.yaml")
        result = await compiled.invoke({"input": "data"})
    """

    def __init__(
        self,
        yaml_loader: Any,
        validator: Any,
        node_factory: Any,
    ):
        """Initialize the compiler.

        Args:
            yaml_loader: YAMLWorkflowLoader instance
            validator: WorkflowValidator instance
            node_factory: NodeExecutorFactoryProtocol instance
        """
        self._yaml_loader = yaml_loader
        self._validator = validator
        self._node_factory = node_factory

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
            compiler = WorkflowCompilerImpl(...)
            compiled = compiler.compile("workflow.yaml")
            result = await compiled.invoke({"input": "data"})
        """
        logger.info(f"Compiling workflow: {source[:100]}...")

        # For now, delegate to legacy implementation
        return self._compile_legacy(source, workflow_name=workflow_name, validate=validate)

    def _compile_legacy(
        self,
        source: str,
        *,
        workflow_name: Optional[str] = None,
        validate: bool = True,
    ) -> "CompiledGraphProtocol":
        """Delegate to legacy implementation (temporary).

        This will be replaced with pure SOLID implementation in Phase 8.
        """
        from victor.workflows.yaml_to_graph_compiler import YAMLToStateGraphCompiler

        # Create legacy compiler
        legacy_compiler = YAMLToStateGraphCompiler(
            orchestrator=None,  # Will be set by execution context
            orchestrators=None,  # Will be set by execution context
        )

        # Load YAML
        yaml_def = self._yaml_loader.load(source)

        # Validate if requested
        if validate and self._validator:
            self._validator.validate(yaml_def)

        # Compile using legacy implementation
        return legacy_compiler.compile(yaml_def, workflow_name=workflow_name)


__all__ = [
    "WorkflowCompilerImpl",
]
