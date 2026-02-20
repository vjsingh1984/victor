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

"""Vertical workflow provider capabilities.

This module provides workflow-related functionality for verticals, including:
- Compute handlers for workflow execution
- Workflow provider integration

Extracted from VerticalBase for SRP compliance.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


class VerticalWorkflowProvider(ABC):
    """Provider of vertical workflow capabilities.

    Handles workflow-related functionality including compute handlers
    and workflow provider integration.

    This is a mix-in class that provides workflow capabilities to verticals
    while maintaining SRP compliance.
    """

    @classmethod
    def get_handlers(cls) -> Dict[str, Any]:
        """Get compute handlers for workflow execution.

        Default implementation auto-imports from victor.{vertical_name}.handlers.
        Override to provide domain-specific handlers for workflow nodes.
        These handlers are registered with the HandlerRegistry during
        vertical integration, replacing the previous import-side-effect
        registration pattern.

        Example:
            # Default implementation (no override needed):
            # For CodingAssistant (name="coding"):
            # - Imports from victor.coding.handlers → HANDLERS
            #
            # Custom override (if needed):
            @classmethod
            def get_handlers(cls) -> Dict[str, Any]:
                from victor.coding.handlers import HANDLERS
                return HANDLERS

        Returns:
            Dict mapping handler name to handler instance. Empty dict if
            handlers module not found.
        """
        # Try auto-import from victor.{vertical_name}.handlers
        try:
            if hasattr(cls, 'name'):
                module_path = f"victor.{cls.name}.handlers"
                module = __import__(module_path, fromlist=["HANDLERS"])
                return getattr(module, "HANDLERS", {})
        except ImportError:
            # Handlers module not found (graceful fallback)
            pass
        return {}

    @classmethod
    def get_workflow_provider(cls) -> Optional[Any]:
        """Get workflow provider for this vertical.

        Default implementation auto-imports from victor.{vertical_name}.workflows.
        Supports multiple import patterns:
        1. Direct module: victor.{vertical}.workflows → {Vertical}WorkflowProvider
        2. Submodule: victor.{vertical}.workflows.provider → {Vertical}WorkflowProvider

        Override to provide vertical-specific workflows.

        Example:
            # Default implementation (no override needed):
            # For CodingAssistant (name="coding"):
            # - Imports CodingWorkflowProvider from victor.coding.workflows
            # - Returns an instance: CodingWorkflowProvider()
            #
            # Custom override (if needed):
            @classmethod
            def get_workflow_provider(cls) -> Optional[Any]:
                from victor.coding.workflows import CodingWorkflowProvider
                return CodingWorkflowProvider()

        Returns:
            Workflow provider instance (WorkflowProviderProtocol) or None if not found
        """
        if not hasattr(cls, 'name'):
            return None

        vertical_name = cls.name.title()
        class_name = f"{vertical_name}WorkflowProvider"

        # Try multiple import patterns
        patterns = [
            # Pattern 1: Direct module (e.g., victor.coding.workflows)
            f"victor.{cls.name}.workflows",
            # Pattern 2: Provider submodule (e.g., victor.coding.workflows.provider)
            f"victor.{cls.name}.workflows.provider",
        ]

        for module_path in patterns:
            try:
                module = __import__(module_path, fromlist=[class_name])
                provider_class = getattr(module, class_name, None)
                if provider_class is not None:
                    # Instantiate the provider class to get an instance
                    # with get_workflow_names() and other instance methods
                    return provider_class()
            except (ImportError, AttributeError):
                continue

        return None


class VerticalWorkflowMixin(VerticalWorkflowProvider):
    """Concrete implementation of vertical workflow capabilities.

    This class provides the default implementation of workflow-related
    functionality. Verticals can inherit from this class or use it as
    a mix-in to get workflow capabilities.
    """

    pass
