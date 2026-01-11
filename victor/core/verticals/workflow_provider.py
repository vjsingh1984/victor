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

        Override to provide domain-specific handlers for workflow nodes.
        These handlers are registered with the HandlerRegistry during
        vertical integration, replacing the previous import-side-effect
        registration pattern.

        Example:
            @classmethod
            def get_handlers(cls) -> Dict[str, Any]:
                from victor.coding.handlers import HANDLERS
                return HANDLERS

        Returns:
            Dict mapping handler name to handler instance
        """
        return {}

    @classmethod
    def get_workflow_provider(cls) -> Optional[Any]:
        """Get workflow provider for this vertical.

        Override to provide vertical-specific workflows.

        Returns:
            Workflow provider (WorkflowProviderProtocol) or None
        """
        return None


class VerticalWorkflowMixin(VerticalWorkflowProvider):
    """Concrete implementation of vertical workflow capabilities.

    This class provides the default implementation of workflow-related
    functionality. Verticals can inherit from this class or use it as
    a mix-in to get workflow capabilities.
    """

    pass
