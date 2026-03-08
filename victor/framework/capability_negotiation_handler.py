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

"""Capability negotiation step handler for vertical integration pipeline.

This module provides a step handler that negotiates capability versions
between verticals and the orchestrator before applying vertical extensions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from victor.agent.vertical_context import VerticalContext
from victor.framework.capability_negotiation import (
    CompatibilityStrategy,
    NegotiationResult,
    CapabilityNegotiationProtocol,
    negotiate_capabilities,
)
from victor.framework.step_handlers import (
    BaseStepHandler,
    StepHandlerProtocol,
    IntegrationResult,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Capability Negotiation Step Handler
# =============================================================================


class CapabilityNegotiationStepHandler(BaseStepHandler):
    """Step handler for capability negotiation.

    This handler negotiates capability versions between the vertical and
    orchestrator before applying any extensions. This ensures that both
    sides agree on compatible versions and enables graceful degradation
    when version mismatches occur.

    Execution Order: 3 (runs before all other steps)

    The negotiation results are stored in the VerticalContext for later
    use by other step handlers.
    """

    order: int = 3
    """Execution order (must run before capability_config at order 5)."""

    def __init__(
        self,
        strategy: CompatibilityStrategy = CompatibilityStrategy.BACKWARD_COMPATIBLE,
        enable_fallback: bool = True,
        fail_on_incompatible: bool = False,
    ):
        """Initialize capability negotiation handler.

        Args:
            strategy: Compatibility strategy for version negotiation
            enable_fallback: Enable fallback to older versions
            fail_on_incompatible: Fail integration if capabilities are incompatible
        """
        self._strategy = strategy
        self._enable_fallback = enable_fallback
        self._fail_on_incompatible = fail_on_incompatible

    def apply(
        self,
        orchestrator: Any,
        vertical: Any,
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Apply capability negotiation.

        Args:
            orchestrator: The orchestrator being configured
            vertical: The vertical providing extensions
            context: The vertical context with configuration
            result: The integration result to update
        """
        logger.debug(f"Negotiating capabilities for vertical: {context.vertical_name}")

        # Perform negotiation
        negotiation_results = negotiate_capabilities(
            vertical=vertical,
            orchestrator=orchestrator,
            strategy=self._strategy,
            enable_fallback=self._enable_fallback,
        )

        # Store results in context
        context.capability_negotiation_results = negotiation_results

        # Analyze results
        successful_capabilities: List[str] = []
        failed_capabilities: List[str] = []
        fallback_capabilities: List[str] = []

        for capability_name, negotiation_result in negotiation_results.items():
            if negotiation_result.is_success:
                successful_capabilities.append(capability_name)

                if negotiation_result.has_fallback:
                    fallback_capabilities.append(capability_name)
                    logger.info(
                        f"Capability '{capability_name}': using fallback "
                        f"v{negotiation_result.agreed_version} "
                        f"(from v{negotiation_result.fallback_version})"
                    )
                else:
                    logger.debug(
                        f"Capability '{capability_name}': negotiated v{negotiation_result.agreed_version}"
                    )

                # Log feature support
                if negotiation_result.unsupported_features:
                    logger.warning(
                        f"Capability '{capability_name}': unsupported features: "
                        f"{negotiation_result.unsupported_features}"
                    )

                if negotiation_result.missing_required_features:
                    logger.warning(
                        f"Capability '{capability_name}': missing required features: "
                        f"{negotiation_result.missing_required_features}"
                    )
            else:
                failed_capabilities.append(capability_name)
                logger.error(
                    f"Capability '{capability_name}': negotiation failed - "
                    f"{negotiation_result.error}"
                )

        # Update integration result
        result.capability_negotiation_results = negotiation_results

        if successful_capabilities:
            result.add_info(
                "capability_negotiation",
                f"Negotiated {len(successful_capabilities)} capabilities: "
                f"{', '.join(successful_capabilities)}",
            )

        if fallback_capabilities:
            result.add_warning(
                "capability_negotiation",
                f"Used fallback for {len(fallback_capabilities)} capabilities: "
                f"{', '.join(fallback_capabilities)}",
            )

        # Fail if required capabilities are incompatible
        if failed_capabilities and self._fail_on_incompatible:
            result.add_error(
                "capability_negotiation",
                f"Failed to negotiate capabilities: {', '.join(failed_capabilities)}",
            )
            result.success = False

        elif failed_capabilities:
            result.add_warning(
                "capability_negotiation",
                f"Some capabilities failed negotiation: {', '.join(failed_capabilities)}",
            )


# =============================================================================
# Capability-Aware Tool Step Handler
# =============================================================================


class CapabilityAwareToolStepHandler(BaseStepHandler):
    """Tool step handler that respects capability negotiation.

    This handler extends ToolStepHandler to check negotiated capabilities
    before applying tools. If tool capability negotiation failed or used
    a fallback version, this handler can adjust tool behavior accordingly.

    Execution Order: 10 (same as ToolStepHandler, but runs after it checks capabilities)
    """

    order: int = 10

    def apply(
        self,
        orchestrator: Any,
        vertical: Any,
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Apply tools with capability awareness.

        Args:
            orchestrator: The orchestrator being configured
            vertical: The vertical providing extensions
            context: The vertical context with configuration
            result: The integration result to update
        """
        # Check if capability negotiation happened
        negotiation_results = getattr(context, "capability_negotiation_results", None)

        if negotiation_results is None:
            # No negotiation performed, proceed normally
            return

        # Check tool capability negotiation result
        tools_result = negotiation_results.get("tools")

        if tools_result is None or not tools_result.is_success:
            # Tool capability negotiation failed
            logger.warning(
                "Tool capability negotiation failed or not available, " "applying minimal tool set"
            )

            # Apply minimal tool set
            self._apply_minimal_tools(orchestrator, vertical, context, result)
            return

        # Tool capability negotiated successfully
        logger.info(f"Tool capability negotiated: v{tools_result.agreed_version}")

        # Apply tools based on negotiated version
        self._apply_tools_for_version(
            orchestrator,
            vertical,
            context,
            result,
            tools_result.agreed_version,
            tools_result.supported_features,
        )

    def _apply_minimal_tools(
        self,
        orchestrator: Any,
        vertical: Any,
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Apply minimal tool set when negotiation fails.

        Args:
            orchestrator: The orchestrator being configured
            vertical: The vertical providing extensions
            context: The vertical context with configuration
            result: The integration result to update
        """
        # Get basic tools
        if hasattr(vertical, "get_tools"):
            tools = vertical.get_tools()

            # Filter to minimal set
            minimal_tools = [t for t in tools if t in ("read", "write")]

            if hasattr(orchestrator, "set_enabled_tools"):
                orchestrator.set_enabled_tools(set(minimal_tools))

            result.tools_applied = minimal_tools
            logger.info(f"Applied minimal tool set: {minimal_tools}")

    def _apply_tools_for_version(
        self,
        orchestrator: Any,
        vertical: Any,
        context: VerticalContext,
        result: IntegrationResult,
        version: Any,
        supported_features: List[str],
    ) -> None:
        """Apply tools based on negotiated version.

        Args:
            orchestrator: The orchestrator being configured
            vertical: The vertical providing extensions
            context: The vertical context with configuration
            result: The integration result to update
            version: Negotiated version
            supported_features: List of supported features
        """
        # Get tools from vertical
        if hasattr(vertical, "get_tools"):
            tools = vertical.get_tools()

            # Apply tool filtering if supported
            if "tool_filtering" in supported_features:
                # Apply full tool set with filtering
                if hasattr(orchestrator, "set_enabled_tools"):
                    orchestrator.set_enabled_tools(set(tools))

                result.tools_applied = tools
                logger.debug(f"Applied full tool set with filtering: {tools}")
            else:
                # Apply basic tool set without filtering
                if hasattr(orchestrator, "set_enabled_tools"):
                    orchestrator.set_enabled_tools(set(tools))

                result.tools_applied = tools
                logger.debug(f"Applied basic tool set: {tools}")


__all__ = [
    "CapabilityNegotiationStepHandler",
    "CapabilityAwareToolStepHandler",
]
