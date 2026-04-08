"""Tool service adapter that wraps ToolCoordinator.

Implements ToolServiceProtocol by delegating to the existing
ToolCoordinator, enabling feature-flagged service layer migration.
"""

from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

if TYPE_CHECKING:
    from victor.agent.coordinators.tool_coordinator import ToolCoordinator
    from victor.agent.protocols import ToolAccessContext

logger = logging.getLogger(__name__)


class ToolServiceAdapter:
    """Adapts ToolCoordinator to ToolServiceProtocol.

    This adapter delegates all tool operations to the existing
    ToolCoordinator, providing a service-layer interface without
    changing behavior.
    """

    def __init__(self, tool_coordinator: "ToolCoordinator") -> None:
        self._tool_coordinator = tool_coordinator

    def get_available_tools(self) -> Set[str]:
        """Get all available tool names from the registry."""
        return self._tool_coordinator.get_available_tools()

    def get_enabled_tools(self) -> Set[str]:
        """Get currently enabled tool names."""
        return self._tool_coordinator.get_enabled_tools()

    def set_enabled_tools(self, tools: Set[str]) -> None:
        """Set which tools are enabled for this session."""
        self._tool_coordinator.set_enabled_tools(tools)

    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a specific tool is enabled."""
        return self._tool_coordinator.is_tool_enabled(tool_name)

    def parse_and_validate_tool_calls(
        self,
        tool_calls: Optional[List[Dict[str, Any]]],
        full_content: str,
        tool_adapter: Any,
    ) -> Tuple[Optional[List[Dict[str, Any]]], str]:
        """Parse and validate tool calls from model output."""
        return self._tool_coordinator.parse_and_validate_tool_calls(
            tool_calls, full_content, tool_adapter
        )

    async def execute_tool_with_retry(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        context: Dict[str, Any],
        tool_executor: Optional[Callable[..., Awaitable[Any]]] = None,
        cache: Optional[Any] = None,
        on_success: Optional[Callable[[str, Dict[str, Any], Any], None]] = None,
        retry_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[Any], bool, Optional[str]]:
        """Execute a tool with retry logic."""
        return await self._tool_coordinator.execute_tool_with_retry(
            tool_name,
            tool_args,
            context,
            tool_executor=tool_executor,
            cache=cache,
            on_success=on_success,
            retry_config=retry_config,
        )

    def validate_tool_call(
        self,
        tool_call: Any,
        sanitizer: Any,
        is_tool_enabled_fn: Optional[Callable[[str], bool]] = None,
    ) -> Any:
        """Validate a single tool call."""
        return self._tool_coordinator.validate_tool_call(
            tool_call, sanitizer, is_tool_enabled_fn=is_tool_enabled_fn
        )

    def normalize_arguments_full(
        self,
        tool_name: str,
        original_name: str,
        raw_args: Any,
        argument_normalizer: Any,
        tool_adapter: Any,
        failed_signatures: Optional[Set[Tuple[str, str]]] = None,
    ) -> Any:
        """Normalize tool arguments."""
        return self._tool_coordinator.normalize_arguments_full(
            tool_name,
            original_name,
            raw_args,
            argument_normalizer,
            tool_adapter,
            failed_signatures=failed_signatures,
        )

    def _build_tool_access_context(self) -> "ToolAccessContext":
        """Build tool access context."""
        return self._tool_coordinator._build_tool_access_context()

    def is_healthy(self) -> bool:
        """Check if the tool service is healthy."""
        return self._tool_coordinator is not None
