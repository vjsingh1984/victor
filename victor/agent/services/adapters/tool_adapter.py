"""Tool service adapter shim.

Provides a service-shaped compatibility wrapper that prefers the canonical
ToolService when available and falls back to the deprecated ToolCoordinator.
"""

from __future__ import annotations

import logging
import warnings
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

logger = logging.getLogger(__name__)


class ToolServiceAdapter:
    """Compatibility shim that routes tool calls to the service first."""

    def __init__(
        self,
        *args: Any,
        tool_service: Optional[Any] = None,
        tool_coordinator: Optional[Any] = None,
        deprecated_tool_coordinator: Optional[Any] = None,
    ) -> None:
        if len(args) > 2:
            raise TypeError("ToolServiceAdapter accepts at most 2 positional arguments.")

        if args:
            warnings.warn(
                "Positional ToolServiceAdapter(...) construction is deprecated. "
                "Use tool_service=... for canonical service wiring or "
                "deprecated_tool_coordinator=... for explicit compatibility mode.",
                DeprecationWarning,
                stacklevel=2,
            )
            if tool_service is not None:
                raise TypeError(
                    "Use either positional tool_service or keyword tool_service, not both."
                )
            tool_service = args[0]
            if len(args) == 2:
                if tool_coordinator is not None or deprecated_tool_coordinator is not None:
                    raise TypeError(
                        "Use either positional tool_coordinator or keyword compatibility "
                        "arguments, not both."
                    )
                tool_coordinator = args[1]

        if tool_coordinator is not None and deprecated_tool_coordinator is not None:
            raise TypeError(
                "Use only one of tool_coordinator or deprecated_tool_coordinator."
            )

        resolved_deprecated_tool_coordinator = deprecated_tool_coordinator
        if tool_coordinator is not None:
            warnings.warn(
                "ToolServiceAdapter(tool_coordinator=...) is deprecated. "
                "Use deprecated_tool_coordinator=... for explicit compatibility "
                "mode or prefer ToolService.",
                DeprecationWarning,
                stacklevel=2,
            )
            resolved_deprecated_tool_coordinator = tool_coordinator

        if tool_service is None and resolved_deprecated_tool_coordinator is not None:
            warnings.warn(
                "ToolServiceAdapter configured with a coordinator fallback only. "
                "This compatibility path is deprecated; prefer ToolService.",
                DeprecationWarning,
                stacklevel=2,
            )
        self._tool_service = tool_service
        self._deprecated_tool_coordinator = resolved_deprecated_tool_coordinator

    def _delegate(self, method_name: str) -> Callable[..., Any]:
        service = self._tool_service
        if service is not None:
            method = getattr(service, method_name, None)
            if callable(method):
                return method

        coordinator = self._deprecated_tool_coordinator
        if coordinator is not None:
            method = getattr(coordinator, method_name, None)
            if callable(method):
                return method

        raise AttributeError(f"Tool service adapter has no delegate for {method_name}")

    def get_available_tools(self) -> Set[str]:
        """Get all available tool names from the registry."""
        return self._delegate("get_available_tools")()

    def get_enabled_tools(self) -> Set[str]:
        """Get currently enabled tool names."""
        return self._delegate("get_enabled_tools")()

    def set_enabled_tools(self, tools: Set[str]) -> None:
        """Set which tools are enabled for this session."""
        self._delegate("set_enabled_tools")(tools)

    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a specific tool is enabled."""
        return self._delegate("is_tool_enabled")(tool_name)

    def resolve_tool_alias(self, tool_name: str) -> str:
        """Resolve a tool alias to its canonical name."""
        return self._delegate("resolve_tool_alias")(tool_name)

    def parse_and_validate_tool_calls(
        self,
        tool_calls: Optional[List[Dict[str, Any]]],
        full_content: str,
        tool_adapter: Any,
    ) -> Tuple[Optional[List[Dict[str, Any]]], str]:
        """Parse and validate tool calls from model output."""
        return self._delegate("parse_and_validate_tool_calls")(
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
        return await self._delegate("execute_tool_with_retry")(
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
        return self._delegate("validate_tool_call")(
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
        return self._delegate("normalize_arguments_full")(
            tool_name,
            original_name,
            raw_args,
            argument_normalizer,
            tool_adapter,
            failed_signatures=failed_signatures,
        )

    def normalize_tool_arguments(self, tool_args: Dict[str, Any], tool_name: str) -> Any:
        """Normalize tool arguments before execution."""
        return self._delegate("normalize_tool_arguments")(tool_args, tool_name)

    def process_tool_results(
        self,
        pipeline_result: Any,
        ctx: Any,
    ) -> List[Dict[str, Any]]:
        """Process tool execution results via the canonical delegate."""
        return self._delegate("process_tool_results")(pipeline_result, ctx)

    def on_tool_complete(
        self,
        result: Any,
        metrics_collector: Optional[Any] = None,
        *,
        read_files_session: Optional[Set[str]] = None,
        required_files: Optional[List[str]] = None,
        required_outputs: Optional[List[str]] = None,
        nudge_sent_flag: Optional[List[bool]] = None,
        add_message: Optional[Callable[[str, str], None]] = None,
        observability: Optional[Any] = None,
        pipeline_calls_used: int = 0,
        tool_name: Optional[str] = None,
        elapsed: float = 0.0,
        session_id: Optional[str] = None,
    ) -> None:
        """Notify the canonical delegate that a tool execution completed."""
        self._delegate("on_tool_complete")(
            result=result,
            metrics_collector=metrics_collector,
            read_files_session=read_files_session,
            required_files=required_files,
            required_outputs=required_outputs,
            nudge_sent_flag=nudge_sent_flag,
            add_message=add_message,
            observability=observability,
            pipeline_calls_used=pipeline_calls_used,
            tool_name=tool_name,
            elapsed=elapsed,
            session_id=session_id,
        )

    def _build_tool_access_context(self) -> Any:
        """Build tool access context."""
        service = self._tool_service
        if service is not None:
            method = getattr(service, "build_tool_access_context", None)
            if callable(method):
                return method()
        coordinator = self._deprecated_tool_coordinator
        if coordinator is not None:
            method = getattr(coordinator, "build_tool_access_context", None)
            if callable(method):
                return method()
        raise AttributeError("Tool service adapter has no delegate for tool access context")

    def build_tool_access_context(self) -> Any:
        """Build tool access context."""
        return self._build_tool_access_context()

    def is_healthy(self) -> bool:
        """Check if the tool service is healthy."""
        return self._tool_service is not None or self._deprecated_tool_coordinator is not None
