"""Internal helpers mapping framework API to orchestrator.

This module contains implementation details that bridge the simplified
framework API to the existing AgentOrchestrator internals. Users should
not import from this module directly.

Phase 7.6 Refactoring:
- configure_tools() now delegates to ToolConfigurator
- stream_with_events() uses EventRegistry for conversion
- setup_observability_integration() uses AgentBridge patterns

Phase 8.0 - Vertical Integration Unification:
- Added vertical parameter to create_orchestrator_from_options()
- Uses VerticalIntegrationPipeline for consistent vertical application
- Achieves parity with FrameworkShim CLI path
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional, Type, Union

from victor.framework.events import (
    Event,
    EventType,
    content_event,
    error_event,
    stream_end_event,
    stream_start_event,
    thinking_event,
    tool_call_event,
    tool_result_event,
)
from victor.framework.tools import ToolSet

# Import capability helpers for protocol-based access
from victor.framework.vertical_integration import _check_capability, _invoke_capability

if TYPE_CHECKING:
    from victor.framework.config import AgentConfig
    from victor.core.verticals.base import VerticalBase


async def create_orchestrator_from_options(
    provider: str,
    model: Optional[str],
    temperature: float,
    max_tokens: int,
    tools: Union[ToolSet, List[str], None],
    thinking: bool,
    airgapped: bool,
    profile: Optional[str],
    workspace: Optional[str],
    config: Optional["AgentConfig"],
    system_prompt: Optional[str] = None,
    enable_observability: bool = True,
    session_id: Optional[str] = None,
    vertical: Optional[Union[Type["VerticalBase"], str]] = None,
) -> Any:
    """Create an AgentOrchestrator from framework options.

    This function bridges the simplified framework API to the
    internal orchestrator creation logic.

    Args:
        provider: LLM provider name
        model: Model identifier
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        tools: Tool configuration
        thinking: Enable extended thinking
        airgapped: Air-gapped mode
        profile: Profile name from profiles.yaml
        workspace: Working directory
        config: Advanced configuration
        system_prompt: Optional custom system prompt (e.g., from a vertical)
        enable_observability: Whether to auto-initialize ObservabilityIntegration
        session_id: Optional session ID for event correlation
        vertical: Optional vertical class or name to apply

    Returns:
        Configured AgentOrchestrator instance
    """
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.config.settings import load_settings
    from victor.core.bootstrap import ensure_bootstrapped

    # Load settings
    settings = load_settings()

    # Apply config overrides
    if config:
        for key, value in config.to_settings_dict().items():
            if hasattr(settings, key):
                setattr(settings, key, value)

    # Apply airgapped mode
    if airgapped:
        settings.airgapped_mode = True

    # Resolve vertical name for bootstrap
    vertical_name = None
    if vertical:
        if isinstance(vertical, str):
            vertical_name = vertical
        elif hasattr(vertical, "name"):
            vertical_name = vertical.name

    # Bootstrap with vertical context BEFORE orchestrator creation
    # This ensures vertical services are registered with correct vertical name
    ensure_bootstrapped(settings, vertical=vertical_name)

    # Create orchestrator using from_settings
    orchestrator = await AgentOrchestrator.from_settings(
        settings,
        profile_name=profile or "default",
        thinking=thinking,
    )

    # Override provider/model if specified
    if provider != "anthropic" or model:
        # Need to switch provider/model - use capability-based check
        if hasattr(orchestrator, "_provider_manager") or hasattr(orchestrator, "provider_manager"):
            pm = getattr(orchestrator, "_provider_manager", None) or getattr(
                orchestrator, "provider_manager", None
            )
            if pm:
                await pm.switch_provider(provider, model or orchestrator.model)

    # Apply vertical configuration using unified pipeline
    # This achieves parity with FrameworkShim CLI path
    if vertical:
        apply_vertical_to_orchestrator(orchestrator, vertical)

    # Configure tools if specified using ToolConfigurator
    if tools:
        configure_tools(orchestrator, tools, airgapped=airgapped)

    # Apply custom system prompt (e.g., from vertical)
    # Note: This is in addition to any prompt from vertical
    if system_prompt:
        apply_system_prompt(orchestrator, system_prompt)

    # Auto-initialize observability integration
    if enable_observability:
        setup_observability_integration(orchestrator, session_id)

    return orchestrator


def apply_vertical_to_orchestrator(
    orchestrator: Any,
    vertical: Union[Type["VerticalBase"], str],
) -> None:
    """Apply vertical configuration to orchestrator using integration pipeline.

    This provides parity with FrameworkShim by using the same
    VerticalIntegrationPipeline for both SDK and CLI paths.

    Args:
        orchestrator: AgentOrchestrator instance
        vertical: Vertical class or name string
    """
    from victor.framework.vertical_integration import (
        VerticalIntegrationPipeline,
        IntegrationResult,
    )
    import logging

    logger = logging.getLogger(__name__)

    # Create and apply pipeline
    pipeline = VerticalIntegrationPipeline()
    result = pipeline.apply(orchestrator, vertical)

    if result.success:
        logger.info(
            f"Applied vertical '{result.vertical_name}' via SDK path: "
            f"tools={len(result.tools_applied)}, "
            f"middleware={result.middleware_count}, "
            f"safety={result.safety_patterns_count}"
        )
    else:
        for error in result.errors:
            logger.error(f"Vertical integration error: {error}")
        for warning in result.warnings:
            logger.warning(f"Vertical integration warning: {warning}")


def setup_observability_integration(
    orchestrator: Any,
    session_id: Optional[str] = None,
) -> Any:
    """Set up ObservabilityIntegration for the orchestrator.

    This wires the EventBus into the orchestrator for unified event handling.

    Args:
        orchestrator: AgentOrchestrator instance
        session_id: Optional session ID for event correlation

    Returns:
        ObservabilityIntegration instance
    """
    from victor.observability.integration import ObservabilityIntegration

    # Create integration with optional session ID
    integration = ObservabilityIntegration(session_id=session_id)

    # Wire into orchestrator
    integration.wire_orchestrator(orchestrator)

    # Store reference on orchestrator for access
    orchestrator.observability = integration

    return integration


def apply_system_prompt(orchestrator: Any, system_prompt: str) -> None:
    """Apply a custom system prompt to the orchestrator.

    SOLID Compliance (DIP): This function only uses public methods.
    It never writes to private attributes to maintain proper encapsulation
    and dependency inversion.

    Args:
        orchestrator: AgentOrchestrator instance
        system_prompt: Custom system prompt text
    """
    import logging

    logger = logging.getLogger(__name__)

    # Use capability-based approach (protocol-first, fallback to hasattr)
    # SOLID Compliance (DIP): Only use public methods, never write to private attributes
    if _check_capability(orchestrator, "custom_prompt"):
        _invoke_capability(orchestrator, "custom_prompt", system_prompt)
        logger.debug("Applied system prompt via custom_prompt capability")
    elif _check_capability(orchestrator, "prompt_builder"):
        # Fallback: try direct prompt builder access via public method only
        prompt_builder = getattr(orchestrator, "prompt_builder", None)
        if prompt_builder:
            if hasattr(prompt_builder, "set_custom_prompt"):
                prompt_builder.set_custom_prompt(system_prompt)
                logger.debug("Applied system prompt via prompt_builder.set_custom_prompt")
            else:
                logger.warning(
                    "Cannot set custom prompt: prompt_builder lacks set_custom_prompt method. "
                    "Consider implementing CapabilityRegistryProtocol."
                )
    else:
        logger.warning(
            "Cannot set custom prompt: orchestrator lacks custom_prompt capability "
            "and prompt_builder. Consider implementing CapabilityRegistryProtocol."
        )


def configure_tools(
    orchestrator: Any,
    tools: Union[ToolSet, List[str]],
    airgapped: bool = False,
) -> None:
    """Configure tools on the orchestrator using ToolConfigurator.

    This delegates to the ToolConfigurator from Phase 7.5 for proper
    tool configuration with support for filters and hooks.

    Args:
        orchestrator: AgentOrchestrator instance
        tools: Tool configuration (ToolSet or list of tool names)
        airgapped: Whether to apply airgapped filter
    """
    from victor.framework.tool_config import (
        AirgappedFilter,
        ToolConfigurator,
        ToolConfigMode,
        get_tool_configurator,
    )

    # Get or create configurator
    configurator = get_tool_configurator()

    # Add airgapped filter if in airgapped mode
    if airgapped:
        configurator.add_filter(AirgappedFilter())

    # Configure using ToolSet or list
    if isinstance(tools, ToolSet):
        # Use configure_from_toolset for ToolSet instances
        configurator.configure_from_toolset(orchestrator, tools)
    else:
        # Convert list to set and configure
        configurator.configure(orchestrator, set(tools), ToolConfigMode.REPLACE)


async def stream_with_events(
    orchestrator: Any,
    prompt: str,
) -> AsyncIterator[Event]:
    """Stream orchestrator response as framework Events.

    This function wraps the orchestrator's stream_chat method
    and converts internal stream chunks to framework Events.

    Args:
        orchestrator: AgentOrchestrator instance
        prompt: User prompt

    Yields:
        Event objects representing agent actions
    """
    # Emit stream start
    yield stream_start_event()

    try:
        async for chunk in orchestrator.stream_chat(prompt):
            # Handle thinking content (extended thinking mode)
            if chunk.metadata and chunk.metadata.get("reasoning_content"):
                yield thinking_event(chunk.metadata["reasoning_content"])

            # Handle regular content
            if chunk.content:
                yield content_event(chunk.content)

            # Handle tool start events
            if chunk.metadata and "tool_start" in chunk.metadata:
                tool_data = chunk.metadata["tool_start"]
                yield tool_call_event(
                    tool_name=tool_data.get("name", "unknown"),
                    tool_id=tool_data.get("id"),
                    arguments=tool_data.get("arguments", {}),
                )

            # Handle tool result events
            if chunk.metadata and "tool_result" in chunk.metadata:
                tool_data = chunk.metadata["tool_result"]
                yield tool_result_event(
                    tool_name=tool_data.get("name", "unknown"),
                    tool_id=tool_data.get("id"),
                    result=str(tool_data.get("result", "")),
                    success=tool_data.get("success", True),
                )

            # Handle tool calls in chunk
            if chunk.tool_calls:
                for tc in chunk.tool_calls:
                    yield tool_call_event(
                        tool_name=tc.get("name", "unknown"),
                        tool_id=tc.get("id"),
                        arguments=tc.get("arguments", {}),
                    )

        # Emit stream end
        yield stream_end_event(success=True)

    except Exception as e:
        yield error_event(str(e), recoverable=False)
        yield stream_end_event(success=False, error=str(e))


def format_context_message(context: Dict[str, Any]) -> Optional[str]:
    """Format context dict into a message string.

    Args:
        context: Context dictionary with file, error, code, etc.

    Returns:
        Formatted string for prepending to conversation, or None
    """
    if not context:
        return None

    parts = []

    if "file" in context:
        parts.append(f"File: {context['file']}")

    if "files" in context:
        parts.append(f"Files: {', '.join(context['files'])}")

    if "error" in context:
        parts.append(f"Error: {context['error']}")

    if "code" in context:
        parts.append(f"```\n{context['code']}\n```")

    # Add any other context as key-value pairs
    for key, value in context.items():
        if key not in ("file", "files", "error", "code"):
            parts.append(f"{key}: {value}")

    return "\n".join(parts) if parts else None


def collect_tool_calls(events: List[Event]) -> List[Dict[str, Any]]:
    """Collect tool calls from a list of events.

    Args:
        events: List of Event objects

    Returns:
        List of tool call dictionaries
    """
    tool_calls = []

    for event in events:
        if event.type == EventType.TOOL_RESULT:
            tool_calls.append(
                {
                    "tool": event.tool_name,
                    "tool_id": event.tool_id,
                    "arguments": event.arguments,
                    "result": event.result,
                    "success": event.success,
                }
            )

    return tool_calls
