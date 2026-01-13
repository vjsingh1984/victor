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

"""Reusable pipeline for vertical extension application.

This module provides a unified pipeline for applying vertical extensions
to orchestrators, ensuring both CLI (FrameworkShim) and SDK (Agent.create())
paths apply identical vertical configurations.

Design Philosophy:
- Single implementation for all vertical integration
- Protocol-based access (no private attribute writes)
- Type-safe configuration through VerticalContext
- SOLID-compliant extension points
- Step handlers for Single Responsibility (Phase 3.1)

Architecture (Refactored with Step Handlers):
    VerticalIntegrationPipeline (Facade)
    │
    └── StepHandlerRegistry
        ├── ToolStepHandler (order=10) - Apply tools filter
        ├── PromptStepHandler (order=20) - Apply system prompt
        ├── ConfigStepHandler (order=40) - Apply stages
        ├── ExtensionsStepHandler (order=45)
        │   ├── MiddlewareStepHandler - Apply middleware
        │   ├── SafetyStepHandler - Apply safety patterns
        │   ├── PromptStepHandler - Apply prompt contributors
        │   └── ConfigStepHandler - Apply mode config & tool deps
        ├── FrameworkStepHandler (order=60)
        │   ├── Workflows - Register workflow definitions
        │   ├── RL Config - Configure RL learners
        │   └── Team Specs - Register team specifications
        └── ContextStepHandler (order=100) - Attach context

Architecture Notes: Pipeline vs Adapter
---------------------------------------

This module contains two distinct components with different responsibilities:

1. **VerticalIntegrationPipeline** (this module):
   - Purpose: Orchestrate the COMPLETE vertical application process
   - Phase: Setup/initialization (one-time during agent creation)
   - Scope: All vertical aspects (tools, prompts, config, extensions, workflows)
   - Usage: Called by FrameworkShim, Agent.create(), SDK initialization
   - Pattern: Facade + Template Method with step handlers
   - Example:
        pipeline = VerticalIntegrationPipeline()
        result = pipeline.apply(orchestrator, CodingAssistant)

2. **VerticalIntegrationAdapter** (victor/agent/vertical_integration_adapter.py):
   - Purpose: Provide runtime implementation for middleware/safety application
   - Phase: Runtime (during request processing)
   - Scope: Only middleware chain and safety pattern application
   - Usage: Called by orchestrator's apply_vertical_middleware/safety_patterns methods
   - Pattern: Adapter (wraps orchestrator, provides single implementation)
   - Example:
        adapter = VerticalIntegrationAdapter(orchestrator)
        adapter.apply_middleware(middleware_list)
        adapter.apply_safety_patterns(patterns)

Key Differences:
┌────────────────────────┬──────────────────────────────┬─────────────────────────────────┐
│ Aspect                 │ VerticalIntegrationPipeline   │ VerticalIntegrationAdapter       │
├────────────────────────┼──────────────────────────────┼─────────────────────────────────┤
│ When to use            │ Agent creation/setup         │ Runtime request processing      │
│ What it handles        │ Entire vertical config       │ Middleware & safety only        │
│ Called by              │ FrameworkShim, SDK           │ AgentOrchestrator methods       │
│ Step handlers          │ Yes (orchestration)          │ No (direct implementation)      │
│ Returns                │ IntegrationResult            │ None (void)                     │
│ Caching                │ Yes (config cache)           │ No (stateful)                   │
│ Location               │ victor.framework.vertical_integration │ victor.agent.vertical_integration_adapter │
└────────────────────────┴──────────────────────────────┴─────────────────────────────────┘

When to use which:
- Use Pipeline when: Creating an agent, applying a vertical, SDK initialization
- Use Adapter when: Implementing orchestrator methods, applying middleware/safety at runtime

The Pipeline delegates to the Adapter:
- The Pipeline's MiddlewareStepHandler calls orchestrator.apply_vertical_middleware()
- The orchestrator method delegates to VerticalIntegrationAdapter.apply_middleware()
- This separation allows: (a) Pipeline to orchestrate the full setup, (b) Adapter to
  provide runtime implementation without requiring Pipeline imports

Usage:
    from victor.framework.vertical_integration import (
        VerticalIntegrationPipeline,
        IntegrationResult,
    )

    # Create pipeline
    pipeline = VerticalIntegrationPipeline()

    # Apply vertical
    result = pipeline.apply(orchestrator, CodingAssistant)

    # Check result
    if result.success:
        print(f"Applied {result.vertical_name}")
        print(f"Tools: {result.tools_applied}")
    else:
        print(f"Errors: {result.errors}")

    # Custom step handlers (Phase 3.1)
    from victor.framework.step_handlers import StepHandlerRegistry

    registry = StepHandlerRegistry.default()
    registry.add_handler(MyCustomStepHandler())
    pipeline = VerticalIntegrationPipeline(step_registry=registry)
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    Union,
    runtime_checkable,
)

from victor.agent.vertical_context import VerticalContext, create_vertical_context

if TYPE_CHECKING:
    from victor.framework.step_handlers import StepHandlerRegistry
    from victor.core.verticals.base import VerticalBase

# Import protocols for runtime isinstance checks

# Import capability registry protocol for type-safe capability access
from victor.framework.protocols import CapabilityRegistryProtocol


def _check_capability(
    obj: Any,
    capability_name: str,
    min_version: Optional[str] = None,
    strict_mode: bool = False,
) -> bool:
    """Check if object has capability via registry with optional version check.

    Uses protocol-based capability discovery. Orchestrators must implement
    CapabilityRegistryProtocol for proper capability checking.

    SOLID Compliance:
    - Uses protocol, not hasattr (DIP - Dependency Inversion)
    - No private attribute access (SRP - Single Responsibility)

    Args:
        obj: Object to check (should implement CapabilityRegistryProtocol)
        capability_name: Name of capability
        min_version: Minimum required version (default: None = any version)
        strict_mode: If True, raise TypeError when obj doesn't implement protocol

    Returns:
        True if capability is available via the registry and meets version requirement

    Raises:
        TypeError: If strict_mode=True and obj doesn't implement CapabilityRegistryProtocol

    Example:
        # Check for any version
        if _check_capability(obj, "enabled_tools"):
            ...

        # Check for minimum version
        if _check_capability(obj, "enabled_tools", min_version="1.1"):
            ...

        # Strict mode (raises error instead of fallback)
        if _check_capability(obj, "enabled_tools", strict_mode=True):
            ...
    """
    # Check capability registry (protocol-based only)
    if isinstance(obj, CapabilityRegistryProtocol):
        return obj.has_capability(capability_name, min_version=min_version)

    # Strict mode: raise TypeError for non-protocol objects
    if strict_mode:
        raise TypeError(
            f"Object must implement CapabilityRegistryProtocol for capability checking. "
            f"Got {type(obj).__name__} instead. "
            f"Ensure your orchestrator uses CapabilityRegistryMixin."
        )

    # For objects not implementing protocol, show deprecation warning and fallback
    if min_version is not None:
        logger.debug(
            f"Version check requested for '{capability_name}' but object does not "
            f"implement CapabilityRegistryProtocol. Falling back to hasattr check."
        )

    warnings.warn(
        f"Object {type(obj).__name__} does not implement CapabilityRegistryProtocol. "
        f"Falling back to hasattr() checks for capability '{capability_name}'. "
        f"This is deprecated and will be removed in a future version. "
        f"Please add CapabilityRegistryMixin to your orchestrator.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Legacy fallback with public method mappings
    public_methods = {
        "enabled_tools": "set_enabled_tools",
        "prompt_builder": "prompt_builder",
        "vertical_middleware": "apply_vertical_middleware",
        "vertical_safety_patterns": "apply_vertical_safety_patterns",
        "vertical_context": "set_vertical_context",
        "adaptive_mode_controller": "adaptive_mode_controller",
    }

    method_name = public_methods.get(capability_name, capability_name)
    return hasattr(obj, method_name) and (
        callable(getattr(obj, method_name, None))
        or not callable(getattr(obj, method_name, None))  # Allow properties
    )


def _invoke_capability(
    obj: Any,
    capability_name: str,
    *args: Any,
    min_version: Optional[str] = None,
    strict_mode: bool = False,
    **kwargs: Any,
) -> Any:
    """Invoke a capability on an object via public methods only.

    SOLID Compliance (DIP): This function only uses public methods.
    It never writes to private attributes (_attr) to maintain
    proper encapsulation and dependency inversion.

    Uses protocol-based capability invocation. Orchestrators must implement
    CapabilityRegistryProtocol for proper capability invocation. When the
    protocol is not implemented, falls back to public method mappings but
    never resorts to private attribute access.

    Args:
        obj: Object implementing the capability (should implement CapabilityRegistryProtocol)
        capability_name: Name of the capability to invoke
        *args: Arguments for capability (value to pass to the capability method)
        min_version: Minimum required version (default: None = no check)
        strict_mode: If True, raise TypeError when obj doesn't implement protocol
        **kwargs: Additional arguments for capability

    Returns:
        Result of capability invocation, True if capability was invoked successfully

    Raises:
        TypeError: If strict_mode=True and obj doesn't implement CapabilityRegistryProtocol
        AttributeError: If capability cannot be invoked via public methods

    Example:
        # Invoke without version check
        _invoke_capability(obj, "enabled_tools", {"read", "write"})

        # Invoke with version requirement
        _invoke_capability(obj, "enabled_tools", {"read", "write"}, min_version="1.1")

        # Strict mode (raises error instead of fallback)
        _invoke_capability(obj, "enabled_tools", {"read", "write"}, strict_mode=True)
    """
    # Use capability registry if available (preferred)
    if isinstance(obj, CapabilityRegistryProtocol):
        try:
            return obj.invoke_capability(capability_name, *args, min_version=min_version, **kwargs)
        except (KeyError, TypeError) as e:
            logger.debug(f"Registry invoke failed for {capability_name}: {e}")
            # Fall through to public method fallback

    # Strict mode: raise TypeError for non-protocol objects
    if strict_mode:
        raise TypeError(
            f"Object must implement CapabilityRegistryProtocol for capability invocation. "
            f"Got {type(obj).__name__} instead. "
            f"Ensure your orchestrator uses CapabilityRegistryMixin."
        )

    # Show deprecation warning when falling back to hasattr
    if not isinstance(obj, CapabilityRegistryProtocol):
        warnings.warn(
            f"Object {type(obj).__name__} does not implement CapabilityRegistryProtocol. "
            f"Falling back to hasattr() checks for capability '{capability_name}'. "
            f"This is deprecated and will be removed in a future version. "
            f"Please add CapabilityRegistryMixin to your orchestrator.",
            DeprecationWarning,
            stacklevel=2,
        )

    # Fallback: use public method mappings only (no private attributes)
    # Note: Version checking not available without protocol implementation
    if min_version is not None:
        logger.debug(
            f"Version check requested for '{capability_name}' but object does not "
            f"implement CapabilityRegistryProtocol. Invoking without version check."
        )

    # Use centralized capability method mappings (single source of truth)
    from victor.agent.capability_registry import get_method_for_capability

    method_name = get_method_for_capability(capability_name)
    method = getattr(obj, method_name, None)
    if callable(method):
        return method(*args, **kwargs)

    # No private attribute fallback - raise clear error instead
    raise AttributeError(
        f"Cannot invoke capability '{capability_name}' on {type(obj).__name__}. "
        f"Expected method '{method_name}' not found. "
        f"Object should implement CapabilityRegistryProtocol."
    )


logger = logging.getLogger(__name__)


# =============================================================================
# Extension Handler Registry (OCP Compliance)
# =============================================================================


@dataclass
class ExtensionHandlerInfo:
    """Information about an extension handler.

    Attributes:
        name: Name of the extension type
        attr_name: Attribute name on VerticalExtensions object
        handler: Handler function to call
        order: Execution order (lower = earlier)
    """

    name: str
    attr_name: str
    handler: Callable[
        ["VerticalIntegrationPipeline", Any, Any, VerticalContext, "IntegrationResult"], None
    ]
    order: int = 100


class ExtensionHandlerRegistry:
    """Registry for extension handlers (OCP compliance).

    .. deprecated::
        This registry is kept for architectural reference but is NOT actively used.
        The active implementation lives in `victor.framework.step_handlers.ExtensionsStepHandler`
        which uses its own `ExtensionHandlerRegistry` instance-based approach.

        This legacy registry represents the original OCP-compliant design for the
        `VerticalIntegrationPipeline`. The codebase has evolved to use step handlers
        with SOLID principles, and the extension handling is now managed by
        `ExtensionsStepHandler` in `step_handlers.py`.

        Keeping this registry serves as documentation of the architectural evolution
        from OCP to SOLID principles. It may be repurposed in future iterations for
        higher-level extension types that operate outside the step handler system.

    This registry allows new extension types to be added without modifying
    the _apply_extensions method. Each extension type registers a handler
    that will be called when that extension is present.

    Example:
        registry = ExtensionHandlerRegistry.default()

        # Register a new extension type
        registry.register(
            name="chain_factory",
            attr_name="chain_factory_provider",
            handler=lambda pipeline, orch, ext, ctx, res: pipeline._apply_chain_factory(orch, ext, ctx, res),
            order=70,
        )

    Note:
        For new extension types, use `victor.framework.step_handlers.ExtensionsStepHandler`
        and its `ExtensionHandlerRegistry` instead.
    """

    def __init__(self) -> None:
        self._handlers: Dict[str, ExtensionHandlerInfo] = {}

    def register(
        self,
        name: str,
        attr_name: str,
        handler: Callable,
        order: int = 100,
    ) -> None:
        """Register an extension handler.

        Args:
            name: Name of the extension type
            attr_name: Attribute name on VerticalExtensions object
            handler: Handler function (pipeline, orch, ext_value, context, result) -> None
            order: Execution order (lower = earlier)
        """
        self._handlers[name] = ExtensionHandlerInfo(
            name=name,
            attr_name=attr_name,
            handler=handler,
            order=order,
        )
        logger.debug(f"Registered extension handler: {name} (order={order})")

    def unregister(self, name: str) -> bool:
        """Unregister an extension handler.

        Args:
            name: Name of the extension type

        Returns:
            True if handler was removed, False if not found
        """
        if name in self._handlers:
            del self._handlers[name]
            return True
        return False

    def get_ordered_handlers(self) -> List[ExtensionHandlerInfo]:
        """Get handlers sorted by execution order.

        Returns:
            List of handler info sorted by order
        """
        return sorted(self._handlers.values(), key=lambda h: h.order)

    def get_handler(self, name: str) -> Optional[ExtensionHandlerInfo]:
        """Get a specific handler by name.

        Args:
            name: Name of the extension type

        Returns:
            Handler info or None if not found
        """
        return self._handlers.get(name)

    @classmethod
    def default(cls) -> "ExtensionHandlerRegistry":
        """Create registry with default extension handlers.

        Returns:
            Registry with built-in extension handlers
        """
        registry = cls()

        # Register default handlers in order
        registry.register(
            name="middleware",
            attr_name="middleware",
            handler=lambda p, o, e, c, r: p._apply_middleware(o, e, c, r),
            order=10,
        )
        registry.register(
            name="safety",
            attr_name="safety_extensions",
            handler=lambda p, o, e, c, r: p._apply_safety(o, e, c, r),
            order=20,
        )
        registry.register(
            name="prompts",
            attr_name="prompt_contributors",
            handler=lambda p, o, e, c, r: p._apply_prompts(o, e, c, r),
            order=30,
        )
        registry.register(
            name="mode_config",
            attr_name="mode_config_provider",
            handler=lambda p, o, e, c, r: p._apply_mode_config(o, e, c, r),
            order=40,
        )
        registry.register(
            name="tool_deps",
            attr_name="tool_dependency_provider",
            handler=lambda p, o, e, c, r: p._apply_tool_deps(o, e, c, r),
            order=50,
        )

        return registry


# Module-level default registry (lazy initialization)
_default_extension_registry: Optional[ExtensionHandlerRegistry] = None


def get_extension_handler_registry() -> ExtensionHandlerRegistry:
    """Get the default extension handler registry.

    Returns:
        The default extension handler registry
    """
    global _default_extension_registry
    if _default_extension_registry is None:
        _default_extension_registry = ExtensionHandlerRegistry.default()
    return _default_extension_registry


def register_extension_handler(
    name: str,
    attr_name: str,
    handler: Callable,
    order: int = 100,
) -> None:
    """Register a new extension handler (OCP extension point).

    This allows adding new extension types without modifying the
    _apply_extensions method.

    Args:
        name: Name of the extension type
        attr_name: Attribute name on VerticalExtensions object
        handler: Handler function (pipeline, orch, ext_value, context, result) -> None
        order: Execution order (lower = earlier)

    Example:
        def apply_chain_factory(pipeline, orch, chain_factory, ctx, result):
            chains = chain_factory.create_chains()
            # Apply chains...

        register_extension_handler(
            name="chain_factory",
            attr_name="chain_factory_provider",
            handler=apply_chain_factory,
            order=60,
        )
    """
    registry = get_extension_handler_registry()
    registry.register(name, attr_name, handler, order)


# =============================================================================
# Integration Result
# =============================================================================


@dataclass
class ExtensionLoadErrorInfo:
    """Information about an extension loading error.

    Captures details about a failed extension load for reporting.

    Attributes:
        extension_type: Type of extension that failed
        vertical_name: Name of the vertical
        error_message: The error message
        is_required: Whether this extension was required
        original_exception_type: Type name of the original exception
    """

    extension_type: str
    vertical_name: str
    error_message: str
    is_required: bool = False
    original_exception_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "extension_type": self.extension_type,
            "vertical_name": self.vertical_name,
            "error_message": self.error_message,
            "is_required": self.is_required,
            "original_exception_type": self.original_exception_type,
        }


# Type alias for validation status
ValidationStatus = Literal["success", "partial", "failed"]


@dataclass
class IntegrationResult:
    """Result from applying a vertical to an orchestrator.

    Attributes:
        success: Whether integration succeeded
        vertical_name: Name of the applied vertical
        context: The created VerticalContext
        tools_applied: Set of tools enabled
        middleware_count: Number of middleware applied
        safety_patterns_count: Number of safety patterns applied
        prompt_hints_count: Number of prompt hints applied
        mode_configs_count: Number of mode configs applied
        workflows_count: Number of workflows registered
        rl_learners_count: Number of RL learners configured
        team_specs_count: Number of team specifications registered
        step_status: Per-step execution status for detailed auditing
        errors: List of errors encountered
        warnings: List of warnings
        info: List of informational messages
        extension_errors: List of ExtensionLoadErrorInfo for extension loading failures
        validation_status: Overall validation status - "success", "partial", or "failed"
    """

    success: bool = True
    vertical_name: Optional[str] = None
    context: Optional[VerticalContext] = None
    tools_applied: Set[str] = field(default_factory=set)
    middleware_count: int = 0
    safety_patterns_count: int = 0
    prompt_hints_count: int = 0
    mode_configs_count: int = 0
    workflows_count: int = 0
    rl_learners_count: int = 0
    team_specs_count: int = 0
    step_status: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    extension_errors: List[ExtensionLoadErrorInfo] = field(default_factory=list)
    validation_status: ValidationStatus = "success"

    def add_error(self, error: str) -> None:
        """Add an error."""
        self.errors.append(error)
        self.success = False
        self._update_validation_status()

    def add_warning(self, warning: str) -> None:
        """Add a warning."""
        self.warnings.append(warning)

    def add_info(self, message: str) -> None:
        """Add an informational message."""
        self.info.append(message)

    def add_extension_error(
        self,
        extension_type: str,
        vertical_name: str,
        error_message: str,
        is_required: bool = False,
        original_exception: Optional[Exception] = None,
    ) -> None:
        """Add an extension loading error.

        Args:
            extension_type: Type of extension that failed
            vertical_name: Name of the vertical
            error_message: The error message
            is_required: Whether this extension was required
            original_exception: The original exception (optional)
        """
        error_info = ExtensionLoadErrorInfo(
            extension_type=extension_type,
            vertical_name=vertical_name,
            error_message=error_message,
            is_required=is_required,
            original_exception_type=(
                type(original_exception).__name__ if original_exception else None
            ),
        )
        self.extension_errors.append(error_info)

        # If required extension failed, mark as failed
        if is_required:
            self.success = False
            self.validation_status = "failed"
        else:
            # Non-required failures result in partial status
            self._update_validation_status()

    def _update_validation_status(self) -> None:
        """Update validation status based on current errors."""
        if not self.success:
            # Check if any required extension failed
            required_failures = [e for e in self.extension_errors if e.is_required]
            if required_failures or self.errors:
                self.validation_status = "failed"
            elif self.extension_errors or self.warnings:
                self.validation_status = "partial"
        elif self.extension_errors or self.warnings:
            self.validation_status = "partial"
        else:
            self.validation_status = "success"

    def has_extension_errors(self) -> bool:
        """Check if there are any extension loading errors.

        Returns:
            True if there are extension errors
        """
        return len(self.extension_errors) > 0

    def get_required_extension_failures(self) -> List[ExtensionLoadErrorInfo]:
        """Get list of required extension failures.

        Returns:
            List of ExtensionLoadErrorInfo for required extension failures
        """
        return [e for e in self.extension_errors if e.is_required]

    def record_step_status(
        self,
        step_name: str,
        status: str,
        details: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
    ) -> None:
        """Record the execution status of a step handler.

        Tracks per-step status for detailed auditing and debugging.
        Each step is recorded with its status, optional details, and duration.

        Args:
            step_name: Unique identifier for the step handler
            status: Status string - "success", "error", "warning", or "skipped"
            details: Optional dictionary with additional details (counts, messages, etc.)
            duration_ms: Optional execution duration in milliseconds

        Example:
            result.record_step_status(
                "tools",
                "success",
                details={"tools_count": 15, "canonical_count": 12},
                duration_ms=5.3,
            )
        """
        step_record: Dict[str, Any] = {"status": status}
        if details is not None:
            step_record["details"] = details
        if duration_ms is not None:
            step_record["duration_ms"] = duration_ms
        self.step_status[step_name] = step_record

    def get_step_status(self, step_name: str) -> Optional[Dict[str, Any]]:
        """Get the status record for a specific step.

        Args:
            step_name: Name of the step to look up

        Returns:
            Status record dict or None if step not recorded
        """
        return self.step_status.get(step_name)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization (SRP: data -> dict).

        Returns:
            Dict representation suitable for JSON serialization
        """
        return {
            "success": self.success,
            "validation_status": self.validation_status,
            "vertical_name": self.vertical_name,
            "tools_applied": list(self.tools_applied),
            "middleware_count": self.middleware_count,
            "safety_patterns_count": self.safety_patterns_count,
            "prompt_hints_count": self.prompt_hints_count,
            "mode_configs_count": self.mode_configs_count,
            "workflows_count": self.workflows_count,
            "rl_learners_count": self.rl_learners_count,
            "team_specs_count": self.team_specs_count,
            "step_status": self.step_status,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
            "extension_errors": [e.to_dict() for e in self.extension_errors],
        }

    def persist(self, base_path: Optional[Path] = None) -> Optional[Path]:
        """Persist integration result to JSONL file for auditing (SRP: persistence).

        Uses append-only JSONL format for efficiency - avoids file locking issues
        and reduces filesystem overhead. Each line is a complete JSON record.

        Args:
            base_path: Base path for audit logs. Defaults to ~/.victor/logs/integration/

        Returns:
            Path to the JSONL file, or None if persistence failed
        """
        import json
        from datetime import datetime, timezone

        try:
            # Use default path if not provided
            if base_path is None:
                base_path = Path.home() / ".victor" / "logs" / "integration"

            # Ensure directory exists
            base_path.mkdir(parents=True, exist_ok=True)

            # Use append-only JSONL file (one per day for easy rotation)
            date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
            filepath = base_path / f"integration_{date_str}.jsonl"

            # Append JSON line (atomic on most filesystems)
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "vertical_name": self.vertical_name,
                "result": self.to_dict(),
            }
            with open(filepath, "a") as f:
                f.write(json.dumps(record) + "\n")

            logger.debug(f"IntegrationResult appended to: {filepath}")
            return filepath

        except Exception as e:
            logger.warning(f"Failed to persist IntegrationResult: {e}")
            return None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IntegrationResult":
        """Create IntegrationResult from dictionary (SRP: deserialization).

        Args:
            data: Dictionary representation (from to_dict)

        Returns:
            IntegrationResult instance
        """
        # Deserialize extension errors
        extension_errors = []
        for error_dict in data.get("extension_errors", []):
            extension_errors.append(
                ExtensionLoadErrorInfo(
                    extension_type=error_dict.get("extension_type", "unknown"),
                    vertical_name=error_dict.get("vertical_name", "unknown"),
                    error_message=error_dict.get("error_message", ""),
                    is_required=error_dict.get("is_required", False),
                    original_exception_type=error_dict.get("original_exception_type"),
                )
            )

        return cls(
            success=data.get("success", True),
            validation_status=data.get("validation_status", "success"),
            vertical_name=data.get("vertical_name"),
            tools_applied=set(data.get("tools_applied", [])),
            middleware_count=data.get("middleware_count", 0),
            safety_patterns_count=data.get("safety_patterns_count", 0),
            prompt_hints_count=data.get("prompt_hints_count", 0),
            mode_configs_count=data.get("mode_configs_count", 0),
            workflows_count=data.get("workflows_count", 0),
            rl_learners_count=data.get("rl_learners_count", 0),
            team_specs_count=data.get("team_specs_count", 0),
            step_status=data.get("step_status", {}),
            errors=data.get("errors", []),
            warnings=data.get("warnings", []),
            info=data.get("info", []),
            extension_errors=extension_errors,
        )


# =============================================================================
# Integration Protocol
# =============================================================================


@runtime_checkable
class OrchestratorVerticalProtocol(Protocol):
    """Protocol for orchestrators that support vertical integration.

    This defines the methods an orchestrator must implement to
    properly receive vertical configuration. Using a protocol
    ensures SOLID compliance (no private attribute access).
    """

    def set_vertical_context(self, context: VerticalContext) -> None:
        """Set the vertical context."""
        ...

    def set_enabled_tools(self, tools: Set[str]) -> None:
        """Set enabled tools from vertical."""
        ...

    def apply_vertical_middleware(self, middleware: List[Any]) -> None:
        """Apply middleware from vertical."""
        ...

    def apply_vertical_safety_patterns(self, patterns: List[Any]) -> None:
        """Apply safety patterns from vertical."""
        ...


# =============================================================================
# Integration Pipeline
# =============================================================================


class VerticalIntegrationPipeline:
    """Reusable pipeline for vertical extension application.

    This pipeline encapsulates all vertical integration logic that was
    previously spread across FrameworkShim._apply_vertical() and
    related methods. It provides:

    1. **Unified Integration**: Same logic for CLI and SDK paths
    2. **Protocol Compliance**: Uses proper methods, not private attrs
    3. **Error Handling**: Graceful degradation with detailed errors
    4. **Extensibility**: Hook points for custom integration steps
    5. **Step Handlers**: Focused classes for each integration concern (Phase 3.1)

    The pipeline now uses a registry of step handlers that can be customized.
    Each step handler implements a single concern (Single Responsibility Principle).
    New steps can be added without modifying existing ones (Open/Closed Principle).

    Architecture Notes: Pipeline vs Adapter
    ---------------------------------------
    This class (VerticalIntegrationPipeline) is the SETUP-PHASE orchestrator.
    It coordinates the complete vertical application process through step handlers.
    For runtime middleware/safety application, see VerticalIntegrationAdapter.

    - Pipeline: Use during agent creation, applies entire vertical config
    - Adapter: Use at runtime, applies middleware/safety to active requests

    Example:
        pipeline = VerticalIntegrationPipeline()

        # Apply with full vertical class
        result = pipeline.apply(orchestrator, CodingAssistant)

        # Apply with vertical name (from registry)
        result = pipeline.apply(orchestrator, "coding")

        # Apply with custom configuration
        result = pipeline.apply(
            orchestrator,
            CodingAssistant,
            config_overrides={"tool_budget": 30},
        )

        # Use custom step handlers (Phase 3.1)
        from victor.framework.step_handlers import StepHandlerRegistry

        registry = StepHandlerRegistry.default()
        registry.add_handler(MyCustomStepHandler())
        pipeline = VerticalIntegrationPipeline(step_registry=registry)
    """

    def __init__(
        self,
        strict_mode: bool = False,
        pre_hooks: Optional[List[Callable[[Any, Type], None]]] = None,
        post_hooks: Optional[List[Callable[[Any, IntegrationResult], None]]] = None,
        step_registry: Optional["StepHandlerRegistry"] = None,
        use_step_handlers: bool = True,
        extension_registry: Optional[ExtensionHandlerRegistry] = None,
        enable_cache: bool = True,
        cache_ttl: int = 3600,
        parallel_enabled: bool = False,
    ):
        """Initialize the pipeline.

        Args:
            strict_mode: If True, fail on any integration error
            pre_hooks: Callables to run before integration
            post_hooks: Callables to run after integration
            step_registry: Custom step handler registry (uses default if None)
            use_step_handlers: If True, use step handlers; if False, use legacy methods
            extension_registry: Custom extension handler registry (OCP compliance)
            enable_cache: If True, enable configuration caching (default: True)
            cache_ttl: Cache time-to-live in seconds (default: 3600 = 1 hour)
            parallel_enabled: If True, enable parallel execution (Phase 2.2, default: False)
        """
        self._strict_mode = strict_mode
        self._pre_hooks = pre_hooks or []
        self._post_hooks = post_hooks or []
        self._use_step_handlers = use_step_handlers
        self._enable_cache = enable_cache
        self._cache_ttl = cache_ttl
        self._parallel_enabled = parallel_enabled

        # Initialize cache (eagerly, to avoid attribute errors in tests)
        self._cache: Dict[str, bytes] = {}

        # Initialize extension handler registry (OCP compliance)
        self._extension_registry = extension_registry or get_extension_handler_registry()

        # Initialize step handler registry
        if step_registry is not None:
            self._step_registry = step_registry
        elif use_step_handlers:
            try:
                from victor.framework.step_handlers import StepHandlerRegistry

                self._step_registry = StepHandlerRegistry.default()
            except ImportError:
                logger.debug("Step handlers not available, using legacy methods")
                self._step_registry = None
                self._use_step_handlers = False
        else:
            self._step_registry = None

    @property
    def step_registry(self) -> Optional["StepHandlerRegistry"]:
        """Get the step handler registry.

        Returns:
            The step handler registry or None if not using step handlers
        """
        return self._step_registry

    def apply(
        self,
        orchestrator: Any,
        vertical: Union[Type["VerticalBase"], str],
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> IntegrationResult:
        """Apply a vertical to an orchestrator.

        This is the main entry point for vertical integration. It:
        1. Resolves the vertical (if string name)
        2. Creates a VerticalContext
        3. Applies all vertical extensions via step handlers (or legacy methods)
        4. Attaches context to orchestrator

        Args:
            orchestrator: AgentOrchestrator instance
            vertical: Vertical class or name string
            config_overrides: Optional configuration overrides

        Returns:
            IntegrationResult with details of what was applied
        """
        result = IntegrationResult()

        # Resolve vertical from name if needed
        vertical_class = self._resolve_vertical(vertical)
        if vertical_class is None:
            result.add_error(f"Vertical not found: {vertical}")
            return result

        result.vertical_name = vertical_class.name

        # Check cache for pre-computed integration (Phase 1: Caching)
        if self._enable_cache:
            cache_key = self._generate_cache_key(vertical_class)
            if cache_key:
                cached_result = self._load_from_cache(cache_key)
                if cached_result:
                    logger.debug(
                        f"Cache HIT for vertical '{vertical_class.name}' "
                        f"(key: {cache_key[:16]}...)"
                    )
                    return cached_result
                else:
                    logger.debug(
                        f"Cache MISS for vertical '{vertical_class.name}' "
                        f"(key: {cache_key[:16]}...)"
                    )

        # Run pre-hooks
        for hook in self._pre_hooks:
            try:
                hook(orchestrator, vertical_class)
            except Exception as e:
                result.add_warning(f"Pre-hook error: {e}")

        # Create context
        context = self._create_context(vertical_class, result)
        if context is None:
            return result
        result.context = context

        # Apply integration steps (Phase 1: Remove legacy path)
        # Step handlers are now mandatory for SOLID compliance
        if self._step_registry is None:
            raise RuntimeError(
                "StepHandlerRegistry required for vertical integration. "
                "Ensure step handlers are initialized. "
                "Use create_integration_pipeline() factory for proper setup."
            )

        # Use step handlers (Phase 3.1 - SOLID compliant single responsibility)
        self._apply_with_step_handlers(orchestrator, vertical_class, context, result)

        # Run post-hooks
        for hook in self._post_hooks:
            try:
                hook(orchestrator, result)
            except Exception as e:
                result.add_warning(f"Post-hook error: {e}")

        logger.debug(
            f"Vertical integration complete: {result.vertical_name}, "
            f"tools={len(result.tools_applied)}, "
            f"middleware={result.middleware_count}, "
            f"safety_patterns={result.safety_patterns_count}"
        )

        # Emit vertical_applied event for observability
        try:
            from victor.core.events import get_observability_bus

            bus = get_observability_bus()
            if bus:
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(
                        bus.emit(
                            topic="vertical.applied",
                            data={
                                "vertical": result.vertical_name,
                                "tools_count": len(result.tools_applied),
                                "middleware_count": result.middleware_count,
                                "safety_patterns_count": result.safety_patterns_count,
                                "prompt_hints_count": result.prompt_hints_count,
                                "workflows_count": result.workflows_count,
                                "rl_learners_count": result.rl_learners_count,
                                "team_specs_count": result.team_specs_count,
                                "success": result.success,
                                "error_count": len(result.errors),
                                "warning_count": len(result.warnings),
                                "category": "vertical",  # Preserve for observability
                            },
                            source="VerticalIntegrationPipeline",
                        )
                    )
                except RuntimeError:
                    # No event loop running
                    logger.debug("No event loop, skipping vertical.applied event emission")
                except Exception as emit_error:
                    logger.debug(f"Failed to create emit task: {emit_error}")
        except Exception as e:
            logger.debug(f"Failed to emit vertical_applied event: {e}")

        # Note: result.persist() available for opt-in audit logging
        # Not called automatically to avoid duplication with EventBus

        # Save to cache (Phase 1: Caching)
        if self._enable_cache and result.success:
            cache_key = self._generate_cache_key(vertical_class)
            if cache_key:
                self._save_to_cache(cache_key, result)

        return result

    def _generate_cache_key(self, vertical: Type["VerticalBase"]) -> Optional[str]:
        """Generate stable cache key from vertical signature.

        The key is based on:
        1. Vertical name
        2. Source code hash (for inline changes)
        3. File modification time (for file changes)

        This ensures cache invalidation when vertical code changes.

        Args:
            vertical: Vertical class

        Returns:
            Cache key string, or None if generation fails
        """
        try:
            # Try to get source file via inspect.getfile()
            # This works for normally imported modules
            try:
                source_file = Path(inspect.getfile(vertical))
            except (TypeError, OSError):
                # For dynamically loaded modules, try module.__file__
                if hasattr(vertical, "__module__"):
                    import sys

                    module_name = vertical.__module__
                    if module_name in sys.modules:
                        module = sys.modules[module_name]
                        if hasattr(module, "__file__") and module.__file__:
                            source_file = Path(module.__file__)
                        else:
                            # Fallback to class-based key
                            return self._generate_class_based_key(vertical)
                    else:
                        return self._generate_class_based_key(vertical)
                else:
                    return self._generate_class_based_key(vertical)

            source_hash = self._hash_source_file(source_file)

            # Combine into key
            key_parts = [
                f"vertical={vertical.name}",
                f"source={source_hash}",
            ]

            key_string = "|".join(key_parts)
            full_hash = hashlib.sha256(key_string.encode()).hexdigest()

            return f"v1_{vertical.name}_{full_hash[:16]}"

        except Exception as e:
            logger.warning(f"Failed to generate cache key: {e}")
            return None

    def _generate_class_based_key(self, vertical: Type["VerticalBase"]) -> Optional[str]:
        """Generate cache key from class properties when source file unavailable.

        This is a fallback for dynamically loaded classes or built-in classes.
        It attempts to include file hash if the module has a __file__ attribute.

        Args:
            vertical: Vertical class

        Returns:
            Cache key string
        """
        try:
            # First, try to get file hash from module
            if hasattr(vertical, "__module__"):
                import sys

                module_name = vertical.__module__

                # Try to get module from sys.modules first
                module = sys.modules.get(module_name)

                # If not in sys.modules, try to get module via inspect
                if not module:
                    try:
                        import inspect

                        # Get module from class using inspect
                        module = inspect.getmodule(vertical)
                    except Exception:
                        pass

                # If we have a module with __file__, use it for hashing
                if module and hasattr(module, "__file__") and module.__file__:
                    try:
                        source_file = Path(module.__file__)
                        if source_file.exists():
                            file_hash = self._hash_source_file(source_file)
                            # Use file hash as primary key component
                            key_parts = [
                                f"vertical={vertical.name}",
                                f"module={module_name}",
                                f"file={file_hash}",
                            ]
                            key_string = "|".join(key_parts)
                            full_hash = hashlib.sha256(key_string.encode()).hexdigest()
                            return f"v2_{vertical.name}_{full_hash[:16]}"
                    except Exception:
                        pass  # Fall back to id-based key

            # Fallback: use class id (won't detect file changes, but stable)
            key_parts = [
                f"vertical={vertical.name}",
                f"id={id(vertical)}",
                f"module={vertical.__module__}",
            ]
            key_string = "|".join(key_parts)
            full_hash = hashlib.sha256(key_string.encode()).hexdigest()
            return f"v3_{vertical.name}_{full_hash[:16]}"

        except Exception:
            # Ultimate fallback
            return f"v4_{vertical.name}_{id(vertical)}"

    def _hash_source_file(self, source_file: Path) -> str:
        """Hash source file content and metadata.

        Args:
            source_file: Path to Python source file

        Returns:
            Hex digest hash combining content and mtime
        """
        try:
            # File content hash (first 16 bytes for performance)
            with open(source_file, "rb") as f:
                content_hash = hashlib.sha256(f.read(16384)).hexdigest()[:16]

            # Modification time (for file changes)
            mtime = source_file.stat().st_mtime_ns
            mtime_hash = hashlib.sha256(str(mtime).encode()).hexdigest()[:8]

            return f"{content_hash}_{mtime_hash}"

        except Exception as e:
            logger.warning(f"Failed to hash source file {source_file}: {e}")
            return f"unknown_{hash(source_file)}"

    def _load_from_cache(self, cache_key: str) -> Optional["IntegrationResult"]:
        """Load integration result from in-memory cache.

        Args:
            cache_key: Cache key

        Returns:
            Cached IntegrationResult or None
        """
        # Simple in-memory cache using JSON serialization
        # Note: Uses to_dict()/from_dict() to avoid pickle issues with
        # unpicklable middleware objects. The context is NOT cached.
        cached_data = self._cache.get(cache_key)
        if cached_data:
            try:
                result = IntegrationResult.from_dict(json.loads(cached_data))

                # Log metrics
                logger.debug(f"Loaded cached integration: {cache_key}")
                return result

            except (json.JSONDecodeError, TypeError, KeyError) as e:
                logger.warning(f"Failed to load from cache: {e}")
                # Clear corrupted cache entry
                del self._cache[cache_key]

        return None

    def _save_to_cache(self, cache_key: str, result: "IntegrationResult") -> None:
        """Save integration result to in-memory cache.

        Uses JSON serialization via to_dict() to avoid pickle issues with
        unpicklable middleware objects. Note: The VerticalContext is NOT
        cached as it contains unpicklable references.

        Args:
            cache_key: Cache key
            result: Integration result to cache
        """
        try:
            # Serialize result using JSON (avoids pickle issues)
            data = json.dumps(result.to_dict())

            # Save to cache
            if not hasattr(self, "_cache"):
                self._cache: Dict[str, str] = {}

            self._cache[cache_key] = data

            # Enforce TTL by storing timestamp
            if not hasattr(self, "_cache_timestamps"):
                self._cache_timestamps: Dict[str, float] = {}
            # Use event loop time if available, otherwise use wall clock time
            try:
                self._cache_timestamps[cache_key] = asyncio.get_event_loop().time()
            except RuntimeError:
                # No event loop running, fall back to wall clock time
                import time

                self._cache_timestamps[cache_key] = time.time()

            logger.debug(f"Cached integration result: {cache_key}")

        except (TypeError, ValueError) as e:
            logger.warning(f"Failed to save to cache: {e}")

    async def apply_async(
        self,
        orchestrator: Any,
        vertical: Union[Type["VerticalBase"], str],
    ) -> IntegrationResult:
        """Apply vertical integration asynchronously (Phase 2.1).

        This async method provides the foundation for parallel step execution.
        Currently executes sequentially, but can be extended for parallel execution.

        Args:
            orchestrator: Orchestrator instance
            vertical: Vertical class or name string

        Returns:
            IntegrationResult with status and metadata

        Example:
            pipeline = VerticalIntegrationPipeline()
            result = await pipeline.apply_async(orchestrator, CodingAssistant)
        """

        # Resolve vertical
        vertical_cls = self._resolve_vertical(vertical)
        if vertical_cls is None:
            result = IntegrationResult(vertical_name=str(vertical))
            result.add_error(f"Vertical not found: {vertical}")
            return result

        # Check cache first (Phase 1)
        if self._enable_cache:
            cache_key = self._generate_cache_key(vertical_cls)
            if cache_key:
                cached_result = self._load_from_cache(cache_key)
                if cached_result:
                    logger.debug(
                        f"Cache HIT for vertical '{vertical_cls.name}' "
                        f"(key: {cache_key[:16]}...)"
                    )
                    return cached_result
                else:
                    logger.debug(
                        f"Cache MISS for vertical '{vertical_cls.name}' "
                        f"(key: {cache_key[:16]}...)"
                    )

        # Create context and result
        result = IntegrationResult(vertical_name=vertical_cls.name)
        context = self._create_context(vertical_cls, result)
        if context is None:
            result.add_error("Failed to create vertical context")
            return result

        # Execute step handlers (Phase 2.1-2.2)
        if self._step_registry is None:
            result.add_error("StepHandlerRegistry required for vertical integration")
            return result

        # Choose execution strategy based on parallel_enabled flag
        if self._parallel_enabled:
            # Parallel execution (Phase 2.2)
            await self._apply_with_step_handlers_parallel(
                orchestrator, vertical_cls, context, result
            )
        else:
            # Sequential execution (Phase 2.1)
            for handler in self._step_registry.get_ordered_handlers():
                try:
                    # Check if handler has async apply method
                    if hasattr(handler, "apply_async"):
                        await handler.apply_async(
                            orchestrator,
                            vertical_cls,
                            context,
                            result,
                            strict_mode=self._strict_mode,
                        )
                    else:
                        # Fallback to sync apply
                        handler.apply(
                            orchestrator,
                            vertical_cls,
                            context,
                            result,
                            strict_mode=self._strict_mode,
                        )
                except Exception as e:
                    if self._strict_mode:
                        result.add_error(f"Step handler '{handler.name}' failed: {e}")
                    else:
                        result.add_warning(f"Step handler '{handler.name}' error: {e}")
                    logger.debug(
                        f"Step handler '{handler.name}' failed: {e}",
                        exc_info=True,
                    )

        # Cache result (Phase 1)
        if self._enable_cache and result.success:
            cache_key = self._generate_cache_key(vertical_cls)
            if cache_key:
                self._save_to_cache(cache_key, result)

        return result

    def _classify_handlers(self, handlers: List[Any]) -> Tuple[List[Any], List[Any]]:
        """Classify handlers into independent and dependent groups (Phase 2.2).

        Independent handlers can run in parallel (no shared state):
        - ToolStepHandler: Reads vertical only
        - PromptStepHandler: Reads vertical only
        - SafetyStepHandler: Reads vertical only

        Dependent handlers must run sequentially:
        - ConfigStepHandler: Depends on tools, prompts
        - MiddlewareStepHandler: Depends on config
        - ExtensionsStepHandler: Depends on all
        - FrameworkStepHandler: Depends on all
        - ContextStepHandler: Must run last

        Args:
            handlers: List of step handlers

        Returns:
            Tuple of (independent_handlers, dependent_handlers)
        """
        independent = []
        dependent = []

        for handler in handlers:
            handler_type = type(handler).__name__

            # Independent handlers (read-only, no side effects)
            if handler_type in [
                "ToolStepHandler",
                "PromptStepHandler",
                "SafetyStepHandler",
            ]:
                independent.append(handler)
            else:
                # Dependent handlers (have side effects or dependencies)
                dependent.append(handler)

        return independent, dependent

    async def _apply_with_step_handlers_parallel(
        self,
        orchestrator: Any,
        vertical: Type["VerticalBase"],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Apply step handlers with parallel execution (Phase 2.2).

        Executes independent handlers concurrently using asyncio.gather,
        then executes dependent handlers sequentially.

        Args:
            orchestrator: Orchestrator instance
            vertical: Vertical class
            context: Vertical context
            result: Integration result
        """
        import asyncio

        if self._step_registry is None:
            return

        handlers = self._step_registry.get_ordered_handlers()
        independent, dependent = self._classify_handlers(handlers)

        # Execute independent handlers in parallel
        if independent:
            logger.debug(f"Executing {len(independent)} independent handlers in parallel")

            tasks = [
                self._run_handler_async(h, orchestrator, vertical, context, result)
                for h in independent
            ]

            # Gather results with exception handling
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check for exceptions
            for i, result_or_exc in enumerate(results):
                if isinstance(result_or_exc, Exception):
                    handler = independent[i]
                    if self._strict_mode:
                        result.add_error(
                            f"Parallel handler '{handler.name}' failed: {result_or_exc}"
                        )
                    else:
                        result.add_warning(
                            f"Parallel handler '{handler.name}' error: {result_or_exc}"
                        )

        # Execute dependent handlers sequentially
        if dependent:
            logger.debug(f"Executing {len(dependent)} dependent handlers sequentially")

            for handler in dependent:
                await self._run_handler_async(handler, orchestrator, vertical, context, result)

    async def _run_handler_async(
        self,
        handler: Any,
        orchestrator: Any,
        vertical: Type["VerticalBase"],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Run a single handler with async support (Phase 2.2).

        Args:
            handler: Step handler instance
            orchestrator: Orchestrator instance
            vertical: Vertical class
            context: Vertical context
            result: Integration result

        Raises:
            Exception: If handler fails and strict_mode is enabled
        """
        try:
            # Check if handler has async apply method
            if hasattr(handler, "apply_async"):
                await handler.apply_async(
                    orchestrator,
                    vertical,
                    context,
                    result,
                    strict_mode=self._strict_mode,
                )
            else:
                # Run sync handler in thread pool to avoid blocking
                import asyncio

                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: handler.apply(
                        orchestrator,
                        vertical,
                        context,
                        result,
                        strict_mode=self._strict_mode,
                    ),
                )
        except Exception as e:
            if self._strict_mode:
                raise
            else:
                logger.debug(
                    f"Handler '{handler.name}' failed: {e}",
                    exc_info=True,
                )

    def _apply_with_step_handlers(
        self,
        orchestrator: Any,
        vertical: Type["VerticalBase"],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Apply vertical using step handlers.

        This is the new Phase 3.1 implementation using focused step handler
        classes for each concern.

        Args:
            orchestrator: Orchestrator instance
            vertical: Vertical class
            context: Vertical context
            result: Integration result
        """
        if self._step_registry is None:
            return

        # Execute all step handlers in order
        for handler in self._step_registry.get_ordered_handlers():
            try:
                handler.apply(
                    orchestrator,
                    vertical,
                    context,
                    result,
                    strict_mode=self._strict_mode,
                )
            except Exception as e:
                if self._strict_mode:
                    result.add_error(f"Step handler '{handler.name}' failed: {e}")
                else:
                    result.add_warning(f"Step handler '{handler.name}' error: {e}")
                logger.debug(
                    f"Step handler '{handler.name}' failed: {e}",
                    exc_info=True,
                )

    # Legacy method removed in Phase 1 refactoring
    # Step handlers now provide SOLID-compliant single responsibility implementation
    # See: victor/framework/step_handlers.py

    def _resolve_vertical(
        self, vertical: Union[Type["VerticalBase"], str]
    ) -> Optional[Type["VerticalBase"]]:
        """Resolve vertical from name or return class.

        Args:
            vertical: Vertical class or name string

        Returns:
            Vertical class or None if not found
        """
        if isinstance(vertical, str):
            from victor.core.verticals.base import VerticalRegistry

            resolved = VerticalRegistry.get(vertical)
            if resolved is None:
                # Try case-insensitive match
                for name in VerticalRegistry.list_names():
                    if name.lower() == vertical.lower():
                        return VerticalRegistry.get(name)
            return resolved
        return vertical

    def _create_context(
        self, vertical: Type["VerticalBase"], result: IntegrationResult
    ) -> Optional[VerticalContext]:
        """Create the VerticalContext for this vertical.

        Args:
            vertical: Vertical class
            result: Result to update

        Returns:
            VerticalContext or None on error
        """
        try:
            config = vertical.get_config()
            context = create_vertical_context(
                name=vertical.name,
                config=config,
            )
            return context
        except Exception as e:
            result.add_error(f"Failed to create context: {e}")
            return None

    # Legacy tool methods removed in Phase 1 refactoring
    # Replaced by ToolStepHandler in victor/framework/step_handlers.py
    # Tiered tools handled by TieredConfigStepHandler

    # Legacy system prompt method removed in Phase 1 refactoring
    # Replaced by PromptStepHandler in victor/framework/step_handlers.py

    # Legacy stages method removed in Phase 1 refactoring
    # Replaced by ConfigStepHandler in victor/framework/step_handlers.py

    # Legacy extension methods removed in Phase 1 refactoring
    # Replaced by ExtensionsStepHandler in victor/framework/step_handlers.py
    # which coordinates:
    #   - MiddlewareStepHandler for middleware
    #   - SafetyStepHandler for safety extensions
    #   - PromptStepHandler for prompt contributors
    #   - ConfigStepHandler for mode config and tool deps

    # Legacy attach_context removed in Phase 1 refactoring
    # Replaced by ContextStepHandler in victor/framework/step_handlers.py

    # Legacy framework integration methods removed in Phase 1 refactoring
    # Replaced by FrameworkStepHandler in victor/framework/step_handlers.py
    # which handles workflows, RL config, and team specs


# =============================================================================
# Factory Functions
# =============================================================================


def create_integration_pipeline(
    strict: bool = False,
    use_step_handlers: bool = True,
    enable_cache: bool = True,
    enable_parallel: bool = False,
) -> VerticalIntegrationPipeline:
    """Create a vertical integration pipeline with feature flags (Phase 2).

    Args:
        strict: Whether to use strict mode
        use_step_handlers: If True, use step handlers (Phase 3.1)
        enable_cache: Enable configuration caching (Phase 1, default: True)
        enable_parallel: Enable parallel execution (Phase 2.2, default: False)

    Returns:
        Configured VerticalIntegrationPipeline

    Example:
        # Basic pipeline
        pipeline = create_integration_pipeline()

        # With parallel execution (Phase 2.2)
        pipeline = create_integration_pipeline(enable_parallel=True)

        result = await pipeline.apply_async(orchestrator, CodingAssistant)
    """
    return VerticalIntegrationPipeline(
        strict_mode=strict,
        use_step_handlers=use_step_handlers,
        enable_cache=enable_cache,
        parallel_enabled=enable_parallel,
    )


def create_integration_pipeline_with_handlers(
    strict: bool = False,
    custom_handlers: Optional[List[Any]] = None,
    enable_parallel: bool = False,
) -> VerticalIntegrationPipeline:
    """Create a pipeline with custom step handlers.

    This factory function allows adding custom step handlers to the
    default set for specialized integration needs.

    Args:
        strict: Whether to use strict mode
        custom_handlers: Optional list of additional step handlers
        enable_parallel: Enable parallel execution (Phase 2.2, default: False)

    Returns:
        Configured VerticalIntegrationPipeline with custom handlers

    Example:
        from victor.framework.step_handlers import BaseStepHandler

        class MyCustomHandler(BaseStepHandler):
            @property
            def name(self) -> str:
                return "my_custom"

            @property
            def order(self) -> int:
                return 55  # Between extensions and framework

            def _do_apply(self, orchestrator, vertical, context, result):
                # Custom integration logic
                pass

        pipeline = create_integration_pipeline_with_handlers(
            custom_handlers=[MyCustomHandler()],
            enable_parallel=True,
        )
    """
    try:
        from victor.framework.step_handlers import StepHandlerRegistry

        registry = StepHandlerRegistry.default()

        if custom_handlers:
            for handler in custom_handlers:
                registry.add_handler(handler)

        return VerticalIntegrationPipeline(
            strict_mode=strict,
            step_registry=registry,
            use_step_handlers=True,
            parallel_enabled=enable_parallel,
        )
    except ImportError:
        # Fallback to legacy if step handlers not available
        return VerticalIntegrationPipeline(
            strict_mode=strict,
            use_step_handlers=False,
        )


def apply_vertical(
    orchestrator: Any,
    vertical: Union[Type["VerticalBase"], str],
) -> IntegrationResult:
    """Apply a vertical to an orchestrator.

    Convenience function for one-shot vertical application.

    Args:
        orchestrator: Orchestrator instance
        vertical: Vertical class or name

    Returns:
        IntegrationResult
    """
    pipeline = VerticalIntegrationPipeline()
    return pipeline.apply(orchestrator, vertical)


__all__ = [
    # Core classes
    "IntegrationResult",
    "ExtensionLoadErrorInfo",
    "ValidationStatus",
    "VerticalIntegrationPipeline",
    "OrchestratorVerticalProtocol",
    # Extension handler registry (OCP compliance)
    "ExtensionHandlerInfo",
    "ExtensionHandlerRegistry",
    "get_extension_handler_registry",
    "register_extension_handler",
    # Factory functions
    "create_integration_pipeline",
    "create_integration_pipeline_with_handlers",
    "apply_vertical",
]
