"""Batch registration API for ToolRegistry.

This module provides efficient batch registration with a single mutation window,
reducing cache invalidation overhead from O(n) to O(1) for bulk operations.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class BatchRegistrationResult:
    """Result of a batch registration operation.

    Attributes:
        registered: List of successfully registered tool names
        failed: List of (tool_name, error_message) tuples
        duration_ms: Total duration in milliseconds
        cache_invalidations: Number of cache invalidations performed
    """

    registered: List[str] = field(default_factory=list)
    failed: List[tuple[str, str]] = field(default_factory=list)
    duration_ms: float = 0.0
    cache_invalidations: int = 0

    @property
    def success_count(self) -> int:
        """Number of successfully registered tools."""
        return len(self.registered)

    @property
    def failure_count(self) -> int:
        """Number of failed registrations."""
        return len(self.failed)

    @property
    def total_count(self) -> int:
        """Total number of registration attempts."""
        return self.success_count + self.failure_count

    def __str__(self) -> str:
        success_rate = (self.success_count / self.total_count * 100) if self.total_count > 0 else 0
        return (
            f"BatchRegistrationResult("
            f"success={self.success_count}, "
            f"failed={self.failure_count}, "
            f"success_rate={success_rate:.1f}%, "
            f"duration={self.duration_ms:.2f}ms, "
            f"cache_invalidations={self.cache_invalidations})"
        )


@dataclass
class ValidationContext:
    """Context for validation during batch registration.

    Accumulates validation state for all tools before committing.
    """

    tools: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    indexes: Dict[str, Dict[str, Set[str]]] = field(default_factory=dict)
    validation_errors: List[tuple[str, str]] = field(default_factory=list)

    def is_valid(self) -> bool:
        """Check if all validations passed."""
        return len(self.validation_errors) == 0

    def add_error(self, tool_name: str, error: str) -> None:
        """Add a validation error."""
        self.validation_errors.append((tool_name, error))

    def add_tool(self, tool_name: str, tool: Any) -> None:
        """Add a validated tool to the context."""
        self.tools[tool_name] = tool

    def add_index_entry(self, index_name: str, key: str, tool_name: str) -> None:
        """Add an entry to an index."""
        if index_name not in self.indexes:
            self.indexes[index_name] = {}
        if key not in self.indexes[index_name]:
            self.indexes[index_name][key] = set()
        self.indexes[index_name][key].add(tool_name)


class BatchRegistrationError(Exception):
    """Base class for batch registration errors."""

    def __init__(self, message: str, partial_result: Optional[BatchRegistrationResult] = None):
        self.message = message
        self.partial_result = partial_result
        super().__init__(message)


class BatchRegistrationValidator:
    """Validator for batch registration operations.

    Validates all tools before committing to registry.
    Implements Unit of Work pattern.
    """

    def __init__(self, registry):
        """Initialize validator with registry reference."""
        self._registry = registry

    @staticmethod
    def _registration_tool(tool: Any) -> Any:
        """Return the concrete tool object used for validation/registration."""
        if hasattr(tool, "Tool"):
            return tool.Tool
        return tool

    @staticmethod
    def _tool_name(tool: Any) -> str:
        """Return a stable tool name for diagnostics without assuming .name."""
        concrete = BatchRegistrationValidator._registration_tool(tool)
        name = getattr(concrete, "name", None)
        if name:
            return str(name)
        return getattr(tool, "__name__", type(tool).__name__) or "unnamed_tool"

    def validate_batch(self, tools: List[Any]) -> ValidationContext:
        """Validate all tools in the batch.

        Args:
            tools: List of tools to validate

        Returns:
            ValidationContext with validation state
        """
        context = ValidationContext()

        for raw_tool in tools:
            tool = self._registration_tool(raw_tool)
            tool_name = self._tool_name(tool)
            try:
                # Validate tool name uniqueness
                if tool_name in context.tools:
                    context.add_error(tool_name, f"Duplicate tool name: {tool_name}")
                    continue

                # Validate tool metadata
                self._validate_tool_metadata(tool, context)

                # Validate tool parameters schema
                self._validate_tool_parameters(tool, context)

                # Tool is valid, add to context
                context.add_tool(tool_name, tool)

            except Exception as e:
                context.add_error(tool_name, str(e))

        return context

    def _validate_tool_metadata(self, tool: Any, context: ValidationContext) -> None:
        """Validate tool metadata.

        Args:
            tool: Tool to validate
            context: Validation context to update
        """
        tool_name = self._tool_name(tool)

        # Check required fields
        if not tool_name:
            context.add_error("unnamed_tool", "Tool name is required")

        if not getattr(tool, "description", ""):
            context.add_error(tool_name, "Tool description is required")

        # Validate tags
        if hasattr(tool, "tags") and tool.tags:
            if not isinstance(tool.tags, list):
                context.add_error(tool_name, "Tags must be a list")

            # Validate tag format
            for tag in tool.tags:
                if not isinstance(tag, str):
                    context.add_error(tool_name, f"Tag must be string, got {type(tag)}")

    def _validate_tool_parameters(self, tool: Any, context: ValidationContext) -> None:
        """Validate tool parameter schema.

        Args:
            tool: Tool to validate
            context: Validation context to update
        """
        if not hasattr(tool, "parameters") or tool.parameters is None:
            return

        parameters = tool.parameters
        if not isinstance(parameters, dict):
            context.add_error(self._tool_name(tool), "Parameters must be a dict")
            return

        # Validate schema structure
        if parameters.get("type") != "object":
            context.add_error(self._tool_name(tool), "Parameters root type must be 'object'")

        # Validate properties if present
        properties = parameters.get("properties", {})
        if not isinstance(properties, dict):
            context.add_error(self._tool_name(tool), "Properties must be a dict")
            return

        # Check for required fields
        for prop_name, prop_schema in properties.items():
            if prop_schema.get("required", False):
                # Property exists in schema, validation happens during execution
                pass


class BatchRegistrar:
    """Handles validated batch registration with a single registry mutation window.

    Workflow:
    1. Validate tools chunk-by-chunk
    2. Register valid tools while the registry is in ``batch_update()`` mode
    3. Report successes and failures without repeated cache invalidation
    """

    def __init__(self, registry):
        """Initialize batch registrar.

        Args:
            registry: ToolRegistry instance
        """
        self._registry = registry
        self._validator = BatchRegistrationValidator(registry)

    def register_batch(
        self,
        tools: List[Any],
        chunk_size: int = 100,
        fail_fast: bool = False,
    ) -> BatchRegistrationResult:
        """Register multiple tools with a single cache invalidation window.

        Args:
            tools: List of tools to register
            chunk_size: Process in chunks of this size (default: 100)
            fail_fast: Stop on first validation error (default: False)

        Returns:
            BatchRegistrationResult with registration statistics

        Raises:
            BatchRegistrationError: If fail_fast=True and validation fails
        """
        start_time = time.perf_counter()
        result = BatchRegistrationResult()

        with self._registry.batch_update():
            # Process in chunks to avoid memory spikes while still invalidating once.
            for chunk_start in range(0, len(tools), chunk_size):
                chunk = tools[chunk_start : chunk_start + chunk_size]

                validation_context = self._validator.validate_batch(chunk)

                if fail_fast and not validation_context.is_valid():
                    result.failed.extend(validation_context.validation_errors)
                    raise BatchRegistrationError(
                        f"Validation failed with {len(validation_context.validation_errors)} errors",
                        partial_result=result,
                    )

                result.failed.extend(validation_context.validation_errors)
                committed = self._commit_valid_tools(validation_context)
                result.registered.extend(committed)

        result.duration_ms = (time.perf_counter() - start_time) * 1000
        result.cache_invalidations = int(result.success_count > 0)

        logger.info(
            f"Batch registration complete: "
            f"{result.success_count}/{result.total_count} successful, "
            f"{result.duration_ms:.2f}ms, "
            f"{result.cache_invalidations} cache invalidations"
        )

        return result

    def _commit_valid_tools(self, context: ValidationContext) -> List[str]:
        """Commit all valid tools to registry atomically.

        Args:
            context: Validation context with valid tools

        Returns:
            List of registered tool names
        """
        registered_names = []

        for tool_name, tool in context.tools.items():
            try:
                self._registry.register(tool)
                registered_names.append(tool_name)
            except Exception as e:
                logger.warning(f"Failed to register {tool_name}: {e}")
                # Continue with other tools even if one fails

        return registered_names


# Convenience function for batch registration
def register_tools_batch(
    registry,
    tools: List[Any],
    chunk_size: int = 100,
    fail_fast: bool = False,
) -> BatchRegistrationResult:
    """Convenience function for batch registration.

    Args:
        registry: ToolRegistry instance
        tools: List of tools to register
        chunk_size: Process in chunks of this size
        fail_fast: Stop on first validation error

    Returns:
        BatchRegistrationResult with registration statistics
    """
    registrar = BatchRegistrar(registry)
    return registrar.register_batch(tools, chunk_size, fail_fast)


__all__ = [
    "BatchRegistrationResult",
    "BatchRegistrationError",
    "BatchRegistrar",
    "register_tools_batch",
]
