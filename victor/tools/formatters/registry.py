"""Formatter registry and centralized formatting logic."""

import logging
import time
from typing import Any, Dict, Optional

from .base import ToolFormatter, FormattedOutput

logger = logging.getLogger(__name__)


def _is_rich_formatting_enabled() -> bool:
    """Check if Rich formatting is enabled via feature flag or settings.

    The feature flag acts as a master switch that can disable Rich formatting:
    - If explicitly disabled via feature flag → return False (overrides settings)
    - Otherwise, use settings.rich_formatting_enabled (defaults to True)
    - If feature flag check fails → fall back to settings

    Returns:
        True if Rich formatting is enabled, False otherwise
    """
    try:
        from victor.core.feature_flags import get_feature_flag_manager, FeatureFlag

        manager = get_feature_flag_manager()

        # Check if feature flag is explicitly disabled (master override)
        if FeatureFlag.USE_RICH_FORMATTING in manager._runtime_overrides:
            # Feature flag is explicitly set, use that value
            return manager._runtime_overrides[FeatureFlag.USE_RICH_FORMATTING]

        # No explicit override, use settings (which defaults to True)
        from victor.config.tool_settings import get_toolSettings
        settings = get_toolSettings()
        return settings.rich_formatting_enabled
    except Exception:
        # If checks fail, fall back to settings
        try:
            from victor.config.tool_settings import get_toolSettings
            settings = get_toolSettings()
            return settings.rich_formatting_enabled
        except Exception:
            # If that also fails, enable by default for safety
            return True

# Global formatter registry
_FORMATTER_MAP: Dict[str, ToolFormatter] = {}


class FormatterRegistry:
    """Central registry for tool output formatters.

    This is a singleton that manages all formatters and provides
    fallback behavior when formatters fail.
    """

    _instance: Optional['FormatterRegistry'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._formatters: Dict[str, ToolFormatter] = {}
        return cls._instance

    def register(self, tool_name: str, formatter: ToolFormatter) -> None:
        """Register a formatter for a tool.

        Args:
            tool_name: Name of the tool (test, code_search, git, etc.)
            formatter: Formatter instance to register
        """
        self._formatters[tool_name] = formatter
        logger.debug(f"Registered formatter for tool: {tool_name}")

    def get_formatter(self, tool_name: str) -> ToolFormatter:
        """Get formatter for a tool, with fallback to generic.

        Args:
            tool_name: Name of the tool

        Returns:
            Formatter instance (generic fallback if not found)
        """
        formatter = self._formatters.get(tool_name)
        if formatter:
            return formatter

        # Fallback to generic formatter
        from .generic import GenericFormatter
        logger.debug(f"No formatter found for {tool_name}, using generic fallback")
        return GenericFormatter()

    def list_formatters(self) -> list[str]:
        """List all registered formatter names.

        Returns:
            List of tool names with registered formatters
        """
        return list(self._formatters.keys())


# Singleton accessor
def get_formatter_registry() -> FormatterRegistry:
    """Get the global formatter registry singleton.

    Returns:
        The singleton FormatterRegistry instance
    """
    return FormatterRegistry()


# Module-level convenience function for formatting
def format_tool_output(
    tool_name: str,
    data: Dict[str, Any],
    **kwargs
) -> FormattedOutput:
    """Format tool output using registered formatter with production guards.

    This is the main entry point for tools to format their output.
    It handles formatter selection, validation, performance guards, and error recovery.

    Args:
        tool_name: Name of the tool (test, code_search, git, etc.)
        data: Tool output data to format
        **kwargs: Formatter-specific options (max_failures, max_files, etc.)

    Returns:
        FormattedOutput with Rich markup (or plain text on error)

    Example:
        test_data = {"summary": {"total": 10, "passed": 8}, "failures": []}
        formatted = format_tool_output("test", test_data, max_failures=5)
        print(formatted.content)

    Performance Guards:
        - Max output size: 1MB (configurable via settings)
        - Max formatting time: 200ms (configurable via settings)
        - Input validation: Enabled by default
        - Fallback behavior: Plain text on errors
    """
    # Get settings for production guards
    try:
        from victor.config.tool_settings import get_tool_settings
        settings = get_tool_settings()
        rich_enabled = settings.rich_formatting_enabled
        max_size = settings.rich_formatting_max_output_size
        max_time_ms = settings.rich_formatting_max_time_ms
        validation_enabled = settings.rich_formatting_validation_enabled
        fallback_enabled = settings.rich_formatting_fallback_enabled
        allowed_tools = set(settings.rich_formatting_tools)
    except Exception:
        # If settings unavailable, use safe defaults
        rich_enabled = True
        max_size = 1_000_000
        max_time_ms = 200
        validation_enabled = True
        fallback_enabled = True
        allowed_tools = {
            "test", "pytest", "run_tests",
            "code_search", "semantic_code_search",
            "git", "http", "https",
            "database", "db", "sql",
            "refactor", "refactoring",
            "docker", "security", "security_scan",
        }

    # Check if Rich formatting is enabled for this tool (feature flag + tool whitelist)
    if not _is_rich_formatting_enabled() or tool_name not in allowed_tools:
        # Return plain text output
        return FormattedOutput(
            content=str(data),
            format_type="plain",
            summary=None,
            contains_markup=False,
        )

    registry = get_formatter_registry()
    formatter = registry.get_formatter(tool_name)

    # Performance guard: Check input size before formatting
    try:
        input_size = len(str(data))
        if input_size > max_size:
            logger.warning(
                f"{tool_name} input too large for Rich formatting: {input_size} bytes "
                f"(limit: {max_size} bytes)"
            )
            return FormattedOutput(
                content=str(data),
                format_type="plain",
                summary=f"{tool_name} output too large for Rich formatting",
                contains_markup=False,
            )
    except Exception as e:
        logger.debug(f"Could not calculate input size for {tool_name}: {e}")

    # Input validation if enabled
    if validation_enabled:
        try:
            if not formatter.validate_input(data):
                logger.warning(f"Invalid input for {tool_name} formatter")
                return FormattedOutput(
                    content=str(data),
                    format_type="plain",
                    summary=f"Invalid {tool_name} data",
                    contains_markup=False,
                )
        except Exception as e:
            logger.debug(f"Validation error for {tool_name}: {e}")
            if not fallback_enabled:
                return FormattedOutput(
                    content=str(data),
                    format_type="plain",
                    summary=f"{tool_name} validation failed",
                    contains_markup=False,
                )

    # Format with performance guard
    start_time = time.time()
    try:
        formatted = formatter.format(data, **kwargs)

        # Performance guard: Check formatting time
        elapsed_ms = (time.time() - start_time) * 1000
        if elapsed_ms > max_time_ms:
            logger.warning(
                f"{tool_name} formatting exceeded timeout: {elapsed_ms:.0f}ms "
                f"(limit: {max_time_ms}ms)"
            )
            # Still return the result, but log the warning

        # Performance guard: Check output size
        try:
            output_size = len(formatted.content)
            if output_size > max_size:
                logger.warning(
                    f"{tool_name} formatted output too large: {output_size} bytes "
                    f"(limit: {max_size} bytes)"
                )
                # Truncate if needed
                formatted.content = formatted.content[:max_size]
                formatted.line_count = len(formatted.content.splitlines())
        except Exception as e:
            logger.debug(f"Could not calculate output size for {tool_name}: {e}")

        return formatted

    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(
            f"Formatter error for {tool_name} after {elapsed_ms:.0f}ms: {e}",
            exc_info=True
        )

        # Try fallback if enabled
        if fallback_enabled:
            fallback = formatter.get_fallback()
            if fallback:
                try:
                    return fallback.format(data, **kwargs)
                except Exception as fallback_error:
                    logger.error(f"Fallback formatter also failed for {tool_name}: {fallback_error}")

        # Final fallback: plain text
        return FormattedOutput(
            content=str(data),
            format_type="plain",
            summary=f"{tool_name} formatting failed",
            contains_markup=False,
        )
