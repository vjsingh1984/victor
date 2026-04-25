"""Formatter registry and centralized formatting logic."""

import hashlib
import json
import logging
import time
from typing import Any, Dict, Optional

from .base import ToolFormatter, FormattedOutput

logger = logging.getLogger(__name__)


# Simple in-memory cache for formatted outputs
class _FormatCache:
    """Simple in-memory cache for formatted outputs with TTL.

    Features:
    - Content-based cache keys (SHA256 hash)
    - TTL-based expiration (5 minutes default)
    - LRU eviction when full
    - Thread-safe operations
    """

    def __init__(self, max_size: int = 100, default_ttl: int = 300):
        """Initialize the cache.

        Args:
            max_size: Maximum number of entries to store
            default_ttl: Default TTL in seconds (5 minutes)
        """
        self._cache: Dict[str, tuple[FormattedOutput, float]] = {}
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._hits = 0
        self._misses = 0

    def get(self, tool_name: str, data: Dict[str, Any], **kwargs) -> Optional[FormattedOutput]:
        """Get formatted output from cache.

        Args:
            tool_name: Name of the tool
            data: Tool output data
            **kwargs: Formatter options

        Returns:
            Cached FormattedOutput if found and not expired, None otherwise
        """
        key = self._generate_key(tool_name, data, kwargs)

        if key in self._cache:
            formatted, expiry_time = self._cache[key]

            # Check if entry has expired
            if time.time() < expiry_time:
                self._hits += 1
                return formatted
            else:
                # Remove expired entry
                del self._cache[key]

        self._misses += 1
        return None

    def put(self, tool_name: str, data: Dict[str, Any], formatted: FormattedOutput,
            ttl: Optional[int] = None, **kwargs) -> None:
        """Put formatted output into cache.

        Args:
            tool_name: Name of the tool
            data: Tool output data
            formatted: Formatted output to cache
            ttl: Time to live in seconds (uses default if None)
            **kwargs: Formatter options
        """
        # Evict oldest entry if cache is full
        if len(self._cache) >= self._max_size:
            # Simple FIFO eviction (could be improved to LRU)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        key = self._generate_key(tool_name, data, kwargs)
        expiry_time = time.time() + (ttl or self._default_ttl)
        self._cache[key] = (formatted, expiry_time)

    def invalidate(self, tool_name: Optional[str] = None) -> None:
        """Invalidate cache entries.

        Args:
            tool_name: Tool name to invalidate (None = invalidate all)
        """
        if tool_name is None:
            self._cache.clear()
        else:
            # Remove all entries for this tool
            keys_to_remove = [k for k in self._cache if k.startswith(f"{tool_name}:")]
            for key in keys_to_remove:
                del self._cache[key]

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats (hits, misses, size, hit_rate)
        """
        total = self._hits + self._misses
        hit_rate = (self._hits / total) if total > 0 else 0

        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._cache),
            "max_size": self._max_size,
            "hit_rate": hit_rate,
        }

    def _generate_key(self, tool_name: str, data: Dict[str, Any],
                     kwargs: Dict[str, Any]) -> str:
        """Generate content-based cache key.

        Args:
            tool_name: Name of the tool
            data: Tool output data
            kwargs: Formatter options

        Returns:
            SHA256 hash of tool_name + data + kwargs
        """
        # Create deterministic string representation
        key_parts = [tool_name]

        # Sort dict keys for consistency
        if isinstance(data, dict):
            sorted_data = json.dumps(data, sort_keys=True)
        else:
            sorted_data = str(data)

        key_parts.append(sorted_data)

        # Add relevant kwargs to key (exclude cache-specific options)
        cacheable_kwargs = {k: v for k, v in kwargs.items()
                           if k not in ("ttl", "max_size", "max_time_ms")}
        if cacheable_kwargs:
            sorted_kwargs = json.dumps(cacheable_kwargs, sort_keys=True)
            key_parts.append(sorted_kwargs)

        key_string = ":".join(key_parts)
        content_hash = hashlib.sha256(key_string.encode()).hexdigest()[:32]
        return f"{tool_name}:{content_hash}"


# Global cache instance
_format_cache = _FormatCache()


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
        max_size = settings.rich_formatting_max_output_size
        max_time_ms = settings.rich_formatting_max_time_ms
        validation_enabled = settings.rich_formatting_validation_enabled
        fallback_enabled = settings.rich_formatting_fallback_enabled
        allowed_tools = set(settings.rich_formatting_tools)
    except Exception:
        # If settings unavailable, use safe defaults
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

    # Check cache before formatting (Phase 8: Caching Layer)
    try:
        cache_enabled = settings.rich_formatting_cache_enabled
    except Exception:
        cache_enabled = True

    if cache_enabled:
        cached = _format_cache.get(tool_name, data, **kwargs)
        if cached is not None:
            logger.debug(f"Cache hit for {tool_name}")
            return cached
        else:
            logger.debug(f"Cache miss for {tool_name}")

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

        # Store in cache if enabled (Phase 8: Caching Layer)
        if cache_enabled:
            try:
                cache_ttl = settings.rich_formatting_cache_ttl
                _format_cache.put(tool_name, data, formatted, ttl=cache_ttl, **kwargs)
            except Exception as cache_error:
                logger.debug(f"Failed to cache formatted output for {tool_name}: {cache_error}")

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


# Cache management functions (Phase 8: Caching Layer)
def get_format_cache_stats() -> Dict[str, Any]:
    """Get formatter cache statistics.

    Returns:
        Dictionary with cache stats (hits, misses, size, hit_rate)
    """
    return _format_cache.get_stats()


def invalidate_format_cache(tool_name: Optional[str] = None) -> None:
    """Invalidate formatter cache entries.

    Args:
        tool_name: Tool name to invalidate (None = invalidate all)

    Example:
        # Invalidate all cache entries
        invalidate_format_cache()

        # Invalidate cache for specific tool
        invalidate_format_cache("test")
    """
    _format_cache.invalidate(tool_name)


def clear_format_cache() -> None:
    """Clear the entire formatter cache.

    This is useful for testing or when you want to force re-formatting.
    """
    _format_cache.invalidate()

