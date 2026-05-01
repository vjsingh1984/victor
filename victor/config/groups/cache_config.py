"""Cache configuration for tool results, selection, and HTTP connections.

This module contains settings for:
- Tool result caching and deduplication
- Tool selection cache (embedding-based)
- Generic result caching
- HTTP connection pooling
"""

from typing import List
from pydantic import BaseModel, Field, field_validator


class CacheSettings(BaseModel):
    """Cache settings for tools, selection, and network resources.

    Controls various caching mechanisms that improve performance by
    avoiding redundant computation and network calls.
    """

    # ==========================================================================
    # Tool Result Caching (opt-in per tool)
    # ==========================================================================
    # Caches tool results to avoid redundant executions for identical inputs.
    # Uses TTL-based expiration with allowlist for opt-in safety.
    tool_cache_enabled: bool = True
    tool_cache_ttl: int = 600  # seconds (10 minutes)
    # Note: tool_cache_dir now uses get_project_paths().global_cache_dir
    tool_cache_allowlist: List[str] = Field(
        default_factory=lambda: [
            "code_search",
            "semantic_code_search",
            "list_directory",
            "plan_files",
        ]
    )

    # ==========================================================================
    # Generic Runtime Cache (feature-flagged integration path)
    # ==========================================================================
    # Generic cache for non-tool payloads (e.g., computed results, API responses).
    # Disabled by default - requires explicit opt-in for each use case.
    generic_result_cache_enabled: bool = False
    generic_result_cache_ttl: int = 300  # seconds (5 minutes)

    # ==========================================================================
    # Tool Selection Result Cache (embedding-based)
    # ==========================================================================
    # Caches semantic tool selection results to avoid repeated embedding computation.
    # Typical 20-40% latency reduction for conversational agents.
    # Short TTL ensures fresh selections as context evolves.
    tool_selection_cache_enabled: bool = True
    tool_selection_cache_ttl: int = 300  # seconds (5 minutes)

    # ==========================================================================
    # Shared HTTP Connection Pool (feature-flagged integration path)
    # ==========================================================================
    # Connection pooling for network tools to reduce connection overhead.
    # Disabled by default - requires feature flag enablement.
    http_connection_pool_enabled: bool = False
    http_connection_pool_max_connections: int = 100
    http_connection_pool_max_connections_per_host: int = 10
    http_connection_pool_connection_timeout: int = 30  # seconds
    http_connection_pool_total_timeout: int = 60  # seconds

    @field_validator("tool_cache_ttl")
    @classmethod
    def validate_tool_cache_ttl(cls, v: int) -> int:
        """Validate tool cache TTL is non-negative.

        Args:
            v: Cache TTL in seconds

        Returns:
            Validated TTL

        Raises:
            ValueError: If TTL is negative
        """
        if v < 0:
            raise ValueError("tool_cache_ttl must be >= 0")
        return v

    @field_validator("generic_result_cache_ttl")
    @classmethod
    def validate_generic_cache_ttl(cls, v: int) -> int:
        """Validate generic cache TTL is non-negative.

        Args:
            v: Cache TTL in seconds

        Returns:
            Validated TTL

        Raises:
            ValueError: If TTL is negative
        """
        if v < 0:
            raise ValueError("generic_result_cache_ttl must be >= 0")
        return v

    @field_validator("tool_selection_cache_ttl")
    @classmethod
    def validate_selection_cache_ttl(cls, v: int) -> int:
        """Validate selection cache TTL is non-negative.

        Args:
            v: Cache TTL in seconds

        Returns:
            Validated TTL

        Raises:
            ValueError: If TTL is negative
        """
        if v < 0:
            raise ValueError("tool_selection_cache_ttl must be >= 0")
        return v

    @field_validator("http_connection_pool_max_connections")
    @classmethod
    def validate_max_connections(cls, v: int) -> int:
        """Validate max connections is positive.

        Args:
            v: Maximum number of connections

        Returns:
            Validated max connections

        Raises:
            ValueError: If max connections is not positive
        """
        if v < 1:
            raise ValueError("http_connection_pool_max_connections must be >= 1")
        return v

    @field_validator("http_connection_pool_max_connections_per_host")
    @classmethod
    def validate_max_connections_per_host(cls, v: int) -> int:
        """Validate max connections per host is positive.

        Args:
            v: Maximum number of connections per host

        Returns:
            Validated max connections per host

        Raises:
            ValueError: If max connections per host is not positive
        """
        if v < 1:
            raise ValueError("http_connection_pool_max_connections_per_host must be >= 1")
        return v

    @field_validator("http_connection_pool_connection_timeout")
    @classmethod
    def validate_connection_timeout(cls, v: int) -> int:
        """Validate connection timeout is positive.

        Args:
            v: Connection timeout in seconds

        Returns:
            Validated timeout

        Raises:
            ValueError: If timeout is not positive
        """
        if v <= 0:
            raise ValueError("http_connection_pool_connection_timeout must be > 0")
        return v

    @field_validator("http_connection_pool_total_timeout")
    @classmethod
    def validate_total_timeout(cls, v: int) -> int:
        """Validate total timeout is positive.

        Args:
            v: Total timeout in seconds

        Returns:
            Validated timeout

        Raises:
            ValueError: If timeout is not positive
        """
        if v <= 0:
            raise ValueError("http_connection_pool_total_timeout must be > 0")
        return v
