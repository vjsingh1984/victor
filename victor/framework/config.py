"""Simplified configuration for the Victor framework API.

For most use cases, you don't need this module - use Agent.create()
keyword arguments instead. AgentConfig is for advanced configurations
that go beyond the simple API.

Example:
    # Simple use case - no config needed
    agent = await Agent.create(provider="anthropic")

    # Advanced use case
    config = AgentConfig(
        tool_budget=100,
        max_iterations=50,
        enable_semantic_search=True,
    )
    agent = await Agent.create(config=config)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class AgentConfig:
    """Advanced configuration options for agents.

    This class provides fine-grained control over agent behavior.
    For simple use cases, prefer Agent.create() keyword arguments.

    Attributes:
        tool_budget: Maximum number of tool calls allowed per session
        max_iterations: Maximum total agent loop iterations
        enable_parallel_tools: Allow multiple tools to run concurrently
        max_concurrent_tools: Maximum concurrent tool executions
        max_context_chars: Maximum context window characters (provider default if None)
        enable_context_compaction: Automatically compact context when nearing limits
        streaming_timeout: Timeout in seconds for streaming responses
        enable_semantic_search: Use embedding-based tool selection
        enable_code_correction: Auto-correct malformed code in responses
        enable_analytics: Collect usage analytics
        enable_tool_cache: Cache idempotent tool results
        tool_cache_ttl: Cache TTL in seconds
        max_recovery_attempts: Maximum retry attempts on failures
        extra: Additional settings passed through to Settings

    Example:
        config = AgentConfig(
            tool_budget=100,
            enable_semantic_search=True,
            extra={"custom_setting": "value"}
        )
    """

    # Tool execution
    tool_budget: int = 50
    max_iterations: int = 25
    enable_parallel_tools: bool = True
    max_concurrent_tools: int = 5

    # Context management
    max_context_chars: Optional[int] = None
    enable_context_compaction: bool = True

    # Streaming
    streaming_timeout: float = 300.0

    # Features
    enable_semantic_search: bool = False
    enable_code_correction: bool = True
    enable_analytics: bool = True

    # Caching
    enable_tool_cache: bool = True
    tool_cache_ttl: int = 600

    # Recovery
    max_recovery_attempts: int = 3

    # Additional settings (pass-through to Settings)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_settings_dict(self) -> Dict[str, Any]:
        """Convert to Settings-compatible dictionary.

        Returns:
            Dictionary that can be used to override Settings attributes
        """
        settings = {
            "tool_call_budget": self.tool_budget,
            "max_iterations": self.max_iterations,
            "parallel_tool_execution": self.enable_parallel_tools,
            "max_concurrent_tools": self.max_concurrent_tools,
            "context_proactive_compaction": self.enable_context_compaction,
            "streaming_timeout": self.streaming_timeout,
            "use_semantic_tool_selection": self.enable_semantic_search,
            "code_correction_enabled": self.enable_code_correction,
            "analytics_enabled": self.enable_analytics,
            "tool_cache_enabled": self.enable_tool_cache,
            "tool_cache_ttl": self.tool_cache_ttl,
            "recovery_max_attempts": self.max_recovery_attempts,
        }
        if self.max_context_chars:
            settings["max_context_chars"] = self.max_context_chars
        settings.update(self.extra)
        return settings

    @classmethod
    def default(cls) -> "AgentConfig":
        """Create default configuration."""
        return cls()

    @classmethod
    def minimal(cls) -> "AgentConfig":
        """Create minimal configuration for simple tasks.

        Disables semantic search, analytics, and reduces budgets
        for quick, lightweight operations.
        """
        return cls(
            tool_budget=15,
            max_iterations=10,
            enable_semantic_search=False,
            enable_analytics=False,
            enable_tool_cache=False,
        )

    @classmethod
    def high_budget(cls) -> "AgentConfig":
        """Create high-budget configuration for complex tasks.

        Increases tool budget and iterations for tasks that
        require extensive exploration or many operations.
        """
        return cls(
            tool_budget=200,
            max_iterations=100,
            max_concurrent_tools=10,
            enable_semantic_search=True,
        )

    @classmethod
    def airgapped(cls) -> "AgentConfig":
        """Create configuration for air-gapped environments.

        Disables features that require network access.
        """
        return cls(
            enable_semantic_search=False,  # May require downloading models
            enable_analytics=False,
            extra={"airgapped_mode": True},
        )
