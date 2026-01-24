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


# =============================================================================
# Graph Configuration (ISP Compliance)
# =============================================================================

"""Focused configuration classes for StateGraph execution (ISP compliance).

This section provides segregated configuration interfaces following the
Interface Segregation Principle (ISP). Each config class addresses one
specific aspect of graph execution, allowing clients to depend only on
the configuration they need.

Classes:
    ExecutionConfig: Controls execution limits (iterations, timeout, recursion)
    CheckpointConfig: Controls state persistence behavior
    InterruptConfig: Controls interrupt behavior for human-in-the-loop
    PerformanceConfig: Controls performance optimizations
    ObservabilityConfig: Controls observability and eventing
    GraphConfig: Facade that composes all focused configs

Migration:
    Use GraphConfig.from_legacy(**kwargs) to migrate from legacy format.
"""

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from victor.framework.graph import CheckpointerProtocol


@dataclass
class ExecutionConfig:
    """Controls execution limits for graph execution.

    Attributes:
        max_iterations: Maximum cycles allowed (default: 25)
        timeout: Overall execution timeout in seconds (None = no limit)
        recursion_limit: Maximum recursion depth (default: 100)

    Example:
        config = ExecutionConfig(max_iterations=50, timeout=300.0)
    """

    max_iterations: int = 25
    timeout: Optional[float] = None
    recursion_limit: int = 100


@dataclass
class CheckpointConfig:
    """Controls state checkpointing behavior.

    Attributes:
        checkpointer: Optional checkpointer for persistence
        checkpoint_at_start: Whether to checkpoint before execution (default: False)
        checkpoint_at_end: Whether to checkpoint after execution (default: True)

    Example:
        from victor.storage.checkpointing import MemoryCheckpointer
        config = CheckpointConfig(
            checkpointer=MemoryCheckpointer(),
            checkpoint_at_end=True
        )
    """

    checkpointer: Optional["CheckpointerProtocol"] = None
    checkpoint_at_start: bool = False
    checkpoint_at_end: bool = True


@dataclass
class InterruptConfig:
    """Controls interrupt behavior for human-in-the-loop workflows.

    Attributes:
        interrupt_before: List of node IDs to interrupt before execution
        interrupt_after: List of node IDs to interrupt after execution

    Example:
        config = InterruptConfig(
            interrupt_before=["critical_operation"],
            interrupt_after=["user_approval"]
        )
    """

    interrupt_before: List[str] = field(default_factory=list)
    interrupt_after: List[str] = field(default_factory=list)


@dataclass
class PerformanceConfig:
    """Controls performance optimizations.

    Attributes:
        use_copy_on_write: Enable copy-on-write state optimization
            (None = use settings default, True = enable, False = disable)
        enable_state_caching: Enable state caching (default: True)

    Example:
        config = PerformanceConfig(use_copy_on_write=True)
    """

    use_copy_on_write: Optional[bool] = None  # None = use settings default
    enable_state_caching: bool = True


@dataclass
class ObservabilityConfig:
    """Controls observability and eventing features.

    Attributes:
        emit_events: Enable EventBus integration for observability (default: True)
        log_node_execution: Log each node execution (default: False)
        graph_id: Optional identifier for this graph execution (for event correlation)

    Example:
        config = ObservabilityConfig(
            emit_events=True,
            graph_id="workflow-123"
        )
    """

    emit_events: bool = True  # Enable EventBus observability integration
    log_node_execution: bool = False
    graph_id: Optional[str] = None  # Optional identifier for event correlation


@dataclass
class GraphConfig:
    """Facade config that composes focused configs.

    This class provides backward compatibility while allowing clients
    to use focused configs directly for ISP compliance.

    Attributes:
        execution: Execution limits configuration
        checkpoint: Checkpointing configuration
        interrupt: Interrupt behavior configuration
        performance: Performance optimization configuration
        observability: Observability and eventing configuration

    Example:
        # Use focused configs (ISP compliant)
        config = GraphConfig(
            execution=ExecutionConfig(max_iterations=50),
            observability=ObservabilityConfig(emit_events=True)
        )

        # Migrate from legacy format
        config = GraphConfig.from_legacy(max_iterations=50, timeout=300.0)
    """

    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    interrupt: InterruptConfig = field(default_factory=InterruptConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)

    @classmethod
    def from_legacy(cls, **kwargs: Any) -> "GraphConfig":
        """Migrate from legacy config format.

        This class method allows gradual migration from legacy code
        that uses the old monolithic GraphConfig.

        Args:
            **kwargs: Legacy config parameters:
                - max_iterations: int
                - timeout: Optional[float]
                - checkpointer: Optional[CheckpointerProtocol]
                - recursion_limit: int
                - interrupt_before: List[str]
                - interrupt_after: List[str]
                - use_copy_on_write: Optional[bool]
                - emit_events: bool
                - graph_id: Optional[str]

        Returns:
            GraphConfig with focused configs populated from legacy kwargs

        Example:
            # Legacy code
            old_config = GraphConfig(max_iterations=50, timeout=300.0)

            # New code (with migration)
            new_config = GraphConfig.from_legacy(max_iterations=50, timeout=300.0)
        """
        return cls(
            execution=ExecutionConfig(
                max_iterations=kwargs.get("max_iterations", 25),
                timeout=kwargs.get("timeout"),
                recursion_limit=kwargs.get("recursion_limit", 100),
            ),
            checkpoint=CheckpointConfig(
                checkpointer=kwargs.get("checkpointer"),
                # Legacy didn't have these, use defaults
                checkpoint_at_start=False,
                checkpoint_at_end=True,
            ),
            interrupt=InterruptConfig(
                interrupt_before=kwargs.get("interrupt_before", []),
                interrupt_after=kwargs.get("interrupt_after", []),
            ),
            performance=PerformanceConfig(
                use_copy_on_write=kwargs.get("use_copy_on_write"),
                enable_state_caching=True,  # Default for legacy migration
            ),
            observability=ObservabilityConfig(
                emit_events=kwargs.get("emit_events", True),
                log_node_execution=False,  # Default for legacy migration
                graph_id=kwargs.get("graph_id"),
            ),
        )

    def to_legacy_dict(self) -> Dict[str, Any]:
        """Convert to legacy dict format for backward compatibility.

        Returns:
            Dict with legacy config keys
        """
        return {
            "max_iterations": self.execution.max_iterations,
            "timeout": self.execution.timeout,
            "checkpointer": self.checkpoint.checkpointer,
            "recursion_limit": self.execution.recursion_limit,
            "interrupt_before": self.interrupt.interrupt_before,
            "interrupt_after": self.interrupt.interrupt_after,
            "use_copy_on_write": self.performance.use_copy_on_write,
            "emit_events": self.observability.emit_events,
            "graph_id": self.observability.graph_id,
        }


# =============================================================================
# Vertical Configuration (Promoted from individual verticals)
# =============================================================================

"""Framework-level configuration for vertical capabilities.

This section promotes duplicated configuration patterns from individual
verticals (Coding, DevOps, RAG, etc.) to framework-level abstractions
that all verticals can benefit from (DRY principle).

Classes promoted from verticals:
    - SafetyConfig: Generic safety configuration (promoted from all verticals)
    - StyleConfig: Code/container style configuration (promoted from Coding/DevOps)
    - ToolConfig: Tool behavior configuration (promoted from all verticals)
"""

from enum import Enum


class SafetyLevel(Enum):
    """Safety enforcement levels for operations.

    Defines how strictly safety rules are enforced:
    - OFF: No safety checks
    - LOW: Warn only, don't block operations
    - MEDIUM: Block dangerous operations, warn risky ones
    - HIGH: Block anything not explicitly allowed
    """

    OFF = "off"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class SafetyConfig:
    """Generic safety configuration (promoted from verticals).

    This configuration class consolidates safety-related settings
    that were duplicated across Coding, DevOps, RAG, and other verticals.

    Attributes:
        level: Overall safety enforcement level
        require_confirmation: Require user confirmation for operations
        blocked_operations: List of operation patterns to block
        audit_log: Enable audit logging of safety checks
        dry_run: Show what would happen without executing

    Example:
        # High safety for production
        config = SafetyConfig(
            level=SafetyLevel.HIGH,
            require_confirmation=True,
            blocked_operations=["rm -rf /", "git push --force"],
        )

        # Low safety for development
        config = SafetyConfig(level=SafetyLevel.LOW, dry_run=False)
    """

    level: SafetyLevel = SafetyLevel.MEDIUM
    require_confirmation: bool = False
    blocked_operations: List[str] = field(default_factory=list)
    audit_log: bool = True
    dry_run: bool = False

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "SafetyConfig":
        """Create from dict (supports loading from profiles).

        Args:
            config: Dictionary with safety configuration

        Returns:
            SafetyConfig instance
        """
        return cls(
            level=SafetyLevel(config.get("level", "medium")),
            require_confirmation=config.get("require_confirmation", False),
            blocked_operations=config.get("blocked_operations", []),
            audit_log=config.get("audit_log", True),
            dry_run=config.get("dry_run", False),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "level": self.level.value,
            "require_confirmation": self.require_confirmation,
            "blocked_operations": self.blocked_operations,
            "audit_log": self.audit_log,
            "dry_run": self.dry_run,
        }


@dataclass
class StyleConfig:
    """Generic style/format configuration (promoted from verticals).

    This configuration class consolidates style-related settings
    that were duplicated across Coding (code_style) and DevOps (container_style).

    Attributes:
        formatter: Preferred formatter (black, prettier, rubocop, etc.)
        formatter_options: Options passed to formatter
        linter: Preferred linter (ruff, eslint, flake8, etc.)
        linter_options: Options passed to linter
        style_guide: Style guide to follow (PEP8, Google, AirBnB, etc.)

    Example:
        # Python code style
        config = StyleConfig(
            formatter="black",
            formatter_options={"line_length": 100},
            linter="ruff",
            style_guide="PEP8",
        )

        # Docker container style
        config = StyleConfig(
            formatter="hadolint",
            linter="dockerfile_lint",
        )
    """

    formatter: Optional[str] = None
    formatter_options: Dict[str, Any] = field(default_factory=dict)
    linter: Optional[str] = None
    linter_options: Dict[str, Any] = field(default_factory=dict)
    style_guide: Optional[str] = None

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "StyleConfig":
        """Create from dict (supports loading from profiles).

        Args:
            config: Dictionary with style configuration

        Returns:
            StyleConfig instance
        """
        return cls(
            formatter=config.get("formatter"),
            formatter_options=config.get("formatter_options", {}),
            linter=config.get("linter"),
            linter_options=config.get("linter_options", {}),
            style_guide=config.get("style_guide"),
        )


@dataclass
class ToolConfig:
    """Generic tool configuration (promoted from verticals).

    This configuration class consolidates tool-related settings
    that were duplicated across all verticals.

    Attributes:
        enabled_tools: Whitelist of tools to enable (empty = all)
        disabled_tools: Blacklist of tools to disable
        tool_settings: Per-tool configuration (tool_name -> settings dict)
        max_tool_budget: Maximum number of tool calls per session
        require_confirmation: Require confirmation before tool use

    Example:
        # Enable only specific tools
        config = ToolConfig(
            enabled_tools=["read", "write", "grep"],
            max_tool_budget=50,
        )

        # Disable dangerous tools
        config = ToolConfig(
            disabled_tools=["shell", "run_command"],
            require_confirmation=True,
        )

        # Per-tool settings
        config = ToolConfig(
            tool_settings={
                "docker": {"runtime": "python3.12"},
                "git": {"default_branch": "main"},
            },
        )
    """

    enabled_tools: List[str] = field(default_factory=list)
    disabled_tools: List[str] = field(default_factory=list)
    tool_settings: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    max_tool_budget: int = 100
    require_confirmation: bool = False

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ToolConfig":
        """Create from dict (supports loading from profiles).

        Args:
            config: Dictionary with tool configuration

        Returns:
            ToolConfig instance
        """
        return cls(
            enabled_tools=config.get("enabled_tools", []),
            disabled_tools=config.get("disabled_tools", []),
            tool_settings=config.get("tool_settings", {}),
            max_tool_budget=config.get("max_tool_budget", 100),
            require_confirmation=config.get("require_confirmation", False),
        )

    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a tool is enabled.

        Args:
            tool_name: Name of the tool to check

        Returns:
            True if tool is enabled, False otherwise
        """
        # If whitelist specified, tool must be in it
        if self.enabled_tools:
            return tool_name in self.enabled_tools

        # If blacklist specified, tool must not be in it
        if self.disabled_tools:
            return tool_name not in self.disabled_tools

        # No restrictions, tool is enabled
        return True

    def get_tool_setting(self, tool_name: str, key: str, default: Any = None) -> Any:
        """Get a setting for a specific tool.

        Args:
            tool_name: Name of the tool
            key: Setting key
            default: Default value if setting not found

        Returns:
            Tool setting value
        """
        return self.tool_settings.get(tool_name, {}).get(key, default)


# =============================================================================
# Safety Enforcement (Framework-Level)
# =============================================================================

"""Framework-level safety enforcement system.

This section provides a unified safety enforcement mechanism that all
verticals can use instead of implementing their own safety checks.

Classes:
    SafetyRule: Individual safety rule with check function
    SafetyEnforcer: Framework-level safety enforcer that manages rules
"""

from typing import Callable, Tuple


@dataclass
class SafetyRule:
    """Individual safety rule for operations.

    Attributes:
        name: Unique name for this rule
        description: Human-readable description
        check_fn: Function that checks if operation violates this rule.
                  Returns True if operation should be blocked.
        level: Safety enforcement level (MEDIUM = warn, HIGH = block)
        allow_override: Whether user can override this rule

    Example:
        def check_git_force_push(operation: str) -> bool:
            return "git push --force" in operation and "main" in operation

        rule = SafetyRule(
            name="protect_main_branch",
            description="Block force push to main/master",
            check_fn=check_git_force_push,
            level=SafetyLevel.HIGH,
            allow_override=False,
        )
    """

    name: str
    description: str
    check_fn: Callable[[str], bool]
    level: SafetyLevel = SafetyLevel.MEDIUM
    allow_override: bool = False


class SafetyEnforcer:
    """Framework-level safety enforcement for operations.

    This class provides a unified safety enforcement mechanism that all
    verticals can use instead of implementing their own safety checks.

    Attributes:
        config: SafetyConfig with enforcement levels and blocked operations
        rules: List of registered safety rules

    Example:
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))

        # Register rules
        enforcer.add_rule(SafetyRule(
            name="protect_main",
            description="Block force push to main",
            check_fn=lambda op: "git push --force" in op and "main" in op,
            level=SafetyLevel.HIGH,
        ))

        # Check operations
        allowed, reason = enforcer.check_operation("git push --force origin main")
        if not allowed:
            print(f"Blocked: {reason}")
    """

    def __init__(self, config: SafetyConfig):
        """Initialize safety enforcer.

        Args:
            config: Safety configuration
        """
        self.config = config
        self.rules: list[SafetyRule] = []

    def add_rule(self, rule: SafetyRule) -> None:
        """Register a safety rule.

        Args:
            rule: SafetyRule to add

        Raises:
            ValueError: If rule name already exists
        """
        if any(r.name == rule.name for r in self.rules):
            raise ValueError(f"Rule '{rule.name}' already registered")

        self.rules.append(rule)

    def remove_rule(self, name: str) -> bool:
        """Remove a safety rule by name.

        Args:
            name: Rule name to remove

        Returns:
            True if rule was removed, False if not found
        """
        for i, rule in enumerate(self.rules):
            if rule.name == name:
                self.rules.pop(i)
                return True
        return False

    def check_operation(
        self, operation: str, context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str]]:
        """Check if operation is allowed by safety rules.

        Args:
            operation: Operation description or command to check
            context: Optional context for rule evaluation (e.g., branch name, environment)

        Returns:
            Tuple of (allowed: bool, reason: Optional[str])
            - allowed: True if operation is allowed, False if blocked
            - reason: Human-readable reason if blocked, None if allowed

        Example:
            allowed, reason = enforcer.check_operation(
                "git push --force origin main",
                context={"branch": "main", "user": "developer"}
            )
            if not allowed:
                print(f"Blocked: {reason}")
        """
        # Dry run mode - always allow but log what would happen (check FIRST)
        if self.config.dry_run:
            return True, f"[DRY RUN] Would execute: {operation}"

        # Check blocked operations list (config-level blocking)
        for blocked in self.config.blocked_operations:
            if blocked.lower() in operation.lower():
                return False, f"Blocked by configuration: '{blocked}'"

        # Check registered rules
        for rule in self.rules:
            try:
                # Check if rule applies
                if rule.check_fn(operation):
                    # Rule triggered - check enforcement level
                    if rule.level == SafetyLevel.HIGH:
                        # HIGH level - always block (unless overridden by config)
                        if rule.allow_override and self.config.level == SafetyLevel.LOW:
                            continue  # Allow due to override
                        return False, f"Blocked by safety rule: {rule.name} - {rule.description}"

                    elif rule.level == SafetyLevel.MEDIUM:
                        # MEDIUM level - block unless config is LOW
                        if self.config.level == SafetyLevel.LOW:
                            continue  # Warn only, don't block
                        return False, f"Blocked by safety rule: {rule.name} - {rule.description}"

                    elif rule.level == SafetyLevel.LOW:
                        # LOW level - warn but don't block (unless config is HIGH)
                        if self.config.level == SafetyLevel.HIGH:
                            return (
                                False,
                                f"Blocked by safety rule: {rule.name} - {rule.description}",
                            )
                        # Warn but allow
                        if self.config.audit_log:
                            # Log warning (would integrate with logging system)
                            pass

                    elif rule.level == SafetyLevel.OFF:
                        # Rule disabled
                        pass

            except Exception:
                # Rule check failed - log and continue
                if self.config.audit_log:
                    # Log error (would integrate with logging system)
                    pass

        return True, None

    def get_rules_by_level(self, level: SafetyLevel) -> list[SafetyRule]:
        """Get all rules with a specific enforcement level.

        Args:
            level: SafetyLevel to filter by

        Returns:
            List of rules with the specified level
        """
        return [rule for rule in self.rules if rule.level == level]

    def clear_rules(self) -> None:
        """Remove all registered rules."""
        self.rules.clear()


# Update __all__ to include new classes
__all__ = [
    # Agent configuration
    "AgentConfig",
    # Graph configuration
    "ExecutionConfig",
    "CheckpointConfig",
    "InterruptConfig",
    "PerformanceConfig",
    "ObservabilityConfig",
    "GraphConfig",
    # Vertical configuration
    "SafetyConfig",
    "StyleConfig",
    "ToolConfig",
    "SafetyLevel",
    # Safety enforcement
    "SafetyRule",
    "SafetyEnforcer",
]
