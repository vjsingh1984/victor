"""Recovery and loop detection configuration.

This module contains settings for:
- Loop detection and recovery
- Empty response handling
- Chat iteration limits
- Continuation prompt management
"""

from pydantic import BaseModel, Field, field_validator


class RecoverySettings(BaseModel):
    """Recovery and loop detection settings.

    Controls how Victor detects and recovers from stuck behavior,
    including loop detection, empty responses, and continuation prompts.
    """

    # ==========================================================================
    # Recovery & Loop Detection Thresholds
    # ==========================================================================
    # These control when Victor forces completion after detecting stuck behavior.
    # Lower values = faster recovery but may cut off legitimate long operations.
    # Higher values = more patience but may waste tokens on stuck loops.

    # Empty response recovery: Force after N consecutive empty responses from model
    recovery_empty_response_threshold: int = 5

    # Loop detection patience: How many consecutive blocked attempts before forcing completion
    # This is separate from the per-task loop_repeat_threshold (which controls when to warn/block)
    recovery_blocked_consecutive_threshold: int = 6
    recovery_blocked_total_threshold: int = 9

    # ==========================================================================
    # Chat Iteration Limits
    # ==========================================================================
    # Maximum number of chat iterations per session to prevent infinite loops.
    # Includes both tool calls and model responses.
    chat_max_iterations: int = Field(
        default=50, ge=1, description="Maximum chat iterations per session"
    )

    # Maximum consecutive tool calls before forcing completion
    # Prevents infinite loops where the agent keeps calling tools without making progress
    max_consecutive_tool_calls: int = Field(
        default=20,
        ge=1,
        description="Max consecutive tool calls before forcing completion",
    )

    # ==========================================================================
    # Continuation Prompts
    # ==========================================================================
    # How many times to prompt model to continue before forcing completion
    # These are global defaults - can be overridden per provider/model via RL learning
    max_continuation_prompts: int = Field(
        default=3,
        ge=0,
        description="Max consecutive continuation prompts before forcing completion",
    )

    # Force response even if errors occurred during tool execution
    force_response_on_error: bool = Field(
        default=True, description="Force synthesis even if errors occurred"
    )

    @field_validator("recovery_empty_response_threshold")
    @classmethod
    def validate_empty_response_threshold(cls, v: int) -> int:
        """Validate empty response threshold is positive.

        Args:
            v: Threshold value

        Returns:
            Validated threshold

        Raises:
            ValueError: If threshold is not positive
        """
        if v < 1:
            raise ValueError("recovery_empty_response_threshold must be >= 1")
        return v

    @field_validator("recovery_blocked_consecutive_threshold")
    @classmethod
    def validate_blocked_consecutive_threshold(cls, v: int) -> int:
        """Validate blocked consecutive threshold is positive.

        Args:
            v: Threshold value

        Returns:
            Validated threshold

        Raises:
            ValueError: If threshold is not positive
        """
        if v < 1:
            raise ValueError("recovery_blocked_consecutive_threshold must be >= 1")
        return v

    @field_validator("recovery_blocked_total_threshold")
    @classmethod
    def validate_blocked_total_threshold(cls, v: int) -> int:
        """Validate blocked total threshold is positive and >= consecutive threshold.

        Args:
            v: Threshold value

        Returns:
            Validated threshold

        Raises:
            ValueError: If threshold is not positive or less than consecutive threshold
        """
        if v < 1:
            raise ValueError("recovery_blocked_total_threshold must be >= 1")
        # Note: Can't validate against consecutive_threshold here since we don't have access to it
        # This will be validated in a model_validator in Settings
        return v
