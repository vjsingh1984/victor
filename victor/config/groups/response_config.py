"""Response handling and completion configuration.

This module contains settings for:
- Response completion and retries
- Token management and reserves
- Response processing behavior
"""

from pydantic import BaseModel, Field, field_validator


class ResponseSettings(BaseModel):
    """Response handling and completion settings.

    Controls how responses are processed, completed, and managed.
    Includes retry logic for incomplete responses and token management.
    """

    # ==========================================================================
    # Response Completion (from VictorSettings merge)
    # ==========================================================================
    # Number of retries when model response is incomplete or truncated.
    # Higher values = more resilience to incomplete responses, but may
    # indicate issues with the provider or token limits.
    response_completion_retries: int = Field(
        default=3, ge=1, description="Max retries for incomplete responses"
    )

    # Token reserve allocated for model's own response generation.
    # Ensures model has enough budget to complete its response even when
    # context is nearly full. Typical values: 2048-8192 tokens.
    response_token_reserve: int = 4096  # Tokens reserved for model response

    @field_validator("response_token_reserve")
    @classmethod
    def validate_token_reserve(cls, v: int) -> int:
        """Validate response token reserve is positive and reasonable.

        Args:
            v: Token reserve count

        Returns:
            Validated token reserve

        Raises:
            ValueError: If token reserve is not positive or too large
        """
        if v < 512:
            raise ValueError("response_token_reserve must be >= 512 (too small for practical use)")
        if v > 32768:
            raise ValueError("response_token_reserve must be <= 32768 (unreasonably large)")
        return v
