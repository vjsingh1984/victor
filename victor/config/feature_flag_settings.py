"""Feature flags for gradual rollout of architecture components."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class FeatureFlagSettings(BaseModel):
    """Feature flags for gradual rollout of architecture components.

    Feature flags allow gradual rollout of new features with minimal risk:
    - Enable features for a percentage of users (rollout_percentage)
    - Instant rollback via environment variables
    - Safe testing in production before full rollout
    - Independent toggles for related features

    Environment Variables:
        VICTOR_ENABLE_PREDICTIVE_TOOLS: Master switch for predictive components
        VICTOR_PREDICTIVE_ROLLOUT_PERCENTAGE: Rollout percentage (0-100)
        VICTOR_ENABLE_HYBRID_DECISIONS: Enable hybrid decision service
        VICTOR_ENABLE_PHASE_AWARE_CONTEXT: Enable phase-based context

    Rollout Strategy:
        Week 1: 1% (canary)
        Week 2: 10% (early adopters)
        Week 3: 50% (broad rollout)
        Week 4: 100% (full rollout)

    Rollback:
        Set VICTOR_ENABLE_PREDICTIVE_TOOLS=false for instant rollback
    """

    # Existing service flags (Phase 3-6)
    use_new_chat_service: bool = False
    use_new_tool_service: bool = False
    use_new_context_service: bool = False
    use_new_provider_service: bool = False
    use_new_recovery_service: bool = False
    use_new_session_service: bool = False
    use_composition_over_inheritance: bool = False
    use_strategy_based_tool_registration: bool = False
    use_provider_pooling: bool = False

    # Priority 1: Hybrid Decision Service (Phase 7)
    enable_hybrid_decisions: bool = Field(
        default=False,
        description="Enable hybrid decision service with deterministic fast-paths",
    )

    # Priority 2: Phase-Based Context Management
    enable_phase_aware_context: bool = Field(
        default=False,
        description="Enable phase-based context management (EXPLORATION, PLANNING, EXECUTION, REVIEW)",
    )

    # Priority 3: Predictive Tool Selection
    enable_predictive_tools: bool = Field(
        default=False,
        description="Master switch for predictive tool selection components",
    )

    predictive_rollout_percentage: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Percentage of requests that use predictive features (0-100)",
    )

    enable_tool_predictor: bool = Field(
        default=False,
        description="Enable ensemble tool prediction (keyword + semantic + co-occurrence)",
    )

    enable_cooccurrence_tracking: bool = Field(
        default=False,
        description="Enable co-occurrence pattern tracking for tool sequences",
    )

    enable_tool_preloading: bool = Field(
        default=False,
        description="Enable async background preloading of tool schemas",
    )

    predictive_confidence_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for predictive tool selection (0.0-1.0)",
    )

    # Priority 4: Learning from Execution (Future)
    enable_execution_learning: bool = Field(
        default=False,
        description="Enable learning from execution outcomes (long-term research)",
    )

    @field_validator("predictive_rollout_percentage")
    @classmethod
    def validate_rollout_percentage(cls, v: int, info) -> int:
        """Validate rollout percentage and auto-enable features if needed."""
        # If rollout percentage > 0, ensure master switch is on
        if v > 0:
            # Check if we're in a validation context with field values
            if hasattr(info, "data") and isinstance(info.data, dict):
                # If enable_predictive_tools is not explicitly set, auto-enable it
                if "enable_predictive_tools" not in info.data:
                    # We'll let the model_defaults handle this
                    pass
        return v

    def should_use_predictive_for_request(self, request_hash: int = 0) -> bool:
        """Determine if predictive features should be used for a specific request.

        Uses consistent hashing based on request_hash to ensure the same
        request always gets the same treatment during rollout.

        Args:
            request_hash: Hash of request identifier (session_id, user_id, etc.)

        Returns:
            True if predictive features should be used for this request
        """
        if not self.enable_predictive_tools:
            return False

        if self.predictive_rollout_percentage >= 100:
            return True

        if self.predictive_rollout_percentage <= 0:
            return False

        # Use consistent hashing to determine if this request is in rollout
        # This ensures the same request always gets the same treatment
        hash_value = abs(request_hash) % 100
        return hash_value < self.predictive_rollout_percentage

    def get_effective_settings(self) -> dict:
        """Get effective feature flag settings considering rollout percentage.

        Returns:
            Dictionary with actual feature states for the current environment
        """
        return {
            "hybrid_decisions_enabled": self.enable_hybrid_decisions,
            "phase_aware_context_enabled": self.enable_phase_aware_context,
            "predictive_tools_enabled": self.enable_predictive_tools,
            "tool_predictor_enabled": self.enable_tool_predictor and self.enable_predictive_tools,
            "cooccurrence_tracking_enabled": self.enable_cooccurrence_tracking
            and self.enable_predictive_tools,
            "tool_preloading_enabled": self.enable_tool_preloading and self.enable_predictive_tools,
            "rollout_percentage": self.predictive_rollout_percentage,
            "confidence_threshold": self.predictive_confidence_threshold,
        }
