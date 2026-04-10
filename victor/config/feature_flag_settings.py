"""Feature flags for gradual rollout of architecture components."""

from __future__ import annotations

from pydantic import BaseModel


class FeatureFlagSettings(BaseModel):
    """Feature flags for gradual rollout of architecture components."""

    use_new_chat_service: bool = False
    use_new_tool_service: bool = False
    use_new_context_service: bool = False
    use_new_provider_service: bool = False
    use_new_recovery_service: bool = False
    use_new_session_service: bool = False
    use_composition_over_inheritance: bool = False
    use_strategy_based_tool_registration: bool = False
    use_provider_pooling: bool = False
