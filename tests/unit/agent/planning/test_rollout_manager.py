"""Tests for feature flags and gradual rollout.

Tests cover:
- Feature flag initialization and defaults
- Environment variable configuration
- Rollout percentage logic
- Consistent hashing for request routing
- Rollout stage progression
- Automatic rollback on errors
- Metrics collection and reporting
- Rollback mechanisms
"""

from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import the module first to avoid circular import issues
from victor.agent.planning import rollout_manager  # noqa: E402

# Then import the classes
RolloutConfig = rollout_manager.RolloutConfig
RolloutManager = rollout_manager.RolloutManager
RolloutMetrics = rollout_manager.RolloutMetrics
RolloutStage = rollout_manager.RolloutStage


class TestFeatureFlagSettings:
    """Test feature flag settings."""

    def test_default_feature_flags(self):
        """Test default feature flag values."""
        from victor.config.feature_flag_settings import FeatureFlagSettings

        flags = FeatureFlagSettings()

        # All predictive features should be disabled by default
        assert flags.enable_predictive_tools is False
        assert flags.enable_tool_predictor is False
        assert flags.enable_cooccurrence_tracking is False
        assert flags.enable_tool_preloading is False
        assert flags.enable_hybrid_decisions is False
        assert flags.enable_phase_aware_context is False

    def test_rollout_percentage_default(self):
        """Test default rollout percentage."""
        from victor.config.feature_flag_settings import FeatureFlagSettings

        flags = FeatureFlagSettings()

        assert flags.predictive_rollout_percentage == 0

    def test_confidence_threshold_default(self):
        """Test default confidence threshold."""
        from victor.config.feature_flag_settings import FeatureFlagSettings

        flags = FeatureFlagSettings()

        assert flags.predictive_confidence_threshold == 0.6

    def test_confidence_threshold_validation(self):
        """Test confidence threshold validation."""
        from victor.config.feature_flag_settings import FeatureFlagSettings
        from pydantic import ValidationError

        # Test invalid high value
        with pytest.raises(ValidationError):
            FeatureFlagSettings(predictive_confidence_threshold=1.5)

        # Test invalid low value
        with pytest.raises(ValidationError):
            FeatureFlagSettings(predictive_confidence_threshold=-0.1)

        # Test valid values
        flags = FeatureFlagSettings(predictive_confidence_threshold=0.8)
        assert flags.predictive_confidence_threshold == 0.8

    def test_rollout_percentage_validation(self):
        """Test rollout percentage validation."""
        from victor.config.feature_flag_settings import FeatureFlagSettings
        from pydantic import ValidationError

        # Test invalid high value
        with pytest.raises(ValidationError):
            FeatureFlagSettings(predictive_rollout_percentage=150)

        # Test invalid low value
        with pytest.raises(ValidationError):
            FeatureFlagSettings(predictive_rollout_percentage=-10)

        # Test valid values
        flags = FeatureFlagSettings(predictive_rollout_percentage=50)
        assert flags.predictive_rollout_percentage == 50

    def test_should_use_predictive_disabled(self):
        """Test that disabled features never get used."""
        from victor.config.feature_flag_settings import FeatureFlagSettings

        flags = FeatureFlagSettings(
            enable_predictive_tools=False,
            predictive_rollout_percentage=100,
        )

        assert flags.should_use_predictive_for_request() is False

    def test_should_use_predictive_full_rollout(self):
        """Test that 100% rollout always returns True."""
        from victor.config.feature_flag_settings import FeatureFlagSettings

        flags = FeatureFlagSettings(
            enable_predictive_tools=True,
            predictive_rollout_percentage=100,
        )

        assert flags.should_use_predictive_for_request() is True

    def test_should_use_predictive_zero_rollout(self):
        """Test that 0% rollout always returns False."""
        from victor.config.feature_flag_settings import FeatureFlagSettings

        flags = FeatureFlagSettings(
            enable_predictive_tools=True,
            predictive_rollout_percentage=0,
        )

        assert flags.should_use_predictive_for_request() is False

    def test_consistent_hashing_for_rollout(self):
        """Test that consistent hashing provides stable routing."""
        from victor.config.feature_flag_settings import FeatureFlagSettings

        flags = FeatureFlagSettings(
            enable_predictive_tools=True,
            predictive_rollout_percentage=50,
        )

        # Same request should get same result
        result1 = flags.should_use_predictive_for_request(request_hash=12345)
        result2 = flags.should_use_predictive_for_request(request_hash=12345)
        assert result1 == result2

        # Different requests might get different results
        result3 = flags.should_use_predictive_for_request(request_hash=54321)
        # We don't assert result3 != result1 because it could be the same by chance

    def test_get_effective_settings(self):
        """Test getting effective feature flag settings."""
        from victor.config.feature_flag_settings import FeatureFlagSettings

        flags = FeatureFlagSettings(
            enable_predictive_tools=True,
            enable_tool_predictor=True,
            enable_cooccurrence_tracking=True,
            enable_tool_preloading=True,
            predictive_rollout_percentage=50,
            predictive_confidence_threshold=0.7,
        )

        effective = flags.get_effective_settings()

        assert effective["predictive_tools_enabled"] is True
        assert effective["tool_predictor_enabled"] is True
        assert effective["cooccurrence_tracking_enabled"] is True
        assert effective["tool_preloading_enabled"] is True
        assert effective["rollout_percentage"] == 50
        assert effective["confidence_threshold"] == 0.7

    def test_effective_settings_with_master_disabled(self):
        """Test that disabled master switch disables all sub-features."""
        from victor.config.feature_flag_settings import FeatureFlagSettings

        flags = FeatureFlagSettings(
            enable_predictive_tools=False,
            enable_tool_predictor=True,
            enable_tool_preloading=True,
        )

        effective = flags.get_effective_settings()

        assert effective["predictive_tools_enabled"] is False
        assert effective["tool_predictor_enabled"] is False
        assert effective["tool_preloading_enabled"] is False


class TestRolloutMetrics:
    """Test rollout metrics."""

    def test_metrics_creation(self):
        """Test creating metrics."""
        metrics = RolloutMetrics()

        assert metrics.total_requests == 0
        assert metrics.predictive_requests == 0
        assert metrics.errors == 0
        assert metrics.latency_ms == 0.0
        assert isinstance(metrics.last_updated, datetime)

    def test_error_rate_calculation(self):
        """Test error rate calculation."""
        metrics = RolloutMetrics(
            total_requests=100,
            predictive_requests=80,
            errors=4,
        )

        assert metrics.error_rate() == 0.05  # 4/80 = 0.05

    def test_error_rate_no_requests(self):
        """Test error rate with no requests."""
        metrics = RolloutMetrics()

        assert metrics.error_rate() == 0.0

    def test_rollout_percentage_calculation(self):
        """Test rollout percentage calculation."""
        metrics = RolloutMetrics(
            total_requests=100,
            predictive_requests=25,
        )

        assert metrics.rollout_percentage() == 25.0


class TestRolloutManager:
    """Test rollout manager."""

    def test_default_initialization(self):
        """Test default initialization."""
        manager = RolloutManager()

        assert manager.get_current_stage() == RolloutStage.CANARY
        assert manager.get_rollout_percentage() == 1

    def test_custom_config(self):
        """Test custom configuration."""
        config = RolloutConfig(
            error_threshold=0.1,
            min_requests_before_rollout=50,
            cooldown_seconds=1800,
        )
        manager = RolloutManager(config=config)

        assert manager.config.error_threshold == 0.1
        assert manager.config.min_requests_before_rollout == 50

    def test_should_use_predictive_canary(self):
        """Test predictive usage in canary stage (1%)."""
        manager = RolloutManager()

        # With 1% rollout, very few requests should use predictive
        count = 0
        for i in range(1000):
            if manager.should_use_predictive(session_id=f"session_{i}"):
                count += 1

        # Should be roughly 1% (allow 0-2% for variance)
        assert 0 <= count <= 20

    def test_should_use_predictive_full_rollout(self):
        """Test predictive usage at 100% rollout."""
        manager = RolloutManager()
        manager._current_stage = RolloutStage.GENERAL

        # All requests should use predictive
        for i in range(100):
            assert manager.should_use_predictive(session_id=f"session_{i}") is True

    def test_should_use_predictive_zero_rollout(self):
        """Test predictive usage at 0% rollout."""
        config = RolloutConfig(stages={RolloutStage.CANARY: 0})
        manager = RolloutManager(config=config)

        # No requests should use predictive
        for i in range(100):
            assert manager.should_use_predictive(session_id=f"session_{i}") is False

    def test_consistent_routing(self):
        """Test that same session always gets same routing."""
        manager = RolloutManager()

        # Same session should get same result
        result1 = manager.should_use_predictive(session_id="test_session")
        result2 = manager.should_use_predictive(session_id="test_session")
        assert result1 == result2

    def test_record_request_success(self):
        """Test recording a successful request."""
        manager = RolloutManager()

        manager.record_request(
            session_id="test_session",
            used_predictive=True,
            success=True,
            latency_ms=100,
        )

        summary = manager.get_metrics_summary()
        assert summary["total_requests"] == 1
        assert summary["predictive_requests"] == 1
        assert summary["errors"] == 0

    def test_record_request_error(self):
        """Test recording a failed request."""
        manager = RolloutManager()

        manager.record_request(
            session_id="test_session",
            used_predictive=True,
            success=False,
            error_message="Test error",
        )

        summary = manager.get_metrics_summary()
        assert summary["total_requests"] == 1
        assert summary["errors"] == 1

    def test_record_request_non_predictive(self):
        """Test recording a non-predictive request."""
        manager = RolloutManager()

        manager.record_request(
            session_id="test_session",
            used_predictive=False,
            success=True,
        )

        summary = manager.get_metrics_summary()
        assert summary["total_requests"] == 1
        assert summary["predictive_requests"] == 0

    def test_latency_tracking(self):
        """Test latency tracking."""
        manager = RolloutManager()

        # Record multiple requests with different latencies
        for latency in [100, 150, 200, 250, 300]:
            manager.record_request(
                session_id="test_session",
                used_predictive=True,
                success=True,
                latency_ms=latency,
            )

        summary = manager.get_metrics_summary()
        # Should have average latency (exponential moving average)
        assert summary["avg_latency_ms"] > 0

    def test_can_advance_to_next_stage(self):
        """Test checking if can advance to next stage."""
        manager = RolloutManager()

        # Initially should not be able to advance (not enough requests)
        assert manager.can_advance_to_next_stage() is False

        # Record enough successful requests
        for i in range(150):
            manager.record_request(
                session_id=f"session_{i}",
                used_predictive=True,
                success=True,
            )

        # Still need to wait for cooldown
        assert manager.can_advance_to_next_stage() is False

    def test_advance_to_next_stage(self):
        """Test advancing to next stage."""
        manager = RolloutManager()
        manager.config.cooldown_seconds = 0  # Disable cooldown
        manager.config.min_requests_before_rollout = 10

        # Record enough requests
        for i in range(10):
            manager.record_request(
                session_id=f"session_{i}",
                used_predictive=True,
                success=True,
            )

        # Should be able to advance
        new_stage = manager.advance_to_next_stage()
        assert new_stage == RolloutStage.EARLY_ADOPTERS
        assert manager.get_current_stage() == RolloutStage.EARLY_ADOPTERS

    def test_cannot_advance_at_final_stage(self):
        """Test that cannot advance from final stage."""
        manager = RolloutManager()
        manager._current_stage = RolloutStage.GENERAL

        new_stage = manager.advance_to_next_stage()
        assert new_stage is None

    def test_rollback(self):
        """Test rollback to canary stage."""
        manager = RolloutManager()
        manager._current_stage = RolloutStage.BETA

        new_stage = manager.rollback(reason="testing")
        assert new_stage == RolloutStage.CANARY
        assert manager.get_current_stage() == RolloutStage.CANARY

    def test_get_metrics_summary(self):
        """Test getting metrics summary."""
        manager = RolloutManager()

        # Record some requests
        for i in range(10):
            manager.record_request(
                session_id=f"session_{i}",
                used_predictive=True,
                success=True,
                latency_ms=150,
            )

        summary = manager.get_metrics_summary()

        assert summary["total_requests"] == 10
        assert summary["predictive_requests"] == 10
        assert summary["errors"] == 0
        assert summary["current_stage"] == "canary"
        assert summary["rollout_percentage"] == 1
        assert summary["error_rate"] == 0.0
        assert summary["avg_latency_ms"] > 0

    def test_error_rate_threshold(self):
        """Test error rate threshold for rollback."""
        config = RolloutConfig(error_threshold=0.1)
        manager = RolloutManager(config=config)

        # Record some successful requests
        for i in range(10):
            manager.record_request(
                session_id=f"session_{i}",
                used_predictive=True,
                success=True,
            )

        # Record some failures (but under threshold)
        for i in range(5):
            manager.record_request(
                session_id=f"session_{i}",
                used_predictive=True,
                success=False,
            )

        # Error rate should be 5/15 = 33%, which exceeds 10% threshold
        summary = manager.get_metrics_summary()
        assert summary["error_rate"] > config.error_threshold


class TestRolloutIntegration:
    """Integration tests for rollout with feature flags."""

    def test_feature_flags_control_rollout(self):
        """Test that feature flags control rollout behavior."""
        from victor.config.feature_flag_settings import FeatureFlagSettings

        flags = FeatureFlagSettings(
            enable_predictive_tools=True,
            predictive_rollout_percentage=50,
        )

        # Should use predictive for roughly 50% of requests
        count = 0
        for i in range(1000):
            if flags.should_use_predictive_for_request(request_hash=i):
                count += 1

        # Should be around 50% (allow 40-60% for variance)
        assert 400 <= count <= 600

    def test_instant_rollback_via_env_var(self, monkeypatch):
        """Test instant rollback via environment variable."""
        # Set environment variable to disable
        monkeypatch.setenv("VICTOR_ENABLE_PREDICTIVE_TOOLS", "false")

        from victor.config.feature_flag_settings import FeatureFlagSettings

        flags = FeatureFlagSettings()

        # Should be disabled regardless of other settings
        assert flags.enable_predictive_tools is False

    def test_rollout_stage_mapping(self):
        """Test rollout stage percentage mapping."""
        config = RolloutConfig()
        manager = RolloutManager(config=config)

        # Check stage percentages
        assert config.stages[RolloutStage.CANARY] == 1
        assert config.stages[RolloutStage.EARLY_ADOPTERS] == 10
        assert config.stages[RolloutStage.BETA] == 50
        assert config.stages[RolloutStage.GENERAL] == 100

    def test_stage_progression(self):
        """Test stage progression over time."""
        manager = RolloutManager()
        manager.config.cooldown_seconds = 0
        manager.config.min_requests_before_rollout = 10

        # Start at canary
        assert manager.get_current_stage() == RolloutStage.CANARY

        # Record requests and advance
        for stage in [RolloutStage.CANARY, RolloutStage.EARLY_ADOPTERS, RolloutStage.BETA]:
            manager._current_stage = stage

            # Record enough requests
            for i in range(10):
                manager.record_request(
                    session_id=f"session_{stage.value}_{i}",
                    used_predictive=True,
                    success=True,
                )

            # Advance to next stage
            manager.advance_to_next_stage()

        # Should be at GENERAL
        assert manager.get_current_stage() == RolloutStage.GENERAL
