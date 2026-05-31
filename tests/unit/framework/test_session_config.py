# Copyright 2026 Vijaykumar Singh <singhvijay@users.noreply.github.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for SessionConfig.__post_init__ validators — Wave J."""

import pytest

from victor.framework.session_config import SessionConfig


class TestSessionConfigValidation:
    """SessionConfig: tool_budget, max_iterations, and threshold range validators."""

    def test_default_config_passes_validation(self):
        """Default SessionConfig should pass all validators."""
        config = SessionConfig()
        assert config is not None
        assert config.tool_budget is None
        assert config.max_iterations is None

    def test_tool_budget_below_one_raises(self):
        """tool_budget must be >= 1 if set."""
        with pytest.raises(ValueError, match="tool_budget must be >= 1"):
            SessionConfig(tool_budget=0)

        with pytest.raises(ValueError, match="tool_budget must be >= 1"):
            SessionConfig(tool_budget=-5)

    def test_tool_budget_one_passes(self):
        """tool_budget=1 should be allowed (minimum valid value)."""
        config = SessionConfig(tool_budget=1)
        assert config.tool_budget == 1

    def test_max_iterations_below_one_raises(self):
        """max_iterations must be >= 1 if set."""
        with pytest.raises(ValueError, match="max_iterations must be >= 1"):
            SessionConfig(max_iterations=0)

        with pytest.raises(ValueError, match="max_iterations must be >= 1"):
            SessionConfig(max_iterations=-10)

    def test_max_iterations_one_passes(self):
        """max_iterations=1 should be allowed (minimum valid value)."""
        config = SessionConfig(max_iterations=1)
        assert config.max_iterations == 1

    def test_compaction_threshold_out_of_range_raises(self):
        """compaction.threshold must be in [0.0, 1.0] if set."""
        from victor.framework.session_config import CompactionConfig

        with pytest.raises(ValueError, match="compaction.threshold must be in \\[0.0, 1.0\\]"):
            SessionConfig(compaction=CompactionConfig(threshold=-0.1))

        with pytest.raises(ValueError, match="compaction.threshold must be in \\[0.0, 1.0\\]"):
            SessionConfig(compaction=CompactionConfig(threshold=1.5))

    def test_compaction_threshold_bounds_pass(self):
        """compaction.threshold at 0.0 and 1.0 should be allowed."""
        from victor.framework.session_config import CompactionConfig

        config_low = SessionConfig(compaction=CompactionConfig(threshold=0.0))
        assert config_low.compaction.threshold == 0.0

        config_high = SessionConfig(compaction=CompactionConfig(threshold=1.0))
        assert config_high.compaction.threshold == 1.0

    def test_bayesian_thresholds_out_of_range_raise(self):
        """bayesian.simple_threshold and complex_threshold must be in [0.0, 1.0]."""
        from victor.framework.bayesian_config import BayesianConfig

        with pytest.raises(ValueError, match="bayesian.simple_threshold must be in \\[0.0, 1.0\\]"):
            SessionConfig(bayesian=BayesianConfig(simple_threshold=-0.1))

        with pytest.raises(
            ValueError, match="bayesian.complex_threshold must be in \\[0.0, 1.0\\]"
        ):
            SessionConfig(bayesian=BayesianConfig(complex_threshold=1.5))

    def test_bayesian_thresholds_bounds_pass(self):
        """bayesian thresholds at 0.0 and 1.0 should be allowed."""
        from victor.framework.bayesian_config import BayesianConfig

        config_low = SessionConfig(bayesian=BayesianConfig(simple_threshold=0.0))
        assert config_low.bayesian.simple_threshold == 0.0

        config_high = SessionConfig(bayesian=BayesianConfig(complex_threshold=1.0))
        assert config_high.bayesian.complex_threshold == 1.0

    def test_from_cli_flags_with_invalid_tool_budget_raises(self):
        """from_cli_flags should reject tool_budget < 1."""
        with pytest.raises(ValueError, match="tool_budget must be >= 1"):
            SessionConfig.from_cli_flags(tool_budget=0)

    def test_from_cli_flags_with_invalid_max_iterations_raises(self):
        """from_cli_flags should reject max_iterations < 1."""
        with pytest.raises(ValueError, match="max_iterations must be >= 1"):
            SessionConfig.from_cli_flags(max_iterations=-1)

    def test_from_cli_flags_with_invalid_compaction_threshold_raises(self):
        """from_cli_flags should reject compaction_threshold outside [0, 1]."""
        with pytest.raises(ValueError, match="compaction.threshold must be in \\[0.0, 1.0\\]"):
            SessionConfig.from_cli_flags(compaction_threshold=1.5)

    def test_from_cli_flags_with_invalid_bayesian_thresholds_raises(self):
        """from_cli_flags should reject bayesian thresholds outside [0, 1]."""
        with pytest.raises(ValueError, match="bayesian.simple_threshold must be in \\[0.0, 1.0\\]"):
            SessionConfig.from_cli_flags(simple_threshold=1.1)

        with pytest.raises(
            ValueError, match="bayesian.complex_threshold must be in \\[0.0, 1.0\\]"
        ):
            SessionConfig.from_cli_flags(complex_threshold=-0.5)
