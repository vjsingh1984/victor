# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for RLConfigFactory (Phase 9.1).

Tests the factory pattern for creating vertical-specific RL configs,
consolidating duplicate config code across verticals.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml

from victor.framework.rl import LearnerType
from victor.framework.rl.config import BaseRLConfig

if TYPE_CHECKING:
    pass


class TestRLConfigFactory:
    """Tests for RLConfigFactory class."""

    def test_create_coding_config(self):
        """Test creating coding vertical config."""
        from victor.framework.rl.config_factory import RLConfigFactory

        config = RLConfigFactory.create("coding")

        assert config is not None
        assert isinstance(config, BaseRLConfig)
        # Coding has additional learners
        assert LearnerType.TOOL_SELECTOR in config.active_learners
        # Should have coding-specific task mappings
        assert "debugging" in config.task_type_mappings or "refactoring" in config.task_type_mappings

    def test_create_devops_config(self):
        """Test creating devops vertical config."""
        from victor.framework.rl.config_factory import RLConfigFactory

        config = RLConfigFactory.create("devops")

        assert config is not None
        assert isinstance(config, BaseRLConfig)
        # DevOps should have deployment-related mappings
        assert "deployment" in config.task_type_mappings or "containerization" in config.task_type_mappings

    def test_create_rag_config(self):
        """Test creating RAG vertical config."""
        from victor.framework.rl.config_factory import RLConfigFactory

        config = RLConfigFactory.create("rag")

        assert config is not None
        assert isinstance(config, BaseRLConfig)
        # RAG should have search-related mappings
        assert "search" in config.task_type_mappings or "synthesis" in config.task_type_mappings

    def test_create_dataanalysis_config(self):
        """Test creating data analysis vertical config."""
        from victor.framework.rl.config_factory import RLConfigFactory

        config = RLConfigFactory.create("dataanalysis")

        assert config is not None
        assert isinstance(config, BaseRLConfig)
        # DataAnalysis should have EDA/visualization mappings
        assert "eda" in config.task_type_mappings or "visualization" in config.task_type_mappings

    def test_create_research_config(self):
        """Test creating research vertical config."""
        from victor.framework.rl.config_factory import RLConfigFactory

        config = RLConfigFactory.create("research")

        assert config is not None
        assert isinstance(config, BaseRLConfig)
        # Research should have research-related mappings
        assert "research" in config.task_type_mappings or "fact_check" in config.task_type_mappings

    def test_factory_returns_base_for_unknown_vertical(self):
        """Test that factory returns base config for unknown verticals."""
        from victor.framework.rl.config_factory import RLConfigFactory

        config = RLConfigFactory.create("unknown_vertical")

        assert config is not None
        assert isinstance(config, BaseRLConfig)
        # Should have default active learners
        assert LearnerType.TOOL_SELECTOR in config.active_learners

    def test_factory_caches_configs(self):
        """Test that factory caches configs for performance."""
        from victor.framework.rl.config_factory import RLConfigFactory

        config1 = RLConfigFactory.create("coding")
        config2 = RLConfigFactory.create("coding")

        # Should return cached instance
        assert config1 is config2

    def test_create_from_yaml(self):
        """Test creating config from YAML file."""
        from victor.framework.rl.config_factory import RLConfigFactory

        # Create test YAML
        yaml_content = {
            "verticals": {
                "test_vertical": {
                    "active_learners": ["tool_selector", "continuation_patience"],
                    "task_type_mappings": {
                        "custom_task": ["read", "write"]
                    },
                    "quality_thresholds": {
                        "custom_task": 0.85
                    },
                    "default_patience": {
                        "anthropic": 5,
                        "openai": 4
                    },
                    "exploration_bonus": 0.2
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            yaml_path = Path(f.name)

        try:
            config = RLConfigFactory.create_from_yaml(yaml_path, "test_vertical")

            assert config is not None
            assert isinstance(config, BaseRLConfig)
            assert "custom_task" in config.task_type_mappings
            assert config.task_type_mappings["custom_task"] == ["read", "write"]
            assert config.quality_thresholds["custom_task"] == 0.85
            assert config.exploration_bonus == 0.2
        finally:
            yaml_path.unlink()

    def test_create_from_yaml_missing_vertical(self):
        """Test creating config from YAML with missing vertical returns base."""
        from victor.framework.rl.config_factory import RLConfigFactory

        yaml_content = {"verticals": {}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            yaml_path = Path(f.name)

        try:
            config = RLConfigFactory.create_from_yaml(yaml_path, "nonexistent")
            assert config is not None
            # Should return base config with defaults
            assert isinstance(config, BaseRLConfig)
        finally:
            yaml_path.unlink()

    def test_vertical_specific_learners(self):
        """Test that verticals can have different active learners."""
        from victor.framework.rl.config_factory import RLConfigFactory

        coding_config = RLConfigFactory.create("coding")
        rag_config = RLConfigFactory.create("rag")

        # Coding and RAG may have different learner sets
        # Both should have TOOL_SELECTOR
        assert LearnerType.TOOL_SELECTOR in coding_config.active_learners
        assert LearnerType.TOOL_SELECTOR in rag_config.active_learners

    def test_config_methods_work(self):
        """Test that config methods from BaseRLConfig work properly."""
        from victor.framework.rl.config_factory import RLConfigFactory

        config = RLConfigFactory.create("coding")

        # Test inherited methods
        tools = config.get_tools_for_task("debugging")
        assert isinstance(tools, list)

        threshold = config.get_quality_threshold("debugging")
        assert 0.0 <= threshold <= 1.0

        patience = config.get_patience("anthropic")
        assert patience > 0

    def test_clear_cache(self):
        """Test that cache can be cleared."""
        from victor.framework.rl.config_factory import RLConfigFactory

        # Create and cache a config
        config1 = RLConfigFactory.create("coding")

        # Clear cache
        RLConfigFactory.clear_cache()

        # Create again - should be new instance
        config2 = RLConfigFactory.create("coding")

        # After cache clear, should be different instances
        assert config1 is not config2


class TestGenericRLHooks:
    """Tests for GenericRLHooks class that consolidates vertical hooks."""

    def test_hooks_initialization(self):
        """Test hooks can be initialized with config."""
        from victor.framework.rl.config_factory import GenericRLHooks, RLConfigFactory

        config = RLConfigFactory.create("coding")
        hooks = GenericRLHooks(config)

        assert hooks.config is config

    def test_hooks_default_config(self):
        """Test hooks use default config if none provided."""
        from victor.framework.rl.config_factory import GenericRLHooks

        hooks = GenericRLHooks()

        assert hooks.config is not None
        assert isinstance(hooks.config, BaseRLConfig)

    def test_get_tool_recommendation(self):
        """Test tool recommendation method."""
        from victor.framework.rl.config_factory import GenericRLHooks, RLConfigFactory

        config = RLConfigFactory.create("coding")
        hooks = GenericRLHooks(config)

        # Get tools for a task type
        tools = hooks.get_tool_recommendation("debugging")
        assert isinstance(tools, list)

    def test_get_tool_recommendation_with_filter(self):
        """Test tool recommendation with available tools filter."""
        from victor.framework.rl.config_factory import GenericRLHooks, RLConfigFactory

        config = RLConfigFactory.create("coding")
        hooks = GenericRLHooks(config)

        # Get tools with filter
        available = ["read", "write"]
        tools = hooks.get_tool_recommendation("debugging", available_tools=available)

        # Should only return tools that are in available list
        for tool in tools:
            assert tool in available

    def test_get_patience_recommendation(self):
        """Test patience recommendation method."""
        from victor.framework.rl.config_factory import GenericRLHooks, RLConfigFactory

        config = RLConfigFactory.create("coding")
        hooks = GenericRLHooks(config)

        patience = hooks.get_patience_recommendation("anthropic", "claude-3-opus")
        assert patience > 0

    def test_get_quality_threshold(self):
        """Test quality threshold method."""
        from victor.framework.rl.config_factory import GenericRLHooks, RLConfigFactory

        config = RLConfigFactory.create("coding")
        hooks = GenericRLHooks(config)

        threshold = hooks.get_quality_threshold("debugging")
        assert 0.0 <= threshold <= 1.0

    def test_hooks_repr(self):
        """Test hooks string representation."""
        from victor.framework.rl.config_factory import GenericRLHooks

        hooks = GenericRLHooks()

        repr_str = repr(hooks)
        assert "GenericRLHooks" in repr_str


class TestUnifiedRLConfig:
    """Tests for unified RL config loading."""

    def test_unified_config_exists(self):
        """Test that unified config YAML exists."""
        from victor.framework.rl.config_factory import RLConfigFactory

        # The factory should be able to load without errors
        # This implicitly tests that the config exists
        config = RLConfigFactory.create("coding")
        assert config is not None

    def test_all_verticals_have_config(self):
        """Test that all standard verticals have config."""
        from victor.framework.rl.config_factory import RLConfigFactory

        verticals = ["coding", "devops", "rag", "dataanalysis", "research"]

        for vertical in verticals:
            config = RLConfigFactory.create(vertical)
            assert config is not None, f"Missing config for {vertical}"
            assert isinstance(config, BaseRLConfig)

    def test_config_consistency_with_existing(self):
        """Test that factory configs are consistent with existing vertical configs."""
        from victor.framework.rl.config_factory import RLConfigFactory

        # Get factory config
        factory_config = RLConfigFactory.create("coding")

        # Verify it has expected structure
        assert hasattr(factory_config, "active_learners")
        assert hasattr(factory_config, "task_type_mappings")
        assert hasattr(factory_config, "quality_thresholds")
        assert hasattr(factory_config, "default_patience")
