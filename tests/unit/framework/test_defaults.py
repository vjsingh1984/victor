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

"""Tests for victor.framework.defaults package."""

import re

import pytest

from victor.framework.defaults import (
    DEFAULT_ACTIVE_LEARNERS,
    DEFAULT_MODES,
    DEFAULT_PATIENCE_MAP,
    DEFAULT_TASK_BUDGETS,
    BaseRLConfig,
    DefaultSafetyExtension,
    ModeConfigRegistry,
    ModeDefinition,
    RegistryBasedModeConfigProvider,
    StageDefinition,
    get_default_stages,
)
from victor.framework.defaults.safety import DefaultSafetyExtension as DirectSafety
from victor.framework.defaults.stages import get_default_stages as direct_get_stages
from victor.framework.stage_manager import get_default_stages as sm_get_stages

# =============================================================================
# Default Stages
# =============================================================================


class TestGetDefaultStages:
    """Tests for get_default_stages()."""

    def test_returns_seven_stages(self):
        stages = get_default_stages()
        assert len(stages) == 7

    def test_stage_names(self):
        stages = get_default_stages()
        expected = {
            "initial",
            "planning",
            "reading",
            "analysis",
            "execution",
            "verification",
            "completion",
        }
        assert set(stages.keys()) == expected

    def test_stages_are_stage_definitions(self):
        stages = get_default_stages()
        for defn in stages.values():
            assert isinstance(defn, StageDefinition)

    def test_ordering_is_sequential(self):
        stages = get_default_stages()
        orders = [defn.order for defn in stages.values()]
        assert sorted(orders) == list(range(7))

    def test_all_stages_have_keywords(self):
        stages = get_default_stages()
        for name, defn in stages.items():
            assert len(defn.keywords) > 0, f"Stage '{name}' has no keywords"

    def test_all_stages_have_tools(self):
        stages = get_default_stages()
        for name, defn in stages.items():
            assert len(defn.tools) > 0, f"Stage '{name}' has no tools"

    def test_initial_stage_is_order_zero(self):
        stages = get_default_stages()
        assert stages["initial"].order == 0

    def test_completion_is_last(self):
        stages = get_default_stages()
        assert stages["completion"].order == 6

    def test_returns_fresh_dict_each_call(self):
        stages1 = get_default_stages()
        stages2 = get_default_stages()
        assert stages1 is not stages2

    def test_reexport_matches_stage_manager(self):
        """Ensure the defaults re-export matches stage_manager source."""
        assert direct_get_stages is sm_get_stages


# =============================================================================
# Default Safety Extension
# =============================================================================


class TestDefaultSafetyExtension:
    """Tests for DefaultSafetyExtension."""

    @pytest.fixture()
    def ext(self):
        return DefaultSafetyExtension()

    def test_get_bash_patterns_returns_list(self, ext):
        patterns = ext.get_bash_patterns()
        assert isinstance(patterns, list)
        assert len(patterns) > 0

    def test_bash_patterns_cover_git(self, ext):
        categories = {p.category for p in ext.get_bash_patterns()}
        assert "git" in categories

    def test_bash_patterns_cover_filesystem(self, ext):
        categories = {p.category for p in ext.get_bash_patterns()}
        assert "filesystem" in categories

    def test_bash_patterns_have_critical_entries(self, ext):
        risk_levels = {p.risk_level for p in ext.get_bash_patterns()}
        assert "CRITICAL" in risk_levels

    def test_get_file_patterns_returns_list(self, ext):
        patterns = ext.get_file_patterns()
        assert isinstance(patterns, list)
        assert len(patterns) > 0

    def test_file_patterns_cover_credentials(self, ext):
        categories = {p.category for p in ext.get_file_patterns()}
        assert "credentials" in categories

    def test_file_patterns_cover_system(self, ext):
        categories = {p.category for p in ext.get_file_patterns()}
        assert "system" in categories

    def test_get_tool_restrictions(self, ext):
        restrictions = ext.get_tool_restrictions()
        assert "write_file" in restrictions
        assert "shell" in restrictions

    def test_category_is_default(self, ext):
        assert ext.get_category() == "default"

    def test_force_push_pattern_matches(self, ext):
        pattern = next(p for p in ext.get_bash_patterns() if "force" in p.description.lower())
        assert re.search(pattern.pattern, "git push origin main --force")

    def test_hard_reset_pattern_matches(self, ext):
        pattern = next(p for p in ext.get_bash_patterns() if "hard reset" in p.description.lower())
        assert re.search(pattern.pattern, "git reset --hard HEAD~3")

    def test_rm_rf_root_pattern_matches(self, ext):
        pattern = next(
            p
            for p in ext.get_bash_patterns()
            if "root" in p.description.lower() and "delete" in p.description.lower()
        )
        assert re.search(pattern.pattern, "rm -rf /")

    def test_env_file_pattern_matches(self, ext):
        pattern = next(p for p in ext.get_file_patterns() if "environment" in p.description.lower())
        assert re.search(pattern.pattern, "config/.env")

    def test_key_file_pattern_matches(self, ext):
        pattern = next(p for p in ext.get_file_patterns() if "certificate" in p.description.lower())
        assert re.search(pattern.pattern, "server.key")
        assert re.search(pattern.pattern, "cert.pem")

    def test_implements_safety_protocol(self, ext):
        from victor.core.verticals.protocols.safety_provider import (
            SafetyExtensionProtocol,
        )

        assert isinstance(ext, SafetyExtensionProtocol)

    def test_reexport_matches_direct(self):
        assert DirectSafety is DefaultSafetyExtension


# =============================================================================
# Subclass inheritance
# =============================================================================


class TestSafetySubclass:
    """Verify that subclassing DefaultSafetyExtension works correctly."""

    def test_subclass_extends_bash_patterns(self):
        from victor.security.safety.types import SafetyPattern

        class CustomSafety(DefaultSafetyExtension):
            def get_bash_patterns(self):
                patterns = super().get_bash_patterns()
                patterns.append(
                    SafetyPattern(
                        pattern=r"custom_danger",
                        description="Custom danger",
                        risk_level="HIGH",
                        category="custom",
                    )
                )
                return patterns

        ext = CustomSafety()
        patterns = ext.get_bash_patterns()
        custom = [p for p in patterns if p.category == "custom"]
        assert len(custom) == 1
        # base patterns still present
        git_patterns = [p for p in patterns if p.category == "git"]
        assert len(git_patterns) > 0


# =============================================================================
# One-stop imports
# =============================================================================


class TestDefaultsReexports:
    """Verify all re-exports are importable from victor.framework.defaults."""

    def test_get_default_stages(self):
        assert callable(get_default_stages)

    def test_default_safety_extension(self):
        assert DefaultSafetyExtension is not None

    def test_stage_definition(self):
        assert StageDefinition is not None

    def test_base_rl_config(self):
        assert BaseRLConfig is not None

    def test_default_active_learners(self):
        assert isinstance(DEFAULT_ACTIVE_LEARNERS, list)

    def test_default_patience_map(self):
        assert isinstance(DEFAULT_PATIENCE_MAP, dict)

    def test_mode_config_registry(self):
        assert ModeConfigRegistry is not None

    def test_mode_definition(self):
        assert ModeDefinition is not None

    def test_registry_based_mode_config_provider(self):
        assert RegistryBasedModeConfigProvider is not None

    def test_default_modes(self):
        assert isinstance(DEFAULT_MODES, dict)

    def test_default_task_budgets(self):
        assert isinstance(DEFAULT_TASK_BUDGETS, dict)
