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

"""Tests for safety migration backward compatibility (Phase 6.2).

These tests verify that the migration to SafetyPatternRegistry maintains
backward compatibility with existing vertical safety modules.
"""

import pytest

from victor.framework.safety import (
    SafetyPatternRegistry,
    Severity,
    register_vertical_patterns,
    register_builtin_scanners,
)


@pytest.fixture
def fresh_registry():
    """Create a fresh registry for testing."""
    return SafetyPatternRegistry()


class TestCodingSafetyMigration:
    """Tests for coding safety module backward compatibility."""

    def test_coding_safety_extension_import(self):
        """Test that CodingSafetyExtension can still be imported."""
        from victor.coding.safety import CodingSafetyExtension

        extension = CodingSafetyExtension()
        assert extension is not None

    def test_coding_safety_extension_category(self):
        """Test that CodingSafetyExtension returns correct category."""
        from victor.coding.safety import CodingSafetyExtension

        extension = CodingSafetyExtension()
        assert extension.get_category() == "coding"

    def test_coding_safety_get_bash_patterns(self):
        """Test that get_bash_patterns returns patterns."""
        from victor.coding.safety import CodingSafetyExtension

        extension = CodingSafetyExtension()
        patterns = extension.get_bash_patterns()
        assert len(patterns) > 0

    def test_coding_safety_get_file_patterns(self):
        """Test that get_file_patterns returns patterns."""
        from victor.coding.safety import CodingSafetyExtension

        extension = CodingSafetyExtension()
        patterns = extension.get_file_patterns()
        assert len(patterns) > 0

    def test_coding_safety_create_rules_functions(self):
        """Test that create_*_safety_rules functions are available."""
        from victor.coding.safety import (
            create_git_safety_rules,
            create_file_safety_rules,
            create_test_safety_rules,
            create_all_coding_safety_rules,
        )

        assert callable(create_git_safety_rules)
        assert callable(create_file_safety_rules)
        assert callable(create_test_safety_rules)
        assert callable(create_all_coding_safety_rules)

    def test_coding_patterns_export(self):
        """Test that pattern exports are available."""
        from victor.coding.safety import (
            GIT_DANGEROUS_PATTERNS,
            CODING_FILE_PATTERNS,
        )

        assert isinstance(GIT_DANGEROUS_PATTERNS, list)
        assert isinstance(CODING_FILE_PATTERNS, list)


class TestDevOpsSafetyMigration:
    """Tests for devops safety module backward compatibility."""

    def test_devops_safety_extension_import(self):
        """Test that DevOpsSafetyExtension can still be imported."""
        from victor.devops.safety import DevOpsSafetyExtension

        extension = DevOpsSafetyExtension()
        assert extension is not None

    def test_devops_safety_extension_category(self):
        """Test that DevOpsSafetyExtension returns correct category."""
        from victor.devops.safety import DevOpsSafetyExtension

        extension = DevOpsSafetyExtension()
        assert extension.get_category() == "devops"

    def test_devops_safety_get_bash_patterns(self):
        """Test that get_bash_patterns returns patterns."""
        from victor.devops.safety import DevOpsSafetyExtension

        extension = DevOpsSafetyExtension()
        patterns = extension.get_bash_patterns()
        assert len(patterns) > 0

    def test_devops_safety_get_credential_patterns(self):
        """Test that get_credential_patterns returns patterns."""
        from victor.devops.safety import DevOpsSafetyExtension

        extension = DevOpsSafetyExtension()
        patterns = extension.get_credential_patterns()
        assert isinstance(patterns, dict)

    def test_devops_safety_create_rules_functions(self):
        """Test that create_*_safety_rules functions are available."""
        from victor.devops.safety import (
            create_deployment_safety_rules,
            create_container_safety_rules,
            create_infrastructure_safety_rules,
            create_all_devops_safety_rules,
        )

        assert callable(create_deployment_safety_rules)
        assert callable(create_container_safety_rules)
        assert callable(create_infrastructure_safety_rules)
        assert callable(create_all_devops_safety_rules)

    def test_devops_patterns_export(self):
        """Test that pattern exports are available."""
        from victor.devops.safety import (
            DESTRUCTIVE_PATTERNS,
            KUBERNETES_PATTERNS,
            DOCKER_PATTERNS,
            TERRAFORM_PATTERNS,
        )

        assert isinstance(DESTRUCTIVE_PATTERNS, list)
        assert isinstance(KUBERNETES_PATTERNS, list)
        assert isinstance(DOCKER_PATTERNS, list)
        assert isinstance(TERRAFORM_PATTERNS, list)


class TestRegistryIntegration:
    """Tests for registry integration with vertical safety modules."""

    def test_coding_and_registry_patterns_both_work(self, fresh_registry):
        """Test that both old extension and new registry work together."""
        from victor.coding.safety import CodingSafetyExtension

        # Old way - using extension
        extension = CodingSafetyExtension()
        extension_patterns = extension.get_bash_patterns()

        # New way - using registry
        register_vertical_patterns("coding", registry=fresh_registry)
        registry_patterns = fresh_registry.list_patterns()

        # Both should have patterns
        assert len(extension_patterns) > 0
        assert len(registry_patterns) > 0

    def test_devops_and_registry_patterns_both_work(self, fresh_registry):
        """Test that both old extension and new registry work together."""
        from victor.devops.safety import DevOpsSafetyExtension

        # Old way - using extension
        extension = DevOpsSafetyExtension()
        extension_patterns = extension.get_bash_patterns()

        # New way - using registry
        register_vertical_patterns("devops", registry=fresh_registry)
        registry_patterns = fresh_registry.list_patterns()

        # Both should have patterns
        assert len(extension_patterns) > 0
        assert len(registry_patterns) > 0

    def test_registry_scan_matches_git_patterns(self, fresh_registry):
        """Test that registry can scan for git-related dangerous patterns."""
        register_vertical_patterns("coding", registry=fresh_registry)

        # Test various git commands
        violations = fresh_registry.scan("git push --force", domain="coding")
        assert len(violations) > 0

        violations = fresh_registry.scan("git reset --hard HEAD~5", domain="coding")
        assert len(violations) > 0

    def test_registry_scan_matches_devops_patterns(self, fresh_registry):
        """Test that registry can scan for devops-related dangerous patterns."""
        register_vertical_patterns("devops", registry=fresh_registry)

        # Test various devops commands
        violations = fresh_registry.scan("kubectl delete namespace production", domain="devops")
        assert len(violations) > 0

        violations = fresh_registry.scan("terraform destroy", domain="devops")
        assert len(violations) > 0


class TestFrameworkSafetyModuleExports:
    """Tests for framework safety module exports."""

    def test_framework_safety_exports_all_types(self):
        """Test that framework safety exports all required types."""
        from victor.framework.safety import (
            SafetyPattern,
            SafetyViolation,
            Severity,
            Action,
            SafetyPatternRegistry,
            BaseScanner,
            SecretScanner,
            CommandScanner,
            FilePathScanner,
            get_vertical_pattern_path,
            register_vertical_patterns,
            register_all_vertical_patterns,
            register_builtin_scanners,
        )

        # All imports should succeed
        assert SafetyPattern is not None
        assert SafetyViolation is not None
        assert Severity is not None
        assert Action is not None
        assert SafetyPatternRegistry is not None

    def test_severity_enum_values(self):
        """Test that Severity enum has expected values."""
        from victor.framework.safety import Severity

        assert Severity.CRITICAL.value == "critical"
        assert Severity.HIGH.value == "high"
        assert Severity.MEDIUM.value == "medium"
        assert Severity.LOW.value == "low"

    def test_action_enum_values(self):
        """Test that Action enum has expected values."""
        from victor.framework.safety import Action

        assert Action.BLOCK.value == "block"
        assert Action.WARN.value == "warn"
        assert Action.LOG.value == "log"
