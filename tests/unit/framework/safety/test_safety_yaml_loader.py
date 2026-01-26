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

"""Tests for safety YAML loader (Phase 6.2)."""

import tempfile
from pathlib import Path

import pytest

from victor.framework.safety import (
    SafetyPatternRegistry,
    Severity,
    get_vertical_pattern_path,
    register_vertical_patterns,
    register_all_vertical_patterns,
    register_builtin_scanners,
)


@pytest.fixture
def fresh_registry():
    """Create a fresh registry for testing."""
    return SafetyPatternRegistry()


class TestGetVerticalPatternPath:
    """Tests for get_vertical_pattern_path function."""

    def test_coding_vertical_path(self):
        """Test path generation for coding vertical."""
        path = get_vertical_pattern_path("coding")
        assert path.name == "patterns.yaml"
        assert "coding" in str(path)
        assert "config" in str(path)

    def test_devops_vertical_path(self):
        """Test path generation for devops vertical."""
        path = get_vertical_pattern_path("devops")
        assert path.name == "patterns.yaml"
        assert "devops" in str(path)


class TestRegisterVerticalPatterns:
    """Tests for register_vertical_patterns function."""

    def test_register_coding_patterns(self, fresh_registry):
        """Test registering coding vertical patterns."""
        count = register_vertical_patterns("coding", registry=fresh_registry)
        assert count > 0
        # Check some expected patterns
        pattern = fresh_registry.get_pattern("coding_git_force_push")
        assert pattern is not None
        assert pattern.severity == Severity.CRITICAL

    def test_register_devops_patterns(self, fresh_registry):
        """Test registering devops vertical patterns."""
        count = register_vertical_patterns("devops", registry=fresh_registry)
        assert count > 0
        pattern = fresh_registry.get_pattern("devops_kubectl_delete_namespace")
        assert pattern is not None
        assert pattern.severity == Severity.CRITICAL

    def test_register_nonexistent_vertical_returns_zero(self, fresh_registry):
        """Test registering from nonexistent vertical returns 0."""
        count = register_vertical_patterns("nonexistent", registry=fresh_registry)
        assert count == 0

    def test_register_with_custom_path(self, fresh_registry):
        """Test registering from custom YAML path."""
        yaml_content = """
version: "1.0"
patterns:
  - name: custom_pattern
    pattern: "dangerous_command"
    severity: high
    message: A custom dangerous pattern
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)

        try:
            count = register_vertical_patterns(
                "custom", yaml_path=temp_path, registry=fresh_registry
            )
            assert count == 1
            pattern = fresh_registry.get_pattern("custom_pattern")
            assert pattern is not None
            assert pattern.severity == Severity.HIGH
        finally:
            temp_path.unlink()

    def test_register_with_invalid_yaml_returns_zero(self, fresh_registry):
        """Test registering from invalid YAML returns 0."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = Path(f.name)

        try:
            count = register_vertical_patterns(
                "invalid", yaml_path=temp_path, registry=fresh_registry
            )
            assert count == 0
        finally:
            temp_path.unlink()


class TestRegisterAllVerticalPatterns:
    """Tests for register_all_vertical_patterns function."""

    def test_register_all_verticals(self, fresh_registry):
        """Test registering all vertical patterns."""
        total = register_all_vertical_patterns(registry=fresh_registry)
        # Should have patterns from coding and devops (others may not have patterns yet)
        assert total >= 10  # At least some patterns from each vertical

    def test_coding_and_devops_have_patterns(self, fresh_registry):
        """Test that coding and devops verticals have patterns registered."""
        register_all_vertical_patterns(registry=fresh_registry)

        # Check at least one pattern from each vertical
        assert fresh_registry.get_pattern("coding_git_force_push") is not None
        assert fresh_registry.get_pattern("devops_kubectl_delete_namespace") is not None


class TestRegisterBuiltinScanners:
    """Tests for register_builtin_scanners function."""

    def test_register_scanners(self, fresh_registry):
        """Test registering built-in scanners."""
        count = register_builtin_scanners(registry=fresh_registry)
        assert count == 3  # secrets, commands, filepaths

    def test_scanners_available_after_registration(self, fresh_registry):
        """Test that scanners are available after registration."""
        register_builtin_scanners(registry=fresh_registry)

        assert fresh_registry.get_scanner("secrets") is not None
        assert fresh_registry.get_scanner("commands") is not None
        assert fresh_registry.get_scanner("filepaths") is not None

    def test_scanners_listed(self, fresh_registry):
        """Test that scanners are listed."""
        register_builtin_scanners(registry=fresh_registry)

        scanners = fresh_registry.list_scanners()
        assert "secrets" in scanners
        assert "commands" in scanners
        assert "filepaths" in scanners


class TestPatternYAMLContents:
    """Tests verifying the content of pattern YAML files."""

    def test_coding_yaml_has_git_patterns(self, fresh_registry):
        """Test coding YAML has git-specific patterns."""
        register_vertical_patterns("coding", registry=fresh_registry)

        # Check git patterns
        force_push = fresh_registry.get_pattern("coding_git_force_push")
        assert force_push is not None
        assert force_push.severity == Severity.CRITICAL
        assert "force" in force_push.message.lower()

        reset_hard = fresh_registry.get_pattern("coding_git_reset_hard")
        assert reset_hard is not None
        assert reset_hard.severity == Severity.HIGH

    def test_coding_yaml_has_file_patterns(self, fresh_registry):
        """Test coding YAML has file operation patterns."""
        register_vertical_patterns("coding", registry=fresh_registry)

        rm_recursive = fresh_registry.get_pattern("coding_rm_recursive")
        assert rm_recursive is not None
        assert rm_recursive.severity == Severity.CRITICAL

        chmod_777 = fresh_registry.get_pattern("coding_chmod_777")
        assert chmod_777 is not None

    def test_devops_yaml_has_kubernetes_patterns(self, fresh_registry):
        """Test devops YAML has Kubernetes patterns."""
        register_vertical_patterns("devops", registry=fresh_registry)

        delete_ns = fresh_registry.get_pattern("devops_kubectl_delete_namespace")
        assert delete_ns is not None
        assert delete_ns.severity == Severity.CRITICAL

        delete_all = fresh_registry.get_pattern("devops_kubectl_delete_all")
        assert delete_all is not None

    def test_devops_yaml_has_docker_patterns(self, fresh_registry):
        """Test devops YAML has Docker patterns."""
        register_vertical_patterns("devops", registry=fresh_registry)

        privileged = fresh_registry.get_pattern("devops_docker_privileged")
        assert privileged is not None
        assert privileged.severity == Severity.CRITICAL

    def test_devops_yaml_has_terraform_patterns(self, fresh_registry):
        """Test devops YAML has Terraform patterns."""
        register_vertical_patterns("devops", registry=fresh_registry)

        destroy = fresh_registry.get_pattern("devops_terraform_destroy")
        assert destroy is not None
        assert destroy.severity == Severity.CRITICAL

        auto_approve = fresh_registry.get_pattern("devops_terraform_apply_auto_approve")
        assert auto_approve is not None


class TestPatternScanning:
    """Tests for pattern scanning functionality."""

    def test_scan_detects_git_force_push(self, fresh_registry):
        """Test scanning detects git force push commands."""
        register_vertical_patterns("coding", registry=fresh_registry)

        violations = fresh_registry.scan("git push --force origin main", domain="coding")
        assert len(violations) > 0
        assert any(v.pattern_name == "coding_git_force_push" for v in violations)

    def test_scan_detects_kubectl_delete(self, fresh_registry):
        """Test scanning detects kubectl delete namespace."""
        register_vertical_patterns("devops", registry=fresh_registry)

        violations = fresh_registry.scan("kubectl delete namespace production", domain="devops")
        assert len(violations) > 0
        assert any(v.pattern_name == "devops_kubectl_delete_namespace" for v in violations)

    def test_scan_filters_by_domain(self, fresh_registry):
        """Test that scanning filters patterns by domain."""
        register_vertical_patterns("coding", registry=fresh_registry)
        register_vertical_patterns("devops", registry=fresh_registry)

        # Scan with coding domain - should only match coding patterns
        violations = fresh_registry.scan("git push --force", domain="coding")
        for v in violations:
            # All should be coding patterns or "all" domain
            assert "coding" in v.pattern_name or v.pattern_name.startswith("coding_")

    def test_safe_command_no_violations(self, fresh_registry):
        """Test that safe commands produce no violations."""
        register_vertical_patterns("coding", registry=fresh_registry)

        violations = fresh_registry.scan("git status", domain="coding")
        assert len(violations) == 0
