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

"""Tests for safety pattern registry system (Phase 6.1).

TDD tests for:
- SafetyPattern dataclass
- SafetyPatternRegistry
- Built-in scanners
- YAML pattern loading
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List

import pytest
import yaml

from victor.framework.safety.types import (
    SafetyPattern,
    Severity,
    Action,
    SafetyViolation,
)
from victor.framework.safety.registry import SafetyPatternRegistry
from victor.framework.safety.scanners import (
    SecretScanner,
    CommandScanner,
    FilePathScanner,
)


# =============================================================================
# Test SafetyPattern Dataclass
# =============================================================================


class TestSafetyPattern:
    """Tests for SafetyPattern dataclass."""

    def test_pattern_has_required_fields(self):
        """Test that SafetyPattern has all required fields."""
        pattern = SafetyPattern(
            name="api_key",
            pattern=r"AKIA[0-9A-Z]{16}",
            severity=Severity.HIGH,
            message="AWS API key detected",
        )

        assert pattern.name == "api_key"
        assert pattern.pattern == r"AKIA[0-9A-Z]{16}"
        assert pattern.severity == Severity.HIGH
        assert pattern.message == "AWS API key detected"
        assert pattern.domains == ["all"]  # Default
        assert pattern.action == Action.BLOCK  # Default

    def test_pattern_matches_content(self):
        """Test that pattern matching works correctly."""
        pattern = SafetyPattern(
            name="force_push",
            pattern=r"git\s+push\s+.*--force",
            severity=Severity.CRITICAL,
            message="Force push detected",
        )

        # Should match
        assert pattern.matches("git push origin main --force")
        assert pattern.matches("git push --force")

        # Should not match
        assert not pattern.matches("git push origin main")
        assert not pattern.matches("git pull --force")

    def test_pattern_domain_filtering(self):
        """Test that domain filtering works correctly."""
        pattern = SafetyPattern(
            name="rm_rf",
            pattern=r"rm\s+-rf\s+/",
            severity=Severity.CRITICAL,
            message="Dangerous rm command",
            domains=["coding", "devops"],
        )

        # Should apply to specified domains
        assert pattern.applies_to_domain("coding")
        assert pattern.applies_to_domain("devops")

        # Should not apply to other domains
        assert not pattern.applies_to_domain("research")
        assert not pattern.applies_to_domain("rag")

    def test_pattern_applies_to_all_domains(self):
        """Test that 'all' domain applies to everything."""
        pattern = SafetyPattern(
            name="secret",
            pattern=r"secret",
            severity=Severity.HIGH,
            message="Secret detected",
            domains=["all"],
        )

        assert pattern.applies_to_domain("coding")
        assert pattern.applies_to_domain("devops")
        assert pattern.applies_to_domain("research")
        assert pattern.applies_to_domain("anything")

    def test_pattern_action_types(self):
        """Test different action types."""
        block_pattern = SafetyPattern(
            name="block_pattern",
            pattern=r"block_me",
            severity=Severity.CRITICAL,
            message="Blocked",
            action=Action.BLOCK,
        )

        warn_pattern = SafetyPattern(
            name="warn_pattern",
            pattern=r"warn_me",
            severity=Severity.MEDIUM,
            message="Warning",
            action=Action.WARN,
        )

        log_pattern = SafetyPattern(
            name="log_pattern",
            pattern=r"log_me",
            severity=Severity.LOW,
            message="Logged",
            action=Action.LOG,
        )

        assert block_pattern.action == Action.BLOCK
        assert warn_pattern.action == Action.WARN
        assert log_pattern.action == Action.LOG


# =============================================================================
# Test SafetyPatternRegistry
# =============================================================================


class TestSafetyPatternRegistry:
    """Tests for SafetyPatternRegistry."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry before each test."""
        SafetyPatternRegistry.reset_instance()
        yield
        SafetyPatternRegistry.reset_instance()

    def test_registry_singleton(self):
        """Test that SafetyPatternRegistry is a singleton."""
        registry1 = SafetyPatternRegistry.get_instance()
        registry2 = SafetyPatternRegistry.get_instance()

        assert registry1 is registry2

    def test_register_pattern(self):
        """Test registering a safety pattern."""
        registry = SafetyPatternRegistry.get_instance()
        pattern = SafetyPattern(
            name="test_pattern",
            pattern=r"test",
            severity=Severity.LOW,
            message="Test pattern",
        )

        registry.register_pattern(pattern)

        assert registry.get_pattern("test_pattern") is not None
        assert registry.get_pattern("test_pattern").name == "test_pattern"

    def test_register_duplicate_raises(self):
        """Test that registering duplicate pattern raises error."""
        registry = SafetyPatternRegistry.get_instance()
        pattern = SafetyPattern(
            name="dup_pattern",
            pattern=r"dup",
            severity=Severity.LOW,
            message="Duplicate",
        )

        registry.register_pattern(pattern)

        with pytest.raises(ValueError, match="already registered"):
            registry.register_pattern(pattern)

    def test_scan_with_patterns(self):
        """Test scanning content with registered patterns."""
        registry = SafetyPatternRegistry.get_instance()

        registry.register_pattern(
            SafetyPattern(
                name="api_key",
                pattern=r"api_key\s*=\s*['\"][^'\"]+['\"]",
                severity=Severity.HIGH,
                message="API key in code",
            )
        )

        registry.register_pattern(
            SafetyPattern(
                name="password",
                pattern=r"password\s*=\s*['\"][^'\"]+['\"]",
                severity=Severity.CRITICAL,
                message="Password in code",
            )
        )

        content = """
        api_key = "sk-12345"
        password = "secret123"
        username = "admin"
        """

        violations = registry.scan(content)

        assert len(violations) == 2
        assert any(v.pattern_name == "api_key" for v in violations)
        assert any(v.pattern_name == "password" for v in violations)

    def test_scan_filters_by_domain(self):
        """Test that scan filters patterns by domain."""
        registry = SafetyPatternRegistry.get_instance()

        # Pattern only for coding
        registry.register_pattern(
            SafetyPattern(
                name="coding_only",
                pattern=r"coding_pattern",
                severity=Severity.MEDIUM,
                message="Coding only",
                domains=["coding"],
            )
        )

        # Pattern for all domains
        registry.register_pattern(
            SafetyPattern(
                name="all_domains",
                pattern=r"universal_pattern",
                severity=Severity.MEDIUM,
                message="All domains",
                domains=["all"],
            )
        )

        content = "coding_pattern universal_pattern"

        # Scan for coding domain
        coding_violations = registry.scan(content, domain="coding")
        assert len(coding_violations) == 2

        # Scan for research domain
        research_violations = registry.scan(content, domain="research")
        assert len(research_violations) == 1
        assert research_violations[0].pattern_name == "all_domains"

    def test_load_patterns_from_yaml(self):
        """Test loading patterns from YAML file."""
        registry = SafetyPatternRegistry.get_instance()

        yaml_content = """
patterns:
  - name: yaml_pattern1
    pattern: "yaml_test_1"
    severity: high
    message: YAML pattern 1
  - name: yaml_pattern2
    pattern: "yaml_test_2"
    severity: critical
    message: YAML pattern 2
    action: warn
    domains:
      - coding
      - devops
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)

        try:
            count = registry.load_from_yaml(temp_path)

            assert count == 2
            assert registry.get_pattern("yaml_pattern1") is not None
            assert registry.get_pattern("yaml_pattern2") is not None
            assert registry.get_pattern("yaml_pattern1").severity == Severity.HIGH
            assert registry.get_pattern("yaml_pattern2").action == Action.WARN
        finally:
            temp_path.unlink()

    def test_list_patterns(self):
        """Test listing all registered patterns."""
        registry = SafetyPatternRegistry.get_instance()

        registry.register_pattern(
            SafetyPattern(
                name="p1", pattern=r"p1", severity=Severity.LOW, message="P1"
            )
        )
        registry.register_pattern(
            SafetyPattern(
                name="p2", pattern=r"p2", severity=Severity.HIGH, message="P2"
            )
        )

        patterns = registry.list_patterns()

        assert len(patterns) == 2
        assert "p1" in patterns
        assert "p2" in patterns

    def test_list_patterns_by_severity(self):
        """Test listing patterns by severity."""
        registry = SafetyPatternRegistry.get_instance()

        registry.register_pattern(
            SafetyPattern(
                name="critical1",
                pattern=r"c1",
                severity=Severity.CRITICAL,
                message="Critical 1",
            )
        )
        registry.register_pattern(
            SafetyPattern(
                name="high1",
                pattern=r"h1",
                severity=Severity.HIGH,
                message="High 1",
            )
        )
        registry.register_pattern(
            SafetyPattern(
                name="critical2",
                pattern=r"c2",
                severity=Severity.CRITICAL,
                message="Critical 2",
            )
        )

        critical_patterns = registry.list_by_severity(Severity.CRITICAL)
        high_patterns = registry.list_by_severity(Severity.HIGH)

        assert len(critical_patterns) == 2
        assert len(high_patterns) == 1

    def test_unregister_pattern(self):
        """Test unregistering a pattern."""
        registry = SafetyPatternRegistry.get_instance()
        pattern = SafetyPattern(
            name="to_remove",
            pattern=r"remove",
            severity=Severity.LOW,
            message="Remove me",
        )

        registry.register_pattern(pattern)
        assert registry.get_pattern("to_remove") is not None

        registry.unregister("to_remove")
        assert registry.get_pattern("to_remove") is None


# =============================================================================
# Test Built-In Scanners
# =============================================================================


class TestBuiltInScanners:
    """Tests for built-in safety scanners."""

    def test_secret_scanner_detects_api_keys(self):
        """Test that SecretScanner detects common API key patterns."""
        scanner = SecretScanner()

        content_with_secrets = """
        # AWS credentials
        aws_access_key = "AKIAIOSFODNN7EXAMPLE"
        aws_secret_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"

        # GitHub token
        GITHUB_TOKEN = "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

        # Generic API key
        api_key = "sk-proj-1234567890abcdef"
        """

        violations = scanner.scan(content_with_secrets)

        # Should detect at least the AWS key and GitHub token patterns
        assert len(violations) >= 2

    def test_secret_scanner_ignores_safe_content(self):
        """Test that SecretScanner doesn't flag safe content."""
        scanner = SecretScanner()

        safe_content = """
        # Configuration
        max_retries = 3
        timeout = 30

        # Regular code
        def get_user(user_id):
            return db.find(user_id)
        """

        violations = scanner.scan(safe_content)

        assert len(violations) == 0

    def test_command_scanner_detects_dangerous_commands(self):
        """Test that CommandScanner detects dangerous shell commands."""
        scanner = CommandScanner()

        dangerous_content = """
        # Dangerous commands
        rm -rf /
        chmod 777 /etc/passwd
        git push --force origin main
        sudo rm -rf /*
        """

        violations = scanner.scan(dangerous_content)

        # Should detect multiple dangerous patterns
        assert len(violations) >= 3

    def test_command_scanner_allows_safe_commands(self):
        """Test that CommandScanner allows safe commands."""
        scanner = CommandScanner()

        safe_content = """
        # Safe commands
        ls -la
        git status
        npm install
        python setup.py install
        """

        violations = scanner.scan(safe_content)

        assert len(violations) == 0

    def test_file_path_scanner_detects_sensitive_paths(self):
        """Test that FilePathScanner detects sensitive file paths."""
        scanner = FilePathScanner()

        sensitive_content = """
        # Accessing sensitive files
        cat /etc/shadow
        vim ~/.ssh/id_rsa
        read_file("/etc/passwd")
        open("~/.aws/credentials")
        """

        violations = scanner.scan(sensitive_content)

        # Should detect sensitive paths
        assert len(violations) >= 2

    def test_scanner_returns_violations_with_metadata(self):
        """Test that scanner violations include proper metadata."""
        scanner = SecretScanner()

        content = 'api_key = "AKIAIOSFODNN7EXAMPLE"'
        violations = scanner.scan(content)

        if violations:
            v = violations[0]
            assert v.pattern_name is not None
            assert v.severity in [
                Severity.CRITICAL,
                Severity.HIGH,
                Severity.MEDIUM,
                Severity.LOW,
            ]
            assert v.message is not None
            assert v.matched_text is not None


# =============================================================================
# Test YAML Schema Validation
# =============================================================================


class TestYAMLSchemaValidation:
    """Tests for YAML pattern file schema validation."""

    def test_valid_yaml_schema(self):
        """Test that valid YAML schema passes validation."""
        yaml_content = {
            "version": "1.0",
            "patterns": [
                {
                    "name": "valid_pattern",
                    "pattern": r"test",
                    "severity": "high",
                    "message": "Test message",
                }
            ],
        }

        from victor.framework.safety.registry import validate_pattern_yaml

        errors = validate_pattern_yaml(yaml_content)
        assert len(errors) == 0

    def test_invalid_yaml_missing_required(self):
        """Test that YAML missing required fields fails validation."""
        yaml_content = {
            "patterns": [
                {
                    "name": "invalid_pattern",
                    # Missing pattern and severity
                }
            ],
        }

        from victor.framework.safety.registry import validate_pattern_yaml

        errors = validate_pattern_yaml(yaml_content)
        assert len(errors) > 0

    def test_invalid_severity_level(self):
        """Test that invalid severity level fails validation."""
        yaml_content = {
            "patterns": [
                {
                    "name": "bad_severity",
                    "pattern": r"test",
                    "severity": "extreme",  # Invalid
                    "message": "Test",
                }
            ],
        }

        from victor.framework.safety.registry import validate_pattern_yaml

        errors = validate_pattern_yaml(yaml_content)
        assert len(errors) > 0


# =============================================================================
# Test Integration with Existing SafetyRegistry
# =============================================================================


class TestLegacyIntegration:
    """Tests for integration with existing SafetyRegistry."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry before each test."""
        SafetyPatternRegistry.reset_instance()
        yield
        SafetyPatternRegistry.reset_instance()

    def test_register_scanner(self):
        """Test registering a scanner for backward compatibility."""
        registry = SafetyPatternRegistry.get_instance()

        # Should be able to register scanners (like the existing ISafetyScanner)
        scanner = SecretScanner()
        registry.register_scanner("secrets", scanner)

        assert registry.get_scanner("secrets") is scanner

    def test_scan_with_scanner(self):
        """Test scanning with registered scanner."""
        registry = SafetyPatternRegistry.get_instance()

        scanner = SecretScanner()
        registry.register_scanner("secrets", scanner)

        content = 'api_key = "AKIAIOSFODNN7EXAMPLE"'

        # Should be able to scan with specific scanner
        violations = registry.scan_with_scanner("secrets", content)

        # SecretScanner should find the AWS key pattern
        assert len(violations) >= 1
