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

"""Unit tests for PrivacyCapabilityProvider and privacy configuration functions.

Tests for framework-level privacy capability that provides PII detection,
secrets masking, and audit logging across all verticals.
"""

import pytest
from typing import Any

from victor.framework.capabilities.privacy import (
    # Configuration functions
    configure_data_privacy,
    configure_secrets_masking,
    configure_audit_logging,
    # Getter functions
    get_privacy_config,
    get_secrets_masking_config,
    get_audit_logging_config,
    # Provider class
    PrivacyCapabilityProvider,
    # Capability entries
    CAPABILITIES,
    get_framework_privacy_capabilities,
)


# =============================================================================
# Mock Orchestrator for Testing
# =============================================================================


class MockOrchestrator:
    """Mock orchestrator for testing privacy configuration."""

    def __init__(self):
        self.privacy_config: dict[str, Any] = {}
        self.secrets_masking_config: dict[str, Any] = {}
        self.audit_logging_config: dict[str, Any] = {}


class MinimalOrchestrator:
    """Minimal orchestrator without pre-existing config attributes."""

    pass


# =============================================================================
# configure_data_privacy() Tests
# =============================================================================


class TestConfigureDataPrivacy:
    """Tests for configure_data_privacy function."""

    def test_configure_with_defaults(self):
        """configure_data_privacy should apply default settings."""
        orchestrator = MockOrchestrator()
        configure_data_privacy(orchestrator)

        assert orchestrator.privacy_config["anonymize_pii"] is True
        assert orchestrator.privacy_config["hash_identifiers"] is True
        assert orchestrator.privacy_config["log_access"] is True
        assert orchestrator.privacy_config["detect_secrets"] is True
        assert orchestrator.privacy_config["pii_columns"] == []
        assert isinstance(orchestrator.privacy_config["secret_patterns"], list)

    def test_configure_with_custom_settings(self):
        """configure_data_privacy should accept custom settings."""
        orchestrator = MockOrchestrator()
        custom_patterns = [r"custom_\w+", r"secret_\d+"]
        pii_columns = ["email", "ssn", "phone"]

        configure_data_privacy(
            orchestrator,
            anonymize_pii=False,
            pii_columns=pii_columns,
            hash_identifiers=False,
            log_access=False,
            detect_secrets=False,
            secret_patterns=custom_patterns,
        )

        assert orchestrator.privacy_config["anonymize_pii"] is False
        assert orchestrator.privacy_config["pii_columns"] == pii_columns
        assert orchestrator.privacy_config["hash_identifiers"] is False
        assert orchestrator.privacy_config["log_access"] is False
        assert orchestrator.privacy_config["detect_secrets"] is False
        assert orchestrator.privacy_config["secret_patterns"] == custom_patterns

    def test_configure_adds_default_secret_patterns(self):
        """configure_data_privacy should add default patterns when none provided."""
        orchestrator = MockOrchestrator()
        configure_data_privacy(orchestrator, secret_patterns=None)

        patterns = orchestrator.privacy_config["secret_patterns"]
        assert len(patterns) > 0
        assert any("sk-" in pattern for pattern in patterns)
        assert any("Bearer" in pattern for pattern in patterns)

    def test_configure_with_minimal_orchestrator(self):
        """configure_data_privacy should work with minimal orchestrator."""
        orchestrator = MinimalOrchestrator()
        configure_data_privacy(orchestrator)

        # Should add attribute via setattr
        assert hasattr(orchestrator, "privacy_config")
        assert orchestrator.privacy_config["anonymize_pii"] is True

    def test_configure_preserves_existing_config_attribute(self):
        """configure_data_privacy should preserve existing config dict."""
        orchestrator = MockOrchestrator()
        orchestrator.privacy_config = {"custom_key": "custom_value"}

        configure_data_privacy(orchestrator)

        # Should replace the entire config
        assert "custom_key" not in orchestrator.privacy_config
        assert "anonymize_pii" in orchestrator.privacy_config


# =============================================================================
# get_privacy_config() Tests
# =============================================================================


class TestGetPrivacyConfig:
    """Tests for get_privacy_config function."""

    def test_get_default_config_from_minimal_orchestrator(self):
        """get_privacy_config should return defaults for unconfigured orchestrator."""
        orchestrator = MinimalOrchestrator()
        config = get_privacy_config(orchestrator)

        assert config["anonymize_pii"] is True
        assert config["pii_columns"] == []
        assert config["hash_identifiers"] is True
        assert config["log_access"] is True
        assert config["detect_secrets"] is True
        assert isinstance(config["secret_patterns"], list)

    def test_get_existing_config(self):
        """get_privacy_config should return existing configuration."""
        orchestrator = MockOrchestrator()
        configure_data_privacy(orchestrator, anonymize_pii=False)

        config = get_privacy_config(orchestrator)
        assert config["anonymize_pii"] is False

    def test_get_config_returns_dict(self):
        """get_privacy_config should always return a dict."""
        orchestrator = MinimalOrchestrator()
        config = get_privacy_config(orchestrator)

        assert isinstance(config, dict)


# =============================================================================
# configure_secrets_masking() Tests
# =============================================================================


class TestConfigureSecretsMasking:
    """Tests for configure_secrets_masking function."""

    def test_configure_with_defaults(self):
        """configure_secrets_masking should apply default settings."""
        orchestrator = MockOrchestrator()
        configure_secrets_masking(orchestrator)

        assert orchestrator.secrets_masking_config["enabled"] is True
        assert orchestrator.secrets_masking_config["replacement"] == "[REDACTED]"
        assert orchestrator.secrets_masking_config["mask_in_arguments"] is True
        assert orchestrator.secrets_masking_config["mask_in_output"] is True
        assert orchestrator.secrets_masking_config["custom_patterns"] == []

    def test_configure_with_custom_settings(self):
        """configure_secrets_masking should accept custom settings."""
        orchestrator = MockOrchestrator()
        custom_patterns = [r"token_\w+", r"auth_\d+"]

        configure_secrets_masking(
            orchestrator,
            enabled=False,
            replacement="***HIDDEN***",
            mask_in_arguments=False,
            mask_in_output=False,
            custom_patterns=custom_patterns,
        )

        assert orchestrator.secrets_masking_config["enabled"] is False
        assert orchestrator.secrets_masking_config["replacement"] == "***HIDDEN***"
        assert orchestrator.secrets_masking_config["mask_in_arguments"] is False
        assert orchestrator.secrets_masking_config["mask_in_output"] is False
        assert orchestrator.secrets_masking_config["custom_patterns"] == custom_patterns

    def test_configure_with_minimal_orchestrator(self):
        """configure_secrets_masking should work with minimal orchestrator."""
        orchestrator = MinimalOrchestrator()
        configure_secrets_masking(orchestrator)

        assert hasattr(orchestrator, "secrets_masking_config")
        assert orchestrator.secrets_masking_config["enabled"] is True

    def test_custom_patterns_default_to_empty_list(self):
        """configure_secrets_masking should default custom_patterns to empty list."""
        orchestrator = MockOrchestrator()
        configure_secrets_masking(orchestrator)

        assert orchestrator.secrets_masking_config["custom_patterns"] == []


# =============================================================================
# get_secrets_masking_config() Tests
# =============================================================================


class TestGetSecretsMaskingConfig:
    """Tests for get_secrets_masking_config function."""

    def test_get_default_config_from_minimal_orchestrator(self):
        """get_secrets_masking_config should return defaults for unconfigured orchestrator."""
        orchestrator = MinimalOrchestrator()
        config = get_secrets_masking_config(orchestrator)

        assert config["enabled"] is True
        assert config["replacement"] == "[REDACTED]"
        assert config["mask_in_arguments"] is True
        assert config["mask_in_output"] is True
        assert config["custom_patterns"] == []

    def test_get_existing_config(self):
        """get_secrets_masking_config should return existing configuration."""
        orchestrator = MockOrchestrator()
        configure_secrets_masking(orchestrator, enabled=False)

        config = get_secrets_masking_config(orchestrator)
        assert config["enabled"] is False

    def test_get_config_returns_dict(self):
        """get_secrets_masking_config should always return a dict."""
        orchestrator = MinimalOrchestrator()
        config = get_secrets_masking_config(orchestrator)

        assert isinstance(config, dict)


# =============================================================================
# configure_audit_logging() Tests
# =============================================================================


class TestConfigureAuditLogging:
    """Tests for configure_audit_logging function."""

    def test_configure_with_defaults(self):
        """configure_audit_logging should apply default settings."""
        orchestrator = MockOrchestrator()
        configure_audit_logging(orchestrator)

        assert orchestrator.audit_logging_config["enabled"] is True
        assert orchestrator.audit_logging_config["log_data_access"] is True
        assert orchestrator.audit_logging_config["log_pii_access"] is True
        assert orchestrator.audit_logging_config["log_secrets_access"] is True
        assert orchestrator.audit_logging_config["log_file_path"] is None

    def test_configure_with_custom_settings(self):
        """configure_audit_logging should accept custom settings."""
        orchestrator = MockOrchestrator()
        log_path = "/var/log/victor/audit.log"

        configure_audit_logging(
            orchestrator,
            enabled=False,
            log_data_access=False,
            log_pii_access=False,
            log_secrets_access=False,
            log_file_path=log_path,
        )

        assert orchestrator.audit_logging_config["enabled"] is False
        assert orchestrator.audit_logging_config["log_data_access"] is False
        assert orchestrator.audit_logging_config["log_pii_access"] is False
        assert orchestrator.audit_logging_config["log_secrets_access"] is False
        assert orchestrator.audit_logging_config["log_file_path"] == log_path

    def test_configure_with_minimal_orchestrator(self):
        """configure_audit_logging should work with minimal orchestrator."""
        orchestrator = MinimalOrchestrator()
        configure_audit_logging(orchestrator)

        assert hasattr(orchestrator, "audit_logging_config")
        assert orchestrator.audit_logging_config["enabled"] is True

    def test_log_file_path_defaults_to_none(self):
        """configure_audit_logging should default log_file_path to None."""
        orchestrator = MockOrchestrator()
        configure_audit_logging(orchestrator)

        assert orchestrator.audit_logging_config["log_file_path"] is None


# =============================================================================
# get_audit_logging_config() Tests
# =============================================================================


class TestGetAuditLoggingConfig:
    """Tests for get_audit_logging_config function."""

    def test_get_default_config_from_minimal_orchestrator(self):
        """get_audit_logging_config should return defaults for unconfigured orchestrator."""
        orchestrator = MinimalOrchestrator()
        config = get_audit_logging_config(orchestrator)

        assert config["enabled"] is True
        assert config["log_data_access"] is True
        assert config["log_pii_access"] is True
        assert config["log_secrets_access"] is True
        assert config["log_file_path"] is None

    def test_get_existing_config(self):
        """get_audit_logging_config should return existing configuration."""
        orchestrator = MockOrchestrator()
        configure_audit_logging(orchestrator, enabled=False)

        config = get_audit_logging_config(orchestrator)
        assert config["enabled"] is False

    def test_get_config_returns_dict(self):
        """get_audit_logging_config should always return a dict."""
        orchestrator = MinimalOrchestrator()
        config = get_audit_logging_config(orchestrator)

        assert isinstance(config, dict)


# =============================================================================
# PrivacyCapabilityProvider Tests
# =============================================================================


class TestPrivacyCapabilityProvider:
    """Tests for PrivacyCapabilityProvider class."""

    @pytest.fixture
    def provider(self) -> PrivacyCapabilityProvider:
        """Create a PrivacyCapabilityProvider instance."""
        return PrivacyCapabilityProvider()

    def test_initialization(self, provider: PrivacyCapabilityProvider):
        """PrivacyCapabilityProvider should initialize correctly."""
        assert len(provider._capabilities) == 3
        assert "data_privacy" in provider._capabilities
        assert "secrets_masking" in provider._capabilities
        assert "audit_logging" in provider._capabilities

    def test_get_capabilities(self, provider: PrivacyCapabilityProvider):
        """get_capabilities should return all registered capabilities."""
        capabilities = provider.get_capabilities()

        assert isinstance(capabilities, dict)
        assert len(capabilities) == 3
        assert "data_privacy" in capabilities
        assert "secrets_masking" in capabilities
        assert "audit_logging" in capabilities

    def test_get_capability_metadata(self, provider: PrivacyCapabilityProvider):
        """get_capability_metadata should return metadata for all capabilities."""
        metadata = provider.get_capability_metadata()

        assert isinstance(metadata, dict)
        assert len(metadata) == 3
        assert "data_privacy" in metadata
        assert "secrets_masking" in metadata
        assert "audit_logging" in metadata

    def test_metadata_content(self, provider: PrivacyCapabilityProvider):
        """Capability metadata should have correct content."""
        metadata = provider.get_capability_metadata()

        # Check data_privacy metadata
        data_privacy_meta = metadata["data_privacy"]
        assert data_privacy_meta.name == "data_privacy"
        assert "privacy" in data_privacy_meta.tags
        assert "framework" in data_privacy_meta.tags

        # Check secrets_masking metadata
        secrets_meta = metadata["secrets_masking"]
        assert secrets_meta.name == "secrets_masking"
        assert "data_privacy" in secrets_meta.dependencies

        # Check audit_logging metadata
        audit_meta = metadata["audit_logging"]
        assert audit_meta.name == "audit_logging"
        assert "data_privacy" in audit_meta.dependencies

    def test_apply_data_privacy(self, provider: PrivacyCapabilityProvider):
        """apply_data_privacy should configure data privacy."""
        orchestrator = MockOrchestrator()
        provider.apply_data_privacy(orchestrator, pii_columns=["email"])

        assert orchestrator.privacy_config["pii_columns"] == ["email"]
        assert "data_privacy" in provider.get_applied()

    def test_apply_secrets_masking(self, provider: PrivacyCapabilityProvider):
        """apply_secrets_masking should configure secrets masking."""
        orchestrator = MockOrchestrator()
        provider.apply_secrets_masking(orchestrator, replacement="***")

        assert orchestrator.secrets_masking_config["replacement"] == "***"
        assert "secrets_masking" in provider.get_applied()

    def test_apply_audit_logging(self, provider: PrivacyCapabilityProvider):
        """apply_audit_logging should configure audit logging."""
        orchestrator = MockOrchestrator()
        provider.apply_audit_logging(orchestrator, log_file_path="/tmp/audit.log")

        assert orchestrator.audit_logging_config["log_file_path"] == "/tmp/audit.log"
        assert "audit_logging" in provider.get_applied()

    def test_apply_all(self, provider: PrivacyCapabilityProvider):
        """apply_all should apply all capabilities."""
        orchestrator = MockOrchestrator()
        provider.apply_all(orchestrator)

        # All three configs should be set
        assert hasattr(orchestrator, "privacy_config")
        assert hasattr(orchestrator, "secrets_masking_config")
        assert hasattr(orchestrator, "audit_logging_config")

        # All three should be marked as applied
        applied = provider.get_applied()
        assert "data_privacy" in applied
        assert "secrets_masking" in applied
        assert "audit_logging" in applied

    def test_get_applied(self, provider: PrivacyCapabilityProvider):
        """get_applied should return set of applied capabilities."""
        orchestrator = MockOrchestrator()

        # Initially empty
        assert len(provider.get_applied()) == 0

        # Apply one capability
        provider.apply_data_privacy(orchestrator)
        applied = provider.get_applied()
        assert "data_privacy" in applied
        assert len(applied) == 1

        # Apply another
        provider.apply_secrets_masking(orchestrator)
        applied = provider.get_applied()
        assert "secrets_masking" in applied
        assert len(applied) == 2

    def test_get_applied_returns_copy(self, provider: PrivacyCapabilityProvider):
        """get_applied should return a copy, not the internal set."""
        orchestrator = MockOrchestrator()
        provider.apply_data_privacy(orchestrator)

        applied1 = provider.get_applied()
        applied2 = provider.get_applied()

        # Modifying returned set should not affect internal state
        applied1.add("fake_capability")
        assert "fake_capability" not in applied2
        assert "fake_capability" not in provider.get_applied()

    def test_capability_callables(self, provider: PrivacyCapabilityProvider):
        """Capabilities should be callable functions."""
        capabilities = provider.get_capabilities()

        for cap_name, cap_func in capabilities.items():
            assert callable(cap_func), f"{cap_name} is not callable"

    def test_metadata_keys_match_capability_keys(self, provider: PrivacyCapabilityProvider):
        """Metadata keys should match capability keys."""
        capabilities = provider.get_capabilities()
        metadata = provider.get_capability_metadata()

        assert set(capabilities.keys()) == set(metadata.keys())


# =============================================================================
# CAPABILITIES List Tests
# =============================================================================


class TestCapabilitiesList:
    """Tests for CAPABILITIES list and get_framework_privacy_capabilities()."""

    def test_capabilities_list_length(self):
        """CAPABILITIES list should have 3 entries."""
        assert len(CAPABILITIES) == 3

    def test_capabilities_list_content(self):
        """CAPABILITIES list should have correct capability entries."""
        capability_names = [entry.capability.name for entry in CAPABILITIES]

        assert "framework_privacy" in capability_names
        assert "framework_secrets_masking" in capability_names
        assert "framework_audit_logging" in capability_names

    def test_get_framework_privacy_capabilities(self):
        """get_framework_privacy_capabilities should return copy of CAPABILITIES."""
        capabilities = get_framework_privacy_capabilities()

        assert len(capabilities) == 3
        assert capabilities is not CAPABILITIES  # Should be a copy

    def test_capability_entries_have_handlers(self):
        """Each capability entry should have a handler function."""
        for entry in CAPABILITIES:
            assert entry.handler is not None
            assert callable(entry.handler)

    def test_capability_entries_have_getters(self):
        """Each capability entry should have a getter function where applicable."""
        for entry in CAPABILITIES:
            # Some capabilities might not have getters
            if entry.getter_handler:
                assert callable(entry.getter_handler)


# =============================================================================
# Integration Tests
# =============================================================================


class TestPrivacyIntegration:
    """Integration tests for privacy capabilities."""

    def test_full_privacy_configuration_workflow(self):
        """Test complete privacy configuration workflow."""
        provider = PrivacyCapabilityProvider()
        orchestrator = MockOrchestrator()

        # Apply all capabilities
        provider.apply_all(orchestrator)

        # Verify all configurations are set
        assert orchestrator.privacy_config["anonymize_pii"] is True
        assert orchestrator.secrets_masking_config["enabled"] is True
        assert orchestrator.audit_logging_config["enabled"] is True

        # Verify getters return the configured values
        privacy_config = get_privacy_config(orchestrator)
        assert privacy_config["anonymize_pii"] is True

        secrets_config = get_secrets_masking_config(orchestrator)
        assert secrets_config["enabled"] is True

        audit_config = get_audit_logging_config(orchestrator)
        assert audit_config["enabled"] is True

    def test_capability_provider_with_minimal_orchestrator(self):
        """PrivacyCapabilityProvider should work with minimal orchestrator."""
        provider = PrivacyCapabilityProvider()
        orchestrator = MinimalOrchestrator()

        # Should not raise errors
        provider.apply_data_privacy(orchestrator)
        provider.apply_secrets_masking(orchestrator)
        provider.apply_audit_logging(orchestrator)

        # Configs should be added as attributes
        assert hasattr(orchestrator, "privacy_config")
        assert hasattr(orchestrator, "secrets_masking_config")
        assert hasattr(orchestrator, "audit_logging_config")

    def test_multiple_configurations_override_previous(self):
        """Later configurations should override earlier ones."""
        orchestrator = MockOrchestrator()

        # First configuration
        configure_data_privacy(orchestrator, anonymize_pii=True)
        assert orchestrator.privacy_config["anonymize_pii"] is True

        # Override configuration
        configure_data_privacy(orchestrator, anonymize_pii=False)
        assert orchestrator.privacy_config["anonymize_pii"] is False

    def test_capability_provider_base_provider_interface(self):
        """PrivacyCapabilityProvider should implement BaseCapabilityProvider interface."""
        from victor.framework.capabilities import BaseCapabilityProvider

        provider = PrivacyCapabilityProvider()

        # Should be instance of BaseCapabilityProvider
        assert isinstance(provider, BaseCapabilityProvider)

        # Should implement required methods
        assert hasattr(provider, "get_capabilities")
        assert hasattr(provider, "get_capability_metadata")
        assert hasattr(provider, "get_capability")
        assert hasattr(provider, "has_capability")
        assert hasattr(provider, "list_capabilities")

    def test_list_capabilities(self):
        """list_capabilities should return all capability names."""
        provider = PrivacyCapabilityProvider()
        capabilities = provider.list_capabilities()

        assert isinstance(capabilities, list)
        assert "data_privacy" in capabilities
        assert "secrets_masking" in capabilities
        assert "audit_logging" in capabilities

    def test_has_capability(self):
        """has_capability should check for capability existence."""
        provider = PrivacyCapabilityProvider()

        assert provider.has_capability("data_privacy") is True
        assert provider.has_capability("secrets_masking") is True
        assert provider.has_capability("audit_logging") is True
        assert provider.has_capability("nonexistent") is False

    def test_get_capability(self):
        """get_capability should return capability function or None."""
        provider = PrivacyCapabilityProvider()

        data_privacy_cap = provider.get_capability("data_privacy")
        assert data_privacy_cap is not None
        assert callable(data_privacy_cap)

        nonexistent = provider.get_capability("nonexistent")
        assert nonexistent is None


__all__ = [
    # Test classes
    "TestConfigureDataPrivacy",
    "TestGetPrivacyConfig",
    "TestConfigureSecretsMasking",
    "TestGetSecretsMaskingConfig",
    "TestConfigureAuditLogging",
    "TestGetAuditLoggingConfig",
    "TestPrivacyCapabilityProvider",
    "TestCapabilitiesList",
    "TestPrivacyIntegration",
]
