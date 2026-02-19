# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit tests for framework privacy capability config storage behavior."""

from victor.framework.capability_config_service import CapabilityConfigService
from victor.framework.capabilities.privacy import (
    configure_data_privacy,
    configure_secrets_masking,
    configure_audit_logging,
    get_privacy_config,
)


class _StubContainer:
    def __init__(self, service: CapabilityConfigService | None = None) -> None:
        self._service = service

    def get_optional(self, service_type):
        if self._service is None:
            return None
        if isinstance(self._service, service_type):
            return self._service
        return None


class _ServiceBackedOrchestrator:
    def __init__(self, service: CapabilityConfigService) -> None:
        self._container = _StubContainer(service)

    def get_service_container(self):
        return self._container


class _LegacyOrchestrator:
    pass


class TestPrivacyCapabilityConfigStorage:
    """Validate framework privacy capability config storage migration path."""

    def test_privacy_store_and_read_from_framework_service(self):
        service = CapabilityConfigService()
        orchestrator = _ServiceBackedOrchestrator(service)

        configure_data_privacy(orchestrator, anonymize_pii=False, pii_columns=["email"])

        assert get_privacy_config(orchestrator)["anonymize_pii"] is False
        assert service.get_config("privacy_config")["pii_columns"] == ["email"]

    def test_secrets_and_audit_store_in_framework_service(self):
        service = CapabilityConfigService()
        orchestrator = _ServiceBackedOrchestrator(service)

        configure_secrets_masking(orchestrator, replacement="***")
        configure_audit_logging(orchestrator, enabled=False, log_file_path="/tmp/audit.log")

        assert service.get_config("secrets_masking_config")["replacement"] == "***"
        assert service.get_config("audit_logging_config") == {
            "enabled": False,
            "log_data_access": True,
            "log_pii_access": True,
            "log_secrets_access": True,
            "log_file_path": "/tmp/audit.log",
        }

    def test_legacy_fallback_preserves_attribute_behavior(self):
        orchestrator = _LegacyOrchestrator()

        configure_data_privacy(orchestrator, anonymize_pii=True)
        configure_secrets_masking(orchestrator, enabled=True)
        configure_audit_logging(orchestrator, enabled=True)

        assert hasattr(orchestrator, "privacy_config")
        assert hasattr(orchestrator, "secrets_masking_config")
        assert hasattr(orchestrator, "audit_logging_config")
