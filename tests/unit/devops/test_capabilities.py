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

"""Unit tests for DevOps vertical capabilities."""

from __future__ import annotations

from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pytest


class MockVerticalContext:
    """Mock vertical context for testing capabilities."""

    def __init__(self):
        self._configs: Dict[str, Any] = {}

    def set_capability_config(self, name: str, config: Dict[str, Any]) -> None:
        """Store capability configuration."""
        self._configs[name] = config

    def get_capability_config(
        self, name: str, default: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Retrieve capability configuration."""
        return self._configs.get(name, default if default is not None else {})

    def has_capability_config(self, name: str) -> bool:
        """Check if capability configuration exists."""
        return name in self._configs


class MockOrchestrator:
    """Mock orchestrator for testing capabilities."""

    def __init__(self):
        self.vertical_context = MockVerticalContext()


class TestDeploymentSafetyCapability:
    """Tests for deployment safety capability."""

    @pytest.fixture
    def orchestrator(self):
        return MockOrchestrator()

    def test_default_configuration(self, orchestrator):
        """Test deployment safety with default settings."""
        from victor.devops.capabilities import configure_deployment_safety

        configure_deployment_safety(orchestrator)

        config = orchestrator.vertical_context.get_capability_config("deployment_safety")
        assert config["require_approval_for_production"] is True
        assert config["require_backup_before_deploy"] is True
        assert config["enable_rollback"] is True
        assert "production" in config["protected_environments"]
        assert "staging" in config["protected_environments"]

    def test_custom_configuration(self, orchestrator):
        """Test deployment safety with custom settings."""
        from victor.devops.capabilities import configure_deployment_safety

        configure_deployment_safety(
            orchestrator,
            require_approval_for_production=False,
            require_backup_before_deploy=False,
            enable_rollback=False,
            protected_environments=["prod", "uat"],
        )

        config = orchestrator.vertical_context.get_capability_config("deployment_safety")
        assert config["require_approval_for_production"] is False
        assert config["require_backup_before_deploy"] is False
        assert config["enable_rollback"] is False
        assert config["protected_environments"] == ["prod", "uat"]


class TestContainerSettingsCapability:
    """Tests for container settings capability."""

    @pytest.fixture
    def orchestrator(self):
        return MockOrchestrator()

    def test_default_configuration(self, orchestrator):
        """Test container settings with default settings."""
        from victor.devops.capabilities import configure_container_settings

        configure_container_settings(orchestrator)

        config = orchestrator.vertical_context.get_capability_config("container_settings")
        assert config["runtime"] == "docker"
        assert config["security_scan_enabled"] is True
        assert config["max_image_size_mb"] == 2000

    def test_podman_runtime(self, orchestrator):
        """Test container settings with podman runtime."""
        from victor.devops.capabilities import configure_container_settings

        configure_container_settings(
            orchestrator,
            runtime="podman",
            default_registry="registry.example.com",
        )

        config = orchestrator.vertical_context.get_capability_config("container_settings")
        assert config["runtime"] == "podman"
        assert config["default_registry"] == "registry.example.com"

    def test_get_container_settings(self):
        """Test getter for container settings."""
        from victor.devops.capabilities import (
            configure_container_settings,
            get_container_settings,
        )

        # Test with orchestrator that has no config set (returns defaults)
        no_config = MockOrchestrator()
        settings = get_container_settings(no_config)
        assert settings["runtime"] == "docker"  # Should return defaults

        # After configuration with our mock orchestrator
        orchestrator = MockOrchestrator()
        configure_container_settings(orchestrator, runtime="podman")
        settings = get_container_settings(orchestrator)
        assert settings["runtime"] == "podman"


class TestInfrastructureSettingsCapability:
    """Tests for infrastructure settings capability."""

    @pytest.fixture
    def orchestrator(self):
        return MockOrchestrator()

    def test_default_configuration(self, orchestrator):
        """Test infrastructure settings with defaults."""
        from victor.devops.capabilities import configure_infrastructure_settings

        configure_infrastructure_settings(orchestrator)

        config = orchestrator.vertical_context.get_capability_config("infrastructure_settings")
        assert config["iac_tool"] == "terraform"
        assert config["require_plan_before_apply"] is True
        assert config["auto_approve_non_destructive"] is False

    def test_opentofu_configuration(self, orchestrator):
        """Test infrastructure settings with OpenTofu."""
        from victor.devops.capabilities import configure_infrastructure_settings

        configure_infrastructure_settings(
            orchestrator,
            iac_tool="opentofu",
            auto_approve_non_destructive=True,
            state_backend="s3",
        )

        config = orchestrator.vertical_context.get_capability_config("infrastructure_settings")
        assert config["iac_tool"] == "opentofu"
        assert config["auto_approve_non_destructive"] is True
        assert config["state_backend"] == "s3"


class TestCICDSettingsCapability:
    """Tests for CI/CD settings capability."""

    @pytest.fixture
    def orchestrator(self):
        return MockOrchestrator()

    def test_default_configuration(self, orchestrator):
        """Test CI/CD settings with defaults."""
        from victor.devops.capabilities import configure_cicd_settings

        configure_cicd_settings(orchestrator)

        config = orchestrator.vertical_context.get_capability_config("cicd_settings")
        assert config["platform"] == "github_actions"
        assert config["run_tests_before_deploy"] is True
        assert config["require_passing_checks"] is True
        assert config["enable_security_scan"] is True

    def test_gitlab_configuration(self, orchestrator):
        """Test CI/CD settings with GitLab CI."""
        from victor.devops.capabilities import configure_cicd_settings

        configure_cicd_settings(
            orchestrator,
            platform="gitlab_ci",
            run_tests_before_deploy=False,
        )

        config = orchestrator.vertical_context.get_capability_config("cicd_settings")
        assert config["platform"] == "gitlab_ci"
        assert config["run_tests_before_deploy"] is False


class TestMonitoringSettingsCapability:
    """Tests for monitoring settings capability."""

    @pytest.fixture
    def orchestrator(self):
        return MockOrchestrator()

    def test_default_configuration(self, orchestrator):
        """Test monitoring settings with defaults."""
        from victor.devops.capabilities import configure_monitoring_settings

        configure_monitoring_settings(orchestrator)

        config = orchestrator.vertical_context.get_capability_config("monitoring_settings")
        assert config["metrics_backend"] == "prometheus"
        assert config["logging_backend"] == "loki"
        assert config["alerting_enabled"] is True
        assert config["dashboard_tool"] == "grafana"

    def test_custom_configuration(self, orchestrator):
        """Test monitoring settings with custom backends."""
        from victor.devops.capabilities import configure_monitoring_settings

        configure_monitoring_settings(
            orchestrator,
            metrics_backend="datadog",
            logging_backend="elasticsearch",
            dashboard_tool="kibana",
        )

        config = orchestrator.vertical_context.get_capability_config("monitoring_settings")
        assert config["metrics_backend"] == "datadog"
        assert config["logging_backend"] == "elasticsearch"
        assert config["dashboard_tool"] == "kibana"


class TestDevOpsCapabilityProvider:
    """Tests for DevOpsCapabilityProvider class."""

    @pytest.fixture
    def provider(self):
        from victor.devops.capabilities import DevOpsCapabilityProvider

        return DevOpsCapabilityProvider()

    @pytest.fixture
    def orchestrator(self):
        return MockOrchestrator()

    def test_list_capabilities(self, provider):
        """Test listing available capabilities."""
        capabilities = provider.list_capabilities()

        assert "deployment_safety" in capabilities
        assert "container_settings" in capabilities
        assert "infrastructure_settings" in capabilities
        assert "cicd_settings" in capabilities
        assert "monitoring_settings" in capabilities
        assert len(capabilities) == 5

    def test_get_capabilities(self, provider):
        """Test getting all capabilities."""
        capabilities = provider.get_capabilities()

        assert len(capabilities) == 5
        assert callable(capabilities["deployment_safety"])
        assert callable(capabilities["container_settings"])

    def test_get_capability_metadata(self, provider):
        """Test getting capability metadata."""
        metadata = provider.get_capability_metadata()

        assert "deployment_safety" in metadata
        assert metadata["deployment_safety"].name == "deployment_safety"
        assert "safety" in metadata["deployment_safety"].tags
        assert metadata["deployment_safety"].version == "1.0"

    def test_has_capability(self, provider):
        """Test checking for capability existence."""
        assert provider.has_capability("deployment_safety") is True
        assert provider.has_capability("container_settings") is True
        assert provider.has_capability("nonexistent") is False

    def test_get_capability(self, provider):
        """Test getting a specific capability."""
        cap = provider.get_capability("deployment_safety")
        assert cap is not None
        assert callable(cap)

        cap = provider.get_capability("nonexistent")
        assert cap is None

    def test_apply_deployment_safety(self, provider, orchestrator):
        """Test applying deployment safety capability."""
        provider.apply_deployment_safety(orchestrator)

        assert "deployment_safety" in provider.get_applied()
        assert orchestrator.vertical_context.has_capability_config("deployment_safety")

    def test_apply_container_settings(self, provider, orchestrator):
        """Test applying container settings capability."""
        provider.apply_container_settings(orchestrator, runtime="podman")

        assert "container_settings" in provider.get_applied()
        config = orchestrator.vertical_context.get_capability_config("container_settings")
        assert config["runtime"] == "podman"

    def test_apply_all(self, provider, orchestrator):
        """Test applying all capabilities."""
        provider.apply_all(orchestrator)

        applied = provider.get_applied()
        assert len(applied) == 5
        assert "deployment_safety" in applied
        assert "container_settings" in applied
        assert "infrastructure_settings" in applied
        assert "cicd_settings" in applied
        assert "monitoring_settings" in applied


class TestCapabilitiesExports:
    """Tests for module exports."""

    def test_capabilities_list_exists(self):
        """Test CAPABILITIES list is defined."""
        from victor.devops.capabilities import CAPABILITIES

        assert len(CAPABILITIES) == 5
        names = [c.capability.name for c in CAPABILITIES]
        assert "devops_deployment_safety" in names
        assert "devops_container" in names
        assert "devops_infrastructure" in names
        assert "devops_cicd" in names
        assert "devops_monitoring" in names

    def test_get_devops_capabilities(self):
        """Test get_devops_capabilities helper."""
        from victor.devops.capabilities import get_devops_capabilities

        capabilities = get_devops_capabilities()
        assert len(capabilities) == 5

    def test_module_all_exports(self):
        """Test module __all__ exports."""
        from victor.devops import capabilities

        assert "DevOpsCapabilityProvider" in capabilities.__all__
        assert "configure_deployment_safety" in capabilities.__all__
        assert "configure_container_settings" in capabilities.__all__
        assert "CAPABILITIES" in capabilities.__all__


class TestDevOpsAssistantCapabilityIntegration:
    """Tests for capability integration with DevOpsAssistant."""

    def test_get_capability_provider(self):
        """Test DevOpsAssistant.get_capability_provider method."""
        from victor.devops.assistant import DevOpsAssistant
        from victor.devops.capabilities import DevOpsCapabilityProvider

        provider = DevOpsAssistant.get_capability_provider()

        assert provider is not None
        assert isinstance(provider, DevOpsCapabilityProvider)
        assert len(provider.list_capabilities()) == 5
