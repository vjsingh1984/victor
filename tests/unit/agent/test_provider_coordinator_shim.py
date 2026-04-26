from pathlib import Path

from victor.agent.provider.coordinator import (
    ProviderCoordinator as CanonicalProviderCoordinator,
)
from victor.agent.provider.coordinator import (
    ProviderCoordinatorConfig as CanonicalProviderCoordinatorConfig,
)
from victor.agent.provider.coordinator import RateLimitInfo as CanonicalRateLimitInfo
from victor.agent.provider_coordinator import (
    ProviderCoordinator as LegacyProviderCoordinator,
)
from victor.agent.provider_coordinator import (
    ProviderCoordinatorConfig as LegacyProviderCoordinatorConfig,
)
from victor.agent.provider_coordinator import RateLimitInfo as LegacyRateLimitInfo


def test_legacy_provider_coordinator_reexports_canonical_types():
    assert LegacyProviderCoordinator is CanonicalProviderCoordinator
    assert LegacyProviderCoordinatorConfig is CanonicalProviderCoordinatorConfig
    assert LegacyRateLimitInfo is CanonicalRateLimitInfo


def test_internal_code_uses_canonical_provider_coordinator_module():
    runtime_builders_source = Path("victor/agent/factory/runtime_builders.py").read_text()
    provider_runtime_source = Path("victor/agent/runtime/provider_runtime.py").read_text()
    provider_package_source = Path("victor/agent/provider/__init__.py").read_text()
    provider_coordinator_source = Path("victor/agent/provider/coordinator.py").read_text()
    orchestrator_source = Path("victor/agent/orchestrator.py").read_text()

    assert "from victor.agent.provider_coordinator import create_provider_coordinator" not in runtime_builders_source
    assert "create_deprecated_provider_coordinator" not in runtime_builders_source
    assert "create_provider_switch_coordinator" not in runtime_builders_source
    assert "from victor.agent.provider_coordinator import ProviderCoordinatorConfig" not in provider_runtime_source
    assert "from victor.agent.provider.coordinator import (" in provider_runtime_source
    assert "ProviderCoordinatorConfig," in provider_runtime_source
    assert "create_provider_coordinator," in provider_runtime_source
    assert "from victor.agent.provider.switch_coordinator import create_provider_switch_coordinator" in provider_runtime_source
    assert "from victor.agent.provider_coordinator import ProviderCoordinator" not in provider_package_source
    assert "from victor.agent.provider.coordinator import (" not in provider_package_source
    assert "from victor.agent.provider_coordinator import ProviderCoordinator" not in provider_coordinator_source
    assert "from victor.agent.provider.coordinator import ProviderCoordinator" in provider_coordinator_source
    assert "from victor.agent.provider_coordinator import ProviderCoordinator" not in orchestrator_source
    assert "from victor.agent.provider.coordinator import ProviderCoordinator" in orchestrator_source
