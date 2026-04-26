from pathlib import Path

from victor.agent.provider.switch_contracts import (
    HookPriority,
    PostSwitchHook,
    SwitchContext,
)
from victor.agent.provider.switch_coordinator import (
    ProviderSwitchCoordinator as CanonicalProviderSwitchCoordinator,
)
from victor.agent.provider_switch_coordinator import (
    ProviderSwitchCoordinator as LegacyProviderSwitchCoordinator,
)
from victor.agent.provider_switch_coordinator import (
    HookPriority as LegacyHookPriority,
)
from victor.agent.provider_switch_coordinator import (
    PostSwitchHook as LegacyPostSwitchHook,
)
from victor.agent.provider_switch_coordinator import (
    SwitchContext as LegacySwitchContext,
)


def test_legacy_provider_switch_coordinator_reexports_canonical_contracts():
    assert LegacyProviderSwitchCoordinator is CanonicalProviderSwitchCoordinator
    assert LegacyHookPriority is HookPriority
    assert LegacySwitchContext is SwitchContext
    assert LegacyPostSwitchHook is PostSwitchHook


def test_post_switch_hooks_use_canonical_switch_contract_imports():
    source = Path("victor/agent/post_switch_hooks.py").read_text()

    assert (
        "from victor.agent.provider.switch_contracts import HookPriority, SwitchContext" in source
    )
    assert "from victor.agent.provider_switch_coordinator import HookPriority" not in source
    assert "from victor.agent.provider_switch_coordinator import SwitchContext" not in source


def test_factory_code_uses_provider_package_instead_of_root_switch_module():
    runtime_builders_source = Path("victor/agent/factory/runtime_builders.py").read_text()
    orchestrator_factory_source = Path("victor/agent/orchestrator_factory.py").read_text()

    assert (
        "from victor.agent.provider_switch_coordinator import ProviderSwitchCoordinator"
        not in runtime_builders_source
    )
    assert (
        "from victor.agent.provider_switch_coordinator import ProviderSwitchCoordinator"
        not in orchestrator_factory_source
    )
    assert "from victor.agent.provider import ProviderSwitchCoordinator" in runtime_builders_source
    assert (
        "from victor.agent.provider import ProviderSwitchCoordinator" in orchestrator_factory_source
    )


def test_provider_package_uses_canonical_switch_coordinator_module():
    provider_package_source = Path("victor/agent/provider/__init__.py").read_text()

    assert "from victor.agent.provider.switch_coordinator import ProviderSwitchCoordinator" in provider_package_source
    assert "from victor.agent.provider_switch_coordinator import ProviderSwitchCoordinator" not in provider_package_source
