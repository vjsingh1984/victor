from importlib import import_module, reload
from pathlib import Path

import pytest

from victor.agent.provider.switch_contracts import (
    HookPriority,
    PostSwitchHook,
    SwitchContext,
)
from victor.agent.provider.switch_coordinator import (
    ProviderSwitchCoordinator as CanonicalProviderSwitchCoordinator,
)


def _reload_legacy_provider_switch_module():
    module = import_module("victor.agent.provider_switch_coordinator")
    return reload(module)


def test_legacy_provider_switch_coordinator_reexports_canonical_contracts_with_warning():
    with pytest.warns(
        DeprecationWarning,
        match="victor\\.agent\\.provider_switch_coordinator is deprecated",
    ):
        legacy_module = _reload_legacy_provider_switch_module()

    assert legacy_module.ProviderSwitchCoordinator is CanonicalProviderSwitchCoordinator
    assert legacy_module.HookPriority is HookPriority
    assert legacy_module.SwitchContext is SwitchContext
    assert legacy_module.PostSwitchHook is PostSwitchHook


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
    assert "from victor.agent.provider import ProviderSwitchCoordinator" in orchestrator_factory_source


def test_provider_package_uses_canonical_switch_coordinator_module():
    provider_package_source = Path("victor/agent/provider/__init__.py").read_text()

    assert (
        "from victor.agent.provider.switch_coordinator import ProviderSwitchCoordinator"
        in provider_package_source
    )
    assert (
        "from victor.agent.provider_switch_coordinator import ProviderSwitchCoordinator"
        not in provider_package_source
    )
