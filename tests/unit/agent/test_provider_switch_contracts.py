from pathlib import Path

from victor.agent.provider.switch_contracts import (
    HookPriority,
    PostSwitchHook,
    SwitchContext,
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
    assert LegacyHookPriority is HookPriority
    assert LegacySwitchContext is SwitchContext
    assert LegacyPostSwitchHook is PostSwitchHook


def test_post_switch_hooks_use_canonical_switch_contract_imports():
    source = Path("victor/agent/post_switch_hooks.py").read_text()

    assert "from victor.agent.provider.switch_contracts import HookPriority, SwitchContext" in source
    assert "from victor.agent.provider_switch_coordinator import HookPriority" not in source
    assert "from victor.agent.provider_switch_coordinator import SwitchContext" not in source
