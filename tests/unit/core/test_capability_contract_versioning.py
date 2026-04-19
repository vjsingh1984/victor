"""TDD tests for per-capability contract versioning."""

import pytest

from victor_sdk.core.capability_contract import (
    CapabilityContract,
    CapabilityContractRegistry,
)


class TestCapabilityContract:

    def test_creation(self):
        c = CapabilityContract("tools", version=2, min_sdk_version=">=0.7.0")
        assert c.name == "tools"
        assert c.version == 2
        assert c.min_sdk_version == ">=0.7.0"

    def test_sdk_compatible_within_range(self):
        c = CapabilityContract("tools", version=2, min_sdk_version=">=0.7.0")
        assert c.is_sdk_compatible("0.7.0") is True
        assert c.is_sdk_compatible("0.8.0") is True
        assert c.is_sdk_compatible("1.0.0") is True

    def test_sdk_incompatible_below_range(self):
        c = CapabilityContract("tools", version=2, min_sdk_version=">=0.7.0")
        assert c.is_sdk_compatible("0.6.0") is False

    def test_sdk_compatible_no_constraint(self):
        c = CapabilityContract("tools", version=2)
        assert c.is_sdk_compatible("0.1.0") is True

    def test_frozen_dataclass(self):
        c = CapabilityContract("tools", version=2)
        with pytest.raises(AttributeError):
            c.name = "other"


class TestCapabilityContractRegistry:

    def test_register_and_lookup(self):
        reg = CapabilityContractRegistry()
        reg.register(CapabilityContract("tools", version=2))
        assert reg.get("tools").version == 2

    def test_lookup_missing(self):
        reg = CapabilityContractRegistry()
        assert reg.get("nonexistent") is None

    def test_check_all_compatible(self):
        reg = CapabilityContractRegistry()
        reg.register(CapabilityContract("tools", version=2, min_sdk_version=">=0.7.0"))
        reg.register(CapabilityContract("safety", version=1, min_sdk_version=">=0.5.0"))
        results = reg.check_all("0.7.0")
        assert all(r.compatible for r in results)

    def test_check_all_with_incompatible(self):
        reg = CapabilityContractRegistry()
        reg.register(CapabilityContract("tools", version=2, min_sdk_version=">=0.7.0"))
        results = reg.check_all("0.6.0")
        assert not results[0].compatible
        assert "tools" in results[0].message
