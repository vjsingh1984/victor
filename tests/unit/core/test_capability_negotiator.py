"""Tests for CapabilityNegotiator."""

from victor.core.verticals.capability_negotiator import (
    CapabilityNegotiator,
    FRAMEWORK_CAPABILITIES,
    NegotiationResult,
)
from victor_sdk.verticals.manifest import ExtensionManifest, ExtensionType


class TestNegotiationResult:
    def test_defaults(self):
        r = NegotiationResult()
        assert r.compatible is True
        assert r.warnings == []
        assert r.errors == []
        assert r.degraded_features == set()


class TestCapabilityNegotiator:
    def test_compatible_manifest(self):
        manifest = ExtensionManifest(
            api_version=2,
            name="coding",
            version="1.0.0",
            provides={ExtensionType.SAFETY, ExtensionType.TOOLS},
        )
        negotiator = CapabilityNegotiator()
        result = negotiator.negotiate(manifest)
        assert result.compatible is True
        assert result.errors == []

    def test_api_version_too_low(self):
        manifest = ExtensionManifest(api_version=0, name="old")
        negotiator = CapabilityNegotiator()
        result = negotiator.negotiate(manifest)
        assert result.compatible is False
        assert any("below minimum" in e for e in result.errors)

    def test_api_version_too_high(self):
        manifest = ExtensionManifest(api_version=99, name="future")
        negotiator = CapabilityNegotiator()
        result = negotiator.negotiate(manifest)
        assert result.compatible is False
        assert any("exceeds" in e for e in result.errors)

    def test_unmet_requirements(self):
        manifest = ExtensionManifest(
            api_version=1,
            name="needy",
            requires={ExtensionType.API_ROUTER},
        )
        # API_ROUTER is not in FRAMEWORK_CAPABILITIES
        negotiator = CapabilityNegotiator(
            framework_capabilities={ExtensionType.SAFETY, ExtensionType.TOOLS}
        )
        result = negotiator.negotiate(manifest)
        assert result.compatible is False
        assert any("Unmet" in e for e in result.errors)

    def test_requirements_met(self):
        manifest = ExtensionManifest(
            api_version=1,
            name="ok",
            requires={ExtensionType.SAFETY},
        )
        negotiator = CapabilityNegotiator()
        result = negotiator.negotiate(manifest)
        assert result.compatible is True

    def test_unknown_provided_types_warn(self):
        manifest = ExtensionManifest(
            api_version=1,
            name="exotic",
            provides={ExtensionType.API_ROUTER},
        )
        negotiator = CapabilityNegotiator(
            framework_capabilities={ExtensionType.SAFETY}
        )
        result = negotiator.negotiate(manifest)
        assert result.compatible is True
        assert len(result.warnings) == 1
        assert ExtensionType.API_ROUTER in result.degraded_features

    def test_all_builtin_verticals_compatible(self):
        """All built-in verticals should produce compatible manifests."""
        from victor_sdk.verticals.protocols.base import VerticalBase

        class FakeBuiltin(VerticalBase):
            @classmethod
            def get_name(cls):
                return "fake-builtin"

            @classmethod
            def get_description(cls):
                return "test"

            @classmethod
            def get_tools(cls):
                return ["read"]

            @classmethod
            def get_system_prompt(cls):
                return "prompt"

        manifest = FakeBuiltin.get_manifest()
        negotiator = CapabilityNegotiator()
        result = negotiator.negotiate(manifest)
        assert result.compatible is True

    def test_framework_capabilities_include_all_standard_types(self):
        expected = {
            ExtensionType.SAFETY,
            ExtensionType.TOOLS,
            ExtensionType.WORKFLOWS,
            ExtensionType.TEAMS,
            ExtensionType.MIDDLEWARE,
            ExtensionType.MODE_CONFIG,
            ExtensionType.RL_CONFIG,
            ExtensionType.ENRICHMENT,
            ExtensionType.CAPABILITIES,
            ExtensionType.SERVICE_PROVIDER,
        }
        assert FRAMEWORK_CAPABILITIES == expected
