"""Tests for vertical metadata propagation in integration telemetry."""

from __future__ import annotations

from typing import List

from victor.core.verticals.base import VerticalBase, VerticalRegistry
from victor.framework.vertical_integration import IntegrationResult, VerticalIntegrationPipeline
from victor_sdk.verticals.manifest import ExtensionManifest, ExtensionType


def test_vertical_applied_payload_includes_manifest_metadata() -> None:
    """Observability payloads should carry manifest version and namespace."""

    class _TelemetryVertical(VerticalBase):
        name = "telemetry_vertical"
        description = "Telemetry test vertical"
        version = "9.8.7"
        _victor_manifest = ExtensionManifest(
            name="telemetry_vertical",
            version="9.8.7",
            api_version=1,
            provides={ExtensionType.TOOLS},
            plugin_namespace="analytics",
        )

        @classmethod
        def get_tools(cls) -> List[str]:
            return ["read"]

        @classmethod
        def get_system_prompt(cls) -> str:
            return "telemetry prompt"

    VerticalRegistry.unregister(_TelemetryVertical.name)
    VerticalRegistry.register(_TelemetryVertical)

    pipeline = VerticalIntegrationPipeline()
    result = IntegrationResult(vertical_name="telemetry_vertical")

    payload = pipeline._build_vertical_applied_payload(object(), result, cache_hit=False)

    assert payload["vertical"] == "telemetry_vertical"
    assert payload["vertical_manifest_version"] == "9.8.7"
    assert payload["vertical_plugin_namespace"] == "analytics"

    VerticalRegistry.unregister(_TelemetryVertical.name)
