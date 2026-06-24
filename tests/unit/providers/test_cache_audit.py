"""TDD tests for the cache audit trail (CacheAuditRecord + generate_cache_audit).

Verifies the audit artifact dataclass fields, the per-provider builder, and
the bulk generate function across all registered providers.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from victor.providers.cache_audit import (
    CacheAuditRecord,
    build_cache_audit_record,
    generate_cache_audit,
)

# ── 1. CacheAuditRecord dataclass ─────────────────────────────────────────────


class TestCacheAuditRecordModel:
    """CacheAuditRecord is a frozen pydantic model with all required fields."""

    def test_record_importable_and_constructible(self):
        record = CacheAuditRecord(
            provider_name="openai",
            cache_supports=True,
            cache_type="auto_prefix",
            serializer_method_name="build_openai_messages",
            tools_ordering="before_messages",
            boundary_position=None,
            spec_compliance="compliant",
            timestamp="2025-01-01T00:00:00+00:00",
        )
        assert record.provider_name == "openai"
        assert record.cache_supports is True
        assert record.cache_type == "auto_prefix"
        assert record.serializer_method_name == "build_openai_messages"
        assert record.tools_ordering == "before_messages"
        assert record.boundary_position is None
        assert record.spec_compliance == "compliant"
        assert record.timestamp == "2025-01-01T00:00:00+00:00"

    def test_record_is_frozen(self):
        record = CacheAuditRecord(
            provider_name="x",
            cache_supports=False,
            cache_type="none",
            serializer_method_name="unknown",
            tools_ordering="n/a",
        )
        with pytest.raises(Exception):
            record.provider_name = "changed"  # type: ignore[misc]

    def test_record_defaults(self):
        record = CacheAuditRecord(
            provider_name="test",
            cache_supports=False,
            cache_type="none",
            serializer_method_name="unknown",
            tools_ordering="n/a",
        )
        assert record.boundary_position is None
        assert record.spec_compliance == "unknown"
        assert record.timestamp == ""

    def test_record_all_required_fields_present(self):
        """Every field mandated by the spec must exist on the model."""
        fields = set(CacheAuditRecord.model_fields.keys())
        expected = {
            "provider_name",
            "cache_supports",
            "cache_type",
            "serializer_method_name",
            "tools_ordering",
            "boundary_position",
            "spec_compliance",
            "timestamp",
        }
        assert expected == fields, f"Missing fields: {expected - fields}"

    def test_cache_type_validated_values(self):
        """cache_type accepts the three canonical values."""
        for ct in ("auto_prefix", "explicit_markers", "none"):
            r = CacheAuditRecord(
                provider_name="t",
                cache_supports=(ct != "none"),
                cache_type=ct,
                serializer_method_name="m",
                tools_ordering="n/a",
            )
            assert r.cache_type == ct

    def test_spec_compliance_validated_values(self):
        for sc in ("compliant", "gap", "unknown"):
            r = CacheAuditRecord(
                provider_name="t",
                cache_supports=False,
                cache_type="none",
                serializer_method_name="m",
                tools_ordering="n/a",
                spec_compliance=sc,
            )
            assert r.spec_compliance == sc


# ── 2. build_cache_audit_record — per-provider builder ─────────────────────────


def _make_mock_provider(
    name: str = "mock",
    supports_caching: bool = True,
    has_boundary: bool = False,
    has_serializer: str | None = "_serialize_message",
    has_payload_builder: bool = False,
    tools_before: bool = True,
):
    """Build a mock provider with configurable caching surface."""
    provider = MagicMock()
    provider.name = name
    provider.supports_prompt_caching.return_value = supports_caching
    provider.supports_kv_prefix_caching.return_value = supports_caching

    if has_boundary:
        provider._find_cache_boundary = MagicMock(return_value=2)
    else:
        # MagicMock auto-creates attributes; delete to simulate absence
        if hasattr(provider, "_find_cache_boundary"):
            del provider._find_cache_boundary

    if has_serializer:
        setattr(provider, has_serializer, MagicMock())
    else:
        if hasattr(provider, has_serializer or "_serialize_message"):
            delattr(provider, has_serializer or "_serialize_message")

    if has_payload_builder:

        def _build_payload(**kwargs):
            keys_order = ["tools", "messages"] if tools_before else ["messages", "tools"]
            return {k: kwargs.get(k, []) for k in keys_order}

        provider._build_request_payload = _build_payload

    return provider


class TestBuildCacheAuditRecord:
    """build_cache_audit_record extracts correct fields from a provider."""

    def test_auto_prefix_provider(self):
        provider = _make_mock_provider(
            name="openai",
            supports_caching=True,
            has_boundary=False,
            has_serializer="build_openai_messages",
        )
        record = build_cache_audit_record(provider)
        assert record.provider_name == "openai"
        assert record.cache_supports is True
        assert record.cache_type == "auto_prefix"
        assert record.serializer_method_name == "build_openai_messages"
        assert record.boundary_position is None

    def test_explicit_markers_provider(self):
        provider = _make_mock_provider(
            name="anthropic",
            supports_caching=True,
            has_boundary=True,
            has_serializer="_serialize_message",
        )
        record = build_cache_audit_record(provider)
        assert record.provider_name == "anthropic"
        assert record.cache_type == "explicit_markers"
        assert record.boundary_position == 2
        assert record.serializer_method_name == "_serialize_message"

    def test_no_cache_provider(self):
        provider = _make_mock_provider(
            name="ollama", supports_caching=False, has_boundary=False, has_serializer=None
        )
        record = build_cache_audit_record(provider)
        assert record.cache_supports is False
        assert record.cache_type == "none"
        assert record.boundary_position is None

    def test_spec_compliance_compliant_when_caching_supported(self):
        provider = _make_mock_provider(name="deepseek", supports_caching=True)
        record = build_cache_audit_record(provider)
        assert record.spec_compliance == "compliant"

    def test_spec_compliance_unknown_when_no_caching(self):
        provider = _make_mock_provider(name="lmstudio", supports_caching=False)
        record = build_cache_audit_record(provider)
        assert record.spec_compliance == "unknown"

    def test_timestamp_is_iso_format(self):
        provider = _make_mock_provider(name="x")
        record = build_cache_audit_record(provider)
        # Parse should not raise
        dt = datetime.fromisoformat(record.timestamp)
        assert dt is not None

    def test_tools_ordering_before_messages(self):
        provider = _make_mock_provider(name="deepseek", has_payload_builder=True, tools_before=True)
        record = build_cache_audit_record(provider)
        assert record.tools_ordering == "before_messages"

    def test_tools_ordering_after_messages(self):
        provider = _make_mock_provider(name="weird", has_payload_builder=True, tools_before=False)
        record = build_cache_audit_record(provider)
        assert record.tools_ordering == "after_messages"

    def test_tools_ordering_na_when_no_builder(self):
        provider = _make_mock_provider(name="simple", has_payload_builder=False)
        record = build_cache_audit_record(provider)
        assert record.tools_ordering == "n/a"

    def test_provider_name_falls_back_to_class_name(self):
        provider = _make_mock_provider()
        del provider.name
        provider.__class__.__name__ = "FakeProvider"
        record = build_cache_audit_record(provider)
        assert record.provider_name == "FakeProvider"


# ── 3. generate_cache_audit — bulk generator ───────────────────────────────────


class TestGenerateCacheAudit:
    """generate_cache_audit produces records for a list of providers."""

    def test_generate_from_explicit_list(self):
        providers = [
            _make_mock_provider(
                name="openai", supports_caching=True, has_serializer="build_openai_messages"
            ),
            _make_mock_provider(
                name="anthropic",
                supports_caching=True,
                has_boundary=True,
                has_serializer="_serialize_message",
            ),
            _make_mock_provider(name="ollama", supports_caching=False, has_serializer=None),
        ]
        records = generate_cache_audit(providers)
        assert len(records) == 3
        names = {r.provider_name for r in records}
        assert names == {"openai", "anthropic", "ollama"}

    def test_generate_returns_cache_audit_records(self):
        providers = [_make_mock_provider(name="x")]
        records = generate_cache_audit(providers)
        assert all(isinstance(r, CacheAuditRecord) for r in records)

    def test_generate_empty_list(self):
        records = generate_cache_audit([])
        assert records == []

    def test_generate_classifies_two_paradigms(self):
        """Auto-prefix family vs explicit-markers family are distinguishable."""
        providers = [
            _make_mock_provider(
                name="openai", supports_caching=True, has_serializer="build_openai_messages"
            ),
            _make_mock_provider(
                name="xai", supports_caching=True, has_serializer="build_openai_messages"
            ),
            _make_mock_provider(
                name="deepseek", supports_caching=True, has_serializer="build_openai_messages"
            ),
            _make_mock_provider(
                name="anthropic",
                supports_caching=True,
                has_boundary=True,
                has_serializer="_serialize_message",
            ),
        ]
        records = generate_cache_audit(providers)
        types = {r.provider_name: r.cache_type for r in records}
        assert types["openai"] == "auto_prefix"
        assert types["xai"] == "auto_prefix"
        assert types["deepseek"] == "auto_prefix"
        assert types["anthropic"] == "explicit_markers"

    def test_generate_discovery_returns_list(self):
        """When called with None, should return a list (may be empty if no providers registered)."""
        records = generate_cache_audit(None)
        assert isinstance(records, list)


# ── 4. Real provider smoke tests ───────────────────────────────────────────────


class TestRealProviderAudit:
    """Smoke-test build_cache_audit_record against real provider classes."""

    def test_anthropic_provider_audit(self):
        from victor.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(api_key="sk-test-fake")
        record = build_cache_audit_record(provider)
        assert record.provider_name == "anthropic"
        assert record.cache_supports is True
        assert record.cache_type == "explicit_markers"
        assert record.serializer_method_name == "_serialize_message"
        assert record.spec_compliance == "compliant"

    def test_openai_provider_audit(self):
        from victor.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(api_key="sk-test-fake")
        record = build_cache_audit_record(provider)
        assert record.provider_name == "openai"
        assert record.cache_supports is True
        assert record.cache_type == "auto_prefix"
        assert record.boundary_position is None
        assert record.spec_compliance == "compliant"

    def test_deepseek_provider_audit(self):
        from victor.providers.deepseek_provider import DeepSeekProvider

        provider = DeepSeekProvider(api_key="sk-test-fake", base_url="https://api.deepseek.com/v1")
        record = build_cache_audit_record(provider)
        assert record.provider_name == "deepseek"
        assert record.cache_supports is True
        assert record.cache_type == "auto_prefix"
        assert record.boundary_position is None
