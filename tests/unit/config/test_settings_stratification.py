"""Tests for Settings stratification into nested config groups.

Verifies that:
1. Flat field access still works after stratification (backward compat)
2. Nested group access works (settings.provider.default_provider)
3. Flat and nested values are synced at construction time
4. Each nested model has correct defaults independently
5. Deprecated eventbus_* fields are removed
"""

import pytest

from victor.config.settings import (
    EventSettings,
    PipelineSettings,
    ProviderSettings,
    ResilienceSettings,
    SearchSettings,
    SecuritySettings,
    Settings,
    ToolSettings,
)


class TestSettingsBackwardCompat:
    """Flat field access still works after stratification."""

    def test_flat_provider_access(self):
        s = Settings(default_provider="anthropic")
        assert s.default_provider == "anthropic"

    def test_flat_tool_access(self):
        s = Settings(tool_retry_enabled=False)
        assert s.tool_retry_enabled is False

    def test_flat_resilience_access(self):
        s = Settings(circuit_breaker_failure_threshold=10)
        assert s.circuit_breaker_failure_threshold == 10

    def test_flat_security_access(self):
        s = Settings(airgapped_mode=True)
        assert s.airgapped_mode is True

    def test_flat_event_access(self):
        s = Settings(event_backend_type="sqlite")
        assert s.event_backend_type == "sqlite"

    def test_flat_pipeline_access(self):
        s = Settings(intelligent_pipeline_enabled=False)
        assert s.intelligent_pipeline_enabled is False

    def test_flat_search_access(self):
        s = Settings(codebase_vector_store="chromadb")
        assert s.codebase_vector_store == "chromadb"


class TestNestedAccess:
    """Nested config group access works correctly."""

    def test_nested_provider_access(self):
        s = Settings(default_provider="anthropic")
        assert s.provider.default_provider == "anthropic"

    def test_nested_provider_model(self):
        s = Settings(default_model="gpt-4")
        assert s.provider.default_model == "gpt-4"

    def test_nested_provider_api_key(self):
        s = Settings(anthropic_api_key="sk-test")
        assert s.provider.anthropic_api_key.get_secret_value() == "sk-test"

    def test_nested_tools_access(self):
        s = Settings(tool_retry_max_attempts=5)
        assert s.tools.tool_retry_max_attempts == 5

    def test_nested_tools_deduplication(self):
        s = Settings(enable_tool_deduplication=False)
        assert s.tools.enable_tool_deduplication is False

    def test_nested_search_access(self):
        s = Settings(codebase_dimension=768)
        assert s.search.codebase_dimension == 768

    def test_nested_resilience_access(self):
        s = Settings(circuit_breaker_timeout=120.0)
        assert s.resilience.circuit_breaker_timeout == 120.0

    def test_nested_security_access(self):
        s = Settings(write_approval_mode="all_writes")
        assert s.security.write_approval_mode == "all_writes"

    def test_nested_security_secret_access(self):
        s = Settings(server_api_key="server-token", server_session_secret="session-secret")
        assert s.security.server_api_key.get_secret_value() == "server-token"
        assert s.security.server_session_secret.get_secret_value() == "session-secret"

    def test_nested_events_access(self):
        s = Settings(event_queue_maxsize=5000)
        assert s.events.event_queue_maxsize == 5000

    def test_nested_pipeline_access(self):
        s = Settings(max_exploration_iterations=100)
        assert s.pipeline.max_exploration_iterations == 100


class TestFlatNestedSync:
    """Flat and nested values are synced at construction time."""

    def test_provider_sync(self):
        s = Settings(default_provider="openai", default_temperature=0.5)
        assert s.default_provider == s.provider.default_provider
        assert s.default_temperature == s.provider.default_temperature

    def test_tools_sync(self):
        s = Settings(fallback_max_tools=12)
        assert s.fallback_max_tools == s.tools.fallback_max_tools

    def test_resilience_sync(self):
        s = Settings(retry_max_attempts=5)
        assert s.retry_max_attempts == s.resilience.retry_max_attempts

    def test_security_sync(self):
        s = Settings(server_max_sessions=50)
        assert s.server_max_sessions == s.security.server_max_sessions

    def test_events_sync(self):
        s = Settings(event_backend_type="redis")
        assert s.event_backend_type == s.events.event_backend_type

    def test_pipeline_sync(self):
        s = Settings(session_idle_timeout=300)
        assert s.session_idle_timeout == s.pipeline.session_idle_timeout

    def test_search_sync(self):
        s = Settings(semantic_similarity_threshold=0.5)
        assert s.semantic_similarity_threshold == s.search.semantic_similarity_threshold


class TestNestedModelDefaults:
    """Nested config groups work independently with correct defaults."""

    def test_provider_settings_defaults(self):
        ps = ProviderSettings()
        assert ps.default_provider == "ollama"
        assert ps.default_model == "qwen3-coder:30b"
        assert ps.default_temperature == 0.7
        assert ps.default_max_tokens == 4096
        assert ps.anthropic_api_key is None
        assert ps.ollama_base_url == "http://localhost:11434"
        assert ps.lmstudio_max_vram_gb == 48.0

    def test_tool_settings_defaults(self):
        ts = ToolSettings()
        assert ts.tool_retry_enabled is True
        assert ts.tool_retry_max_attempts == 3
        assert ts.fallback_max_tools == 8
        assert ts.use_semantic_tool_selection is True
        assert ts.embedding_provider == "sentence-transformers"
        assert ts.tool_validation_mode == "lenient"

    def test_search_settings_defaults(self):
        ss = SearchSettings()
        assert ss.codebase_vector_store == "lancedb"
        assert ss.codebase_dimension == 384
        assert ss.semantic_similarity_threshold == 0.25
        assert ss.enable_hybrid_search is False

    def test_resilience_settings_defaults(self):
        rs = ResilienceSettings()
        assert rs.resilience_enabled is True
        assert rs.circuit_breaker_failure_threshold == 5
        assert rs.retry_max_attempts == 3
        assert rs.rate_limiting_enabled is True
        assert rs.rate_limit_requests_per_minute == 50

    def test_security_settings_defaults(self):
        ss = SecuritySettings()
        assert ss.airgapped_mode is False
        assert ss.write_approval_mode == "risky_only"
        assert ss.code_executor_network_disabled is True
        assert ss.headless_mode is False

    def test_event_settings_defaults(self):
        es = EventSettings()
        assert es.event_backend_type == "in_memory"
        assert es.event_delivery_guarantee == "at_most_once"
        assert es.event_queue_maxsize == 10000
        assert es.event_queue_overflow_policy == "drop_newest"

    def test_pipeline_settings_defaults(self):
        ps = PipelineSettings()
        assert ps.intelligent_pipeline_enabled is True
        assert ps.max_exploration_iterations == 200
        assert ps.recovery_empty_response_threshold == 5
        assert ps.session_idle_timeout == 180


class TestDeprecatedFieldsRemoved:
    """Deprecated eventbus_* fields are removed."""

    def test_eventbus_fields_not_on_settings(self):
        """Verify deprecated eventbus_* fields are no longer defined on Settings."""
        settings_fields = set(Settings.model_fields.keys())
        deprecated_fields = {
            "eventbus_backend",
            "eventbus_queue_maxsize",
            "eventbus_backpressure_strategy",
            "eventbus_sampling_enabled",
            "eventbus_sampling_default_rate",
            "eventbus_batching_enabled",
            "eventbus_batch_size",
            "eventbus_batch_flush_interval_ms",
        }
        overlap = settings_fields & deprecated_fields
        assert not overlap, f"Deprecated fields still on Settings: {overlap}"


class TestNestedModelsExcludedFromDump:
    """Nested models are excluded from model_dump to avoid duplication."""

    def test_nested_groups_not_in_dump(self):
        s = Settings()
        dump = s.model_dump()
        for group_name in (
            "provider",
            "tools",
            "search",
            "resilience",
            "security",
            "events",
            "pipeline",
        ):
            assert group_name not in dump, f"{group_name} should be excluded from dump"

    def test_flat_fields_in_dump(self):
        s = Settings()
        dump = s.model_dump()
        assert "default_provider" in dump
        assert "tool_retry_enabled" in dump
        assert "resilience_enabled" in dump


class TestAllNestedGroupsPopulated:
    """All 7 nested groups are populated after construction."""

    def test_all_groups_exist(self):
        s = Settings()
        assert isinstance(s.provider, ProviderSettings)
        assert isinstance(s.tools, ToolSettings)
        assert isinstance(s.search, SearchSettings)
        assert isinstance(s.resilience, ResilienceSettings)
        assert isinstance(s.security, SecuritySettings)
        assert isinstance(s.events, EventSettings)
        assert isinstance(s.pipeline, PipelineSettings)
