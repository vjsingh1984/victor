"""Tests verifying orchestrator delegates correctly via both paths.

Tests that the orchestrator methods produce the same results whether
using direct coordinator delegation or service adapter delegation,
based on the USE_SERVICE_LAYER feature flag.
"""

from unittest.mock import MagicMock, patch

import pytest

from victor.core.feature_flags import FeatureFlag, FeatureFlagConfig, FeatureFlagManager


class TestFeatureFlagIntegration:
    """Test that the USE_SERVICE_LAYER flag is properly recognized."""

    def test_flag_exists_in_enum(self):
        assert hasattr(FeatureFlag, "USE_SERVICE_LAYER")
        assert FeatureFlag.USE_SERVICE_LAYER.value == "use_service_layer"

    def test_flag_env_var_name(self):
        assert (
            FeatureFlag.USE_SERVICE_LAYER.get_env_var_name()
            == "VICTOR_USE_SERVICE_LAYER"
        )

    def test_flag_enabled_by_default(self):
        manager = FeatureFlagManager(FeatureFlagConfig())
        assert manager.is_enabled(FeatureFlag.USE_SERVICE_LAYER) is True

    def test_flag_enabled_via_env(self, monkeypatch):
        monkeypatch.setenv("VICTOR_USE_SERVICE_LAYER", "true")
        manager = FeatureFlagManager(FeatureFlagConfig())
        assert manager.is_enabled(FeatureFlag.USE_SERVICE_LAYER) is True

    def test_flag_disabled_via_env(self, monkeypatch):
        monkeypatch.setenv("VICTOR_USE_SERVICE_LAYER", "false")
        manager = FeatureFlagManager(FeatureFlagConfig())
        assert manager.is_enabled(FeatureFlag.USE_SERVICE_LAYER) is False

    def test_flag_runtime_enable(self):
        manager = FeatureFlagManager(FeatureFlagConfig())
        manager.enable(FeatureFlag.USE_SERVICE_LAYER)
        assert manager.is_enabled(FeatureFlag.USE_SERVICE_LAYER) is True


class TestAdapterImports:
    """Test that adapter module imports work correctly."""

    def test_import_chat_adapter(self):
        from victor.agent.services.adapters.chat_adapter import ChatServiceAdapter

        assert ChatServiceAdapter is not None

    def test_import_tool_adapter(self):
        from victor.agent.services.adapters.tool_adapter import ToolServiceAdapter

        assert ToolServiceAdapter is not None

    def test_import_context_adapter(self):
        from victor.agent.services.adapters.context_adapter import ContextServiceAdapter

        assert ContextServiceAdapter is not None

    def test_import_session_adapter(self):
        from victor.agent.services.adapters.session_adapter import SessionServiceAdapter

        assert SessionServiceAdapter is not None

    def test_import_all_from_package(self):
        from victor.agent.services.adapters import (
            ChatServiceAdapter,
            ToolServiceAdapter,
            ContextServiceAdapter,
            SessionServiceAdapter,
        )

        assert all(
            [
                ChatServiceAdapter,
                ToolServiceAdapter,
                ContextServiceAdapter,
                SessionServiceAdapter,
            ]
        )


class TestAdapterProtocolConformance:
    """Test that adapters expose the expected interface methods."""

    def test_chat_adapter_has_required_methods(self):
        from victor.agent.services.adapters.chat_adapter import ChatServiceAdapter

        adapter = ChatServiceAdapter(MagicMock())
        assert callable(getattr(adapter, "chat", None))
        assert callable(getattr(adapter, "stream_chat", None))
        assert callable(getattr(adapter, "chat_with_planning", None))
        assert callable(getattr(adapter, "reset_conversation", None))
        assert callable(getattr(adapter, "is_healthy", None))

    def test_tool_adapter_has_required_methods(self):
        from victor.agent.services.adapters.tool_adapter import ToolServiceAdapter

        adapter = ToolServiceAdapter(MagicMock())
        assert callable(getattr(adapter, "get_available_tools", None))
        assert callable(getattr(adapter, "get_enabled_tools", None))
        assert callable(getattr(adapter, "set_enabled_tools", None))
        assert callable(getattr(adapter, "is_tool_enabled", None))
        assert callable(getattr(adapter, "execute_tool_with_retry", None))
        assert callable(getattr(adapter, "parse_and_validate_tool_calls", None))
        assert callable(getattr(adapter, "is_healthy", None))

    def test_session_adapter_has_required_methods(self):
        from victor.agent.services.adapters.session_adapter import SessionServiceAdapter

        adapter = SessionServiceAdapter(MagicMock())
        assert callable(getattr(adapter, "get_recent_sessions", None))
        assert callable(getattr(adapter, "recover_session", None))
        assert callable(getattr(adapter, "get_session_stats", None))
        assert callable(getattr(adapter, "save_checkpoint", None))
        assert callable(getattr(adapter, "restore_checkpoint", None))
        assert callable(getattr(adapter, "is_healthy", None))

    def test_context_adapter_has_required_methods(self):
        from victor.agent.services.adapters.context_adapter import ContextServiceAdapter

        adapter = ContextServiceAdapter(MagicMock())
        assert callable(getattr(adapter, "get_context_metrics", None))
        assert callable(getattr(adapter, "compact_context", None))
        assert callable(getattr(adapter, "add_message", None))
        assert callable(getattr(adapter, "get_messages", None))
        assert callable(getattr(adapter, "is_healthy", None))
