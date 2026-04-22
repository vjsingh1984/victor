"""Tests for new config groups extracted from Settings class.

Tests cover:
- WorkflowSettings
- ResponseSettings
- CacheSettings
"""

import pytest

from victor.config.groups.workflow_config import WorkflowSettings
from victor.config.groups.response_config import ResponseSettings
from victor.config.groups.cache_config import CacheSettings


class TestWorkflowSettings:
    """Tests for WorkflowSettings."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        settings = WorkflowSettings()

        assert settings.workflow_definition_cache_enabled is True
        assert settings.workflow_definition_cache_ttl == 3600
        assert settings.workflow_definition_cache_max_entries == 100
        assert settings.stategraph_copy_on_write_enabled is True

    def test_custom_values(self):
        """Test custom values."""
        settings = WorkflowSettings(
            workflow_definition_cache_enabled=False,
            workflow_definition_cache_ttl=7200,
            workflow_definition_cache_max_entries=200,
            stategraph_copy_on_write_enabled=False,
        )

        assert settings.workflow_definition_cache_enabled is False
        assert settings.workflow_definition_cache_ttl == 7200
        assert settings.workflow_definition_cache_max_entries == 200
        assert settings.stategraph_copy_on_write_enabled is False

    def test_cache_ttl_validation(self):
        """Test cache TTL validation."""
        # Valid TTL
        settings = WorkflowSettings(workflow_definition_cache_ttl=0)
        assert settings.workflow_definition_cache_ttl == 0

        # Invalid TTL (negative)
        with pytest.raises(ValueError, match="workflow_definition_cache_ttl must be >= 0"):
            WorkflowSettings(workflow_definition_cache_ttl=-1)

    def test_max_entries_validation(self):
        """Test max entries validation."""
        # Valid max entries
        settings = WorkflowSettings(workflow_definition_cache_max_entries=1)
        assert settings.workflow_definition_cache_max_entries == 1

        # Invalid max entries (zero)
        with pytest.raises(ValueError, match="workflow_definition_cache_max_entries must be >= 1"):
            WorkflowSettings(workflow_definition_cache_max_entries=0)


class TestResponseSettings:
    """Tests for ResponseSettings."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        settings = ResponseSettings()

        assert settings.response_completion_retries == 3
        assert settings.response_token_reserve == 4096

    def test_custom_values(self):
        """Test custom values."""
        settings = ResponseSettings(
            response_completion_retries=5,
            response_token_reserve=8192,
        )

        assert settings.response_completion_retries == 5
        assert settings.response_token_reserve == 8192

    def test_token_reserve_validation(self):
        """Test token reserve validation."""
        # Valid token reserve
        settings = ResponseSettings(response_token_reserve=512)
        assert settings.response_token_reserve == 512

        settings = ResponseSettings(response_token_reserve=32768)
        assert settings.response_token_reserve == 32768

        # Invalid token reserve (too small)
        with pytest.raises(ValueError, match="response_token_reserve must be >= 512"):
            ResponseSettings(response_token_reserve=511)

        # Invalid token reserve (too large)
        with pytest.raises(ValueError, match="response_token_reserve must be <= 32768"):
            ResponseSettings(response_token_reserve=32769)


class TestCacheSettings:
    """Tests for CacheSettings."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        settings = CacheSettings()

        assert settings.tool_cache_enabled is True
        assert settings.tool_cache_ttl == 600
        assert len(settings.tool_cache_allowlist) == 4
        assert settings.generic_result_cache_enabled is False
        assert settings.generic_result_cache_ttl == 300
        assert settings.tool_selection_cache_enabled is True
        assert settings.tool_selection_cache_ttl == 300
        assert settings.http_connection_pool_enabled is False
        assert settings.http_connection_pool_max_connections == 100
        assert settings.http_connection_pool_max_connections_per_host == 10
        assert settings.http_connection_pool_connection_timeout == 30
        assert settings.http_connection_pool_total_timeout == 60

    def test_custom_values(self):
        """Test custom values."""
        settings = CacheSettings(
            tool_cache_enabled=False,
            tool_cache_ttl=1200,
            generic_result_cache_enabled=True,
            generic_result_cache_ttl=600,
            tool_selection_cache_enabled=False,
            tool_selection_cache_ttl=0,
            http_connection_pool_enabled=True,
            http_connection_pool_max_connections=200,
            http_connection_pool_max_connections_per_host=20,
            http_connection_pool_connection_timeout=60,
            http_connection_pool_total_timeout=120,
        )

        assert settings.tool_cache_enabled is False
        assert settings.tool_cache_ttl == 1200
        assert settings.generic_result_cache_enabled is True
        assert settings.generic_result_cache_ttl == 600
        assert settings.tool_selection_cache_enabled is False
        assert settings.tool_selection_cache_ttl == 0
        assert settings.http_connection_pool_enabled is True
        assert settings.http_connection_pool_max_connections == 200
        assert settings.http_connection_pool_max_connections_per_host == 20
        assert settings.http_connection_pool_connection_timeout == 60
        assert settings.http_connection_pool_total_timeout == 120

    def test_tool_cache_ttl_validation(self):
        """Test tool cache TTL validation."""
        # Valid TTL
        settings = CacheSettings(tool_cache_ttl=0)
        assert settings.tool_cache_ttl == 0

        # Invalid TTL (negative)
        with pytest.raises(ValueError, match="tool_cache_ttl must be >= 0"):
            CacheSettings(tool_cache_ttl=-1)

    def test_generic_cache_ttl_validation(self):
        """Test generic cache TTL validation."""
        # Valid TTL
        settings = CacheSettings(generic_result_cache_ttl=0)
        assert settings.generic_result_cache_ttl == 0

        # Invalid TTL (negative)
        with pytest.raises(ValueError, match="generic_result_cache_ttl must be >= 0"):
            CacheSettings(generic_result_cache_ttl=-1)

    def test_selection_cache_ttl_validation(self):
        """Test selection cache TTL validation."""
        # Valid TTL
        settings = CacheSettings(tool_selection_cache_ttl=0)
        assert settings.tool_selection_cache_ttl == 0

        # Invalid TTL (negative)
        with pytest.raises(ValueError, match="tool_selection_cache_ttl must be >= 0"):
            CacheSettings(tool_selection_cache_ttl=-1)

    def test_max_connections_validation(self):
        """Test max connections validation."""
        # Valid max connections
        settings = CacheSettings(http_connection_pool_max_connections=1)
        assert settings.http_connection_pool_max_connections == 1

        # Invalid max connections (zero)
        with pytest.raises(ValueError, match="http_connection_pool_max_connections must be >= 1"):
            CacheSettings(http_connection_pool_max_connections=0)

    def test_max_connections_per_host_validation(self):
        """Test max connections per host validation."""
        # Valid max connections per host
        settings = CacheSettings(http_connection_pool_max_connections_per_host=1)
        assert settings.http_connection_pool_max_connections_per_host == 1

        # Invalid max connections per host (zero)
        with pytest.raises(ValueError, match="http_connection_pool_max_connections_per_host must be >= 1"):
            CacheSettings(http_connection_pool_max_connections_per_host=0)

    def test_connection_timeout_validation(self):
        """Test connection timeout validation."""
        # Valid timeout
        settings = CacheSettings(http_connection_pool_connection_timeout=1)
        assert settings.http_connection_pool_connection_timeout == 1

        # Invalid timeout (zero)
        with pytest.raises(ValueError, match="http_connection_pool_connection_timeout must be > 0"):
            CacheSettings(http_connection_pool_connection_timeout=0)

    def test_total_timeout_validation(self):
        """Test total timeout validation."""
        # Valid timeout
        settings = CacheSettings(http_connection_pool_total_timeout=1)
        assert settings.http_connection_pool_total_timeout == 1

        # Invalid timeout (zero)
        with pytest.raises(ValueError, match="http_connection_pool_total_timeout must be > 0"):
            CacheSettings(http_connection_pool_total_timeout=0)


class TestConfigGroupsIntegration:
    """Integration tests for new config groups with Settings."""

    def test_workflow_settings_in_main_settings(self):
        """Test that WorkflowSettings is accessible from Settings."""
        from victor.config.settings import Settings

        settings = Settings()

        # Nested access only
        assert settings.workflow is not None
        assert isinstance(settings.workflow, WorkflowSettings)

    def test_response_settings_in_main_settings(self):
        """Test that ResponseSettings is accessible from Settings."""
        from victor.config.settings import Settings

        settings = Settings()

        # Nested access only
        assert settings.response is not None
        assert isinstance(settings.response, ResponseSettings)

    def test_cache_settings_in_main_settings(self):
        """Test that CacheSettings is accessible from Settings."""
        from victor.config.settings import Settings

        settings = Settings()

        # Nested access
        assert settings.cache is not None
        assert isinstance(settings.cache, CacheSettings)

        # Flat access removed - all access is now via nested groups
        # NOTE: Flat fields have been removed, use nested access only

    def test_flat_field_overrides_sync_to_nested_groups(self):
        """Test that nested initialization works correctly."""
        from victor.config.settings import Settings

        # Initialize with nested structure
        settings = Settings(
            **{"workflow": {"workflow_definition_cache_enabled": False}},
            **{"response": {"response_completion_retries": 5}},
            **{"cache": {"tool_cache_enabled": False}},
        )

        # Verify nested values are set correctly
        assert settings.workflow.workflow_definition_cache_enabled is False
        assert settings.response.response_completion_retries == 5
        assert settings.cache.tool_cache_enabled is False

    def test_nested_group_independence(self):
        """Test that nested groups are independent from each other."""
        from victor.config.settings import Settings

        settings = Settings()

        # Modifying one group shouldn't affect others
        workflow_id = id(settings.workflow)
        response_id = id(settings.response)
        cache_id = id(settings.cache)

        assert workflow_id != response_id != cache_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
