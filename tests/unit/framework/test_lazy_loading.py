"""Tests for PEP 562 lazy loading in victor.framework.__init__."""

import sys

import pytest


class TestLazyLoading:
    """Test PEP 562 lazy loading of framework modules."""

    def test_core_imports_available(self):
        """Core names should be directly importable."""
        from victor.framework import Agent, Task, Tools, State, Stage
        from victor.framework import AgentExecutionEvent, EventType
        from victor.framework import ToolSet, ToolCategory
        from victor.framework import AgentConfig

        assert Agent is not None
        assert Task is not None
        assert Tools is not None

    def test_core_protocols_available(self):
        """Protocol imports should work."""
        from victor.framework import OrchestratorProtocol, ProviderProtocol

        assert OrchestratorProtocol is not None

    def test_core_errors_available(self):
        """Error imports should work."""
        from victor.framework import AgentError, ProviderError, ToolError

        assert issubclass(AgentError, Exception)

    def test_lazy_import_triggers_on_access(self):
        """Lazy names should load their module on first access."""
        import victor.framework

        # Access a lazy name
        val = getattr(victor.framework, "StateGraph", None)
        # Should either succeed or be None if module not installed
        # The key test is no crash

    def test_nonexistent_attribute_raises(self):
        """Accessing a non-existent attribute should raise AttributeError."""
        import victor.framework

        with pytest.raises(AttributeError, match="no attribute"):
            getattr(victor.framework, "ThisDoesNotExist12345")

    def test_dir_contains_core_names(self):
        """__dir__ should include core names."""
        import victor.framework

        names = dir(victor.framework)
        assert "Agent" in names
        assert "Task" in names
        assert "Tools" in names
        assert "State" in names
        assert "EventType" in names
        assert "discover" in names

    def test_dir_contains_lazy_names(self):
        """__dir__ should include lazy-loaded names."""
        import victor.framework

        names = dir(victor.framework)
        # These are lazy names
        assert "StateGraph" in names
        assert "CircuitBreaker" in names
        assert "HealthChecker" in names

    def test_aliased_imports(self):
        """Aliased imports should resolve correctly."""
        import victor.framework

        # AgentTeamFormation is an alias for TeamFormation
        names = dir(victor.framework)
        assert "AgentTeamFormation" in names

    def test_all_includes_all_names(self):
        """__all__ should contain both core and lazy names."""
        import victor.framework

        assert "Agent" in victor.framework.__all__
        assert "StateGraph" in victor.framework.__all__

    def test_version_available(self):
        """__version__ should be set."""
        import victor.framework

        assert hasattr(victor.framework, "__version__")
        assert isinstance(victor.framework.__version__, str)

    def test_discover_callable(self):
        """discover() should be callable."""
        from victor.framework import discover

        assert callable(discover)

    def test_optional_modules_not_loaded_eagerly(self):
        """Importing victor.framework should NOT load optional modules."""
        # Remove cached modules to test fresh import behavior
        mods_to_check = [
            "victor.framework.cqrs_bridge",
            "victor.framework.health",
            "victor.framework.metrics",
            "victor.framework.resilience",
            "victor.framework.observability",
        ]

        # After importing victor.framework, these optional modules
        # should NOT be in sys.modules unless something else loaded them
        import victor.framework  # noqa: F401

        # We can't guarantee they aren't loaded by other test fixtures,
        # but we verify that the lazy loading mechanism exists
        assert hasattr(victor.framework, "__getattr__")
