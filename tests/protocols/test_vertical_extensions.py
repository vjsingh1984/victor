# Tests for vertical-specific extension implementations
# Migrated from victor/tests/unit/protocols/test_vertical_protocols.py

import pytest

from victor_contracts import VerticalExtensions


class TestCodingVerticalExtensions:
    """Tests for the actual CodingAssistant extensions."""

    def test_coding_has_extensions(self):
        """CodingAssistant should provide extensions."""
        try:
            from victor_coding.assistant import CodingAssistant
        except ImportError:
            pytest.skip("victor-coding package not installed")

        extensions = CodingAssistant.get_extensions()
        assert extensions is not None
        assert isinstance(extensions, VerticalExtensions)

    def test_coding_middleware(self):
        """CodingAssistant should provide middleware."""
        try:
            from victor_coding.assistant import CodingAssistant
        except ImportError:
            pytest.skip("victor-coding package not installed")

        extensions = CodingAssistant.get_extensions()
        assert len(extensions.middleware) >= 1
        # Should have CodeCorrectionMiddleware or similar
        middleware_names = [type(m).__name__ for m in extensions.middleware]
        assert any("Code" in name or "Git" in name for name in middleware_names)

    @pytest.mark.skip(reason="Extension loading from external packages requires extension loader refactoring")
    def test_coding_safety_patterns(self):
        """CodingAssistant should provide safety patterns."""
        try:
            from victor_coding.assistant import CodingAssistant
        except ImportError:
            pytest.skip("victor-coding package not installed")

        extensions = CodingAssistant.get_extensions()
        patterns = extensions.get_all_safety_patterns()
        assert len(patterns) > 0

        # Should have git-related patterns
        descriptions = [p.description for p in patterns]
        assert any("git" in d.lower() or "push" in d.lower() for d in descriptions)

    @pytest.mark.skip(reason="Extension loading from external packages requires extension loader refactoring")
    def test_coding_task_hints(self):
        """CodingAssistant should provide task hints."""
        try:
            from victor_coding.assistant import CodingAssistant
        except ImportError:
            pytest.skip("victor-coding package not installed")

        extensions = CodingAssistant.get_extensions()
        hints = extensions.get_all_task_hints()
        assert len(hints) > 0

        # Should have coding-related hints
        assert any(hint_type in hints for hint_type in ["edit", "code_generation", "refactor"])

    @pytest.mark.skip(reason="Extension loading from external packages requires extension loader refactoring")
    def test_coding_mode_configs(self):
        """CodingAssistant should provide mode configs."""
        try:
            from victor_coding.assistant import CodingAssistant
        except ImportError:
            pytest.skip("victor-coding package not installed")

        extensions = CodingAssistant.get_extensions()
        modes = extensions.get_all_mode_configs()
        assert len(modes) > 0

        # Should have common modes
        assert "fast" in modes or "default" in modes


class TestResearchVerticalExtensions:
    """Tests for ResearchAssistant extensions."""

    def test_research_has_extensions(self):
        """ResearchAssistant should provide extensions (even if empty)."""
        try:
            from victor_research.assistant import ResearchAssistant
        except ImportError:
            pytest.skip("victor-research package not installed")

        extensions = ResearchAssistant.get_extensions()
        assert extensions is not None
        assert isinstance(extensions, VerticalExtensions)

    @pytest.mark.skip(reason="Extension loading from external packages requires extension loader refactoring")
    def test_research_complete_extensions(self):
        """ResearchAssistant now has complete extensions."""
        try:
            from victor_research.assistant import ResearchAssistant
        except ImportError:
            pytest.skip("victor-research package not installed")

        extensions = ResearchAssistant.get_extensions()
        # Research vertical now has safety extensions defined
        assert len(extensions.safety_extensions) >= 1
        # Safety extension should have get_bash_patterns
        patterns = extensions.get_all_safety_patterns()
        assert len(patterns) > 0


# =============================================================================
# Tests for Vertical Provider Protocols (isinstance() checks)
# =============================================================================


class TestVerticalIntegrationWithProtocols:
    """Tests for vertical integration using isinstance() checks."""

    def test_coding_vertical_compatibility(self):
        """CodingAssistant should work with the new protocol checks."""
        from victor_coding.assistant import CodingAssistant

        # Verify methods exist (duck typing compatibility)
        assert hasattr(CodingAssistant, "get_workflow_provider")
        assert hasattr(CodingAssistant, "get_rl_config_provider")
        assert hasattr(CodingAssistant, "get_rl_hooks")
        assert hasattr(CodingAssistant, "get_team_spec_provider")
