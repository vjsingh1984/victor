"""Tests for bootstrap hints decoupling and phase rename."""

from unittest.mock import patch, MagicMock


class TestLoadVerticalPackageHints:
    """Tests for _load_vertical_package_hints dynamic loading."""

    def test_defaults_present_without_entry_points(self):
        """Hardcoded fallbacks are returned when no entry points exist."""
        from victor.core.bootstrap import _load_vertical_package_hints

        hints = _load_vertical_package_hints()
        assert hints["coding"] == "victor-coding"
        assert hints["research"] == "victor-research"
        assert hints["devops"] == "victor-devops"
        assert hints["investment"] == "victor-invest"

    def test_entry_point_extends_hints(self):
        """An entry point adds a new vertical hint."""
        from victor.core.bootstrap import _load_vertical_package_hints

        fake_ep = MagicMock()
        fake_ep.name = "rag"
        fake_ep.load.return_value = "victor-rag"

        with patch("importlib.metadata.entry_points") as mock_eps:
            mock_eps.return_value = MagicMock()
            mock_eps.return_value.select.return_value = [fake_ep]
            hints = _load_vertical_package_hints()

        assert hints["rag"] == "victor-rag"
        # defaults still present
        assert hints["coding"] == "victor-coding"

    def test_entry_point_overrides_default(self):
        """An entry point can override a built-in hint."""
        from victor.core.bootstrap import _load_vertical_package_hints

        fake_ep = MagicMock()
        fake_ep.name = "coding"
        fake_ep.load.return_value = "victor-coding-pro"

        with patch("importlib.metadata.entry_points") as mock_eps:
            mock_eps.return_value = MagicMock()
            mock_eps.return_value.select.return_value = [fake_ep]
            hints = _load_vertical_package_hints()

        assert hints["coding"] == "victor-coding-pro"

    def test_entry_point_failure_falls_back(self):
        """If entry_points() raises, the defaults are still returned."""
        from victor.core.bootstrap import _load_vertical_package_hints

        with patch("importlib.metadata.entry_points", side_effect=Exception("boom")):
            hints = _load_vertical_package_hints()

        assert hints["coding"] == "victor-coding"


class TestBootstrapPhaseRename:
    """Tests for the coding -> vertical_services phase rename."""

    def test_phase_named_vertical_services(self):
        """The bootstrap DAG should have 'vertical_services', not 'coding'."""
        from victor.core.bootstrap import _BOOTSTRAP_PHASES

        phase_names = [p.name for p in _BOOTSTRAP_PHASES]
        assert "vertical_services" in phase_names
        assert "coding" not in phase_names

    def test_vertical_services_phase_is_optional(self):
        """The vertical_services phase should be optional."""
        from victor.core.bootstrap import _BOOTSTRAP_PHASES

        phase = next(p for p in _BOOTSTRAP_PHASES if p.name == "vertical_services")
        assert phase.optional is True

    def test_vertical_services_depends_on_capabilities(self):
        """The vertical_services phase should depend on capabilities."""
        from victor.core.bootstrap import _BOOTSTRAP_PHASES

        phase = next(p for p in _BOOTSTRAP_PHASES if p.name == "vertical_services")
        assert "capabilities" in phase.depends_on
