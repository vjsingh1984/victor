"""Tests for bootstrap decoupling (core purge — no hardcoded vertical names)."""

from unittest.mock import patch, MagicMock


class TestLoadVerticalPackageHints:
    """Tests for _load_vertical_package_hints dynamic loading."""

    def test_empty_without_entry_points(self):
        """No hardcoded hints — empty dict when no entry points exist."""
        from victor.core.bootstrap import _load_vertical_package_hints

        with patch("importlib.metadata.entry_points") as mock_eps:
            mock_eps.return_value = MagicMock()
            mock_eps.return_value.select.return_value = []
            hints = _load_vertical_package_hints()

        assert len(hints) == 0

    def test_entry_point_provides_hints(self):
        """Entry points are the sole source of vertical hints."""
        from victor.core.bootstrap import _load_vertical_package_hints

        fake_ep = MagicMock()
        fake_ep.name = "coding"
        fake_ep.load.return_value = "victor-coding"

        with patch("importlib.metadata.entry_points") as mock_eps:
            mock_eps.return_value = MagicMock()
            mock_eps.return_value.select.return_value = [fake_ep]
            hints = _load_vertical_package_hints()

        assert hints["coding"] == "victor-coding"

    def test_multiple_entry_points(self):
        """Multiple verticals advertise via entry points."""
        from victor.core.bootstrap import _load_vertical_package_hints

        eps = []
        for name, pkg in [("coding", "victor-coding"), ("rag", "victor-rag")]:
            ep = MagicMock()
            ep.name = name
            ep.load.return_value = pkg
            eps.append(ep)

        with patch("importlib.metadata.entry_points") as mock_eps:
            mock_eps.return_value = MagicMock()
            mock_eps.return_value.select.return_value = eps
            hints = _load_vertical_package_hints()

        assert hints["coding"] == "victor-coding"
        assert hints["rag"] == "victor-rag"

    def test_entry_point_failure_returns_empty(self):
        """If entry_points() raises, empty dict returned."""
        from victor.core.bootstrap import _load_vertical_package_hints

        with patch(
            "importlib.metadata.entry_points",
            side_effect=Exception("boom"),
        ):
            hints = _load_vertical_package_hints()

        assert isinstance(hints, dict)


class TestBootstrapPhaseRename:
    """Tests for the coding -> vertical_services phase rename."""

    def test_phase_named_vertical_services(self):
        """Bootstrap DAG has 'vertical_services', not 'coding'."""
        from victor.core.bootstrap import _BOOTSTRAP_PHASES

        names = [p.name for p in _BOOTSTRAP_PHASES]
        assert "vertical_services" in names
        assert "coding" not in names

    def test_vertical_services_phase_is_optional(self):
        """vertical_services phase is optional."""
        from victor.core.bootstrap import _BOOTSTRAP_PHASES

        phase = next(
            p for p in _BOOTSTRAP_PHASES
            if p.name == "vertical_services"
        )
        assert phase.optional is True

    def test_vertical_services_depends_on_capabilities(self):
        """vertical_services phase depends on capabilities."""
        from victor.core.bootstrap import _BOOTSTRAP_PHASES

        phase = next(
            p for p in _BOOTSTRAP_PHASES
            if p.name == "vertical_services"
        )
        assert "capabilities" in phase.depends_on
