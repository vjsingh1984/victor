"""Tests for graph tool module-level analysis modes (WS-1/WS-6)."""

import sqlite3

import pytest


class TestGraphToolModuleModes:
    """Test the 4 new graph modes: coupling, cohesion, hotspots, tdd_priority."""

    def test_coupling_mode_literal(self):
        """Verify 'coupling' is in the GraphMode type."""
        from victor.tools.graph_tool import GraphMode
        # GraphMode is a Literal type - just verify the string is valid
        assert "coupling" in GraphMode.__args__

    def test_cohesion_mode_literal(self):
        from victor.tools.graph_tool import GraphMode
        assert "cohesion" in GraphMode.__args__

    def test_hotspots_mode_literal(self):
        from victor.tools.graph_tool import GraphMode
        assert "hotspots" in GraphMode.__args__

    def test_tdd_priority_mode_literal(self):
        from victor.tools.graph_tool import GraphMode
        assert "tdd_priority" in GraphMode.__args__
