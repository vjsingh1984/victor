"""Tests for ModuleAnalyzer."""

import sqlite3
from pathlib import Path

import pytest

from victor.analysis.module_analyzer import ModuleAnalyzer, ModuleMetrics


@pytest.fixture
def in_memory_db():
    """Create an in-memory SQLite database with schema."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE graph_node (
            node_id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            name TEXT NOT NULL,
            file TEXT NOT NULL,
            line INTEGER,
            end_line INTEGER,
            lang TEXT,
            signature TEXT,
            docstring TEXT,
            parent_id TEXT,
            embedding_ref TEXT,
            metadata TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE graph_edge (
            src TEXT NOT NULL,
            dst TEXT NOT NULL,
            type TEXT NOT NULL,
            weight REAL,
            metadata TEXT,
            PRIMARY KEY (src, dst, type)
        )
    """)
    conn.execute("""
        CREATE TABLE graph_module_metric (
            module_path TEXT PRIMARY KEY,
            pagerank_score REAL DEFAULT 0.0,
            betweenness REAL DEFAULT 0.0,
            afferent_coupling INTEGER DEFAULT 0,
            efferent_coupling INTEGER DEFAULT 0,
            instability REAL DEFAULT 0.0,
            abstractness REAL DEFAULT 0.0,
            distance_main_seq REAL DEFAULT 0.0,
            cohesion_lcom4 REAL DEFAULT 0.0,
            hotspot_score REAL DEFAULT 0.0,
            symbol_count INTEGER DEFAULT 0,
            change_frequency INTEGER DEFAULT 0,
            tdd_priority REAL DEFAULT 0.0,
            computed_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE TABLE graph_module_metric_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            module_path TEXT NOT NULL,
            hotspot_score REAL,
            tdd_priority REAL,
            computed_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    return conn


class FakeDB:
    """Fake database manager wrapping a connection."""

    def __init__(self, conn):
        self.connection = conn

    def _get_raw_connection(self):
        return self.connection


def _seed_graph(conn, modules, edges):
    """Seed graph_node and graph_edge tables."""
    node_id = 0
    for mod, symbols in modules.items():
        for sym_name, sym_type in symbols:
            conn.execute(
                "INSERT INTO graph_node (node_id, type, name, file) VALUES (?, ?, ?, ?)",
                (f"n{node_id}", sym_type, sym_name, mod),
            )
            node_id += 1
    for src_mod, dst_mod in edges:
        # Find a node in each module
        src_node = conn.execute(
            "SELECT node_id FROM graph_node WHERE file = ? LIMIT 1", (src_mod,)
        ).fetchone()
        dst_node = conn.execute(
            "SELECT node_id FROM graph_node WHERE file = ? LIMIT 1", (dst_mod,)
        ).fetchone()
        if src_node and dst_node:
            conn.execute(
                "INSERT OR IGNORE INTO graph_edge (src, dst, type, weight) VALUES (?, ?, 'CALLS', 1.0)",
                (src_node[0], dst_node[0]),
            )
    conn.commit()


class TestModuleAnalyzer:
    """Tests for ModuleAnalyzer."""

    def test_compute_coupling(self, in_memory_db):
        """Test afferent/efferent coupling computation."""
        analyzer = ModuleAnalyzer(db=FakeDB(in_memory_db))
        adj_out = {"a.py": {"b.py", "c.py"}, "b.py": {"c.py"}}
        adj_in = {"b.py": {"a.py"}, "c.py": {"a.py", "b.py"}}

        ca, ce, instability = analyzer._compute_coupling("a.py", adj_out, adj_in)
        assert ce == 2  # a.py depends on b.py and c.py
        assert ca == 0  # nothing depends on a.py
        assert instability == 1.0  # Ce / (Ca + Ce) = 2/2

        ca, ce, instability = analyzer._compute_coupling("c.py", adj_out, adj_in)
        assert ca == 2  # a.py and b.py depend on c.py
        assert ce == 0
        assert instability == 0.0

    def test_compute_coupling_balanced(self, in_memory_db):
        """Test balanced coupling."""
        analyzer = ModuleAnalyzer(db=FakeDB(in_memory_db))
        adj_out = {"a.py": {"b.py"}}
        adj_in = {"a.py": {"c.py"}}

        ca, ce, instability = analyzer._compute_coupling("a.py", adj_out, adj_in)
        assert ca == 1
        assert ce == 1
        assert instability == 0.5

    def test_compute_pagerank_triangle(self, in_memory_db):
        """Test PageRank on a triangle graph."""
        analyzer = ModuleAnalyzer(db=FakeDB(in_memory_db))
        modules = {"a.py", "b.py", "c.py"}
        adj = {
            "a.py": {"b.py"},
            "b.py": {"c.py"},
            "c.py": {"a.py"},
        }

        scores = analyzer._compute_pagerank(modules, adj)
        # Triangle graph: all nodes should have roughly equal PageRank
        assert len(scores) == 3
        values = list(scores.values())
        assert all(abs(v - values[0]) < 0.01 for v in values)

    def test_compute_pagerank_star(self, in_memory_db):
        """Test PageRank on a star graph (hub pattern)."""
        analyzer = ModuleAnalyzer(db=FakeDB(in_memory_db))
        modules = {"hub.py", "a.py", "b.py", "c.py"}
        adj = {
            "a.py": {"hub.py"},
            "b.py": {"hub.py"},
            "c.py": {"hub.py"},
            "hub.py": set(),
        }

        scores = analyzer._compute_pagerank(modules, adj)
        # Hub should have highest PageRank
        assert scores["hub.py"] > scores["a.py"]
        assert scores["hub.py"] > scores["b.py"]

    def test_compute_betweenness(self, in_memory_db):
        """Test betweenness centrality on a chain."""
        analyzer = ModuleAnalyzer(db=FakeDB(in_memory_db))
        modules = {"a.py", "b.py", "c.py"}
        adj = {"a.py": {"b.py"}, "b.py": {"c.py"}, "c.py": set()}

        betweenness = analyzer._compute_betweenness(modules, adj)
        # b.py is on the path from a.py to c.py
        assert betweenness["b.py"] >= betweenness["a.py"]
        assert betweenness["b.py"] >= betweenness["c.py"]

    def test_persist_and_get_cached(self, in_memory_db):
        """Test SQLite roundtrip."""
        analyzer = ModuleAnalyzer(db=FakeDB(in_memory_db))
        metrics = [
            ModuleMetrics(
                module_path="test.py",
                pagerank_score=0.5,
                betweenness=0.3,
                afferent_coupling=2,
                efferent_coupling=3,
                instability=0.6,
                hotspot_score=0.7,
                tdd_priority=0.8,
            ),
            ModuleMetrics(
                module_path="other.py",
                pagerank_score=0.2,
                hotspot_score=0.3,
                tdd_priority=0.4,
            ),
        ]

        analyzer.persist(metrics)

        cached = analyzer.get_cached(order_by="hotspot_score", limit=10)
        assert len(cached) == 2
        assert cached[0]["module_path"] == "test.py"  # Higher hotspot
        assert cached[0]["hotspot_score"] == 0.7

    def test_has_cached_metrics(self, in_memory_db):
        """Test cache detection."""
        analyzer = ModuleAnalyzer(db=FakeDB(in_memory_db))
        assert not analyzer.has_cached_metrics()

        metrics = [ModuleMetrics(module_path="test.py", hotspot_score=0.5)]
        analyzer.persist(metrics)
        assert analyzer.has_cached_metrics()

    def test_hotspot_formula(self, in_memory_db):
        """Test hotspot formula: 0.4*pr + 0.3*coupling + 0.3*change."""
        analyzer = ModuleAnalyzer(db=FakeDB(in_memory_db))
        m = ModuleMetrics(
            module_path="test.py",
            pagerank_score=1.0,
            instability=1.0,
            change_frequency=10,
        )
        score = analyzer._compute_hotspot(m, max_change=10)
        expected = 0.4 * 1.0 + 0.3 * 1.0 + 0.3 * 1.0  # = 1.0
        assert abs(score - expected) < 1e-6

    def test_hotspot_zero(self, in_memory_db):
        """Test hotspot with all zeros."""
        analyzer = ModuleAnalyzer(db=FakeDB(in_memory_db))
        m = ModuleMetrics(module_path="test.py")
        score = analyzer._compute_hotspot(m, max_change=1)
        assert score == 0.0

    def test_abstractness(self, in_memory_db):
        """Test abstractness computation."""
        analyzer = ModuleAnalyzer(db=FakeDB(in_memory_db))
        module_to_nodes = {
            "abs.py": [
                {"node_id": "1", "type": "class", "name": "BaseProvider"},
                {"node_id": "2", "type": "class", "name": "ConcreteProvider"},
                {"node_id": "3", "type": "function", "name": "helper"},
            ]
        }
        # 1 abstract out of 2 class-type nodes
        result = analyzer._compute_abstractness("abs.py", module_to_nodes)
        assert result == 0.5

    def test_compute_all_empty_graph(self, in_memory_db):
        """Test compute_all with empty graph."""
        analyzer = ModuleAnalyzer(db=FakeDB(in_memory_db))
        metrics = analyzer.compute_all()
        assert metrics == []
