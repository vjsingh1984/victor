"""Module-level graph analysis: coupling, cohesion, PageRank, hotspots.

Computes and persists module-level metrics using the graph_module_metric
table in .victor/project.db.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModuleMetrics:
    """Metrics for a single module (file or package)."""

    module_path: str
    pagerank_score: float = 0.0
    betweenness: float = 0.0
    afferent_coupling: int = 0  # Ca: incoming deps
    efferent_coupling: int = 0  # Ce: outgoing deps
    instability: float = 0.0  # I = Ce / (Ca + Ce)
    abstractness: float = 0.0  # A = abstract / total types
    distance_main_seq: float = 0.0  # D = |A + I - 1|
    cohesion_lcom4: float = 0.0
    hotspot_score: float = 0.0
    symbol_count: int = 0
    change_frequency: int = 0
    tdd_priority: float = 0.0


class ModuleAnalyzer:
    """Compute and persist module-level metrics from the code graph."""

    def __init__(self, db=None, project_path: Optional[Path] = None):
        self._db = db
        self._project_path = project_path or Path.cwd()

    def _get_db(self):
        """Get or create database connection."""
        if self._db is not None:
            return self._db
        from victor.core.database import ProjectDatabaseManager

        return ProjectDatabaseManager(self._project_path)

    def compute_all(self) -> list[ModuleMetrics]:
        """Compute all module metrics from the graph."""
        modules, adj_out, adj_in = self._load_module_graph()
        if not modules:
            return []

        pageranks = self._compute_pagerank(modules, adj_out)
        betweenness = self._compute_betweenness(modules, adj_out)
        module_to_nodes = self._load_module_to_nodes()

        results = []
        # Normalize change frequencies for hotspot calc
        change_freqs = {}
        for mod in modules:
            change_freqs[mod] = self._get_change_frequency(mod)
        max_change = max(change_freqs.values()) if change_freqs else 1
        if max_change == 0:
            max_change = 1

        for mod in modules:
            ca, ce, instability = self._compute_coupling(mod, adj_out, adj_in)
            abstractness = self._compute_abstractness(mod, module_to_nodes)
            distance = abs(abstractness + instability - 1.0)
            cohesion = self._compute_cohesion(mod, module_to_nodes)
            change_freq = change_freqs.get(mod, 0)

            m = ModuleMetrics(
                module_path=mod,
                pagerank_score=pageranks.get(mod, 0.0),
                betweenness=betweenness.get(mod, 0.0),
                afferent_coupling=ca,
                efferent_coupling=ce,
                instability=instability,
                abstractness=abstractness,
                distance_main_seq=distance,
                cohesion_lcom4=cohesion,
                symbol_count=len(module_to_nodes.get(mod, [])),
                change_frequency=change_freq,
            )
            m.hotspot_score = self._compute_hotspot(m, max_change)
            m.tdd_priority = self._compute_tdd_priority(m, max_change)
            results.append(m)

        return results

    def persist(self, metrics: list[ModuleMetrics]) -> None:
        """UPSERT metrics into graph_module_metric table."""
        db = self._get_db()
        conn = db.connection if hasattr(db, "connection") else db._get_raw_connection()
        for m in metrics:
            conn.execute(
                """INSERT INTO graph_module_metric
                   (module_path, pagerank_score, betweenness, afferent_coupling,
                    efferent_coupling, instability, abstractness, distance_main_seq,
                    cohesion_lcom4, hotspot_score, symbol_count, change_frequency,
                    tdd_priority, computed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                   ON CONFLICT(module_path) DO UPDATE SET
                    pagerank_score=excluded.pagerank_score,
                    betweenness=excluded.betweenness,
                    afferent_coupling=excluded.afferent_coupling,
                    efferent_coupling=excluded.efferent_coupling,
                    instability=excluded.instability,
                    abstractness=excluded.abstractness,
                    distance_main_seq=excluded.distance_main_seq,
                    cohesion_lcom4=excluded.cohesion_lcom4,
                    hotspot_score=excluded.hotspot_score,
                    symbol_count=excluded.symbol_count,
                    change_frequency=excluded.change_frequency,
                    tdd_priority=excluded.tdd_priority,
                    computed_at=datetime('now')
                """,
                (
                    m.module_path,
                    m.pagerank_score,
                    m.betweenness,
                    m.afferent_coupling,
                    m.efferent_coupling,
                    m.instability,
                    m.abstractness,
                    m.distance_main_seq,
                    m.cohesion_lcom4,
                    m.hotspot_score,
                    m.symbol_count,
                    m.change_frequency,
                    m.tdd_priority,
                ),
            )
            # Also insert history
            conn.execute(
                """INSERT INTO graph_module_metric_history
                   (module_path, hotspot_score, tdd_priority)
                   VALUES (?, ?, ?)""",
                (m.module_path, m.hotspot_score, m.tdd_priority),
            )
        conn.commit()

    def get_cached(self, order_by: str = "hotspot_score", limit: int = 50) -> list[dict]:
        """Get cached metrics from the database."""
        allowed_cols = {
            "hotspot_score",
            "tdd_priority",
            "pagerank_score",
            "instability",
            "betweenness",
        }
        if order_by not in allowed_cols:
            order_by = "hotspot_score"
        db = self._get_db()
        conn = db.connection if hasattr(db, "connection") else db._get_raw_connection()
        cursor = conn.execute(
            f"SELECT * FROM graph_module_metric ORDER BY {order_by} DESC LIMIT ?",
            (limit,),
        )
        cols = [desc[0] for desc in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def has_cached_metrics(self) -> bool:
        """Check if cached metrics exist."""
        db = self._get_db()
        conn = db.connection if hasattr(db, "connection") else db._get_raw_connection()
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM graph_module_metric")
            return cursor.fetchone()[0] > 0
        except Exception:
            return False

    # === Internal methods ===

    def _load_module_graph(self) -> tuple[set[str], dict[str, set[str]], dict[str, set[str]]]:
        """Load module-level adjacency from graph_edge/graph_node tables."""
        db = self._get_db()
        conn = db.connection if hasattr(db, "connection") else db._get_raw_connection()
        try:
            rows = conn.execute(
                """SELECT DISTINCT n1.file, n2.file
                   FROM graph_edge e
                   JOIN graph_node n1 ON e.src = n1.node_id
                   JOIN graph_node n2 ON e.dst = n2.node_id
                   WHERE n1.file IS NOT NULL AND n2.file IS NOT NULL
                     AND n1.file != n2.file"""
            ).fetchall()
        except Exception:
            return set(), {}, {}

        modules: set[str] = set()
        adj_out: dict[str, set[str]] = {}
        adj_in: dict[str, set[str]] = {}
        for src_file, dst_file in rows:
            modules.add(src_file)
            modules.add(dst_file)
            adj_out.setdefault(src_file, set()).add(dst_file)
            adj_in.setdefault(dst_file, set()).add(src_file)
        return modules, adj_out, adj_in

    def _load_module_to_nodes(self) -> dict[str, list[dict]]:
        """Load mapping of modules to their node info."""
        db = self._get_db()
        conn = db.connection if hasattr(db, "connection") else db._get_raw_connection()
        try:
            rows = conn.execute(
                "SELECT node_id, type, name, file FROM graph_node WHERE file IS NOT NULL"
            ).fetchall()
        except Exception:
            return {}
        result: dict[str, list[dict]] = {}
        for node_id, ntype, name, fpath in rows:
            result.setdefault(fpath, []).append({"node_id": node_id, "type": ntype, "name": name})
        return result

    def _compute_coupling(self, module: str, adj_out: dict, adj_in: dict) -> tuple[int, int, float]:
        """Compute afferent (Ca) and efferent (Ce) coupling + instability."""
        ca = len(adj_in.get(module, set()))
        ce = len(adj_out.get(module, set()))
        total = ca + ce
        instability = ce / total if total > 0 else 0.0
        return ca, ce, instability

    def _compute_abstractness(self, module: str, module_to_nodes: dict) -> float:
        """Compute abstractness: ratio of abstract types to total types."""
        nodes = module_to_nodes.get(module, [])
        if not nodes:
            return 0.0
        type_nodes = [n for n in nodes if n["type"] in ("class", "interface", "protocol")]
        if not type_nodes:
            return 0.0
        # Heuristic: classes with "Base", "Abstract", "Protocol", "Interface" in name
        abstract_count = sum(
            1
            for n in type_nodes
            if any(
                kw in n["name"]
                for kw in ("Base", "Abstract", "Protocol", "Interface", "ABC", "Mixin")
            )
        )
        return abstract_count / len(type_nodes)

    def _compute_pagerank(
        self,
        modules: set[str],
        adj: dict[str, set[str]],
        damping: float = 0.85,
        iterations: int = 100,
    ) -> dict[str, float]:
        """Compute PageRank, delegating to Rust when available."""
        if not modules:
            return {}
        # Try Rust backend for large graphs
        try:
            from victor.native.graph_algo_loader import pagerank as native_pagerank

            # Convert set adjacency to list adjacency for the native interface
            adj_dict = {k: list(v) for k, v in adj.items()}
            # Add modules with no outgoing edges
            for m in modules:
                if m not in adj_dict:
                    adj_dict[m] = []
            return native_pagerank(adj_dict, damping, iterations)
        except ImportError:
            pass

        # Pure Python fallback (power iteration)
        n = len(modules)
        if n == 0:
            return {}
        mod_list = sorted(modules)
        scores = dict.fromkeys(mod_list, 1.0 / n)

        for _ in range(iterations):
            new_scores: dict[str, float] = {}
            for m in mod_list:
                rank = (1.0 - damping) / n
                for src in mod_list:
                    if m in adj.get(src, set()):
                        out_deg = len(adj.get(src, set()))
                        if out_deg > 0:
                            rank += damping * scores[src] / out_deg
                new_scores[m] = rank
            scores = new_scores

        return scores

    def _compute_betweenness(self, modules: set[str], adj: dict[str, set[str]]) -> dict[str, float]:
        """Compute betweenness centrality (Brandes algorithm)."""
        if not modules:
            return {}
        try:
            from victor.native.graph_algo_loader import betweenness_centrality

            adj_dict = {k: list(v) for k, v in adj.items()}
            for m in modules:
                if m not in adj_dict:
                    adj_dict[m] = []
            return betweenness_centrality(adj_dict)
        except ImportError:
            pass

        # Pure Python Brandes
        from collections import deque

        mod_list = sorted(modules)
        cb: dict[str, float] = dict.fromkeys(mod_list, 0.0)

        for s in mod_list:
            stack: list[str] = []
            pred: dict[str, list[str]] = {m: [] for m in mod_list}
            sigma: dict[str, int] = dict.fromkeys(mod_list, 0)
            sigma[s] = 1
            dist: dict[str, int] = dict.fromkeys(mod_list, -1)
            dist[s] = 0
            queue: deque[str] = deque([s])

            while queue:
                v = queue.popleft()
                stack.append(v)
                for w in adj.get(v, set()):
                    if w not in dist:
                        continue
                    if dist[w] < 0:
                        queue.append(w)
                        dist[w] = dist[v] + 1
                    if dist[w] == dist[v] + 1:
                        sigma[w] += sigma[v]
                        pred[w].append(v)

            delta: dict[str, float] = dict.fromkeys(mod_list, 0.0)
            while stack:
                w = stack.pop()
                for v in pred[w]:
                    if sigma[w] > 0:
                        delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
                if w != s:
                    cb[w] += delta[w]

        # Normalize
        n = len(mod_list)
        if n > 2:
            norm = 1.0 / ((n - 1) * (n - 2))
            cb = {m: v * norm for m, v in cb.items()}

        return cb

    def _compute_cohesion(self, module: str, module_to_nodes: dict) -> float:
        """Compute LCOM4 cohesion (0=low cohesion, 1=high cohesion)."""
        nodes = module_to_nodes.get(module, [])
        if len(nodes) <= 1:
            return 1.0
        # Simplified: connected component count over total as proxy
        # Lower component count = higher cohesion
        # For now, use 1/symbol_count as a simple heuristic
        return min(1.0, 1.0 / max(1, len(nodes) - 1))

    def _get_change_frequency(self, module: str) -> int:
        """Get git change frequency for a module (last 90 days)."""
        try:
            result = subprocess.run(
                ["git", "log", "--oneline", "--since=90 days ago", "--", module],
                capture_output=True,
                text=True,
                cwd=str(self._project_path),
                timeout=5,
            )
            if result.returncode == 0:
                return len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0
        except Exception:
            pass
        return 0

    def _compute_hotspot(self, m: ModuleMetrics, max_change: int) -> float:
        """Compute hotspot score: 0.4*pagerank + 0.3*coupling_ratio + 0.3*change_freq_norm."""
        coupling_ratio = m.instability  # Ce / (Ca + Ce), already [0, 1]
        change_norm = m.change_frequency / max_change if max_change > 0 else 0.0
        return 0.4 * m.pagerank_score + 0.3 * coupling_ratio + 0.3 * change_norm

    def _compute_tdd_priority(self, m: ModuleMetrics, max_change: int) -> float:
        """Compute TDD priority score.

        Formula: 0.25*pagerank + 0.20*coupling + 0.20*change_freq
                 + 0.20*(1-coverage) + 0.15*(1-cohesion)
        """
        change_norm = m.change_frequency / max_change if max_change > 0 else 0.0
        coverage = self._get_test_coverage(m.module_path)
        return (
            0.25 * m.pagerank_score
            + 0.20 * m.instability
            + 0.20 * change_norm
            + 0.20 * (1.0 - coverage)
            + 0.15 * (1.0 - m.cohesion_lcom4)
        )

    def _get_test_coverage(self, module: str) -> float:
        """Read test coverage for a module from .coverage or coverage.json."""
        # Try coverage.json first
        coverage_json = self._project_path / "coverage.json"
        if coverage_json.exists():
            try:
                import json

                data = json.loads(coverage_json.read_text())
                files = data.get("files", {})
                for path, info in files.items():
                    if module in path:
                        summary = info.get("summary", {})
                        return summary.get("percent_covered", 0.0) / 100.0
            except Exception:
                pass

        # Try htmlcov/status.json
        status_json = self._project_path / "htmlcov" / "status.json"
        if status_json.exists():
            try:
                import json

                data = json.loads(status_json.read_text())
                files = data.get("files", {})
                for path, info in files.items():
                    if module in path:
                        return info.get("index", {}).get("pc_covered", 0.0) / 100.0
            except Exception:
                pass

        # Default: no coverage data available, assume 0
        return 0.0
