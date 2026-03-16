# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Graph insights, conversation insights, and embedding status queries.

Handles:
- Code graph analysis (design patterns, PageRank, hub detection)
- Conversation history mining
- Embedding/cache health inspection
- Analyzer coverage section building
"""

import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from victor.core.verticals.import_resolver import import_module_with_fallback

logger = logging.getLogger(__name__)


def _collect_embedding_status(root_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Summarize embedding/cache health for init.md enrichment."""
    root = Path(root_path).resolve() if root_path else Path.cwd()
    try:
        from victor.storage.cache.embedding_cache_manager import EmbeddingCacheManager
    except Exception as exc:  # pragma: no cover - optional dependency at runtime
        logger.debug("Embedding manager unavailable: %s", exc)
        return None

    try:
        status = EmbeddingCacheManager.get_instance().get_status()
    except Exception as exc:  # pragma: no cover - runtime I/O
        logger.debug("Failed to collect embedding status: %s", exc)
        return None

    caches: List[Dict[str, Any]] = []
    for cache in status.caches:
        caches.append(
            {
                "name": cache.name,
                "type": cache.cache_type.value,
                "files": cache.file_count,
                "size": cache.size_str,
                "age": cache.age_str,
            }
        )

    info: Dict[str, Any] = {
        "total_files": status.total_files,
        "total_size": status.total_size_str,
        "caches": caches,
    }

    # Optional: inspect code embedding store (LanceDB) for metadata richness
    try:
        from victor.config.settings import get_project_paths, load_settings

        settings = load_settings()
        default_dir = get_project_paths(root).embeddings_dir
        persist_dir = Path(getattr(settings, "codebase_persist_directory", None) or default_dir)

        import lancedb  # type: ignore

        if persist_dir.exists():
            db = lancedb.connect(str(persist_dir))
            table_names = db.list_tables().tables
            table_name = getattr(settings, "codebase_embedding_table", None) or (
                table_names[0] if table_names else None
            )
            if table_name:
                table = db.open_table(table_name)
                row_count = table.count_rows()

                sample_keys: Dict[str, int] = defaultdict(int)
                chunk_types: Set[str] = set()
                span_lengths: List[int] = []

                try:
                    sample_rows = table.head(min(50, row_count)).to_list()  # small sample
                except Exception:
                    sample_rows = []

                for row in sample_rows:
                    for key, value in row.items():
                        if key in {"id", "vector"}:
                            continue
                        if value not in (None, "", [], {}):
                            sample_keys[key] += 1
                        if key == "chunk_type" and isinstance(value, str):
                            chunk_types.add(value)
                        if key in {"line_start", "line_end"}:
                            try:
                                start = int(row.get("line_start", 0))
                                end = int(row.get("line_end", 0))
                                if start and end and end >= start:
                                    span_lengths.append(end - start + 1)
                            except Exception:
                                pass

                info["code_embeddings"] = {
                    "path": str(persist_dir),
                    "table": table_name,
                    "rows": row_count,
                    "metadata_keys": sorted(sample_keys.items(), key=lambda x: -x[1])[:8],
                    "chunk_types": sorted(chunk_types)[:6] if chunk_types else [],
                    "max_span": max(span_lengths) if span_lengths else None,
                }
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.debug("Skipping code embedding metadata inspection: %s", exc)

    return info


def _build_analyzer_section(
    stats: Dict[str, Any],
    graph_insights: Dict[str, Any],
    embedding_status: Optional[Dict[str, Any]],
) -> List[str]:
    """Construct analyzer coverage section from index + graph data."""
    if not stats:
        return []

    lines: List[str] = []
    lines.append("## Analyzer Coverage\n")

    total_symbols = stats.get("total_symbols", 0)
    total_files = stats.get("total_files", 0)
    lines.append(
        f"- **Symbol index**: {total_symbols} symbols across {total_files} files "
        "(multi-language tree-sitter + regex fallback, includes call sites and references)"
    )

    if graph_insights.get("has_graph"):
        graph_stats = graph_insights.get("stats", {})
        edge_types = graph_stats.get("edge_types", {})
        call_edges = edge_types.get("CALLS", 0)
        ref_edges = edge_types.get("REFERENCES", 0)
        import_edges = edge_types.get("IMPORTS", 0)
        inherits = edge_types.get("INHERITS", 0)
        comp = edge_types.get("COMPOSED_OF", 0)

        lines.append(
            f"- **Code graph**: {graph_stats.get('total_nodes', 0)} nodes, "
            f"{graph_stats.get('total_edges', 0)} edges "
            f"(calls: {call_edges}, refs: {ref_edges}, imports: {import_edges}, "
            f"inheritance: {inherits}, composition: {comp}); module PageRank and coupling detection ready"
        )

        hubs = graph_insights.get("hub_classes", [])[:2]
        if hubs:
            hub_preview = ", ".join(f"`{hub['name']}` ({hub['degree']} links)" for hub in hubs)
            lines.append(f"  - Hub classes: {hub_preview}")

        modules = graph_insights.get("important_modules", [])[:3]
        if modules:
            module_preview = ", ".join(f"`{mod['module']}` ({mod['role']})" for mod in modules)
            lines.append(f"  - Module PageRank leaders: {module_preview}")

        languages = graph_insights.get("languages", [])
        if languages:
            top_langs = ", ".join(f"{lang} ({count})" for lang, count in languages[:3])
            lines.append(f"  - Graph coverage by language: {top_langs}")

        if edge_types:
            missing_edges = graph_insights.get("edge_gaps", [])
            if missing_edges:
                lines.append(
                    f"  - Missing edge types: {', '.join(missing_edges)} (re-run index with tree-sitter deps to capture)"
                )
            elif call_edges == 0:
                lines.append(
                    "  - Calls not captured; verify tree-sitter call extraction is installed."
                )

        call_hotspots = graph_insights.get("call_hotspots", [])
        if call_hotspots:
            hot_preview = ", ".join(
                f"`{sym['name']}` ({sym['in_degree']} callers)" for sym in call_hotspots[:3]
            )
            lines.append(f"  - Call hotspots: {hot_preview}")
    else:
        lines.append(
            "- **Code graph**: not detected yet (run `victor index` to populate tree-sitter call/reference graph)"
        )

    if embedding_status and embedding_status.get("caches"):
        cache_preview = ", ".join(
            f"{cache['name']} ({cache['files']} files, {cache['size']}, {cache['age']})"
            for cache in embedding_status["caches"][:3]
        )
        lines.append(
            f"- **Semantic embeddings**: {cache_preview} "
            "(tree-sitter chunking for code spans; tool/intent/conversation caches ready)"
        )
        if embedding_status.get("code_embeddings"):
            ce = embedding_status["code_embeddings"]
            meta_keys = ", ".join(k for k, _ in ce.get("metadata_keys", [])) or "none"
            lines.append(
                f"  - Code embeddings: {ce.get('rows', 0)} vectors @ {ce.get('path')} "
                f"(metadata keys: {meta_keys})"
            )

    lines.append("")
    return lines


async def extract_conversation_insights(root_path: Optional[str] = None) -> Dict[str, Any]:
    """Extract insights from conversation history to enhance init.md.

    Analyzes stored conversations to identify:
    - Frequently asked questions/topics
    - Common file references
    - Learned patterns and hot spots
    """
    import sqlite3

    root = Path(root_path).resolve() if root_path else Path.cwd()
    db_path = root / ".victor" / "conversation.db"

    if not db_path.exists():
        return {"error": "No conversation history found"}

    insights = {
        "common_topics": [],
        "hot_files": [],
        "faq": [],
        "learned_patterns": [],
        "session_count": 0,
        "message_count": 0,
    }

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(DISTINCT session_id) FROM messages")
        insights["session_count"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM messages")
        insights["message_count"] = cursor.fetchone()[0]

        cursor.execute("""
            SELECT content FROM messages
            WHERE role = 'user'
              AND content NOT LIKE '%<TOOL_OUTPUT%'
              AND content NOT LIKE '%Complete this Python function%'
              AND content NOT LIKE '%Complete the following Python%'
              AND length(content) BETWEEN 20 AND 500
        """)

        queries = [row[0] for row in cursor.fetchall()]
        query_counter = Counter()

        topic_keywords = [
            "component",
            "architecture",
            "test",
            "bug",
            "fix",
            "add",
            "create",
            "refactor",
            "improve",
            "explain",
            "how",
            "what",
            "why",
            "where",
            "error",
            "issue",
            "implement",
            "feature",
            "config",
            "setup",
        ]

        for query in queries:
            query_lower = query.lower()
            for keyword in topic_keywords:
                if keyword in query_lower:
                    query_counter[keyword] += 1

        insights["common_topics"] = query_counter.most_common(10)

        cursor.execute("""
            SELECT content FROM messages
            WHERE role = 'assistant'
              AND (content LIKE '%.py%' OR content LIKE '%.ts%' OR content LIKE '%.js%')
        """)

        file_counter = Counter()
        file_pattern = re.compile(r"`([a-zA-Z_/]+\.(py|ts|js|go|rs))[:`]")

        for row in cursor.fetchall():
            matches = file_pattern.findall(row[0])
            for match in matches:
                file_path = match[0]
                if "/" in file_path and not file_path.startswith("/"):
                    file_counter[file_path] += 1

        insights["hot_files"] = file_counter.most_common(15)

        cursor.execute("""
            SELECT pattern_name, pattern_type, COUNT(*) as count
            FROM patterns
            GROUP BY pattern_type
            ORDER BY count DESC
        """)
        insights["learned_patterns"] = [
            {"name": row[0], "type": row[1], "count": row[2]} for row in cursor.fetchall()
        ]

        cursor.execute("""
            SELECT content, COUNT(*) as times
            FROM messages
            WHERE role = 'user'
              AND content LIKE '%?%'
              AND content NOT LIKE '%Complete%function%'
              AND length(content) BETWEEN 15 AND 200
            GROUP BY content
            HAVING times > 1
            ORDER BY times DESC
            LIMIT 5
        """)
        insights["faq"] = [{"question": row[0], "times_asked": row[1]} for row in cursor.fetchall()]

        conn.close()

    except Exception as e:
        insights["error"] = str(e)

    return insights


async def extract_graph_insights(root_path: Optional[str] = None) -> Dict[str, Any]:
    """Extract insights from the code graph for init.md enrichment.

    Analyzes the code graph to detect:
    - Design patterns (Provider, Factory, Facade, etc.)
    - Most important symbols (PageRank)
    - Hub classes (high centrality)
    - File dependencies
    - Graph statistics
    """
    from pathlib import Path
    from victor.tools.graph_tool import GraphAnalyzer, _load_graph
    from victor.core.schema import Tables

    root = Path(root_path).resolve() if root_path else Path.cwd()
    graph_db_path = root / ".victor" / "project.db"
    _NT = Tables.GRAPH_NODE
    _ET = Tables.GRAPH_EDGE

    insights: Dict[str, Any] = {
        "has_graph": False,
        "patterns": [],
        "important_symbols": [],
        "hub_classes": [],
        "stats": {},
        "important_modules": [],
        "module_coupling": [],
        "languages": [],
        "call_hotspots": [],
        "edge_gaps": [],
        "pagerank": [],
        "centrality": [],
        "components": [],
    }

    graph_registry_module, _resolved = import_module_with_fallback(
        "victor.coding.codebase.graph.registry"
    )
    if graph_registry_module is None or not hasattr(graph_registry_module, "create_graph_store"):
        insights["error"] = "Graph store backend is unavailable"
        return insights

    if not graph_db_path.exists():
        return insights

    try:
        import sqlite3
        import json

        conn = sqlite3.connect(graph_db_path)
        try:
            cur = conn.execute(f"SELECT COUNT(*) FROM {_NT}")
            total_nodes = cur.fetchone()[0]

            if total_nodes == 0:
                return insights

            cur = conn.execute(f"SELECT COUNT(*) FROM {_ET}")
            total_edges = cur.fetchone()[0]

            cur = conn.execute(f"SELECT type, COUNT(*) FROM {_NT} GROUP BY type")
            node_types = dict(cur.fetchall())

            cur = conn.execute(f"SELECT lang, COUNT(*) FROM {_NT} GROUP BY lang")
            languages = [(row[0], row[1]) for row in cur.fetchall() if row[0]]

            cur = conn.execute(f"SELECT type, COUNT(*) FROM {_ET} GROUP BY type")
            edge_types = dict(cur.fetchall())

            insights["has_graph"] = True
            insights["stats"] = {
                "total_nodes": total_nodes,
                "total_edges": total_edges,
                "node_types": node_types,
                "edge_types": edge_types,
            }
            insights["languages"] = languages
            expected_edges = {
                "CALLS",
                "REFERENCES",
                "IMPORTS",
                "INHERITS",
                "IMPLEMENTS",
                "COMPOSED_OF",
            }
            insights["edge_gaps"] = sorted(list(expected_edges - set(edge_types.keys())))

            cur = conn.execute(f"""
                SELECT n.name, n.type, n.file, n.line,
                       (SELECT COUNT(*) FROM {_ET} WHERE src = n.node_id) +
                       (SELECT COUNT(*) FROM {_ET} WHERE dst = n.node_id) as degree
                FROM {_NT} n
                WHERE n.type IN ('class', 'struct', 'interface')
                ORDER BY degree DESC
                LIMIT 5
            """)
            hub_results = cur.fetchall()
            insights["hub_classes"] = [
                {"name": r[0], "type": r[1], "file": r[2], "line": r[3], "degree": r[4]}
                for r in hub_results
                if r[4] >= 5
            ][:3]

            cur = conn.execute(f"""
                SELECT n.name, n.type, n.file, n.line,
                       (SELECT COUNT(*) FROM {_ET} WHERE dst = n.node_id AND type = 'CALLS') as in_calls,
                       (SELECT COUNT(*) FROM {_ET} WHERE src = n.node_id AND type = 'CALLS') as out_calls
                FROM {_NT} n
                WHERE n.type IN ('function', 'method', 'class')
                ORDER BY in_calls DESC
                LIMIT 8
            """)
            important_results = cur.fetchall()
            insights["important_symbols"] = [
                {
                    "name": r[0],
                    "type": r[1],
                    "file": r[2],
                    "line": r[3],
                    "in_degree": r[4],
                    "out_degree": r[5],
                    "score": r[4] / max(total_edges, 1),
                }
                for r in important_results
                if r[4] > 0
            ]
            insights["call_hotspots"] = insights["important_symbols"][:5]

            # Optional richer graph analytics
            try:
                from victor.tools.graph_tool import GraphAnalyzer
                from victor.storage.graph.sqlite_store import SqliteGraphStore

                ga = GraphAnalyzer()
                store = SqliteGraphStore(project_path=root)
                nodes = await store.get_all_nodes()
                edges = await store.get_all_edges()
                for n in nodes:
                    ga.add_node(n)
                for e in edges:
                    ga.add_edge(e)

                pagerank_top = ga.pagerank(
                    top_k=8,
                    edge_types=[
                        "CALLS",
                        "REFERENCES",
                        "INHERITS",
                        "IMPLEMENTS",
                        "COMPOSED_OF",
                        "IMPORTS",
                    ],
                )
                centrality_top = ga.degree_centrality(
                    top_k=8,
                    edge_types=[
                        "CALLS",
                        "REFERENCES",
                        "INHERITS",
                        "IMPLEMENTS",
                        "COMPOSED_OF",
                        "IMPORTS",
                    ],
                )

                visited: Set[str] = set()
                components: List[int] = []
                for node_id in ga.nodes:
                    if node_id in visited:
                        continue
                    stack = [node_id]
                    size = 0
                    while stack:
                        nid = stack.pop()
                        if nid in visited:
                            continue
                        visited.add(nid)
                        size += 1
                        neighbors = [t for t, _et, _w in ga.outgoing.get(nid, [])] + [
                            s for s, _et, _w in ga.incoming.get(nid, [])
                        ]
                        for n2 in neighbors:
                            if n2 not in visited:
                                stack.append(n2)
                    components.append(size)

                components.sort(reverse=True)

                insights["pagerank"] = pagerank_top[:5]
                insights["centrality"] = centrality_top[:5]
                insights["components"] = components[:3]
            except Exception as exc:
                logger.debug(f"Graph analytics fallback skipped: {exc}")

            # Module-level analysis
            cur = conn.execute(f"""
                SELECT
                    src_n.file as src_module,
                    dst_n.file as dst_module,
                    COUNT(*) as ref_count
                FROM {_ET} e
                JOIN {_NT} src_n ON e.src = src_n.node_id
                JOIN {_NT} dst_n ON e.dst = dst_n.node_id
                WHERE e.type = 'REFERENCES'
                  AND src_n.file != dst_n.file
                  AND src_n.file IS NOT NULL
                  AND dst_n.file IS NOT NULL
                  AND src_n.file NOT LIKE 'tests/%'
                  AND dst_n.file NOT LIKE 'tests/%'
                GROUP BY src_n.file, dst_n.file
                HAVING ref_count >= 2
                """)
            module_edges = cur.fetchall()

            if module_edges:
                module_in_degree: Dict[str, int] = defaultdict(int)
                module_out_degree: Dict[str, int] = defaultdict(int)
                module_weighted_in: Dict[str, int] = defaultdict(int)
                all_modules: Set[str] = set()

                for src_mod, dst_mod, count in module_edges:
                    all_modules.add(src_mod)
                    all_modules.add(dst_mod)
                    module_out_degree[src_mod] += 1
                    module_in_degree[dst_mod] += 1
                    module_weighted_in[dst_mod] += count

                module_importance = [
                    (mod, module_weighted_in[mod], module_in_degree[mod], module_out_degree[mod])
                    for mod in all_modules
                ]
                module_importance.sort(key=lambda x: x[1], reverse=True)

                insights["important_modules"] = []
                for mod, weighted_in, in_deg, out_deg in module_importance[:8]:
                    if in_deg > out_deg * 2 and in_deg >= 3:
                        role = "service"
                    elif out_deg > in_deg * 2 and out_deg >= 3:
                        role = "orchestrator"
                    elif in_deg >= 2 and out_deg >= 2:
                        role = "intermediary"
                    elif in_deg > 0 and out_deg == 0:
                        role = "leaf"
                    elif out_deg > 0 and in_deg == 0:
                        role = "entry"
                    else:
                        role = "peripheral"

                    insights["important_modules"].append(
                        {
                            "module": mod,
                            "weighted_importance": weighted_in,
                            "in_degree": in_deg,
                            "out_degree": out_deg,
                            "role": role,
                        }
                    )

                coupling_issues = []
                for mod, weighted_in, in_deg, out_deg in module_importance:
                    total_degree = in_deg + out_deg
                    if total_degree >= 8:
                        if in_deg > 5 and out_deg > 5:
                            pattern = "hub"
                        elif in_deg > 5:
                            pattern = "high_fan_in"
                        else:
                            pattern = "high_fan_out"
                        coupling_issues.append(
                            {
                                "module": mod,
                                "pattern": pattern,
                                "in_degree": in_deg,
                                "out_degree": out_deg,
                            }
                        )

                insights["module_coupling"] = coupling_issues[:5]

        finally:
            conn.close()

    except Exception as e:
        logger.warning(f"Failed to extract graph insights: {e}")

    return insights
