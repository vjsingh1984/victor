#!/usr/bin/env python3
"""
Comprehensive Graph Analysis for Victor Codebase.

Performs:
1. Connected component analysis
2. PageRank and centrality analysis
3. Dead code detection
4. Code coupling analysis
5. Duplication detection
6. Module dependency analysis
7. Graph metrics and topology analysis

Outputs detailed reports for manual verification.
"""

import sqlite3
import json
from collections import defaultdict
from pathlib import Path
from datetime import datetime
import networkx as nx
from typing import Dict, List, Set, Tuple, Any

DB_PATH = Path(__file__).parent.parent / ".victor" / "project.db"
OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "analysis_reports"


def load_graph_from_db() -> Tuple[Dict, List[Dict]]:
    """Load graph from SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    nodes = {}
    edges = []

    # Load nodes
    cursor = conn.execute("""
        SELECT node_id, type, name, file, line, end_line, signature, docstring, parent_id
        FROM graph_node
    """)
    for row in cursor:
        nodes[row['node_id']] = {
            'type': row['type'],
            'name': row['name'],
            'file': row['file'],
            'line': row['line'],
            'end_line': row['end_line'],
            'signature': row['signature'],
            'docstring': row['docstring'],
            'parent_id': row['parent_id']
        }

    # Load edges
    cursor = conn.execute("SELECT src, dst, type, weight FROM graph_edge")
    for row in cursor:
        edges.append({
            'src': row['src'],
            'dst': row['dst'],
            'type': row['type'],
            'weight': row['weight'] or 1.0
        })

    conn.close()
    return nodes, edges


def filter_victor_nodes(nodes: Dict, edges: List) -> Tuple[Dict, List]:
    """Filter to only victor codebase (exclude tests and external)."""
    victor_nodes = {}
    for k, v in nodes.items():
        if v['file'] and 'victor/' in v['file']:
            # Exclude test files
            if '/test' not in v['file'].lower() and 'test_' not in v['file'].lower():
                victor_nodes[k] = v

    victor_edges = [e for e in edges
                    if e['src'] in victor_nodes and e['dst'] in victor_nodes]

    return victor_nodes, victor_edges


def build_networkx_graph(nodes: Dict, edges: List, directed: bool = True) -> nx.DiGraph:
    """Build NetworkX graph from nodes and edges."""
    G = nx.DiGraph() if directed else nx.Graph()

    for node_id, node in nodes.items():
        G.add_node(node_id, **node)

    for edge in edges:
        if edge['src'] in nodes and edge['dst'] in nodes:
            G.add_edge(edge['src'], edge['dst'],
                      edge_type=edge['type'],
                      weight=edge['weight'])

    return G


def get_module_from_file(file_path: str) -> str:
    """Extract module name from file path."""
    if not file_path:
        return "unknown"
    path = Path(file_path)
    if "victor" in path.parts:
        victor_idx = path.parts.index("victor")
        module_parts = list(path.parts[victor_idx:])
        if module_parts[-1].endswith('.py'):
            module_parts[-1] = module_parts[-1][:-3]
        return ".".join(module_parts[:3])
    return str(path.parent.name) if path.parent.name else "root"


def get_submodule(file_path: str) -> str:
    """Get top-level submodule (e.g., victor.agent, victor.tools)."""
    if not file_path:
        return "unknown"
    path = Path(file_path)
    if "victor" in path.parts:
        victor_idx = path.parts.index("victor")
        if len(path.parts) > victor_idx + 1:
            return f"victor.{path.parts[victor_idx + 1]}"
    return "victor"


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_connected_components(G: nx.DiGraph) -> Dict:
    """Analyze connected components."""
    # Weakly connected components
    weak_components = list(nx.weakly_connected_components(G))
    weak_components.sort(key=len, reverse=True)

    # Strongly connected components
    strong_components = list(nx.strongly_connected_components(G))
    strong_components.sort(key=len, reverse=True)

    # Analyze component composition
    component_analysis = []
    for i, comp in enumerate(weak_components[:10]):
        modules = defaultdict(int)
        types = defaultdict(int)
        for node_id in comp:
            if node_id in G.nodes:
                node = G.nodes[node_id]
                modules[get_submodule(node.get('file', ''))] += 1
                types[node.get('type', 'unknown')] += 1

        component_analysis.append({
            'component_id': i + 1,
            'size': len(comp),
            'top_modules': dict(sorted(modules.items(), key=lambda x: x[1], reverse=True)[:5]),
            'node_types': dict(types)
        })

    return {
        'num_weak_components': len(weak_components),
        'num_strong_components': len(strong_components),
        'weak_component_sizes': [len(c) for c in weak_components[:20]],
        'strong_component_sizes': [len(c) for c in strong_components[:20]],
        'component_details': component_analysis,
        'largest_weak_component_ratio': len(weak_components[0]) / G.number_of_nodes() if weak_components else 0
    }


def analyze_centrality(G: nx.DiGraph) -> Dict:
    """Compute various centrality metrics."""
    results = {}

    # PageRank (most important)
    print("  Computing PageRank...")
    pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)
    top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:50]
    results['pagerank'] = [
        {
            'node_id': n,
            'name': G.nodes[n].get('name', ''),
            'type': G.nodes[n].get('type', ''),
            'module': get_module_from_file(G.nodes[n].get('file', '')),
            'score': round(s, 8)
        }
        for n, s in top_pagerank
    ]

    # In-degree centrality (most referenced)
    print("  Computing in-degree centrality...")
    in_degree = dict(G.in_degree())
    top_indegree = sorted(in_degree.items(), key=lambda x: x[1], reverse=True)[:50]
    results['most_referenced'] = [
        {
            'node_id': n,
            'name': G.nodes[n].get('name', ''),
            'type': G.nodes[n].get('type', ''),
            'module': get_module_from_file(G.nodes[n].get('file', '')),
            'in_degree': d
        }
        for n, d in top_indegree
    ]

    # Out-degree centrality (most referencing)
    print("  Computing out-degree centrality...")
    out_degree = dict(G.out_degree())
    top_outdegree = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:50]
    results['most_referencing'] = [
        {
            'node_id': n,
            'name': G.nodes[n].get('name', ''),
            'type': G.nodes[n].get('type', ''),
            'module': get_module_from_file(G.nodes[n].get('file', '')),
            'out_degree': d
        }
        for n, d in top_outdegree
    ]

    # Betweenness centrality (on smaller subgraph for performance)
    print("  Computing betweenness centrality (sampled)...")
    try:
        betweenness = nx.betweenness_centrality(G, k=min(500, G.number_of_nodes()))
        top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:30]
        results['betweenness'] = [
            {
                'node_id': n,
                'name': G.nodes[n].get('name', ''),
                'type': G.nodes[n].get('type', ''),
                'module': get_module_from_file(G.nodes[n].get('file', '')),
                'score': round(s, 8)
            }
            for n, s in top_betweenness
        ]
    except:
        results['betweenness'] = []

    return results


def analyze_dead_code(G: nx.DiGraph, nodes: Dict) -> Dict:
    """Detect potential dead code (unreferenced functions/classes)."""
    dead_code_candidates = []

    # Find nodes with no incoming edges (except from their parent/file)
    for node_id, node in nodes.items():
        if node['type'] in ('function', 'class'):
            in_edges = list(G.in_edges(node_id, data=True))

            # Filter out CONTAINS edges (parent-child relationships)
            meaningful_in_edges = [
                e for e in in_edges
                if e[2].get('edge_type') not in ('CONTAINS',)
            ]

            if len(meaningful_in_edges) == 0:
                # Check if it's a public API (doesn't start with _)
                is_private = node['name'].startswith('_')
                is_dunder = node['name'].startswith('__') and node['name'].endswith('__')

                dead_code_candidates.append({
                    'node_id': node_id,
                    'name': node['name'],
                    'type': node['type'],
                    'file': node['file'],
                    'line': node['line'],
                    'is_private': is_private,
                    'is_dunder': is_dunder,
                    'module': get_module_from_file(node['file']),
                    'has_docstring': bool(node.get('docstring')),
                    'severity': 'low' if is_private or is_dunder else 'medium'
                })

    # Sort by severity and module
    dead_code_candidates.sort(key=lambda x: (x['severity'], x['module']))

    # Categorize by module
    by_module = defaultdict(list)
    for item in dead_code_candidates:
        by_module[item['module']].append(item)

    return {
        'total_candidates': len(dead_code_candidates),
        'by_severity': {
            'medium': len([x for x in dead_code_candidates if x['severity'] == 'medium']),
            'low': len([x for x in dead_code_candidates if x['severity'] == 'low'])
        },
        'by_module': {m: len(v) for m, v in sorted(by_module.items(), key=lambda x: len(x[1]), reverse=True)},
        'candidates': dead_code_candidates[:100]  # Top 100 for report
    }


def analyze_code_coupling(G: nx.DiGraph, nodes: Dict) -> Dict:
    """Analyze code coupling between modules."""
    module_graph = nx.DiGraph()
    module_edges = defaultdict(lambda: defaultdict(int))

    # Build module-level graph
    for src, dst, data in G.edges(data=True):
        if src in nodes and dst in nodes:
            src_mod = get_submodule(nodes[src]['file'])
            dst_mod = get_submodule(nodes[dst]['file'])
            if src_mod != dst_mod:
                module_edges[src_mod][dst_mod] += 1

    # Build module graph
    for src_mod, dsts in module_edges.items():
        for dst_mod, count in dsts.items():
            module_graph.add_edge(src_mod, dst_mod, weight=count)

    # Calculate coupling metrics
    coupling_analysis = []
    for mod in module_graph.nodes():
        in_coupling = sum(d['weight'] for _, _, d in module_graph.in_edges(mod, data=True))
        out_coupling = sum(d['weight'] for _, _, d in module_graph.out_edges(mod, data=True))

        coupling_analysis.append({
            'module': mod,
            'afferent_coupling': in_coupling,  # incoming dependencies
            'efferent_coupling': out_coupling,  # outgoing dependencies
            'total_coupling': in_coupling + out_coupling,
            'instability': round(out_coupling / (in_coupling + out_coupling), 3) if (in_coupling + out_coupling) > 0 else 0
        })

    coupling_analysis.sort(key=lambda x: x['total_coupling'], reverse=True)

    # Find tightly coupled module pairs
    tight_coupling = []
    for src, dst, data in module_graph.edges(data=True):
        if data['weight'] >= 20:  # Significant coupling
            tight_coupling.append({
                'source': src,
                'target': dst,
                'edge_count': data['weight']
            })
    tight_coupling.sort(key=lambda x: x['edge_count'], reverse=True)

    # Circular dependencies
    cycles = list(nx.simple_cycles(module_graph))

    return {
        'module_coupling': coupling_analysis,
        'tight_coupling_pairs': tight_coupling[:30],
        'circular_dependencies': [list(c) for c in cycles[:20]],
        'module_graph_density': nx.density(module_graph),
        'module_count': module_graph.number_of_nodes()
    }


def analyze_duplications(G: nx.DiGraph, nodes: Dict) -> Dict:
    """Detect potential code duplications."""
    duplicates = []

    # 1. Name-based duplicates
    by_name_type = defaultdict(list)
    for node_id, node in nodes.items():
        if node['type'] in ('function', 'class'):
            key = (node['name'], node['type'])
            by_name_type[key].append((node_id, node))

    name_duplicates = []
    for (name, typ), node_list in by_name_type.items():
        if len(node_list) > 1:
            files = set(n['file'] for _, n in node_list)
            if len(files) > 1:  # Same name in different files
                name_duplicates.append({
                    'name': name,
                    'type': typ,
                    'count': len(node_list),
                    'locations': [
                        {
                            'file': n['file'],
                            'line': n['line'],
                            'module': get_module_from_file(n['file']),
                            'signature': n.get('signature', '')[:100]
                        }
                        for _, n in node_list
                    ]
                })

    name_duplicates.sort(key=lambda x: x['count'], reverse=True)

    # 2. Signature-based duplicates (similar function signatures)
    by_signature = defaultdict(list)
    for node_id, node in nodes.items():
        if node['type'] == 'function' and node.get('signature'):
            # Normalize signature
            sig = node['signature'].replace(' ', '').lower()
            if len(sig) > 20:  # Only significant signatures
                by_signature[sig].append((node_id, node))

    signature_duplicates = []
    for sig, node_list in by_signature.items():
        if len(node_list) > 1:
            files = set(n['file'] for _, n in node_list)
            if len(files) > 1:
                signature_duplicates.append({
                    'signature': sig[:80],
                    'count': len(node_list),
                    'locations': [
                        {
                            'name': n['name'],
                            'file': n['file'],
                            'line': n['line'],
                            'module': get_module_from_file(n['file'])
                        }
                        for _, n in node_list[:5]
                    ]
                })

    signature_duplicates.sort(key=lambda x: x['count'], reverse=True)

    # 3. Similar connection patterns
    connection_patterns = defaultdict(list)
    for node_id in G.nodes():
        if nodes.get(node_id, {}).get('type') == 'function':
            # Get sorted list of called functions
            called = tuple(sorted(
                nodes.get(dst, {}).get('name', '')
                for _, dst in G.out_edges(node_id)
                if nodes.get(dst, {}).get('type') == 'function'
            ))
            if len(called) >= 3:
                connection_patterns[called].append(node_id)

    similar_patterns = []
    for pattern, node_ids in connection_patterns.items():
        if len(node_ids) > 1:
            similar_patterns.append({
                'pattern_size': len(pattern),
                'functions': [
                    {
                        'name': nodes[nid]['name'],
                        'file': nodes[nid]['file'],
                        'module': get_module_from_file(nodes[nid]['file'])
                    }
                    for nid in node_ids[:5]
                ],
                'shared_calls': list(pattern)[:10]
            })

    similar_patterns.sort(key=lambda x: (len(x['functions']), x['pattern_size']), reverse=True)

    return {
        'name_duplicates': name_duplicates[:50],
        'signature_duplicates': signature_duplicates[:30],
        'similar_call_patterns': similar_patterns[:30],
        'summary': {
            'total_name_duplicates': len(name_duplicates),
            'total_signature_duplicates': len(signature_duplicates),
            'total_pattern_duplicates': len(similar_patterns)
        }
    }


def analyze_graph_metrics(G: nx.DiGraph) -> Dict:
    """Compute overall graph metrics."""
    metrics = {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'density': round(nx.density(G), 6),
        'avg_in_degree': round(sum(d for _, d in G.in_degree()) / G.number_of_nodes(), 2),
        'avg_out_degree': round(sum(d for _, d in G.out_degree()) / G.number_of_nodes(), 2),
    }

    # Node type distribution
    type_dist = defaultdict(int)
    for node_id in G.nodes():
        type_dist[G.nodes[node_id].get('type', 'unknown')] += 1
    metrics['node_type_distribution'] = dict(type_dist)

    # Edge type distribution
    edge_type_dist = defaultdict(int)
    for _, _, data in G.edges(data=True):
        edge_type_dist[data.get('edge_type', 'unknown')] += 1
    metrics['edge_type_distribution'] = dict(edge_type_dist)

    # Module distribution
    module_dist = defaultdict(int)
    for node_id in G.nodes():
        mod = get_submodule(G.nodes[node_id].get('file', ''))
        module_dist[mod] += 1
    metrics['module_distribution'] = dict(sorted(module_dist.items(), key=lambda x: x[1], reverse=True))

    return metrics


def generate_module_dependency_data(G: nx.DiGraph, nodes: Dict) -> Dict:
    """Generate data for module dependency diagrams."""
    module_edges = defaultdict(lambda: defaultdict(int))
    module_nodes = defaultdict(set)

    for node_id, node in nodes.items():
        mod = get_submodule(node['file'])
        module_nodes[mod].add(node_id)

    for src, dst, data in G.edges(data=True):
        if src in nodes and dst in nodes:
            src_mod = get_submodule(nodes[src]['file'])
            dst_mod = get_submodule(nodes[dst]['file'])
            edge_type = data.get('edge_type', 'UNKNOWN')
            if src_mod != dst_mod:
                module_edges[(src_mod, dst_mod)][edge_type] += 1

    # Format for diagram generation
    dependencies = []
    for (src, dst), edge_types in module_edges.items():
        total = sum(edge_types.values())
        if total >= 5:  # Only significant edges
            dependencies.append({
                'source': src,
                'target': dst,
                'total': total,
                'by_type': dict(edge_types)
            })

    dependencies.sort(key=lambda x: x['total'], reverse=True)

    return {
        'modules': [
            {'name': m, 'size': len(n)}
            for m, n in sorted(module_nodes.items(), key=lambda x: len(x[1]), reverse=True)
        ],
        'dependencies': dependencies
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*70)
    print("COMPREHENSIVE GRAPH ANALYSIS FOR VICTOR CODEBASE")
    print(f"Started: {datetime.now().isoformat()}")
    print("="*70)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n1. Loading graph from database...")
    nodes, edges = load_graph_from_db()
    print(f"   Loaded {len(nodes)} nodes and {len(edges)} edges")

    print("\n2. Filtering to Victor codebase...")
    victor_nodes, victor_edges = filter_victor_nodes(nodes, edges)
    print(f"   Victor codebase: {len(victor_nodes)} nodes, {len(victor_edges)} edges")

    print("\n3. Building NetworkX graph...")
    G = build_networkx_graph(victor_nodes, victor_edges)
    print(f"   Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Run all analyses
    results = {}

    print("\n4. Analyzing graph metrics...")
    results['graph_metrics'] = analyze_graph_metrics(G)

    print("\n5. Analyzing connected components...")
    results['connected_components'] = analyze_connected_components(G)

    print("\n6. Analyzing centrality metrics...")
    results['centrality'] = analyze_centrality(G)

    print("\n7. Analyzing dead code candidates...")
    results['dead_code'] = analyze_dead_code(G, victor_nodes)

    print("\n8. Analyzing code coupling...")
    results['coupling'] = analyze_code_coupling(G, victor_nodes)

    print("\n9. Analyzing duplications...")
    results['duplications'] = analyze_duplications(G, victor_nodes)

    print("\n10. Generating module dependency data...")
    results['module_dependencies'] = generate_module_dependency_data(G, victor_nodes)

    # Save results
    results['metadata'] = {
        'generated_at': datetime.now().isoformat(),
        'total_nodes': len(victor_nodes),
        'total_edges': len(victor_edges)
    }

    output_file = OUTPUT_DIR / "graph_analysis_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n\nResults saved to: {output_file}")

    # Generate summary report
    generate_summary_report(results)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


def generate_summary_report(results: Dict):
    """Generate a human-readable summary report."""
    report_file = OUTPUT_DIR / "ANALYSIS_SUMMARY_REPORT.md"

    with open(report_file, 'w') as f:
        f.write("# Victor Codebase Graph Analysis Report\n\n")
        f.write(f"Generated: {results['metadata']['generated_at']}\n\n")

        f.write("## Executive Summary\n\n")
        metrics = results['graph_metrics']
        f.write(f"- **Total Nodes**: {metrics['nodes']:,}\n")
        f.write(f"- **Total Edges**: {metrics['edges']:,}\n")
        f.write(f"- **Graph Density**: {metrics['density']:.6f}\n")
        f.write(f"- **Average In-Degree**: {metrics['avg_in_degree']:.2f}\n")
        f.write(f"- **Average Out-Degree**: {metrics['avg_out_degree']:.2f}\n\n")

        f.write("### Node Type Distribution\n\n")
        f.write("| Type | Count |\n|------|-------|\n")
        for typ, count in sorted(metrics['node_type_distribution'].items(), key=lambda x: x[1], reverse=True):
            f.write(f"| {typ} | {count:,} |\n")

        f.write("\n### Module Distribution\n\n")
        f.write("| Module | Nodes |\n|--------|-------|\n")
        for mod, count in list(metrics['module_distribution'].items())[:15]:
            f.write(f"| {mod} | {count:,} |\n")

        # Connected Components
        f.write("\n## Connected Components\n\n")
        cc = results['connected_components']
        f.write(f"- **Weakly Connected Components**: {cc['num_weak_components']}\n")
        f.write(f"- **Strongly Connected Components**: {cc['num_strong_components']}\n")
        f.write(f"- **Main Component Coverage**: {cc['largest_weak_component_ratio']*100:.1f}%\n\n")

        f.write("### Top Components\n\n")
        for comp in cc['component_details'][:5]:
            f.write(f"**Component {comp['component_id']}** ({comp['size']:,} nodes)\n")
            f.write(f"- Top modules: {comp['top_modules']}\n")
            f.write(f"- Node types: {comp['node_types']}\n\n")

        # Centrality
        f.write("## Central Nodes (PageRank)\n\n")
        f.write("Most important nodes by PageRank:\n\n")
        f.write("| Rank | Name | Type | Module | Score |\n")
        f.write("|------|------|------|--------|-------|\n")
        for i, node in enumerate(results['centrality']['pagerank'][:20], 1):
            f.write(f"| {i} | {node['name'][:30]} | {node['type']} | {node['module']} | {node['score']:.6f} |\n")

        # Dead Code
        f.write("\n## Dead Code Analysis\n\n")
        dc = results['dead_code']
        f.write(f"**Total Candidates**: {dc['total_candidates']}\n\n")
        f.write("| Severity | Count |\n|----------|-------|\n")
        for sev, count in dc['by_severity'].items():
            f.write(f"| {sev} | {count} |\n")

        f.write("\n### By Module\n\n")
        f.write("| Module | Dead Code Candidates |\n|--------|----------------------|\n")
        for mod, count in list(dc['by_module'].items())[:15]:
            f.write(f"| {mod} | {count} |\n")

        f.write("\n### Sample Dead Code Candidates (Medium Severity)\n\n")
        f.write("| Name | Type | File | Line |\n")
        f.write("|------|------|------|------|\n")
        medium_candidates = [c for c in dc['candidates'] if c['severity'] == 'medium'][:30]
        for c in medium_candidates:
            short_file = '/'.join(Path(c['file']).parts[-3:]) if c['file'] else 'unknown'
            f.write(f"| {c['name'][:30]} | {c['type']} | {short_file} | {c['line']} |\n")

        # Coupling
        f.write("\n## Code Coupling Analysis\n\n")
        coupling = results['coupling']
        f.write(f"- **Module Graph Density**: {coupling['module_graph_density']:.4f}\n")
        f.write(f"- **Total Modules**: {coupling['module_count']}\n")
        f.write(f"- **Circular Dependencies**: {len(coupling['circular_dependencies'])}\n\n")

        f.write("### Module Coupling Metrics\n\n")
        f.write("| Module | Afferent | Efferent | Total | Instability |\n")
        f.write("|--------|----------|----------|-------|-------------|\n")
        for m in coupling['module_coupling'][:15]:
            f.write(f"| {m['module']} | {m['afferent_coupling']} | {m['efferent_coupling']} | {m['total_coupling']} | {m['instability']:.2f} |\n")

        f.write("\n### Tightly Coupled Module Pairs\n\n")
        f.write("| Source | Target | Edge Count |\n")
        f.write("|--------|--------|------------|\n")
        for pair in coupling['tight_coupling_pairs'][:20]:
            f.write(f"| {pair['source']} | {pair['target']} | {pair['edge_count']} |\n")

        if coupling['circular_dependencies']:
            f.write("\n### Circular Dependencies\n\n")
            for cycle in coupling['circular_dependencies'][:10]:
                f.write(f"- {' -> '.join(cycle)} -> {cycle[0]}\n")

        # Duplications
        f.write("\n## Duplication Analysis\n\n")
        dup = results['duplications']
        f.write(f"- **Name Duplicates**: {dup['summary']['total_name_duplicates']}\n")
        f.write(f"- **Signature Duplicates**: {dup['summary']['total_signature_duplicates']}\n")
        f.write(f"- **Similar Call Patterns**: {dup['summary']['total_pattern_duplicates']}\n\n")

        f.write("### Name Duplicates (Potential Consolidation)\n\n")
        for d in dup['name_duplicates'][:20]:
            f.write(f"**{d['type']} `{d['name']}`** ({d['count']} occurrences)\n")
            for loc in d['locations'][:3]:
                short_file = '/'.join(Path(loc['file']).parts[-3:]) if loc['file'] else 'unknown'
                f.write(f"  - {short_file}:{loc['line']}\n")
            f.write("\n")

        f.write("### Similar Call Patterns (Possible Abstraction Candidates)\n\n")
        for p in dup['similar_call_patterns'][:10]:
            f.write(f"**{len(p['functions'])} functions share {p['pattern_size']} calls**\n")
            for func in p['functions'][:3]:
                f.write(f"  - `{func['name']}` in {func['module']}\n")
            f.write(f"  - Shared calls: {', '.join(p['shared_calls'][:5])}\n\n")

        f.write("\n---\n\n")
        f.write("*This report is for manual verification. Please review each item carefully before taking action.*\n")

    print(f"Summary report saved to: {report_file}")


if __name__ == "__main__":
    main()
