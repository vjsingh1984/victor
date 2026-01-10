#!/usr/bin/env python3
"""
Graph analysis script for Victor codebase.
Performs connected component analysis, PageRank, and duplication detection.
"""

import sqlite3
import json
from collections import defaultdict
from pathlib import Path

# Using only stdlib for portability
import heapq

DB_PATH = Path(__file__).parent.parent / ".victor" / "project.db"

def load_graph():
    """Load graph from SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    nodes = {}
    edges = []

    # Load nodes
    cursor = conn.execute("""
        SELECT node_id, type, name, file, line, signature, parent_id
        FROM graph_node
    """)
    for row in cursor:
        nodes[row['node_id']] = {
            'type': row['type'],
            'name': row['name'],
            'file': row['file'],
            'line': row['line'],
            'signature': row['signature'],
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


def build_adjacency_list(nodes, edges, edge_types=None):
    """Build adjacency list from nodes and edges."""
    adj = defaultdict(set)
    adj_reverse = defaultdict(set)

    for edge in edges:
        if edge_types and edge['type'] not in edge_types:
            continue
        if edge['src'] in nodes and edge['dst'] in nodes:
            adj[edge['src']].add(edge['dst'])
            adj_reverse[edge['dst']].add(edge['src'])

    return adj, adj_reverse


def find_connected_components(nodes, adj, adj_reverse):
    """Find weakly connected components using BFS."""
    visited = set()
    components = []

    # Build undirected adjacency for weak connectivity
    undirected = defaultdict(set)
    for src, dsts in adj.items():
        for dst in dsts:
            undirected[src].add(dst)
            undirected[dst].add(src)

    for node in nodes:
        if node not in visited:
            component = set()
            queue = [node]
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                component.add(current)
                for neighbor in undirected[current]:
                    if neighbor not in visited:
                        queue.append(neighbor)
            if component:
                components.append(component)

    return sorted(components, key=len, reverse=True)


def compute_pagerank(nodes, adj, damping=0.85, iterations=100):
    """Compute PageRank scores."""
    n = len(nodes)
    if n == 0:
        return {}

    node_list = list(nodes.keys())
    node_idx = {node: i for i, node in enumerate(node_list)}

    # Initialize PageRank
    pr = {node: 1.0 / n for node in node_list}

    for _ in range(iterations):
        new_pr = {}
        for node in node_list:
            rank = (1 - damping) / n
            # Sum contributions from incoming edges
            for other in node_list:
                if node in adj.get(other, set()):
                    out_degree = len(adj.get(other, set()))
                    if out_degree > 0:
                        rank += damping * pr[other] / out_degree
            new_pr[node] = rank
        pr = new_pr

    return pr


def get_module_from_file(file_path):
    """Extract module name from file path."""
    if not file_path:
        return "unknown"
    path = Path(file_path)
    if "victor" in path.parts:
        victor_idx = path.parts.index("victor")
        module_parts = list(path.parts[victor_idx:])
        if module_parts[-1].endswith('.py'):
            module_parts[-1] = module_parts[-1][:-3]
        return ".".join(module_parts[:3])  # Top 2-3 levels
    return str(path.parent.name) if path.parent.name else "root"


def analyze_module_dependencies(nodes, edges):
    """Analyze module-level dependencies."""
    module_nodes = defaultdict(set)
    module_edges = defaultdict(lambda: defaultdict(int))

    # Group nodes by module
    for node_id, node in nodes.items():
        module = get_module_from_file(node['file'])
        module_nodes[module].add(node_id)

    # Count cross-module edges
    for edge in edges:
        if edge['src'] in nodes and edge['dst'] in nodes:
            src_module = get_module_from_file(nodes[edge['src']]['file'])
            dst_module = get_module_from_file(nodes[edge['dst']]['file'])
            if src_module != dst_module:
                module_edges[src_module][dst_module] += 1

    return module_nodes, module_edges


def find_potential_duplicates(nodes, edges):
    """Find nodes with similar signatures/names that might be duplicates."""
    duplicates = []

    # Group by name and type
    by_name_type = defaultdict(list)
    for node_id, node in nodes.items():
        if node['type'] in ('function', 'class'):
            key = (node['name'], node['type'])
            by_name_type[key].append((node_id, node))

    # Find duplicates
    for (name, typ), node_list in by_name_type.items():
        if len(node_list) > 1:
            files = set()
            for node_id, node in node_list:
                files.add(node['file'])
            if len(files) > 1:  # Same name in different files
                duplicates.append({
                    'name': name,
                    'type': typ,
                    'count': len(node_list),
                    'locations': [(n['file'], n['line']) for _, n in node_list]
                })

    return sorted(duplicates, key=lambda x: x['count'], reverse=True)


def find_similar_connections(nodes, adj):
    """Find nodes with very similar connection patterns (potential duplication)."""
    # Group nodes by their connection signature
    connection_signatures = defaultdict(list)

    for node_id, neighbors in adj.items():
        if node_id in nodes and nodes[node_id]['type'] == 'function':
            # Create signature from sorted neighbor names
            neighbor_names = tuple(sorted(
                nodes.get(n, {}).get('name', '') for n in neighbors if n in nodes
            ))
            if len(neighbor_names) >= 3:  # Only consider nodes with 3+ connections
                connection_signatures[neighbor_names].append(node_id)

    # Find groups with similar connections
    similar_groups = []
    for sig, node_ids in connection_signatures.items():
        if len(node_ids) > 1:
            similar_groups.append({
                'nodes': [(nodes[nid]['name'], nodes[nid]['file']) for nid in node_ids],
                'shared_connections': len(sig),
                'connection_sample': list(sig)[:5]
            })

    return sorted(similar_groups, key=lambda x: (len(x['nodes']), x['shared_connections']), reverse=True)[:50]


def main():
    print("Loading graph from database...")
    nodes, edges = load_graph()
    print(f"Loaded {len(nodes)} nodes and {len(edges)} edges")

    # Filter to victor codebase only
    victor_nodes = {k: v for k, v in nodes.items()
                    if v['file'] and 'victor/' in v['file'] and 'test' not in v['file'].lower()}
    victor_edges = [e for e in edges
                    if e['src'] in victor_nodes and e['dst'] in victor_nodes]
    print(f"Victor codebase: {len(victor_nodes)} nodes, {len(victor_edges)} edges")

    print("\n" + "="*60)
    print("1. CONNECTED COMPONENT ANALYSIS")
    print("="*60)

    # Build adjacency for all edge types
    adj, adj_rev = build_adjacency_list(victor_nodes, victor_edges)

    # Find connected components
    components = find_connected_components(victor_nodes, adj, adj_rev)
    print(f"\nFound {len(components)} connected components")

    # Analyze top components
    for i, comp in enumerate(components[:10]):
        modules = defaultdict(int)
        for node_id in comp:
            if node_id in victor_nodes:
                module = get_module_from_file(victor_nodes[node_id]['file'])
                modules[module] += 1
        top_modules = sorted(modules.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\nComponent {i+1}: {len(comp)} nodes")
        print(f"  Top modules: {top_modules}")

    print("\n" + "="*60)
    print("2. MODULE-LEVEL DEPENDENCY ANALYSIS")
    print("="*60)

    module_nodes, module_edges = analyze_module_dependencies(victor_nodes, victor_edges)

    print(f"\nFound {len(module_nodes)} modules")
    top_modules = sorted(module_nodes.items(), key=lambda x: len(x[1]), reverse=True)[:20]
    print("\nTop modules by size:")
    for module, node_set in top_modules:
        print(f"  {module}: {len(node_set)} nodes")

    # Find modules with most cross-module dependencies
    print("\nModules with most outgoing cross-module edges:")
    module_out_degree = [(m, sum(deps.values())) for m, deps in module_edges.items()]
    for module, count in sorted(module_out_degree, key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {module}: {count} outgoing edges")

    print("\n" + "="*60)
    print("3. PAGERANK ANALYSIS (Central Nodes)")
    print("="*60)

    # Compute PageRank for the main component
    if components:
        main_component_nodes = {k: victor_nodes[k] for k in components[0] if k in victor_nodes}
        main_adj, _ = build_adjacency_list(main_component_nodes, victor_edges)

        print(f"\nComputing PageRank for main component ({len(main_component_nodes)} nodes)...")
        pr = compute_pagerank(main_component_nodes, main_adj, iterations=50)

        # Top PageRank nodes
        top_pr = heapq.nlargest(30, pr.items(), key=lambda x: x[1])
        print("\nTop 30 nodes by PageRank (most central):")
        for node_id, score in top_pr:
            node = victor_nodes[node_id]
            print(f"  {score:.6f} | {node['type']:10} | {node['name'][:40]:40} | {get_module_from_file(node['file'])}")

    print("\n" + "="*60)
    print("4. DUPLICATE ANALYSIS")
    print("="*60)

    print("\n4a. Potential Name Duplicates:")
    duplicates = find_potential_duplicates(victor_nodes, victor_edges)
    print(f"Found {len(duplicates)} potential duplicate names")
    for dup in duplicates[:20]:
        print(f"\n  {dup['type']} '{dup['name']}' appears {dup['count']} times:")
        for file, line in dup['locations'][:5]:
            short_file = '/'.join(Path(file).parts[-3:]) if file else 'unknown'
            print(f"    - {short_file}:{line}")

    print("\n4b. Similar Connection Patterns (potential code duplication):")
    similar = find_similar_connections(victor_nodes, adj)
    print(f"Found {len(similar)} groups with similar connections")
    for group in similar[:10]:
        print(f"\n  {len(group['nodes'])} functions with {group['shared_connections']} shared connections:")
        for name, file in group['nodes'][:3]:
            short_file = '/'.join(Path(file).parts[-3:]) if file else 'unknown'
            print(f"    - {name} in {short_file}")

    print("\n" + "="*60)
    print("5. EDGE ANALYSIS (Connection Patterns)")
    print("="*60)

    # Analyze edge density between modules
    print("\nHighest edge density between modules:")
    module_pairs = []
    for src_mod, dsts in module_edges.items():
        for dst_mod, count in dsts.items():
            if count >= 10:  # Only significant connections
                module_pairs.append((src_mod, dst_mod, count))

    for src, dst, count in sorted(module_pairs, key=lambda x: x[2], reverse=True)[:20]:
        print(f"  {src} -> {dst}: {count} edges")

    # Export summary for diagram generation
    summary = {
        'total_nodes': len(victor_nodes),
        'total_edges': len(victor_edges),
        'num_components': len(components),
        'component_sizes': [len(c) for c in components[:10]],
        'top_modules': [(m, len(n)) for m, n in top_modules],
        'module_dependencies': [(s, d, c) for s, d, c in sorted(module_pairs, key=lambda x: x[2], reverse=True)[:50]],
        'top_pagerank': [(victor_nodes[n]['name'], victor_nodes[n]['type'],
                          get_module_from_file(victor_nodes[n]['file']), s)
                         for n, s in top_pr[:20]] if components else [],
        'duplicates': [{'name': d['name'], 'type': d['type'], 'count': d['count']}
                       for d in duplicates[:30]]
    }

    output_path = Path(__file__).parent / "graph_analysis_results.json"
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
