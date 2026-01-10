#!/usr/bin/env python3
"""
Incremental Graph Analysis - prints results as it progresses.
"""

import sqlite3
import json
from collections import defaultdict
from pathlib import Path
from datetime import datetime
import networkx as nx

DB_PATH = Path(__file__).parent.parent / ".victor" / "project.db"
OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "analysis_reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_graph():
    """Load graph from SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    nodes = {}
    edges = []

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
            'signature': row['signature'],
            'docstring': row['docstring'],
            'parent_id': row['parent_id']
        }

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


def get_submodule(file_path):
    if not file_path:
        return "unknown"
    path = Path(file_path)
    if "victor" in path.parts:
        victor_idx = path.parts.index("victor")
        if len(path.parts) > victor_idx + 1:
            return f"victor.{path.parts[victor_idx + 1]}"
    return "victor"


def get_module(file_path):
    if not file_path:
        return "unknown"
    path = Path(file_path)
    if "victor" in path.parts:
        victor_idx = path.parts.index("victor")
        parts = list(path.parts[victor_idx:])
        if parts[-1].endswith('.py'):
            parts[-1] = parts[-1][:-3]
        return ".".join(parts[:3])
    return "unknown"


print("="*70)
print("INCREMENTAL GRAPH ANALYSIS")
print(f"Started: {datetime.now()}")
print("="*70)

# Load data
print("\n[1] LOADING DATA...")
nodes, edges = load_graph()
print(f"    Total nodes: {len(nodes):,}")
print(f"    Total edges: {len(edges):,}")

# Filter to victor
victor_nodes = {k: v for k, v in nodes.items()
                if v['file'] and 'victor/' in v['file'] and '/test' not in v['file'].lower()}
victor_edges = [e for e in edges if e['src'] in victor_nodes and e['dst'] in victor_nodes]
print(f"    Victor nodes: {len(victor_nodes):,}")
print(f"    Victor edges: {len(victor_edges):,}")

# Build graph
G = nx.DiGraph()
for node_id, node in victor_nodes.items():
    G.add_node(node_id, **node)
for edge in victor_edges:
    if edge['src'] in victor_nodes and edge['dst'] in victor_nodes:
        G.add_edge(edge['src'], edge['dst'], edge_type=edge['type'], weight=edge['weight'])

print(f"    NetworkX graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

# ============================================================================
print("\n" + "="*70)
print("[2] GRAPH METRICS")
print("="*70)

type_dist = defaultdict(int)
for nid in G.nodes():
    type_dist[G.nodes[nid].get('type', 'unknown')] += 1

print("\n  Node Type Distribution:")
for typ, count in sorted(type_dist.items(), key=lambda x: x[1], reverse=True):
    print(f"    {typ:15}: {count:,}")

edge_type_dist = defaultdict(int)
for _, _, data in G.edges(data=True):
    edge_type_dist[data.get('edge_type', 'unknown')] += 1

print("\n  Edge Type Distribution:")
for typ, count in sorted(edge_type_dist.items(), key=lambda x: x[1], reverse=True):
    print(f"    {typ:15}: {count:,}")

module_dist = defaultdict(int)
for nid in G.nodes():
    mod = get_submodule(G.nodes[nid].get('file', ''))
    module_dist[mod] += 1

print("\n  Module Distribution:")
for mod, count in sorted(module_dist.items(), key=lambda x: x[1], reverse=True)[:15]:
    print(f"    {mod:25}: {count:,}")

# Save metrics
metrics = {
    'nodes': G.number_of_nodes(),
    'edges': G.number_of_edges(),
    'density': nx.density(G),
    'node_types': dict(type_dist),
    'edge_types': dict(edge_type_dist),
    'modules': dict(module_dist)
}
with open(OUTPUT_DIR / "01_metrics.json", 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"\n  >> Saved to 01_metrics.json")

# ============================================================================
print("\n" + "="*70)
print("[3] CONNECTED COMPONENTS")
print("="*70)

weak_comps = list(nx.weakly_connected_components(G))
weak_comps.sort(key=len, reverse=True)
strong_comps = list(nx.strongly_connected_components(G))
strong_comps.sort(key=len, reverse=True)

print(f"\n  Weakly connected components: {len(weak_comps)}")
print(f"  Strongly connected components: {len(strong_comps)}")
print(f"\n  Top 10 weak component sizes: {[len(c) for c in weak_comps[:10]]}")
print(f"  Largest component coverage: {len(weak_comps[0])/G.number_of_nodes()*100:.1f}%")

# Analyze top components
comp_details = []
print("\n  Component Analysis:")
for i, comp in enumerate(weak_comps[:5]):
    modules = defaultdict(int)
    types = defaultdict(int)
    for nid in comp:
        if nid in G.nodes:
            modules[get_submodule(G.nodes[nid].get('file', ''))] += 1
            types[G.nodes[nid].get('type', 'unknown')] += 1

    top_mods = sorted(modules.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"\n    Component {i+1} ({len(comp):,} nodes):")
    print(f"      Types: {dict(types)}")
    print(f"      Top modules: {dict(top_mods)}")

    comp_details.append({
        'id': i+1,
        'size': len(comp),
        'modules': dict(modules),
        'types': dict(types)
    })

with open(OUTPUT_DIR / "02_components.json", 'w') as f:
    json.dump({
        'weak_count': len(weak_comps),
        'strong_count': len(strong_comps),
        'weak_sizes': [len(c) for c in weak_comps[:20]],
        'strong_sizes': [len(c) for c in strong_comps[:20]],
        'details': comp_details
    }, f, indent=2)
print(f"\n  >> Saved to 02_components.json")

# ============================================================================
print("\n" + "="*70)
print("[4] CENTRALITY - PageRank (Top 30 Central Nodes)")
print("="*70)

print("\n  Computing PageRank...")
pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)
top_pr = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:50]

print("\n  Rank | Score      | Type       | Name                             | Module")
print("  " + "-"*90)
for i, (nid, score) in enumerate(top_pr[:30], 1):
    node = G.nodes[nid]
    name = node.get('name', '')[:30]
    typ = node.get('type', '')[:10]
    mod = get_module(node.get('file', ''))
    print(f"  {i:4} | {score:.8f} | {typ:10} | {name:32} | {mod}")

with open(OUTPUT_DIR / "03_pagerank.json", 'w') as f:
    json.dump([
        {'rank': i+1, 'node_id': nid, 'score': score,
         'name': G.nodes[nid].get('name', ''),
         'type': G.nodes[nid].get('type', ''),
         'module': get_module(G.nodes[nid].get('file', ''))}
        for i, (nid, score) in enumerate(top_pr)
    ], f, indent=2)
print(f"\n  >> Saved to 03_pagerank.json")

# ============================================================================
print("\n" + "="*70)
print("[5] MOST REFERENCED (In-Degree) - Top 30")
print("="*70)

in_degree = sorted(G.in_degree(), key=lambda x: x[1], reverse=True)[:50]
print("\n  Rank | In-Degree | Type       | Name                             | Module")
print("  " + "-"*90)
for i, (nid, deg) in enumerate(in_degree[:30], 1):
    node = G.nodes[nid]
    name = node.get('name', '')[:30]
    typ = node.get('type', '')[:10]
    mod = get_module(node.get('file', ''))
    print(f"  {i:4} | {deg:9} | {typ:10} | {name:32} | {mod}")

with open(OUTPUT_DIR / "04_in_degree.json", 'w') as f:
    json.dump([
        {'rank': i+1, 'node_id': nid, 'in_degree': deg,
         'name': G.nodes[nid].get('name', ''),
         'type': G.nodes[nid].get('type', ''),
         'module': get_module(G.nodes[nid].get('file', ''))}
        for i, (nid, deg) in enumerate(in_degree)
    ], f, indent=2)
print(f"\n  >> Saved to 04_in_degree.json")

# ============================================================================
print("\n" + "="*70)
print("[6] MOST REFERENCING (Out-Degree) - Top 30")
print("="*70)

out_degree = sorted(G.out_degree(), key=lambda x: x[1], reverse=True)[:50]
print("\n  Rank | Out-Degree | Type       | Name                             | Module")
print("  " + "-"*90)
for i, (nid, deg) in enumerate(out_degree[:30], 1):
    node = G.nodes[nid]
    name = node.get('name', '')[:30]
    typ = node.get('type', '')[:10]
    mod = get_module(node.get('file', ''))
    print(f"  {i:4} | {deg:10} | {typ:10} | {name:32} | {mod}")

with open(OUTPUT_DIR / "05_out_degree.json", 'w') as f:
    json.dump([
        {'rank': i+1, 'node_id': nid, 'out_degree': deg,
         'name': G.nodes[nid].get('name', ''),
         'type': G.nodes[nid].get('type', ''),
         'module': get_module(G.nodes[nid].get('file', ''))}
        for i, (nid, deg) in enumerate(out_degree)
    ], f, indent=2)
print(f"\n  >> Saved to 05_out_degree.json")

# ============================================================================
print("\n" + "="*70)
print("[7] DEAD CODE CANDIDATES (Unreferenced Functions/Classes)")
print("="*70)

dead_code = []
for nid, node in victor_nodes.items():
    if node['type'] in ('function', 'class'):
        in_edges = list(G.in_edges(nid, data=True))
        meaningful = [e for e in in_edges if e[2].get('edge_type') != 'CONTAINS']
        if len(meaningful) == 0:
            is_private = node['name'].startswith('_')
            is_dunder = node['name'].startswith('__') and node['name'].endswith('__')
            dead_code.append({
                'name': node['name'],
                'type': node['type'],
                'file': node['file'],
                'line': node['line'],
                'module': get_module(node['file']),
                'is_private': is_private,
                'is_dunder': is_dunder,
                'severity': 'low' if is_private or is_dunder else 'medium'
            })

dead_code.sort(key=lambda x: (x['severity'], x['module']))

by_module = defaultdict(list)
for item in dead_code:
    by_module[item['module']].append(item)

print(f"\n  Total dead code candidates: {len(dead_code)}")
print(f"  Medium severity (public): {len([x for x in dead_code if x['severity'] == 'medium'])}")
print(f"  Low severity (private): {len([x for x in dead_code if x['severity'] == 'low'])}")

print("\n  By Module:")
for mod, items in sorted(by_module.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
    print(f"    {mod:30}: {len(items)} candidates")

print("\n  Sample Medium-Severity Candidates:")
medium = [x for x in dead_code if x['severity'] == 'medium'][:20]
for item in medium:
    short_file = '/'.join(Path(item['file']).parts[-3:]) if item['file'] else 'unknown'
    print(f"    {item['type']:8} {item['name'][:35]:35} @ {short_file}:{item['line']}")

with open(OUTPUT_DIR / "06_dead_code.json", 'w') as f:
    json.dump({
        'total': len(dead_code),
        'by_severity': {'medium': len([x for x in dead_code if x['severity'] == 'medium']),
                        'low': len([x for x in dead_code if x['severity'] == 'low'])},
        'by_module': {m: len(v) for m, v in by_module.items()},
        'candidates': dead_code
    }, f, indent=2)
print(f"\n  >> Saved to 06_dead_code.json")

# ============================================================================
print("\n" + "="*70)
print("[8] MODULE COUPLING ANALYSIS")
print("="*70)

module_edges = defaultdict(lambda: defaultdict(int))
for src, dst, data in G.edges(data=True):
    src_mod = get_submodule(G.nodes[src].get('file', ''))
    dst_mod = get_submodule(G.nodes[dst].get('file', ''))
    if src_mod != dst_mod:
        module_edges[src_mod][dst_mod] += 1

# Build module graph
MG = nx.DiGraph()
for src, dsts in module_edges.items():
    for dst, count in dsts.items():
        MG.add_edge(src, dst, weight=count)

print(f"\n  Module graph: {MG.number_of_nodes()} modules, {MG.number_of_edges()} edges")
print(f"  Module graph density: {nx.density(MG):.4f}")

# Coupling metrics
coupling = []
for mod in MG.nodes():
    aff = sum(d['weight'] for _, _, d in MG.in_edges(mod, data=True))
    eff = sum(d['weight'] for _, _, d in MG.out_edges(mod, data=True))
    total = aff + eff
    instab = eff / total if total > 0 else 0
    coupling.append({
        'module': mod,
        'afferent': aff,  # incoming deps
        'efferent': eff,  # outgoing deps
        'total': total,
        'instability': round(instab, 3)
    })

coupling.sort(key=lambda x: x['total'], reverse=True)

print("\n  Module Coupling (Top 15):")
print("  Module                     | Aff    | Eff    | Total  | Instab")
print("  " + "-"*70)
for c in coupling[:15]:
    print(f"  {c['module']:27} | {c['afferent']:6} | {c['efferent']:6} | {c['total']:6} | {c['instability']:.2f}")

# Tight coupling pairs
tight = []
for src, dst, data in MG.edges(data=True):
    if data['weight'] >= 20:
        tight.append((src, dst, data['weight']))
tight.sort(key=lambda x: x[2], reverse=True)

print("\n  Tightly Coupled Module Pairs (>= 20 edges):")
for src, dst, w in tight[:15]:
    print(f"    {src:25} -> {dst:25}: {w} edges")

# Circular dependencies
cycles = list(nx.simple_cycles(MG))
print(f"\n  Circular dependencies found: {len(cycles)}")
for cycle in cycles[:5]:
    print(f"    {' -> '.join(cycle)} -> {cycle[0]}")

with open(OUTPUT_DIR / "07_coupling.json", 'w') as f:
    json.dump({
        'density': nx.density(MG),
        'module_count': MG.number_of_nodes(),
        'edge_count': MG.number_of_edges(),
        'coupling': coupling,
        'tight_pairs': [{'src': s, 'dst': d, 'weight': w} for s, d, w in tight],
        'cycles': [list(c) for c in cycles[:20]]
    }, f, indent=2)
print(f"\n  >> Saved to 07_coupling.json")

# ============================================================================
print("\n" + "="*70)
print("[9] DUPLICATION ANALYSIS")
print("="*70)

# Name duplicates
by_name_type = defaultdict(list)
for nid, node in victor_nodes.items():
    if node['type'] in ('function', 'class'):
        key = (node['name'], node['type'])
        by_name_type[key].append(node)

name_dups = []
for (name, typ), nodes_list in by_name_type.items():
    if len(nodes_list) > 1:
        files = set(n['file'] for n in nodes_list)
        if len(files) > 1:
            name_dups.append({
                'name': name,
                'type': typ,
                'count': len(nodes_list),
                'locations': [
                    {'file': n['file'], 'line': n['line'], 'module': get_module(n['file'])}
                    for n in nodes_list
                ]
            })

name_dups.sort(key=lambda x: x['count'], reverse=True)

print(f"\n  Name duplicates (same name, different files): {len(name_dups)}")
print("\n  Top Name Duplicates:")
for d in name_dups[:15]:
    print(f"    {d['type']:8} '{d['name']}' appears in {d['count']} locations:")
    for loc in d['locations'][:3]:
        short = '/'.join(Path(loc['file']).parts[-3:]) if loc['file'] else '?'
        print(f"      - {short}:{loc['line']}")

# Signature duplicates
by_sig = defaultdict(list)
for nid, node in victor_nodes.items():
    if node['type'] == 'function' and node.get('signature'):
        sig = node['signature'].replace(' ', '').lower()
        if len(sig) > 20:
            by_sig[sig].append(node)

sig_dups = []
for sig, nodes_list in by_sig.items():
    if len(nodes_list) > 1:
        files = set(n['file'] for n in nodes_list)
        if len(files) > 1:
            sig_dups.append({
                'signature': sig[:80],
                'count': len(nodes_list),
                'locations': [
                    {'name': n['name'], 'file': n['file'], 'line': n['line']}
                    for n in nodes_list[:5]
                ]
            })

sig_dups.sort(key=lambda x: x['count'], reverse=True)
print(f"\n  Signature duplicates: {len(sig_dups)}")

# Similar call patterns
call_patterns = defaultdict(list)
for nid in G.nodes():
    if victor_nodes.get(nid, {}).get('type') == 'function':
        called = tuple(sorted(
            victor_nodes.get(dst, {}).get('name', '')
            for _, dst in G.out_edges(nid)
            if victor_nodes.get(dst, {}).get('type') == 'function'
        ))
        if len(called) >= 3:
            call_patterns[called].append(nid)

pattern_dups = []
for pattern, nids in call_patterns.items():
    if len(nids) > 1:
        pattern_dups.append({
            'pattern_size': len(pattern),
            'functions': [
                {'name': victor_nodes[n]['name'], 'module': get_module(victor_nodes[n]['file'])}
                for n in nids[:5]
            ],
            'shared_calls': list(pattern)[:10]
        })

pattern_dups.sort(key=lambda x: (len(x['functions']), x['pattern_size']), reverse=True)
print(f"  Similar call patterns: {len(pattern_dups)}")

with open(OUTPUT_DIR / "08_duplications.json", 'w') as f:
    json.dump({
        'summary': {
            'name_duplicates': len(name_dups),
            'signature_duplicates': len(sig_dups),
            'pattern_duplicates': len(pattern_dups)
        },
        'name_duplicates': name_dups[:50],
        'signature_duplicates': sig_dups[:30],
        'pattern_duplicates': pattern_dups[:30]
    }, f, indent=2)
print(f"\n  >> Saved to 08_duplications.json")

# ============================================================================
print("\n" + "="*70)
print("[10] MODULE DEPENDENCY DATA FOR DIAGRAMS")
print("="*70)

# Format for diagram generation
dep_data = []
for (src, dst), count in sorted(
    [((s, d), sum(v.values()) if isinstance(list(module_edges[s].values())[0] if module_edges[s] else 0, int) else module_edges[s][d])
     for s in module_edges for d in module_edges[s]],
    key=lambda x: module_edges[x[0][0]][x[0][1]], reverse=True
):
    if module_edges[src][dst] >= 5:
        dep_data.append({
            'source': src,
            'target': dst,
            'weight': module_edges[src][dst]
        })

dep_data.sort(key=lambda x: x['weight'], reverse=True)

print(f"\n  Significant module dependencies (>=5 edges): {len(dep_data)}")
print("\n  Top Dependencies:")
for d in dep_data[:20]:
    print(f"    {d['source']:25} -> {d['target']:25}: {d['weight']}")

modules_data = [
    {'name': mod, 'size': count}
    for mod, count in sorted(module_dist.items(), key=lambda x: x[1], reverse=True)
]

with open(OUTPUT_DIR / "09_module_deps.json", 'w') as f:
    json.dump({
        'modules': modules_data,
        'dependencies': dep_data
    }, f, indent=2)
print(f"\n  >> Saved to 09_module_deps.json")

# ============================================================================
print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print(f"Finished: {datetime.now()}")
print("="*70)
print(f"\nAll results saved to: {OUTPUT_DIR}")
