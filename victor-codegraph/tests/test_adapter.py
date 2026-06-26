"""Adapter tests — ProximaRecord shape projection (no DB, no embed by default)."""

from __future__ import annotations

from victor_codegraph import parse, to_proxima_records

SAMPLE = '''\
def a():
    return b()


def b():
    return 1
'''


def test_symbol_records_shape():
    parsed = parse(SAMPLE, file_path="m.py")
    records = to_proxima_records(parsed, repo_graph_id="repo1")
    nodes = [r for r in records if "graph_node" in r["labels"]]
    assert len(nodes) == 2
    n = nodes[0]
    assert n["oid"].startswith("graph/repo1/node/")
    assert n["labels"] == ["graph_node", "code_symbol"]
    assert n["branch_id"] == "main"
    assert n["props"]["lang"] == "python"
    assert n["props"]["ast_kind"] in ("FUNCTION", "METHOD", "CONSTRUCTOR")
    assert n["embeddings"] == []  # no embedder supplied


def test_edge_records_reference_node_oids():
    parsed = parse(SAMPLE, file_path="m.py")
    records = to_proxima_records(parsed, repo_graph_id="repo1")
    edges = [r for r in records if "graph_edge" in r["labels"]]
    assert edges, "expected a CALLS edge (a -> b)"
    e = edges[0]
    assert e["edge"]["from_oid"].startswith("graph/repo1/node/")
    assert e["edge"]["to_oid"].startswith("graph/repo1/node/")
    assert e["edge"]["edge_type"] == "CALLS"
    # edge props always carry a call-site line (0 when unknown; >0 once the parser
    # preserves call_site through resolution — see the call-site-fidelity change).
    assert "line" in e["props"]
    assert isinstance(e["props"]["line"], int)


def test_embedder_populates_embedding_cell():
    parsed = parse(SAMPLE, file_path="m.py")
    records = to_proxima_records(
        parsed, repo_graph_id="repo1", embedder=lambda text: [0.0] * 384
    )
    node = next(r for r in records if "graph_node" in r["labels"])
    assert len(node["embeddings"]) == 1
    cell = node["embeddings"][0]
    assert cell["modality"] == "code"
    assert cell["dim"] == 384
    assert len(cell["values"]) == 384
