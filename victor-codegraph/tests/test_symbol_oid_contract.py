"""Pins the ADR-044 stable, line-independent symbol-oid contract.

The correlated-CPG join key must survive edits that don't change *what a symbol is*:
a line move keeps the oid; a rename / signature change mints a new one; a body edit
bumps ``content_version`` but not the oid. Every cross-surface consumer (Victor, the
ProximaDB SDK, the AnvaiOps connector) derives identity from this one function, so if it
ever drifts this test fails loudly.
"""

from __future__ import annotations

from victor_codegraph import parse, stable_symbol_oid, to_proxima_records


def _node(records, name):
    for r in records:
        if "graph_node" in r.get("labels", []) and r["props"].get("name") == name:
            return r
    raise AssertionError(f"no node record for {name!r}")


def _records(content, *, repo="repo1", stable=True):
    return to_proxima_records(
        parse(content, file_path="m.py"), repo_graph_id=repo, stable_oid=stable
    )


def _oid_of(content, name):
    return _node(_records(content), name)["oid"]


def test_stable_oid_unchanged_when_symbol_moves_lines():
    base = "def first():\n    return second()\n\n\ndef second():\n    return 1\n"
    moved = "\n\n\n" + base  # both symbols pushed down; identity must not churn
    assert _oid_of(base, "second") == _oid_of(moved, "second"), "line move must not churn the oid"
    assert _oid_of(base, "first") == _oid_of(moved, "first")


def test_stable_oid_changes_on_rename():
    a = "def alpha():\n    return 1\n"
    b = "def beta():\n    return 1\n"
    assert _oid_of(a, "alpha") != _oid_of(b, "beta"), "rename must mint a new identity"


def test_stable_oid_changes_on_signature_change():
    one = "def f(x):\n    return x\n"
    two = "def f(x, y):\n    return x\n"
    assert _oid_of(one, "f") != _oid_of(two, "f"), "signature change must mint a new identity"


def test_body_edit_bumps_content_version_not_oid():
    a = _node(_records("def f(x):\n    return x\n"), "f")
    b = _node(_records("def f(x):\n    return x + 1  # changed body\n"), "f")
    assert a["oid"] == b["oid"], "body edit must not churn the oid"
    assert (
        a["props"]["content_version"] != b["props"]["content_version"]
    ), "body edit must bump content_version"


def test_dual_emit_and_default_off_gate():
    parsed = parse("def f(x):\n    return x\n", file_path="m.py")
    # Default OFF: the record oid is the legacy line-coupled id (byte-identical behavior),
    # but BOTH ids are always emitted for mixed-read.
    off = _node(to_proxima_records(parsed, repo_graph_id="r"), "f")
    assert off["oid"] == off["props"]["legacy_oid"]
    assert off["props"]["stable_oid"] != off["props"]["legacy_oid"]
    # Gate ON: the record oid is the canonical stable id.
    on = _node(to_proxima_records(parsed, repo_graph_id="r", stable_oid=True), "f")
    assert on["oid"] == on["props"]["stable_oid"]
    assert on["props"]["legacy_oid"] == off["props"]["legacy_oid"]


def test_stable_symbol_oid_is_pure_and_structural():
    # The pure derivation depends only on structural coordinates — never on line/col.
    k = stable_symbol_oid("repo1", "python", "m.py::f", "(x)")
    assert k == stable_symbol_oid("repo1", "python", "m.py::f", "(x)")  # deterministic
    assert k != stable_symbol_oid("repo2", "python", "m.py::f", "(x)")  # repo
    assert k != stable_symbol_oid("repo1", "rust", "m.py::f", "(x)")  # language
    assert k != stable_symbol_oid("repo1", "python", "m.py::g", "(x)")  # name (FQN)
    assert k != stable_symbol_oid("repo1", "python", "m.py::f", "(x, y)")  # overload
