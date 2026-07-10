"""Pins the canonical 1-based line-number contract (see model.LINE_BASE).

This is the cross-surface contract — Victor, the ProximaDB SDK, and AnvaiOps all
consume this package and must agree on a symbol's (file, name, line) coordinate.
If a refactor ever shifts line numbers, this test fails loudly.
"""

from __future__ import annotations

from victor_codegraph import LINE_BASE, parse


def test_line_base_is_one():
    assert LINE_BASE == 1


def test_parse_emits_1_based_lines():
    # The def is on the FIRST line → start_line must be 1 (not 0).
    parsed = parse(
        "def first():\n    return second()\n\n\ndef second():\n    return 1\n",
        file_path="m.py",
    )
    by_name = {s.simple_name: s for s in parsed.symbols}
    assert by_name["first"].location.start_line == 1
    assert by_name["second"].location.start_line == 5  # 1-based line of the 2nd def


def test_call_site_line_is_1_based():
    parsed = parse("def a():\n    return b()\n\n\ndef b():\n    return 1\n", file_path="m.py")
    calls = [r for r in parsed.relations if r.relation_type.name == "CALLS"]
    assert calls
    # the b() call is on line 2 (1-based), not 1
    assert any(r.call_site is not None and r.call_site.start_line == 2 for r in calls)
