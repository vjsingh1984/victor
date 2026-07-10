"""Tests for edit-batch per-file isolation, structured reporting, and fuzzy fallback.

Regression coverage for the dogfooding failure where a batched ``edit`` call
spanning two files discarded a *valid* edit to file A because an op targeting
file B had a stale ``old_str``. Edits are now isolated per file.
"""

import pytest

from victor.tools.file_editor_tool import (
    _find_unique_fuzzy_span,
    _resolve_replace,
    edit,
)

# ---------------------------------------------------------------------------
# Pure helpers (no editor backend required — always run).
# ---------------------------------------------------------------------------


class TestResolveReplace:
    def test_exact_unique_match(self):
        content = "def foo():\n    x = 1\n    return x\n"
        res = _resolve_replace(content, "    x = 1", "    x = 2", "f.py")
        assert res["ok"] is True
        assert res["fuzzy"] is False
        assert "x = 2" in res["new_content"]

    def test_ambiguous_match_is_rejected(self):
        res = _resolve_replace("a\na\n", "a", "b", "f.py")
        assert res["ok"] is False
        assert "Ambiguous" in res["reason"] or "ambiguous" in res["reason"]

    def test_missing_match_returns_rich_detail(self):
        content = "def foo():\n    x = 1\n"
        res = _resolve_replace(content, "    nonexistent", "y", "f.py")
        assert res["ok"] is False
        assert "old_str not found" in res["reason"]

    def test_fuzzy_whitespace_drift_resolves(self):
        # old_str collapses extra spacing between tokens vs the file.
        content = "def foo():\n    return 1\n"
        res = _resolve_replace(content, "def    foo():", "def bar():", "f.py")
        assert res["ok"] is True
        assert res["fuzzy"] is True
        assert "def bar():" in res["new_content"]
        assert "return 1" in res["new_content"]

    def test_fuzzy_indentation_drift_resolves(self):
        content = "class A:\n\t def method(self):\n\t     pass\n"
        # Same logical lines, different leading whitespace (tabs vs spaces).
        old = "    def method(self):\n        pass"
        res = _resolve_replace(content, old, "    def method(self):\n        return 1", "f.py")
        assert res["ok"] is True
        assert res["fuzzy"] is True

    def test_fuzzy_is_unique_only(self):
        # Two whitespace-equivalent spans -> ambiguous -> no fuzzy guess.
        content = "x = 1\nx  =  1\n"
        assert _find_unique_fuzzy_span(content, "x = 1") is None


# ---------------------------------------------------------------------------
# Integration: per-file isolation through the real ``edit`` tool.
# ---------------------------------------------------------------------------


def _editor_available() -> bool:
    try:
        from victor.tools.file_editor_tool import _is_file_editor_available

        return _is_file_editor_available()
    except Exception:
        return False


requires_editor = pytest.mark.skipif(
    not _editor_available(), reason="Enhanced editor not registered (requires vertical)"
)


@pytest.mark.integration
@requires_editor
class TestPerFileIsolation:
    @pytest.mark.asyncio
    async def test_good_edit_survives_sibling_failure(self, tmp_path):
        """The transcript scenario: A's valid op must commit even though B's op fails."""
        file_a = tmp_path / "a.py"
        file_b = tmp_path / "b.py"
        file_a.write_text("VALUE_A = 1\n")
        file_b.write_text("VALUE_B = 2\n")

        result = await edit(
            ops=[
                {
                    "type": "replace",
                    "path": str(file_a),
                    "old_str": "VALUE_A = 1",
                    "new_str": "VALUE_A = 99",
                },
                # Stale old_str (reconstructed from memory) — does not match file B.
                {
                    "type": "replace",
                    "path": str(file_b),
                    "old_str": "VALUE_B = 999",
                    "new_str": "VALUE_B = 0",
                },
            ]
        )

        # File A's good edit is preserved...
        assert file_a.read_text() == "VALUE_A = 99\n"
        # ...and File B is untouched.
        assert file_b.read_text() == "VALUE_B = 2\n"
        # ...and the failure is reported structurally for targeted retry.
        assert result.get("partial") is True
        assert any(str(file_b) in str(f["path"]) for f in result["failed"])

    @pytest.mark.asyncio
    async def test_all_ops_fail_returns_failure(self, tmp_path):
        file_a = tmp_path / "a.py"
        file_a.write_text("VALUE_A = 1\n")
        result = await edit(
            ops=[
                {
                    "type": "replace",
                    "path": str(file_a),
                    "old_str": "NOPE",
                    "new_str": "x",
                },
            ]
        )
        assert result["success"] is False
        assert result["failed"]
        assert file_a.read_text() == "VALUE_A = 1\n"

    @pytest.mark.asyncio
    async def test_fuzzy_fallback_applies_through_tool(self, tmp_path):
        file_a = tmp_path / "a.py"
        file_a.write_text("def foo():\n    return 1\n")
        result = await edit(
            ops=[
                {
                    "type": "replace",
                    "path": str(file_a),
                    "old_str": "def    foo():",
                    "new_str": "def bar():",
                },
            ]
        )
        assert result["success"] is True
        assert "def bar():" in file_a.read_text()

    @pytest.mark.asyncio
    async def test_same_file_group_is_atomic(self, tmp_path):
        """If a later same-file op fails, the earlier same-file op is rolled back too."""
        file_a = tmp_path / "a.py"
        file_b = tmp_path / "b.py"
        file_a.write_text("A1\nA2\n")
        file_b.write_text("B1\n")

        result = await edit(
            ops=[
                {
                    "type": "replace",
                    "path": str(file_a),
                    "old_str": "A1",
                    "new_str": "A1x",
                },
                {
                    "type": "replace",
                    "path": str(file_a),
                    "old_str": "NOMATCH",
                    "new_str": "z",
                },
                {
                    "type": "replace",
                    "path": str(file_b),
                    "old_str": "B1",
                    "new_str": "B1x",
                },
            ]
        )
        # File A unchanged (its group failed atomically)...
        assert file_a.read_text() == "A1\nA2\n"
        # ...File B's independent group still applied.
        assert file_b.read_text() == "B1x\n"
        assert result.get("partial") is True

    @pytest.mark.asyncio
    async def test_same_file_valid_sibling_reported_as_rolled_back(self, tmp_path):
        """A valid same-file op is reported (not silently dropped) when a sibling fails."""
        file_a = tmp_path / "a.py"
        file_b = tmp_path / "b.py"
        file_a.write_text("A1\nA2\n")
        file_b.write_text("B1\n")

        result = await edit(
            ops=[
                {"type": "replace", "path": str(file_a), "old_str": "A1", "new_str": "A1x"},
                {"type": "replace", "path": str(file_a), "old_str": "NOMATCH", "new_str": "z"},
                {"type": "replace", "path": str(file_b), "old_str": "B1", "new_str": "B1x"},
            ]
        )

        # file_a rolled back atomically; file_b's independent group still applied.
        assert file_a.read_text() == "A1\nA2\n"
        assert file_b.read_text() == "B1x\n"
        assert result.get("partial") is True
        # The otherwise-valid file_a op is surfaced with a rollback reason so the
        # model knows to re-read and retry the whole file.
        rolled_back = [f for f in result["failed"] if "rolled back" in f.get("error", "")]
        assert rolled_back, f"expected a rolled-back sibling report, got: {result.get('failed')}"
