import pytest
from unittest.mock import patch, AsyncMock
from victor.tools.unified.fs_tool import fs_tool


@pytest.mark.asyncio
async def test_fs_tool_cat_output_shape():
    """Test that `fs cat` correctly takes a string command and returns plaintext."""
    with patch("victor.tools.unified.fs_tool.read", new_callable=AsyncMock) as mock_read:
        # Mock the underlying python tool to return a truncated plaintext string
        mock_read.return_value = "1: print('hello')\n2: print('world')"

        # Test the NEW input shape (single bash-like command string)
        result = await fs_tool("fs cat main.py")

        mock_read.assert_called_once_with(
            "main.py", offset=0, limit=0, search="", ctx=2, regex=False
        )
        # Ensure it returns the pure plaintext, not JSON
        assert "print('hello')" in result
        assert "{" not in result  # No JSON wrapper


@pytest.mark.asyncio
async def test_fs_tool_ls_markdown_formatting():
    """Test that `fs ls` formats JSON dictionary outputs into a Markdown table."""
    with patch("victor.tools.unified.fs_tool.ls", new_callable=AsyncMock) as mock_ls:
        mock_ls.return_value = [
            {"name": "main.py", "type": "file", "size": "14KB"},
            {"name": "src", "type": "dir", "size": "-"},
        ]

        result = await fs_tool("fs ls .")

        mock_ls.assert_called_once_with(".", recursive=False, depth=2, pattern="", limit=1000)
        # Verify markdown table formatting
        assert "| Type | Size/Lines | Path |" in result
        assert "| file | 14KB | main.py |" in result
        assert "| dir | - | src |" in result


@pytest.mark.asyncio
async def test_fs_tool_cat_paging_and_search_options():
    """`fs cat` forwards paging + in-file search flags to read()."""
    with patch("victor.tools.unified.fs_tool.read", new_callable=AsyncMock) as mock_read:
        mock_read.return_value = "slice"
        await fs_tool("fs cat big.py --offset 200 --limit 50 --search 'def login' --ctx 3 --regex")
        mock_read.assert_called_once_with(
            "big.py", offset=200, limit=50, search="def login", ctx=3, regex=True
        )


@pytest.mark.asyncio
async def test_fs_tool_ls_options():
    """`fs ls` forwards recursive/depth/pattern/limit to ls()."""
    with patch("victor.tools.unified.fs_tool.ls", new_callable=AsyncMock) as mock_ls:
        mock_ls.return_value = []
        await fs_tool("fs ls src -r --depth 4 --pattern '*.py' --limit 25")
        mock_ls.assert_called_once_with("src", recursive=True, depth=4, pattern="*.py", limit=25)


@pytest.mark.asyncio
async def test_fs_tool_write_options():
    """`fs write` forwards validate/format/dry-run to write()."""
    with patch("victor.tools.unified.fs_tool.write", new_callable=AsyncMock) as mock_write:
        mock_write.return_value = "ok"
        await fs_tool("fs write app.py -c 'x = 1' --validate --dry-run")
        mock_write.assert_called_once_with(
            "app.py", "x = 1", validate=True, format_code=False, dry_run=True
        )


@pytest.mark.asyncio
async def test_fs_tool_patch_replaces_first_match(tmp_path):
    """Test that `fs patch --search/--replace` edits the target file."""
    target = tmp_path / "app.py"
    target.write_text("foo\nfoo\n", encoding="utf-8")

    result = await fs_tool(f'fs patch {target} --search "foo" --replace "bar"')

    assert "replaced 1 occurrence" in result
    assert target.read_text(encoding="utf-8") == "bar\nfoo\n"


@pytest.mark.asyncio
async def test_fs_tool_patch_recovery_hint(tmp_path):
    """Test that failed patches return Markdown headers and recovery hints."""
    target = tmp_path / "app.py"
    target.write_text("baz\n", encoding="utf-8")

    # Pass the options just like bash
    result = await fs_tool(f'fs patch {target} --search "foo" --replace "bar"')

    # Assert markdown headers and explicit recovery hints
    assert "### ❌ ERROR" in result
    assert "Patch failed: String not found" in result
    assert "### 💡 SYSTEM HINT" in result
    assert f"Use `fs cat {target}` to refresh your view" in result


# ---------------------------------------------------------------------------
# Shell-operator guard (the silent-argparse-failure fix)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "op_cmd",
    [
        "fs cat x 2>/dev/null || echo y",  # || + redirect (the transcript case)
        "fs ls . | grep py",  # pipe
        "fs cat a ; fs cat b",  # sequence (space-separated)
        "fs cat a && echo done",  # &&
        "fs cat a > out.txt",  # redirect to file
        "fs cat a >> out.txt",  # append redirect
    ],
)
async def test_fs_tool_rejects_shell_operators(op_cmd):
    """`fs` is not a shell — operators return a crisp, actionable message."""
    result = await fs_tool(op_cmd)
    assert "SHELL OPERATOR NOT SUPPORTED" in result
    assert "`shell` tool" in result
    # Must NOT be a generic parse error (the old behavior).
    assert "Error parsing command" not in result


@pytest.mark.asyncio
async def test_fs_tool_operator_inside_quoted_arg_not_flagged():
    """A `|` inside a quoted argument is content, not a shell operator."""
    with patch("victor.tools.unified.fs_tool.read", new_callable=AsyncMock) as mock_read:
        mock_read.return_value = "ok"
        await fs_tool('fs cat "a | b.txt"')
        mock_read.assert_called_once()  # reached read(), not the guard


# ---------------------------------------------------------------------------
# Robust `fs edit` (the hard blocker from the transcript)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fs_edit_new_without_mode_is_crisp_error(tmp_path):
    """`fs edit --new` with no --old/--insert/--append → short actionable error,
    not the old ~1100-char usage blob."""
    target = tmp_path / "app.py"
    target.write_text("x = 1\n", encoding="utf-8")

    result = await fs_tool(f'fs edit {target} --new "DEBUG = False"')

    assert "### ❌ ERROR" in result
    assert "without a target mode" in result
    assert "replace" in result and "insert" in result and "append" in result
    # Crisp — nowhere near the old 1110-char blob.
    assert len(result) < 700


@pytest.mark.asyncio
async def test_fs_edit_replace_delegates(tmp_path):
    """`fs edit --old/--new` builds a replace op and delegates to edit()."""
    target = tmp_path / "app.py"
    target.write_text("DEBUG = True\n", encoding="utf-8")

    with patch("victor.tools.file_editor_tool.edit", new_callable=AsyncMock) as mock_edit:
        mock_edit.return_value = {"success": True, "message": "Applied 1 operation(s)"}
        result = await fs_tool(f'fs edit {target} --old "DEBUG = True" --new "DEBUG = False"')

    mock_edit.assert_called_once()
    ops = mock_edit.call_args.kwargs["ops"]
    assert ops[0]["type"] == "replace"
    assert ops[0]["old_str"] == "DEBUG = True"
    assert ops[0]["new_str"] == "DEBUG = False"
    assert "Applied" in result


@pytest.mark.asyncio
async def test_fs_edit_insert_after_anchor(tmp_path):
    """`fs edit --insert ANCHOR --new TEXT` splices text after the anchor line."""
    target = tmp_path / "app.py"
    target.write_text("def a():\n    return 1\n\ndef b():\n    return 2\n", encoding="utf-8")
    new_code = "def inserted():\n    return 0\n"

    with patch("victor.tools.file_editor_tool.edit", new_callable=AsyncMock) as mock_edit:
        mock_edit.return_value = {"success": True, "message": "ok"}
        await fs_tool(f'fs edit {target} --insert "def b():" --new "{new_code}"')

    ops = mock_edit.call_args.kwargs["ops"]
    assert ops[0]["type"] == "modify"
    spliced = ops[0]["content"]
    # Inserted text lands immediately AFTER the anchor line.
    assert spliced.index("def inserted") > spliced.index("def b():")
    assert spliced.count("def inserted") == 1


@pytest.mark.asyncio
async def test_fs_edit_insert_anchor_not_found(tmp_path):
    """A missing anchor yields a clear error, not a stack trace."""
    target = tmp_path / "app.py"
    target.write_text("x = 1\n", encoding="utf-8")

    result = await fs_tool(f'fs edit {target} --insert "nope" --new "y = 2"')
    assert "Insert anchor not found" in result


@pytest.mark.asyncio
async def test_fs_edit_insert_anchor_ambiguous(tmp_path):
    """An anchor matching multiple lines is rejected with a clear message."""
    target = tmp_path / "app.py"
    target.write_text("dup\ndup\ndup\n", encoding="utf-8")

    result = await fs_tool(f'fs edit {target} --insert "dup" --new "new"')
    assert "ambiguous" in result


@pytest.mark.asyncio
async def test_fs_edit_append(tmp_path):
    """`fs edit --append --new TEXT` appends to the end of the file."""
    target = tmp_path / "app.py"
    target.write_text("a\nb\n", encoding="utf-8")

    with patch("victor.tools.file_editor_tool.edit", new_callable=AsyncMock) as mock_edit:
        mock_edit.return_value = {"success": True, "message": "ok"}
        await fs_tool(f'fs edit {target} --append --new "c"')

    ops = mock_edit.call_args.kwargs["ops"]
    assert ops[0]["content"].endswith("a\nb\nc") or ops[0]["content"].endswith("a\nb\nc\n")


@pytest.mark.asyncio
async def test_fs_edit_new_file_source(tmp_path):
    """`--new-file` reads the new text from a file (robust for multiline)."""
    target = tmp_path / "app.py"
    target.write_text("keep\n", encoding="utf-8")
    src = tmp_path / "snippet.txt"
    src.write_text('line with "quotes" and {braces}\nmulti\n', encoding="utf-8")

    with patch("victor.tools.file_editor_tool.edit", new_callable=AsyncMock) as mock_edit:
        mock_edit.return_value = {"success": True, "message": "ok"}
        await fs_tool(f"fs edit {target} --append --new-file {src}")

    ops = mock_edit.call_args.kwargs["ops"]
    assert 'line with "quotes" and {braces}' in ops[0]["content"]


@pytest.mark.asyncio
async def test_fs_edit_dry_run_passes_preview(tmp_path):
    """`--dry-run` calls edit() in preview/no-commit mode and surfaces a diff."""
    target = tmp_path / "app.py"
    target.write_text("x = 1\n", encoding="utf-8")

    with patch("victor.tools.file_editor_tool.edit", new_callable=AsyncMock) as mock_edit:
        mock_edit.return_value = {
            "success": True,
            "message": "preview",
            "diff": "-x = 1\n+DEBUG = False",
        }
        result = await fs_tool(f'fs edit {target} --old "x = 1" --new "DEBUG = False" --dry-run')

    assert mock_edit.call_args.kwargs["preview"] is True
    assert mock_edit.call_args.kwargs["commit"] is False
    assert "```diff" in result
    assert "DEBUG = False" in result
