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
