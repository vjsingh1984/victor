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

        mock_read.assert_called_once_with("main.py")
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

        mock_ls.assert_called_once_with(".")
        # Verify markdown table formatting
        assert "| Type | Size/Lines | Path |" in result
        assert "| file | 14KB | main.py |" in result
        assert "| dir | - | src |" in result


@pytest.mark.asyncio
async def test_fs_tool_patch_recovery_hint():
    """Test that failed patches return Markdown headers and recovery hints."""
    with patch("victor.tools.unified.fs_tool.apply_patch", new_callable=AsyncMock) as mock_replace:
        mock_replace.side_effect = ValueError("String not found")

        # Pass the options just like bash
        result = await fs_tool('fs patch app.py --search "foo" --replace "bar"')

        # Assert markdown headers and explicit recovery hints
        assert "### ❌ ERROR" in result
        assert "Patch failed: String not found" in result
        assert "### 💡 SYSTEM HINT" in result
        assert "Use `fs cat app.py` to refresh your view" in result
