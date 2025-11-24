"""Tests for file editor tool integration."""

import asyncio
import tempfile
from pathlib import Path

from victor.tools.file_editor_tool import FileEditorTool


async def test_file_editor_tool():
    """Test file editor tool operations."""
    print("🧪 Testing File Editor Tool Integration\n")
    print("=" * 70)

    tool = FileEditorTool()

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.py"

        # Test 1: Start transaction
        print("\n1️⃣ Starting transaction...")
        result = await tool.execute(
            operation="start_transaction",
            description="Test transaction"
        )
        assert result.success, f"Start failed: {result.error}"
        print(f"✓ {result.output}")

        # Test 2: Add file creation
        print("\n2️⃣ Adding file creation...")
        result = await tool.execute(
            operation="add_create",
            path=str(test_file),
            content="def hello():\n    print('Hello, World!')\n"
        )
        assert result.success, f"Add create failed: {result.error}"
        print(f"✓ {result.output}")

        # Test 3: Check status
        print("\n3️⃣ Checking status...")
        result = await tool.execute(operation="status")
        assert result.success, f"Status failed: {result.error}"
        print(f"✓ Transaction status:\n{result.output}")

        # Test 4: Preview changes
        print("\n4️⃣ Previewing changes...")
        result = await tool.execute(operation="preview")
        assert result.success, f"Preview failed: {result.error}"
        print(f"✓ {result.output}")

        # Test 5: Commit changes
        print("\n5️⃣ Committing changes...")
        result = await tool.execute(operation="commit")
        assert result.success, f"Commit failed: {result.error}"
        print(f"✓ {result.output}")

        # Verify file was created
        assert test_file.exists(), "File was not created"
        content = test_file.read_text()
        assert "Hello, World!" in content, "File content incorrect"
        print(f"✓ File created successfully: {test_file}")

        # Test 6: Modify file
        print("\n6️⃣ Modifying file...")
        result = await tool.execute(
            operation="start_transaction",
            description="Modify test file"
        )
        assert result.success

        new_content = "def hello():\n    print('Hello, Victor!')\n"
        result = await tool.execute(
            operation="add_modify",
            path=str(test_file),
            new_content=new_content
        )
        assert result.success, f"Add modify failed: {result.error}"

        result = await tool.execute(operation="commit")
        assert result.success, f"Commit failed: {result.error}"

        # Verify modification
        content = test_file.read_text()
        assert "Victor" in content, "File modification failed"
        print(f"✓ File modified successfully")

        # Test 7: Dry run
        print("\n7️⃣ Testing dry run...")
        result = await tool.execute(
            operation="start_transaction",
            description="Dry run test"
        )
        assert result.success

        result = await tool.execute(
            operation="add_modify",
            path=str(test_file),
            new_content="# This won't be applied\n"
        )
        assert result.success

        result = await tool.execute(operation="commit", dry_run=True)
        assert result.success, f"Dry run failed: {result.error}"

        # Verify file wasn't changed
        content = test_file.read_text()
        assert "Victor" in content, "Dry run modified file!"
        print(f"✓ Dry run successful - file unchanged")

        # Test 8: Abort transaction
        print("\n8️⃣ Testing abort...")
        result = await tool.execute(
            operation="start_transaction",
            description="Test abort"
        )
        assert result.success

        result = await tool.execute(
            operation="add_delete",
            path=str(test_file)
        )
        assert result.success

        result = await tool.execute(operation="abort")
        assert result.success, f"Abort failed: {result.error}"

        # Verify file still exists
        assert test_file.exists(), "File was deleted after abort!"
        print(f"✓ Transaction aborted - file still exists")

    print("\n\n✨ All tests passed!")
    print("\nFile editor tool is ready for agent integration!")


if __name__ == "__main__":
    asyncio.run(test_file_editor_tool())
