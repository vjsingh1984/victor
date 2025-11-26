# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Demo of Git tool with AI-powered features.

This demonstrates:
1. Git status and diff operations
2. AI-generated commit messages
3. Smart staging
4. Branch management
5. PR creation with auto-descriptions
6. Conflict analysis

Usage:
    python examples/git_tool_demo.py
"""

import asyncio
import tempfile
from pathlib import Path
import subprocess

# For demo purposes - in real usage, this would come from agent
from victor.tools.git_tool import GitTool


def run_command(cmd: str, cwd: Path = None) -> str:
    """Run shell command and return output."""
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        cwd=cwd
    )
    return result.stdout


async def main():
    """Demo git tool operations."""
    print("🎯 Git Tool Demo (without AI provider)")
    print("=" * 70)
    print("\nNote: AI features require LLM provider integration")
    print("      This demo shows core git operations\n")

    # Create temporary git repo for demo
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Initialize git repo
        print("1️⃣ Setting up demo repository...")
        print("-" * 70)
        run_command("git init", tmpdir)
        run_command("git config user.email 'demo@example.com'", tmpdir)
        run_command("git config user.name 'Demo User'", tmpdir)

        # Change to temp directory for git operations
        import os
        original_dir = os.getcwd()
        os.chdir(tmpdir)

        try:
            # Create git tool (without provider for basic demo)
            git_tool = GitTool(provider=None)

            # Create some files
            (tmpdir / "README.md").write_text("""# Demo Project

This is a demo project for testing Victor's git tool.

## Features
- Git integration
- AI-powered commit messages
- Smart operations
""")

            (tmpdir / "main.py").write_text("""def main():
    print("Hello, World!")

if __name__ == "__main__":
    main()
""")

            # Test 1: Git status
            print("\n2️⃣ Git Status")
            print("-" * 70)
            result = await git_tool.execute(operation="status")
            print(result.output if result.success else f"Error: {result.error}")

            # Test 2: Stage files
            print("\n3️⃣ Staging Files")
            print("-" * 70)
            result = await git_tool.execute(
                operation="stage",
                files=["README.md", "main.py"]
            )
            print(result.output if result.success else f"Error: {result.error}")

            # Test 3: Show diff
            print("\n4️⃣ Staged Diff")
            print("-" * 70)
            result = await git_tool.execute(operation="diff", staged=True)
            if result.success:
                print(result.output[:500] + "..." if len(result.output) > 500 else result.output)
            else:
                print(f"Error: {result.error}")

            # Test 4: Commit (without AI since no provider)
            print("\n5️⃣ Commit Changes")
            print("-" * 70)
            result = await git_tool.execute(
                operation="commit",
                message="Initial commit: Add README and main.py\n\nThis is a demo commit showing Victor's git tool capabilities.",
                generate_ai=False  # No AI provider in demo
            )
            print(result.output if result.success else f"Error: {result.error}")

            # Test 5: Git log
            print("\n6️⃣ Commit History")
            print("-" * 70)
            result = await git_tool.execute(operation="log", limit=5)
            print(result.output if result.success else f"Error: {result.error}")

            # Test 6: Create and switch branch
            print("\n7️⃣ Branch Operations")
            print("-" * 70)
            result = await git_tool.execute(operation="branch", branch="feature/add-tests")
            print(result.output if result.success else f"Error: {result.error}")

            # Test 7: Make more changes
            print("\n8️⃣ Making More Changes")
            print("-" * 70)
            (tmpdir / "tests.py").write_text("""import pytest

def test_main():
    assert True
""")

            result = await git_tool.execute(operation="stage", files=["tests.py"])
            print(result.output if result.success else f"Error: {result.error}")

            # Test 8: Commit with manual message
            result = await git_tool.execute(
                operation="commit",
                message="test: Add basic test file",
                generate_ai=False
            )
            print(f"\nCommit result: {result.output if result.success else result.error}")

            # Test 9: Switch back to main
            print("\n9️⃣ Switching Branches")
            print("-" * 70)
            result = await git_tool.execute(operation="branch", branch="main")
            print(result.output if result.success else f"Error: {result.error}")

            # Test 10: List all branches
            print("\n🔟 List All Branches")
            print("-" * 70)
            result = await git_tool.execute(operation="branch")
            print(result.output if result.success else f"Error: {result.error}")

            # Test 11: Make changes on main
            print("\n1️⃣1️⃣ Making Changes on Main Branch")
            print("-" * 70)
            (tmpdir / "utils.py").write_text("""def helper():
    return "Helper function"
""")

            result = await git_tool.execute(operation="stage")
            result = await git_tool.execute(
                operation="commit",
                message="feat: Add utility functions",
                generate_ai=False
            )
            print(result.output if result.success else f"Error: {result.error}")

            # Test 12: Final log showing multiple commits
            print("\n1️⃣2️⃣ Final Commit History")
            print("-" * 70)
            result = await git_tool.execute(operation="log", limit=10)
            print(result.output if result.success else f"Error: {result.error}")

            print("\n\n✨ Demo Complete!")
            print("\nGit Tool Features Demonstrated:")
            print("  ✓ Repository status checking")
            print("  ✓ File staging (individual and bulk)")
            print("  ✓ Diff viewing (staged and unstaged)")
            print("  ✓ Committing changes")
            print("  ✓ Viewing commit history")
            print("  ✓ Branch creation and switching")
            print("  ✓ Multiple commits and branch management")

            print("\n\n🤖 With AI Provider Available:")
            print("  • AI-generated commit messages from diff analysis")
            print("  • Intelligent commit message formatting")
            print("  • Auto-generated PR titles and descriptions")
            print("  • Conflict resolution suggestions")
            print("  • Smart file grouping for related changes")

            print("\n\n📚 Example with AI (requires provider):")
            print("""
# In agent conversation:
User: "Commit my changes"

Victor: Let me analyze your changes and generate a commit message...
[Calls git tool with operation="suggest_commit"]

Victor: I suggest this commit message:
"feat(auth): Add password hashing with PBKDF2

Implements secure password storage using PBKDF2-HMAC-SHA256
with 32-byte salts and 100,000 iterations. Updates authentication
module to verify hashed passwords instead of plaintext.

Breaking change: Existing passwords need to be rehashed."

Victor: Should I proceed with this message?
User: "Yes"

[Commits with AI-generated message]
Victor: ✓ Changes committed successfully!
""")

        finally:
            # Restore original directory
            os.chdir(original_dir)


if __name__ == "__main__":
    asyncio.run(main())
