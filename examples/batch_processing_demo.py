"""Demo of Victor's Batch Processing Tool.

Demonstrates efficient multi-file operations:
- Parallel search across files
- Bulk find and replace
- Multi-file analysis
- Dry-run previews

Usage:
    python examples/batch_processing_demo.py
"""

import asyncio
import tempfile
from pathlib import Path
from victor.tools.batch_processor_tool import BatchProcessorTool


def setup_demo_files(temp_dir: Path) -> None:
    """Create demo files for testing."""
    print("\n📁 Setting up demo files...")

    # Create demo Python files
    files = {
        "app.py": """
import logging
from typing import List

# TODO: Add error handling
def process_data(items: List[str]) -> None:
    print("Processing items...")
    for item in items:
        # TODO: Validate item
        print(f"Item: {item}")

logger = logging.getLogger(__name__)
""",
        "utils.py": """
import os
from pathlib import Path

# TODO: Add type hints
def read_config(path):
    print(f"Reading config from {path}")
    return {}

# TODO: Improve error handling
def save_data(data, filename):
    print(f"Saving to {filename}")
    with open(filename, 'w') as f:
        f.write(str(data))
""",
        "models.py": """
from dataclasses import dataclass
from typing import Optional

@dataclass
class User:
    # TODO: Add validation
    id: int
    name: str
    email: Optional[str] = None

    def __str__(self):
        print(f"User: {self.name}")
        return self.name
""",
        "tests/test_app.py": """
import pytest
from app import process_data

# TODO: Add more test cases
def test_process_data():
    print("Testing process_data")
    items = ["a", "b", "c"]
    process_data(items)
    # TODO: Add assertions
""",
        "config.yaml": """
app:
  name: Victor Demo
  version: 1.0.0
  # TODO: Add more config
  debug: true
""",
    }

    for file_path, content in files.items():
        full_path = temp_dir / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content.strip())

    print(f"✓ Created {len(files)} demo files")


async def demo_batch_search():
    """Demo batch search across multiple files."""
    print("\n\n🔍 Batch Search Demo")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        setup_demo_files(temp_path)

        tool = BatchProcessorTool(max_workers=4)

        print("\n1️⃣ Search for 'TODO' comments across all files...")
        result = await tool.execute(
            operation="search",
            path=str(temp_path),
            pattern="TODO",
            file_pattern="**/*.py",
        )

        if result.success:
            print(result.output)
        else:
            print(f"❌ Error: {result.error}")

        print("\n2️⃣ Search for 'print' statements (code smell)...")
        result = await tool.execute(
            operation="search",
            path=str(temp_path),
            pattern="print\\(",
            file_pattern="*.py",
            regex=True,
        )

        if result.success:
            print(result.output)

        print("\n3️⃣ Search in specific file pattern...")
        result = await tool.execute(
            operation="search",
            path=str(temp_path),
            pattern="def ",
            file_pattern="**/test_*.py",
        )

        if result.success:
            print(result.output)


async def demo_batch_replace():
    """Demo batch find and replace operations."""
    print("\n\n🔄 Batch Replace Demo")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        setup_demo_files(temp_path)

        tool = BatchProcessorTool(max_workers=4)

        print("\n1️⃣ DRY RUN: Preview replacing 'print' with 'logger.info'...")
        result = await tool.execute(
            operation="replace",
            path=str(temp_path),
            find="print",
            replace="logger.info",
            file_pattern="*.py",
            dry_run=True,
        )

        if result.success:
            print(result.output)

        print("\n2️⃣ EXECUTE: Replace 'TODO' with 'FIXME'...")
        result = await tool.execute(
            operation="replace",
            path=str(temp_path),
            find="TODO",
            replace="FIXME",
            file_pattern="**/*.py",
            dry_run=False,
        )

        if result.success:
            print(result.output)

        print("\n3️⃣ Verify changes with search...")
        result = await tool.execute(
            operation="search",
            path=str(temp_path),
            pattern="FIXME",
            file_pattern="**/*.py",
        )

        if result.success:
            print(result.output)

        print("\n4️⃣ Regex replace: Fix function definitions...")
        result = await tool.execute(
            operation="replace",
            path=str(temp_path),
            find=r"def (\w+)\(([^)]*)\):",
            replace=r"def \1(\2) -> None:",
            file_pattern="*.py",
            regex=True,
            dry_run=True,
        )

        if result.success:
            print(result.output)


async def demo_batch_analyze():
    """Demo batch file analysis."""
    print("\n\n📊 Batch Analysis Demo")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        setup_demo_files(temp_path)

        tool = BatchProcessorTool(max_workers=4)

        print("\n1️⃣ Analyze all Python files...")
        result = await tool.execute(
            operation="analyze",
            path=str(temp_path),
            file_pattern="**/*.py",
        )

        if result.success:
            print(result.output)

        print("\n2️⃣ Analyze all files (including config)...")
        result = await tool.execute(
            operation="analyze",
            path=str(temp_path),
            file_pattern="*.*",
        )

        if result.success:
            print(result.output)


async def demo_list_files():
    """Demo file listing."""
    print("\n\n📋 File Listing Demo")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        setup_demo_files(temp_path)

        tool = BatchProcessorTool(max_workers=4)

        print("\n1️⃣ List all Python files...")
        result = await tool.execute(
            operation="list",
            path=str(temp_path),
            file_pattern="**/*.py",
        )

        if result.success:
            print(result.output)

        print("\n2️⃣ List only test files...")
        result = await tool.execute(
            operation="list",
            path=str(temp_path),
            file_pattern="**/test_*.py",
        )

        if result.success:
            print(result.output)


async def demo_real_world_workflow():
    """Demo a real-world workflow."""
    print("\n\n🎯 Real-World Workflow Demo")
    print("=" * 70)
    print("\nScenario: Migrating from print() to proper logging")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        setup_demo_files(temp_path)

        tool = BatchProcessorTool(max_workers=4)

        print("\n1️⃣ STEP 1: Find all print statements...")
        result = await tool.execute(
            operation="search",
            path=str(temp_path),
            pattern="print\\(",
            file_pattern="**/*.py",
            regex=True,
        )

        if result.success:
            print(result.output)

        print("\n2️⃣ STEP 2: Preview the replacement...")
        result = await tool.execute(
            operation="replace",
            path=str(temp_path),
            find="print(",
            replace="logger.info(",
            file_pattern="**/*.py",
            dry_run=True,
        )

        if result.success:
            print(result.output)

        print("\n3️⃣ STEP 3: Execute the replacement...")
        result = await tool.execute(
            operation="replace",
            path=str(temp_path),
            find="print(",
            replace="logger.info(",
            file_pattern="**/*.py",
            dry_run=False,
        )

        if result.success:
            print(result.output)

        print("\n4️⃣ STEP 4: Verify no print statements remain...")
        result = await tool.execute(
            operation="search",
            path=str(temp_path),
            pattern="print\\(",
            file_pattern="**/*.py",
            regex=True,
        )

        if result.success:
            if "Found in 0 files" in result.output:
                print("✅ SUCCESS: All print statements replaced with logger.info!")
            print(result.output)

        print("\n5️⃣ STEP 5: Analyze final state...")
        result = await tool.execute(
            operation="analyze",
            path=str(temp_path),
            file_pattern="**/*.py",
        )

        if result.success:
            print(result.output)


async def demo_performance():
    """Demo parallel processing performance."""
    print("\n\n⚡ Performance Demo")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create many files
        print("\n1️⃣ Creating 100 demo files...")
        for i in range(100):
            file_path = temp_path / f"file_{i}.py"
            content = f"""
# File {i}
def function_{i}():
    print("Function {i}")
    # TODO: Implement logic
    return {i}
"""
            file_path.write_text(content)

        print("✓ Created 100 files")

        print("\n2️⃣ Searching across all files with parallel processing...")
        tool = BatchProcessorTool(max_workers=8)

        import time
        start = time.time()

        result = await tool.execute(
            operation="search",
            path=str(temp_path),
            pattern="TODO",
            file_pattern="*.py",
        )

        elapsed = time.time() - start

        if result.success:
            print(f"\n⏱️  Processed 100 files in {elapsed:.3f} seconds")
            print(f"📈 Throughput: {100/elapsed:.1f} files/second")
            print(f"\n{result.output}")


async def main():
    """Run all batch processing demos."""
    print("🎯 Victor Batch Processing Tool Demo")
    print("=" * 70)
    print("\nDemonstrating efficient multi-file operations\n")

    # Run demos
    await demo_batch_search()
    await demo_batch_replace()
    await demo_batch_analyze()
    await demo_list_files()
    await demo_real_world_workflow()
    await demo_performance()

    print("\n\n✨ Demo Complete!")
    print("\nVictor's Batch Processing Tool provides:")
    print("  • Parallel file processing (configurable workers)")
    print("  • Pattern-based file selection (glob patterns)")
    print("  • Regex support for advanced search/replace")
    print("  • Dry-run mode for safe previews")
    print("  • Progress tracking and error handling")
    print("  • Multi-file analysis and reporting")
    print("\nPerfect for:")
    print("  • Code refactoring across projects")
    print("  • Finding security issues or code smells")
    print("  • Bulk updates (API migrations, etc.)")
    print("  • Codebase analysis and metrics")
    print("  • Documentation updates")
    print("\nReady for enterprise use!")


if __name__ == "__main__":
    asyncio.run(main())
