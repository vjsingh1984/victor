#!/usr/bin/env python3
"""
File Operations Demo - High-Performance Parallel File System Operations

This script demonstrates the capabilities of the Rust-based file operations
module, which provides 2-3x faster directory traversal and 3-5x faster
metadata collection compared to pure Python implementations.

Performance Characteristics:
- walk_directory_parallel: 2-3x faster than os.walk
- collect_metadata: 3-5x faster than individual stat calls
- filter_by_extension: Near-instant set-based filtering
- filter_by_size: Parallel filtering with Rayon
- get_directory_stats: Batch statistics collection

Usage:
    python examples/file_operations_demo.py
"""

import os
import sys
import time
from pathlib import Path
from typing import List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from victor.native.rust import file_ops

    RUST_AVAILABLE = file_ops.RUST_AVAILABLE
except ImportError:
    RUST_AVAILABLE = False
    print("ERROR: Rust extension not available!")
    print("Install with: pip install victor-ai[native]")
    sys.exit(1)


def demo_basic_walk():
    """Demonstrate basic directory walking."""
    print("\n" + "=" * 70)
    print("DEMO 1: Basic Directory Walking")
    print("=" * 70)

    # Walk current directory
    print("\nWalking 'victor' directory...")
    files = file_ops.walk_directory(
        "victor", patterns=["*.py"], max_depth=2, ignore_patterns=["__pycache__", "*.pyc"]
    )

    print(f"Found {len(files)} Python files:")
    for f in files[:5]:  # Show first 5
        print(f"  - {f.path} ({f.size} bytes, depth {f.depth})")

    if len(files) > 5:
        print(f"  ... and {len(files) - 5} more")


def demo_pattern_matching():
    """Demonstrate glob pattern matching."""
    print("\n" + "=" * 70)
    print("DEMO 2: Glob Pattern Matching")
    print("=" * 70)

    # Find different file types
    patterns = {
        "Python files": ["*.py"],
        "Rust files": ["*.rs"],
        "Markdown files": ["*.md"],
        "Config files": ["*.toml", "*.yaml", "*.yml"],
    }

    for name, pattern in patterns.items():
        files = file_ops.walk_directory(
            ".", patterns=pattern, max_depth=1, ignore_patterns=["__pycache__", ".git"]
        )
        print(f"\n{name}: {len(files)} files")
        if files:
            for f in files[:3]:
                print(f"  - {Path(f.path).name}")


def demo_parallel_traversal():
    """Demonstrate parallel traversal performance."""
    print("\n" + "=" * 70)
    print("DEMO 3: Parallel Traversal Performance")
    print("=" * 70)

    # Walk with different depths
    for max_depth in [1, 3, 5]:
        start = time.time()
        files = file_ops.walk_directory(
            "victor",
            patterns=["*"],
            max_depth=max_depth,
            ignore_patterns=["__pycache__", "*.pyc", ".git"],
        )
        elapsed = time.time() - start

        print(f"\nMax depth {max_depth}: {len(files)} entries in {elapsed:.3f}s")


def demo_filtering():
    """Demonstrate file filtering capabilities."""
    print("\n" + "=" * 70)
    print("DEMO 4: File Filtering")
    print("=" * 70)

    # Get all files
    all_files = file_ops.walk_directory(
        ".", patterns=["*"], max_depth=1, ignore_patterns=["__pycache__", ".git", "*.so"]
    )

    print(f"\nTotal files found: {len(all_files)}")

    # Filter by extension
    py_files = file_ops.filter_files_by_extension(all_files, ["py"])
    print(f"Python files: {len(py_files)}")

    # Filter by size
    small_files = file_ops.filter_files_by_size(all_files, max_size=1024)
    medium_files = file_ops.filter_files_by_size(all_files, min_size=1024, max_size=10 * 1024)
    large_files = file_ops.filter_files_by_size(all_files, min_size=10 * 1024)

    print(f"Small files (< 1KB): {len(small_files)}")
    print(f"Medium files (1KB-10KB): {len(medium_files)}")
    print(f"Large files (> 10KB): {len(large_files)}")


def demo_metadata_collection():
    """Demonstrate metadata collection."""
    print("\n" + "=" * 70)
    print("DEMO 5: Metadata Collection")
    print("=" * 70)

    # Get some Python files
    py_files = file_ops.walk_directory(
        "victor", patterns=["*.py"], max_depth=2, ignore_patterns=["__pycache__"]
    )

    if py_files:
        paths = [f.path for f in py_files[:10]]  # First 10 files

        print(f"\nCollecting metadata for {len(paths)} files...")
        start = time.time()
        metadata = file_ops.get_file_metadata(paths)
        elapsed = time.time() - start

        print(f"Collected metadata in {elapsed:.3f}s")

        print("\nSample metadata:")
        for m in metadata[:3]:
            print(f"  - {Path(m.path).name}")
            print(f"    Size: {m.size} bytes")
            print(f"    Modified: {m.modified}")
            print(
                f"    Permissions: file={m.is_file}, dir={m.is_dir}, "
                f"symlink={m.is_symlink}, readonly={m.is_readonly}"
            )


def demo_directory_stats():
    """Demonstrate directory statistics."""
    print("\n" + "=" * 70)
    print("DEMO 6: Directory Statistics")
    print("=" * 70)

    # Get stats for victor directory
    print("\nAnalyzing 'victor' directory...")
    stats = file_ops.get_directory_statistics("victor", max_depth=3)

    print(
        f"\nTotal size: {stats['total_size']:,} bytes "
        f"({stats['total_size'] / 1024 / 1024:.2f} MB)"
    )
    print(f"File count: {stats['file_count']:,}")
    print(f"Directory count: {stats['dir_count']:,}")

    print("\nLargest files:")
    for path, size in stats["largest_files"][:5]:
        print(f"  - {Path(path).name}: {size:,} bytes")


def demo_grouping():
    """Demonstrate grouping by directory."""
    print("\n" + "=" * 70)
    print("DEMO 7: Group Files by Directory")
    print("=" * 70)

    # Get Python files
    py_files = file_ops.walk_directory(
        "victor", patterns=["*.py"], max_depth=2, ignore_patterns=["__pycache__"]
    )

    # Group by directory
    grouped = file_ops.group_files_by_directory(py_files)

    print(f"\nFound {len(grouped)} directories with Python files")

    # Show top 5 directories
    sorted_dirs = sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True)[:5]

    for dir_path, files in sorted_dirs:
        print(f"\n{dir_path}:")
        print(f"  {len(files)} Python files")
        for f in files[:3]:
            print(f"    - {Path(f.path).name}")


def demo_time_filtering():
    """Demonstrate time-based filtering."""
    print("\n" + "=" * 70)
    print("DEMO 8: Time-Based Filtering")
    print("=" * 70)

    import time as time_module

    # Get recent files
    one_day_ago = int(time_module.time()) - 86400
    one_week_ago = int(time_module.time()) - 604800

    all_files = file_ops.walk_directory(
        ".", patterns=["*.py", "*.md", "*.txt"], max_depth=1, ignore_patterns=["__pycache__"]
    )

    recent_files = file_ops.filter_files_by_modified_time(all_files, since=one_week_ago)

    print(f"\nFiles modified in last week: {len(recent_files)}")

    very_recent = file_ops.filter_files_by_modified_time(all_files, since=one_day_ago)

    print(f"Files modified in last 24 hours: {len(very_recent)}")

    if very_recent:
        print("\nRecently modified files:")
        for f in very_recent[:5]:
            if f.modified:
                import datetime

                mod_time = datetime.datetime.fromtimestamp(f.modified)
                print(f"  - {Path(f.path).name}: {mod_time.strftime('%Y-%m-%d %H:%M')}")


def demo_code_discovery():
    """Demonstrate code file discovery."""
    print("\n" + "=" * 70)
    print("DEMO 9: Code File Discovery")
    print("=" * 70)

    # Find all code files
    print("\nFinding code files in 'victor' directory...")
    code_files = file_ops.find_code_files("victor", max_depth=2)

    # Group by extension
    ext_counts = {}
    for f in code_files:
        ext = Path(f.path).suffix
        ext_counts[ext] = ext_counts.get(ext, 0) + 1

    print(f"\nFound {len(code_files)} code files")
    print("\nBreakdown by extension:")
    for ext, count in sorted(ext_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {ext or '(no extension)'}: {count} files")


def demo_performance_comparison():
    """Compare Rust vs Python performance."""
    print("\n" + "=" * 70)
    print("DEMO 10: Performance Comparison (Rust vs Python)")
    print("=" * 70)

    import os

    # Rust implementation
    print("\nRust implementation:")
    start = time.time()
    rust_files = file_ops.walk_directory(
        "victor", patterns=["*.py"], max_depth=3, ignore_patterns=["__pycache__", "*.pyc"]
    )
    rust_time = time.time() - start
    print(f"  Found {len(rust_files)} files in {rust_time:.3f}s")

    # Python implementation (os.walk)
    print("\nPython implementation (os.walk):")
    start = time.time()
    python_files = []
    for root, dirs, files in os.walk("victor"):
        # Filter out ignored directories
        dirs[:] = [d for d in dirs if d not in ["__pycache__", ".git", ".pytest_cache"]]

        for file in files:
            if file.endswith(".py") and not file.endswith(".pyc"):
                python_files.append(os.path.join(root, file))
    python_time = time.time() - start
    print(f"  Found {len(python_files)} files in {python_time:.3f}s")

    # Speedup
    if python_time > 0:
        speedup = python_time / rust_time
        print(f"\nSpeedup: {speedup:.2f}x")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("File Operations Demo - High-Performance Rust Extensions")
    print("=" * 70)
    print(f"\nRust extension available: {RUST_AVAILABLE}")

    if not RUST_AVAILABLE:
        print("\nERROR: Rust extension not available!")
        print("Install with: pip install victor-ai[native]")
        return

    try:
        demo_basic_walk()
        demo_pattern_matching()
        demo_parallel_traversal()
        demo_filtering()
        demo_metadata_collection()
        demo_directory_stats()
        demo_grouping()
        demo_time_filtering()
        demo_code_discovery()
        demo_performance_comparison()

        print("\n" + "=" * 70)
        print("All demos completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
