#!/usr/bin/env python3
"""Check documentation file sizes against limits."""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# File size limits by content type (in lines)
SIZE_LIMITS = {
    "quick-start": 300,
    "how-to": 500,
    "reference": 700,
    "architecture": 800,
    "tutorial": 600,
    "journey": 1000,  # Journeys can be longer
}


def categorize_file(file_path: Path) -> str:
    """Categorize file based on path and name."""
    path_str = str(file_path)

    # Skip archive
    if "docs/archive/" in path_str:
        return "skip"

    # Journeys
    if "docs/journeys/" in path_str:
        return "journey"

    # Getting started
    if "docs/getting-started/" in path_str:
        return "quick-start"

    # Architecture
    if "docs/architecture/" in path_str:
        return "architecture"

    # Reference
    if "docs/reference/" in path_str:
        return "reference"

    # Tutorials
    if "docs/tutorials/" in path_str or file_path.name.startswith("tutorial"):
        return "tutorial"

    # How-to guides
    if "docs/guides/" in path_str:
        return "how-to"

    # Default to how-to limit
    return "how-to"


def count_lines(file_path: Path) -> int:
    """Count non-empty lines in file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for line in f if line.strip())
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return 0


def check_file_sizes(docs_dir: Path) -> List[Tuple[Path, int, int, str]]:
    """Check all markdown files in docs directory."""
    violations = []

    for md_file in docs_dir.rglob("*.md"):
        category = categorize_file(md_file)

        if category == "skip":
            continue

        line_count = count_lines(md_file)
        limit = SIZE_LIMITS.get(category, SIZE_LIMITS["how-to"])

        if line_count > limit:
            violations.append((md_file, line_count, limit, category))

    return violations


def main():
    """Main entry point."""
    docs_dir = Path("docs")

    if not docs_dir.exists():
        print(f"Error: {docs_dir} directory not found", file=sys.stderr)
        sys.exit(1)

    violations = check_file_sizes(docs_dir)

    if violations:
        print("❌ File size limit violations:", file=sys.stderr)
        print()

        for file_path, line_count, limit, category in sorted(
            violations,
            key=lambda x: x[1] - x[2],
            reverse=True
        ):
            excess = line_count - limit
            print(
                f"  {file_path}: {line_count} lines "
                f"(limit: {limit} for {category}, excess: {excess} lines)",
                file=sys.stderr
            )

        print()
        print(f"Total violations: {len(violations)}", file=sys.stderr)
        print()
        print("To fix:", file=sys.stderr)
        print("  1. Split large files into multiple focused documents", file=sys.stderr)
        print("  2. Move detailed content to subdirectories", file=sys.stderr)
        print("  3. Reference related files instead of duplicating content", file=sys.stderr)
        sys.exit(1)
    else:
        print("✅ All files within size limits")
        sys.exit(0)


if __name__ == "__main__":
    main()
