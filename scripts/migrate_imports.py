#!/usr/bin/env python3
"""
Migration script to update imports from victor.* to victor_coding.*.

This script helps migrate codebases from the old import structure to the new
decoupled package structure.

Usage:
    python scripts/migrate_imports.py --check          # Dry run, show what would change
    python scripts/migrate_imports.py --apply          # Apply changes
    python scripts/migrate_imports.py --path /my/code  # Migrate specific directory

Example:
    # Check what imports need updating
    python scripts/migrate_imports.py --check --path .

    # Apply the changes
    python scripts/migrate_imports.py --apply --path .
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Modules that have moved from victor.* to victor_coding.*
MIGRATED_MODULES = [
    "codebase",
    "lsp",
    "languages",
    "editing",
    "testgen",
    "refactor",
    "review",
    "security",
    "docgen",
    "completion",
    "deps",
    "coverage",
]

# Patterns to match and replace
# Note: Use non-capturing group for submodules to capture full path
IMPORT_PATTERNS = [
    # from victor.module.submodule import X
    (
        r"from victor\.({modules})((?:\.\w+)*) import",
        r"from victor_coding.\1\2 import",
    ),
    # import victor.module.submodule
    (
        r"import victor\.({modules})((?:\.\w+)*)",
        r"import victor_coding.\1\2",
    ),
    # from victor import module (less common but possible)
    # This is trickier and may need manual review
]


def compile_patterns() -> List[Tuple[re.Pattern, str]]:
    """Compile regex patterns with module list."""
    modules_pattern = "|".join(MIGRATED_MODULES)
    compiled = []
    for pattern, replacement in IMPORT_PATTERNS:
        compiled_pattern = re.compile(pattern.format(modules=modules_pattern))
        compiled.append((compiled_pattern, replacement))
    return compiled


def find_python_files(path: Path) -> List[Path]:
    """Find all Python files in the given path."""
    if path.is_file():
        return [path] if path.suffix == ".py" else []

    files = []
    for py_file in path.rglob("*.py"):
        # Skip common directories to ignore
        parts = py_file.parts
        if any(
            skip in parts
            for skip in [
                "__pycache__",
                ".git",
                ".venv",
                "venv",
                "node_modules",
                "build",
                "dist",
                ".eggs",
                "*.egg-info",
            ]
        ):
            continue
        files.append(py_file)

    return sorted(files)


def analyze_file(
    file_path: Path, patterns: List[Tuple[re.Pattern, str]]
) -> List[Tuple[int, str, str]]:
    """Analyze a file for imports that need updating.

    Returns:
        List of (line_number, old_line, new_line) tuples
    """
    changes = []
    try:
        content = file_path.read_text()
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}", file=sys.stderr)
        return changes

    lines = content.split("\n")
    for i, line in enumerate(lines, 1):
        for pattern, replacement in patterns:
            if pattern.search(line):
                new_line = pattern.sub(replacement, line)
                if new_line != line:
                    changes.append((i, line, new_line))
                    break  # Only one replacement per line

    return changes


def apply_changes(file_path: Path, changes: List[Tuple[int, str, str]]) -> bool:
    """Apply import changes to a file.

    Returns:
        True if changes were applied, False otherwise
    """
    if not changes:
        return False

    try:
        content = file_path.read_text()
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return False

    # Apply replacements
    for _, old_line, new_line in changes:
        content = content.replace(old_line, new_line, 1)

    try:
        file_path.write_text(content)
        return True
    except Exception as e:
        print(f"Error writing {file_path}: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Migrate imports from victor.* to victor_coding.*")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check for imports that need updating (dry run)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply import changes",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("."),
        help="Path to migrate (default: current directory)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output",
    )

    args = parser.parse_args()

    if not args.check and not args.apply:
        parser.error("Please specify either --check or --apply")

    if args.check and args.apply:
        parser.error("Cannot specify both --check and --apply")

    # Compile patterns
    patterns = compile_patterns()

    # Find Python files
    files = find_python_files(args.path)

    if not files:
        print(f"No Python files found in {args.path}")
        return 0

    print(f"Scanning {len(files)} Python files...")

    total_changes = 0
    files_with_changes = 0

    for file_path in files:
        changes = analyze_file(file_path, patterns)

        if changes:
            files_with_changes += 1
            total_changes += len(changes)

            if args.verbose or args.check:
                print(f"\n{file_path}:")
                for line_num, old_line, new_line in changes:
                    print(f"  Line {line_num}:")
                    print(f"    - {old_line.strip()}")
                    print(f"    + {new_line.strip()}")

            if args.apply:
                if apply_changes(file_path, changes):
                    print(f"  Updated {file_path} ({len(changes)} changes)")
                else:
                    print(f"  Failed to update {file_path}")

    # Summary
    print(f"\n{'=' * 60}")
    if args.check:
        print("DRY RUN - No changes made")
    print(f"Files scanned: {len(files)}")
    print(f"Files with deprecated imports: {files_with_changes}")
    print(f"Total imports to migrate: {total_changes}")

    if args.check and total_changes > 0:
        print("\nRun with --apply to update imports")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
