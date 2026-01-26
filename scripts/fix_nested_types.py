#!/usr/bin/env python3
"""
Fix nested type parameters like Dict[str, Callable] -> Dict[str, Callable[..., Any]]
"""
import re
import sys
from pathlib import Path

# Nested type fixes
NESTED_FIXES = [
    (r"Dict\[str, Callable\]", "Dict[str, Callable[..., Any]]"),
    (r"dict\[str, Callable\]", "dict[str, Callable[..., Any]]"),
    (r"Dict\[str, Dict\]", "Dict[str, Dict[str, Any]]"),
    (r"dict\[str, dict\]", "dict[str, dict[str, Any]]"),
    (r"Dict\[str, List\]", "Dict[str, List[Any]]"),
    (r"dict\[str, list\]", "dict[str, list[Any]]"),
    (r"List\[Dict\]", "List[Dict[str, Any]]"),
    (r"list\[dict\]", "list[dict[str, Any]]"),
    (r"List\[List\]", "List[List[Any]]"),
    (r"list\[list\]", "list[list[Any]]"),
    (r"Dict\[str, Pattern\]", "Dict[str, Pattern[str]]"),
    (r"dict\[str, Pattern\]", "dict[str, Pattern[str]]"),
]


def fix_file(file_path: Path) -> int:
    """Fix nested type parameters in a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content

        for pattern, replacement in NESTED_FIXES:
            content = re.sub(pattern, replacement, content)

        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return 1

        return 0
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)
        return 0


def main():
    """Main function."""
    victor_dir = Path("/Users/vijaysingh/code/codingagent/victor")

    # Find all Python files
    python_files = list(victor_dir.rglob("*.py"))

    print(f"Processing {len(python_files)} Python files for nested type fixes...")
    print()

    total_fixed = 0

    for file_path in sorted(python_files):
        if fix_file(file_path):
            total_fixed += 1
            print(f"Fixed: {file_path.relative_to(victor_dir)}")

    print(f"\nTotal files fixed: {total_fixed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
