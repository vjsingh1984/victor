#!/usr/bin/env python3
"""
Script to fix all missing type parameters for generic types in the Victor AI codebase.
"""
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Patterns to find and fix
PATTERNS = [
    # Pattern: (regex_pattern, replacement, description)
    # dict/Dict fixes
    (r'(\W): dict\s*([,\)\]\}])', r'\1: dict[str, Any]\2', 'bare dict to dict[str, Any]'),
    (r'(\W): Dict\s*([,\)\]\}])', r'\1: Dict[str, Any]\2', 'bare Dict to Dict[str, Any]'),

    # list/List fixes
    (r'(\W): list\s*([,\)\]\}])', r'\1: list[Any]\2', 'bare list to list[Any]'),
    (r'(\W): List\s*([,\)\]\}])', r'\1: List[Any]\2', 'bare List to List[Any]'),

    # tuple/Tuple fixes
    (r'(\W): tuple\s*([,\)\]\}])', r'\1: tuple[Any, ...]\2', 'bare tuple to tuple[Any, ...]'),
    (r'(\W): Tuple\s*([,\)\]\}])', r'\1: Tuple[Any, ...]\2', 'bare Tuple to Tuple[Any, ...]'),

    # Callable fixes
    (r'(\W): Callable\s*([,\)\]\}])', r'\1: Callable[..., Any]\2', 'bare Callable to Callable[..., Any]'),

    # Type fixes
    (r'(\W): Type\s*([,\)\]\}])', r'\1: Type[Any]\2', 'bare Type to Type[Any]'),

    # Pattern fixes
    (r'(\W): Pattern\s*([,\)\]\}])', r'\1: Pattern[str]\2', 'bare Pattern to Pattern[str]'),

    # Counter fixes
    (r'(\W): Counter\s*([,\)\]\}])', r'\1: Counter[str]\2', 'bare Counter to Counter[str]'),

    # ItemsView fixes
    (r'(\W): ItemsView\s*\[([^\]]+)\]', r'\1: ItemsView[\2, Any]', 'ItemsView with 1 arg to 2 args'),

    # ObjectPool fixes
    (r'(\W): ObjectPool\s*([,\)\]\}])', r'\1: ObjectPool[Any]\2', 'bare ObjectPool to ObjectPool[Any]'),

    # LRUCache fixes
    (r'(\W): LRUCache\s*([,\)\]\}])', r'\1: LRUCache[Any, Any]\2', 'bare LRUCache to LRUCache[Any, Any]'),

    # TimedCache fixes
    (r'(\W): TimedCache\s*([,\)\]\}])', r'\1: TimedCache[Any, Any]\2', 'bare TimedCache to TimedCache[Any, Any]'),
]

# Special patterns that need more context
SPECIAL_PATTERNS = [
    # dict with 3 args (should be 2)
    (r': dict\[([^,\]]+),\s*([^,\]]+),\s*([^\]]+)\]', r': dict[\1, \2]', 'dict with 3 args to 2 args'),
]

# Files to exclude from processing
EXCLUDE_FILES = {
    'scripts/fix_type_parameters.py',
    'scripts/solid_metrics.py',
    'scripts/verify_solid_deployment.py',
}

def should_process_file(file_path: Path) -> bool:
    """Check if file should be processed."""
    if file_path.name in EXCLUDE_FILES:
        return False
    if 'scripts/' in str(file_path) and file_path.name.startswith('fix_'):
        return False
    return True

def fix_file(file_path: Path) -> Tuple[int, List[str]]:
    """Fix type parameters in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        changes_made = []

        # Apply special patterns first (more specific)
        for pattern, replacement, desc in SPECIAL_PATTERNS:
            matches = list(re.finditer(pattern, content))
            if matches:
                content = re.sub(pattern, replacement, content)
                changes_made.append(f"  - {desc}: {len(matches)} occurrence(s)")

        # Apply general patterns
        for pattern, replacement, desc in PATTERNS:
            matches = list(re.finditer(pattern, content))
            if matches:
                content = re.sub(pattern, replacement, content)
                changes_made.append(f"  - {desc}: {len(matches)} occurrence(s)")

        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return len(changes_made), changes_made

        return 0, []
    except Exception as e:
        print(f"  Error processing {file_path}: {e}", file=sys.stderr)
        return 0, []

def main():
    """Main function to fix all Python files in the victor directory."""
    victor_dir = Path('/Users/vijaysingh/code/codingagent/victor')

    if not victor_dir.exists():
        print(f"Error: {victor_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    # Find all Python files
    python_files = list(victor_dir.rglob('*.py'))
    python_files = [f for f in python_files if should_process_file(f)]

    print(f"Found {len(python_files)} Python files to process")
    print()

    total_files_modified = 0
    total_changes = 0

    for file_path in sorted(python_files):
        changes_count, changes = fix_file(file_path)
        if changes_count > 0:
            total_files_modified += 1
            total_changes += changes_count
            print(f"Modified: {file_path.relative_to(victor_dir)}")
            for change in changes:
                print(change)
            print()

    print(f"\nSummary:")
    print(f"  Files modified: {total_files_modified}")
    print(f"  Total change types applied: {total_changes}")
    print()

    return 0

if __name__ == '__main__':
    sys.exit(main())
