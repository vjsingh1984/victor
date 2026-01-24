#!/usr/bin/env python3
"""
Comprehensive script to fix all missing type parameters.
"""
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

# All type replacements
TYPE_FIXES = {
    'Callable': ('Callable[..., Any]', None),
    'Dict': ('Dict[str, Any]', None),
    'dict': ('dict[str, Any]', None),
    'List': ('List[Any]', None),
    'list': ('list[Any]', None),
    'Tuple': ('Tuple[Any, ...]', None),
    'tuple': ('tuple[Any, ...]', None),
    'Set': ('Set[Any]', None),
    'set': ('set[Any]', None),
    'frozenset': ('frozenset[Any]', None),
    'Type': ('Type[Any]', None),
    'Pattern': ('Pattern[str]', None),
    'Counter': ('Counter[str]', None),
    'deque': ('deque[Any]', None),
    'OrderedDict': ('OrderedDict[str, Any]', None),
    'Task': ('Task[Any, Any]', 'from victor.framework import Task'),
    'SingletonRegistry': ('SingletonRegistry[Any]', None),
    'UniversalRegistry': ('UniversalRegistry[Any]', None),
    'ServiceDescriptor': ('ServiceDescriptor[Any]', None),
    'ItemsView': ('ItemsView[Any, Any]', None),
    'ObjectPool': ('ObjectPool[Any]', None),
    'LRUCache': ('LRUCache[Any, Any]', None),
    'TimedCache': ('TimedCache[Any, Any]', None),
    'weakref.ref': ('weakref.ref[Any]', None),
    'CompletedProcess': ('CompletedProcess[Any]', None),
}

def fix_file(file_path: Path) -> Tuple[int, List[str]]:
    """Fix type parameters in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        changes_made = []

        # Apply each fix
        for type_name, (replacement, _) in TYPE_FIXES.items():
            # Pattern to match: variable: Type followed by delimiter
            # But NOT: Type[...]
            delimiter_class = r'[,\)\]\}:\n]'
            pattern = rf':\s*{re.escape(type_name)}\s*({delimiter_class})'

            def replacer(match):
                # Ensure it's not already followed by [
                delimiter = match.group(1)
                # Check what comes after
                pos = match.end()
                if pos < len(content):
                    next_char = content[pos:pos+1].strip()
                    if next_char == '[':
                        # Already has type params, skip
                        return match.group(0)
                return f': {replacement}{delimiter}'

            # Find all matches
            matches = list(re.finditer(pattern, content))
            if matches:
                # Filter out matches that already have [
                valid_matches = []
                for m in matches:
                    pos = m.end()
                    # Check if next non-space char is [
                    while pos < len(content) and content[pos].isspace():
                        pos += 1
                    if pos >= len(content) or content[pos] != '[':
                        valid_matches.append(m)

                if valid_matches:
                    # Apply replacements in reverse order to maintain positions
                    for m in reversed(valid_matches):
                        delimiter = m.group(1)
                        old_text = m.group(0)
                        new_text = f': {replacement}{delimiter}'
                        content = content[:m.start()] + new_text + content[m.end():]

                    changes_made.append(f"  - {type_name} -> {replacement}: {len(valid_matches)} occurrence(s)")

        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return len(changes_made), changes_made

        return 0, []
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)
        return 0, []

def main():
    """Main function."""
    victor_dir = Path('/Users/vijaysingh/code/codingagent/victor')

    # Find all Python files
    python_files = list(victor_dir.rglob('*.py'))

    print(f"Processing {len(python_files)} Python files...")
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
