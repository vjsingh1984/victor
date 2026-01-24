#!/usr/bin/env python3
"""
Improved script to fix all missing type parameters for generic types.
Uses AST parsing for more accurate fixes.
"""
import ast
import re
import sys
from pathlib import Path
from typing import List, Set, Tuple

# Type mappings for common bare generics
TYPE_MAPPINGS = {
    'dict': 'dict[str, Any]',
    'Dict': 'Dict[str, Any]',
    'list': 'list[Any]',
    'List': 'List[Any]',
    'tuple': 'tuple[Any, ...]',
    'Tuple': 'Tuple[Any, ...]',
    'set': 'set[Any]',
    'Set': 'Set[Any]',
    'frozenset': 'frozenset[Any]',
    'Callable': 'Callable[..., Any]',
    'Type': 'Type[Any]',
    'Pattern': 'Pattern[str]',
    'Counter': 'Counter[str]',
    'ItemsView': 'ItemsView[Any, Any]',
    'ObjectPool': 'ObjectPool[Any]',
    'LRUCache': 'LRUCache[Any, Any]',
    'TimedCache': 'TimedCache[Any, Any]',
    'SingletonRegistry': 'SingletonRegistry[Any]',
    'UniversalRegistry': 'UniversalRegistry[Any]',
    'ServiceDescriptor': 'ServiceDescriptor[Any]',
    'weakref.ref': 'weakref.ref[Any]',
    'CompletedProcess': 'CompletedProcess[Any]',
}

def has_type_params(subscript: ast.Subscript) -> bool:
    """Check if a subscript already has type parameters."""
    return bool(subscript.slice)

def fix_file_with_ast(file_path: Path) -> Tuple[int, List[str]]:
    """Fix type parameters using AST for accuracy."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse the file
        try:
            tree = ast.parse(content, filename=str(file_path))
        except SyntaxError:
            return 0, []

        # Find all problematic type annotations
        class TypeAnnotationVisitor(ast.NodeVisitor):
            def __init__(self):
                self.fixes = []

            def visit_AnnAssign(self, node):
                if node.annotation:
                    self._check_annotation(node.annotation)
                self.generic_visit(node)

            def visit_FunctionDef(self, node):
                # Check return type
                if node.returns:
                    self._check_annotation(node.returns)

                # Check argument annotations
                for arg in node.args.args:
                    if arg.annotation:
                        self._check_annotation(arg.annotation)

                # Check keyword argument annotations
                for arg in node.args.kwonlyargs:
                    if arg.annotation:
                        self._check_annotation(arg.annotation)

                self.generic_visit(node)

            def visit_AsyncFunctionDef(self, node):
                # Same as FunctionDef
                if node.returns:
                    self._check_annotation(node.returns)

                for arg in node.args.args:
                    if arg.annotation:
                        self._check_annotation(arg.annotation)

                for arg in node.args.kwonlyargs:
                    if arg.annotation:
                        self._check_annotation(arg.annotation)

                self.generic_visit(node)

            def _check_annotation(self, annotation):
                if isinstance(annotation, ast.Subscript):
                    # Get the type name
                    if isinstance(annotation.value, ast.Name):
                        type_name = annotation.value.id
                    elif isinstance(annotation.value, ast.Attribute):
                        type_name = f"{annotation.value.value.id}.{annotation.value.attr}"
                    else:
                        return

                    # Check if it's a bare generic (empty slice)
                    if type_name in TYPE_MAPPINGS:
                        slice_content = annotation.slice

                        # Check if slice is empty or missing type params
                        if isinstance(slice_content, ast.Tuple):
                            if not slice_content.elts:
                                self.fixes.append((type_name, TYPE_MAPPINGS[type_name]))
                        elif isinstance(slice_content, ast.Name) and slice_content.id == 'Any':
                            pass  # Already has type param
                        else:
                            # Check if we need to add type params
                            if type_name in ['dict', 'Dict'] and self._count_subscript_elements(slice_content) != 2:
                                self.fixes.append((type_name, TYPE_MAPPINGS[type_name]))
                            elif type_name in ['list', 'List', 'set', 'Set', 'frozenset'] and self._count_subscript_elements(slice_content) != 1:
                                self.fixes.append((type_name, TYPE_MAPPINGS[type_name]))
                            elif type_name == 'tuple' and self._count_subscript_elements(slice_content) < 2:
                                self.fixes.append((type_name, TYPE_MAPPINGS[type_name]))
                            elif type_name in TYPE_MAPPINGS and not hasattr(slice_content, 'elts'):
                                # Single element but should be more
                                if type_name in ['dict', 'Dict', 'ItemsView', 'LRUCache', 'TimedCache', 'tuple', 'Tuple']:
                                    self.fixes.append((type_name, TYPE_MAPPINGS[type_name]))

            def _count_subscript_elements(self, slice_node):
                """Count elements in a subscript slice."""
                if isinstance(slice_node, ast.Tuple):
                    return len(slice_node.elts)
                return 1

        visitor = TypeAnnotationVisitor()
        visitor.visit(tree)

        if not visitor.fixes:
            return 0, []

        # Apply fixes using text replacement
        new_content = content
        changes_made = []

        for old_type, new_type in visitor.fixes:
            # More sophisticated regex to avoid false matches
            pattern = rf':\s*{re.escape(old_type)}\s*\['
            replacement = f': {new_type}['

            if re.search(pattern, new_content):
                new_content = re.sub(pattern, replacement, new_content)
                changes_made.append(f"  - {old_type} -> {new_type}")

        if new_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return len(changes_made), changes_made

        return 0, []

    except Exception as e:
        print(f"  Error processing {file_path}: {e}", file=sys.stderr)
        return 0, []

def fix_file_with_regex(file_path: Path) -> Tuple[int, List[str]]:
    """Fallback: Use regex for simpler cases."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        changes_made = []

        # Fix bare generics without brackets
        for old_type, new_type in TYPE_MAPPINGS.items():
            # Look for type annotations without brackets
            # Match: : TypeName followed by space, comma, paren, bracket, or newline
            # But NOT: : TypeName[...]
            pattern = rf':\s*{re.escape(old_type)}\s*([,\)\]\}:\n])'
            replacement = f': {new_type}\\1'

            # Only apply if not already followed by [
            def replacer(match):
                suffix = match.group(1)
                return f': {new_type}{suffix}'

            matches = list(re.finditer(pattern, content))
            if matches:
                # Filter out matches that already have [ after the type name
                filtered_matches = []
                for m in matches:
                    # Check if there's already a [ after this occurrence
                    after_match = content[m.end():]
                    if not after_match.lstrip().startswith('['):
                        filtered_matches.append(m)

                if filtered_matches:
                    # Apply replacement only for filtered matches
                    for m in reversed(filtered_matches):  # Reverse to maintain positions
                        content = content[:m.start()] + replacer(m) + content[m.end():]
                    changes_made.append(f"  - {old_type} -> {new_type}: {len(filtered_matches)} occurrence(s)")

        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return len(changes_made), changes_made

        return 0, []

    except Exception as e:
        print(f"  Error processing {file_path}: {e}", file=sys.stderr)
        return 0, []

def main():
    """Main function."""
    victor_dir = Path('/Users/vijaysingh/code/codingagent/victor')

    if not victor_dir.exists():
        print(f"Error: {victor_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    # Get list of files with errors
    result = subprocess.run(
        ['mypy', str(victor_dir), '--config-file', 'pyproject.toml'],
        capture_output=True,
        text=True
    )

    error_files = set()
    for line in result.stdout.split('\n'):
        if 'type-arg' in line:
            match = re.match(r'^([^:]+):', line)
            if match:
                error_files.add(Path(match.group(1)))

    print(f"Found {len(error_files)} files with type parameter errors")
    print()

    total_files_modified = 0
    total_changes = 0

    for file_path in sorted(error_files):
        print(f"Processing: {file_path.relative_to(victor_dir)}")

        # Try AST-based fix first
        changes_count, changes = fix_file_with_ast(file_path)

        # Fall back to regex if AST didn't find anything
        if changes_count == 0:
            changes_count, changes = fix_file_with_regex(file_path)

        if changes_count > 0:
            total_files_modified += 1
            total_changes += changes_count
            print(f"  Modified ({changes_count} changes):")
            for change in changes:
                print(change)
        print()

    print(f"\nSummary:")
    print(f"  Files modified: {total_files_modified}")
    print(f"  Total change types applied: {total_changes}")
    print()

    return 0

if __name__ == '__main__':
    import subprocess
    sys.exit(main())
