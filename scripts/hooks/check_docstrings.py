#!/usr/bin/env python3
"""
Check that public functions and classes have docstrings.

Usage: check_docstrings.py <file_path>
"""
import ast
import sys
from pathlib import Path


def has_docstring(node):
    """Check if an AST node has a docstring."""
    return (
        ast.get_docstring(node) is not None
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module))
        else False
    )


def is_public(name):
    """Check if a name is public (doesn't start with underscore)."""
    return not name.startswith("_")


def check_file(file_path):
    """Check a Python file for missing docstrings."""
    with open(file_path, "r") as f:
        try:
            tree = ast.parse(f.read(), filename=str(file_path))
        except SyntaxError:
            return 0, []  # Skip files with syntax errors

    issues = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and is_public(node.name):
            if not has_docstring(node):
                issues.append(f"  Class {node.name} is missing docstring")

            # Check methods
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and is_public(item.name):
                    if not has_docstring(item):
                        issues.append(f"  Method {node.name}.{item.name} is missing docstring")

        elif isinstance(node, ast.FunctionDef) and is_public(node.name):
            if not has_docstring(node):
                issues.append(f"  Function {node.name} is missing docstring")

    return len(issues), issues


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: check_docstrings.py <file_path>")
        sys.exit(1)

    file_path = Path(sys.argv[1])

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    count, issues = check_file(file_path)

    if count > 0:
        print(f"Found {count} docstring issues in {file_path}:")
        for issue in issues:
            print(issue)

        # Don't fail, just warn
        sys.exit(0)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
