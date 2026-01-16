#!/usr/bin/env python3
"""
Check cyclomatic complexity of Python code.

Usage: check_complexity.py <file_path>
"""
import ast
import sys
from pathlib import Path


class ComplexityChecker(ast.NodeVisitor):
    """AST visitor to check cyclomatic complexity."""

    def __init__(self, max_complexity=10):
        """Initialize the checker."""
        self.max_complexity = max_complexity
        self.function_complexity = {}
        self.current_function = None
        self.complexity = 1

    def visit_FunctionDef(self, node):
        """Visit a function definition."""
        old_function = self.current_function
        old_complexity = self.complexity

        self.current_function = f"{node.name}"
        self.complexity = 1

        self.generic_visit(node)

        complexity = self.complexity
        self.function_complexity[self.current_function] = complexity

        self.current_function = old_function
        self.complexity = old_complexity

    def visit_AsyncFunctionDef(self, node):
        """Visit an async function definition."""
        self.visit_FunctionDef(node)

    def visit_If(self, node):
        """Visit an if statement."""
        self.complexity += 1
        self.generic_visit(node)

    def visit_While(self, node):
        """Visit a while loop."""
        self.complexity += 1
        self.generic_visit(node)

    def visit_For(self, node):
        """Visit a for loop."""
        self.complexity += 1
        self.generic_visit(node)

    def visit_Try(self, node):
        """Visit a try statement."""
        self.complexity += len(node.handlers)
        self.generic_visit(node)

    def visit_With(self, node):
        """Visit a with statement."""
        self.complexity += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        """Visit an except handler."""
        if node.type is not None:
            self.complexity += 1
        self.generic_visit(node)


def check_file(file_path, max_complexity=10):
    """Check a Python file for complexity issues."""
    with open(file_path, "r") as f:
        try:
            tree = ast.parse(f.read(), filename=str(file_path))
        except SyntaxError:
            return 0, []  # Skip files with syntax errors

    checker = ComplexityChecker(max_complexity=max_complexity)
    checker.visit(tree)

    issues = []
    for func, complexity in checker.function_complexity.items():
        if complexity > max_complexity:
            issues.append(f"  {func}: complexity {complexity} (max: {max_complexity})")

    return len(issues), issues


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: check_complexity.py <file_path>")
        sys.exit(1)

    file_path = Path(sys.argv[1])

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    count, issues = check_file(file_path)

    if count > 0:
        print(f"Found {count} complexity issues in {file_path}:")
        for issue in issues:
            print(issue)

        # Don't fail, just warn
        sys.exit(0)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
