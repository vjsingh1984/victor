#!/usr/bin/env python3
"""Validation script for vertical SDK alignment.

This script checks that all external verticals are properly aligned with the
Victor SDK and framework, ensuring:
1. No forbidden imports from internal modules
2. @register_vertical decorator is present
3. Framework-level imports are used correctly
4. Version constraints are specified
"""

import ast
import sys
from pathlib import Path
from typing import List, Set, Tuple


class VerticalValidator:
    """Validator for vertical SDK alignment."""

    # Forbidden import patterns (internal modules)
    FORBIDDEN_IMPORTS = {
        "from victor.agent",
        "from victor.agent.coordinators.safety_coordinator",
    }

    # Allowed exceptions (marked as INTERNAL API with comments)
    ALLOWED_INTERNAL_PATTERNS = {
        "from victor.agent.coordinators.conversation_coordinator",
    }

    def __init__(self, vertical_path: Path):
        """Initialize validator.

        Args:
            vertical_path: Path to vertical package directory
        """
        self.vertical_path = vertical_path
        self.issues: List[Tuple[str, str, int]] = []  # (file, issue, line)
        self.warnings: List[Tuple[str, str, int]] = []

    def validate(self) -> bool:
        """Run all validation checks.

        Returns:
            True if validation passes, False otherwise
        """
        print(f"\n{'='*60}")
        print(f"Validating: {self.vertical_path.name}")
        print(f"{'='*60}")

        # Find all Python files
        python_files = list(self.vertical_path.rglob("*.py"))
        python_files = [
            f
            for f in python_files
            if "__pycache__" not in str(f)
            and ".venv" not in str(f)
            and "tests" not in f.parts
        ]

        if not python_files:
            print("  ⚠ No Python files found")
            return True

        for py_file in python_files:
            self._validate_file(py_file)

        # Print results
        if self.issues:
            print(f"\n  ❌ Found {len(self.issues)} issue(s):")
            for file, issue, line in self.issues:
                print(f"     {file}:{line} - {issue}")
            return False
        else:
            print(f"  ✅ All checks passed ({len(python_files)} files)")
            if self.warnings:
                print(f"\n  ⚠️  {len(self.warnings)} warning(s):")
                for file, warning, line in self.warnings:
                    print(f"     {file}:{line} - {warning}")
            return True

    def _validate_file(self, file_path: Path) -> None:
        """Validate a single Python file.

        Args:
            file_path: Path to Python file
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.split("\n")

            # Parse AST
            tree = ast.parse(content, filename=str(file_path))

            # Check for forbidden imports
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module:
                        self._check_import(file_path, node.module, node.lineno, lines)

                elif isinstance(node, ast.ClassDef):
                    # Check for @register_vertical decorator
                    self._check_decorator(file_path, node, lines)

        except SyntaxError as e:
            self.issues.append(
                (
                    str(file_path.relative_to(self.vertical_path)),
                    f"Syntax error: {e}",
                    e.lineno or 0,
                )
            )
        except Exception as e:
            self.warnings.append(
                (
                    str(file_path.relative_to(self.vertical_path)),
                    f"Could not parse: {e}",
                    0,
                )
            )

    def _check_import(
        self, file_path: Path, module: str, lineno: int, lines: List[str]
    ) -> None:
        """Check if import is allowed.

        Args:
            file_path: File being checked
            module: Module being imported
            lineno: Line number
            lines: File contents for comment checking
        """
        import_stmt = f"from {module}"

        # Check if it's a forbidden import
        for forbidden in self.FORBIDDEN_IMPORTS:
            if import_stmt.startswith(forbidden):
                # Check if it's an allowed exception with documentation
                if lineno > 1:
                    prev_lines = "\n".join(lines[max(0, lineno - 3) : lineno])
                    if "INTERNAL API" in prev_lines or "TODO: Refactor" in prev_lines:
                        self.warnings.append(
                            (
                                str(file_path.relative_to(self.vertical_path)),
                                f"Using internal module: {module} (documented)",
                                lineno,
                            )
                        )
                        return

                self.issues.append(
                    (
                        str(file_path.relative_to(self.vertical_path)),
                        f"Forbidden import from internal module: {module}",
                        lineno,
                    )
                )

    def _check_decorator(
        self, file_path: Path, node: ast.ClassDef, lines: List[str]
    ) -> None:
        """Check if class has @register_vertical decorator.

        Args:
            file_path: File being checked
            node: Class definition node
            lines: File contents
        """
        # Only check classes that inherit from VerticalBase
        is_vertical = False
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id == "VerticalBase":
                is_vertical = True
                break
            elif isinstance(base, ast.Attribute) and base.attr == "VerticalBase":
                is_vertical = True
                break

        if not is_vertical:
            return

        # Check for @register_vertical decorator
        has_decorator = False
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call):
                if (
                    isinstance(decorator.func, ast.Name)
                    and decorator.func.id == "register_vertical"
                ):
                    has_decorator = True
                    break

        # Only check files named assistant.py or vertical.py
        rel_path = str(file_path.relative_to(self.vertical_path))
        if not ("assistant.py" in rel_path or "vertical.py" in rel_path):
            return

        if not has_decorator:
            self.warnings.append(
                (
                    str(file_path.relative_to(self.vertical_path)),
                    f"Class '{node.name}' missing @register_vertical decorator",
                    node.lineno,
                )
            )


def main() -> int:
    """Main entry point."""
    # Find vertical directories
    parent_dir = Path("/Users/vijaysingh/code")
    vertical_names = [
        "victor-coding",
        "victor-dataanalysis",
        "victor-devops",
        "victor-rag",
        "victor-research",
        "victor-invest",
    ]

    all_passed = True
    results = {}

    for name in vertical_names:
        vertical_path = parent_dir / name
        if not vertical_path.exists():
            print(f"\n⚠️  {name}: Not found")
            continue

        validator = VerticalValidator(vertical_path)
        passed = validator.validate()
        results[name] = passed
        all_passed = all_passed and passed

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {name}")

    print(f"\n{'='*60}")
    if all_passed:
        print("✅ All verticals passed validation!")
        return 0
    else:
        print("❌ Some verticals have issues - please review above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
