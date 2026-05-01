"""Guard tests: victor/ (core) must NOT import from external vertical packages.

This is the reverse of test_external_vertical_import_boundaries.py which checks
that verticals don't import from core. This file checks that core doesn't import
from verticals — even in try/except blocks or via importlib.import_module().

External vertical packages should be discovered via entry points and the
CapabilityRegistry, not imported by name.
"""

import ast
from pathlib import Path
from typing import List, Tuple

import pytest

# Root of the victor core package (not victor-sdk, not external verticals)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
VICTOR_ROOT = REPO_ROOT / "victor"

# External vertical package prefixes that core must never import
EXTERNAL_VERTICAL_PREFIXES = (
    "victor_coding",
    "victor_devops",
    "victor_research",
    "victor_rag",
    "victor_dataanalysis",
    "victor_invest",
)


def _collect_all_imports(root: Path) -> List[Tuple[str, int, str]]:
    """AST-walk all .py files under root and collect import statements.

    Returns list of (relative_path, line_number, module_name) for imports
    that match external vertical prefixes. Only collects actual ast.Import
    and ast.ImportFrom nodes — NOT string literals in docstrings.
    """
    violations = []
    for py_file in root.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        rel = str(py_file.relative_to(REPO_ROOT))
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=rel)
        except (SyntaxError, UnicodeDecodeError):
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if any(node.module.startswith(p) for p in EXTERNAL_VERTICAL_PREFIXES):
                    violations.append((rel, node.lineno, f"from {node.module}"))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if any(alias.name.startswith(p) for p in EXTERNAL_VERTICAL_PREFIXES):
                        violations.append((rel, node.lineno, f"import {alias.name}"))
    return violations


def _collect_importlib_calls(root: Path) -> List[Tuple[str, int, str]]:
    """Find importlib.import_module("victor_coding...") calls in AST.

    Returns list of (relative_path, line_number, module_string) for dynamic
    imports of external vertical packages.
    """
    violations = []
    for py_file in root.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        rel = str(py_file.relative_to(REPO_ROOT))
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=rel)
        except (SyntaxError, UnicodeDecodeError):
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            # Match importlib.import_module(...)
            func = node.func
            is_import_module = False
            if isinstance(func, ast.Attribute) and func.attr == "import_module":
                is_import_module = True
            elif isinstance(func, ast.Name) and func.id == "import_module":
                is_import_module = True
            if not is_import_module:
                continue
            # Check first positional argument is a string literal
            if (
                node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                mod_name = node.args[0].value
                if any(mod_name.startswith(p) for p in EXTERNAL_VERTICAL_PREFIXES):
                    violations.append((rel, node.lineno, f'import_module("{mod_name}")'))
    return violations


class TestCoreDoesNotImportExternalVerticals:
    """Ensure victor/ has zero static or dynamic imports from external verticals."""

    def test_no_static_imports_of_external_verticals_in_core(self):
        """AST-walk all .py under victor/ and assert zero import statements
        referencing external vertical packages (even try/except guarded ones).

        Docstring occurrences (string literals) are NOT flagged — only actual
        ast.Import / ast.ImportFrom nodes.
        """
        violations = _collect_all_imports(VICTOR_ROOT)
        assert not violations, (
            f"Core (victor/) has {len(violations)} static import(s) from external "
            f"verticals. Use entry points or CapabilityRegistry instead:\n"
            + "\n".join(f"  {f}:{line}: {mod}" for f, line, mod in violations)
        )

    def test_no_importlib_import_module_of_external_verticals(self):
        """Find importlib.import_module("victor_coding...") patterns.

        Dynamic imports by package name are still coupling — use
        CapabilityRegistry or entry point discovery instead.
        """
        violations = _collect_importlib_calls(VICTOR_ROOT)
        assert not violations, (
            f"Core (victor/) has {len(violations)} dynamic import(s) of external "
            f"verticals via importlib. Use CapabilityRegistry instead:\n"
            + "\n".join(f"  {f}:{line}: {mod}" for f, line, mod in violations)
        )
