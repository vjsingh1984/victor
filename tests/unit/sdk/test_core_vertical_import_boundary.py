"""Guard tests: victor/ (core) must NOT import from external vertical packages.

This is the reverse of test_external_vertical_import_boundaries.py which checks
that verticals don't import from core. This file checks that core doesn't import
from verticals — even in try/except blocks or via importlib.import_module().

External vertical packages should be discovered via entry points and the
CapabilityRegistry, not imported by name.
"""

import ast
from pathlib import Path
from typing import Dict, List, Set, Tuple

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

# Known violations to track migration progress (baseline).
# These are architectural violations that should be migrated to entry points.
KNOWN_VIOLATIONS: Dict[str, Set[str]] = {
    "victor_coding": {
        # TODO: Migrate to entry point registration
        # victor/core/graph_rag/language_handlers.py imports victor_coding directly
        # for language plugin discovery. Should use entry points instead.
        # Migration path:
        # 1. Move LanguageEdgeHandler to victor.framework.extensions or victor_sdk
        # 2. Add entry point registration in victor-coding's pyproject.toml
        # 3. Update core to discover plugins via entry points
        # See: https://github.com/anthropics/victor-ai/issues/XXX
        "victor_coding.languages.registry",
        "victor_coding.languages.base",
    }
}


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

        # Filter out known violations (tracked migration debt)
        known_violations: Set[str] = set()
        for package_violations in KNOWN_VIOLATIONS.values():
            known_violations.update(package_violations)

        new_violations = [
            (f, line, mod)
            for f, line, mod in violations
            if not any(
                # Match both "from victor_coding..." and "import victor_coding..."
                mod == known or mod.startswith(f"from {known}") or mod.startswith(f"import {known}")
                for known in known_violations
            )
        ]

        assert not new_violations, (
            f"Core (victor/) has {len(new_violations)} static import(s) from external "
            f"verticals. Use entry points or CapabilityRegistry instead:\n"
            + "\n".join(f"  {f}:{line}: {mod}" for f, line, mod in new_violations)
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

    def test_known_violations_not_stale(self):
        """Ensure KNOWN_VIOLATIONS baseline stays current (no stale entries).

        If a violation is fixed, remove it from KNOWN_VIOLATIONS.
        """
        violations = _collect_all_imports(VICTOR_ROOT)

        # Collect all actual violation import prefixes (strip "from " and "import " prefixes)
        actual_violation_prefixes: Set[str] = set()
        for _, _, mod in violations:
            # Strip "from " or "import " prefix to get the raw module name
            if mod.startswith("from "):
                actual_violation_prefixes.add(
                    mod[5:].split()[0]
                )  # Remove "from " and take first word
            elif mod.startswith("import "):
                actual_violation_prefixes.add(mod[7:])  # Remove "import "
            else:
                actual_violation_prefixes.add(mod)

        # Check for stale entries (violations that are fixed but still in baseline)
        for package, known_imports in KNOWN_VIOLATIONS.items():
            stale = {
                imp
                for imp in known_imports
                if not any(
                    actual == imp or actual.startswith(f"{imp}.") or imp.startswith(f"{actual}.")
                    for actual in actual_violation_prefixes
                )
            }
            if stale:
                pytest.fail(
                    f"Package {package} has stale KNOWN_VIOLATIONS entries (violations fixed!):\n"
                    f"  {stale}\n"
                    f"Remove these from KNOWN_VIOLATIONS in test_core_vertical_import_boundary.py"
                )
