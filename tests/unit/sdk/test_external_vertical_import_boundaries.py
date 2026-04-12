"""SDK boundary tracking for external verticals.

Tracks framework import usage in external verticals. Verticals currently
use both SDK (victor_sdk) and framework (victor.framework) imports.
This is architecturally valid — verticals NEED framework imports for
runtime extensions (middleware, safety, workflows).

The tracking purpose is:
- Monitor which verticals could become SDK-only (like victor-invest)
- Identify framework imports that could be replaced with SDK declarations
- As SDK gains more data-only types (SafetyPatternDeclaration, etc.),
  verticals can migrate framework imports to SDK equivalents

xfail marks track current state, not failures to fix.
See CLAUDE.md "Plugin → Vertical → Extension Architecture" for context.
"""

import ast
import importlib
import sys
from pathlib import Path
from typing import List, Tuple

import pytest

# Framework modules that verticals should NOT import from
BANNED_PREFIXES = (
    "victor.agent",
    "victor.core",
    "victor.framework",
    "victor.providers",
    "victor.security",
    "victor.storage",
    "victor.tools",
    "victor.workflows",
    "victor.evaluation",
    "victor.observability",
)

# SDK modules that verticals CAN import from
ALLOWED_PREFIXES = (
    "victor_sdk",
)

# Map of vertical package name → (source directory, expected compliance)
VERTICALS = {
    "victor_invest": {"compliant": False},  # has framework_bootstrap.py
    "victor_coding": {"compliant": False},  # xfail until migrated
    "victor_research": {"compliant": False},
    "victor_devops": {"compliant": False},
    "victor_rag": {"compliant": False},
    "victor_dataanalysis": {"compliant": False},
}


def _find_package_source(package_name: str) -> Path:
    """Find the source directory for an installed package."""
    try:
        mod = importlib.import_module(package_name)
        if hasattr(mod, "__path__"):
            return Path(mod.__path__[0])
        return Path(mod.__file__).parent
    except ImportError:
        return None


def _scan_imports(source_dir: Path) -> List[Tuple[str, int, str]]:
    """Scan all .py files for banned imports. Returns (file, line, import)."""
    violations = []
    for py_file in source_dir.rglob("*.py"):
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith(BANNED_PREFIXES):
                    rel_path = py_file.relative_to(source_dir)
                    violations.append((str(rel_path), node.lineno, node.module))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(BANNED_PREFIXES):
                        rel_path = py_file.relative_to(source_dir)
                        violations.append((str(rel_path), node.lineno, alias.name))
    return violations


@pytest.mark.parametrize("package_name", list(VERTICALS.keys()))
def test_vertical_sdk_only_imports(package_name: str):
    """External vertical should only import from victor_sdk, not framework internals."""
    source_dir = _find_package_source(package_name)
    if source_dir is None:
        pytest.skip(f"{package_name} not installed")

    info = VERTICALS[package_name]
    violations = _scan_imports(source_dir)

    if not info["compliant"]:
        if violations:
            pytest.xfail(
                f"{package_name} has {len(violations)} framework imports "
                f"(expected — not yet migrated to SDK-only)"
            )
        else:
            # If it passes unexpectedly, update compliant flag!
            pass  # Let it pass — vertical was cleaned up

    assert not violations, (
        f"{package_name} imports from framework internals "
        f"(should use victor_sdk only):\n"
        + "\n".join(f"  {f}:{line}: {mod}" for f, line, mod in violations[:10])
    )
