"""SDK boundary enforcement for external vertical definition layers.

Enforces that vertical DEFINITION files (assistant.py, plugin.py) have
no MODULE-LEVEL imports from victor.framework. Imports inside function
bodies, methods, TYPE_CHECKING blocks, and __getattr__ are allowed
(they're lazy/deferred and don't execute on module import).

Architecture: Plugin (bootstrap) -> Vertical (config) -> Extension (runtime)
See CLAUDE.md "Plugin -> Vertical -> Extension Architecture"
"""

import ast
import importlib
from pathlib import Path
from typing import List, Tuple

import pytest

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

DEFINITION_FILES = {"assistant.py", "plugin.py"}

VERTICALS = [
    "victor_invest",
    "victor_coding",
    "victor_research",
    "victor_devops",
    "victor_rag",
    "victor_dataanalysis",
]


def _find_source(pkg: str) -> Path:
    try:
        mod = importlib.import_module(pkg)
        return Path(mod.__path__[0]) if hasattr(mod, "__path__") else None
    except ImportError:
        return None


def _get_module_level_imports(source: str) -> List[Tuple[int, str]]:
    """Extract only MODULE-LEVEL imports (not inside functions/methods/if blocks)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    violations = []
    for node in ast.iter_child_nodes(tree):
        # Only check top-level statements
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith(BANNED_PREFIXES):
                violations.append((node.lineno, node.module))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(BANNED_PREFIXES):
                    violations.append((node.lineno, alias.name))
        # Check inside TYPE_CHECKING — these are OK (type-only)
        # Check inside if/try blocks at module level — still module-level
        elif isinstance(node, ast.If):
            # Skip TYPE_CHECKING blocks
            test = node.test
            if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
                continue
            if isinstance(test, ast.Attribute) and getattr(test, "attr", "") == "TYPE_CHECKING":
                continue
            # Other if blocks at module level: check their imports
            for child in ast.walk(node):
                if isinstance(child, ast.ImportFrom) and child.module:
                    if child.module.startswith(BANNED_PREFIXES):
                        violations.append((child.lineno, child.module))
    return violations


def _scan_definition_imports(source_dir: Path) -> List[Tuple[str, int, str]]:
    """Scan definition-layer files for module-level banned imports."""
    results = []
    for py_file in source_dir.rglob("*.py"):
        rel = str(py_file.relative_to(source_dir))
        if Path(rel).name not in DEFINITION_FILES:
            continue
        source = py_file.read_text(encoding="utf-8")
        for lineno, module in _get_module_level_imports(source):
            results.append((rel, lineno, module))
    return results


@pytest.mark.parametrize("pkg", VERTICALS)
def test_vertical_definition_layer_sdk_only(pkg: str):
    """Definition files (assistant.py, plugin.py) must not have module-level framework imports.

    Imports inside function bodies, TYPE_CHECKING blocks, and __getattr__
    are allowed (deferred/lazy).
    """
    src = _find_source(pkg)
    if src is None:
        pytest.skip(f"{pkg} not installed")

    violations = _scan_definition_imports(src)
    assert not violations, (
        f"{pkg} definition-layer files have module-level framework imports "
        f"(should use victor_sdk or defer to function bodies):\n"
        + "\n".join(f"  {f}:{line}: {mod}" for f, line, mod in violations)
    )
