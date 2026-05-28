"""Architecture guardrails for externalized vertical imports."""

from __future__ import annotations

import ast
from pathlib import Path

_FORBIDDEN_PREFIXES = (
    "victor.coding",
    "victor.devops",
    "victor.research",
    "victor.rag",
    "victor.dataanalysis",
    "victor.verticals.contrib.coding",
)

_ALLOWED_IMPORT_FILES = {
    Path("core/utils/coding_support.py"),
}


def _iter_runtime_python_files() -> list[Path]:
    repo_root = Path(__file__).resolve().parents[4]
    runtime_root = repo_root / "victor"
    return sorted(runtime_root.rglob("*.py"))


def test_runtime_code_has_no_direct_legacy_vertical_imports() -> None:
    """Core/framework runtime should rely on resolver-based loading instead of direct imports."""
    repo_root = Path(__file__).resolve().parents[4]
    runtime_root = repo_root / "victor"
    violations: list[str] = []

    for path in _iter_runtime_python_files():
        rel = path.relative_to(runtime_root)

        # Contrib vertical packages are the source modules and can import internally.
        if len(rel.parts) >= 2 and rel.parts[:2] == ("verticals", "contrib"):
            continue
        if rel in _ALLOWED_IMPORT_FILES:
            continue

        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(module):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for prefix in _FORBIDDEN_PREFIXES:
                        if alias.name == prefix or alias.name.startswith(f"{prefix}."):
                            violations.append(
                                f"{rel}:{node.lineno} import {alias.name}"
                            )
            elif isinstance(node, ast.ImportFrom):
                if not node.module:
                    continue
                for prefix in _FORBIDDEN_PREFIXES:
                    if node.module == prefix or node.module.startswith(f"{prefix}."):
                        violations.append(
                            f"{rel}:{node.lineno} from {node.module} import ..."
                        )

    assert not violations, "Direct legacy vertical imports found:\n" + "\n".join(
        violations
    )
