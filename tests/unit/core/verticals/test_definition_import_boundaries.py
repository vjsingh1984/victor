"""Guardrails for migrated vertical definition-layer import boundaries."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
FORBIDDEN_DEFINITION_IMPORT_PREFIXES = (
    "victor.framework",
    "victor.core.verticals",
    "victor.tools.tool_names",
    "victor.framework.tool_naming",
)


@dataclass(frozen=True)
class DefinitionBoundaryTarget:
    """Definition-layer file that must stay free of runtime-only imports."""

    label: str
    path: Path
    allowed_victor_import_prefixes: tuple[str, ...] = ()
    requires_sdk_import: bool = False


DEFINITION_BOUNDARY_TARGETS = [
    DefinitionBoundaryTarget(
        label="coding assistant",
        path=REPO_ROOT / "victor/verticals/contrib/coding/assistant.py",
        requires_sdk_import=True,
    ),
    DefinitionBoundaryTarget(
        label="rag assistant",
        path=REPO_ROOT / "victor/verticals/contrib/rag/assistant.py",
        allowed_victor_import_prefixes=("victor.verticals.contrib.rag.prompt_metadata",),
        requires_sdk_import=True,
    ),
    DefinitionBoundaryTarget(
        label="rag prompt metadata",
        path=REPO_ROOT / "victor/verticals/contrib/rag/prompt_metadata.py",
    ),
    DefinitionBoundaryTarget(
        label="devops assistant",
        path=REPO_ROOT / "victor/verticals/contrib/devops/assistant.py",
        allowed_victor_import_prefixes=("victor.verticals.contrib.devops.prompt_metadata",),
        requires_sdk_import=True,
    ),
    DefinitionBoundaryTarget(
        label="devops prompt metadata",
        path=REPO_ROOT / "victor/verticals/contrib/devops/prompt_metadata.py",
    ),
    DefinitionBoundaryTarget(
        label="dataanalysis assistant",
        path=REPO_ROOT / "victor/verticals/contrib/dataanalysis/assistant.py",
        allowed_victor_import_prefixes=("victor.verticals.contrib.dataanalysis.prompt_metadata",),
        requires_sdk_import=True,
    ),
    DefinitionBoundaryTarget(
        label="dataanalysis prompt metadata",
        path=REPO_ROOT / "victor/verticals/contrib/dataanalysis/prompt_metadata.py",
    ),
    DefinitionBoundaryTarget(
        label="research assistant",
        path=REPO_ROOT / "victor/verticals/contrib/research/assistant.py",
        allowed_victor_import_prefixes=("victor.verticals.contrib.research.prompt_metadata",),
        requires_sdk_import=True,
    ),
    DefinitionBoundaryTarget(
        label="research prompt metadata",
        path=REPO_ROOT / "victor/verticals/contrib/research/prompt_metadata.py",
    ),
    DefinitionBoundaryTarget(
        label="external security assistant",
        path=REPO_ROOT / "examples/external_vertical/src/victor_security/assistant.py",
        requires_sdk_import=True,
    ),
]


def _module_name_for_path(path: Path) -> str:
    """Return the Python module name for a repo-relative file path."""
    return ".".join(path.relative_to(REPO_ROOT).with_suffix("").parts)


def _resolve_from_import(path: Path, node: ast.ImportFrom) -> str | None:
    """Resolve an ``ImportFrom`` node to an absolute module path when possible."""
    if node.level == 0:
        return node.module

    current_module = _module_name_for_path(path)
    package_parts = current_module.split(".")[:-1]
    if node.level > len(package_parts):
        return node.module

    base_parts = package_parts[: len(package_parts) - node.level + 1]
    if node.module:
        return ".".join(base_parts + node.module.split("."))
    return ".".join(base_parts)


def _collect_imports(path: Path) -> list[str]:
    """Collect imported module names from a Python source file."""
    tree = ast.parse(path.read_text(), filename=str(path))
    imports: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module_name = _resolve_from_import(path, node)
            if module_name:
                imports.append(module_name)

    return imports


@pytest.mark.parametrize("target", DEFINITION_BOUNDARY_TARGETS, ids=lambda item: item.label)
def test_migrated_definition_files_only_import_sdk_or_local_definition_helpers(
    target: DefinitionBoundaryTarget,
) -> None:
    """Migrated definition files should avoid all unexpected ``victor.*`` imports."""
    imports = _collect_imports(target.path)
    unexpected_imports = sorted(
        {
            module
            for module in imports
            if module.startswith("victor.")
            and not any(
                module == allowed or module.startswith(f"{allowed}.")
                for allowed in target.allowed_victor_import_prefixes
            )
        }
    )

    assert unexpected_imports == [], (
        f"{target.label} should stay SDK-only aside from explicitly allowed local "
        f"definition helpers; found unexpected Victor imports: {unexpected_imports}"
    )


@pytest.mark.parametrize("target", DEFINITION_BOUNDARY_TARGETS, ids=lambda item: item.label)
def test_migrated_definition_files_do_not_import_forbidden_runtime_prefixes(
    target: DefinitionBoundaryTarget,
) -> None:
    """Definition-layer files should fail on known runtime/framework import prefixes."""
    imports = _collect_imports(target.path)
    forbidden_imports = sorted(
        {
            module
            for module in imports
            if any(
                module == prefix or module.startswith(f"{prefix}.")
                for prefix in FORBIDDEN_DEFINITION_IMPORT_PREFIXES
            )
        }
    )

    assert forbidden_imports == [], (
        f"{target.label} should not import runtime/framework-only prefixes; "
        f"found: {forbidden_imports}"
    )


@pytest.mark.parametrize(
    "target",
    [item for item in DEFINITION_BOUNDARY_TARGETS if item.requires_sdk_import],
    ids=lambda item: item.label,
)
def test_migrated_definition_entrypoints_import_victor_sdk(
    target: DefinitionBoundaryTarget,
) -> None:
    """Definition entrypoints should continue to anchor on ``victor_sdk``."""
    imports = _collect_imports(target.path)

    assert any(
        module == "victor_sdk" or module.startswith("victor_sdk.") for module in imports
    ), f"{target.label} should import victor_sdk"
