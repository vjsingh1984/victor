import ast
import importlib
from pathlib import Path

ALLOWED_IMPORT_FILES = {
    Path("tests/unit/agent/test_provider_switch_contracts.py"),
}
BANNED_PROVIDER_EXPORTS = {
    "ProviderCoordinator",
    "ProviderSwitchCoordinator",
}


def _find_import_violations(module_name: str) -> list[str]:
    violations: list[str] = []

    for root in (Path("victor"), Path("tests")):
        for path in root.rglob("*.py"):
            if path in ALLOWED_IMPORT_FILES:
                continue

            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == module_name:
                            violations.append(f"{path}:{node.lineno} import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module == module_name:
                        names = ", ".join(alias.name for alias in node.names)
                        violations.append(f"{path}:{node.lineno} from {node.module} import {names}")
                    elif node.module == "victor.agent":
                        for alias in node.names:
                            if alias.name == module_name.rsplit(".", 1)[-1]:
                                violations.append(
                                    f"{path}:{node.lineno} from victor.agent import {alias.name}"
                                )

    return violations


def _find_provider_export_violations() -> list[str]:
    violations: list[str] = []

    for root in (Path("victor"), Path("tests")):
        for path in root.rglob("*.py"):
            if path in ALLOWED_IMPORT_FILES:
                continue

            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue

                if node.module == "victor.agent.provider":
                    banned = sorted(
                        alias.name for alias in node.names if alias.name in BANNED_PROVIDER_EXPORTS
                    )
                    if banned:
                        violations.append(
                            f"{path}:{node.lineno} from victor.agent.provider import "
                            + ", ".join(banned)
                        )

    return violations


def test_removed_root_provider_shim_modules_are_not_importable() -> None:
    """Provider root shims were removed in v1.0.0 and must stay absent."""
    for module_name in (
        "victor.agent.provider_coordinator",
        "victor.agent.provider_switch_coordinator",
    ):
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        raise AssertionError(f"{module_name} unexpectedly imported successfully")


def test_removed_root_provider_shim_modules_stay_unreferenced():
    provider_coordinator_violations = _find_import_violations("victor.agent.provider_coordinator")
    provider_switch_violations = _find_import_violations("victor.agent.provider_switch_coordinator")
    provider_export_violations = _find_provider_export_violations()

    assert (
        not provider_coordinator_violations
    ), "removed root provider coordinator shim should stay unreferenced:\n" + "\n".join(
        provider_coordinator_violations
    )
    assert (
        not provider_switch_violations
    ), "removed root provider switch shim should stay unreferenced:\n" + "\n".join(
        provider_switch_violations
    )
    assert (
        not provider_export_violations
    ), "removed provider coordinator exports should stay unreferenced:\n" + "\n".join(
        provider_export_violations
    )
