import ast
from pathlib import Path


ALLOWED_IMPORT_FILES = {
    Path("tests/unit/agent/test_provider_coordinator_shim.py"),
    Path("tests/unit/agent/test_provider_switch_contracts.py"),
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


def test_root_provider_shim_modules_remain_compatibility_only():
    provider_coordinator_violations = _find_import_violations(
        "victor.agent.provider_coordinator"
    )
    provider_switch_violations = _find_import_violations(
        "victor.agent.provider_switch_coordinator"
    )

    assert not provider_coordinator_violations, (
        "root provider coordinator shim should remain compatibility-only:\n"
        + "\n".join(provider_coordinator_violations)
    )
    assert not provider_switch_violations, (
        "root provider switch shim should remain compatibility-only:\n"
        + "\n".join(provider_switch_violations)
    )
