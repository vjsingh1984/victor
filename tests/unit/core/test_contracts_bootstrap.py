from __future__ import annotations

from pathlib import Path
from types import ModuleType

from victor._contracts_bootstrap import prefer_repo_local_victor_contracts


def test_prefer_repo_local_victor_contracts_prepends_checkout_and_clears_stale_modules(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    package_file = repo_root / "victor" / "__init__.py"
    contracts_root = repo_root / "victor-contracts"
    contracts_package = contracts_root / "victor_contracts" / "__init__.py"

    package_file.parent.mkdir(parents=True)
    package_file.write_text("", encoding="utf-8")
    (repo_root / "pyproject.toml").write_text(
        "[project]\nname='victor-ai'\n", encoding="utf-8"
    )
    contracts_package.parent.mkdir(parents=True)
    contracts_package.write_text("", encoding="utf-8")

    sys_path = ["/fake/site-packages", str(repo_root)]
    modules = {
        "victor_contracts": ModuleType("victor_contracts"),
        "victor_contracts.discovery": ModuleType("victor_contracts.discovery"),
    }
    modules["victor_contracts"].__file__ = (
        "/fake/site-packages/victor_contracts/__init__.py"
    )
    modules["victor_contracts.discovery"].__file__ = (
        "/fake/site-packages/victor_contracts/discovery.py"
    )

    result = prefer_repo_local_victor_contracts(
        package_file, sys_path=sys_path, modules=modules
    )

    assert result == contracts_root
    assert sys_path[0] == str(contracts_root)
    assert "victor_contracts" not in modules
    assert "victor_contracts.discovery" not in modules


def test_prefer_repo_local_victor_contracts_noops_without_sibling_checkout(
    tmp_path: Path,
) -> None:
    package_file = tmp_path / "victor" / "__init__.py"
    package_file.parent.mkdir(parents=True)
    package_file.write_text("", encoding="utf-8")

    sys_path = ["/fake/site-packages"]
    result = prefer_repo_local_victor_contracts(
        package_file, sys_path=sys_path, modules={}
    )

    assert result is None
    assert sys_path == ["/fake/site-packages"]
