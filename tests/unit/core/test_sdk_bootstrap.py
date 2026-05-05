from __future__ import annotations

from pathlib import Path
from types import ModuleType

from victor._sdk_bootstrap import prefer_repo_local_victor_sdk


def test_prefer_repo_local_victor_sdk_prepends_checkout_and_clears_stale_modules(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    package_file = repo_root / "victor" / "__init__.py"
    sdk_root = repo_root / "victor-sdk"
    sdk_package = sdk_root / "victor_sdk" / "__init__.py"

    package_file.parent.mkdir(parents=True)
    package_file.write_text("", encoding="utf-8")
    (repo_root / "pyproject.toml").write_text("[project]\nname='victor-ai'\n", encoding="utf-8")
    sdk_package.parent.mkdir(parents=True)
    sdk_package.write_text("", encoding="utf-8")

    sys_path = ["/fake/site-packages", str(repo_root)]
    modules = {
        "victor_sdk": ModuleType("victor_sdk"),
        "victor_sdk.discovery": ModuleType("victor_sdk.discovery"),
    }
    modules["victor_sdk"].__file__ = "/fake/site-packages/victor_sdk/__init__.py"
    modules["victor_sdk.discovery"].__file__ = "/fake/site-packages/victor_sdk/discovery.py"

    result = prefer_repo_local_victor_sdk(package_file, sys_path=sys_path, modules=modules)

    assert result == sdk_root
    assert sys_path[0] == str(sdk_root)
    assert "victor_sdk" not in modules
    assert "victor_sdk.discovery" not in modules


def test_prefer_repo_local_victor_sdk_noops_without_sibling_checkout(tmp_path: Path) -> None:
    package_file = tmp_path / "victor" / "__init__.py"
    package_file.parent.mkdir(parents=True)
    package_file.write_text("", encoding="utf-8")

    sys_path = ["/fake/site-packages"]
    result = prefer_repo_local_victor_sdk(package_file, sys_path=sys_path, modules={})

    assert result is None
    assert sys_path == ["/fake/site-packages"]
