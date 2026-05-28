"""Bootstrap helpers for in-repo victor/victor-contracts development workflows."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import MutableMapping, MutableSequence, Optional


def prefer_repo_local_victor_contracts(
    package_file: str | Path,
    *,
    sys_path: MutableSequence[str] | None = None,
    modules: MutableMapping[str, ModuleType] | None = None,
) -> Optional[Path]:
    """Prefer the sibling ``victor-contracts`` checkout when running from this repo.

    This keeps ``../.venv/bin/victor`` aligned with the source tree during local
    development, even if the environment also has an older wheel-installed
    ``victor_contracts`` on ``site-packages``.
    """
    package_path = Path(package_file).resolve()
    repo_root = package_path.parent.parent
    contracts_root = repo_root / "victor-contracts"
    contracts_package = contracts_root / "victor_contracts" / "__init__.py"

    if not ((repo_root / "pyproject.toml").is_file() and contracts_package.is_file()):
        return None

    path_list = sys.path if sys_path is None else sys_path
    module_map = sys.modules if modules is None else modules
    contracts_root_str = str(contracts_root)

    if contracts_root_str in path_list:
        existing_index = path_list.index(contracts_root_str)
        if existing_index != 0:
            path_list.pop(existing_index)
            path_list.insert(0, contracts_root_str)
    else:
        path_list.insert(0, contracts_root_str)

    loaded_contracts = module_map.get("victor_contracts")
    loaded_contracts_file = getattr(loaded_contracts, "__file__", None)
    if loaded_contracts_file:
        try:
            loaded_path = Path(loaded_contracts_file).resolve()
            if contracts_root not in loaded_path.parents:
                for name in list(module_map):
                    if name == "victor_contracts" or name.startswith(
                        "victor_contracts."
                    ):
                        module_map.pop(name, None)
        except OSError:
            pass

    return contracts_root
