"""Bootstrap helpers for in-repo victor/victor-sdk development workflows."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import MutableMapping, MutableSequence, Optional


def prefer_repo_local_victor_sdk(
    package_file: str | Path,
    *,
    sys_path: MutableSequence[str] | None = None,
    modules: MutableMapping[str, ModuleType] | None = None,
) -> Optional[Path]:
    """Prefer the sibling ``victor-sdk`` checkout when running from this repo.

    This keeps ``../.venv/bin/victor`` aligned with the source tree during local
    development, even if the environment also has an older wheel-installed
    ``victor_sdk`` on ``site-packages``.
    """
    package_path = Path(package_file).resolve()
    repo_root = package_path.parent.parent
    sdk_root = repo_root / "victor-sdk"
    sdk_package = sdk_root / "victor_sdk" / "__init__.py"

    if not ((repo_root / "pyproject.toml").is_file() and sdk_package.is_file()):
        return None

    path_list = sys.path if sys_path is None else sys_path
    module_map = sys.modules if modules is None else modules
    sdk_root_str = str(sdk_root)

    if sdk_root_str in path_list:
        existing_index = path_list.index(sdk_root_str)
        if existing_index != 0:
            path_list.pop(existing_index)
            path_list.insert(0, sdk_root_str)
    else:
        path_list.insert(0, sdk_root_str)

    loaded_sdk = module_map.get("victor_sdk")
    loaded_sdk_file = getattr(loaded_sdk, "__file__", None)
    if loaded_sdk_file:
        try:
            loaded_path = Path(loaded_sdk_file).resolve()
            if sdk_root not in loaded_path.parents:
                for name in list(module_map):
                    if name == "victor_sdk" or name.startswith("victor_sdk."):
                        module_map.pop(name, None)
        except OSError:
            pass

    return sdk_root
