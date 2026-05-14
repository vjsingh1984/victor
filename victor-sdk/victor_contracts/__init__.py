"""Semantic import alias for the Victor contracts package.

The current distribution and import namespace are still ``victor-sdk`` and
``victor_sdk`` for compatibility. New code should prefer ``victor_contracts``
when it means the protocol/definition contract package rather than a runtime
client SDK.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
from types import ModuleType
from typing import Any

import victor_sdk as _victor_sdk
from victor_sdk import *  # noqa: F401,F403
from victor_sdk import __all__ as __all__  # noqa: F401
from victor_sdk import __version__ as __version__  # noqa: F401


class _VictorContractsAliasLoader(importlib.abc.Loader):
    """Load ``victor_contracts.*`` modules by aliasing ``victor_sdk.*``."""

    def __init__(self, sdk_name: str) -> None:
        self._sdk_name = sdk_name

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        module = importlib.import_module(self._sdk_name)
        sys.modules[spec.name] = module
        return module

    def exec_module(self, module: ModuleType) -> None:
        return None


class _VictorContractsAliasFinder(importlib.abc.MetaPathFinder):
    """Redirect nested ``victor_contracts`` imports to ``victor_sdk`` modules."""

    _PREFIX = "victor_contracts."
    _SDK_PREFIX = "victor_sdk."

    def find_spec(
        self,
        fullname: str,
        path: Any,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if not fullname.startswith(self._PREFIX):
            return None

        sdk_name = self._SDK_PREFIX + fullname[len(self._PREFIX) :]
        sdk_spec = importlib.util.find_spec(sdk_name)
        if sdk_spec is None:
            return None

        is_package = sdk_spec.submodule_search_locations is not None
        spec = importlib.machinery.ModuleSpec(
            fullname,
            _VictorContractsAliasLoader(sdk_name),
            is_package=is_package,
        )
        spec.origin = sdk_spec.origin
        if is_package:
            spec.submodule_search_locations = sdk_spec.submodule_search_locations
        return spec


def _install_alias_finder() -> None:
    if any(isinstance(finder, _VictorContractsAliasFinder) for finder in sys.meta_path):
        return
    sys.meta_path.insert(0, _VictorContractsAliasFinder())


_install_alias_finder()

# Present this package as having the same search path as victor_sdk. This keeps
# tooling that inspects package resources or submodule search paths working.
__path__ = _victor_sdk.__path__
