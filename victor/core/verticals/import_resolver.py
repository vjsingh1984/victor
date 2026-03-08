"""Helpers for resolving vertical module imports across package layouts.

This module centralizes import fallback behavior used during the external
vertical extraction migration:

1. External package namespace (preferred): ``victor_<vertical>``
2. Legacy in-package namespace: ``victor.<vertical>``
3. Contrib fallback namespace: ``victor.verticals.contrib.<vertical>``
"""

from __future__ import annotations

import importlib
import logging
from types import ModuleType
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

_PACKAGE_OVERRIDES = {
    "dataanalysis": "victor_dataanalysis",
}


def normalize_vertical_name(vertical_name: str) -> str:
    """Normalize a vertical name for import path construction."""
    normalized = vertical_name.strip().lower().replace("-", "_")
    if normalized in {"data_analysis"}:
        return "dataanalysis"
    return normalized


def _dedupe(candidates: List[str]) -> List[str]:
    """Deduplicate candidate paths while preserving order."""
    seen = set()
    ordered: List[str] = []
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        ordered.append(candidate)
        seen.add(candidate)
    return ordered


def _join_module_path(package: str, module_suffix: str) -> str:
    """Join package + optional module suffix into a module path."""
    suffix = module_suffix.strip(".")
    if not suffix:
        return package
    return f"{package}.{suffix}"


def _external_package_candidates(vertical_name: str) -> List[str]:
    """Return external package candidates for a vertical."""
    normalized = normalize_vertical_name(vertical_name)

    candidates: List[str] = []
    override = _PACKAGE_OVERRIDES.get(normalized) or _PACKAGE_OVERRIDES.get(vertical_name)
    if override:
        candidates.append(override)
    else:
        candidates.append(f"victor_{normalized}")

    # Support historical spellings like data_analysis vs dataanalysis.
    if "_" in normalized:
        candidates.append(f"victor_{normalized.replace('_', '')}")

    return _dedupe(candidates)


def vertical_module_candidates(vertical_name: str, module_suffix: str) -> List[str]:
    """Build ordered import candidates for a vertical module suffix."""
    normalized = normalize_vertical_name(vertical_name)
    suffix = module_suffix.strip(".")
    candidates: List[str] = []

    for package in _external_package_candidates(normalized):
        candidates.append(_join_module_path(package, suffix))

    candidates.append(_join_module_path(f"victor.{normalized}", suffix))
    candidates.append(_join_module_path(f"victor.verticals.contrib.{normalized}", suffix))
    return _dedupe(candidates)


def module_import_candidates(module_path: str) -> List[str]:
    """Expand a module path into ordered compatibility candidates."""
    path = module_path.strip()
    if not path:
        return []

    parts = path.split(".")

    # Contrib path: victor.verticals.contrib.<vertical>[.<suffix>...]
    if len(parts) >= 4 and parts[:3] == ["victor", "verticals", "contrib"]:
        vertical = parts[3]
        suffix = ".".join(parts[4:])
        return _dedupe(vertical_module_candidates(vertical, suffix) + [path])

    # Legacy path: victor.<vertical>[.<suffix>...]
    if len(parts) >= 2 and parts[0] == "victor":
        vertical = parts[1]
        suffix = ".".join(parts[2:])
        return _dedupe(vertical_module_candidates(vertical, suffix) + [path])

    # External path: victor_<vertical>[.<suffix>...]
    root = parts[0]
    if root.startswith("victor_"):
        vertical = root[len("victor_") :]
        suffix = ".".join(parts[1:])
        return _dedupe([path] + vertical_module_candidates(vertical, suffix))

    return [path]


def import_module_with_fallback(module_path: str) -> Tuple[Optional[ModuleType], Optional[str]]:
    """Import first available module from compatibility candidates."""
    for candidate in module_import_candidates(module_path):
        try:
            module = importlib.import_module(candidate)
            return module, candidate
        except Exception as exc:
            logger.debug("Failed importing module candidate '%s': %s", candidate, exc)
            continue
    return None, None
