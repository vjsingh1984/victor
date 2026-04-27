"""Tests for sidecar vertical package manifest support."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from victor.core.verticals.manifest_contract import (
    get_or_create_vertical_manifest,
    load_vertical_package_manifest_for_module,
)
from victor_sdk.verticals.manifest import ExtensionManifest, ExtensionType


def _write_sidecar_package(root: Path, package_name: str = "sample_vertical_pkg") -> str:
    package_dir = root / package_name
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "assistant.py").write_text(
        """
class SampleVertical:
    description = "sample vertical"
""".strip(),
        encoding="utf-8",
    )
    (package_dir / "victor-vertical.toml").write_text(
        f"""
[vertical]
name = "sidecar_vertical"
version = "2.3.4"
description = "sidecar vertical"
license = "Apache-2.0"
requires_victor = ">=0.7.0"
authors = [{{name = "Victor"}}]

[vertical.class]
module = "{package_name}.assistant"
class_name = "SampleVertical"
provides_tools = ["read", "write"]
provides_workflows = ["triage"]
provides_capabilities = ["indexing"]

[vertical.dependencies]
verticals = ["shared_memory", "prompt_guard"]
""".strip(),
        encoding="utf-8",
    )
    return f"{package_name}.assistant"


def test_load_vertical_package_manifest_for_module_reads_sidecar(tmp_path, monkeypatch):
    """Package-level metadata should normalize into an ExtensionManifest."""
    module_name = _write_sidecar_package(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    manifest = load_vertical_package_manifest_for_module(module_name)

    assert manifest is not None
    assert manifest.name == "sidecar_vertical"
    assert manifest.version == "2.3.4"
    assert manifest.min_framework_version == ">=0.7.0"
    assert manifest.provides == {
        ExtensionType.TOOLS,
        ExtensionType.WORKFLOWS,
        ExtensionType.CAPABILITIES,
    }
    assert [dep.extension_name for dep in manifest.extension_dependencies] == [
        "shared_memory",
        "prompt_guard",
    ]


def test_get_or_create_vertical_manifest_prefers_explicit_manifest_over_sidecar(
    tmp_path,
    monkeypatch,
):
    """Explicit class manifests should win over package sidecar metadata."""
    module_name = _write_sidecar_package(tmp_path, package_name="explicit_vertical_pkg")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    module = importlib.import_module(module_name)
    vertical_cls = module.SampleVertical
    vertical_cls._victor_manifest = ExtensionManifest(  # type: ignore[attr-defined]
        name="explicit_vertical",
        version="9.9.9",
        provides={ExtensionType.SAFETY},
    )

    manifest = get_or_create_vertical_manifest(vertical_cls)

    assert manifest is vertical_cls._victor_manifest
    assert manifest.name == "explicit_vertical"
    assert manifest.provides == {ExtensionType.SAFETY}

    sys.modules.pop(module_name, None)
    sys.modules.pop("explicit_vertical_pkg", None)
