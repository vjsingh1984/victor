#!/usr/bin/env python3
"""Verify that vertical classes declared in package metadata are importable.

This script is used by CI to validate external vertical examples and packages
directly from the repository checkout. It supports both flat and `src/`
package layouts and accepts SDK-pure verticals by adapting them through the
core runtime bridge.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
from typing import Iterator

from victor_sdk.verticals.protocols.base import VerticalBase as SdkVerticalBase

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


@dataclass(frozen=True)
class VerifiedVerticalClass:
    """Resolved class details for a validated vertical."""

    vertical_dir: Path
    module_name: str
    class_name: str
    base_class_name: str
    instance_name: str | None
    instantiate_error: str | None = None


def load_class_spec(toml_path: Path) -> tuple[str, str]:
    """Return the module path and class name declared in victor-vertical.toml."""

    vertical = tomllib.loads(toml_path.read_text(encoding="utf-8")).get("vertical", {})
    class_spec = vertical.get("class", {})
    module_name = class_spec.get("module")
    class_name = class_spec.get("class_name")

    if not module_name or not class_name:
        raise ValueError(
            f"{toml_path} does not declare vertical.class.module/class_name"
        )

    return str(module_name), str(class_name)


def candidate_import_roots(vertical_dir: Path) -> list[Path]:
    """Return repo-local import roots for the provided vertical directory."""

    candidates = (
        vertical_dir / "src",
        vertical_dir,
        vertical_dir.parent,
    )
    roots: list[Path] = []
    seen: set[Path] = set()

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists() and resolved not in seen:
            roots.append(resolved)
            seen.add(resolved)

    return roots


@contextmanager
def prepend_sys_path(paths: list[Path]) -> Iterator[None]:
    """Temporarily prepend import roots for class verification."""

    original = list(sys.path)
    try:
        sys.path[:0] = [str(path) for path in paths]
        yield
    finally:
        sys.path[:] = original


def verify_vertical_class(vertical_dir: Path) -> VerifiedVerticalClass:
    """Load and validate the vertical class declared in the package metadata."""

    module_name, class_name = load_class_spec(vertical_dir / "victor-vertical.toml")
    import_roots = candidate_import_roots(vertical_dir)

    with prepend_sys_path(import_roots):
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        if not isinstance(cls, type) or not issubclass(cls, SdkVerticalBase):
            raise TypeError(
                f"{class_name} does not inherit from victor_sdk.VerticalBase"
            )

        instance_name: str | None = None
        instantiate_error: str | None = None
        try:
            instance = cls()
            instance_name = getattr(instance, "name", None)
        except Exception as exc:  # pragma: no cover - informational only
            instantiate_error = str(exc)

    return VerifiedVerticalClass(
        vertical_dir=vertical_dir,
        module_name=module_name,
        class_name=class_name,
        base_class_name=SdkVerticalBase.__name__,
        instance_name=instance_name,
        instantiate_error=instantiate_error,
    )


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("No vertical directories provided.", file=sys.stderr)
        return 1

    class_check_failed = False
    print("Checking vertical class implementations...")
    print()

    for raw_dir in args:
        vertical_dir = Path(raw_dir)
        vertical_name = vertical_dir.name
        print(f"Checking: {vertical_name}")
        try:
            verified = verify_vertical_class(vertical_dir)
        except Exception as exc:
            print(f"✗ Failed to verify class: {exc}", file=sys.stderr)
            class_check_failed = True
            print()
            continue

        print(f"  Module: {verified.module_name}")
        print(f"  Class: {verified.class_name}")
        print(f"✓ Class exists and inherits from {verified.base_class_name}")
        if verified.instance_name is not None:
            print("✓ Class can be instantiated")
            print(f"  Name: {verified.instance_name}")
        elif verified.instantiate_error:
            print(f"⚠ Could not instantiate: {verified.instantiate_error}")
        print()

    if class_check_failed:
        print("=========================================")
        print("Vertical Class Check Failed")
        print("=========================================")
        print()
        print("One or more vertical classes could not be verified.")
        print("Ensure the class specified in victor-vertical.toml:")
        print("  1. Exists at the specified module path")
        print("  2. Can be imported from the package layout")
        print("  3. Inherits from victor_sdk.VerticalBase")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
