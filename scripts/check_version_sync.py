#!/usr/bin/env python3
"""Verify that victor-ai and victor-sdk versions are consistent.

victor-ai and victor-sdk maintain independent version numbers.

Checks:
  1. VERSION file exists and matches victor-ai pyproject.toml
  2. victor-sdk/VERSION exists and matches victor-sdk pyproject.toml
  3. victor-ai's dependency on victor-sdk uses a compatible range

Exit code 0 on success, 1 on mismatch.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read_version(path: Path) -> str:
    if not path.exists():
        print(f"ERROR: {path} not found")
        sys.exit(1)
    return path.read_text().strip()


def extract_toml_version(toml_path: Path) -> str:
    """Extract version = "X.Y.Z" from a pyproject.toml."""
    text = toml_path.read_text()
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        print(f"ERROR: No version found in {toml_path}")
        sys.exit(1)
    return match.group(1)


def extract_sdk_dependency(toml_path: Path) -> str:
    """Extract the victor-sdk dependency spec from victor-ai's pyproject.toml."""
    text = toml_path.read_text()
    match = re.search(r'"(victor-sdk[^"]*)"', text)
    if not match:
        print(f"ERROR: No victor-sdk dependency found in {toml_path}")
        sys.exit(1)
    return match.group(1)


def main():
    errors = []

    # Check victor-ai version
    ai_version = read_version(ROOT / "VERSION")
    ai_toml_version = extract_toml_version(ROOT / "pyproject.toml")
    print(f"victor-ai VERSION file: {ai_version}")
    print(f"victor-ai pyproject.toml: {ai_toml_version}")
    if ai_version != ai_toml_version:
        errors.append(f"victor-ai version ({ai_toml_version}) != VERSION ({ai_version})")

    # Check victor-sdk version
    sdk_version = read_version(ROOT / "victor-sdk" / "VERSION")
    sdk_toml_version = extract_toml_version(ROOT / "victor-sdk" / "pyproject.toml")
    print(f"victor-sdk VERSION file: {sdk_version}")
    print(f"victor-sdk pyproject.toml: {sdk_toml_version}")
    if sdk_version != sdk_toml_version:
        errors.append(f"victor-sdk version ({sdk_toml_version}) != VERSION ({sdk_version})")

    # Check victor-ai's dependency on victor-sdk
    dep_spec = extract_sdk_dependency(ROOT / "pyproject.toml")
    print(f"victor-ai depends on: {dep_spec}")

    # Verify the SDK version satisfies the dependency spec
    if ">=" in dep_spec:
        # Extract lower bound: "victor-sdk>=X.Y.Z,<A.B"
        lower_match = re.search(r">=([0-9.]+)", dep_spec)
        if lower_match:
            lower_bound = lower_match.group(1)
            if sdk_version < lower_bound:
                errors.append(
                    f"victor-sdk version ({sdk_version}) is below "
                    f"victor-ai's lower bound ({lower_bound})"
                )
    elif "==" in dep_spec:
        # Exact pin — still valid, just not decoupled
        pin_match = re.search(r"==([0-9.]+)", dep_spec)
        if pin_match and pin_match.group(1) != sdk_version:
            errors.append(
                f"victor-ai pins victor-sdk=={pin_match.group(1)} "
                f"but SDK version is {sdk_version}"
            )

    if errors:
        print()
        for err in errors:
            print(f"FAIL: {err}")
        print(f"\nTo fix: update VERSION files and run 'make sync-version'")
        sys.exit(1)

    print("\nAll versions consistent.")
    sys.exit(0)


if __name__ == "__main__":
    main()
