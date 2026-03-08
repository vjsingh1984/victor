#!/usr/bin/env python3
"""Verify that victor-ai and victor-sdk versions are in sync.

Checks:
  1. VERSION file exists and is valid semver-like
  2. pyproject.toml (victor-ai) version matches VERSION
  3. victor-sdk/pyproject.toml version matches VERSION
  4. victor-ai dependency on victor-sdk pins the correct version

Exit code 0 on success, 1 on mismatch.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read_version_file() -> str:
    path = ROOT / "VERSION"
    if not path.exists():
        print("ERROR: VERSION file not found at repo root")
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


def extract_sdk_dependency_pin(toml_path: Path) -> str:
    """Extract the victor-sdk version pin from victor-ai's pyproject.toml."""
    text = toml_path.read_text()
    # Match: "victor-sdk==X.Y.Z" or "victor-sdk>=X.Y.Z"
    match = re.search(r'"victor-sdk([><=!~]+)([^"]+)"', text)
    if not match:
        print(f"ERROR: No victor-sdk dependency found in {toml_path}")
        sys.exit(1)
    return match.group(1), match.group(2).split(",")[0].strip()


def main():
    version = read_version_file()
    print(f"VERSION file: {version}")

    errors = []

    # Check victor-ai pyproject.toml
    ai_toml = ROOT / "pyproject.toml"
    ai_version = extract_toml_version(ai_toml)
    print(f"victor-ai pyproject.toml: {ai_version}")
    if ai_version != version:
        errors.append(f"victor-ai version ({ai_version}) != VERSION ({version})")

    # Check victor-sdk pyproject.toml
    sdk_toml = ROOT / "victor-sdk" / "pyproject.toml"
    sdk_version = extract_toml_version(sdk_toml)
    print(f"victor-sdk pyproject.toml: {sdk_version}")
    if sdk_version != version:
        errors.append(f"victor-sdk version ({sdk_version}) != VERSION ({version})")

    # Check victor-ai's dependency pin on victor-sdk
    op, pinned = extract_sdk_dependency_pin(ai_toml)
    print(f"victor-ai depends on victor-sdk{op}{pinned}")
    if op != "==":
        errors.append(
            f"victor-ai should pin victor-sdk with ==, got {op}"
        )
    if pinned != version:
        errors.append(
            f"victor-ai pins victor-sdk=={pinned}, expected =={version}"
        )

    if errors:
        print()
        for err in errors:
            print(f"FAIL: {err}")
        print(f"\nTo fix: update VERSION file and run 'make sync-version'")
        sys.exit(1)

    print("\nAll versions in sync.")
    sys.exit(0)


if __name__ == "__main__":
    main()
