#!/usr/bin/env python3
"""Synchronize all version strings from the VERSION file.

Updates:
  - pyproject.toml (victor-ai) version
  - victor-sdk/pyproject.toml version
  - victor-ai's victor-sdk dependency pin
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    version_file = ROOT / "VERSION"
    if not version_file.exists():
        print("ERROR: VERSION file not found")
        sys.exit(1)

    version = version_file.read_text().strip()
    print(f"Syncing all packages to version {version}")

    # Update victor-ai pyproject.toml
    ai_toml = ROOT / "pyproject.toml"
    text = ai_toml.read_text()
    text = re.sub(
        r'^(version\s*=\s*)"[^"]+"',
        rf'\g<1>"{version}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r'"victor-sdk[><=!~]+[^"]*"',
        f'"victor-sdk=={version}"',
        text,
        count=1,
    )
    ai_toml.write_text(text)
    print(f"  Updated {ai_toml}")

    # Update victor-sdk pyproject.toml
    sdk_toml = ROOT / "victor-sdk" / "pyproject.toml"
    text = sdk_toml.read_text()
    text = re.sub(
        r'^(version\s*=\s*)"[^"]+"',
        rf'\g<1>"{version}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    sdk_toml.write_text(text)
    print(f"  Updated {sdk_toml}")

    print(f"\nAll packages synced to {version}")
    print("Run 'python scripts/check_version_sync.py' to verify")


if __name__ == "__main__":
    main()
