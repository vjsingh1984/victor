#!/usr/bin/env python3
"""Synchronize package version strings from VERSION files.

victor-ai and victor-sdk have independent version files:
  - VERSION          → victor-ai version
  - victor-sdk/VERSION → victor-sdk version

Updates:
  - pyproject.toml (victor-ai) version
  - victor-sdk/pyproject.toml version
  - victor-ai's victor-sdk dependency lower bound

Usage:
  python scripts/sync_version.py          # Sync both packages
  python scripts/sync_version.py --ai     # Sync victor-ai only
  python scripts/sync_version.py --sdk    # Sync victor-sdk only
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sync_ai():
    """Sync victor-ai version from root VERSION file."""
    version_file = ROOT / "VERSION"
    if not version_file.exists():
        print("ERROR: VERSION file not found")
        sys.exit(1)

    version = version_file.read_text().strip()
    print(f"Syncing victor-ai to version {version}")

    ai_toml = ROOT / "pyproject.toml"
    text = ai_toml.read_text()
    text = re.sub(
        r'^(version\s*=\s*)"[^"]+"',
        rf'\g<1>"{version}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    ai_toml.write_text(text)
    print(f"  Updated {ai_toml}")


def sync_sdk():
    """Sync victor-sdk version from its own VERSION file."""
    sdk_version_file = ROOT / "victor-sdk" / "VERSION"
    if not sdk_version_file.exists():
        print("ERROR: victor-sdk/VERSION file not found")
        sys.exit(1)

    version = sdk_version_file.read_text().strip()
    print(f"Syncing victor-sdk to version {version}")

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

    # Update victor-ai's SDK dependency lower bound
    ai_toml = ROOT / "pyproject.toml"
    text = ai_toml.read_text()
    text = re.sub(
        r'"victor-sdk[><=!~,. 0-9]+"',
        f'"victor-sdk>={version},<1.0"',
        text,
        count=1,
    )
    ai_toml.write_text(text)
    print(f"  Updated SDK dependency bound in {ai_toml}")


def main():
    if "--ai" in sys.argv:
        sync_ai()
    elif "--sdk" in sys.argv:
        sync_sdk()
    else:
        sync_ai()
        sync_sdk()

    print("\nSync complete. Run 'python scripts/check_version_sync.py' to verify.")


if __name__ == "__main__":
    main()
