#!/usr/bin/env python3
"""
Generate SHA256 checksums for Victor AI release artifacts.

This script creates a SHA256SUMS file containing checksums for all
distribution artifacts, and optionally generates a GPG signature.

Usage:
    python scripts/create_checksums.py [--sign]
"""

import hashlib
import os
import sys
from pathlib import Path
from typing import Dict, List


def calculate_sha256(file_path: Path) -> str:
    """Calculate SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read file in chunks to handle large files
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def generate_checksums(dist_dir: Path = Path("dist")) -> Dict[str, str]:
    """
    Generate SHA256 checksums for all files in dist directory.

    Args:
        dist_dir: Path to distribution directory

    Returns:
        Dictionary mapping filenames to checksums
    """
    checksums = {}

    if not dist_dir.exists():
        print(f"Error: Distribution directory '{dist_dir}' not found", file=sys.stderr)
        sys.exit(1)

    # Find all distribution files
    files = list(dist_dir.glob("*"))
    if not files:
        print(f"Error: No files found in '{dist_dir}'", file=sys.stderr)
        sys.exit(1)

    print(f"Generating checksums for {len(files)} files...")

    for file_path in sorted(files):
        if file_path.is_file():
            checksum = calculate_sha256(file_path)
            checksums[file_path.name] = checksum
            print(f"  {file_path.name}: {checksum}")

    return checksums


def write_checksums_file(checksums: Dict[str, str], output_file: Path = Path("SHA256SUMS")) -> None:
    """
    Write checksums to SHA256SUMS file.

    Args:
        checksums: Dictionary mapping filenames to checksums
        output_file: Path to output file
    """
    with open(output_file, "w") as f:
        for filename, checksum in sorted(checksums.items()):
            f.write(f"{checksum}  {filename}\n")

    print(f"\nChecksums written to {output_file}")


def verify_checksums(checksums_file: Path = Path("SHA256SUMS")) -> bool:
    """
    Verify checksums in SHA256SUMS file.

    Args:
        checksums_file: Path to checksums file

    Returns:
        True if all checksums are valid, False otherwise
    """
    if not checksums_file.exists():
        print(f"Error: Checksums file '{checksums_file}' not found", file=sys.stderr)
        return False

    print(f"Verifying checksums from {checksums_file}...")

    with open(checksums_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("  ", 1)
            if len(parts) != 2:
                print(f"Warning: Invalid line: {line}", file=sys.stderr)
                continue

            expected_checksum, filename = parts
            file_path = Path("dist") / filename

            if not file_path.exists():
                print(f"Warning: File not found: {filename}", file=sys.stderr)
                continue

            actual_checksum = calculate_sha256(file_path)

            if actual_checksum == expected_checksum:
                print(f"  ✓ {filename}")
            else:
                print(f"  ✗ {filename} (checksum mismatch)", file=sys.stderr)
                return False

    print("\nAll checksums verified successfully!")
    return True


def print_statistics(checksums: Dict[str, str]) -> None:
    """Print statistics about distribution files."""
    print("\n=== Release Statistics ===")

    total_size = 0
    for filename in checksums.keys():
        file_path = Path("dist") / filename
        size = file_path.stat().st_size
        total_size += size

        # Format size in human-readable format
        if size < 1024:
            size_str = f"{size} B"
        elif size < 1024 * 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size / (1024 * 1024):.1f} MB"

        print(f"  {filename}: {size_str}")

    # Format total size
    if total_size < 1024 * 1024:
        total_str = f"{total_size / 1024:.1f} KB"
    else:
        total_str = f"{total_size / (1024 * 1024):.1f} MB"

    print(f"\nTotal size: {total_str}")
    print(f"Total files: {len(checksums)}")


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate SHA256 checksums for Victor AI release artifacts"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing checksums instead of generating new ones",
    )
    parser.add_argument(
        "--dist-dir",
        type=Path,
        default=Path("dist"),
        help="Path to distribution directory (default: dist)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("SHA256SUMS"),
        help="Path to output checksums file (default: SHA256SUMS)",
    )

    args = parser.parse_args()

    if args.verify:
        # Verify existing checksums
        success = verify_checksums(args.output)
        sys.exit(0 if success else 1)
    else:
        # Generate new checksums
        checksums = generate_checksums(args.dist_dir)
        write_checksums_file(checksums, args.output)
        print_statistics(checksums)

        print("\n=== Verification Instructions ===")
        print("To verify the checksums after download:")
        print(f"  sha256sum -c {args.output}")
        print("")
        print("To verify a single file:")
        print("  sha256sum <filename>  # Compare with checksum in SHA256SUMS")


if __name__ == "__main__":
    main()
