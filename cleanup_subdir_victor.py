#!/usr/bin/env python3
"""Clean up subdirectory .victor directories, keeping only the root .victor.

This script:
1. Identifies all .victor directories
2. Preserves the root .victor
3. Removes subdirectory .victor directories (duplicate embeddings/indexes)
4. Reports space savings

Usage:
    python cleanup_subdir_victor.py [--dry-run]
"""

import argparse
import shutil
from pathlib import Path
import sys


def get_directory_size(path: Path) -> int:
    """Get total size of directory in bytes."""
    total = 0
    try:
        for item in path.rglob('*'):
            if item.is_file():
                total += item.stat().st_size
            elif item.is_dir() and not item.is_symlink():
                try:
                    total += sum(
                        f.stat().st_size
                        for f in item.iterdir()
                        if f.is_file() and not f.is_symlink()
                    )
                except (PermissionError, OSError):
                    pass
    except (PermissionError, OSError):
        pass
    return total


def format_size(bytes_size: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f}TB"


def find_victor_dirs(root_path: Path) -> tuple[Path, list[Path]]:
    """Find all .victor directories, separating root from subdirectories.

    Returns:
        Tuple of (root_victor, subdirectory_victors)
    """
    all_victor_dirs = sorted(root_path.rglob('.victor'))

    # Root .victor is the one directly in the project root
    root_victor = root_path / '.victor'
    if root_victor in all_victor_dirs:
        subdirs = [d for d in all_victor_dirs if d != root_victor]
    else:
        # No root .victor found, all are subdirectories
        root_victor = None
        subdirs = all_victor_dirs

    return root_victor, subdirs


def main():
    parser = argparse.ArgumentParser(
        description='Clean up subdirectory .victor directories'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be deleted without actually deleting'
    )
    parser.add_argument(
        '--project-root',
        type=Path,
        default=Path.cwd(),
        help='Project root directory (default: current directory)'
    )

    args = parser.parse_args()
    project_root = args.project_root.resolve()

    print("=" * 70)
    print("Subdirectory .victor Cleanup")
    print("=" * 70)
    print(f"Project root: {project_root}")
    print(f"Mode: {'DRY RUN (no changes)' if args.dry_run else 'LIVE (will delete)'}")
    print()

    # Find all .victor directories
    root_victor, subdirs = find_victor_dirs(project_root)

    if root_victor and root_victor.exists():
        root_size = get_directory_size(root_victor)
        print(f"✓ Root .victor: {root_victor}")
        print(f"  Size: {format_size(root_size)}")
        print(f"  Action: KEEP (this is the master)")
    else:
        print(f"⚠ No root .victor found at {project_root / '.victor'}")
        root_size = 0

    print()
    print(f"Subdirectory .victor directories to remove: {len(subdirs)}")
    print()

    total_subdir_size = 0
    for subdir in subdirs:
        if not subdir.exists():
            continue

        size = get_directory_size(subdir)
        total_subdir_size += size

        # Show relative path from project root
        try:
            rel_path = subdir.relative_to(project_root)
        except ValueError:
            rel_path = subdir

        print(f"  📁 {rel_path}")
        print(f"     Size: {format_size(size)}")

        # Show what's inside
        contents = list(subdir.iterdir())
        if contents:
            for item in contents:
                if item.is_dir():
                    item_size = get_directory_size(item)
                    print(f"       └─ {item.name}/ ({format_size(item_size)})")
                else:
                    item_size = item.stat().st_size
                    print(f"       └─ {item.name} ({format_size(item_size)})")

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Root .victor size:     {format_size(root_size)}")
    print(f"Subdir total size:     {format_size(total_subdir_size)}")
    print(f"Space to be freed:     {format_size(total_subdir_size)}")

    if total_subdir_size > 0 and root_size > 0:
        percentage = (total_subdir_size / (root_size + total_subdir_size)) * 100
        print(f"Reduction:            {percentage:.1f}%")

    print()
    print(f"Directories to remove: {len(subdirs)}")

    if not args.dry_run:
        print()
        response = input("Continue with deletion? [yes/NO]: ").strip().lower()

        if response != 'yes':
            print("Aborted.")
            return 1

        print()
        print("Deleting subdirectory .victor directories...")
        for subdir in subdirs:
            if not subdir.exists():
                continue

            try:
                shutil.rmtree(subdir)
                print(f"  ✓ Deleted: {subdir.relative_to(project_root)}")
            except Exception as e:
                print(f"  ✗ Failed to delete {subdir}: {e}")

        print()
        print("✅ Cleanup complete!")
        print(f"   Root .victor preserved at: {root_victor}")
    else:
        print()
        print("DRY RUN: No changes made.")
        print("Run without --dry-run to actually delete these directories.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
