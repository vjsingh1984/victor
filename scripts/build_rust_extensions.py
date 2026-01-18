#!/usr/bin/env python3
"""
Build script for victor_native Rust extensions.

This script builds the Rust extensions using maturin with proper error handling
and build verification steps.

Usage:
    python scripts/build_rust_extensions.py [--release] [--debug]

Options:
    --release    Build with release optimizations (default)
    --debug      Build with debug symbols
"""

import argparse
import subprocess
import sys
from pathlib import Path


def check_maturin_installed():
    """Check if maturin is installed, install if not."""
    try:
        result = subprocess.run(
            ["maturin", "--version"],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✅ Found maturin: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print("⚠️  Maturin not found. Installing...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "maturin>=1.4"],
                check=True
            )
            print("✅ Maturin installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install maturin. Please install manually:")
            print("   pip install maturin>=1.4")
            return False


def check_rust_toolchain():
    """Check if Rust toolchain is installed."""
    try:
        result = subprocess.run(
            ["rustc", "--version"],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✅ Found Rust: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Rust toolchain not found. Please install Rust:")
        print("   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh")
        return False


def build_rust_extensions(profile="release"):
    """
    Build Rust extensions using maturin.

    Args:
        profile: Build profile ("release" or "debug")

    Returns:
        True if build succeeded, False otherwise
    """
    rust_dir = Path(__file__).parent.parent / "rust"

    if not rust_dir.exists():
        print(f"❌ Rust directory not found: {rust_dir}")
        return False

    print(f"\n🔨 Building Rust extensions with {profile} profile...")
    print(f"   Working directory: {rust_dir}")

    # Build command
    cmd = ["maturin", "develop"]

    if profile == "release":
        cmd.append("--release")
    elif profile == "debug":
        cmd.append("--release-with-debug")

    try:
        # Run maturin build
        result = subprocess.run(
            cmd,
            cwd=rust_dir,
            check=True,
            capture_output=True,
            text=True
        )

        print("✅ Build output:")
        print(result.stdout)

        if result.stderr:
            print("⚠️  Build warnings:")
            print(result.stderr)

        return verify_build()

    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed with error:")
        print(e.stdout)
        print(e.stderr)
        return False


def verify_build():
    """
    Verify that the built module can be imported.

    Returns:
        True if verification succeeded, False otherwise
    """
    print("\n🔍 Verifying build...")

    try:
        # Try to import the module
        result = subprocess.run(
            [sys.executable, "-c", "import victor_native; print(victor_native.__version__)"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10
        )

        version = result.stdout.strip()
        print(f"✅ Build verification successful!")
        print(f"   victor_native version: {version}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Build verification failed:")
        print(e.stdout)
        print(e.stderr)
        return False
    except subprocess.TimeoutExpired:
        print("❌ Build verification timed out")
        return False


def run_tests():
    """Run Rust unit tests."""
    print("\n🧪 Running Rust unit tests...")

    rust_dir = Path(__file__).parent.parent / "rust"

    try:
        result = subprocess.run(
            ["cargo", "test", "--lib"],
            cwd=rust_dir,
            check=True,
            capture_output=True,
            text=True
        )

        print("✅ All Rust tests passed")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Rust tests failed:")
        print(e.stdout)
        print(e.stderr)
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build victor_native Rust extensions"
    )
    parser.add_argument(
        "--release",
        action="store_true",
        default=True,
        help="Build with release optimizations (default)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Build with debug symbols"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run Rust unit tests after building"
    )

    args = parser.parse_args()

    # Determine build profile
    if args.debug:
        profile = "debug"
    else:
        profile = "release"

    print("🚀 Victor Native Rust Extensions Build Script")
    print("=" * 60)

    # Check prerequisites
    if not check_rust_toolchain():
        sys.exit(1)

    if not check_maturin_installed():
        sys.exit(1)

    # Build extensions
    if not build_rust_extensions(profile):
        print("\n❌ Build failed. Please check the errors above.")
        sys.exit(1)

    # Run tests if requested
    if args.test:
        if not run_tests():
            print("\n⚠️  Tests failed, but build succeeded.")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("✅ Build completed successfully!")
    print(f"   Profile: {profile}")
    print(f"   Module: victor_native")
    print("=" * 60)


if __name__ == "__main__":
    main()
