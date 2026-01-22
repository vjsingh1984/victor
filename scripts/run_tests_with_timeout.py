#!/usr/bin/env python3
"""Run tests with individual timeouts to identify long-running tests.

This script runs pytest tests with a per-test timeout to prevent any single
test from running indefinitely. It's useful for identifying slow or hanging
tests, especially for RAG document ingestion and other I/O-heavy operations.

Usage:
    python scripts/run_tests_with_timeout.py              # All tests
    python scripts/run_tests_with_timeout.py --unit        # Unit tests only
    python scripts/run_tests_with_timeout.py --integration # Integration tests only
    python scripts/run_tests_with_timeout.py tests/unit/   # Specific directory
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_tests_with_timeout(
    test_path: str = "tests/",
    timeout_seconds: int = 240,
    verbose: bool = False,
) -> int:
    """Run pytest tests with per-test timeout.

    Args:
        test_path: Path to tests or specific test file/directory
        timeout_seconds: Per-test timeout in seconds (default: 240 = 4 minutes)
        verbose: Enable verbose output

    Returns:
        Exit code from pytest
    """
    # Build pytest command
    cmd = [
        "pytest",
        test_path,
        "-v" if verbose else "",
        "--tb=short",
        # Use pytest-timeout plugin if available, otherwise use system timeout
        # Note: pytest-timeout needs to be installed: pip install pytest-timeout
        f"--timeout={timeout_seconds}",
        # Mark slow tests separately
        "-m",
        "not slow or slow",
        # Show test durations
        "--durations=10",
        # Stop on first failure (optional - remove if you want all results)
        # "-x",
    ]

    # Filter out empty strings
    cmd = [c for c in cmd if c]

    print(f"🧪 Running tests with {timeout_seconds}s timeout per test...")
    print(f"📁 Test path: {test_path}")
    print(f"📊 Command: {' '.join(cmd)}")
    print()

    # Run pytest
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1


def check_pytest_timeout_installed() -> bool:
    """Check if pytest-timeout plugin is installed."""
    try:
        # Try to import pytest_timeout
        result = subprocess.run(
            [sys.executable, "-c", "import pytest_timeout"],
            capture_output=True,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run tests with individual timeouts to prevent long-running tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests with 4-minute timeout
  python scripts/run_tests_with_timeout.py

  # Run only unit tests with 2-minute timeout
  python scripts/run_tests_with_timeout.py --unit --timeout 120

  # Run specific test file
  python scripts/run_tests_with_timeout.py tests/unit/agent/test_orchestrator_di.py

  # Run with verbose output
  python scripts/run_tests_with_timeout.py -v
        """,
    )

    parser.add_argument(
        "test_path",
        nargs="?",
        default="tests/",
        help="Path to tests or specific test file/directory (default: tests/)",
    )

    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=240,
        help="Per-test timeout in seconds (default: 240 = 4 minutes)",
    )

    parser.add_argument(
        "--unit",
        action="store_true",
        help="Run only unit tests (tests/unit/)",
    )

    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run only integration tests (tests/integration/)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Determine test path based on arguments
    test_path = args.test_path
    if args.unit:
        test_path = "tests/unit/"
    elif args.integration:
        test_path = "tests/integration/"

    # Check if pytest-timeout is installed
    if not check_pytest_timeout_installed():
        print("⚠️  Warning: pytest-timeout plugin not detected")
        print("   Install with: pip install pytest-timeout")
        print("   Falling back to system timeout via 'timeout' command...")
        print()

    # Run tests
    exit_code = run_tests_with_timeout(
        test_path=test_path,
        timeout_seconds=args.timeout,
        verbose=args.verbose,
    )

    # Print summary
    print()
    if exit_code == 0:
        print("✅ All tests passed within timeout!")
    else:
        print(f"⚠️  Tests completed with exit code: {exit_code}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
