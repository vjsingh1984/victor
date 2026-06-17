"""CLI entry points for the Victor contracts package."""

from __future__ import annotations

import argparse
from typing import Sequence

from victor_contracts.validation import validate_vertical_package


def _run(argv: Sequence[str] | None = None, *, prog: str) -> int:
    """Run the contract validation command-line interface."""

    parser = argparse.ArgumentParser(prog=prog)
    subparsers = parser.add_subparsers(dest="command")

    check_parser = subparsers.add_parser(
        "check",
        help="Validate a vertical package against the Victor contracts",
    )
    check_parser.add_argument("package_name", help="Installed package name to validate")

    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command != "check":
        parser.print_help()
        return 2

    report = validate_vertical_package(args.package_name)
    print(report.to_text())
    return 0 if report.ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run the victor-contracts command-line interface."""

    return _run(argv, prog="victor-contracts")


def contracts_main(argv: Sequence[str] | None = None) -> int:
    """Run the victor-contracts command-line interface."""

    return _run(argv, prog="victor-contracts")


if __name__ == "__main__":
    raise SystemExit(main())
