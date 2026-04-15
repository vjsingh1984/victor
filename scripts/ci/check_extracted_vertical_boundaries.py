#!/usr/bin/env python3
"""Audit extracted vertical repositories against Victor's SDK boundary contract.

By default this script looks for the extracted plugin repos that currently
track the contract-first migration:
  - ../victor-coding
  - ../victor-research
  - ../victor-devops

Additional repos can be audited by passing explicit paths. Hybrid app repos
such as victor-invest should be passed explicitly once they are on the same
plugin contract track.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_RELATIVE_PATHS = (
    "../victor-coding",
    "../victor-research",
    "../victor-devops",
)


def normalize_paths(paths: Iterable[str | Path], *, cwd: Path) -> list[Path]:
    """Return de-duplicated absolute paths preserving input order."""
    normalized: list[Path] = []
    seen: set[Path] = set()

    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = (cwd / path).resolve()
        else:
            path = path.resolve()

        if path in seen:
            continue
        seen.add(path)
        normalized.append(path)

    return normalized


def discover_default_paths(*, repo_root: Path) -> list[Path]:
    """Discover existing extracted vertical repo paths next to the core repo."""
    return [
        path
        for path in normalize_paths(DEFAULT_RELATIVE_PATHS, cwd=repo_root)
        if path.exists() and path.is_dir()
    ]


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description="Audit extracted vertical repositories for forbidden Victor runtime imports."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Explicit extracted vertical repository paths to audit.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a text summary.",
    )
    return parser


def run_audit(
    paths: Sequence[Path],
    *,
    json_output: bool,
    stdout: object,
) -> int:
    """Run the contract audit and emit a summary."""
    from victor.core.verticals.contract_audit import VerticalContractAuditor

    reports = VerticalContractAuditor().audit_paths(paths)

    if json_output:
        stdout.write(json.dumps([report.to_dict() for report in reports], indent=2))
        stdout.write("\n")
    else:
        for report in reports:
            status = "PASSED" if report.passed else "FAILED"
            stdout.write(
                f"{report.root_path}: {status} "
                f"(errors={report.error_count}, warnings={report.warning_count})\n"
            )
            for issue in report.issues:
                location = issue.path or ""
                if issue.line is not None:
                    location = f"{location}:{issue.line}" if location else str(issue.line)
                location_prefix = f" [{location}]" if location else ""
                stdout.write(
                    f"  - {issue.level.upper()} {issue.code}{location_prefix}: {issue.message}\n"
                )

    return 1 if any(not report.passed for report in reports) else 0


def main(
    argv: Sequence[str] | None = None,
    *,
    repo_root: Path | None = None,
    stdout: object | None = None,
) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    output = stdout or sys.stdout

    if args.paths:
        paths = normalize_paths(args.paths, cwd=root)
    else:
        paths = discover_default_paths(repo_root=root)

    if not paths:
        output.write("No extracted vertical repositories found to audit.\n")
        return 0

    return run_audit(paths, json_output=args.json, stdout=output)


if __name__ == "__main__":
    raise SystemExit(main())
