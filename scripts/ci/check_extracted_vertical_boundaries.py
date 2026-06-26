#!/usr/bin/env python3
"""Audit vertical packages against Victor's contract boundary.

The first-party verticals now live in-repo under ``verticals/`` (folded in from
their former standalone repos). This audit enforces that they import only the
``victor_contracts`` contract surface — never ``victor`` framework internals —
which is what keeps the monorepo from eroding the decoupling that the separate
repos used to provide.

Additional repos/paths can be audited by passing explicit paths (e.g. a still
external ``../victor-invest``).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_RELATIVE_EXTRACTED_REPO_PATHS = (
    # In-repo first-party verticals (folded in under verticals/).
    "verticals/victor-coding",
    "verticals/victor-research",
    "verticals/victor-devops",
    "verticals/victor-rag",
    "verticals/victor-dataanalysis",
    # Still-external verticals are audited only when checked out next to the repo.
    "../victor-invest",
)


def normalize_extracted_repo_paths(
    paths: Iterable[str | Path],
    *,
    cwd: Path,
) -> list[Path]:
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


def discover_default_extracted_repo_paths(*, repo_root: Path) -> list[Path]:
    """Discover existing extracted vertical repos next to the core repo."""
    return [
        path
        for path in normalize_extracted_repo_paths(
            DEFAULT_RELATIVE_EXTRACTED_REPO_PATHS,
            cwd=repo_root,
        )
        if path.exists() and path.is_dir()
    ]


def _load_vertical_contract_auditor(repo_root: Path) -> type:
    """Load the auditor module without importing the vertical runtime package."""

    module_path = repo_root / "victor" / "core" / "verticals" / "contract_audit.py"
    if not module_path.exists():
        module_path = (
            Path(__file__).resolve().parents[2] / "victor/core/verticals/contract_audit.py"
        )
    spec = importlib.util.spec_from_file_location(
        "_victor_contract_audit_cli",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load contract auditor from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.VerticalContractAuditor


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
    repo_root: Path,
    json_output: bool,
    stdout: object,
) -> int:
    """Run the contract audit and emit a summary."""
    VerticalContractAuditor = _load_vertical_contract_auditor(repo_root)
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
        paths = normalize_extracted_repo_paths(args.paths, cwd=root)
    else:
        paths = discover_default_extracted_repo_paths(repo_root=root)

    if not paths:
        output.write("No extracted vertical repositories found to audit.\n")
        return 0

    return run_audit(paths, repo_root=root, json_output=args.json, stdout=output)


if __name__ == "__main__":
    raise SystemExit(main())
