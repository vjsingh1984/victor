#!/usr/bin/env python3
"""Fail fast on foundational repo drift in first-party operational surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import importlib
import re
import sys

import yaml

BANNED_REPO_URLS = ("https://github.com/vijay-singh/codingagent",)

REPO_URL_SCAN_GLOBS = (
    "README.md",
    ".github/workflows/*.yml",
    "victor/**/victor-vertical.toml",
)

MARKDOWN_SCAN_GLOBS = (
    "README.md",
    "docs/**/*.md",
    "feps/**/*.md",
)

PRIMARY_VERTICAL_CONTRACT_FILES = (
    Path("pyproject.toml"),
    Path("scripts/scaffold_vertical.py"),
    Path("docs/guides/vertical-quickstart.md"),
    Path("docs/guides/FRAMEWORK_CAPABILITIES.md"),
    Path("docs/reference/verticals/index.md"),
    Path("docs/verticals/api_reference.md"),
    Path("docs/verticals/best_practices.md"),
    Path("docs/verticals/migration_guide.md"),
    Path("docs/demos/10-minute-vertical-demo.md"),
    Path("victor-sdk/README.md"),
    Path("victor-sdk/VERTICAL_DEVELOPMENT.md"),
    Path("victor-sdk/examples/minimal_vertical/README.md"),
    Path("victor-sdk/examples/minimal_vertical/pyproject.toml"),
    Path("examples/external_vertical/README.md"),
    Path("examples/external_vertical/pyproject.toml"),
)

STALE_VERTICAL_CONTRACT_PATTERNS = {
    '[project.entry-points."victor.verticals"]': (
        "primary vertical contract docs/examples must use victor.plugins, not victor.verticals"
    ),
    '[project.entry-points."victor.plugins.': (
        "primary vertical contract docs/examples must use the canonical victor.plugins"
        " group, not nested victor.plugins.* groups"
    ),
    "Must inherit from victor.core.verticals.VerticalBase": (
        "primary vertical contract docs/examples must point authors at victor_sdk.VerticalBase"
    ),
    "from victor.core.verticals import VerticalBase": (
        "primary vertical contract docs/examples must not import victor.core.verticals.VerticalBase"
    ),
    "from victor.verticals import VerticalBase": (
        "primary vertical contract docs/examples must point authors at victor_sdk.VerticalBase,"
        " not victor.verticals.VerticalBase"
    ),
    "from victor.framework.vertical_base import": (
        "primary vertical contract docs/examples must point authors at victor_sdk.VerticalBase,"
        " not victor.framework.vertical_base"
    ),
    "from victor.framework.extensions import VerticalBase": (
        "primary vertical contract docs/examples must point authors at victor_sdk.VerticalBase,"
        " not victor.framework.extensions.VerticalBase"
    ),
    "from victor.framework.extensions import StageDefinition": (
        "primary vertical contract docs/examples must point authors at victor_sdk.StageDefinition,"
        " not victor.framework.extensions.StageDefinition"
    ),
    "from victor.framework.extensions import VerticalConfig": (
        "primary vertical contract docs/examples must point authors at victor_sdk.VerticalConfig,"
        " not victor.framework.extensions.VerticalConfig"
    ),
    "from victor.core.verticals.registration import register_vertical": (
        "primary vertical contract docs/examples must import register_vertical from victor_sdk,"
        " not victor.core.verticals.registration"
    ),
    "victor.core.verticals.registration.register_vertical": (
        "primary vertical contract docs/examples must reference victor_sdk.register_vertical,"
        " not victor.core.verticals.registration.register_vertical"
    ),
    "from victor.verticals.base import StageDefinition": (
        "primary vertical contract docs/examples must point authors at victor_sdk.StageDefinition"
    ),
    "from victor.verticals.base import VerticalConfig": (
        "primary vertical contract docs/examples must point authors at victor_sdk.VerticalConfig"
    ),
    "VerticalRegistry.register(": (
        "primary vertical contract docs/examples must register external verticals through"
        " VictorPlugin/context.register_vertical(), not VerticalRegistry.register()"
    ),
    'get_entry_point("victor.verticals"': (
        "primary vertical contract docs/examples must use victor.plugins, not victor.verticals,"
        " for entry-point lookups"
    ),
    'get_entry_point_group("victor.verticals"': (
        "primary vertical contract docs/examples must use victor.plugins, not victor.verticals,"
        " for entry-point lookups"
    ),
}

ARCHIVED_DOC_BANNERS = {
    Path("docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md"): "Archived planning document",
}

LEGACY_PATHS_THAT_MUST_STAY_REMOVED = {
    Path("victor/agent/protocols.py"): (
        "legacy monolithic protocol module reintroduced; keep victor/agent/protocols/ as the"
        " canonical source"
    ),
}

ROADMAP_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]*ROADMAP\.md(?:#[^)]+)?)\)")
TARGET_HEADER_RE = re.compile(r"^[A-Za-z0-9_.-]+:\s*(?:#.*)?$")


@dataclass(frozen=True)
class HygieneFinding:
    """A single repo hygiene violation."""

    path: Path
    message: str


def _load_toml_module():
    """Load a TOML parser compatible with Python 3.10+."""
    try:
        return importlib.import_module("tomllib")
    except ModuleNotFoundError:
        return importlib.import_module("tomli")


tomllib = _load_toml_module()


def _iter_unique_files(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    files: set[Path] = set()
    for pattern in patterns:
        files.update(path for path in root.glob(pattern) if path.is_file())
    return sorted(files)


def _relative(path: Path, root: Path) -> Path:
    return path.relative_to(root)


def _workflow_on_config(data: object) -> object | None:
    if not isinstance(data, dict):
        return None
    if "on" in data:
        return data["on"]
    if True in data:
        return data[True]
    return None


def check_workflow_yaml(root: Path) -> list[HygieneFinding]:
    """Ensure workflow files parse and expose a real trigger configuration."""
    findings: list[HygieneFinding] = []
    workflows_dir = root / ".github" / "workflows"
    for path in sorted(workflows_dir.glob("*.yml")):
        rel_path = _relative(path, root)
        try:
            loaded = yaml.safe_load(path.read_text())
        except yaml.YAMLError as exc:
            findings.append(
                HygieneFinding(rel_path, f"workflow YAML failed to parse: {exc}")
            )
            continue

        if not isinstance(loaded, dict):
            findings.append(
                HygieneFinding(rel_path, "workflow must parse to a mapping")
            )
            continue

        name = loaded.get("name")
        if not isinstance(name, str) or not name.strip():
            findings.append(
                HygieneFinding(
                    rel_path, "workflow is missing a non-empty top-level name"
                )
            )

        on_config = _workflow_on_config(loaded)
        if on_config in (None, "", []):
            findings.append(
                HygieneFinding(
                    rel_path, "workflow is missing a non-empty top-level trigger (`on`)"
                ),
            )

    return findings


def check_banned_repo_urls(root: Path) -> list[HygieneFinding]:
    """Keep first-party operational files pointed at the canonical repo."""
    findings: list[HygieneFinding] = []
    for path in _iter_unique_files(root, REPO_URL_SCAN_GLOBS):
        text = path.read_text()
        for banned_url in BANNED_REPO_URLS:
            if banned_url in text:
                findings.append(
                    HygieneFinding(
                        _relative(path, root),
                        f"contains banned repo URL: {banned_url}",
                    )
                )
    return findings


def check_uppercase_roadmap_links(root: Path) -> list[HygieneFinding]:
    """Reject markdown links that point to the non-canonical uppercase roadmap path."""
    findings: list[HygieneFinding] = []
    for path in _iter_unique_files(root, MARKDOWN_SCAN_GLOBS):
        text = path.read_text()
        for match in ROADMAP_LINK_RE.finditer(text):
            findings.append(
                HygieneFinding(
                    _relative(path, root),
                    f"contains non-canonical roadmap link target: {match.group(1)}",
                )
            )
    return findings


def check_archived_doc_banners(root: Path) -> list[HygieneFinding]:
    """Ensure known archived planning docs advertise their non-canonical status."""
    findings: list[HygieneFinding] = []
    for rel_path, banner_text in ARCHIVED_DOC_BANNERS.items():
        path = root / rel_path
        if not path.is_file():
            findings.append(HygieneFinding(rel_path, "archived document is missing"))
            continue
        text = path.read_text()
        if banner_text not in text[:500]:
            findings.append(
                HygieneFinding(
                    rel_path,
                    f"archived document is missing the required banner text: {banner_text!r}",
                )
            )
    return findings


def check_removed_legacy_paths(root: Path) -> list[HygieneFinding]:
    """Prevent removed duplicate-source files from quietly returning."""
    findings: list[HygieneFinding] = []
    for rel_path, message in LEGACY_PATHS_THAT_MUST_STAY_REMOVED.items():
        if (root / rel_path).exists():
            findings.append(HygieneFinding(rel_path, message))
    return findings


def _extract_make_target(lines: list[str], target_name: str) -> list[str]:
    start_index: int | None = None
    for index, line in enumerate(lines):
        if line.startswith(f"{target_name}:"):
            start_index = index + 1
            break
    if start_index is None:
        return []

    block: list[str] = []
    for line in lines[start_index:]:
        if TARGET_HEADER_RE.match(line):
            break
        block.append(line)
    return block


def check_makefile_lint_gate(root: Path) -> list[HygieneFinding]:
    """Keep the local lint target aligned with the current mypy gate."""
    findings: list[HygieneFinding] = []
    path = root / "Makefile"
    lines = path.read_text().splitlines()
    lint_block = _extract_make_target(lines, "lint")
    rel_path = _relative(path, root)

    if not lint_block:
        return [HygieneFinding(rel_path, "Makefile is missing the lint target block")]

    mypy_lines = [line.strip() for line in lint_block if "mypy " in line]
    if not mypy_lines:
        findings.append(
            HygieneFinding(rel_path, "lint target is missing a mypy command")
        )
        return findings

    if not any("mypy victor" in line for line in mypy_lines):
        findings.append(HygieneFinding(rel_path, "lint target must run `mypy victor`"))

    if any("|| true" in line for line in mypy_lines):
        findings.append(
            HygieneFinding(
                rel_path, "lint target must not suppress mypy failure with `|| true`"
            )
        )

    return findings


def check_vertical_extra_metadata(root: Path) -> list[HygieneFinding]:
    """Reject stale no-op metadata for extracted vertical package extras."""

    path = root / "pyproject.toml"
    if not path.is_file():
        return []
    rel_path = _relative(path, root)
    try:
        data = tomllib.loads(path.read_text())
    except Exception as exc:
        return [HygieneFinding(rel_path, f"pyproject.toml failed to parse: {exc}")]

    project = data.get("project", {})
    optional_deps = project.get("optional-dependencies", {})
    if not isinstance(optional_deps, dict):
        return [
            HygieneFinding(rel_path, "project.optional-dependencies must be a mapping")
        ]
    project_name = str(project.get("name", "")).strip().lower()

    findings: list[HygieneFinding] = []
    for extra_name in ("coding", "research", "devops", "verticals"):
        values = optional_deps.get(extra_name)
        if not isinstance(values, list) or not values:
            findings.append(
                HygieneFinding(
                    rel_path,
                    f"extracted vertical extra '{extra_name}' must map to real package dependencies",
                )
            )

    for legacy_name in ("rag", "dataanalysis"):
        values = optional_deps.get(legacy_name)
        if isinstance(values, list) and not values:
            findings.append(
                HygieneFinding(
                    rel_path,
                    f"legacy no-op vertical extra '{legacy_name}' must not be present",
                )
            )

    if project_name:
        self_reference_prefix = f"{project_name}["
        for extra_name in ("dev", "ci", "verticals"):
            values = optional_deps.get(extra_name)
            if not isinstance(values, list):
                continue
            for value in values:
                if not isinstance(value, str):
                    continue
                if value.lower().startswith(self_reference_prefix):
                    findings.append(
                        HygieneFinding(
                            rel_path,
                            f"extra '{extra_name}' must not self-reference {project_name}[...]"
                            " because nested extras do not resolve reliably in CI",
                        )
                    )
                    break

    return findings


def check_primary_vertical_contract_docs(root: Path) -> list[HygieneFinding]:
    """Keep the primary authoring surfaces aligned with the plugin-first SDK contract."""

    findings: list[HygieneFinding] = []
    for rel_path in PRIMARY_VERTICAL_CONTRACT_FILES:
        path = root / rel_path
        if not path.is_file():
            continue

        text = path.read_text(encoding="utf-8")
        for needle, message in STALE_VERTICAL_CONTRACT_PATTERNS.items():
            if needle in text:
                findings.append(HygieneFinding(rel_path, message))

    return findings


def _workflow_has_blocking_trivy_step(path: Path) -> bool:
    try:
        loaded = yaml.safe_load(path.read_text())
    except yaml.YAMLError:
        return False

    if not isinstance(loaded, dict):
        return False

    jobs = loaded.get("jobs")
    if not isinstance(jobs, dict):
        return False

    for job in jobs.values():
        if not isinstance(job, dict):
            continue
        steps = job.get("steps")
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            uses = step.get("uses")
            if not isinstance(uses, str) or "aquasecurity/trivy-action" not in uses:
                continue
            config = step.get("with")
            if not isinstance(config, dict):
                continue
            severity = str(config.get("severity", ""))
            exit_code = str(config.get("exit-code", ""))
            ignore_unfixed = str(config.get("ignore-unfixed", "")).lower()
            continue_on_error = bool(step.get("continue-on-error", False))
            if (
                "CRITICAL" in severity
                and exit_code == "1"
                and ignore_unfixed == "true"
                and not continue_on_error
            ):
                return True
    return False


def _workflow_has_blocking_pip_audit_step(path: Path) -> bool:
    try:
        loaded = yaml.safe_load(path.read_text())
    except yaml.YAMLError:
        return False

    if not isinstance(loaded, dict):
        return False

    jobs = loaded.get("jobs")
    if not isinstance(jobs, dict):
        return False

    for job in jobs.values():
        if not isinstance(job, dict):
            continue
        steps = job.get("steps")
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            uses = step.get("uses")
            run = step.get("run")
            has_pip_audit_action = (
                isinstance(uses, str) and "pypa/gh-action-pip-audit" in uses
            )
            has_pip_audit_command = isinstance(run, str) and "pip-audit" in run
            if not has_pip_audit_action and not has_pip_audit_command:
                continue
            if bool(step.get("continue-on-error", False)):
                continue
            if isinstance(run, str) and "|| true" in run:
                continue
            return True
    return False


def _workflow_has_blocking_bandit_high_step(path: Path) -> bool:
    try:
        loaded = yaml.safe_load(path.read_text())
    except yaml.YAMLError:
        return False

    if not isinstance(loaded, dict):
        return False

    jobs = loaded.get("jobs")
    if not isinstance(jobs, dict):
        return False

    for job in jobs.values():
        if not isinstance(job, dict):
            continue
        steps = job.get("steps")
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            run = step.get("run")
            if not isinstance(run, str):
                continue
            if "bandit -r victor/" not in run:
                continue
            if (
                "--severity-level high" not in run
                or "--confidence-level high" not in run
            ):
                continue
            if bool(step.get("continue-on-error", False)):
                continue
            return True
    return False


def check_security_baseline(root: Path) -> list[HygieneFinding]:
    """Ensure the documented security baseline matches enforced workflow reality."""
    findings: list[HygieneFinding] = []

    blocking_sources = [
        root / ".github" / "workflows" / "ci-fast.yml",
        root / ".github" / "workflows" / "security.yml",
    ]
    if not any(
        _workflow_has_blocking_trivy_step(path)
        for path in blocking_sources
        if path.is_file()
    ):
        findings.append(
            HygieneFinding(
                Path(".github/workflows"),
                "missing a blocking Trivy step for CRITICAL findings (`exit-code: 1` without continue-on-error)",
            )
        )

    security_workflow = root / ".github" / "workflows" / "security.yml"
    if security_workflow.is_file():
        if not _workflow_has_blocking_pip_audit_step(security_workflow):
            findings.append(
                HygieneFinding(
                    Path(".github/workflows/security.yml"),
                    "missing a blocking pip-audit step in the security baseline",
                )
            )
        if not _workflow_has_blocking_bandit_high_step(security_workflow):
            findings.append(
                HygieneFinding(
                    Path(".github/workflows/security.yml"),
                    "missing a blocking Bandit step for HIGH severity + HIGH confidence findings",
                )
            )

    security_doc = root / "SECURITY.md"
    if security_doc.is_file():
        text = security_doc.read_text()
        if "## Current CI Enforcement Baseline" not in text:
            findings.append(
                HygieneFinding(
                    Path("SECURITY.md"),
                    "missing the Current CI Enforcement Baseline section",
                ),
            )
        if "**Blocking today**" not in text:
            findings.append(
                HygieneFinding(
                    Path("SECURITY.md"),
                    "missing the Blocking today security baseline summary",
                ),
            )
        if "**Advisory today**" not in text:
            findings.append(
                HygieneFinding(
                    Path("SECURITY.md"),
                    "missing the Advisory today security baseline summary",
                ),
            )
        if "### Current Thresholds" not in text:
            findings.append(
                HygieneFinding(
                    Path("SECURITY.md"), "missing the Current Thresholds section"
                ),
            )
        if "Trivy filesystem scan" not in text:
            findings.append(
                HygieneFinding(
                    Path("SECURITY.md"),
                    "missing Trivy filesystem scan threshold documentation",
                ),
            )
        if "| Dependency audit | Blocking |" not in text:
            findings.append(
                HygieneFinding(
                    Path("SECURITY.md"),
                    "missing blocking dependency-audit threshold documentation",
                ),
            )
        if "| Bandit (SAST) | Blocking |" not in text:
            findings.append(
                HygieneFinding(
                    Path("SECURITY.md"),
                    "missing blocking Bandit threshold documentation",
                ),
            )
    else:
        findings.append(
            HygieneFinding(Path("SECURITY.md"), "security policy document is missing")
        )

    return findings


def run_checks(root: Path) -> list[HygieneFinding]:
    """Run all foundational repo hygiene checks."""
    findings: list[HygieneFinding] = []
    findings.extend(check_workflow_yaml(root))
    findings.extend(check_banned_repo_urls(root))
    findings.extend(check_uppercase_roadmap_links(root))
    findings.extend(check_archived_doc_banners(root))
    findings.extend(check_removed_legacy_paths(root))
    findings.extend(check_makefile_lint_gate(root))
    findings.extend(check_vertical_extra_metadata(root))
    findings.extend(check_primary_vertical_contract_docs(root))
    findings.extend(check_security_baseline(root))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path.cwd(), help="Repository root to scan"
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    findings = run_checks(root)
    if not findings:
        print("Repo hygiene checks passed")
        return 0

    print("Repo hygiene checks failed:")
    for finding in findings:
        print(f"- {finding.path}: {finding.message}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
