from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def load_repo_hygiene_module():
    module_path = Path(__file__).resolve().parents[3] / "scripts" / "ci" / "repo_hygiene_check.py"
    spec = importlib.util.spec_from_file_location("repo_hygiene_check", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


repo_hygiene_check = load_repo_hygiene_module()


def write_file(root: Path, relative_path: str, content: str) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_workflow_check_flags_missing_top_level_on(tmp_path: Path) -> None:
    write_file(
        tmp_path,
        ".github/workflows/bad.yml",
        "name: Broken\non:\npermissions:\n  contents: read\n  pull_request:\n    paths:\n      - 'victor/**'\n",
    )
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(
        tmp_path, "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md", "Archived planning document\n"
    )

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any("non-empty top-level trigger" in finding.message for finding in findings)


def test_banned_repo_url_is_flagged(tmp_path: Path) -> None:
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(
        tmp_path, "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md", "Archived planning document\n"
    )
    write_file(
        tmp_path,
        "victor/verticals/contrib/coding/victor-vertical.toml",
        'repository = "https://github.com/vijay-singh/codingagent"\n',
    )

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any("banned repo URL" in finding.message for finding in findings)


def test_uppercase_roadmap_markdown_link_is_flagged(tmp_path: Path) -> None:
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(
        tmp_path, "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md", "Archived planning document\n"
    )
    write_file(tmp_path, "README.md", "[Roadmap](ROADMAP.md)\n")

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any("non-canonical roadmap link target" in finding.message for finding in findings)


def test_archived_doc_requires_banner(tmp_path: Path) -> None:
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(tmp_path, "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md", "# Missing banner\n")

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any("required banner text" in finding.message for finding in findings)


def test_legacy_monolithic_protocol_module_is_flagged(tmp_path: Path) -> None:
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(
        tmp_path, "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md", "Archived planning document\n"
    )
    write_file(tmp_path, "victor/agent/protocols.py", "class Legacy: ...\n")

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any("legacy monolithic protocol module" in finding.message for finding in findings)


def test_makefile_lint_gate_rejects_advisory_mypy(tmp_path: Path) -> None:
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(tmp_path, ".github/workflows/ci-fast.yml", "name: CI\non: push\njobs: {}\n")
    write_file(tmp_path, ".github/workflows/security.yml", "name: Security\non: push\njobs: {}\n")
    write_file(
        tmp_path,
        "Makefile",
        "lint:\n\truff check victor tests\n\tmypy victor --ignore-missing-imports || true\n",
    )
    write_file(
        tmp_path, "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md", "Archived planning document\n"
    )
    write_file(
        tmp_path,
        "SECURITY.md",
        "## Current CI Enforcement Baseline\n### Current Thresholds\nTrivy filesystem scan\n",
    )

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any("must not suppress mypy failure" in finding.message for finding in findings)


def test_security_baseline_requires_blocking_trivy_path(tmp_path: Path) -> None:
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(tmp_path, ".github/workflows/ci-fast.yml", "name: CI\non: push\njobs: {}\n")
    write_file(tmp_path, ".github/workflows/security.yml", "name: Security\non: push\njobs: {}\n")
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(
        tmp_path, "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md", "Archived planning document\n"
    )
    write_file(
        tmp_path,
        "SECURITY.md",
        "## Current CI Enforcement Baseline\n### Current Thresholds\nTrivy filesystem scan\n",
    )

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any("blocking Trivy step" in finding.message for finding in findings)


def test_security_baseline_requires_blocking_pip_audit_path(tmp_path: Path) -> None:
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(
        tmp_path,
        ".github/workflows/security.yml",
        "name: Security\non: push\njobs:\n  sec:\n    steps:\n      - uses: aquasecurity/trivy-action@master\n        with:\n          severity: CRITICAL\n          exit-code: '1'\n          ignore-unfixed: 'true'\n",
    )
    write_file(tmp_path, ".github/workflows/ci-fast.yml", "name: CI\non: push\njobs: {}\n")
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(
        tmp_path, "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md", "Archived planning document\n"
    )
    write_file(
        tmp_path,
        "SECURITY.md",
        "## Current CI Enforcement Baseline\n**Blocking today**\n**Advisory today**\n### Current Thresholds\nTrivy filesystem scan\n| Dependency audit | Blocking |\n| Bandit (SAST) | Blocking |\n",
    )

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any("blocking pip-audit step" in finding.message for finding in findings)


def test_security_baseline_requires_blocking_bandit_high_path(tmp_path: Path) -> None:
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(
        tmp_path,
        ".github/workflows/security.yml",
        "name: Security\non: push\njobs:\n  sec:\n    steps:\n      - uses: aquasecurity/trivy-action@master\n        with:\n          severity: CRITICAL\n          exit-code: '1'\n          ignore-unfixed: 'true'\n      - uses: pypa/gh-action-pip-audit@v1.0.8\n",
    )
    write_file(tmp_path, ".github/workflows/ci-fast.yml", "name: CI\non: push\njobs: {}\n")
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(
        tmp_path, "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md", "Archived planning document\n"
    )
    write_file(
        tmp_path,
        "SECURITY.md",
        "## Current CI Enforcement Baseline\n**Blocking today**\n**Advisory today**\n### Current Thresholds\nTrivy filesystem scan\n| Dependency audit | Blocking |\n| Bandit (SAST) | Blocking |\n",
    )

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any("blocking Bandit step" in finding.message for finding in findings)


def test_security_baseline_requires_ignore_unfixed_for_blocking_trivy_path(tmp_path: Path) -> None:
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(
        tmp_path,
        ".github/workflows/ci-fast.yml",
        "name: CI\non: push\njobs:\n  sec:\n    steps:\n      - uses: aquasecurity/trivy-action@master\n        with:\n          severity: CRITICAL\n          exit-code: '1'\n",
    )
    write_file(tmp_path, ".github/workflows/security.yml", "name: Security\non: push\njobs: {}\n")
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(
        tmp_path, "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md", "Archived planning document\n"
    )
    write_file(
        tmp_path,
        "SECURITY.md",
        "## Current CI Enforcement Baseline\n**Blocking today**\n**Advisory today**\n### Current Thresholds\nTrivy filesystem scan\n",
    )

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any("blocking Trivy step" in finding.message for finding in findings)


def test_security_baseline_requires_threshold_docs(tmp_path: Path) -> None:
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(
        tmp_path,
        ".github/workflows/ci-fast.yml",
        "name: CI\non: push\njobs:\n  sec:\n    steps:\n      - uses: aquasecurity/trivy-action@master\n        with:\n          severity: CRITICAL\n          exit-code: '1'\n          ignore-unfixed: 'true'\n",
    )
    write_file(tmp_path, ".github/workflows/security.yml", "name: Security\non: push\njobs: {}\n")
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(
        tmp_path, "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md", "Archived planning document\n"
    )
    write_file(tmp_path, "SECURITY.md", "## Current CI Enforcement Baseline\n")

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any("Current Thresholds section" in finding.message for finding in findings)


def test_security_baseline_requires_blocking_and_advisory_summary_docs(tmp_path: Path) -> None:
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(
        tmp_path,
        ".github/workflows/ci-fast.yml",
        "name: CI\non: push\njobs:\n  sec:\n    steps:\n      - uses: aquasecurity/trivy-action@master\n        with:\n          severity: CRITICAL\n          exit-code: '1'\n          ignore-unfixed: 'true'\n",
    )
    write_file(tmp_path, ".github/workflows/security.yml", "name: Security\non: push\njobs: {}\n")
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(
        tmp_path, "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md", "Archived planning document\n"
    )
    write_file(
        tmp_path, "SECURITY.md", "## Current CI Enforcement Baseline\n### Current Thresholds\n"
    )

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any("Blocking today" in finding.message for finding in findings)
    assert any("Advisory today" in finding.message for finding in findings)


def test_current_repo_passes_hygiene_checks() -> None:
    findings = repo_hygiene_check.run_checks(Path.cwd())

    assert findings == []
