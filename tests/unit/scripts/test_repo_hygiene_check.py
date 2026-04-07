from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace


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


def test_load_toml_module_falls_back_to_tomli(monkeypatch) -> None:
    calls: list[str] = []
    tomli_module = SimpleNamespace(loads=lambda text: {"ok": text})

    def fake_import_module(name: str):
        calls.append(name)
        if name == "tomllib":
            raise ModuleNotFoundError(name)
        if name == "tomli":
            return tomli_module
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(repo_hygiene_check.importlib, "import_module", fake_import_module)

    loaded = repo_hygiene_check._load_toml_module()

    assert loaded is tomli_module
    assert calls == ["tomllib", "tomli"]


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


def test_vertical_extras_require_real_package_dependencies(tmp_path: Path) -> None:
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(
        tmp_path, "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md", "Archived planning document\n"
    )
    write_file(
        tmp_path,
        "pyproject.toml",
        """
[project]
name = "victor-ai"

[project.optional-dependencies]
coding = []
research = ["victor-research>=0.6.0"]
devops = ["victor-devops>=0.6.0"]
verticals = ["victor-coding>=0.6.0", "victor-research>=0.6.0", "victor-devops>=0.6.0"]
        """.strip() + "\n",
    )

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any("extracted vertical extra 'coding'" in finding.message for finding in findings)


def test_legacy_noop_vertical_extras_are_rejected(tmp_path: Path) -> None:
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(
        tmp_path, "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md", "Archived planning document\n"
    )
    write_file(
        tmp_path,
        "pyproject.toml",
        """
[project]
name = "victor-ai"

[project.optional-dependencies]
coding = ["victor-coding>=0.6.0"]
research = ["victor-research>=0.6.0"]
devops = ["victor-devops>=0.6.0"]
verticals = ["victor-coding>=0.6.0", "victor-research>=0.6.0", "victor-devops>=0.6.0"]
rag = []
        """.strip() + "\n",
    )

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any("legacy no-op vertical extra 'rag'" in finding.message for finding in findings)


def test_self_referential_ci_extra_is_rejected(tmp_path: Path) -> None:
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(
        tmp_path, "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md", "Archived planning document\n"
    )
    write_file(
        tmp_path,
        "pyproject.toml",
        """
[project]
name = "victor-ai"

[project.optional-dependencies]
coding = ["victor-coding>=0.6.0"]
research = ["victor-research>=0.6.0"]
devops = ["victor-devops>=0.6.0"]
verticals = ["victor-coding>=0.6.0", "victor-research>=0.6.0", "victor-devops>=0.6.0"]
ci = ["victor-ai[dev]", "pytest-split>=0.8"]
        """.strip() + "\n",
    )

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any("must not self-reference victor-ai" in finding.message for finding in findings)


def test_primary_vertical_contract_docs_reject_legacy_entry_point_guidance(tmp_path: Path) -> None:
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(
        tmp_path, "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md", "Archived planning document\n"
    )
    write_file(
        tmp_path,
        "pyproject.toml",
        """
[project]
name = "victor-ai"

[project.optional-dependencies]
coding = ["victor-coding>=0.6.0"]
research = ["victor-research>=0.6.0"]
devops = ["victor-devops>=0.6.0"]
verticals = ["victor-coding>=0.6.0", "victor-research>=0.6.0", "victor-devops>=0.6.0"]

[project.entry-points."victor.plugins"]
benchmark = "victor.benchmark:plugin"

# Requirements for external verticals:
#   1. Must inherit from victor.core.verticals.VerticalBase
        """.strip() + "\n",
    )

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any(
        "victor_sdk.VerticalBase" in finding.message or "victor.plugins" in finding.message
        for finding in findings
    )


def test_primary_vertical_contract_docs_reject_legacy_victor_verticals_examples(
    tmp_path: Path,
) -> None:
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(
        tmp_path, "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md", "Archived planning document\n"
    )
    write_file(
        tmp_path,
        "pyproject.toml",
        """
[project]
name = "victor-ai"

[project.optional-dependencies]
coding = ["victor-coding>=0.6.0"]
research = ["victor-research>=0.6.0"]
devops = ["victor-devops>=0.6.0"]
verticals = ["victor-coding>=0.6.0", "victor-research>=0.6.0", "victor-devops>=0.6.0"]
        """.strip() + "\n",
    )
    write_file(
        tmp_path,
        "victor-sdk/README.md",
        """
[project.entry-points."victor.verticals"]
security = "victor_security:SecurityVertical"
        """.strip() + "\n",
    )

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any("victor.plugins" in finding.message for finding in findings)


def test_primary_vertical_contract_docs_reject_nested_victor_plugins_groups(
    tmp_path: Path,
) -> None:
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(
        tmp_path, "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md", "Archived planning document\n"
    )
    write_file(
        tmp_path,
        "docs/verticals/best_practices.md",
        """
[project.entry-points."victor.plugins.my_company"]
tool = "my_package.plugin:plugin"
        """.strip() + "\n",
    )

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any("nested victor.plugins.* groups" in finding.message for finding in findings)


def test_primary_vertical_contract_docs_reject_framework_shim_examples(tmp_path: Path) -> None:
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(
        tmp_path, "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md", "Archived planning document\n"
    )
    write_file(
        tmp_path,
        "pyproject.toml",
        """
[project]
name = "victor-ai"

[project.optional-dependencies]
coding = ["victor-coding>=0.6.0"]
research = ["victor-research>=0.6.0"]
devops = ["victor-devops>=0.6.0"]
verticals = ["victor-coding>=0.6.0", "victor-research>=0.6.0", "victor-devops>=0.6.0"]
        """.strip() + "\n",
    )
    write_file(
        tmp_path,
        "victor-sdk/VERTICAL_DEVELOPMENT.md",
        """
from victor.framework.vertical_base import VerticalBase
        """.strip() + "\n",
    )

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any("victor.framework.vertical_base" in finding.message for finding in findings)


def test_primary_vertical_contract_docs_reject_core_register_vertical_imports(
    tmp_path: Path,
) -> None:
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(
        tmp_path, "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md", "Archived planning document\n"
    )
    write_file(
        tmp_path,
        "docs/verticals/api_reference.md",
        """
from victor.core.verticals.registration import register_vertical

# Public API reference
victor.core.verticals.registration.register_vertical
        """.strip() + "\n",
    )

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any(
        "import register_vertical from victor_sdk" in finding.message for finding in findings
    )
    assert any("victor_sdk.register_vertical" in finding.message for finding in findings)


def test_primary_vertical_contract_docs_reject_legacy_entry_point_lookup_examples(
    tmp_path: Path,
) -> None:
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(
        tmp_path, "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md", "Archived planning document\n"
    )
    write_file(
        tmp_path,
        "docs/verticals/api_reference.md",
        """
from victor.framework.entry_point_registry import get_entry_point

coding = get_entry_point("victor.verticals", "coding")
        """.strip() + "\n",
    )

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any("entry-point lookups" in finding.message for finding in findings)


def test_primary_vertical_contract_docs_reject_runtime_vertical_authoring_examples(
    tmp_path: Path,
) -> None:
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(
        tmp_path, "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md", "Archived planning document\n"
    )
    write_file(
        tmp_path,
        "docs/reference/verticals/index.md",
        """
from victor.verticals import VerticalBase
from victor.verticals.base import StageDefinition
from victor.verticals.base import VerticalConfig
from victor.verticals import VerticalRegistry

VerticalRegistry.register(MyVertical)
        """.strip() + "\n",
    )

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any("victor_sdk.VerticalBase" in finding.message for finding in findings)
    assert any("victor_sdk.StageDefinition" in finding.message for finding in findings)
    assert any("victor_sdk.VerticalConfig" in finding.message for finding in findings)
    assert any("VictorPlugin/context.register_vertical" in finding.message for finding in findings)


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


def test_security_baseline_accepts_shell_based_blocking_pip_audit(tmp_path: Path) -> None:
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(
        tmp_path,
        ".github/workflows/ci-fast.yml",
        "name: CI\non: push\njobs:\n  sec:\n    steps:\n      - uses: aquasecurity/trivy-action@0.33.1\n        with:\n          severity: CRITICAL\n          exit-code: '1'\n          ignore-unfixed: 'true'\n",
    )
    write_file(
        tmp_path,
        ".github/workflows/security.yml",
        "name: Security\non: push\njobs:\n  sec:\n    steps:\n      - uses: aquasecurity/trivy-action@0.33.1\n        with:\n          severity: CRITICAL\n          exit-code: '1'\n          ignore-unfixed: 'true'\n      - name: Run pip-audit\n        run: |\n          pip install pip-audit\n          pip-audit --format json --output audit-report.json\n      - name: Run Bandit\n        run: |\n          bandit -r victor/ --severity-level high --confidence-level high\n",
    )
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

    assert not any("blocking pip-audit step" in finding.message for finding in findings)


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
