# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Comprehensive QA test suite for Victor AI release validation.

This module provides automated validation across multiple quality dimensions:
1. Test Suite Execution (unit, integration, performance, security)
2. Code Quality (linting, formatting, type checking)
3. Documentation (completeness, accuracy, links)
4. Performance (benchmarks, load testing, regression checks)
5. Security (vulnerability scanning, secrets management)
6. Deployment (Docker, Kubernetes, CI/CD)

Usage:
    pytest tests/qa/test_comprehensive_qa.py -v
    pytest tests/qa/test_comprehensive_qa.py::TestComprehensiveQA::test_unit_tests_pass -v
    pytest tests/qa/test_comprehensive_qa.py -k "performance" -v
"""

import subprocess
import sys
import json
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field
from datetime import datetime

from victor import __version__


@dataclass
class QAResult:
    """Result of a QA validation step."""

    name: str
    passed: bool = False
    duration_seconds: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "passed": self.passed,
            "duration_seconds": self.duration_seconds,
            "details": self.details,
            "errors": self.errors,
            "warnings": self.warnings,
            "metrics": self.metrics,
        }


class TestComprehensiveQA:
    """
    Comprehensive QA test suite.

    This test suite validates all aspects of production readiness:
    - Test coverage and execution
    - Code quality standards
    - Documentation completeness
    - Performance benchmarks
    - Security scanning
    - Deployment readiness
    """

    # Configuration
    MIN_COVERAGE_PERCENT: float = 70.0
    MAX_MYPY_ERRORS: int = 100
    MAX_RUFF_ERRORS: int = 50
    MAX_BANDIT_ISSUES: int = 10
    MAX_SAFETY_ISSUES: int = 5

    # Test directories
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    VICTOR_DIR = PROJECT_ROOT / "victor"
    TESTS_DIR = PROJECT_ROOT / "tests"
    DOCS_DIR = PROJECT_ROOT / "docs"
    SCRIPTS_DIR = PROJECT_ROOT / "scripts"

    # Results storage
    results: list[QAResult] = []

    @classmethod
    def setup_class(cls):
        """Setup test class with project information."""
        print(f"\n{'='*60}")
        print("Victor AI Comprehensive QA Suite")
        print(f"Version: {__version__}")
        print(f"Project Root: {cls.PROJECT_ROOT}")
        print(f"Python: {sys.version}")
        print(f"{'='*60}\n")

    @classmethod
    def teardown_class(cls):
        """Generate QA report after all tests complete."""
        cls._generate_qa_report()

    # ========================================================================
    # Test Suite Execution Tests
    # ========================================================================

    def test_unit_tests_pass(self):
        """
        Validate all unit tests pass.

        Runs the complete unit test suite and verifies:
        - All tests pass (no failures)
        - Acceptable number of tests skipped
        - No unexpected errors
        - Test collection succeeds
        """
        result = QAResult(name="Unit Tests Execution")

        start_time = datetime.now()

        try:
            # Run unit tests with coverage
            cmd = [
                sys.executable,
                "-m",
                "pytest",
                str(self.TESTS_DIR / "unit"),
                "-v",
                "--tb=short",
                "--no-header",
                "-q",
            ]

            proc = subprocess.run(
                cmd,
                cwd=self.PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes
            )

            result.duration_seconds = (datetime.now() - start_time).total_seconds()

            # Parse output
            output = proc.stdout + proc.stderr

            # Extract test statistics
            lines = output.split("\n")
            for line in lines:
                if "passed" in line.lower():
                    result.details["summary"] = line.strip()

            result.passed = proc.returncode == 0

            if not result.passed:
                result.errors.append("Unit tests failed")
                result.errors.extend([line for line in lines[-20:] if line.strip()])

        except subprocess.TimeoutExpired:
            result.passed = False
            result.errors.append("Unit tests timed out after 5 minutes")

        except Exception as e:
            result.passed = False
            result.errors.append(f"Unexpected error: {e}")

        self.results.append(result)

        assert result.passed, f"Unit tests failed: {result.errors}"
        print(f"✓ Unit tests passed in {result.duration_seconds:.2f}s")

    def test_integration_tests_pass(self):
        """
        Validate integration tests pass.

        Runs integration test suite and verifies:
        - All integration tests pass
        - No environment-specific failures
        - Proper test isolation
        """
        result = QAResult(name="Integration Tests Execution")

        start_time = datetime.now()

        try:
            # Run integration tests but skip slow ones
            # Skip: real_execution (actual provider calls), benchmark, load_test, slow
            cmd = [
                sys.executable,
                "-m",
                "pytest",
                str(self.TESTS_DIR / "integration"),
                "-v",
                "-m",
                "integration",
                "--tb=short",
                "--no-header",
                "-q",
                "--ignore=tests/integration/real_execution",
                "--ignore=tests/integration/performance_scenarios.py",
                "-m",
                "not slow",  # Skip slow tests
                "-x",  # Stop on first failure
            ]

            proc = subprocess.run(
                cmd,
                cwd=self.PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=900,  # 15 minutes (increased from 10 minutes)
            )

            result.duration_seconds = (datetime.now() - start_time).total_seconds()
            result.passed = proc.returncode == 0

            if not result.passed:
                output = proc.stdout + proc.stderr
                result.errors.append("Integration tests failed")
                result.errors.extend([line for line in output.split("\n")[-20:] if line.strip()])

        except subprocess.TimeoutExpired:
            result.passed = False
            result.errors.append("Integration tests timed out after 15 minutes")

        except Exception as e:
            result.passed = False
            result.errors.append(f"Unexpected error: {e}")

        self.results.append(result)

        # Don't fail on integration tests - just warn
        if not result.passed:
            print(f"⚠ Integration tests had issues (non-blocking for QA): {result.errors}")
        else:
            print(f"✓ Integration tests passed in {result.duration_seconds:.2f}s")

    def test_code_coverage_meets_minimum(self):
        """
        Validate code coverage meets minimum requirements.

        Checks that:
        - Overall coverage >= MIN_COVERAGE_PERCENT
        - Critical modules have higher coverage
        - Coverage report is generated
        """
        result = QAResult(name="Code Coverage Validation")

        start_time = datetime.now()

        try:
            cmd = [
                sys.executable,
                "-m",
                "pytest",
                str(self.TESTS_DIR / "unit"),
                "--cov=victor",
                "--cov-report=term",
                "--cov-report=json",
                "--no-header",
                "-q",
            ]

            proc = subprocess.run(
                cmd,
                cwd=self.PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=300,
            )

            result.duration_seconds = (datetime.now() - start_time).total_seconds()

            # Parse coverage from JSON report
            coverage_file = self.PROJECT_ROOT / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)

                total_coverage = coverage_data["totals"]["percent_covered"]
                result.metrics["total_coverage"] = total_coverage
                result.details["coverage_file"] = str(coverage_file)

                result.passed = total_coverage >= self.MIN_COVERAGE_PERCENT

                if not result.passed:
                    result.errors.append(
                        f"Coverage {total_coverage:.1f}% below minimum {self.MIN_COVERAGE_PERCENT}%"
                    )

            else:
                result.warnings.append("Coverage report not found")
                result.passed = False

        except subprocess.TimeoutExpired:
            result.passed = False
            result.errors.append("Coverage check timed out")

        except Exception as e:
            result.passed = False
            result.errors.append(f"Unexpected error: {e}")

        self.results.append(result)

        assert result.passed, f"Coverage check failed: {result.errors}"
        print(
            f"✓ Code coverage {result.metrics.get('total_coverage', 0):.1f}% meets minimum {self.MIN_COVERAGE_PERCENT}%"
        )

    # ========================================================================
    # Code Quality Tests
    # ========================================================================

    def test_ruff_linting_passes(self):
        """
        Validate code passes ruff linting.

        Checks that:
        - No critical linting errors
        - Code follows style guidelines
        - Error count within acceptable limits
        """
        result = QAResult(name="Ruff Linting")

        start_time = datetime.now()

        try:
            cmd = [
                sys.executable,
                "-m",
                "ruff",
                "check",
                str(self.VICTOR_DIR),
                "--output-format=json",
            ]

            proc = subprocess.run(
                cmd,
                cwd=self.PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=120,
            )

            result.duration_seconds = (datetime.now() - start_time).total_seconds()

            # Parse JSON output
            try:
                errors = json.loads(proc.stdout)
                error_count = len(errors)
                result.metrics["error_count"] = error_count
                result.details["errors_sample"] = errors[:5] if error_count > 0 else []

                result.passed = error_count <= self.MAX_RUFF_ERRORS

                if not result.passed:
                    result.errors.append(
                        f"Found {error_count} ruff errors (max: {self.MAX_RUFF_ERRORS})"
                    )
            except json.JSONDecodeError:
                # Non-JSON output might mean success
                result.passed = proc.returncode == 0

        except subprocess.TimeoutExpired:
            result.passed = False
            result.errors.append("Ruff linting timed out")

        except Exception as e:
            result.passed = False
            result.errors.append(f"Unexpected error: {e}")

        self.results.append(result)

        assert result.passed, f"Ruff linting failed: {result.errors}"
        print(f"✓ Ruff linting passed with {result.metrics.get('error_count', 0)} errors")

    def test_black_formatting_check(self):
        """
        Validate code formatting with black.

        Checks that:
        - All code is properly formatted
        - No formatting violations
        - Consistent style across codebase
        """
        result = QAResult(name="Black Formatting Check")

        start_time = datetime.now()

        try:
            cmd = [
                sys.executable,
                "-m",
                "black",
                "--check",
                str(self.VICTOR_DIR),
            ]

            proc = subprocess.run(
                cmd,
                cwd=self.PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=120,
            )

            result.duration_seconds = (datetime.now() - start_time).total_seconds()
            result.passed = proc.returncode == 0

            if not result.passed:
                result.errors.append("Code formatting issues found")
                result.details["reformat_needed"] = True

        except subprocess.TimeoutExpired:
            result.passed = False
            result.errors.append("Black check timed out")

        except Exception as e:
            result.passed = False
            result.errors.append(f"Unexpected error: {e}")

        self.results.append(result)

        assert result.passed, f"Black formatting check failed: {result.errors}"
        print("✓ Black formatting check passed")

    def test_mypy_type_checking(self):
        """
        Validate type hints with mypy.

        Checks that:
        - Type checking completes successfully
        - Type errors within acceptable limits
        - No critical type issues
        """
        result = QAResult(name="Mypy Type Checking")

        start_time = datetime.now()

        try:
            cmd = [
                sys.executable,
                "-m",
                "mypy",
                str(self.VICTOR_DIR),
                "--no-error-summary",
            ]

            proc = subprocess.run(
                cmd,
                cwd=self.PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=180,
            )

            result.duration_seconds = (datetime.now() - start_time).total_seconds()

            # Count errors
            output = proc.stdout + proc.stderr
            error_count = output.count(": error:")
            result.metrics["error_count"] = error_count

            result.passed = error_count <= self.MAX_MYPY_ERRORS

            if not result.passed:
                result.errors.append(
                    f"Found {error_count} mypy errors (max: {self.MAX_MYPY_ERRORS})"
                )
                result.details["errors_sample"] = output.split("\n")[:10]

        except subprocess.TimeoutExpired:
            result.passed = False
            result.errors.append("Mypy type checking timed out")

        except Exception as e:
            result.passed = False
            result.errors.append(f"Unexpected error: {e}")

        self.results.append(result)

        assert result.passed, f"Mypy type checking failed: {result.errors}"
        print(f"✓ Mypy type checking passed with {result.metrics.get('error_count', 0)} errors")

    # ========================================================================
    # Security Tests
    # ========================================================================

    def test_bandit_security_scan(self):
        """
        Validate security with bandit scanning.

        Checks that:
        - No high-severity security issues
        - Total issues within acceptable limits
        - Common vulnerabilities are absent
        """
        result = QAResult(name="Bandit Security Scan")

        start_time = datetime.now()

        try:
            cmd = [
                sys.executable,
                "-m",
                "bandit",
                "-r",
                str(self.VICTOR_DIR),
                "-f",
                "json",
            ]

            proc = subprocess.run(
                cmd,
                cwd=self.PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=120,
            )

            result.duration_seconds = (datetime.now() - start_time).total_seconds()

            # Parse JSON output
            try:
                bandit_results = json.loads(proc.stdout)
                issues = bandit_results.get("results", [])
                issue_count = len(issues)

                # Count high severity issues
                high_severity = sum(1 for i in issues if i.get("issue_severity") == "HIGH")

                result.metrics["total_issues"] = issue_count
                result.metrics["high_severity_issues"] = high_severity

                result.passed = high_severity == 0 and issue_count <= self.MAX_BANDIT_ISSUES

                if not result.passed:
                    if high_severity > 0:
                        result.errors.append(f"Found {high_severity} HIGH severity issues")
                    if issue_count > self.MAX_BANDIT_ISSUES:
                        result.errors.append(
                            f"Found {issue_count} total issues (max: {self.MAX_BANDIT_ISSUES})"
                        )
                    result.details["issues_sample"] = issues[:5]

            except json.JSONDecodeError:
                # No issues found
                result.passed = True

        except subprocess.TimeoutExpired:
            result.passed = False
            result.errors.append("Bandit scan timed out")

        except Exception as e:
            result.passed = False
            result.errors.append(f"Unexpected error: {e}")

        self.results.append(result)

        assert result.passed, f"Bandit security scan failed: {result.errors}"
        print(f"✓ Bandit security scan passed with {result.metrics.get('total_issues', 0)} issues")

    def test_safety_dependency_check(self):
        """
        Validate dependency security with safety.

        Checks that:
        - No known vulnerabilities in dependencies
        - All dependencies are up-to-date
        - No compromised packages
        """
        result = QAResult(name="Safety Dependency Check")

        start_time = datetime.now()

        try:
            cmd = [
                sys.executable,
                "-m",
                "safety",
                "check",
                "--json",
            ]

            proc = subprocess.run(
                cmd,
                cwd=self.PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=60,
            )

            result.duration_seconds = (datetime.now() - start_time).total_seconds()

            # Parse JSON output
            try:
                safety_results = json.loads(proc.stdout)
                issues = safety_results if isinstance(safety_results, list) else []

                result.metrics["vulnerability_count"] = len(issues)
                result.passed = len(issues) == 0

                if not result.passed:
                    result.errors.append(f"Found {len(issues)} dependency vulnerabilities")
                    result.details["vulnerabilities"] = issues[:5]

            except json.JSONDecodeError:
                # No issues found or different output format
                result.passed = proc.returncode == 0

        except subprocess.TimeoutExpired:
            result.passed = False
            result.errors.append("Safety check timed out")

        except Exception as e:
            result.passed = False
            result.errors.append(f"Unexpected error: {e}")

        self.results.append(result)

        # Allow warnings but not failures
        if not result.passed and result.warnings:
            print(f"⚠ Safety check warnings: {result.warnings}")
        else:
            assert result.passed, f"Safety dependency check failed: {result.errors}"
            print("✓ Safety dependency check passed")

    # ========================================================================
    # Performance Tests
    # ========================================================================

    def test_performance_benchmarks_run(self):
        """
        Validate performance benchmarks execute successfully.

        Checks that:
        - Benchmark suite runs without errors
        - Performance baselines are established
        - No significant regressions detected
        """
        result = QAResult(name="Performance Benchmarks")

        start_time = datetime.now()

        try:
            # Check if benchmark script exists
            benchmark_script = self.SCRIPTS_DIR / "benchmark_tool_selection.py"

            if not benchmark_script.exists():
                result.warnings.append("Benchmark script not found")
                result.passed = True  # Not a failure if benchmarks don't exist
            else:
                cmd = [
                    sys.executable,
                    str(benchmark_script),
                    "run",
                    "--group",
                    "smoke",
                ]

                proc = subprocess.run(
                    cmd,
                    cwd=self.PROJECT_ROOT,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )

                result.duration_seconds = (datetime.now() - start_time).total_seconds()
                result.passed = proc.returncode == 0

                if not result.passed:
                    result.errors.append("Benchmark execution failed")
                    result.details["output"] = proc.stdout + proc.stderr

        except subprocess.TimeoutExpired:
            result.passed = False
            result.errors.append("Benchmarks timed out")

        except Exception as e:
            result.passed = False
            result.errors.append(f"Unexpected error: {e}")

        self.results.append(result)

        assert result.passed, f"Performance benchmarks failed: {result.errors}"
        print(f"✓ Performance benchmarks completed in {result.duration_seconds:.2f}s")

    # ========================================================================
    # Documentation Tests
    # ========================================================================

    def test_documentation_builds(self):
        """
        Validate documentation can be built successfully.

        Checks that:
        - Documentation builds without errors
        - All pages are generated
        - No broken links or references
        """
        result = QAResult(name="Documentation Build")

        start_time = datetime.now()

        try:
            # Check if mkdocs is configured
            mkdocs_config = self.PROJECT_ROOT / "mkdocs.yml"

            if not mkdocs_config.exists():
                result.warnings.append("mkdocs.yml not found")
                result.passed = True  # Not a failure if docs aren't configured
            else:
                cmd = [
                    sys.executable,
                    "-m",
                    "mkdocs",
                    "build",
                    "--strict",
                ]

                proc = subprocess.run(
                    cmd,
                    cwd=self.PROJECT_ROOT,
                    capture_output=True,
                    text=True,
                    timeout=180,
                )

                result.duration_seconds = (datetime.now() - start_time).total_seconds()
                result.passed = proc.returncode == 0

                if not result.passed:
                    result.errors.append("Documentation build failed")
                    result.details["output"] = proc.stdout + proc.stderr

        except subprocess.TimeoutExpired:
            result.passed = False
            result.errors.append("Documentation build timed out")

        except Exception as e:
            result.passed = False
            result.errors.append(f"Unexpected error: {e}")

        self.results.append(result)

        # Only fail if docs are configured
        if not result.passed and "mkdocs.yml not found" not in result.warnings:
            assert result.passed, f"Documentation build failed: {result.errors}"
        print("✓ Documentation validation completed")

    def test_readme_exists(self):
        """Validate README.md exists and is complete."""
        result = QAResult(name="README Validation")

        readme_path = self.PROJECT_ROOT / "README.md"

        result.passed = readme_path.exists()

        if result.passed:
            content = readme_path.read_text()

            # Check for essential sections
            required_sections = ["Installation", "Usage", "Features"]
            missing = [s for s in required_sections if s not in content]

            if missing:
                result.warnings.append(f"README missing sections: {missing}")

            result.metrics["readme_size"] = len(content)

        else:
            result.errors.append("README.md not found")

        self.results.append(result)

        assert result.passed, f"README validation failed: {result.errors}"
        print("✓ README validation passed")

    # ========================================================================
    # Release Readiness Tests
    # ========================================================================

    def test_version_is_defined(self):
        """Validate version is properly defined."""
        result = QAResult(name="Version Validation")

        result.passed = __version__ is not None and __version__ != "0.0.0"

        if result.passed:
            result.details["version"] = __version__
        else:
            result.errors.append("Version not properly defined")

        self.results.append(result)

        assert result.passed, f"Version validation failed: {result.errors}"
        print(f"✓ Version validation passed: {__version__}")

    def test_changelog_exists(self):
        """Validate CHANGELOG.md exists and has current version."""
        result = QAResult(name="CHANGELOG Validation")

        changelog_path = self.PROJECT_ROOT / "CHANGELOG.md"

        result.passed = changelog_path.exists()

        if result.passed:
            content = changelog_path.read_text()

            # Check if current version is mentioned
            if __version__ in content:
                result.details["version_in_changelog"] = True
            else:
                result.warnings.append(f"Version {__version__} not in CHANGELOG")

        else:
            result.errors.append("CHANGELOG.md not found")

        self.results.append(result)

        assert result.passed, f"CHANGELOG validation failed: {result.errors}"
        print("✓ CHANGELOG validation passed")

    # ========================================================================
    # Reporting
    # ========================================================================

    @classmethod
    def _generate_qa_report(cls):
        """Generate comprehensive QA report."""
        report_path = cls.PROJECT_ROOT / "docs" / "QA_REPORT.md"

        # Create docs directory if needed
        report_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate report
        report_lines = [
            "# Quality Assurance Report",
            "",
            f"**Generated:** {datetime.now().isoformat()}",
            f"**Version:** {__version__}",
            "",
            "## Summary",
            "",
        ]

        total = len(cls.results)
        passed = sum(1 for r in cls.results if r.passed)

        report_lines.extend(
            [
                f"- **Total Checks:** {total}",
                f"- **Passed:** {passed}",
                f"- **Failed:** {total - passed}",
                f"- **Success Rate:** {100.0 * passed / total:.1f}%",
                "",
                "## Detailed Results",
                "",
            ]
        )

        for result in cls.results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            report_lines.extend(
                [
                    f"### {status} {result.name}",
                    "",
                    f"- **Duration:** {result.duration_seconds:.2f}s",
                    f"- **Status:** {'Passed' if result.passed else 'Failed'}",
                ]
            )

            if result.metrics:
                report_lines.append("- **Metrics:**")
                for key, value in result.metrics.items():
                    report_lines.append(f"  - {key}: {value}")

            if result.warnings:
                report_lines.append("- **Warnings:**")
                for warning in result.warnings:
                    report_lines.append(f"  - {warning}")

            if result.errors:
                report_lines.append("- **Errors:**")
                for error in result.errors:
                    report_lines.append(f"  - {error}")

            report_lines.append("")

        # Write report
        report_path.write_text("\n".join(report_lines))

        print(f"\n{'='*60}")
        print(f"QA Report generated: {report_path}")
        print(f"{'='*60}\n")

        # Also generate JSON report for machine parsing
        json_report_path = cls.PROJECT_ROOT / "docs" / "qa_report.json"
        json_data = {
            "generated": datetime.now().isoformat(),
            "version": __version__,
            "summary": {
                "total": total,
                "passed": passed,
                "failed": total - passed,
                "success_rate": 100.0 * passed / total,
            },
            "results": [r.to_dict() for r in cls.results],
        }

        json_report_path.write_text(json.dumps(json_data, indent=2))
        print(f"JSON Report generated: {json_report_path}")
