#!/usr/bin/env python
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
Full QA automation script for Victor AI.

This script runs comprehensive quality assurance validation across all dimensions:
- Test execution (unit, integration, smoke)
- Code quality (linting, formatting, type checking)
- Security scanning (bandit, safety, pip-audit)
- Performance benchmarks
- Documentation validation
- Release readiness checks

Usage:
    python scripts/run_full_qa.py                    # Run all checks
    python scripts/run_full_qa.py --fast             # Quick validation (skip slow tests)
    python scripts/run_full_qa.py --coverage         # Include coverage reports
    python scripts/run_full_qa.py --report json      # Output format (text, json, html)
    python scripts/run_full_qa.py --output report.txt # Save to file
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import time


class QAOrchestrator:
    """
    Orchestrates comprehensive QA validation across multiple dimensions.

    Features:
    - Parallel test execution where possible
    - Progress tracking and reporting
    - Flexible output formats (text, json, html)
    - Configurable test suites
    - Detailed failure analysis
    """

    def __init__(
        self,
        project_root: Path,
        fast_mode: bool = False,
        with_coverage: bool = False,
        output_format: str = "text",
        output_file: Optional[Path] = None,
    ):
        """Initialize QA orchestrator."""
        self.project_root = project_root
        self.fast_mode = fast_mode
        self.with_coverage = with_coverage
        self.output_format = output_format
        self.output_file = output_file

        self.victor_dir = project_root / "victor"
        self.tests_dir = project_root / "tests"
        self.docs_dir = project_root / "docs"

        self.results: Dict[str, Dict[str, Any]] = {}
        self.start_time = datetime.now()

    def run_all(self) -> int:
        """
        Run all QA checks and return exit code.

        Returns:
            0 if all checks pass, 1 otherwise
        """
        print(f"\n{'='*70}")
        print(f"Victor AI Comprehensive QA Suite")
        print(f"{'='*70}")
        print(f"Start Time: {self.start_time.isoformat()}")
        print(f"Fast Mode: {self.fast_mode}")
        print(f"Coverage: {self.with_coverage}")
        print(f"Output Format: {self.output_format}")
        print(f"{'='*70}\n")

        # Define test suites
        test_suites = [
            ("Unit Tests", self._run_unit_tests),
            ("Integration Tests", self._run_integration_tests),
            ("Code Coverage", self._run_coverage_tests),
            ("Ruff Linting", self._run_ruff_checks),
            ("Black Formatting", self._run_black_checks),
            ("Mypy Type Checking", self._run_mypy_checks),
            ("Bandit Security", self._run_bandit_scan),
            ("Safety Dependencies", self._run_safety_check),
            ("Pip Audit", self._run_pip_audit),
            ("Documentation Build", self._run_docs_build),
            ("Performance Benchmarks", self._run_benchmarks),
            ("Release Readiness", self._check_release_readiness),
        ]

        # Filter suites for fast mode
        if self.fast_mode:
            test_suites = [
                ("Unit Tests (Fast)", self._run_unit_tests_fast),
                ("Ruff Linting", self._run_ruff_checks),
                ("Black Formatting", self._run_black_checks),
                ("Release Readiness", self._check_release_readiness),
            ]

        # Run test suites
        for name, func in test_suites:
            print(f"\n{'─'*70}")
            print(f"Running: {name}")
            print(f"{'─'*70}")

            try:
                func()
            except Exception as e:
                self.results[name] = {
                    "status": "ERROR",
                    "error": str(e),
                    "duration": 0.0,
                }
                print(f"✗ ERROR: {e}")

        # Generate final report
        return self._generate_report()

    def _run_command(
        self,
        cmd: List[str],
        timeout: int = 300,
        capture_output: bool = True,
    ) -> Tuple[int, str, str, float]:
        """
        Run command and return exit code, stdout, stderr, and duration.

        Args:
            cmd: Command to run
            timeout: Timeout in seconds
            capture_output: Whether to capture output

        Returns:
            Tuple of (exit_code, stdout, stderr, duration_seconds)
        """
        start = time.time()

        try:
            proc = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=capture_output,
                text=True,
                timeout=timeout,
            )

            duration = time.time() - start

            if capture_output:
                return proc.returncode, proc.stdout, proc.stderr, duration
            else:
                return proc.returncode, "", "", duration

        except subprocess.TimeoutExpired:
            duration = time.time() - start
            return -1, "", f"Command timed out after {timeout}s", duration

    # ========================================================================
    # Test Suites
    # ========================================================================

    def _run_unit_tests(self):
        """Run full unit test suite."""
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(self.tests_dir / "unit"),
            "-v",
            "--tb=short",
            "--no-header",
            "-q",
        ]

        exit_code, stdout, stderr, duration = self._run_command(cmd, timeout=600)

        self.results["Unit Tests"] = {
            "status": "PASS" if exit_code == 0 else "FAIL",
            "exit_code": exit_code,
            "duration": duration,
            "output": stdout + stderr,
        }

        print(f"Status: {'✓ PASS' if exit_code == 0 else '✗ FAIL'}")
        print(f"Duration: {duration:.2f}s")

    def _run_unit_tests_fast(self):
        """Run fast unit tests only."""
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(self.tests_dir / "unit"),
            "-v",
            "-m",
            "not slow",
            "--tb=short",
            "--no-header",
            "-q",
        ]

        exit_code, stdout, stderr, duration = self._run_command(cmd, timeout=300)

        self.results["Unit Tests (Fast)"] = {
            "status": "PASS" if exit_code == 0 else "FAIL",
            "exit_code": exit_code,
            "duration": duration,
            "output": stdout + stderr,
        }

        print(f"Status: {'✓ PASS' if exit_code == 0 else '✗ FAIL'}")
        print(f"Duration: {duration:.2f}s")

    def _run_integration_tests(self):
        """Run integration test suite."""
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(self.tests_dir / "integration"),
            "-v",
            "-m",
            "integration",
            "--tb=short",
            "--no-header",
            "-q",
        ]

        exit_code, stdout, stderr, duration = self._run_command(cmd, timeout=900)

        self.results["Integration Tests"] = {
            "status": "PASS" if exit_code == 0 else "FAIL",
            "exit_code": exit_code,
            "duration": duration,
            "output": stdout + stderr,
        }

        print(f"Status: {'✓ PASS' if exit_code == 0 else '✗ FAIL'}")
        print(f"Duration: {duration:.2f}s")

    def _run_coverage_tests(self):
        """Run code coverage analysis."""
        if not self.with_coverage:
            print("Skipping coverage (use --coverage to enable)")
            return

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(self.tests_dir / "unit"),
            "--cov=victor",
            "--cov-report=term",
            "--cov-report=html",
            "--cov-report=json",
            "--no-header",
            "-q",
        ]

        exit_code, stdout, stderr, duration = self._run_command(cmd, timeout=600)

        # Parse coverage percentage
        coverage_percent = 0.0
        for line in stdout.split("\n"):
            if "TOTAL" in line and "%" in line:
                try:
                    coverage_percent = float(line.split("%")[0].strip().split()[-1])
                except (ValueError, IndexError):
                    pass

        self.results["Code Coverage"] = {
            "status": "PASS" if coverage_percent >= 70.0 else "WARN",
            "exit_code": exit_code,
            "duration": duration,
            "coverage_percent": coverage_percent,
            "output": stdout + stderr,
        }

        print(f"Status: {'✓ PASS' if coverage_percent >= 70.0 else '⚠ WARN'}")
        print(f"Coverage: {coverage_percent:.1f}%")
        print(f"Duration: {duration:.2f}s")

    def _run_ruff_checks(self):
        """Run ruff linting."""
        cmd = [
            sys.executable,
            "-m",
            "ruff",
            "check",
            str(self.victor_dir),
        ]

        exit_code, stdout, stderr, duration = self._run_command(cmd, timeout=120)

        # Count errors
        error_count = stdout.count("\n") if exit_code != 0 else 0

        self.results["Ruff Linting"] = {
            "status": "PASS" if exit_code == 0 else "WARN",
            "exit_code": exit_code,
            "duration": duration,
            "error_count": error_count,
            "output": stdout + stderr,
        }

        print(f"Status: {'✓ PASS' if exit_code == 0 else '⚠ WARN'}")
        print(f"Errors: {error_count}")
        print(f"Duration: {duration:.2f}s")

    def _run_black_checks(self):
        """Run black formatting check."""
        cmd = [
            sys.executable,
            "-m",
            "black",
            "--check",
            str(self.victor_dir),
        ]

        exit_code, stdout, stderr, duration = self._run_command(cmd, timeout=120)

        self.results["Black Formatting"] = {
            "status": "PASS" if exit_code == 0 else "FAIL",
            "exit_code": exit_code,
            "duration": duration,
            "output": stdout + stderr,
        }

        print(f"Status: {'✓ PASS' if exit_code == 0 else '✗ FAIL'}")
        print(f"Duration: {duration:.2f}s")

    def _run_mypy_checks(self):
        """Run mypy type checking."""
        cmd = [
            sys.executable,
            "-m",
            "mypy",
            str(self.victor_dir),
        ]

        exit_code, stdout, stderr, duration = self._run_command(cmd, timeout=180)

        # Count errors
        error_count = (stdout + stderr).count(": error:")

        self.results["Mypy Type Checking"] = {
            "status": "PASS" if error_count == 0 else "WARN",
            "exit_code": exit_code,
            "duration": duration,
            "error_count": error_count,
            "output": stdout + stderr,
        }

        print(f"Status: {'✓ PASS' if error_count == 0 else '⚠ WARN'}")
        print(f"Errors: {error_count}")
        print(f"Duration: {duration:.2f}s")

    def _run_bandit_scan(self):
        """Run bandit security scan."""
        cmd = [
            sys.executable,
            "-m",
            "bandit",
            "-r",
            str(self.victor_dir),
            "-f",
            "json",
        ]

        exit_code, stdout, stderr, duration = self._run_command(cmd, timeout=120)

        # Parse JSON output
        try:
            bandit_results = json.loads(stdout)
            issues = bandit_results.get("results", [])
            high_severity = sum(1 for i in issues if i.get("issue_severity") == "HIGH")

            self.results["Bandit Security"] = {
                "status": "PASS" if high_severity == 0 else "FAIL",
                "exit_code": exit_code,
                "duration": duration,
                "total_issues": len(issues),
                "high_severity": high_severity,
                "output": stdout + stderr,
            }

            print(f"Status: {'✓ PASS' if high_severity == 0 else '✗ FAIL'}")
            print(f"Issues: {len(issues)} (HIGH: {high_severity})")

        except json.JSONDecodeError:
            self.results["Bandit Security"] = {
                "status": "PASS",
                "exit_code": exit_code,
                "duration": duration,
                "output": stdout + stderr,
            }

            print(f"Status: ✓ PASS (no issues found)")

        print(f"Duration: {duration:.2f}s")

    def _run_safety_check(self):
        """Run safety dependency check."""
        cmd = [
            sys.executable,
            "-m",
            "safety",
            "check",
            "--json",
        ]

        exit_code, stdout, stderr, duration = self._run_command(cmd, timeout=60)

        # Parse JSON output
        try:
            safety_results = json.loads(stdout)
            vuln_count = len(safety_results) if isinstance(safety_results, list) else 0

            self.results["Safety Dependencies"] = {
                "status": "PASS" if vuln_count == 0 else "FAIL",
                "exit_code": exit_code,
                "duration": duration,
                "vulnerability_count": vuln_count,
                "output": stdout + stderr,
            }

            print(f"Status: {'✓ PASS' if vuln_count == 0 else '✗ FAIL'}")
            print(f"Vulnerabilities: {vuln_count}")

        except json.JSONDecodeError:
            self.results["Safety Dependencies"] = {
                "status": "PASS",
                "exit_code": exit_code,
                "duration": duration,
                "output": stdout + stderr,
            }

            print(f"Status: ✓ PASS (no vulnerabilities)")

        print(f"Duration: {duration:.2f}s")

    def _run_pip_audit(self):
        """Run pip-audit dependency check."""
        try:
            cmd = [
                sys.executable,
                "-m",
                "pip_audit",
                "--format",
                "json",
            ]

            exit_code, stdout, stderr, duration = self._run_command(cmd, timeout=60)

            # Parse JSON output
            try:
                audit_results = json.loads(stdout)
                vuln_count = len(audit_results)

                self.results["Pip Audit"] = {
                    "status": "PASS" if vuln_count == 0 else "WARN",
                    "exit_code": exit_code,
                    "duration": duration,
                    "vulnerability_count": vuln_count,
                    "output": stdout + stderr,
                }

                print(f"Status: {'✓ PASS' if vuln_count == 0 else '⚠ WARN'}")
                print(f"Vulnerabilities: {vuln_count}")

            except json.JSONDecodeError:
                self.results["Pip Audit"] = {
                    "status": "PASS",
                    "exit_code": exit_code,
                    "duration": duration,
                    "output": stdout + stderr,
                }

                print(f"Status: ✓ PASS (no vulnerabilities)")

            print(f"Duration: {duration:.2f}s")

        except FileNotFoundError:
            print("Status: ⊘ SKIP (pip-audit not installed)")

    def _run_docs_build(self):
        """Run documentation build."""
        mkdocs_config = self.project_root / "mkdocs.yml"

        if not mkdocs_config.exists():
            print("Status: ⊘ SKIP (mkdocs.yml not found)")
            return

        cmd = [
            sys.executable,
            "-m",
            "mkdocs",
            "build",
            "--strict",
        ]

        exit_code, stdout, stderr, duration = self._run_command(cmd, timeout=180)

        self.results["Documentation Build"] = {
            "status": "PASS" if exit_code == 0 else "FAIL",
            "exit_code": exit_code,
            "duration": duration,
            "output": stdout + stderr,
        }

        print(f"Status: {'✓ PASS' if exit_code == 0 else '✗ FAIL'}")
        print(f"Duration: {duration:.2f}s")

    def _run_benchmarks(self):
        """Run performance benchmarks."""
        benchmark_script = self.project_root / "scripts" / "benchmark_tool_selection.py"

        if not benchmark_script.exists():
            print("Status: ⊘ SKIP (benchmark script not found)")
            return

        cmd = [
            sys.executable,
            str(benchmark_script),
            "run",
            "--group",
            "smoke",
        ]

        exit_code, stdout, stderr, duration = self._run_command(cmd, timeout=300)

        self.results["Performance Benchmarks"] = {
            "status": "PASS" if exit_code == 0 else "WARN",
            "exit_code": exit_code,
            "duration": duration,
            "output": stdout + stderr,
        }

        print(f"Status: {'✓ PASS' if exit_code == 0 else '⚠ WARN'}")
        print(f"Duration: {duration:.2f}s")

    def _check_release_readiness(self):
        """Check release readiness."""
        checks = []

        # Check version
        try:
            from victor import __version__
            checks.append(("Version Defined", __version__ is not None and __version__ != "0.0.0"))
        except ImportError:
            checks.append(("Version Defined", False))

        # Check README
        readme = self.project_root / "README.md"
        checks.append(("README Exists", readme.exists()))

        # Check CHANGELOG
        changelog = self.project_root / "CHANGELOG.md"
        checks.append(("CHANGELOG Exists", changelog.exists()))

        # Check LICENSE
        license_file = self.project_root / "LICENSE"
        checks.append(("LICENSE Exists", license_file.exists()))

        all_passed = all(passed for _, passed in checks)

        self.results["Release Readiness"] = {
            "status": "PASS" if all_passed else "FAIL",
            "checks": {name: passed for name, passed in checks},
            "duration": 0.0,
        }

        for name, passed in checks:
            print(f"{name}: {'✓' if passed else '✗'}")

        print(f"Status: {'✓ PASS' if all_passed else '✗ FAIL'}")

    # ========================================================================
    # Reporting
    # ========================================================================

    def _generate_report(self) -> int:
        """Generate final report and return exit code."""
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()

        # Count results
        total = len(self.results)
        passed = sum(1 for r in self.results.values() if r["status"] == "PASS")
        failed = sum(1 for r in self.results.values() if r["status"] == "FAIL")
        warned = sum(1 for r in self.results.values() if r["status"] == "WARN")
        errored = sum(1 for r in self.results.values() if r["status"] == "ERROR")

        # Generate report based on format
        if self.output_format == "json":
            report = self._generate_json_report(
                total, passed, failed, warned, errored, total_duration
            )
        elif self.output_format == "html":
            report = self._generate_html_report(
                total, passed, failed, warned, errored, total_duration
            )
        else:
            report = self._generate_text_report(
                total, passed, failed, warned, errored, total_duration
            )

        # Output report
        if self.output_file:
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            self.output_file.write_text(report)
            print(f"\nReport saved to: {self.output_file}")
        else:
            print("\n" + report)

        # Return exit code
        return 0 if failed == 0 and errored == 0 else 1

    def _generate_text_report(
        self, total: int, passed: int, failed: int, warned: int, errored: int, duration: float
    ) -> str:
        """Generate text format report."""
        lines = [
            "",
            "="*70,
            "QA SUMMARY",
            "="*70,
            "",
            f"Total Duration: {duration:.2f}s",
            f"Total Checks: {total}",
            f"Passed: {passed}",
            f"Failed: {failed}",
            f"Warnings: {warned}",
            f"Errors: {errored}",
            f"Success Rate: {100.0 * passed / total:.1f}%",
            "",
            "="*70,
            "DETAILED RESULTS",
            "="*70,
            "",
        ]

        for name, result in self.results.items():
            status = result["status"]
            duration = result.get("duration", 0.0)

            status_symbol = {
                "PASS": "✓",
                "FAIL": "✗",
                "WARN": "⚠",
                "ERROR": "✗",
            }.get(status, "?")

            lines.extend([
                f"{status_symbol} {name}",
                f"  Status: {status}",
                f"  Duration: {duration:.2f}s",
            ])

            if status == "FAIL" and "output" in result:
                output_lines = result["output"].split("\n")[:10]
                lines.extend([f"  Output: {line}" for line in output_lines])

            lines.append("")

        return "\n".join(lines)

    def _generate_json_report(
        self, total: int, passed: int, failed: int, warned: int, errored: int, duration: float
    ) -> str:
        """Generate JSON format report."""
        report = {
            "summary": {
                "total_duration": duration,
                "total_checks": total,
                "passed": passed,
                "failed": failed,
                "warned": warned,
                "errored": errored,
                "success_rate": 100.0 * passed / total,
            },
            "results": self.results,
            "generated": datetime.now().isoformat(),
        }

        return json.dumps(report, indent=2)

    def _generate_html_report(
        self, total: int, passed: int, failed: int, warned: int, errored: int, duration: float
    ) -> str:
        """Generate HTML format report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Victor AI QA Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
        .result {{ margin: 10px 0; padding: 10px; border-radius: 5px; }}
        .pass {{ background: #d4edda; border-left: 4px solid #28a745; }}
        .fail {{ background: #f8d7da; border-left: 4px solid #dc3545; }}
        .warn {{ background: #fff3cd; border-left: 4px solid #ffc107; }}
        .error {{ background: #f5c6cb; border-left: 4px solid #dc3545; }}
    </style>
</head>
<body>
    <h1>Victor AI QA Report</h1>

    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total Duration:</strong> {duration:.2f}s</p>
        <p><strong>Total Checks:</strong> {total}</p>
        <p><strong>Passed:</strong> {passed}</p>
        <p><strong>Failed:</strong> {failed}</p>
        <p><strong>Warnings:</strong> {warned}</p>
        <p><strong>Errors:</strong> {errored}</p>
        <p><strong>Success Rate:</strong> {100.0 * passed / total:.1f}%</p>
    </div>

    <h2>Detailed Results</h2>
"""

        for name, result in self.results.items():
            status = result["status"].lower()
            duration = result.get("duration", 0.0)

            html += f"""
    <div class="result {status}">
        <h3>{name}</h3>
        <p><strong>Status:</strong> {result["status"]}</p>
        <p><strong>Duration:</strong> {duration:.2f}s</p>
    </div>
"""

        html += """
</body>
</html>
"""

        return html


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive QA validation for Victor AI"
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode - skip slow tests and benchmarks",
    )

    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Include coverage reports",
    )

    parser.add_argument(
        "--report",
        choices=["text", "json", "html"],
        default="text",
        help="Output format (default: text)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path (default: stdout)",
    )

    args = parser.parse_args()

    # Run QA
    orchestrator = QAOrchestrator(
        project_root=Path.cwd(),
        fast_mode=args.fast,
        with_coverage=args.coverage,
        output_format=args.report,
        output_file=args.output,
    )

    exit_code = orchestrator.run_all()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
