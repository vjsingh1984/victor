#!/usr/bin/env python3
"""
Victor Migration Validation Script

This script validates that a migration from Victor 0.5.x to 1.0.0 was successful.

Usage:
    python scripts/migration/validate_migration.py [--directory ./]
    python scripts/migration/validate_migration.py --check-imports
    python scripts/migration/validate_migration.py --check-config
"""

import argparse
import ast
import os
import sys
from pathlib import Path
from typing import List, Tuple


class MigrationValidator:
    """Validates Victor 0.5.x to 1.0.0 migration."""

    def __init__(self, directory: Path = None):
        self.directory = directory or Path.cwd()
        self.issues = []
        self.warnings = []

    def validate_all(self) -> bool:
        """Run all validation checks."""
        print(f"Validating migration in {self.directory}\n")

        checks = [
            ("Imports", self.check_imports),
            ("Configuration", self.check_configuration),
            ("Code Patterns", self.check_code_patterns),
            ("Dependencies", self.check_dependencies),
        ]

        results = []
        for name, check in checks:
            print(f"Checking {name}...")
            try:
                result = check()
                results.append(result)
                if result:
                    print(f"✓ {name} validation passed")
                else:
                    print(f"✗ {name} validation failed")
            except Exception as e:
                print(f"✗ {name} validation error: {e}")
                results.append(False)
            print()

        return all(results)

    def check_imports(self) -> bool:
        """Check for old import patterns."""
        old_imports = {
            "victor.config.Config": "victor.config.settings.Settings",
            "victor.state_machine": "victor.agent.coordinators.state_coordinator",
            "victor.tool_selector": "victor.agent.coordinators.tool_coordinator",
            "victor.event_bus.EventBus": "victor.core.events.create_event_backend",
            "victor.protocols.base": "victor.protocols",
        }

        issues_found = False

        for py_file in self.directory.rglob("*.py"):
            # Skip tests and migrations
            if "test" in py_file.parts or "migration" in py_file.parts:
                continue

            content = py_file.read_text()

            for old_import, new_import in old_imports.items():
                if old_import in content:
                    self.issues.append(
                        f"{py_file}: Found old import '{old_import}', should use '{new_import}'"
                    )
                    issues_found = True

        if issues_found:
            print("Old import patterns found:")
            for issue in self.issues[:5]:
                print(f"  ✗ {issue}")
            if len(self.issues) > 5:
                print(f"  ... and {len(self.issues) - 5} more")

        return not issues_found

    def check_configuration(self) -> bool:
        """Check configuration files."""
        issues_found = False

        # Check for .env file
        env_file = self.directory / ".env"
        if env_file.exists():
            content = env_file.read_text()

            # Check for deprecated variables
            deprecated_vars = [
                "VICTOR_API_KEY=",
                "VICTOR_USE_SEMANTIC=",
                "VICTOR_CACHE_ENABLED=",
            ]

            for var in deprecated_vars:
                if var in content:
                    self.issues.append(
                        f"{env_file}: Found deprecated variable '{var.strip('=')}'"
                    )
                    issues_found = True

        # Check for old Config usage
        for py_file in self.directory.rglob("*.py"):
            content = py_file.read_text()
            if "Config()" in content and "from victor.config" in content:
                self.issues.append(
                    f"{py_file}: Found old Config() usage, should use Settings()"
                )
                issues_found = True

        if issues_found:
            print("Configuration issues found:")
            for issue in self.issues[:5]:
                print(f"  ✗ {issue}")
            if len(self.issues) > 5:
                print(f"  ... and {len(self.issues) - 5} more")

        return not issues_found

    def check_code_patterns(self) -> bool:
        """Check for old code patterns."""
        issues_found = False

        for py_file in self.directory.rglob("*.py"):
            # Skip tests
            if "test" in py_file.parts:
                continue

            content = py_file.read_text()
            lines = content.split('\n')

            for i, line in enumerate(lines, 1):
                # Check for direct orchestrator instantiation
                if "AgentOrchestrator(" in line:
                    self.issues.append(
                        f"{py_file}:{i}: Found AgentOrchestrator() instantiation, "
                        "use bootstrap_orchestrator()"
                    )
                    issues_found = True

                # Check for API key in provider constructors
                if "api_key=" in line and "Provider(" in line:
                    self.issues.append(
                        f"{py_file}:{i}: Found api_key parameter in provider, "
                        "use environment variables"
                    )
                    issues_found = True

                # Check for singleton access
                if "SharedToolRegistry.get_instance()" in line:
                    self.issues.append(
                        f"{py_file}:{i}: Found SharedToolRegistry singleton, "
                        "use DI container"
                    )
                    issues_found = True

                # Check for EventBus instantiation
                if "EventBus()" in line:
                    self.issues.append(
                        f"{py_file}:{i}: Found EventBus() instantiation, "
                        "use create_event_backend()"
                    )
                    issues_found = True

                # Check for sync chat calls
                if "orchestrator.chat(" in line and "await" not in lines[max(0, i-2):i]:
                    self.warnings.append(
                        f"{py_file}:{i}: Possible sync chat call, should use await"
                    )

        if issues_found:
            print("Code pattern issues found:")
            for issue in self.issues[:5]:
                print(f"  ✗ {issue}")
            if len(self.issues) > 5:
                print(f"  ... and {len(self.issues) - 5} more")

        if self.warnings:
            print("\nWarnings:")
            for warning in self.warnings[:5]:
                print(f"  ⚠ {warning}")
            if len(self.warnings) > 5:
                print(f"  ... and {len(self.warnings) - 5} more")

        return not issues_found

    def check_dependencies(self) -> bool:
        """Check Python dependencies."""
        try:
            import victor
            version = getattr(victor, "__version__", "unknown")

            # Parse version
            try:
                major, minor, patch = map(int, version.split('.')[:3])
                if major >= 1:
                    print(f"✓ Victor version: {version}")
                    return True
                else:
                    print(f"✗ Victor version {version} is not 1.0.0+")
                    return False
            except:
                print(f"⚠ Could not parse Victor version: {version}")
                return False

        except ImportError:
            print("✗ Victor not installed")
            return False

    def generate_report(self) -> str:
        """Generate validation report."""
        report = []
        report.append("=" * 80)
        report.append("Migration Validation Report")
        report.append("=" * 80)
        report.append(f"Directory: {self.directory}")
        report.append(f"Timestamp: {os.popen('date').read().strip()}")
        report.append("")

        if self.issues:
            report.append(f"Issues Found: {len(self.issues)}")
            report.append("")
            for issue in self.issues:
                report.append(f"  ✗ {issue}")
            report.append("")

        if self.warnings:
            report.append(f"Warnings: {len(self.warnings)}")
            report.append("")
            for warning in self.warnings:
                report.append(f"  ⚠ {warning}")
            report.append("")

        if not self.issues and not self.warnings:
            report.append("✓ No issues found - migration appears successful!")
        else:
            report.append("Please address the issues above to complete migration.")

        report.append("=" * 80)

        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(
        description="Validate Victor migration from 0.5.x to 1.0.0"
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path.cwd(),
        help="Directory to validate (default: current directory)"
    )
    parser.add_argument(
        "--check-imports",
        action="store_true",
        help="Only check imports"
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Only check configuration"
    )
    parser.add_argument(
        "--check-patterns",
        action="store_true",
        help="Only check code patterns"
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Save validation report to file"
    )

    args = parser.parse_args()

    validator = MigrationValidator(args.directory)

    # Run specific check or all checks
    if args.check_imports:
        success = validator.check_imports()
    elif args.check_config:
        success = validator.check_configuration()
    elif args.check_patterns:
        success = validator.check_code_patterns()
    else:
        success = validator.validate_all()

    # Generate report
    if args.report:
        report = validator.generate_report()
        args.report.write_text(report)
        print(f"\nReport saved to: {args.report}")

    # Print summary
    print("\n" + "=" * 80)
    if success:
        print("✓ Validation passed - migration looks good!")
    else:
        print("✗ Validation failed - issues found that need attention")
    print("=" * 80)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
