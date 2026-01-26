#!/usr/bin/env python3
"""Protocol Conformance Checker for Victor

This tool verifies that verticals and other components correctly implement
their declared protocols. It checks method signatures, required attributes,
and generates compliance reports.

Usage:
    python scripts/check_protocol_conformance.py --vertical victor/coding
    python scripts/check_protocol_conformance.py --protocol ToolProvider
    python scripts/check_protocol_conformance.py --all-verticals

Exit Codes:
    0: All checks passed
    1: Violations found
    2: Error occurred
"""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, get_type_hints

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class Severity(Enum):
    """Violation severity levels."""

    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class Violation:
    """Represents a protocol conformance violation."""

    severity: Severity
    component: str
    protocol: str
    message: str
    suggestion: Optional[str] = None
    line_number: Optional[int] = None

    def __str__(self) -> str:
        """Format violation for display."""
        location = f"{self.component}:{self.line_number}" if self.line_number else self.component
        return f"[{self.severity.value}] {location} - {self.message}"


@dataclass
class ComplianceReport:
    """Protocol compliance report for a component."""

    component_name: str
    protocol_name: str
    is_compliant: bool
    violations: List[Violation] = field(default_factory=list)
    missing_methods: List[str] = field(default_factory=list)
    signature_mismatches: List[str] = field(default_factory=list)
    extra_methods: List[str] = field(default_factory=list)

    def add_violation(self, violation: Violation) -> None:
        """Add a violation to the report."""
        self.violations.append(violation)

    @property
    def error_count(self) -> int:
        """Count of ERROR severity violations."""
        return sum(1 for v in self.violations if v.severity == Severity.ERROR)

    @property
    def warning_count(self) -> int:
        """Count of WARNING severity violations."""
        return sum(1 for v in self.violations if v.severity == Severity.WARNING)


class ProtocolConformanceChecker:
    """Checks protocol conformance for Victor components."""

    def __init__(self, verbose: bool = False):
        """Initialize checker.

        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose
        self.reports: List[ComplianceReport] = []

    def check_protocol_conformance(
        self,
        component_class: Type,
        protocol_class: Type,
    ) -> ComplianceReport:
        """Check if a component correctly implements a protocol.

        Args:
            component_class: The component class to check
            protocol_class: The protocol class to check against

        Returns:
            ComplianceReport with findings
        """
        report = ComplianceReport(
            component_name=component_class.__name__,
            protocol_name=protocol_class.__name__,
            is_compliant=True,
        )

        # Get protocol members
        protocol_members = self._get_protocol_members(protocol_class)
        component_members = self._get_component_members(component_class)

        # Check for missing methods
        missing = self._check_missing_methods(component_members, protocol_members)
        if missing:
            report.missing_methods = missing
            for method in missing:
                report.add_violation(
                    Violation(
                        severity=Severity.ERROR,
                        component=component_class.__name__,
                        protocol=protocol_class.__name__,
                        message=f"Missing required method: {method}",
                        suggestion=f"Add method: {self._get_method_signature(protocol_class, method)}",
                    )
                )
            report.is_compliant = False

        # Check signature mismatches
        mismatches = self._check_signature_mismatches(
            component_class, protocol_class, protocol_members
        )
        if mismatches:
            report.signature_mismatches = list(mismatches.keys())
            for method, details in mismatches.items():
                report.add_violation(
                    Violation(
                        severity=Severity.ERROR,
                        component=component_class.__name__,
                        protocol=protocol_class.__name__,
                        message=f"Signature mismatch for {method}: {details}",
                        suggestion=f"Ensure signature matches: {self._get_method_signature(protocol_class, method)}",
                    )
                )
            report.is_compliant = False

        # Check for extra methods (warnings only)
        extra = self._check_extra_methods(component_members, protocol_members)
        if extra:
            report.extra_methods = extra
            for method in extra:
                report.add_violation(
                    Violation(
                        severity=Severity.WARNING,
                        component=component_class.__name__,
                        protocol=protocol_class.__name__,
                        message=f"Extra method not in protocol: {method}",
                        suggestion="Consider if this method should be in the protocol",
                    )
                )

        self.reports.append(report)
        return report

    def _get_protocol_members(self, protocol_class: Type) -> Dict[str, inspect.Signature]:
        """Get protocol method signatures.

        Args:
            protocol_class: Protocol class to inspect

        Returns:
            Dict mapping method names to signatures
        """
        members = {}
        for name, member in inspect.getmembers(protocol_class):
            if name.startswith("_"):
                continue
            if callable(member) and not isinstance(member, (classmethod, staticmethod)):
                try:
                    sig = inspect.signature(member)
                    members[name] = sig
                except ValueError:
                    # Built-in methods without signatures
                    continue
        return members

    def _get_component_members(self, component_class: Type) -> Dict[str, inspect.Signature]:
        """Get component method signatures.

        Args:
            component_class: Component class to inspect

        Returns:
            Dict mapping method names to signatures
        """
        members = {}
        for name, member in inspect.getmembers(component_class):
            if name.startswith("_"):
                continue
            if callable(member) and not isinstance(member, (classmethod, staticmethod)):
                try:
                    sig = inspect.signature(member)
                    members[name] = sig
                except ValueError:
                    continue
        return members

    def _check_missing_methods(
        self,
        component_members: Dict[str, inspect.Signature],
        protocol_members: Dict[str, inspect.Signature],
    ) -> List[str]:
        """Check for missing required methods.

        Args:
            component_members: Component's methods
            protocol_members: Protocol's methods

        Returns:
            List of missing method names
        """
        missing = []
        for name in protocol_members:
            if name not in component_members:
                missing.append(name)
        return missing

    def _check_signature_mismatches(
        self,
        component_class: Type,
        protocol_class: Type,
        protocol_members: Dict[str, inspect.Signature],
    ) -> Dict[str, str]:
        """Check for method signature mismatches.

        Args:
            component_class: Component class
            protocol_class: Protocol class
            protocol_members: Protocol's methods

        Returns:
            Dict mapping method names to error messages
        """
        mismatches = {}
        for name, protocol_sig in protocol_members.items():
            if not hasattr(component_class, name):
                continue
            component_method = getattr(component_class, name)
            try:
                component_sig = inspect.signature(component_method)

                # Compare parameter counts
                protocol_params = list(protocol_sig.parameters.values())
                component_params = list(component_sig.parameters.values())

                # Skip 'self' comparison for non-protocol methods
                if protocol_params and protocol_params[0].name == "self":
                    protocol_params = protocol_params[1:]
                if component_params and component_params[0].name == "self":
                    component_params = component_params[1:]

                if len(protocol_params) != len(component_params):
                    mismatches[name] = (
                        f"Parameter count mismatch: "
                        f"expected {len(protocol_params)}, got {len(component_params)}"
                    )
                    continue

                # Compare parameter names (excluding *args, **kwargs)
                for pp, cp in zip(protocol_params, component_params):
                    if pp.kind != cp.kind:
                        mismatches[name] = (
                            f"Parameter kind mismatch for {pp.name}: "
                            f"expected {pp.kind}, got {cp.kind}"
                        )
                        break
            except ValueError:
                # Methods without signature
                continue

        return mismatches

    def _check_extra_methods(
        self,
        component_members: Dict[str, inspect.Signature],
        protocol_members: Dict[str, inspect.Signature],
    ) -> List[str]:
        """Check for extra methods not in protocol.

        Args:
            component_members: Component's methods
            protocol_members: Protocol's methods

        Returns:
            List of extra method names
        """
        extra = []
        for name in component_members:
            if name not in protocol_members:
                extra.append(name)
        return extra

    def _get_method_signature(self, cls: Type, method_name: str) -> str:
        """Get method signature as string.

        Args:
            cls: Class containing the method
            method_name: Name of the method

        Returns:
            Method signature as string
        """
        if not hasattr(cls, method_name):
            return f"{method_name}()"
        method = getattr(cls, method_name)
        try:
            sig = inspect.signature(method)
            return f"{method_name}{sig}"
        except ValueError:
            return f"{method_name}()"

    def check_vertical_protocol(
        self,
        vertical_path: Path,
        protocol_name: Optional[str] = None,
    ) -> List[ComplianceReport]:
        """Check a vertical's protocol conformance.

        Args:
            vertical_path: Path to vertical directory
            protocol_name: Optional specific protocol to check

        Returns:
            List of compliance reports
        """
        reports = []

        # Import vertical module
        vertical_name = vertical_path.name
        module_path = f"victor.{vertical_name}"

        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            print(f"Error importing {module_path}: {e}")
            return reports

        # Find vertical class (inherits from VerticalBase)
        from victor.core.verticals.base import VerticalBase

        vertical_class = None
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, VerticalBase) and obj != VerticalBase:
                vertical_class = obj
                break

        if not vertical_class:
            print(f"No VerticalBase subclass found in {vertical_path}")
            return reports

        # Check protocols
        if protocol_name:
            # Check specific protocol
            try:
                protocol = self._import_protocol(protocol_name)
                report = self.check_protocol_conformance(vertical_class, protocol)
                reports.append(report)
            except ImportError as e:
                print(f"Error importing protocol {protocol_name}: {e}")
        else:
            # Auto-detect protocols from annotations
            # Check if vertical claims to implement certain protocols
            protocols_to_check = self._get_protocols_for_vertical(vertical_class)
            for protocol in protocols_to_check:
                report = self.check_protocol_conformance(vertical_class, protocol)
                reports.append(report)

        return reports

    def _import_protocol(self, protocol_name: str) -> Type:
        """Import a protocol class by name.

        Args:
            protocol_name: Protocol name (can include module path)

        Returns:
            Protocol class

        Raises:
            ImportError: If protocol cannot be imported
        """
        # Try direct import
        try:
            module_path, class_name = protocol_name.rsplit(".", 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ValueError, ImportError, AttributeError):
            # Try from victor.protocols
            try:
                from victor import protocols

                return getattr(protocols, protocol_name)
            except AttributeError:
                raise ImportError(f"Cannot import protocol: {protocol_name}")

    def _get_protocols_for_vertical(self, vertical_class: Type) -> List[Type]:
        """Get protocols that a vertical should implement.

        Args:
            vertical_class: Vertical class

        Returns:
            List of protocol classes
        """
        # Common protocols for verticals
        from victor.protocols.agent_tools import ToolRegistryProtocol
        from victor.protocols.agent_conversation import ConversationControllerProtocol

        protocols = []

        # Check if vertical has methods suggesting ToolRegistryProtocol
        if hasattr(vertical_class, "get_tools"):
            protocols.append(ToolRegistryProtocol)

        # Check if vertical has conversation methods
        if hasattr(vertical_class, "get_system_prompt"):
            protocols.append(ConversationControllerProtocol)

        return protocols

    def print_report(self, report: ComplianceReport) -> None:
        """Print a compliance report.

        Args:
            report: Report to print
        """
        status = "✓ COMPLIANT" if report.is_compliant else "✗ NON-COMPLIANT"
        print(f"\n{status}: {report.component_name} -> {report.protocol_name}")
        print("=" * 80)

        if report.missing_methods:
            print(f"\nMissing Methods ({len(report.missing_methods)}):")
            for method in report.missing_methods:
                print(f"  - {method}")

        if report.signature_mismatches:
            print(f"\nSignature Mismatches ({len(report.signature_mismatches)}):")
            for method in report.signature_mismatches:
                print(f"  - {method}")

        if report.extra_methods:
            print(f"\nExtra Methods ({len(report.extra_methods)}):")
            for method in report.extra_methods:
                print(f"  - {method}")

        if self.verbose or not report.is_compliant:
            print("\nDetailed Violations:")
            for violation in report.violations:
                print(f"  {violation}")
                if violation.suggestion:
                    print(f"    Suggestion: {violation.suggestion}")

    def print_summary(self) -> None:
        """Print summary of all reports."""
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        total = len(self.reports)
        compliant = sum(1 for r in self.reports if r.is_compliant)
        non_compliant = total - compliant

        total_errors = sum(r.error_count for r in self.reports)
        total_warnings = sum(r.warning_count for r in self.reports)

        print(f"Total Checks: {total}")
        print(f"Compliant: {compliant}")
        print(f"Non-Compliant: {non_compliant}")
        print(f"Total Errors: {total_errors}")
        print(f"Total Warnings: {total_warnings}")

        if non_compliant > 0:
            print("\nNon-Compliant Components:")
            for report in self.reports:
                if not report.is_compliant:
                    print(
                        f"  - {report.component_name} ({report.error_count} errors, {report.warning_count} warnings)"
                    )


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Check protocol conformance for Victor components")
    parser.add_argument(
        "--vertical",
        type=Path,
        help="Path to vertical directory (e.g., victor/coding)",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        help="Protocol name to check (e.g., ToolProvider)",
    )
    parser.add_argument(
        "--all-verticals",
        action="store_true",
        help="Check all verticals in victor/",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    args = parser.parse_args()

    checker = ProtocolConformanceChecker(verbose=args.verbose)

    # Determine what to check
    if args.all_verticals:
        # Check all verticals
        victor_dir = Path("victor")
        vertical_dirs = [
            d for d in victor_dir.iterdir() if d.is_dir() and not d.name.startswith("_")
        ]

        for vertical_dir in vertical_dirs:
            # Skip non-vertical directories
            if not (vertical_dir / "assistant.py").exists():
                continue
            reports = checker.check_vertical_protocol(vertical_dir)
            for report in reports:
                checker.print_report(report)

    elif args.vertical:
        # Check specific vertical
        if not args.vertical.exists():
            print(f"Error: Vertical path does not exist: {args.vertical}")
            return 2

        reports = checker.check_vertical_protocol(args.vertical, args.protocol)
        for report in reports:
            checker.print_report(report)

    else:
        parser.print_help()
        return 2

    checker.print_summary()

    # Exit code based on compliance
    if any(not r.is_compliant for r in checker.reports):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
