"""Tests for Protocol Conformance Checker tool."""

from __future__ import annotations

import sys
from pathlib import Path


# Add scripts to path
scripts_dir = Path(__file__).parent.parent.parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))


def test_protocol_conformance_checker_imports():
    """Test that the checker can be imported."""
    import check_protocol_conformance

    assert check_protocol_conformance is not None


def test_severity_enum():
    """Test Severity enum values."""
    from check_protocol_conformance import Severity

    assert Severity.ERROR.value == "ERROR"
    assert Severity.WARNING.value == "WARNING"
    assert Severity.INFO.value == "INFO"


def test_violation_creation():
    """Test Violation dataclass creation."""
    from check_protocol_conformance import Violation, Severity

    violation = Violation(
        severity=Severity.ERROR,
        component="TestComponent",
        protocol="TestProtocol",
        message="Test message",
        suggestion="Test suggestion",
    )

    assert violation.severity == Severity.ERROR
    assert violation.component == "TestComponent"
    assert violation.protocol == "TestProtocol"
    assert violation.message == "Test message"
    assert violation.suggestion == "Test suggestion"
    assert "TestComponent" in str(violation)


def test_compliance_report_creation():
    """Test ComplianceReport dataclass creation."""
    from check_protocol_conformance import ComplianceReport, Violation, Severity

    report = ComplianceReport(
        component_name="TestComponent",
        protocol_name="TestProtocol",
        is_compliant=True,
    )

    assert report.component_name == "TestComponent"
    assert report.protocol_name == "TestProtocol"
    assert report.is_compliant
    assert report.error_count == 0
    assert report.warning_count == 0

    # Add violations
    error_violation = Violation(
        severity=Severity.ERROR,
        component="TestComponent",
        protocol="TestProtocol",
        message="Error message",
    )
    report.add_violation(error_violation)

    assert report.error_count == 1
    assert not report.is_compliant


def test_protocol_conformance_checker_initialization():
    """Test ProtocolConformanceChecker initialization."""
    from check_protocol_conformance import ProtocolConformanceChecker

    checker = ProtocolConformanceChecker(verbose=True)
    assert checker.verbose is True
    assert checker.reports == []


def test_get_protocol_members():
    """Test getting protocol members."""
    from check_protocol_conformance import ProtocolConformanceChecker

    # Create a test protocol
    class TestProtocol:
        def method_one(self, arg1: str) -> int:
            """Test method one."""
            return 0

        def _private_method(self):
            """Private method should be ignored."""
            pass

    checker = ProtocolConformanceChecker()
    members = checker._get_protocol_members(TestProtocol)

    assert "method_one" in members
    assert "_private_method" not in members


def test_check_missing_methods():
    """Test checking for missing methods."""
    from check_protocol_conformance import ProtocolConformanceChecker

    checker = ProtocolConformanceChecker()

    protocol_members = {"method_one": None, "method_two": None}
    component_members = {"method_one": None}

    missing = checker._check_missing_methods(component_members, protocol_members)

    assert "method_two" in missing
    assert "method_one" not in missing


def test_check_extra_methods():
    """Test checking for extra methods."""
    from check_protocol_conformance import ProtocolConformanceChecker

    checker = ProtocolConformanceChecker()

    protocol_members = {"method_one": None}
    component_members = {"method_one": None, "extra_method": None}

    extra = checker._check_extra_methods(component_members, protocol_members)

    assert "extra_method" in extra
    assert "method_one" not in extra
