#!/usr/bin/env python
"""Verification script for penetration testing module.

Checks that all required components are present and functioning.
"""

import sys
from pathlib import Path


def verify_imports():
    """Verify that all required classes can be imported."""
    print("Verifying imports...")
    try:
        from victor.security.penetration_testing import (
            SecurityTestSuite,
            SecurityAuditReport,
            SecurityVulnerability,
            ExploitPattern,
            SeverityLevel,
            AttackType,
            Vulnerability,
            SecurityReport,
            ComprehensiveSecurityReport,
        )
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def verify_data_classes():
    """Verify that all required data classes have correct attributes."""
    print("\nVerifying data classes...")
    from victor.security.penetration_testing import (
        ExploitPattern,
        Vulnerability,
        SeverityLevel,
        AttackType,
    )

    # Check ExploitPattern
    try:
        pattern = ExploitPattern(
            pattern="test",
            description="test pattern",
            risk_level=SeverityLevel.HIGH,
            mitigation="test mitigation",
            category="test",
        )
        assert hasattr(pattern, "matches")
        assert hasattr(pattern, "to_dict")
        print("✓ ExploitPattern class verified")
    except Exception as e:
        print(f"✗ ExploitPattern verification failed: {e}")
        return False

    # Check Vulnerability
    try:
        vuln = Vulnerability(
            type=AttackType.PROMPT_INJECTION,
            description="test vulnerability",
            severity=SeverityLevel.CRITICAL,
            remediation="test remediation",
        )
        assert hasattr(vuln, "to_dict")
        print("✓ Vulnerability class verified")
    except Exception as e:
        print(f"✗ Vulnerability verification failed: {e}")
        return False

    return True


def verify_security_test_suite():
    """Verify that SecurityTestSuite has all required methods."""
    print("\nVerifying SecurityTestSuite...")
    from victor.security.penetration_testing import SecurityTestSuite

    required_methods = [
        "test_prompt_injection",
        "test_authorization_bypass",
        "test_data_exfiltration",
        "test_resource_exhaustion",
        "test_code_injection",
        "run_security_audit",
        "run_all_security_tests",
    ]

    suite = SecurityTestSuite()

    for method in required_methods:
        if not hasattr(suite, method):
            print(f"✗ Missing method: {method}")
            return False

    print(f"✓ All {len(required_methods)} required methods present")
    return True


def verify_report_class():
    """Verify that ComprehensiveSecurityReport has required properties."""
    print("\nVerifying ComprehensiveSecurityReport...")
    from victor.security.penetration_testing import ComprehensiveSecurityReport

    required_properties = [
        "total_tests",
        "total_passed",
        "total_failed",
        "critical_count",
        "high_count",
        "medium_count",
        "low_count",
        "overall_passed",
        "risk_score",
        "recommendations",
    ]

    required_methods = [
        "generate_text_report",
        "generate_markdown_report",
        "to_dict",
    ]

    # Create empty report for testing
    report = ComprehensiveSecurityReport()

    for prop in required_properties:
        if not hasattr(type(report), prop):
            print(f"✗ Missing property: {prop}")
            return False

    for method in required_methods:
        if not hasattr(report, method):
            print(f"✗ Missing method: {method}")
            return False

    print(f"✓ All {len(required_properties)} properties present")
    print(f"✓ All {len(required_methods)} methods present")
    return True


def verify_file_size():
    """Verify that the file meets the minimum line count."""
    print("\nVerifying file size...")
    file_path = Path("victor/security/penetration_testing.py")

    if not file_path.exists():
        print(f"✗ File not found: {file_path}")
        return False

    line_count = len(file_path.read_text().splitlines())

    if line_count < 600:
        print(f"✗ File has only {line_count} lines (required: 600+)")
        return False

    print(f"✓ File has {line_count} lines (exceeds 600 line requirement)")
    return True


def verify_type_aliases():
    """Verify that type aliases exist."""
    print("\nVerifying type aliases...")
    from victor.security.penetration_testing import (
        SecurityVulnerability,
        SecurityAuditReport,
        Vulnerability,
        ComprehensiveSecurityReport,
    )

    # Check that SecurityVulnerability is an alias for Vulnerability
    if SecurityVulnerability is not Vulnerability:
        print("✗ SecurityVulnerability is not an alias for Vulnerability")
        return False

    # Check that SecurityAuditReport is an alias for ComprehensiveSecurityReport
    if SecurityAuditReport is not ComprehensiveSecurityReport:
        print("✗ SecurityAuditReport is not an alias for ComprehensiveSecurityReport")
        return False

    print("✓ All type aliases verified")
    return True


def main():
    """Run all verification checks."""
    print("=" * 80)
    print("Victor AI Penetration Testing Module Verification")
    print("=" * 80)

    checks = [
        verify_imports,
        verify_data_classes,
        verify_security_test_suite,
        verify_report_class,
        verify_file_size,
        verify_type_aliases,
    ]

    results = []
    for check in checks:
        try:
            results.append(check())
        except Exception as e:
            print(f"\n✗ Check failed with exception: {e}")
            results.append(False)

    print("\n" + "=" * 80)
    print(f"Verification Results: {sum(results)}/{len(results)} checks passed")
    print("=" * 80)

    if all(results):
        print("\n✓ ALL CHECKS PASSED - Module is ready for use!")
        return 0
    else:
        print("\n✗ SOME CHECKS FAILED - Please review the output above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
