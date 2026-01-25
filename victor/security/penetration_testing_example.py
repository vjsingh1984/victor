#!/usr/bin/env python
"""Example usage of the penetration testing module.

This file demonstrates how to use the SecurityTestSuite to perform
comprehensive security audits on Victor AI agents.
"""

from pathlib import Path
from victor.security.penetration_testing import (
    SecurityTestSuite,
    SecurityAuditReport,
    SecurityVulnerability,
    ExploitPattern,
    SeverityLevel,
    AttackType,
)


async def example_basic_usage() -> None:
    """Basic usage example."""
    from victor.agent.orchestrator import AgentOrchestrator

    # Create the agent to test
    agent = AgentOrchestrator(
        settings=None,  # type: ignore[arg-type]
        provider=None,  # type: ignore[arg-type]
        model=None,  # type: ignore[arg-type]
    )

    # Create security test suite
    suite = SecurityTestSuite(
        max_test_duration_ms=5000.0,
        safe_mode=True,  # Prevents dangerous operations
    )

    # Run comprehensive security audit
    report: SecurityAuditReport = await suite.run_security_audit(
        agent=agent,
        output_format="markdown",
        output_path=Path("security_audit_report.md"),
    )

    # Check results
    print(f"Risk Score: {report.risk_score}/10.0")
    print(f"Tests Passed: {report.total_passed}/{report.total_tests}")
    print(f"Vulnerabilities Found: {len(report.total_vulnerabilities)}")

    # Display recommendations
    print("\nRecommendations:")
    for i, rec in enumerate(report.recommendations, 1):
        print(f"{i}. {rec}")


async def example_individual_tests() -> None:
    """Run individual security tests."""
    from victor.agent.orchestrator import AgentOrchestrator

    agent = AgentOrchestrator(
        settings=None,  # type: ignore[arg-type]
        provider=None,  # type: ignore[arg-type]
        model=None,  # type: ignore[arg-type]
    )
    suite = SecurityTestSuite()

    # Test specific vulnerabilities
    prompt_injection_report = await suite.test_prompt_injection(agent)
    print(f"Prompt Injection Tests: {prompt_injection_report.passed}")

    auth_bypass_report = await suite.test_authorization_bypass(agent)
    print(f"Authorization Bypass Tests: {auth_bypass_report.passed}")

    code_injection_report = await suite.test_code_injection(agent)
    print(f"Code Injection Tests: {code_injection_report.passed}")


async def example_custom_exploit_patterns() -> None:
    """Create custom exploit patterns for specific threats."""
    # Define custom exploit pattern
    sql_injection_pattern = ExploitPattern(
        pattern=r"('|\")|(\-\-)|(;)|(\bor\b|\band\b).*=",
        description="SQL Injection attack pattern",
        risk_level=SeverityLevel.CRITICAL,
        mitigation="Use parameterized queries and input validation",
        category="injection",
        cwe_id="CWE-89",
        references=[
            "https://owasp.org/www-community/attacks/SQL_Injection",
            "https://cwe.mitre.org/data/definitions/89.html",
        ],
    )

    # Test against user input
    user_input = "admin' OR '1'='1"

    if sql_injection_pattern.matches(user_input):
        print("⚠️  SQL Injection detected!")
        print(f"Mitigation: {sql_injection_pattern.mitigation}")


async def example_report_formats() -> None:
    """Generate reports in different formats."""
    from victor.agent.orchestrator import AgentOrchestrator

    agent = AgentOrchestrator(
        settings=None,  # type: ignore[arg-type]
        provider=None,  # type: ignore[arg-type]
        model=None,  # type: ignore[arg-type]
    )
    suite = SecurityTestSuite()

    # Run audit
    report = await suite.run_security_audit(agent)

    # Text report
    text_report = report.generate_text_report()
    print("Text Report:")
    print(text_report[:500] + "...")

    # Markdown report
    markdown_report = report.generate_markdown_report()
    Path("security_report.md").write_text(markdown_report)

    # JSON report
    import json

    json_report = json.dumps(report.to_dict(), indent=2)
    Path("security_report.json").write_text(json_report)


async def example_filter_by_severity() -> None:
    """Filter vulnerabilities by severity level."""
    from victor.agent.orchestrator import AgentOrchestrator

    agent = AgentOrchestrator(
        settings=None,  # type: ignore[arg-type]
        provider=None,  # type: ignore[arg-type]
        model=None,  # type: ignore[arg-type]
    )
    suite = SecurityTestSuite()

    # Run audit
    report = await suite.run_security_audit(agent)

    # Filter critical vulnerabilities
    critical_vulns = [
        v for v in report.total_vulnerabilities if v.severity == SeverityLevel.CRITICAL
    ]

    if critical_vulns:
        print(f"Found {len(critical_vulns)} CRITICAL vulnerabilities:")
        for vuln in critical_vulns:
            print(f"  - {vuln.description}")
            print(f"    Remediation: {vuln.remediation}")
    else:
        print("No critical vulnerabilities found!")


async def main() -> None:
    """Run all examples."""
    print("=" * 80)
    print("Victor AI Penetration Testing Examples")
    print("=" * 80)
    print()

    # Note: These examples require an actual AgentOrchestrator instance
    # Uncomment to run:

    # await example_basic_usage()
    # await example_individual_tests()
    await example_custom_exploit_patterns()
    # await example_report_formats()
    # await example_filter_by_severity()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
