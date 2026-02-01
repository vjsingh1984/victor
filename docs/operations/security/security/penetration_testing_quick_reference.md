# Penetration Testing Module - Quick Reference Guide

## Quick Start

```python
from victor.security.penetration_testing import SecurityTestSuite
from victor.agent.orchestrator import AgentOrchestrator
from pathlib import Path

# Create agent and test suite
agent = AgentOrchestrator()
suite = SecurityTestSuite(safe_mode=True)

# Run comprehensive security audit
report = await suite.run_security_audit(
    agent=agent,
    output_format="markdown",
    output_path=Path("security_report.md")
)

# Check results
print(f"Risk Score: {report.risk_score}/10.0")
print(f"Status: {'PASSED' if report.overall_passed else 'FAILED'}")
```

## Key Classes

### SecurityTestSuite
Main testing class with all security test methods.

**Initialization:**
```python
suite = SecurityTestSuite(
    max_test_duration_ms=5000.0,  # Max time per test
    safe_mode=True                # Prevent dangerous operations
)
```

**Methods:**
- `test_prompt_injection(agent)` - Test prompt injection attacks
- `test_authorization_bypass(agent)` - Test authorization bypass
- `test_data_exfiltration(agent)` - Test data exfiltration
- `test_resource_exhaustion(agent)` - Test DoS vulnerabilities
- `test_code_injection(agent)` - Test code injection
- `run_security_audit(agent, output_format, output_path)` - Full audit
- `run_all_security_tests(agent)` - Run all tests

### SecurityAuditReport
Comprehensive security audit results.

**Properties:**
```python
report.risk_score              # 0.0-10.0 risk score
report.total_tests             # Total tests run
report.total_passed            # Tests passed
report.total_failed            # Tests failed
report.critical_count          # Critical vulnerabilities
report.high_count              # High vulnerabilities
report.medium_count            # Medium vulnerabilities
report.low_count               # Low vulnerabilities
report.overall_passed          # True if no critical/high vulns
report.recommendations         # List of recommendations
```

**Methods:**
```python
report.generate_text_report()      # Human-readable text
report.generate_markdown_report()  # Markdown format
report.to_dict()                   # Dict for JSON serialization
```

### ExploitPattern
Custom exploit pattern definition.

```python
from victor.security.penetration_testing import ExploitPattern, SeverityLevel

pattern = ExploitPattern(
    pattern=r"admin('|\"|;)",           # Regex pattern
    description="Admin bypass attempt",  # Description
    risk_level=SeverityLevel.HIGH,       # Severity
    mitigation="Validate user input",    # Fix
    category="injection",                # Category
    cwe_id="CWE-89",                     # Optional CWE ID
    references=["https://owasp.org/"]    # Optional refs
)

# Test input
if pattern.matches(user_input):
    print("Attack detected!")
```

## Report Formats

### Text Report
```python
text_report = report.generate_text_report()
print(text_report)
```

### Markdown Report
```python
markdown_report = report.generate_markdown_report()
Path("report.md").write_text(markdown_report)
```

### JSON Report
```python
import json

json_report = json.dumps(report.to_dict(), indent=2)
Path("report.json").write_text(json_report)
```

## Individual Tests

Run specific security tests:

```python
# Prompt injection testing
prompt_report = await suite.test_prompt_injection(agent)
print(f"Vulnerabilities: {len(prompt_report.vulnerabilities_found)}")

# Authorization bypass testing
auth_report = await suite.test_authorization_bypass(agent)
print(f"Passed: {auth_report.passed}")

# Code injection testing
code_report = await suite.test_code_injection(agent)
for vuln in code_report.vulnerabilities_found:
    print(f"{vuln.severity.value}: {vuln.description}")
```

## Filtering Vulnerabilities

```python
# Filter by severity
from victor.security.penetration_testing import SeverityLevel

critical_vulns = [
    v for v in report.total_vulnerabilities
    if v.severity == SeverityLevel.CRITICAL
]

# Filter by type
from victor.security.penetration_testing import AttackType

injection_vulns = [
    v for v in report.total_vulnerabilities
    if v.type == AttackType.CODE_INJECTION
]
```

## Severity Levels

```python
from victor.security.penetration_testing import SeverityLevel

SeverityLevel.CRITICAL  # Exploitable, complete compromise
SeverityLevel.HIGH      # Exploitable, significant impact
SeverityLevel.MEDIUM   # Exploitable with conditions
SeverityLevel.LOW      # Hard to exploit, minimal impact
SeverityLevel.INFO     # Informational only
```

## Attack Types

```python
from victor.security.penetration_testing import AttackType

AttackType.PROMPT_INJECTION        # Prompt manipulation
AttackType.AUTHORIZATION_BYPASS    # Privilege escalation
AttackType.DATA_EXFILTRATION       # Data theft
AttackType.RESOURCE_EXHAUSTION     # DoS attacks
AttackType.MALICIOUS_FILE_UPLOAD   # File exploits
AttackType.CODE_INJECTION          # Code execution
AttackType.SESSION_HIJACKING       # Session theft
```

## Risk Score Calculation

```
Weighted scoring:
- CRITICAL: 10.0 points each
- HIGH: 7.0 points each
- MEDIUM: 4.0 points each
- LOW: 1.0 point each

Normalized to 0-10 scale:
risk_score = min(total_score / 5.0, 10.0)

Risk Levels:
- 8.0-10.0: CRITICAL
- 6.0-7.9: HIGH
- 4.0-5.9: MEDIUM
- 0.0-3.9: LOW
```

## Recommendations

Access prioritized recommendations:

```python
for i, rec in enumerate(report.recommendations, 1):
    print(f"{i}. {rec}")
```

Example output:
```
1. URGENT: Address 3 critical vulnerability(ies) immediately...
2. HIGH PRIORITY: Fix 5 high-severity issue(s) within 7 days...
3. Implement prompt injection defenses: Use system prompt...
4. Strengthen authorization: Implement role-based access control...
```

## Safe Mode

When `safe_mode=True` (default):
- Prevents actual file deletion
- Blocks dangerous network operations
- Sanitizes output for testing
- Logs all security checks

Set `safe_mode=False` only in controlled testing environments.

## Integration with CI/CD

```python
# In your test suite
async def test_security_compliance():
    from victor.security.penetration_testing import SecurityTestSuite

    suite = SecurityTestSuite()
    report = await suite.run_security_audit(agent)

    # Fail CI/CD if critical vulnerabilities found
    assert report.critical_count == 0, \
        f"Found {report.critical_count} critical vulnerabilities"

    # Fail if risk score too high
    assert report.risk_score < 8.0, \
        f"Risk score {report.risk_score} exceeds threshold"

    # Save report for artifacts
    Path("security-report.json").write_text(
        json.dumps(report.to_dict(), indent=2)
    )
```

## Logging

The module provides detailed logging:

```python
import logging

logging.basicConfig(level=logging.INFO)
# Logs show:
# - Test start/completion
# - Vulnerability detection
# - Risk score calculation
# - Report generation
```

## Type Aliases

For convenience and backward compatibility:

```python
from victor.security.penetration_testing import (
    SecurityVulnerability,  # Alias for Vulnerability
    SecurityAuditReport,     # Alias for ComprehensiveSecurityReport
)
```

## Best Practices

1. **Run Regularly**: Integrate into CI/CD pipeline
2. **Review Reports**: Check recommendations after each audit
3. **Fix Critical First**: Prioritize CRITICAL and HIGH vulnerabilities
4. **Track Trends**: Monitor risk scores over time
5. **Safe Testing**: Always use `safe_mode=True` in production
6. **Comprehensive Testing**: Use `run_security_audit()` for full coverage

## Troubleshooting

**Tests timeout?**
```python
suite = SecurityTestSuite(max_test_duration_ms=10000.0)
```

**Too many false positives?**
- Review vulnerability patterns
- Adjust indicators in check methods
- Customize exploit patterns

**Need custom tests?**
```python
# Extend SecurityTestSuite
class CustomTestSuite(SecurityTestSuite):
    async def test_custom_vulnerability(self, agent):
        # Your custom test logic
        pass
```

## Additional Resources

- Full documentation: `docs/security/penetration_testing_implementation.md`
- Usage examples: `victor/security/penetration_testing_example.py`
- Verification: `verify_penetration_testing.py`
- OWASP AI Top 10: https://owasp.org/www-project-top-ten-for-large-language-model-applications/
- CVSS Calculator: https://www.first.org/cvss/calculator/3.1

---

**Last Updated:** February 01, 2026
**Reading Time:** 1 min
