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

"""Security Prompt Contributor - Task hints and prompt extensions for security analysis.

This module demonstrates how to implement PromptContributorProtocol for an external
vertical. Prompt contributors provide:

1. Task Type Hints: Guidance for specific types of tasks (vulnerability scan, code review)
2. System Prompt Sections: Additional context appended to the base system prompt
3. Grounding Rules: Rules for how the AI should base its responses on evidence
4. Priority: Order in which prompt sections are assembled

These contributions help the AI understand how to approach security-specific tasks
while maintaining consistency with the Victor framework.
"""

from typing import Dict, Optional

from victor.core.verticals.protocols import PromptContributorProtocol
from victor.core.vertical_types import TaskTypeHint


# Security-specific task type hints
SECURITY_TASK_TYPE_HINTS: Dict[str, TaskTypeHint] = {
    "vulnerability_scan": TaskTypeHint(
        task_type="vulnerability_scan",
        hint="""[SECURITY SCAN] Comprehensive vulnerability assessment:
1. Run automated scanners (bandit, trivy, semgrep) first
2. Analyze scanner output for high-severity issues
3. Manual review of critical code paths (auth, crypto, input handling)
4. Check OWASP Top 10 vulnerability categories
5. Document findings with severity ratings and remediation steps""",
        tool_budget=25,
        priority_tools=["bash", "grep", "read", "code_search"],
    ),
    "secret_detection": TaskTypeHint(
        task_type="secret_detection",
        hint="""[SECRET DETECTION] Find exposed credentials and secrets:
1. Use grep patterns for common secret formats (API keys, passwords, tokens)
2. Run gitleaks or similar tool for comprehensive detection
3. Check .env files, config files, and package manifests
4. Review git history for accidentally committed secrets
5. Classify by type and recommend rotation procedures""",
        tool_budget=15,
        priority_tools=["grep", "bash", "read"],
    ),
    "dependency_audit": TaskTypeHint(
        task_type="dependency_audit",
        hint="""[DEPENDENCY AUDIT] Check for vulnerable dependencies:
1. Identify package managers in use (npm, pip, maven, etc.)
2. Run appropriate audit tools (npm audit, safety check, trivy)
3. Analyze CVE severity and exploitability
4. Check for available patches or updates
5. Recommend upgrade path with breaking change analysis""",
        tool_budget=20,
        priority_tools=["bash", "read", "grep"],
    ),
    "code_review": TaskTypeHint(
        task_type="code_review",
        hint="""[SECURITY CODE REVIEW] Manual security analysis of source code:
1. Map the application's attack surface
2. Review authentication and authorization logic
3. Check input validation and output encoding
4. Analyze cryptographic implementations
5. Look for injection vulnerabilities (SQL, XSS, command)
6. Assess session management and state handling""",
        tool_budget=30,
        priority_tools=["read", "grep", "code_search", "overview"],
    ),
    "config_security": TaskTypeHint(
        task_type="config_security",
        hint="""[CONFIG SECURITY] Review security configurations:
1. Check web server configs (CORS, headers, TLS)
2. Review container/orchestration configs (Docker, K8s)
3. Analyze cloud configurations (IAM, network, storage)
4. Verify secrets management (no plaintext, proper rotation)
5. Check for security misconfigurations and defaults""",
        tool_budget=20,
        priority_tools=["read", "grep", "bash"],
    ),
    "compliance_check": TaskTypeHint(
        task_type="compliance_check",
        hint="""[COMPLIANCE CHECK] Verify compliance with security standards:
1. Identify applicable standards (SOC2, HIPAA, PCI-DSS, etc.)
2. Map controls to codebase components
3. Check logging and audit trail implementation
4. Verify encryption at rest and in transit
5. Review access control implementations
6. Document compliance gaps with remediation priority""",
        tool_budget=35,
        priority_tools=["read", "grep", "bash", "overview"],
    ),
    "incident_response": TaskTypeHint(
        task_type="incident_response",
        hint="""[INCIDENT RESPONSE] Analyze potential security incidents:
1. Gather evidence without modifying state
2. Check logs for indicators of compromise
3. Identify affected components and scope
4. Look for persistence mechanisms or backdoors
5. Document timeline of events
6. Recommend immediate containment actions""",
        tool_budget=25,
        priority_tools=["read", "grep", "bash"],
    ),
    # Default fallback for general security tasks
    "general": TaskTypeHint(
        task_type="general",
        hint="""[GENERAL SECURITY] For general security queries:
1. Understand the security concern or question
2. Analyze relevant code and configurations
3. Search for security patterns and anti-patterns
4. Provide findings with severity assessment
5. Recommend specific remediation steps
6. Reference relevant security standards (CWE, OWASP)""",
        tool_budget=15,
        priority_tools=["read", "grep", "bash", "overview"],
    ),
}


class SecurityPromptContributor(PromptContributorProtocol):
    """Contributes security-specific prompts and task hints.

    Implements PromptContributorProtocol to provide:
    - Task type hints for different security analysis scenarios
    - Additional system prompt content for security context
    - Grounding rules emphasizing evidence-based analysis

    Example:
        contributor = SecurityPromptContributor()

        # Get hints for vulnerability scanning
        hints = contributor.get_task_type_hints()
        vuln_hint = hints.get("vulnerability_scan")

        # Get additional system prompt section
        section = contributor.get_system_prompt_section()
    """

    def get_task_type_hints(self) -> Dict[str, TaskTypeHint]:
        """Return security-specific task type hints.

        Returns:
            Dict mapping task types to TaskTypeHint objects.
        """
        return SECURITY_TASK_TYPE_HINTS.copy()

    def get_system_prompt_section(self) -> str:
        """Return additional system prompt content for security context.

        This section is appended to the base system prompt and provides
        security-specific guidance and checklists.

        Returns:
            System prompt section for security analysis.
        """
        return """
## Security Analysis Checklist

Before finalizing any security assessment:
- [ ] All findings have been verified (not just scanner output)
- [ ] Severity ratings are justified and consistent
- [ ] Remediation steps are specific and actionable
- [ ] False positives have been identified and excluded
- [ ] CWE/CVE references are included where applicable
- [ ] Impact assessment considers the application context
- [ ] No sensitive data (credentials, PII) is included in output

## Severity Rating Guidelines

- **CRITICAL**: Remote code execution, auth bypass, data breach risk
- **HIGH**: Privilege escalation, significant data exposure, SQLi/XSS
- **MEDIUM**: Information disclosure, CSRF, insecure configuration
- **LOW**: Best practice violations, minor info leaks
- **INFO**: Observations, recommendations, potential improvements

## Evidence Requirements

For each finding:
1. **Location**: Exact file path and line numbers
2. **Evidence**: Code snippet or command output demonstrating the issue
3. **Reproduction**: Steps to verify the vulnerability exists
4. **Context**: Why this matters in the application's threat model
5. **Fix**: Specific code changes or configuration updates

## Responsible Disclosure

- Focus on detection, not exploitation
- Never exfiltrate discovered sensitive data
- Handle credentials as radioactive - report, don't use
- Consider the impact of any active testing
- Document everything for audit trails
""".strip()

    def get_grounding_rules(self) -> str:
        """Get security-specific grounding rules.

        Security analysis must be evidence-based. These rules ensure
        the AI grounds its findings in actual tool output and code.

        Returns:
            Grounding rules text for security analysis.
        """
        return """SECURITY GROUNDING: Base ALL security findings on actual evidence from tools.
- Quote exact code or command output demonstrating vulnerabilities
- Never assume vulnerability exists without verification
- Clearly distinguish between confirmed findings and potential issues
- Provide specific file paths and line numbers for all findings
- If unable to verify, state limitations clearly
- Cross-reference findings with multiple sources when possible""".strip()

    def get_priority(self) -> int:
        """Get priority for prompt section ordering.

        Security is a specialized vertical, so we use a moderate priority
        to ensure our prompt section appears in an appropriate position.

        Lower values appear first in the assembled prompt.

        Returns:
            Priority value (5 = relatively high priority for vertical content).
        """
        return 5

    def get_context_hints(self, task_type: Optional[str] = None) -> Optional[str]:
        """Return contextual hints based on detected task type.

        This method can be used to get specific hints when the task type
        is known from task classification.

        Args:
            task_type: Detected task type (e.g., "vulnerability_scan")

        Returns:
            Hint string for the task type, or None if not found.
        """
        if task_type and task_type in SECURITY_TASK_TYPE_HINTS:
            return SECURITY_TASK_TYPE_HINTS[task_type].hint
        return None

    def get_security_standards_reference(self) -> str:
        """Get reference information for common security standards.

        Provides quick reference for security standards commonly used
        in findings and compliance checks.

        Returns:
            Formatted string with security standards reference.
        """
        return """
## Security Standards Quick Reference

### OWASP Top 10 (2021)
- A01: Broken Access Control
- A02: Cryptographic Failures
- A03: Injection
- A04: Insecure Design
- A05: Security Misconfiguration
- A06: Vulnerable Components
- A07: Authentication Failures
- A08: Data Integrity Failures
- A09: Logging Failures
- A10: SSRF

### CWE (Common Weakness Enumeration)
Reference: https://cwe.mitre.org/
Format: CWE-XXX (e.g., CWE-89 for SQL Injection)

### CVE (Common Vulnerabilities and Exposures)
Reference: https://cve.mitre.org/
Format: CVE-YYYY-NNNNN (e.g., CVE-2024-12345)

### Common Compliance Frameworks
- SOC 2 Type II: Service organization controls
- PCI-DSS: Payment card data security
- HIPAA: Healthcare data protection
- GDPR: EU data privacy
- ISO 27001: Information security management
""".strip()
