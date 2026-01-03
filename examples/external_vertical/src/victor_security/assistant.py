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

"""Security Assistant - External vertical for security auditing and vulnerability analysis.

This module demonstrates how to create a complete VerticalBase subclass that can be
installed as an external package and discovered by Victor via entry points.

Key Implementation Points:
1. Inherit from VerticalBase
2. Set name, description, version class attributes
3. Implement required abstract methods: get_tools(), get_system_prompt()
4. Override optional methods for enhanced functionality:
   - get_stages(): Custom workflow stages
   - get_tiered_tool_config(): Tool tier configuration
   - get_safety_extension(): Security-specific safety patterns
   - get_prompt_contributor(): Task hints and prompt contributions
   - get_provider_hints(): Provider recommendations
   - get_evaluation_criteria(): Performance evaluation criteria
"""

from typing import Any

from victor.core.verticals import StageDefinition, VerticalBase
from victor.core.verticals.protocols import (
    PromptContributorProtocol,
    SafetyExtensionProtocol,
    TieredToolConfig,
)


class SecurityAssistant(VerticalBase):
    """Security assistant for vulnerability analysis, code security review, and auditing.

    This vertical specializes in:
    - Static code security analysis
    - Dependency vulnerability scanning
    - Configuration security review
    - Compliance checking (OWASP, CWE, etc.)
    - Security best practices guidance

    Example usage:
        # Get vertical configuration
        config = SecurityAssistant.get_config()

        # Get extensions for framework integration
        extensions = SecurityAssistant.get_extensions()

        # Create an agent with security vertical
        agent = await SecurityAssistant.create_agent(
            provider="anthropic",
            model="claude-sonnet-4-5-20250514",
        )

    Attributes:
        name: Unique identifier for the vertical ("security")
        description: Human-readable description
        version: Semantic version of the vertical
    """

    # Required class attributes
    name = "security"
    description = "Security auditing, vulnerability analysis, and compliance checking"
    version = "0.1.0"

    @classmethod
    def get_tools(cls) -> list[str]:
        """Get the list of tools for security tasks.

        Returns tools suitable for security analysis:
        - read: Read source code for security review
        - code_search: Search for security patterns (API keys, passwords, vulnerabilities)
        - shell: Run security scanning tools (trivy, bandit, snyk, etc.)
        - ls: Navigate project structure
        - overview: Understand codebase architecture

        For external verticals, you can also register custom tools via the
        "victor.tools" entry point in pyproject.toml.

        Returns:
            List of tool names to enable for this vertical.
        """
        # Using canonical tool names from Victor's tool registry
        # These tools are suitable for security analysis without
        # write access (security auditing should be read-only)
        return [
            # Core reading tools - essential for code review
            "read",  # Read source files
            "code_search",  # Search for patterns (secrets, vulnerabilities)
            "ls",  # Navigate directory structure
            "overview",  # Understand codebase architecture
            # Shell for running security scanners
            "shell",  # Run security tools (bandit, trivy, snyk, etc.)
            # Optional write tools for generating reports
            "write",  # Write security reports
        ]

    @classmethod
    def get_system_prompt(cls) -> str:
        """Get the system prompt for security tasks.

        The system prompt establishes the AI's security expert persona and
        provides guidance for conducting security analysis effectively.

        Returns:
            System prompt text with security domain expertise.
        """
        return """You are a security expert specialized in code security analysis, \
vulnerability detection, and security best practices.

## Your Primary Role

You are designed for SECURITY ANALYSIS. Your job is to:
- Review code for security vulnerabilities (OWASP Top 10, CWE, etc.)
- Identify hardcoded secrets, API keys, and credentials
- Analyze dependencies for known vulnerabilities
- Assess configuration security (Docker, Kubernetes, cloud)
- Provide actionable remediation guidance

IMPORTANT: You focus on security issues, not general code quality.
Prioritize security concerns over style or performance.

## Security Analysis Process

1. **Reconnaissance**: Understand the application architecture and technology stack
2. **Static Analysis**: Search for common vulnerability patterns in code
3. **Dependency Check**: Review package manifests for vulnerable dependencies
4. **Configuration Review**: Examine configs for security misconfigurations
5. **Secret Detection**: Scan for hardcoded secrets, keys, and tokens
6. **Report**: Provide clear findings with severity ratings and remediation steps

## Vulnerability Categories (OWASP Top 10 2021)

1. **A01 Broken Access Control**: Missing authorization checks, IDOR vulnerabilities
2. **A02 Cryptographic Failures**: Weak encryption, exposed secrets
3. **A03 Injection**: SQL, NoSQL, OS command, LDAP injection
4. **A04 Insecure Design**: Missing security controls in design
5. **A05 Security Misconfiguration**: Default configs, verbose errors
6. **A06 Vulnerable Components**: Outdated dependencies with CVEs
7. **A07 Auth Failures**: Weak passwords, session management flaws
8. **A08 Data Integrity**: Insecure deserialization, CI/CD vulnerabilities
9. **A09 Logging Failures**: Insufficient logging and monitoring
10. **A10 SSRF**: Server-Side Request Forgery

## Available Tools

- **read**: Read source code for security review
- **code_search**: Search for security patterns (passwords, API keys)
- **shell**: Run security scanners (bandit, trivy, snyk, etc.)
- **ls/overview**: Understand project structure and architecture
- **write**: Generate security reports

## Output Format

When reporting vulnerabilities:
1. **Title**: Brief description of the issue
2. **Severity**: CRITICAL / HIGH / MEDIUM / LOW / INFO
3. **Location**: File path and line number(s)
4. **Description**: What the vulnerability is and why it's dangerous
5. **Impact**: What an attacker could do by exploiting this
6. **Remediation**: Specific steps to fix the issue
7. **References**: CWE, CVE, or OWASP references if applicable

## Security Scanning Commands

Common security tools you can invoke via shell:
- `bandit -r .` - Python security linter
- `trivy fs .` - Vulnerability scanner for dependencies and containers
- `semgrep --config=auto .` - Multi-language static analysis
- `gitleaks detect` - Secret detection in git history
- `npm audit` / `yarn audit` - JavaScript dependency vulnerabilities
- `safety check` - Python dependency vulnerabilities

## Ethical Guidelines

- Only analyze code you have authorization to review
- Never attempt to exploit vulnerabilities
- Report findings responsibly
- Avoid actions that could cause harm to systems
- Do not exfiltrate sensitive data discovered during analysis

## Quality Standards

- Focus on actionable, exploitable vulnerabilities
- Minimize false positives - verify before reporting
- Prioritize by severity and exploitability
- Provide specific, implementable remediation steps
- Reference authoritative security resources
"""

    @classmethod
    def get_stages(cls) -> dict[str, StageDefinition]:
        """Get security-specific stage definitions.

        Security analysis follows a structured workflow:
        1. INITIAL: Understand the scope and technology stack
        2. RECONNAISSANCE: Map the application architecture
        3. ANALYSIS: Perform security scans and manual review
        4. VERIFICATION: Validate findings and reduce false positives
        5. REPORTING: Document findings with remediation guidance
        6. COMPLETION: Final summary and recommendations

        Returns:
            Dictionary mapping stage names to StageDefinition objects.
        """
        return {
            "INITIAL": StageDefinition(
                name="INITIAL",
                description="Understanding the security assessment scope",
                tools={"read", "ls", "overview"},
                keywords=["security", "audit", "review", "scan", "check", "vulnerabilities"],
                next_stages={"RECONNAISSANCE"},
            ),
            "RECONNAISSANCE": StageDefinition(
                name="RECONNAISSANCE",
                description="Mapping application architecture and technology stack",
                tools={"ls", "read", "overview", "code_search"},
                keywords=["architecture", "stack", "dependencies", "structure", "components"],
                next_stages={"ANALYSIS"},
            ),
            "ANALYSIS": StageDefinition(
                name="ANALYSIS",
                description="Performing security scans and code review",
                tools={"code_search", "shell", "read"},
                keywords=[
                    "scan",
                    "analyze",
                    "search",
                    "find",
                    "detect",
                    "vulnerability",
                    "secret",
                ],
                next_stages={"VERIFICATION", "ANALYSIS"},
            ),
            "VERIFICATION": StageDefinition(
                name="VERIFICATION",
                description="Validating findings and assessing severity",
                tools={"read", "code_search", "shell"},
                keywords=["verify", "confirm", "validate", "check", "assess", "severity"],
                next_stages={"REPORTING", "ANALYSIS"},
            ),
            "REPORTING": StageDefinition(
                name="REPORTING",
                description="Documenting findings with remediation guidance",
                tools={"write", "read"},
                keywords=["report", "document", "summarize", "recommend", "remediate"],
                next_stages={"COMPLETION", "VERIFICATION"},
            ),
            "COMPLETION": StageDefinition(
                name="COMPLETION",
                description="Security assessment complete",
                tools=set(),
                keywords=["done", "complete", "finished", "summary"],
                next_stages=set(),
            ),
        }

    @classmethod
    def get_tiered_tool_config(cls) -> TieredToolConfig | None:
        """Get tiered tool configuration for security analysis.

        Security analysis prioritizes read operations and specialized
        scanning tools, with limited write access (reports only).

        Tiers:
        - mandatory: Essential tools always available (read, ls, code_search)
        - vertical_core: Security-specific core tools (shell for scanners, code_search)
        - semantic_pool: Additional tools selected based on task context

        Returns:
            TieredToolConfig for security vertical
        """
        return TieredToolConfig(
            # Tier 1: Mandatory - always included for any task
            mandatory={
                "read",  # Read source files - essential
                "ls",  # List directory - essential
                "code_search",  # Pattern search - essential for security scanning
            },
            # Tier 2: Vertical Core - essential for security tasks
            vertical_core={
                "shell",  # Run security scanners (bandit, trivy, etc.)
                "code_search",  # Semantic search for security patterns
                "overview",  # Understand codebase architecture
            },
            # Security analysis should be primarily read-only
            # Hide write tools during analysis phases
            readonly_only_for_analysis=True,
        )

    @classmethod
    def get_provider_hints(cls) -> dict[str, Any]:
        """Get security-specific provider hints.

        Security analysis benefits from:
        - Large context windows (for analyzing multiple files)
        - Strong reasoning (for understanding vulnerability impact)
        - Tool calling support (for running security scanners)

        Returns:
            Dictionary with provider preferences.
        """
        return {
            "preferred_providers": ["anthropic", "openai", "google"],
            "min_context_window": 100000,  # Need to analyze multiple files
            "requires_tool_calling": True,  # Essential for running scanners
            "features": ["code_analysis", "reasoning"],
        }

    @classmethod
    def get_evaluation_criteria(cls) -> list[str]:
        """Get security-specific evaluation criteria.

        Criteria for evaluating the quality of security analysis:
        - accuracy: Correct identification of vulnerabilities
        - coverage: Comprehensive scanning across the codebase
        - severity_accuracy: Correct severity classification
        - false_positive_rate: Minimizing incorrect findings
        - remediation_quality: Actionable fix recommendations

        Returns:
            List of evaluation criteria descriptions.
        """
        return [
            "accuracy",  # Correct vulnerability identification
            "coverage",  # Comprehensive analysis
            "severity_accuracy",  # Correct severity ratings
            "false_positive_rate",  # Minimize false positives
            "remediation_quality",  # Actionable recommendations
            "reference_quality",  # Proper CWE/CVE references
        ]

    # =========================================================================
    # Extension Methods - Integrate with Victor Framework
    # =========================================================================

    @classmethod
    def get_safety_extension(cls) -> SafetyExtensionProtocol | None:
        """Get safety extension for security operations.

        Security analysis can involve running powerful scanning tools
        and examining sensitive code. The safety extension helps prevent
        accidental dangerous operations.

        Returns:
            SecuritySafetyExtension instance
        """
        from victor_security.safety import SecuritySafetyExtension

        return SecuritySafetyExtension()

    @classmethod
    def get_prompt_contributor(cls) -> PromptContributorProtocol | None:
        """Get prompt contributor for security task hints.

        Provides security-specific task type hints for different
        types of security analysis (vuln scan, secret detection, etc.)

        Returns:
            SecurityPromptContributor instance
        """
        from victor_security.prompts import SecurityPromptContributor

        return SecurityPromptContributor()

    @classmethod
    def get_mode_config(cls) -> dict[str, Any]:
        """Get security-specific mode configurations.

        Modes for different security analysis scenarios:
        - quick_scan: Fast high-level security check
        - thorough: Comprehensive security audit
        - compliance: Focus on compliance requirements

        Returns:
            Dictionary mapping mode names to configuration dicts.
        """
        return {
            "quick_scan": {
                "name": "quick_scan",
                "tool_budget": 15,
                "max_iterations": 20,
                "temperature": 0.3,  # Lower temperature for consistent analysis
                "description": "Quick security scan for high-severity issues",
            },
            "thorough": {
                "name": "thorough",
                "tool_budget": 50,
                "max_iterations": 50,
                "temperature": 0.5,
                "description": "Comprehensive security audit covering all vulnerability categories",
            },
            "compliance": {
                "name": "compliance",
                "tool_budget": 40,
                "max_iterations": 40,
                "temperature": 0.3,
                "description": "Compliance-focused audit (SOC2, HIPAA, PCI-DSS)",
            },
        }

    @classmethod
    def get_task_type_hints(cls) -> dict[str, Any]:
        """Get security-specific task type hints.

        Task hints help the agent understand how to approach
        different types of security analysis tasks.

        Returns:
            Dictionary mapping task types to hint configurations.
        """
        return {
            "vulnerability_scan": {
                "task_type": "vulnerability_scan",
                "hint": "[SECURITY SCAN] Run automated scans, manual review.",
                "tool_budget": 25,
                "priority_tools": ["shell", "code_search", "read"],
            },
            "secret_detection": {
                "task_type": "secret_detection",
                "hint": "[SECRET SCAN] Search for API keys, passwords, tokens.",
                "tool_budget": 15,
                "priority_tools": ["code_search", "shell", "read"],
            },
            "dependency_audit": {
                "task_type": "dependency_audit",
                "hint": "[DEPENDENCY AUDIT] Check package manifests for CVEs.",
                "tool_budget": 20,
                "priority_tools": ["shell", "read", "code_search"],
            },
            "code_review": {
                "task_type": "code_review",
                "hint": "[CODE REVIEW] Manual security review. Focus on auth.",
                "tool_budget": 30,
                "priority_tools": ["read", "code_search", "overview"],
            },
        }
