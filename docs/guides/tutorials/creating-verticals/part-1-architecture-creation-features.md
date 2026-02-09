# Creating Verticals - Part 1

**Part 1 of 3:** Architecture, Basic Creation, and Advanced Features

---

## Navigation

- **[Part 1: Architecture & Features](#)** (Current)
- [Part 2: Security & Configuration](part-2-security-config.md)
- [Part 3: Testing & Best Practices](part-3-testing-best-practices.md)
- [**Complete Guide**](../CREATING_VERTICALS.md)

---
# Creating Custom Verticals for Victor AI

This comprehensive guide teaches you how to create custom verticals (domains) for Victor AI.

## Table of Contents

1. [Introduction](#introduction)
2. [Vertical Architecture](#vertical-architecture)
3. [Basic Vertical Creation](#basic-vertical-creation)
4. [Advanced Vertical Features](#advanced-vertical-features)
5. [Vertical Configuration](#vertical-configuration)
6. [Testing Verticals](#testing-verticals)
7. [Publishing Verticals](#publishing-verticals)
8. [Examples](#examples)

## Introduction

### What are Verticals?

Verticals are domain-specific specializations of Victor AI. Each vertical provides:
- Specialized tools for a domain
- Custom system prompts
- Domain-specific workflows
- Unique modes and capabilities

### Built-in Verticals

- **Coding**: Code analysis, refactoring, testing
- **DevOps**: CI/CD, Docker, Kubernetes
- **RAG**: Document ingestion, vector search
- **DataAnalysis**: Pandas, visualization, statistics
- **Research**: Web search, citations, synthesis
- **Benchmark**: SWE-bench, evaluation harnesses

### Why Create Custom Verticals?

- **Domain specialization**: Customize for specific industries
- **Proprietary tools**: Integrate internal tooling
- **Custom workflows**: Automate domain-specific tasks
- **Company standards**: Enforce coding standards and practices

## Vertical Architecture

### Vertical Base Class

All verticals inherit from `VerticalBase`:

```python
from victor.core.verticals.base import VerticalBase
from typing import List, Dict, Any

class MyVertical(VerticalBase):
    """Custom vertical for my domain."""

    name = "my_vertical"  # Unique vertical identifier
    description = "Description of what this vertical does"
    version = "0.5.0"

    def get_tools(self) -> List[Any]:
        """Return list of tools provided by this vertical."""
        return []

    def get_system_prompt(self) -> str:
        """Return system prompt for this vertical."""
        return "You are a specialized assistant for..."
```

### Vertical Directory Structure

```
my_vertical/
├── __init__.py              # Vertical export
├── vertical.py              # Main vertical class
├── tools/                   # Vertical-specific tools
│   ├── __init__.py
│   ├── tool1.py
│   └── tool2.py
├── workflows/               # Vertical-specific workflows
│   ├── __init__.py
│   ├── workflow1.yaml
│   └── workflow2.yaml
├── prompts/                 # Prompt templates
│   ├── __init__.py
│   └── templates.py
├── config/                  # Configuration files
│   ├── modes.yaml
│   ├── capabilities.yaml
│   └── teams.yaml
└── tests/                   # Tests
    ├── test_vertical.py
    └── test_tools.py
```

## Basic Vertical Creation

### Step 1: Create Vertical Class

```python
# my_vertical/vertical.py

from victor.core.verticals.base import VerticalBase
from victor.tools.base import BaseTool
from typing import List

class SecurityVertical(VerticalBase):
    """Security analysis and auditing vertical."""

    name = "security"
    description = "Security-focused code analysis and vulnerability detection"
    version = "0.5.0"

    def get_tools(self) -> List[type[BaseTool]]:
        """Import and return security tools."""
        from my_vertical.tools import (
            VulnerabilityScanner,
            SecurityAuditTool,
            ComplianceChecker
        )
        return [
            VulnerabilityScanner,
            SecurityAuditTool,
            ComplianceChecker
        ]

    def get_system_prompt(self) -> str:
        """Return security-focused system prompt."""
        return """You are a security expert AI assistant specializing in:
- Identifying security vulnerabilities
- Recommending secure coding practices
- Performing security audits
- Ensuring compliance with security standards

Always prioritize security in your analysis and recommendations."""
```

### Step 2: Create Vertical Tools

```python
# my_vertical/tools/vulnerability_scanner.py

from victor.tools.base import BaseTool, CostTier
from typing import Dict, Any, List

class VulnerabilityScanner(BaseTool):
    """Scan code for security vulnerabilities."""

    name = "vulnerability_scanner"
    description = "Scan code for common security vulnerabilities including SQL injection, XSS, and more"
    cost_tier = CostTier.MEDIUM
    category = "security"

    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Code to scan for vulnerabilities"
            },
            "language": {
                "type": "string",
                "description": "Programming language"
            }
        },
        "required": ["code"]
    }

    async def execute(
        self,
        code: str,
        language: str = "python"
    ) -> Dict[str, Any]:
        """Scan code for vulnerabilities."""
        vulnerabilities = []

        # SQL Injection detection
        if "execute(" in code or "exec(" in code:
            vulnerabilities.append({
                "type": "SQL Injection",
                "severity": "HIGH",
                "description": "Potential SQL injection vulnerability",
                "line": self._find_line_number(code, "execute")
            })

        # XSS detection
        if language in ["javascript", "typescript"]:
            if "innerHTML" in code or "document.write" in code:
                vulnerabilities.append({
                    "type": "XSS",
                    "severity": "HIGH",
                    "description": "Potential XSS vulnerability",
                    "line": self._find_line_number(code, "innerHTML")
                })

        # Hardcoded secrets detection
        import re
        secret_patterns = {
            r"api[_-]?key\s*=\s*['\"][^'\"]+['\"]": "API Key",
            r"password\s*=\s*['\"][^'\"]+['\"]": "Hardcoded Password",
            r"secret\s*=\s*['\"][^'\"]+['\"]": "Hardcoded Secret"
        }

        for pattern, vuln_type in secret_patterns.items():
            if re.search(pattern, code, re.IGNORECASE):
                vulnerabilities.append({
                    "type": vuln_type,
                    "severity": "CRITICAL",
                    "description": f"Hardcoded {vuln_type} detected",
                    "recommendation": "Use environment variables or secret management"
                })

        return {
            "success": True,
            "vulnerabilities": vulnerabilities,
            "summary": {
                "total": len(vulnerabilities),
                "critical": len([v for v in vulnerabilities if v["severity"] == "CRITICAL"]),
                "high": len([v for v in vulnerabilities if v["severity"] == "HIGH"]),
                "medium": len([v for v in vulnerabilities if v["severity"] == "MEDIUM"]),
                "low": len([v for v in vulnerabilities if v["severity"] == "LOW"])
            }
        }

    def _find_line_number(self, code: str, pattern: str) -> int:
        """Find line number of pattern in code."""
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            if pattern in line:
                return i
        return 0
```

### Step 3: Create Vertical Configuration

```yaml
# my_vertical/config/modes.yaml

vertical_name: security
default_mode: audit

modes:
  audit:
    name: audit
    display_name: Security Audit
    exploration: thorough
    edit_permission: sandbox
    tool_budget_multiplier: 2.0
    max_iterations: 15
    description: Comprehensive security audit mode

  scan:
    name: scan
    display_name: Vulnerability Scan
    exploration: standard
    edit_permission: none
    tool_budget_multiplier: 1.0
    max_iterations: 5
    description: Quick vulnerability scanning mode

  compliance:
    name: compliance
    display_name: Compliance Check
    exploration: thorough
    edit_permission: none
    tool_budget_multiplier: 1.5
    max_iterations: 10
    description: Check compliance with security standards
```

```yaml
# my_vertical/config/capabilities.yaml

vertical_name: security

capabilities:
  vulnerability_scan:
    type: tool
    description: "Scan code for security vulnerabilities"
    enabled: true

  security_audit:
    type: workflow
    description: "Perform comprehensive security audit"
    enabled: true
    handler: "my_vertical.workflows:SecurityAuditWorkflow"

  compliance_check:
    type: validator
    description: "Check compliance with security standards"
    enabled: true
    config:
      standards:
        - OWASP
        - PCI-DSS
        - SOC2

  threat_modeling:
    type: workflow
    description: "Generate threat model for application"
    enabled: true

handlers:
  audit_report: my_vertical.reporting:SecurityAuditReportGenerator
  compliance_validator: my_vertical.validators:ComplianceValidator
```

```yaml
# my_vertical/config/teams.yaml

teams:
  - name: security_review_team
    display_name: "Security Review Team"
    description: "Multi-agent team for comprehensive security review"
    formation: parallel
    communication_style: structured
    max_iterations: 5
    roles:
      - name: vulnerability_scanner
        display_name: "Vulnerability Scanner"
        description: "Focuses on finding security vulnerabilities"
        persona: "You are a security expert specializing in vulnerability detection..."
        tool_categories: [security, analysis]
        capabilities: [vulnerability_scan, static_analysis]

      - name: compliance_auditor
        display_name: "Compliance Auditor"
        description: "Ensures compliance with security standards"
        persona: "You are a compliance expert specializing in security standards..."
        tool_categories: [analysis, validation]
        capabilities: [compliance_check, standards_validation]

      - name: threat_analyst
        display_name: "Threat Analyst"
        description: "Analyzes potential threats and attack vectors"
        persona: "You are a threat modeling expert..."
        tool_categories: [analysis, security]
        capabilities: [threat_modeling, risk_assessment]
```

### Step 4: Export Vertical

```python
# my_vertical/__init__.py

from my_vertical.vertical import SecurityVertical

__all__ = ["SecurityVertical"]
```

### Step 5: Register Vertical

```toml
# pyproject.toml

[project]
name = "victor-security"
version = "0.5.0"

[project.entry-points."victor.verticals"]
security = "my_vertical:SecurityVertical"
```

## Advanced Vertical Features

### Feature 1: Custom Workflows

```yaml
# my_vertical/workflows/security_audit.yaml

workflows:
  security_audit:
    description: "Comprehensive security audit workflow"

    nodes:
      - id: scan_vulnerabilities
        type: agent
        role: vulnerability_scanner
        goal: "Scan codebase for security vulnerabilities"
        tool_budget: 10
        next: [analyze_dependencies]

      - id: analyze_dependencies
        type: agent
        role: dependency_analyzer
        goal: "Analyze dependencies for known vulnerabilities"
        tool_budget: 5
        next: [check_compliance]

      - id: check_compliance
        type: agent
        role: compliance_auditor
        goal: "Check compliance with security standards"
        tool_budget: 5
        next: [generate_report]

      - id: generate_report
        type: compute
        handler: generate_audit_report
        inputs:
          vulnerabilities: $ctx.scan_vulnerabilities.vulnerabilities
          dependencies: $ctx.analyze_dependencies.risks
          compliance: $ctx.check_compliance.results
        next: [end]

      - id: end
        type: end
```

### Feature 2: Custom Modes with Complexity Mapping

```python
# my_vertical/mode_config.py

from victor.core.mode_config import RegistryBasedModeConfigProvider

class SecurityModeConfigProvider(RegistryBasedModeConfigProvider):
    """Security vertical mode configuration with complexity mapping."""

    def __init__(self):
        super().__init__(vertical="security")

    def get_mode_for_complexity(self, complexity: str) -> str:
        """Map task complexity to appropriate mode."""
        mapping = {
            "simple": "scan",        # Quick scan
            "moderate": "audit",     # Full audit
            "complex": "compliance"  # Full compliance check
        }
        return mapping.get(complexity, "audit")

    def get_tool_budget_for_mode(self, mode: str) -> int:
        """Get tool budget for mode."""
        budgets = {
            "scan": 10,
            "audit": 25,
            "compliance": 40
        }
        return budgets.get(mode, 20)
```

### Feature 3: Prompt Templates

```python
# my_vertical/prompts/templates.py

from victor.framework.prompt_builder import PromptSection

class SecurityPromptSection(PromptSection):
    """Security-focused prompt section."""

    def render(self, context: dict) -> str:
        return f"""
## Security Guidelines

---

## See Also

- [Documentation Home](../../README.md)


**Reading Time:** 6 min
**Last Updated:** February 08, 2026**
