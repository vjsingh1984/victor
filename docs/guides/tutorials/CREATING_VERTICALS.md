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

You are a security expert. When analyzing code:

1. **Identify Vulnerabilities**: Look for:
   - SQL injection
   - Cross-site scripting (XSS)
   - Authentication/authorization issues
   - Hardcoded secrets
   - Insecure dependencies

2. **Assess Severity**: Rate findings as:
   - CRITICAL: Immediate action required
   - HIGH: Should be fixed soon
   - MEDIUM: Important to fix
   - LOW: Nice to have

3. **Provide Remediation**: For each issue, suggest:
   - Specific code fix
   - Explanation of the vulnerability
   - Prevention strategies

4. **Follow Standards**: Ensure compliance with:
   - OWASP Top 10
   - Secure coding practices
   - Industry standards

Current context: {context.get('task_description', 'N/A')}
"""

class CompliancePromptSection(PromptSection):
    """Compliance-focused prompt section."""

    def render(self, context: dict) -> str:
        standards = context.get("standards", ["OWASP"])
        return f"""
## Compliance Requirements

Ensure code complies with: {', '.join(standards)}

Check for:
- Input validation
- Output encoding
- Authentication/authorization
- Session management
- Cryptographic practices
- Error handling
- Logging
- Data protection

Report any compliance violations with specific references to the standard.
"""
```

### Feature 4: Event Handlers

```python
# my_vertical/handlers/audit_events.py

from victor.protocols import IEventHandler
from typing import Dict, Any

class SecurityAuditEventHandler(IEventHandler):
    """Handle security audit events."""

    async def on_tool_complete(self, event_data: Dict[str, Any]):
        """Handle tool completion event."""
        tool_name = event_data.get("tool_name")

        if tool_name == "vulnerability_scanner":
            vulnerabilities = event_data.get("result", {}).get("vulnerabilities", [])

            if vulnerabilities:
                await self._notify_security_team(vulnerabilities)
                await self._log_audit_finding(vulnerabilities)

    async def on_workflow_complete(self, event_data: Dict[str, Any]):
        """Handle workflow completion event."""
        workflow_name = event_data.get("workflow_name")

        if workflow_name == "security_audit":
            await self._generate_audit_report(event_data)
            await self._schedule_follow_up()

    async def _notify_security_team(self, vulnerabilities):
        """Notify security team of findings."""
        # Implementation
        pass

    async def _log_audit_finding(self, vulnerabilities):
        """Log audit findings."""
        # Implementation
        pass

    async def _generate_audit_report(self, event_data):
        """Generate audit report."""
        # Implementation
        pass

    async def _schedule_follow_up(self):
        """Schedule follow-up audit."""
        # Implementation
        pass
```

### Feature 5: Custom Validators

```python
# my_vertical/validators/compliance_validator.py

from victor.framework.validation import ValidationPipeline, Validator

class OWASPComplianceValidator(Validator):
    """Validate OWASP Top 10 compliance."""

    def __init__(self):
        super().__init__(name="owasp_compliance")

    async def validate(self, data: Dict[str, Any]) -> bool:
        """Validate OWASP compliance."""
        code = data.get("code", "")

        checks = {
            "A01:2021 – Broken Access Control": self._check_access_control(code),
            "A02:2021 – Cryptographic Failures": self._check_cryptography(code),
            "A03:2021 – Injection": self._check_injection(code),
            # ... more checks
        }

        return all(checks.values())

    def _check_access_control(self, code: str) -> bool:
        """Check for proper access control."""
        # Implementation
        return True

    def _check_cryptography(self, code: str) -> bool:
        """Check for proper cryptography."""
        # Implementation
        return True

    def _check_injection(self, code: str) -> bool:
        """Check for injection vulnerabilities."""
        # Implementation
        return True
```

## Vertical Configuration

### Configuration Provider

```python
# my_vertical/config.py

from victor.core.capabilities import CapabilityLoader
from victor.core.teams import BaseYAMLTeamProvider

class SecurityConfigurationProvider:
    """Centralized configuration provider for security vertical."""

    def __init__(self):
        self.capabilities = CapabilityLoader.from_vertical('security')
        self.teams = BaseYAMLTeamProvider('security')

    def get_capabilities(self):
        """Get all security capabilities."""
        return self.capabilities.load_capabilities('security')

    def get_capability(self, capability_name: str):
        """Get specific capability."""
        return self.capabilities.get_capability('security', capability_name)

    def get_teams(self):
        """Get all security teams."""
        return self.teams.load_teams()

    def get_team(self, team_name: str):
        """Get specific team."""
        return self.teams.get_team(team_name)
```

## Testing Verticals

### Unit Tests

```python
# tests/unit/test_security_vertical.py

import pytest
from my_vertical import SecurityVertical

@pytest.fixture
def security_vertical():
    """Create security vertical instance."""
    return SecurityVertical()

def test_vertical_properties(security_vertical):
    """Test vertical properties."""
    assert security_vertical.name == "security"
    assert security_vertical.version == "0.5.0"

def test_get_tools(security_vertical):
    """Test tool retrieval."""
    tools = security_vertical.get_tools()
    assert len(tools) > 0
    tool_names = [tool.name for tool in tools]
    assert "vulnerability_scanner" in tool_names

def test_system_prompt(security_vertical):
    """Test system prompt generation."""
    prompt = security_vertical.get_system_prompt()
    assert "security" in prompt.lower()
    assert "vulnerabilities" in prompt.lower()

def test_mode_config(security_vertical):
    """Test mode configuration."""
    mode = security_vertical.get_mode_config("audit")
    assert mode.name == "audit"
    assert mode.tool_budget_multiplier == 2.0
```

### Integration Tests

```python
# tests/integration/test_security_vertical_integration.py

import pytest
from victor import Agent

@pytest.mark.asyncio
async def test_security_audit_workflow():
    """Test security audit workflow."""
    agent = await Agent.create(
        vertical="security",
        mode="audit"
    )

    response = await agent.run(
        "Perform a security audit on this code: "
        "def login(username, password): "
        "  query = f'SELECT * FROM users WHERE username={username}'"
    )

    assert "vulnerability" in response.content.lower()
    assert "sql injection" in response.content.lower()

@pytest.mark.asyncio
async def test_vulnerability_scanning_tool():
    """Test vulnerability scanning tool."""
    agent = await Agent.create(vertical="security")

    response = await agent.run(
        "Scan this code for vulnerabilities: "
        "api_key = 'sk-1234567890'"
    )

    assert "hardcoded" in response.content.lower()
    assert "api key" in response.content.lower()
```

## Publishing Verticals

### Package Structure

```toml
# pyproject.toml

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "victor-security"
version = "0.5.0"
description = "Security analysis vertical for Victor AI"
authors = [{name = "Your Name", email = "your.email@example.com"}]
license = {text = "MIT"}
readme = "README.md"
requires-python = ">=3.10"

dependencies = [
    "victor-ai>=0.5.0",
]

[project.entry-points."victor.verticals"]
security = "victor_security.vertical:SecurityVertical"

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.0.0",
    "mypy>=0.5.0",
]

[project.urls]
Homepage = "https://github.com/yourusername/victor-security"
Documentation = "https://victor-security.readthedocs.io"
Repository = "https://github.com/yourusername/victor-security"
```

### Publishing Steps

```bash
# 1. Create distribution
python -m build

# 2. Upload to PyPI
twine upload dist/*

# 3. Users install your vertical
pip install victor-security

# 4. Your vertical is now available!
victor chat --vertical security
```

## Examples

### Example 1: Complete Security Vertical

```python
# victor_security/vertical.py

from victor.core.verticals.base import VerticalBase
from victor.tools.base import BaseTool
from typing import List

class SecurityVertical(VerticalBase):
    """Complete security analysis vertical."""

    name = "security"
    description = "Security-focused code analysis and auditing"
    version = "0.5.0"

    def get_tools(self) -> List[type[BaseTool]]:
        """Return security tools."""
        from victor_security.tools import (
            VulnerabilityScanner,
            DependencyAnalyzer,
            ComplianceChecker,
            SecretScanner
        )
        return [
            VulnerabilityScanner,
            DependencyAnalyzer,
            ComplianceChecker,
            SecretScanner
        ]

    def get_system_prompt(self) -> str:
        """Return security-focused system prompt."""
        from victor_security.prompts import get_security_prompt
        return get_security_prompt()
```

### Example 2: DevOps Vertical

```python
# victor_devops/vertical.py

from victor.core.verticals.base import VerticalBase

class DevOpsVertical(VerticalBase):
    """DevOps automation vertical."""

    name = "devops"
    description = "CI/CD, Docker, Kubernetes automation"
    version = "0.5.0"

    def get_tools(self):
        """Return DevOps tools."""
        from victor_devops.tools import (
            DockerBuilder,
            KubernetesDeployer,
            CIConfigGenerator,
            TerraformValidator
        )
        return [
            DockerBuilder,
            KubernetesDeployer,
            CIConfigGenerator,
            TerraformValidator
        ]

    def get_system_prompt(self):
        """Return DevOps system prompt."""
        return """You are a DevOps expert specializing in:
- Docker containerization
- Kubernetes orchestration
- CI/CD pipelines
- Infrastructure as Code
- Automation and monitoring

Provide practical, production-ready solutions."""
```

## Best Practices

### 1. Modularity

```python
# Good: Modular tools
class VulnerabilityScanner(BaseTool):
    """Single responsibility tool."""

class ComplianceChecker(BaseTool):
    """Single responsibility tool."""

# Bad: Monolithic tool
class SecurityTool(BaseTool):
    """Does everything (hard to maintain)."""
```

### 2. Configuration Driven

```yaml
# Good: YAML configuration
capabilities:
  vulnerability_scan:
    enabled: true
    config:
      severity_threshold: "HIGH"

# Bad: Hardcoded configuration
VULNERABILITY_SCAN_ENABLED = True
SEVERITY_THRESHOLD = "HIGH"
```

### 3. Comprehensive Testing

```python
# Test tools individually
@pytest.mark.asyncio
async def test_vulnerability_scanner():
    scanner = VulnerabilityScanner()
    result = await scanner.execute(code="...")
    assert result["success"]

# Test vertical integration
@pytest.mark.asyncio
async def test_security_vertical():
    vertical = SecurityVertical()
    assert len(vertical.get_tools()) > 0
```

### 4. Documentation

```python
class SecurityVertical(VerticalBase):
    """Security analysis vertical.

    This vertical provides tools for:
    - Vulnerability scanning
    - Security auditing
    - Compliance checking
    - Threat modeling

    Example usage:
        victor chat --vertical security "Audit this code"
    """
```

## Conclusion

Creating custom verticals allows you to extend Victor AI for specific domains and use cases. Follow these patterns and best practices to create robust, reusable verticals.

For more examples, see:
- `victor/coding/` - Coding vertical implementation
- `victor/devops/` - DevOps vertical implementation
- `docs/examples/external_vertical/` - Example external vertical

Happy vertical building! 🏗️
