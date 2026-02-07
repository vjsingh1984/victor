# Creating Verticals - Part 2

**Part 2 of 3:** Security, Compliance, and Configuration

---

## Navigation

- [Part 1: Architecture & Features](part-1-architecture-creation-features.md)
- **[Part 2: Security & Config](#)** (Current)
- [Part 3: Testing & Best Practices](part-3-testing-best-practices.md)
- [**Complete Guide**](../CREATING_VERTICALS.md)

---

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
