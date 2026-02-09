# Creating Verticals - Part 3

**Part 3 of 3:** Testing, Publishing, Examples, and Best Practices

---

## Navigation

- [Part 1: Architecture & Features](part-1-architecture-creation-features.md)
- [Part 2: Security & Config](part-2-security-config.md)
- **[Part 3: Testing & Best Practices](#)** (Current)
- [**Complete Guide**](../CREATING_VERTICALS.md)

---

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
```text

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
```text

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
```text

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
```text

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

Creating custom verticals allows you to extend Victor AI for specific domains and use cases. Follow these patterns and
  best practices to create robust,
  reusable verticals.

For more examples, see:
- `victor/coding/` - Coding vertical implementation
- `victor/devops/` - DevOps vertical implementation
- `docs/examples/external_vertical/` - Example external vertical

Happy vertical building! 🏗️

---

## See Also

- [Documentation Home](../../README.md)


**Last Updated:** February 01, 2026
**Reading Time:** 2 min
