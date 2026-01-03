# Victor Vertical Development Guide

This guide covers how to create custom domain verticals for Victor, allowing you to extend its capabilities with specialized assistants for different domains.

## Overview

Victor's vertical system allows you to:

- Create specialized assistants for specific domains (security, healthcare, finance, etc.)
- Define custom tools, prompts, and workflows for each domain
- Configure tiered tool selection for context-efficient execution
- Register personas for consistent agent behavior
- Compose tool chains for complex operations

## Architecture

### Vertical Components

Each vertical consists of these core components:

```
victor/{vertical_name}/
├── __init__.py           # Package init, exports VerticalAssistant
├── assistant.py          # Main VerticalAssistant class
├── safety.py             # Safety patterns and constraints
├── prompts.py            # Prompt contributors
├── mode_config.py        # Agent mode configurations
├── service_provider.py   # Dependency injection services (optional)
├── tool_dependencies.py  # Tool dependency declarations (optional)
└── workflows/            # YAML workflow definitions
    └── {workflow_name}.yaml
```

### Key Concepts

1. **VerticalBase**: Abstract base class that all verticals inherit from
2. **VerticalRegistry**: Singleton that manages vertical discovery and access
3. **VerticalIntegrationPipeline**: Applies vertical configurations to the orchestrator
4. **Protocols**: ISP-compliant interfaces for different extension points

## Quick Start

### 1. Create the Vertical Package

```python
# victor/security/__init__.py
from victor.security.assistant import SecurityAssistant

__all__ = ["SecurityAssistant"]
```

### 2. Implement the Assistant Class

```python
# victor/security/assistant.py
from typing import Dict, List, Optional, Any
from victor.core.verticals.base import VerticalBase
from victor.core.vertical_types import TieredToolConfig

class SecurityAssistant(VerticalBase):
    """Security analysis vertical assistant."""

    @classmethod
    def get_name(cls) -> str:
        return "security"

    @classmethod
    def get_description(cls) -> str:
        return "Security analysis and vulnerability assessment assistant"

    @classmethod
    def get_tools(cls) -> List[str]:
        """Return list of tool names this vertical uses."""
        return [
            "read",          # Read files for security review
            "grep",          # Search for security patterns
            "bash",          # Run security scanners
            "web_fetch",     # Fetch CVE databases
        ]

    @classmethod
    def get_tiered_tool_config(cls) -> Optional[TieredToolConfig]:
        """Return tiered tool configuration for context-efficient selection."""
        return TieredToolConfig(
            mandatory={"read", "grep"},
            vertical_core={"bash", "web_fetch"},
            semantic_pool={"write", "edit"},
            readonly_only_for_analysis=False,
        )

    @classmethod
    def get_system_prompt(cls) -> str:
        return """You are a security analysis assistant specializing in:
- Vulnerability assessment
- Code security review
- Security best practices
- Threat modeling

Always prioritize security and never execute potentially harmful code."""

    @classmethod
    def get_priority_keywords(cls) -> List[str]:
        """Keywords that trigger this vertical."""
        return [
            "security", "vulnerability", "cve", "exploit",
            "authentication", "authorization", "encryption",
            "threat", "attack", "penetration", "audit",
        ]
```

### 3. Add Safety Patterns

```python
# victor/security/safety.py
from victor.core.verticals.protocols import SafetyExtensionProtocol, SafetyPattern

class SecuritySafetyExtension(SafetyExtensionProtocol):
    """Safety patterns for security vertical."""

    def get_bash_patterns(self) -> list[SafetyPattern]:
        return [
            SafetyPattern(
                pattern=r"nmap\s+-sS",
                description="SYN scan requires explicit approval",
                severity="high",
                action="require_approval",
            ),
            SafetyPattern(
                pattern=r"sqlmap",
                description="SQL injection testing",
                severity="high",
                action="require_approval",
            ),
        ]

    def get_file_patterns(self) -> list[SafetyPattern]:
        return [
            SafetyPattern(
                pattern=r"/etc/shadow",
                description="Accessing shadow password file",
                severity="critical",
                action="block",
            ),
        ]
```

### 4. Configure Prompt Contributors

```python
# victor/security/prompts.py
from typing import Dict
from victor.core.verticals.protocols import PromptContributorProtocol, TaskTypeHint

class SecurityPromptContributor(PromptContributorProtocol):
    """Prompt contributions for security analysis."""

    def get_priority(self) -> int:
        return 100  # Higher = later in prompt

    def get_system_prompt_section(self) -> str:
        return """
## Security Guidelines
- Always check for common vulnerabilities (OWASP Top 10)
- Report findings with severity levels
- Suggest remediation steps
- Never exploit vulnerabilities without explicit permission
"""

    def get_task_type_hints(self) -> Dict[str, TaskTypeHint]:
        return {
            "security_audit": TaskTypeHint(
                keywords=["audit", "review", "assess"],
                suggested_tools=["grep", "read", "bash"],
                prompt_additions="Focus on systematic security review.",
            ),
            "vulnerability_scan": TaskTypeHint(
                keywords=["scan", "vulnerability", "cve"],
                suggested_tools=["bash", "web_fetch"],
                prompt_additions="Check CVE databases for known issues.",
            ),
        }
```

### 5. Add Mode Configurations

```python
# victor/security/mode_config.py
from victor.core.verticals.protocols import ModeConfigProviderProtocol, ModeConfig

class SecurityModeConfigProvider(ModeConfigProviderProtocol):
    """Mode configurations for security vertical."""

    def get_mode_configs(self) -> Dict[str, ModeConfig]:
        return {
            "audit": ModeConfig(
                name="Security Audit",
                description="Comprehensive security review mode",
                tool_budget_multiplier=2.0,
                allowed_tools=["read", "grep", "bash"],
                system_prompt_additions="Perform thorough security analysis.",
            ),
            "quick_scan": ModeConfig(
                name="Quick Scan",
                description="Fast vulnerability check",
                tool_budget_multiplier=0.5,
                allowed_tools=["grep", "read"],
                system_prompt_additions="Quick security scan, prioritize critical issues.",
            ),
        }
```

## Advanced Features

### Tool Tier Registry

Register your vertical's tool tiers globally:

```python
from victor.core.tool_tier_registry import get_tool_tier_registry
from victor.core.vertical_types import TieredToolConfig

registry = get_tool_tier_registry()
registry.register(
    "security",
    TieredToolConfig(
        mandatory={"read", "grep"},
        vertical_core={"bash", "web_fetch"},
        semantic_pool={"write", "edit"},
    ),
    parent="base",  # Inherit from base tier
    description="Security analysis tools",
)
```

### Chain Registry

Register reusable tool chains:

```python
from victor.framework.chain_registry import chain, get_chain_registry

@chain("security:vulnerability_scan", description="Scan for vulnerabilities")
def vulnerability_scan_chain():
    from victor.tools.composition import as_runnable
    return (
        as_runnable(grep_tool, pattern="TODO|FIXME|HACK")
        | as_runnable(analyze_tool)
        | format_results
    )

# Use in YAML workflows:
# handler: chain:security:vulnerability_scan
```

### Persona Registry

Register domain-specific personas:

```python
from victor.framework.persona_registry import persona, PersonaSpec

@persona("security:analyst")
def security_analyst():
    return PersonaSpec(
        name="security_analyst",
        role="Security Analyst",
        expertise=["vulnerability assessment", "threat modeling", "secure coding"],
        communication_style="professional",
        behavioral_traits=["thorough", "cautious", "detail-oriented"],
    )
```

### YAML Workflows

Create domain workflows in `victor/{vertical}/workflows/`:

```yaml
# victor/security/workflows/security_audit.yaml
workflows:
  security_audit:
    description: "Comprehensive security audit workflow"
    metadata:
      version: "1.0"
      vertical: "security"

    nodes:
      - id: gather_files
        type: agent
        role: file_gatherer
        goal: "Identify all security-relevant files"
        tool_budget: 10
        output: files_list
        next: [analyze_code]

      - id: analyze_code
        type: compute
        handler: chain:security:vulnerability_scan
        inputs:
          files: $ctx.files_list
        output: scan_results
        next: [generate_report]

      - id: generate_report
        type: agent
        role: report_writer
        goal: "Generate security audit report"
        tool_budget: 5
        output: final_report
```

## Protocol Reference

### Core Protocols

| Protocol | Purpose | Key Methods |
|----------|---------|-------------|
| `VerticalBase` | Base class for verticals | `get_name()`, `get_tools()`, `get_system_prompt()` |
| `SafetyExtensionProtocol` | Safety patterns | `get_bash_patterns()`, `get_file_patterns()` |
| `PromptContributorProtocol` | Prompt contributions | `get_system_prompt_section()`, `get_task_type_hints()` |
| `ModeConfigProviderProtocol` | Mode configurations | `get_mode_configs()` |
| `ToolDependencyProviderProtocol` | Tool dependencies | `get_tool_dependencies()` |
| `WorkflowProviderProtocol` | Workflow definitions | `get_workflows()` |
| `TieredToolConfigProviderProtocol` | Tiered tools | `get_tiered_tool_config()` |
| `VerticalPersonaProviderProtocol` | Personas | `get_persona_specs()` |

### VerticalExtensions Container

The `VerticalExtensions` dataclass aggregates all extensions:

```python
from victor.core.verticals.protocols import VerticalExtensions

extensions = VerticalExtensions(
    middleware=[CustomMiddleware()],
    safety_extensions=[SecuritySafetyExtension()],
    prompt_contributors=[SecurityPromptContributor()],
    mode_config_provider=SecurityModeConfigProvider(),
    tiered_tool_config=my_tiered_config,
)
```

## Registering Your Vertical

### Built-in Registration

Add to `victor/core/verticals/__init__.py`:

```python
def _register_builtin_verticals(self) -> None:
    # ... existing verticals ...
    from victor.security import SecurityAssistant
    self._register_class(SecurityAssistant)
```

### External Plugin (entry_points)

For pip-installable verticals:

```toml
# pyproject.toml
[project.entry-points."victor.verticals"]
security = "victor_security:SecurityAssistant"
```

## Testing Your Vertical

```python
# tests/unit/security/test_security_assistant.py
import pytest
from victor.security import SecurityAssistant

class TestSecurityAssistant:
    def test_get_name(self):
        assert SecurityAssistant.get_name() == "security"

    def test_get_tools(self):
        tools = SecurityAssistant.get_tools()
        assert "read" in tools
        assert "grep" in tools

    def test_priority_keywords(self):
        keywords = SecurityAssistant.get_priority_keywords()
        assert "security" in keywords
        assert "vulnerability" in keywords

    def test_tiered_config(self):
        config = SecurityAssistant.get_tiered_tool_config()
        assert config is not None
        assert "read" in config.mandatory
```

## Best Practices

1. **Keep verticals focused**: Each vertical should handle a specific domain
2. **Use tiered tools**: Configure mandatory, vertical_core, and semantic_pool appropriately
3. **Implement safety patterns**: Protect against dangerous operations
4. **Register personas**: Provide consistent agent behavior
5. **Create reusable chains**: Use the ChainRegistry for complex operations
6. **Write YAML workflows**: Define common workflows declaratively
7. **Test thoroughly**: Unit test all protocol implementations

## Example Verticals

Reference implementations in the codebase:

- `victor/coding/` - Code development and review
- `victor/devops/` - DevOps and infrastructure
- `victor/rag/` - Document retrieval and analysis
- `victor/dataanalysis/` - Data analysis and visualization
- `victor/research/` - Web research and synthesis

## Troubleshooting

### Vertical Not Loading

1. Check `VerticalRegistry` registration
2. Verify `__init__.py` exports the assistant class
3. Check for import errors in the vertical module

### Tools Not Available

1. Verify tools are listed in `get_tools()`
2. Check tiered config includes the tool
3. Ensure tool is registered in ToolRegistry

### Prompts Not Applied

1. Check prompt contributor priority
2. Verify `get_extensions()` includes the contributor
3. Check VerticalIntegrationPipeline logs
