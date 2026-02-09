# Coordinator Recipes - Part 1

**Part 1 of 4:** Recipes 1-2 (Configuration, Prompts)

---

## Navigation

- **[Part 1: Recipes 1-2](#)** (Current)
- [Part 2: Recipe 3](part-2-recipe-3.md)
- [Part 3: Recipes 4-6](part-3-recipes-4-6.md)
- [Part 4: Recipes 7-9](part-4-recipes-7-9.md)
- [**Complete Guide**](../coordinator_recipes.md)

---
# Coordinator-Based Architecture: Recipes

**Version**: 1.0
**Date**: 2025-01-13
**Audience**: Developers, Advanced Users

---

## Table of Contents

1. [Introduction](#introduction)
2. [Recipe 1: Add File-Based Configuration Provider](#recipe-1-add-file-based-configuration-provider)
3. [Recipe 2: Add Project-Specific Prompt Prompts](#recipe-2-add-project-specific-prompt-contributors)
4. [Recipe 3: Export Analytics to Database](#recipe-3-export-analytics-to-database)
5. [Recipe 4: Implement Smart Context Compaction](#recipe-4-implement-smart-context-compaction)
6. [Recipe 5: Add Custom Middleware Integration](#recipe-5-add-custom-middleware-integration)
7. [Recipe 6: Multi-Tenant Configuration](#recipe-6-multi-tenant-configuration)
8. [Recipe 7: Real-Time Analytics Dashboard](#recipe-7-real-time-analytics-dashboard)
9. [Recipe 8: Custom Tool Selection Strategy](#recipe-8-custom-tool-selection-strategy)
10. [Recipe 9: A/B Testing Coordinator](#recipe-9-ab-testing-coordinator)

---

## Introduction

This document provides step-by-step recipes for common coordinator customization tasks. Each recipe includes:

- **Problem Statement**: What problem does this recipe solve?
- **Solution Overview**: High-level approach
- **Step-by-Step Instructions**: Detailed implementation
- **Code Example**: Complete, runnable code
- **Testing**: How to test the implementation
- **Production Considerations**: Things to consider for production use

### Prerequisites

- Victor installed (`pip install victor-ai`)
- Basic knowledge of Python and async/await
- Understanding of coordinator architecture (see [Quick Start](coordinator_quickstart.md))

---

## Recipe 1: Add File-Based Configuration Provider

### Problem Statement

You want to load orchestrator configuration from YAML/JSON files instead of (or in addition to) environment variables
  and settings objects.

### Solution Overview

Create a custom `IConfigProvider` that reads configuration from files, and register it with `ConfigCoordinator`.

### Step-by-Step Instructions

#### Step 1: Create the File Config Provider

```python
# file_config_provider.py
from pathlib import Path
from typing import Dict
import yaml
import json
from victor.protocols import IConfigProvider

class FileConfigProvider(IConfigProvider):
    """Load configuration from YAML or JSON files."""

    def __init__(self, config_path: Path, priority: int = 75):
        """
        Args:
            config_path: Path to config file (YAML or JSON)
            priority: Provider priority (higher = checked first)
        """
        self.config_path = config_path
        self._priority = priority

    def priority(self) -> int:
        return self._priority

    async def get_config(self, session_id: str) -> Dict:
        """Load config from file."""
        if not self.config_path.exists():
            return {}  # Let next provider try

        try:
            if self.config_path.suffix in ['.yml', '.yaml']:
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f) or {}
            elif self.config_path.suffix == '.json':
                with open(self.config_path, 'r') as f:
                    return json.load(f) or {}
            else:
                return {}
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}
```

#### Step 2: Use with Orchestrator

```python
# main.py
from pathlib import Path
from victor.agent.orchestrator import AgentOrchestrator
from victor.agent.coordinators import ConfigCoordinator
from victor.config.settings import Settings

# Create file config provider
config_path = Path("config/orchestrator.yml")
file_provider = FileConfigProvider(config_path, priority=100)

# Create config coordinator with file provider
config_coordinator = ConfigCoordinator(providers=[
    file_provider,              # Try file first
    EnvironmentConfigProvider(),  # Fallback to environment
])

# Create orchestrator
orchestrator = AgentOrchestrator(
    settings=Settings(),
    provider=provider,
    model="claude-sonnet-4-5",
    _config_coordinator=config_coordinator
)
```

#### Step 3: Create Configuration File

```yaml
# config/orchestrator.yml
model: claude-sonnet-4-5
temperature: 0.7
max_tokens: 4096
thinking: false

# Tool selection
tool_selection:
  strategy: hybrid
  hybrid_alpha: 0.7

# Context management
context_compaction_strategy: semantic
context_compaction_threshold: 0.8

# Analytics
enable_analytics: true
analytics_export_interval: 60
```

### Testing

```python
import pytest
import tempfile
from pathlib import Path

@pytest.mark.asyncio
async def test_file_config_provider():
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        f.write("model: test-model\ntemperature: 0.5\n")
        temp_path = Path(f.name)

    try:
        provider = FileConfigProvider(temp_path)
        config = await provider.get_config("test-session")

        assert config['model'] == 'test-model'
        assert config['temperature'] == 0.5
    finally:
        temp_path.unlink()
```

### Production Considerations

- **File permissions**: Ensure config files have appropriate permissions (e.g., 0600)
- **Secrets management**: Don't store API keys in config files; use environment variables or secret managers
- **Validation**: Add schema validation for configuration files
- **Hot reload**: Consider implementing config file watching for dynamic updates
- **Fallbacks**: Always have fallback providers (e.g., environment variables)

---

## Recipe 2: Add Project-Specific Prompt Contributors

### Problem Statement

You want to add project-specific instructions to prompts (e.g., coding standards, compliance requirements, team
  conventions).

### Solution Overview

Create custom `IPromptContributor` implementations that inject project-specific prompts.

### Step-by-Step Instructions

#### Step 1: Create Custom Prompt Contributors

```python
# prompt_contributors.py
from victor.agent.coordinators.prompt_coordinator import BasePromptContributor
from victor.protocols import PromptContext

class ProjectStandardsContributor(BasePromptContributor):
    """Add project coding standards to prompts."""

    def __init__(self, project_path: str):
        self.project_path = project_path

    def priority(self) -> int:
        return 60  # Medium priority

    async def get_contribution(self, context: PromptContext) -> str:
        """Read project standards from files."""

        # Check for project standards file
        standards_file = Path(self.project_path) / ".victor" / "standards.md"
        if not standards_file.exists():
            return ""

        with open(standards_file, 'r') as f:
            standards = f.read()

        return f"""

## Project-Specific Standards
{standards}
"""

class ComplianceContributor(BasePromptContributor):
    """Add compliance requirements based on industry."""

    def __init__(self, industry: str):
        self.industry = industry

    def priority(self) -> int:
        return 70  # High priority

    async def get_contribution(self, context: PromptContext) -> str:
        """Add compliance instructions."""

        requirements = {
            "healthcare": """
## Healthcare Compliance
- Follow HIPAA guidelines for patient data
- No personal health information (PHI) in code
- Ensure audit logging for data access
""",
            "finance": """
## Financial Compliance
- Follow PCI-DSS standards for payment data
- Implement SOC2 controls
- No hardcoding of credentials
""",
            "general": """
## General Compliance
- Follow GDPR data protection guidelines
- Implement proper error handling
- Use secure communication channels
"""
        }

        return requirements.get(self.industry, requirements["general"])

class TechStackContributor(BasePromptContributor):
    """Add technology-specific instructions."""

    def __init__(self, tech_stack: list):
        self.tech_stack = tech_stack

    def priority(self) -> int:
        return 50  # Medium-low priority

    async def get_contribution(self, context: PromptContext) -> str:
        """Add tech stack specific guidance."""

        guidelines = {
            "python": "- Use type hints for all functions\n- Follow PEP 8 style guide\n- Use async/await for I/O
  operations",
            "typescript": "- Use strict mode\n- Prefer interfaces over types\n- Use async/await over promises",
            "rust": "- Use Result types for error handling\n- Prefer iterators over collections\n- Use clap for CLI
  argument parsing",
        }

        instructions = [
            f"\n## {tech.title()} Guidelines\n{guidelines.get(tech, '')}"
            for tech in self.tech_stack
            if tech in guidelines
        ]

        return "\n".join(instructions)
```

#### Step 2: Register with Orchestrator

```python
# main.py
from victor.agent.coordinators import PromptCoordinator
from victor.agent.orchestrator import AgentOrchestrator

# Create custom contributors
project_contributor = ProjectStandardsContributor(project_path="/path/to/project")
compliance_contributor = ComplianceContributor(industry="healthcare")
tech_stack_contributor = TechStackContributor(tech_stack=["python", "typescript"])

# Create prompt coordinator
prompt_coordinator = PromptCoordinator(contributors=[
    compliance_contributor,    # High priority, applied first
    project_contributor,       # Medium priority
    tech_stack_contributor,    # Lower priority
])

# Create orchestrator
orchestrator = AgentOrchestrator(
    settings=Settings(),
    provider=provider,
    model=model,
    _prompt_coordinator=prompt_coordinator
)
```

#### Step 3: Create Project Standards File

```markdown
# .victor/standards.md

## Code Style

- Max line length: 100 characters
- Use 4 spaces for indentation
- Docstrings required for all public functions

## Testing

- Minimum 80% code coverage
- All functions must have unit tests
- Integration tests for API endpoints

## Security

- No secrets in code
- Use environment variables for configuration
- Run security scans before commits
```

### Testing

```python
@pytest.mark.asyncio
async def test_project_standards_contributor():
    contributor = ProjectStandardsContributor(project_path="/tmp/test_project")

    # Create test standards file
    standards_dir = Path("/tmp/test_project/.victor")
    standards_dir.mkdir(parents=True, exist_ok=True)

    standards_file = standards_dir / "standards.md"
    standards_file.write_text("# Test Standards\n- Follow PEP 8")

    context = PromptContext({"task": "code_generation"})
    contribution = await contributor.get_contribution(context)

    assert "Test Standards" in contribution
    assert "Follow PEP 8" in contribution
```

### Production Considerations

- **File caching**: Cache prompt contributions to avoid repeated file reads
- **Validation**: Validate prompt contribution size to avoid token limit issues
- **Version control**: Track prompt contributors in version control
- **Dynamic updates**: Consider hot-reloading prompt files when they change
- **Fallbacks**: Provide default prompts if files are missing

---

## Recipe 3: Export Analytics to Database

### Problem Statement

You want to store usage analytics in a database for reporting, billing, or analysis.

### Solution Overview

Create a custom `IAnalyticsExporter` that writes analytics events to a database.

### Step-by-Step Instructions

#### Step 1: Create Database Exporter

```python
# analytics_exporters.py
from typing import List
from victor.protocols import IAnalyticsExporter, ExportResult, AnalyticsEvent
import asyncpg

class PostgreSQLAnalyticsExporter(BaseAnalyticsExporter):
    """Export analytics to PostgreSQL database."""

    def __init__(self, connection_string: str, batch_size: int = 100):
        """
        Args:
            connection_string: PostgreSQL connection string

---

## See Also

- [Documentation Home](../../README.md)


**Reading Time:** 6 min
**Last Updated:** February 08, 2026**
