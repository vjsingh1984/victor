# Developer Journey

**Target Audience:** Contributors who want to develop Victor or build extensions
**Time Commitment:** 2 hours
**Prerequisites:** Python 3.10+, Git familiarity, completed [Intermediate Journey](intermediate.md)

## Journey Overview

This journey prepares you to contribute to Victor's codebase and develop extensions. By the end, you'll be able to:
- Set up a development environment
- Write and run tests
- Create custom tools and verticals
- Contribute to the project

## Visual Guide

```mermaid
flowchart TB
    A([Development Setup<br/>15 min]) --> B([Testing Strategy<br/>20 min])
    B --> C([Extension Development<br/>30 min])
    C --> D([Vertical Development<br/>30 min])
    D --> E([Contribution Workflow<br/>15 min])
    E --> F([Active Contributor])

    style A fill:#e8f5e9,stroke:#2e7d32
    style B fill:#e3f2fd,stroke:#1565c0
    style C fill:#fff3e0,stroke:#e65100
    style D fill:#f3e5f5,stroke:#6a1b9a
    style E fill:#fce4ec,stroke:#880e4f
    style F fill:#c8e6c9,stroke:#1b5e20,stroke-width:3px
```text

See [Contributor Workflow Diagram](../diagrams/user-journeys/contributor-workflow.mmd) for detailed contribution
  process.

## Step 1: Development Setup (15 minutes)

### Prerequisites

- Python 3.10 or higher
- Git
- Make (for running make commands)
- Docker (for integration tests)

### Clone and Install

```bash
# Fork the repository on GitHub first
# Then clone your fork

git clone https://github.com/YOUR_USERNAME/victor.git
cd victor

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Verify Installation

```bash
# Run tests
make test

# Run linters
make lint

# Check all checks pass
make qa
```text

**📖 Full Guide:** [Development Setup](../contributing/setup.md)

### Project Structure

```
victor/
├── victor/
│   ├── agent/           # Agent coordinators and orchestration
│   ├── providers/       # LLM provider implementations
│   ├── tools/           # Tool implementations (55+ tools)
│   ├── core/            # Core services (DI, events, security)
│   ├── framework/       # Workflow engine and validation
│   ├── workflows/       # YAML workflow definitions
│   └── coding/          # Coding vertical
├── tests/
│   ├── unit/            # Unit tests
│   ├── integration/     # Integration tests
│   └── conftest.py      # Test fixtures
└── docs/                # Documentation
```text

**📖 Architecture:** [Architecture Overview](../architecture/overview.md)

## Step 2: Testing Strategy (20 minutes)

### Test Organization

```bash
tests/
├── unit/                # Fast, isolated tests
│   ├── agents/          # Agent coordinator tests
│   ├── providers/       # Provider tests
│   ├── tools/           # Tool tests
│   └── core/            # Core service tests
└── integration/         # Slower, external dependency tests
    ├── test_providers.py
    └── test_tools.py
```

### Running Tests

```bash
# Run all tests
make test

# Run unit tests only
pytest tests/unit -v

# Run specific test file
pytest tests/unit/tools/test_file_tools.py -v

# Run specific test
pytest tests/unit/tools/test_file_tools.py::test_read_file -v

# Run with coverage
make test-cov

# Run integration tests
pytest tests/integration -v

# Skip slow tests
pytest -m "not slow"
```text

**Test Markers:**
- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow tests
- `@pytest.mark.benchmark` - Performance benchmarks

### Writing Tests

```python
# tests/unit/tools/test_my_tool.py
import pytest
from victor.tools.my_tool import MyTool

class TestMyTool:
    @pytest.fixture
    def tool(self):
        return MyTool()

    def test_tool_execution(self, tool):
        result = tool.execute(param="value")
        assert result.success
        assert result.data == "expected"

    @pytest.mark.unit
    def test_error_handling(self, tool):
        with pytest.raises(ValueError):
            tool.execute(invalid_param=True)
```

**📖 Full Guide:** [Testing Guide](../contributing/testing.md)

### Test Fixtures

Key fixtures in `tests/conftest.py`:
- `reset_singletons` - Auto-reset singletons between tests
- `mock_provider` - Mock LLM provider for testing
- `temp_dir` - Temporary directory for file operations
- `sample_codebase` - Sample code for testing

## Step 3: Extension Development (30 minutes)

### Creating Custom Tools

Tools are the primary extension mechanism in Victor.

#### Tool Structure

```python
# victor/tools/my_tool.py
from typing import Dict, Any, Optional
from victor.tools.base import BaseTool

class MyTool(BaseTool):
    """Description of what this tool does."""

    @property
    def name(self) -> str:
        return "my_tool"

    @property
    def description(self) -> str:
        return "Detailed description for LLM"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "param1": {
                    "type": "string",
                    "description": "Parameter description"
                }
            },
            "required": ["param1"]
        }

    @property
    def cost_tier(self) -> int:
        return 1  # 1=low, 2=medium, 3=high

    async def execute(
        self,
        param1: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute the tool."""
        try:
            # Tool implementation
            result = self._do_work(param1)

            return {
                "success": True,
                "data": result
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _do_work(self, param: str) -> Any:
        # Actual implementation
        return f"Processed: {param}"
```text

#### Register Tool

```python
# victor/tools/__init__.py
from victor.tools.my_tool import MyTool

def register_custom_tools():
    return [MyTool()]
```

#### Test Tool

```python
# tests/unit/tools/test_my_tool.py
import pytest
from victor.tools.my_tool import MyTool

def test_my_tool_execution():
    tool = MyTool()
    result = await tool.execute(param1="test")
    assert result["success"] == True
```text

**📖 Tutorial:** [Creating Tools](../guides/tutorials/CREATING_TOOLS.md)

### Creating Custom Capabilities

Capabilities define what operations a vertical can perform.

```python
# victor/coding/capabilities/my_capability.py
from victor.core.capabilities import CapabilityBase, CapabilitySpec

class MyCapability(CapabilityBase):
    """Custom capability definition."""

    @classmethod
    def get_spec(cls) -> CapabilitySpec:
        return CapabilitySpec(
            name="my_capability",
            description="Description of capability",
            version="1.0.0",
            dependencies=[],
            config_schema={
                "type": "object",
                "properties": {
                    "option1": {"type": "boolean"}
                }
            }
        )
```

**📖 Guide:** [Capability Development](../contributing/extensions/capabilities.md)

## Step 4: Vertical Development (30 minutes)

Verticals encapsulate domain-specific functionality.

### Vertical Structure

```python
# victor/myvertical/assistant.py
from victor.core.verticals import VerticalBase
from victor.tools.my_tool import MyTool

class MyVerticalAssistant(VerticalBase):
    """My custom vertical assistant."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._tool = MyTool()

    @property
    def name(self) -> str:
        return "myvertical"

    @property
    def description(self) -> str:
        return "Description of this vertical"

    def get_tools(self) -> List[BaseTool]:
        """Return vertical-specific tools."""
        return [
            self._tool,
            # ... more tools
        ]

    def get_system_prompt(self) -> str:
        """Return system prompt for this vertical."""
        return """You are an expert in my vertical.
        You have access to specialized tools for...
        """
```text

### Register Vertical

```python
# victor/myvertical/__init__.py
from victor.myvertical.assistant import MyVerticalAssistant

__all__ = ["MyVerticalAssistant"]
```

Or use entry points for external verticals:

```toml
# pyproject.toml
[project.entry-points."victor.verticals"]
myvertical = "victor.myvertical:MyVerticalAssistant"
```text

### Test Vertical

```python
# tests/verticals/test_myvertical.py
import pytest
from victor.myvertical.assistant import MyVerticalAssistant

def test_vertical_initialization():
    vertical = MyVerticalAssistant({})
    assert vertical.name == "myvertical"
    assert len(vertical.get_tools()) > 0
```

**📖 Tutorial:** [Creating Verticals](../guides/tutorials/CREATING_VERTICALS.md)

## Step 5: Contribution Workflow (15 minutes)

### Contribution Process

```mermaid
flowchart LR
    A[Fork & Clone] --> B[Create Branch]
    B --> C[Make Changes]
    C --> D[Write Tests]
    D --> E[Run Checks]
    E --> F{Checks Pass?}
    F -->|No| C
    F -->|Yes| G[Commit & Push]
    G --> H[Create PR]
    H --> I[Reviews & CI]
    I --> J{Approved?}
    J -->|No| K[Address Feedback]
    K --> G
    J -->|Yes| L[Merge PR]
```text

### Making Changes

```bash
# Create feature branch
git checkout -b feature/my-feature

# Make changes
vim victor/tools/my_tool.py

# Run tests
make test

# Run linters
make lint

# Commit changes
git add .
git commit -m "feat: add my custom tool

- Add MyTool with parameter validation
- Add unit tests
- Update documentation"

# Push to fork
git push origin feature/my-feature
```

### Creating Pull Request

1. Visit GitHub and create PR from your fork
2. Fill out PR template:
   - Description of changes
   - Testing performed
   - Breaking changes (if any)
   - Checklist completion

### Code Review Process

- **Automated Checks:** CI runs tests, linting, security scan
- **Maintainer Review:** Code review by project maintainers
- **Feedback Cycle:** Address review comments
- **Approval:** Merge after approval

**📖 Full Guide:** [Contributing Guide](../contributing/)

## What's Next?

Congratulations! 🎉 You're ready to contribute to Victor.

### Your First Contribution

**Good First Issues:**
- → [GitHub Issues: "good first issue"](https://github.com/vjsingh1984/victor/labels/good%20first%20issue)
- Fix typos in documentation
- Add missing tests
- Small bug fixes

**Feature Proposals:**
- → [FEP Process](../contributing/FEP_PROCESS.md)
- Submit feature proposals via GitHub Discussions
- Get feedback before implementation

### Advanced Development

**For Architects:**
- → [Advanced Journey](advanced.md)
- System architecture design
- Protocol design
- Performance optimization

**Advanced Topics:**
- [Coordinator Development](../guides/coordinators.md)
- [Workflow Engine](../architecture/workflows.md)
- [Event System](../architecture/event-system.md)
- [Dependency Injection](../architecture/dependency-injection.md)

### Reference

- **Protocols:** [Protocol Reference](../architecture/protocols.md)
- **Best Practices:** [Best Practices](../architecture/best-practices/)
- **Design Patterns:** [Design Patterns](../architecture/patterns/)

---

## See Also

- [Documentation Home](../../README.md)


**Last Updated:** January 31, 2026
**Reading Time:** 10 minutes
**Next Journey:** [Advanced Journey](advanced.md)
