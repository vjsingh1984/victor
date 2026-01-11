# Development Guide

**Everything you need to contribute to Victor.**

## Quick Start

```bash
# Clone and install
git clone https://github.com/vjsingh1984/victor.git
cd victor
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
make test
pytest tests/unit -v

# Format code
make format
```

## Documentation Map

```
docs/
├── README.md                          # Project overview
├── getting-started/                   # User onboarding
├── user-guide/                        # Daily usage
├── development/                       # Developer docs
│   ├── index.md                       # This file
│   ├── architecture/                  # System design
│   │   ├── overview.md                # High-level architecture
│   │   ├── components.md              # Component reference
│   │   ├── framework-vertical-integration.md
│   │   ├── state-machine.md           # Conversation stages
│   │   └── data-flow.md               # EventBus patterns
│   ├── extending/                     # Extension guides
│   │   ├── verticals.md               # Vertical development
│   │   └── plugins.md                 # Plugin system
│   ├── testing/                       # Testing strategy
│   └── releasing/                     # Release process
├── guides/                            # How-to guides
│   ├── vertical-quickstart.md         # Vertical quick reference
│   ├── tool-reference.md              # Tool catalog
│   ├── workflow-quickstart.md         # Workflow patterns
│   ├── multi-agent-quickstart.md      # Team coordination
│   ├── development/                   # Dev-specific guides
│   ├── observability/                 # Monitoring & debugging
│   ├── workflow-development/          # Workflow DSL
│   └── integration/                   # Integration guides
└── reference/                         # API & config reference
    ├── providers-comparison.md        # Provider matrix
    ├── api/                            # HTTP/MCP APIs
    ├── configuration/                  # Settings reference
    ├── providers/                      # Provider docs
    ├── tools/                          # Tool documentation
    └── verticals/                      # Built-in verticals
```

## Quick Reference

### Architecture

| Layer | Components | File |
|-------|------------|------|
| **Clients** | CLI, TUI, HTTP API, MCP | `victor/cli/`, `victor/integrations/` |
| **Orchestrator** | AgentOrchestrator, Controllers | `victor/agent/` |
| **Framework** | StateGraph, Workflows, Teams | `victor/framework/` |
| **Verticals** | 5 built-in + custom | `victor/{vertical}/` |
| **Providers** | 21 LLM providers | `victor/providers/` |
| **Tools** | 55+ specialized tools | `victor/tools/` |

### Verticals

| Vertical | Tools | Use Case |
|----------|-------|----------|
| **coding** | 30+ | Code analysis, refactoring, testing |
| **research** | 9 | Web search, synthesis, citations |
| **devops** | 13 | Docker, CI/CD, infrastructure |
| **data_analysis** | 11 | Pandas, visualization, statistics |
| **rag** | 10 | Document retrieval, vector search |

### Key Protocols

| Protocol | Purpose | Location |
|----------|---------|----------|
| `BaseProvider` | LLM abstraction | `victor/providers/base.py` |
| `BaseTool` | Tool interface | `victor/tools/base.py` |
| `VerticalBase` | Domain extension | `victor/core/verticals/base.py` |
| `CapabilityRegistryProtocol` | Capability discovery | `victor/framework/protocols.py` |

## Development Tasks

| Task | Command | Description |
|------|---------|-------------|
| **Run tests** | `make test` | Unit tests only |
| **Run all tests** | `make test-all` | Including integration |
| **Format code** | `make format` | Black + ruff |
| **Lint code** | `make lint` | Check formatting |
| **Type check** | `mypy victor` | Type validation |

## Extension Points

### Add a Provider

```python
from victor.providers.base import BaseProvider

class MyProvider(BaseProvider):
    @property
    def name(self) -> str:
        return "myprovider"

    async def chat(self, message: str, **kwargs) -> ChatResponse:
        # Implementation
        pass
```

[Provider Reference →](../reference/providers/)

### Add a Tool

```python
from victor.tools.base import BaseTool

class MyTool(BaseTool):
    name = "my_tool"
    description = "Does something useful"

    async def execute(self, **kwargs) -> ToolResult:
        # Implementation
        pass
```

[Tool Catalog →](../reference/tools/catalog.md)

### Create a Vertical

```python
from victor.core.verticals import VerticalBase

class MyVertical(VerticalBase):
    name = "my_vertical"

    @classmethod
    def get_tools(cls) -> list[str]:
        return ["read", "write", "grep"]

    @classmethod
    def get_system_prompt(cls) -> str:
        return "You are an expert in..."
```

[Vertical Quickstart →](../guides/vertical-quickstart.md)

### Create a Workflow

```yaml
workflows:
  my_workflow:
    nodes:
      - id: step1
        type: agent
        role: researcher
        goal: "Research the topic"
        next: [step2]

      - id: step2
        type: compute
        handler: summarize
        next: []
```

[Workflow Quickstart →](../guides/workflow-quickstart.md)

## Testing

| Test Type | Marker | Command |
|-----------|--------|---------|
| Unit | `@pytest.mark.unit` | `pytest -m unit` |
| Integration | `@pytest.mark.integration` | `pytest -m integration` |
| Slow | `@pytest.mark.slow` | `pytest -m "not slow"` |
| Workflow | `@pytest.mark.workflows` | `pytest -m workflows` |

[Testing Guide →](testing/strategy.md)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run `make test && make lint`
5. Commit with conventional commits
6. Push and create PR

[Contribution Workflow →](../../CONTRIBUTING.md)

## Resources

| Topic | Link |
|-------|------|
| Architecture Overview | [architecture/overview.md](architecture/overview.md) |
| Component Reference | [architecture/components.md](architecture/components.md) |
| Provider Comparison | [../reference/providers-comparison.md](../reference/providers-comparison.md) |
| Tool Reference | [../guides/tool-reference.md](../guides/tool-reference.md) |
| Multi-Agent Teams | [../guides/multi-agent-quickstart.md](../guides/multi-agent-quickstart.md) |

---

**Next**: [Setup Guide →](setup.md)
