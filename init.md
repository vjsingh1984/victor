# init.md

## Project Overview
**codingagent** is a sophisticated AI agent framework primarily written in Python, designed to automate complex coding and DevOps tasks. It supports 24 LLM providers (cloud and local), 34 tool modules, and 9 specialized domain verticals. The system is built to be highly extensible, targeting developers who need a production-ready agentic workflow.

## System Flow
User → Web UI/CLI → AgentOrchestrator → Provider → Tools → System/Files

## Package Layout
| Path | Type | Description |
|------|------|-------------|
| `victor/` | **ACTIVE** | Backend / core logic and framework |
| `web/server/` | Active | Backend API server |
| `web/ui/` | Active | Web-based user interface |
| `vscode-victor/` | Active | VS Code editor extension |
| `docs/` | Active | Documentation and playbooks |
| `examples/` | Active | Sample workflows and plugin examples |
| `scripts/` | Active | Automation and helper scripts |
| `templates/` | Active | Scaffold and template files |
| `victor_test/` | Active | Lightweight demos and test environments |

## Key Entry Points
| Component | Type | Path | Description |
|-----------|------|------|-------------|
| `AgentOrchestrator` | class | `victor/agent/orchestrator.py:446` | Central coordinator for agent execution flow |
| `ServiceContainer` | class | `victor/core/container.py:275` | Dependency injection container for system services |
| `VerticalBase` | class | `victor/core/verticals/base.py:267` | Base class for domain-specific vertical implementations |
| `BaseTool` | class | `victor/tools/base.py:262` | Abstract base for all agent tools |
| `BaseProvider` | class | `victor/providers/base.py:251` | Interface for LLM provider integrations |
| `ToolRegistry` | class | `victor/tools/registry.py:83` | Registry for discovering and managing available tools |
| `Settings` | class | `victor/config/settings.py:608` | Global application configuration management |
| `Message` | class | `victor/context/manager.py:40` | Core data structure for conversation history |
| `WorkflowExecutor` | class | `victor/workflows/executor.py:345` | Engine for executing defined agent workflows |
| `ArgumentNormalizer` | class | `victor/agent/argument_normalizer.py:58` | Handles tool argument parsing and normalization |

## Architecture Patterns
- **Inheritance Backbone**: The system relies on heavy subclassing for extensibility. Key bases include `VerticalBase` (67 subclasses), `BaseTool` (55), `BaseSlashCommand` (46), and `BaseProvider` (36).
- **Hub Classes**: `Message`, `StreamChunk`, and `ArgumentNormalizer` act as high-connectivity hubs, serving as the primary data exchange formats across the system.
- **Facade Pattern**: `AgentOrchestrator` serves as a facade, delegating complex interactions between the agent, the LLM provider, and the tool registry.
- **Plugin/Extension System**: New capabilities are added via "Verticals" (inheriting `VerticalBase`) and "Tools" (inheriting `BaseTool`), which are registered at runtime.
- **Service Locator/DI**: `ServiceContainer` provides a centralized way to manage dependencies and shared services.
- **Coupling Hotspots**: `victor/config/settings.py` and `victor/ui/emoji.py` exhibit extremely high fan-in, acting as utility services used by almost every other module.

## Development Commands
```bash
pip install -e ".[dev]"
pytest
uvicorn web.server.main:app --reload
```

## Dependencies
Core (24 packages): pydantic-settings, python-dotenv, aiofiles, aiohttp, rich, prompt-toolkit, textual, openai, jsonschema, tiktoken, pyyaml.

## Configuration
Config follows a hierarchical override order: `.env` → `~/.victor/profiles.yaml` → CLI flags.
Centralized management is handled by the `Settings` class.

## Codebase Scale
56,814 symbols, 3,696 files, 279,368 graph edges, 12.0% test coverage.

Run `/init --update` to refresh after code changes.
