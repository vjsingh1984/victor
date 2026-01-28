# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Reference

**Victor** is a provider-agnostic AI coding assistant supporting 21 LLM providers with 55 specialized tools across 5 domain verticals (Coding, DevOps, RAG, Data Analysis, Research). The architecture follows a two-layer coordinator pattern with 20 specialized coordinators, protocol-based design (98 protocols), and dependency injection (55+ services in ServiceContainer).

**Development Commands**:
```bash
# Setup
pip install -e ".[dev]"               # Install for development
pip install -e ".[dev,docs,build]"    # Install with all dev dependencies

# Testing
make test                             # Run unit tests
make test-all                         # All tests including integration
make test-cov                         # Run tests with coverage
pytest tests/unit/path/test_file.py::test_name -v  # Single test

# Code Quality
make lint                             # ruff + black --check + mypy
make format                           # black + ruff --fix

# Developer Tools (Phase 4)
make dev-tools                        # Run all developer tools
make check-protocol                   # Check protocol conformance
make lint-vertical                    # Lint verticals
make validate-config                  # Validate YAML configs
make profile-coordinators             # Profile coordinator performance
make coverage-report                  # Generate coverage reports
make generate-docs                    # Generate documentation

# Quality Assurance (Phase 5)
make qa                               # Comprehensive QA validation
make qa-fast                          # Quick QA validation (skip slow tests)
make qa-report                        # Generate detailed JSON report
make release-checklist                # Show release checklist
make release-validate                 # Validate release readiness

# Security
make security                         # Quick security scans
make security-full                    # Comprehensive security scans

# Performance & Load Testing
make benchmark                        # Run performance benchmarks
make load-test                        # Run load tests with Locust
make load-test-quick                  # Run quick load tests (pytest-based)
make load-test-report                 # Generate load test report

# Build & Distribution
make build                            # Build Python packages (sdist + wheel)
make build-binary                     # Build standalone binary
make docker                           # Build Docker image
make clean                            # Clean build artifacts

# Code Validation
victor validate files main.py         # Validate code syntax
victor validate files src/*.py -s     # Strict mode (exit 1 on errors)
victor validate languages             # List supported languages
victor validate check app.ts          # Check validation support for file

# Run application
victor                                # TUI mode (or 'vic')
victor chat --no-tui                  # CLI mode
victor chat --provider anthropic --model claude-sonnet-4-5
victor serve                          # Start API server for IDE integrations
```

## Architecture Overview

Victor uses a **two-layer coordinator architecture** that separates application-specific orchestration from framework-agnostic workflow infrastructure:

```
┌─────────────────────────────────────────────────────────────────┐
│  CLIENTS: CLI/TUI │ VS Code (HTTP) │ MCP Server │ API Server    │
└─────────────────────────┬───────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              ServiceContainer (DI) - 55+ services               │
└─────────────────────────┬───────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                 AgentOrchestrator (Facade)                      │
│  Delegates to 20 specialized coordinators (app + framework)     │
└─────────────────────────┬───────────────────────────────────────┘
                          ▼
┌───────────┬─────────────┬───────────────┬───────────────────────┐
│ PROVIDERS │   TOOLS     │  WORKFLOWS    │  VERTICALS            │
│  21       │   55        │  StateGraph   │  Coding/DevOps/RAG/   │
│           │             │  + YAML       │  DataAnalysis/Research│
└───────────┴─────────────┴───────────────┴───────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│            Event Bus (In-Memory, Kafka, SQS, RabbitMQ, Redis)   │
└─────────────────────────────────────────────────────────────────┘
```

### Two-Layer Coordinator Design

**Application Layer** (`victor/agent/coordinators/`): Victor-specific business logic
- ChatCoordinator, ToolCoordinator, ContextCoordinator, PromptCoordinator
- SessionCoordinator, ProviderCoordinator, ModeCoordinator, etc.

**Framework Layer** (`victor/framework/coordinators/`): Domain-agnostic infrastructure
- YAMLWorkflowCoordinator, GraphExecutionCoordinator, HITLCoordinator, CacheCoordinator

**Benefits**: Single responsibility, clear boundaries, reusability across verticals, independent testing.

### Key Directories

| Directory | Purpose | Key Components |
|-----------|---------|---------------|
| `victor/agent/` | Core orchestration | `orchestrator.py` (facade), `coordinators/` (20 coordinators), `tool_pipeline.py` |
| `victor/providers/` | LLM provider implementations | `BaseProvider`, 21 providers (Anthropic, OpenAI, Google, Ollama, etc.) |
| `victor/tools/` | 55+ specialized tools | `BaseTool`, file ops, git, testing, web search |
| `victor/core/` | Foundation services | DI container, events, verticals base classes, registries |
| `victor/core/language_capabilities/` | Unified language system | Registry (40+ languages), validators, extractors, feature flags |
| `victor/core/security/` | Security infrastructure | RBAC, audit logging, authorization (RBAC + ABAC) |
| `victor/framework/` | Workflow engine | StateGraph DSL, validation, resilience, parallel execution |
| `victor/workflows/` | YAML workflow definitions | Compiler, scheduler, execution engine |
| `victor/protocols/` | Interface definitions (98 protocols) | Loose coupling, testability |
| `victor/config/` | Configuration system | YAML configs for modes, teams, capabilities, personas |
| `victor/optimization/` | Unified optimization | `workflow/`, `runtime/`, `core/` (workflow profiling, lazy loading, hot path) |
| `victor/coding/` | Coding vertical domain | Code analysis, refactoring, testing tools |
| `victor/devops/` | DevOps vertical domain | Docker, Kubernetes, CI/CD tools |
| `victor/rag/` | RAG vertical domain | Document search, retrieval, Q&A |
| `victor/dataanalysis/` | Data Analysis vertical | Pandas, visualization, stats tools |
| `victor/research/` | Research vertical | Literature search, analysis tools |
| `victor/security_analysis/` | Security analysis vertical | Vulnerability scanning, CVE databases, penetration testing |

### Key Design Patterns

**Protocol-Based Design** (98 protocols): All major components depend on abstractions.
```python
from victor.agent.protocols import ToolExecutorProtocol
from victor.core.container import ServiceContainer

container = ServiceContainer()
executor = container.get(ToolExecutorProtocol)
```

**Dependency Injection**: ServiceContainer manages 55+ services with lifecycle control (singleton, scoped, transient).

**Provider System**: Inherit `BaseProvider`, implement `chat()`, `stream_chat()`, `supports_tools()`, `name`.

**Tool System**: Inherit `BaseTool`, define `name`, `description`, `parameters` (JSON Schema), `cost_tier`, `execute()`.

**Two-Layer Coordinator Pattern**: Application layer (`victor/agent/coordinators/`, Victor-specific) + Framework layer (`victor/framework/coordinators/`, domain-agnostic).

**Event-Driven Architecture**: 5 pluggable backends (In-Memory, Kafka, SQS, RabbitMQ, Redis) for async communication.

**YAML-First Configuration**: Modes, capabilities, teams configured via YAML in `victor/config/`.

**Step Handlers**: Extensible pipeline for vertical integration with SOLID compliance.

## Adding Components

### New Provider
1. Create `victor/providers/your_provider.py` inheriting `BaseProvider`
2. Implement required methods (`chat()`, `stream_chat()`, `supports_tools()`, `name`)
3. Register in `ProviderRegistry`
4. Add tests in `tests/unit/providers/`

### New Tool
1. Create `victor/tools/your_tool.py` inheriting `BaseTool`
2. Define `name`, `description`, `parameters` (JSON Schema), `cost_tier`, `execute()`
3. Register in tool registry
4. Add tests in `tests/unit/tools/`

### New Coordinator
1. Create in `victor/agent/coordinators/` (application) or `victor/framework/coordinators/` (framework)
2. Inherit appropriate protocol dependencies for loose coupling
3. Follow Single Responsibility Principle - one clear purpose
4. Delegate to specialized services, don't implement everything
5. Add tests in `tests/unit/agent/coordinators/` or `tests/unit/framework/`

### New Step Handler (Vertical Integration)
Step handlers are the primary extension mechanism for vertical development (SOLID-compliant):

```python
from victor.framework.step_handlers import BaseStepHandler, StepHandlerRegistry

class CustomToolsHandler(BaseStepHandler):
    """Handle custom tool registration with validation."""

    @property
    def name(self) -> str:
        return "custom_tools"

    @property
    def order(self) -> int:
        return 12  # After default tools (10), before tiered config (15)

    def _do_apply(self, orchestrator, vertical, context, result):
        tools = vertical.get_tools()
        validated = self._validate_tools(tools)
        context.apply_enabled_tools(validated)
        result.add_info(f"Applied {len(validated)} tools")

# Register
registry = StepHandlerRegistry.default()
registry.add_handler(CustomToolsHandler())
```

Built-in handlers (order): `CapabilityConfig` (5), `Tool` (10), `TieredConfig` (15), `Prompt` (20), `Config` (40), `Extensions` (45), `Middleware` (50), `Framework` (60), `Context` (100).

### External Vertical (Plugin)
Register via entry points in `pyproject.toml`:
```toml
[project.entry-points."victor.verticals"]
security = "victor_security:SecurityAssistant"
```
Requirements: Inherit `VerticalBase`, define `name`, implement `get_tools()` and `get_system_prompt()`.

### External Capability (Open/Closed Principle)
Register via entry points:
```toml
[project.entry-points."victor.capabilities"]
my_capability = "my_package.capabilities:MyCapabilityClass"
```
Requirements: Inherit `CapabilityBase`, implement `get_spec()` classmethod.

## Testing

**Key Fixtures** (tests/conftest.py):
- `reset_singletons`: Auto-resets all singletons between tests (prevents test pollution)
- `isolate_environment_variables`: Isolates from `.env` and API keys
- `auto_mock_docker_for_orchestrator`: Mocks Docker for orchestrator tests
- `mock_code_execution_manager`, `mock_docker_client`: Mock Docker for code execution tests
- Workflow: `empty_workflow_graph`, `linear_workflow_graph`, `branching_workflow_graph`
- HITL: `hitl_executor`, `auto_approve_handler`, `auto_reject_handler`

**Test Markers**:
- `@pytest.mark.unit` - Unit tests (fast, isolated)
- `@pytest.mark.integration` - Integration tests (require external services)
- `@pytest.mark.slow` - Slow tests (deselect with `-m "not slow"`)
- `@pytest.mark.workflows` - Workflow-related tests
- `@pytest.mark.hitl` - Human-in-the-loop tests
- `@pytest.mark.benchmark` - Performance benchmarks
- `@pytest.mark.load_test` - Load and scalability tests

**Provider-Specific Tests**: Use `@requires_ollama()` decorator or `is_ollama_available()` check for tests requiring Ollama.

**HTTP Mocking**: Use `respx` for HTTP mocking in async tests.

**macOS Multiprocessing**: Tests use 'spawn' start method to avoid semaphore leak warnings.

## Canonical Imports

| Type | Import |
|------|--------|
| `ToolCall` | `from victor.agent.tool_calling.base import ToolCall` |
| `StreamChunk` | `from victor.providers.base import StreamChunk` |
| Protocols | `from victor.protocols import ...` |
| Team types | `from victor.teams import TeamFormation, create_coordinator` |
| `EventBus` | `from victor.observability.event_bus import EventBus` |
| StateGraph | `from victor.framework.graph import StateGraph, START, END` |
| Agent/Task | `from victor.framework import Agent, Task, State` |
| Resilience | `from victor.framework.resilience import ExponentialBackoffStrategy, CircuitBreaker` |
| Validation | `from victor.framework.validation import ValidationPipeline, ThresholdValidator` |
| Mode config | `from victor.core.mode_config import ModeConfigRegistry, ModeDefinition` |
| Registry | `from victor.core.registries import UniversalRegistry, CacheStrategy` |
| RBAC | `from victor.core.security.auth import RBACManager, Permission, Role` |
| Audit | `from victor.core.security.audit import AuditManager, AuditEvent` |
| Authorization | `from victor.core.security.authorization import EnhancedAuthorizer` |
| Security Analysis | `from victor.security_analysis import SecurityAnalysisAssistant` |
| Optimization | `from victor.optimization import WorkflowOptimizer, LazyComponentLoader, json_dumps` |

## Important Notes

**Air-Gapped Mode** (`airgapped_mode=True`): Only local providers (Ollama, LMStudio, vLLM), no web tools.

**Tool Selection**: Pluggable strategy (`keyword`, `semantic`, `hybrid`). Default: `hybrid` using sentence-transformers embeddings.

**Agent Modes**: `BUILD` (default, full edits), `PLAN` (2.5x exploration, sandbox), `EXPLORE` (3.0x exploration, no edits).

**Conversation State**: Tracks stages (`INITIAL` → `PLANNING` → `READING` → `ANALYZING` → `EXECUTING` → `VERIFICATION` → `COMPLETION`).

**Project Context**: Loads `.victor.md`, `CLAUDE.md`, or `.victor/init.md` from working directory.

**Lazy Loading**: Verticals lazy-load by default (72.8% faster startup). Set `VICTOR_LAZY_LOADING=false` to disable.

**Vector Storage**: LanceDB is the default vector store for semantic code search and conversation embeddings. ChromaDB and ProximaDB (experimental) are optional alternatives.

**Event Bus Backends**: In-Memory (default, single-instance), Kafka (distributed, exactly-once), SQS (serverless, at-least-once), RabbitMQ (reliable, at-least-once), Redis (fast, at-least-once).

**Multi-Provider Workflows**: Switch providers mid-conversation without losing context, or use different providers for cost/quality optimization.

**Protocol Conformance**: Use `make check-protocol` to verify protocol conformance. See `docs/architecture/BEST_PRACTICES.md` for protocol usage guidelines.

**Code Validation**: Unified language capability system validates code before file writes. Supports 40+ languages with tiered support:
- **Tier 1** (Python, JS, TS): Native AST + Tree-sitter + LSP
- **Tier 2** (Go, Java, Rust, C/C++): Native/Tree-sitter + LSP
- **Tier 3** (Ruby, PHP, etc.): Tree-sitter + optional LSP
- **Config files** (JSON, YAML, TOML, XML): Native Python validators

Use `victor validate files <files>` to validate code, or set `VICTOR_VALIDATION_ENABLED=false` to disable.

## Code Style

- Type hints on all public APIs
- Google-style docstrings
- Line length: 100 chars (black enforced)
- Async/await for all I/O

## Entry Points

| Entry Point | Description |
|-------------|-------------|
| `victor.ui.cli:app` | Main CLI/TUI entry point (`victor` or `vic`) |
| `victor.api.server:app` | FastAPI HTTP server for IDE integrations |

**Developer Tools CLI Commands**:
```bash
victor-check-protocol    # Check protocol conformance
victor-lint-vertical      # Lint verticals
victor-validate-config    # Validate YAML configs
victor-profile-coordinators  # Profile coordinator performance
victor-coverage-report    # Generate coverage reports
victor-generate-docs      # Generate documentation
```

## Optional Dependencies

Victor uses optional dependencies for modular installation. Install what you need:

```bash
# Core language support (tree-sitter grammars)
pip install victor-ai[lang-all]           # All language grammars
pip install victor-ai[lang-web]           # HTML, CSS, JS, TS
pip install victor-ai[lang-jvm]           # Scala, Kotlin, Java
pip install victor-ai[lang-systems]       # C, C++, Bash

# API server for IDE integrations
pip install victor-ai[api]                # FastAPI + Uvicorn

# Vector storage alternatives
pip install victor-ai[vector-alt]         # ChromaDB (alternative to LanceDB)
pip install victor-ai[vector-experimental] # ProximaDB (experimental)

# Database support
pip install victor-ai[database]           # PostgreSQL + DuckDB

# Observability and monitoring
pip install victor-ai[observability]      # OpenTelemetry + Prometheus

# Experiments and A/B testing
pip install victor-ai[experiments]        # SciPy for statistical analysis

# All optional dependencies
pip install victor-ai[all]                # Everything
```

**Key Optional Groups**:
- `lang-*`: Additional tree-sitter language grammars for code analysis
- `api`: FastAPI server for IDE integrations (VS Code, JetBrains)
- `vector-alt`: Alternative vector stores (ChromaDB)
- `database`: Database support (PostgreSQL, DuckDB)
- `observability`: OpenTelemetry, Prometheus, structured logging
- `experiments`: Statistical analysis for A/B testing

## Environment Variables

Key environment variables for configuration:

```bash
# Provider API Keys
ANTHROPIC_API_KEY=sk-...                  # Anthropic (Claude)
OPENAI_API_KEY=sk-...                     # OpenAI (GPT)
GOOGLE_API_KEY=...                        # Google (Gemini)
AZURE_API_KEY=...                         # Azure OpenAI
AWS_ACCESS_KEY_ID=...                     # AWS Bedrock

# Local Providers
OLLAMA_BASE_URL=http://localhost:11434    # Ollama local server
LMSTUDIO_BASE_URL=http://localhost:1234   # LM Studio local server

# Vector Storage
VICTOR_VECTOR_STORE=lancedb               # lancedb (default), chromadb, proximadb
LANCEDB_URI=./data/lancedb                # LanceDB storage path

# Event Bus Backend
VICTOR_EVENT_BUS_BACKEND=in-memory        # in-memory (default), kafka, sqs, rabbitmq, redis
KAFKA_BOOTSTRAP_SERVERS=localhost:9092    # Kafka configuration

# Performance
VICTOR_LAZY_LOADING=true                 # Enable lazy loading (default: true)
VICTOR_CACHE_ENABLED=true                 # Enable caching (default: true)

# Code Validation
VICTOR_VALIDATION_ENABLED=true            # Enable code validation (default: true)
VICTOR_STRICT_VALIDATION=false            # Block writes on any error (default: false)
VICTOR_INDEXING_ENABLED=true              # Enable code indexing (default: true)

# Air-Gapped Mode
VICTOR_AIRGAPPED_MODE=false               # Air-gapped mode (local only, no web tools)

# API Server
VICTOR_API_HOST=0.0.0.0                   # API server host
VICTOR_API_PORT=8000                      # API server port
```

## Performance Notes

**Lazy Loading**: Verticals lazy-load by default (72.8% faster startup).
- Disable: `VICTOR_LAZY_LOADING=false`
- Affects: Vertical imports, tool registration, provider initialization

**Vector Storage**: LanceDB is the default vector store for optimal performance.
- Alternative: ChromaDB (`pip install victor-ai[vector-alt]`)
- Used for: Semantic code search, conversation embeddings

**Event Bus Backends**: Choose based on deployment scale:
- **In-Memory** (default): Single-instance, fastest
- **Kafka**: Distributed, exactly-once semantics
- **SQS**: Serverless, at-least-once, AWS-native
- **RabbitMQ**: Reliable, at-least-once
- **Redis**: Fast, at-least-once, simple setup

**Tool Selection**: Pluggable strategy for tool selection:
- `keyword`: Fast, exact match (default fallback)
- `semantic`: Embeddings-based, more flexible
- `hybrid`: Best of both (default, uses sentence-transformers)

## Troubleshooting

**Common Issues**:

1. **Singleton State Pollution**: Tests failing with unexpected state
   ```bash
   # Use the reset_singletons fixture (autouse in conftest.py)
   pytest tests/unit/test_file.py -v
   ```

2. **Provider Not Available**: `ModuleNotFoundError` for provider
   ```bash
   # Install optional provider dependencies
   pip install victor-ai[google]  # For Google Gemini
   ```

3. **Environment Variables Not Loaded**: API keys missing
   ```bash
   # Create .env file in project root
   echo "ANTHROPIC_API_KEY=sk-..." > .env
   ```

4. **Protocol Conformance Errors**: Type mismatches
   ```bash
   # Check protocol conformance
   make check-protocol
   ```

5. **YAML Config Validation Errors**: Invalid configuration
   ```bash
   # Validate all YAML configs
   make validate-config
   ```

6. **Docker Issues**: Tests requiring Docker fail
   ```python
   # Use auto_mock_docker_for_orchestrator fixture
   def test_with_docker_mock(auto_mock_docker_for_orchestrator):
       # Test logic here
       pass
   ```

7. **Memory Issues**: Large codebase analysis
   ```bash
   # Increase lazy loading for lower memory footprint
   export VICTOR_LAZY_LOADING=true
   ```

## Release and CI

**Pre-Release Checklist**:
```bash
# 1. Run comprehensive QA
make qa

# 2. Generate QA report
make qa-report

# 3. View release checklist
make release-checklist

# 4. Validate release readiness
make release-validate

# 5. Run security scans
make security-full
```

**CI/CD Requirements**: (from `.github/workflows/`)
- All unit tests pass (`make test`)
- Black formatting passes (`make lint`)
- Ruff linting passes
- MyPy type checking passes
- Trivy security scan passes (no critical/high vulnerabilities)
- Docker image builds successfully
- Python package builds successfully

**Version Bump**:
```bash
# Bump version in pyproject.toml
make release VERSION=0.5.1

# This will:
# - Update version in pyproject.toml
# - Commit with "Release v0.5.1"
# - Create git tag "v0.5.1"
# - Push to trigger release workflow
```

## Documentation

See `docs/` for comprehensive documentation:
- `MIGRATION_GUIDE.md` - Adopting new patterns
- `architecture/REFACTORING_OVERVIEW.md` - Architecture details
- `architecture/BEST_PRACTICES.md` - Usage patterns (protocol-first, DI, events, coordinators)
- `victor/framework/README.md` - Framework API documentation
- `architecture/coordinator_separation.md` - Two-layer coordinator design
- `DEVELOPER_ONBOARDING.md` - New developer start guide

## SOLID Improvement Patterns

Victor follows SOLID principles with specific patterns for maintainability and extensibility:

### Single Responsibility Principle (SRP)

**Pattern: Externalize Cross-Cutting Concerns**

Separate infrastructure concerns from business logic using dedicated services.

**Example: Vertical Integration Caching**
```python
# ❌ Before: Embedded caching (SRP violation)
class VerticalIntegrationPipeline:
    def __init__(self):
        self._cache: Dict[str, bytes] = {}  # Mixed concerns

    def apply(self, orchestrator, vertical, config):
        # Integration logic + cache management mixed
        cache_key = self._generate_cache_key(vertical, config)
        cached = self._load_from_cache(cache_key)
        # ... integration logic ...

# ✅ After: Externalized service (SRP compliant)
class VerticalIntegrationPipeline:
    def __init__(self, cache_service: VerticalIntegrationCache = None):
        self._cache_service = cache_service or VerticalIntegrationCache()

    def apply(self, orchestrator, vertical, config):
        # Cache logic delegated to service
        cached = self._cache_service.get(vertical, config)
        # ... integration logic ...
```

**Benefits**:
- Testability: Mock cache service in tests
- Flexibility: Swap for Redis/Memcached implementations
- Maintainability: Each service has single responsibility

**Usage**:
```python
from victor.core.cache import VerticalIntegrationCache

# DI-managed (recommended)
container.register(VerticalIntegrationCache, lambda c: VerticalIntegrationCache())

# Manual instantiation
cache = VerticalIntegrationCache(ttl=3600, enable_cache=True)
pipeline = VerticalIntegrationPipeline(cache_service=cache)
```

### Open/Closed Principle (OCP)

**Pattern: Declarative Classification**

Use properties for classification instead of brittle string matching.

**Example: Handler Independence**
```python
# ❌ Before: String matching (brittle)
def _classify_handlers(handlers):
    independent = []
    for handler in handlers:
        if type(handler).__name__ in ["ToolStepHandler", "PromptStepHandler"]:
            independent.append(handler)  # Must modify to add new handlers
    return independent

# ✅ After: Declarative property (extensible)
class BaseStepHandler:
    @property
    def is_independent(self) -> bool:
        return False  # Default: dependent

class ToolStepHandler(BaseStepHandler):
    @property
    def is_independent(self) -> bool:
        return True  # Declare independence

def _classify_handlers(handlers):
    independent = [h for h in handlers if h.is_independent]
    return independent  # No modification needed for new handlers
```

**Benefits**:
- Extensible: New handlers declare own independence
- Type-safe: No string matching
- Self-documenting: Property shows handler nature

**Creating Independent Handlers**:
```python
from victor.framework.step_handlers import BaseStepHandler

class CustomToolsHandler(BaseStepHandler):
    @property
    def name(self) -> str:
        return "custom_tools"

    @property
    def order(self) -> int:
        return 15

    @property
    def is_independent(self) -> bool:
        return True  # Can run in parallel with other independent handlers

    def _do_apply(self, orchestrator, vertical, context, result):
        # Handler logic
        pass
```

### Dependency Inversion Principle (DIP)

**Pattern: Protocol-Based Dependency Injection**

Depend on abstractions (protocols), not concrete implementations.

**Example: Cache Service Injection**
```python
from typing import Protocol

class ICache(Protocol):
    def get(self, key: str) -> Optional[Any]: ...
    def set(self, key: str, value: Any) -> None: ...

class VerticalIntegrationPipeline:
    def __init__(self, cache: ICache):  # Depends on abstraction
        self._cache = cache

# Usage with different implementations
in_memory_cache = InMemoryCache()
pipeline1 = VerticalIntegrationPipeline(in_memory_cache)

redis_cache = RedisCache()
pipeline2 = VerticalIntegrationPipeline(redis_cache)  # Same interface
```

**Benefits**:
- Swappable: Easy to substitute implementations
- Testable: Mock dependencies in tests
- Decoupled: No tight coupling to concrete classes

### Vertical Scaffolding

**Enhanced CLI for Production-Ready Verticals**

The vertical scaffolding CLI generates complete, production-ready verticals with all necessary components:

```bash
# Generate full vertical with all components
victor vertical create myvertical --description "My vertical" --with-all

# Selective component generation
victor vertical create myvertical --with-tests --with-packaging

# Available flags:
--with-tests          # Generate test files (tests/test_assistant.py, tests/conftest.py)
--with-packaging      # Generate pyproject.toml for external distribution
--with-docs           # Generate README.md documentation
--with-workflows      # Generate example workflow template
--with-all            # Generate all optional components (recommended for production)
```

**Generated Structure**:
```
victor/myvertical/
├── __init__.py              # Package exports
├── assistant.py             # Main vertical class
├── safety.py                # Safety patterns
├── prompts.py               # System prompts
├── mode_config.py           # Mode configurations
├── service_provider.py      # DI registration (optional)
├── tests/
│   ├── test_assistant.py    # Test suite
│   └── conftest.py          # Pytest fixtures
├── pyproject.toml           # Packaging (external verticals)
├── README.md                # Documentation
└── workflows/
    └── example.yaml         # Example workflow
```

## Anti-Patterns to Avoid

**God Object**: One class doing everything. Split into focused coordinators with single responsibilities.

**Service Locator**: Using container inside components. Use constructor injection with explicit protocol dependencies.

**Tight Coupling Through Events**: Using events for synchronous request/response. Use direct calls or proper async patterns instead.

**Over-Abstracting**: Too many layers of abstraction (e.g., IToolExecutorFactoryCreatorFactory). Use simple factories or DI directly.

**Circular Dependencies**: Services depending on each other. Break cycles with events or third service.

See `docs/architecture/BEST_PRACTICES.md` for detailed anti-patterns and solutions.