# Victor Documentation

Comprehensive documentation for Victor - Enterprise-Ready AI Coding Assistant.

## Quick Links

| Document | Description |
|----------|-------------|
| [Installation](guides/INSTALLATION.md) | Complete installation guide |
| [Quick Start](guides/QUICKSTART.md) | First steps with Victor |
| [First Run](guides/FIRST_RUN.md) | 2-minute onboarding walkthrough |
| [Local Models](guides/LOCAL_MODELS.md) | Local setup presets and tips |
| [Provider Setup](guides/PROVIDER_SETUP.md) | Configure 21 LLM providers |
| [User Guide](USER_GUIDE.md) | Complete usage guide |
| [Tool Catalog](TOOL_CATALOG.md) | All 55 available tools |
| [Verticals](VERTICALS.md) | 5 domain-specific assistant templates |
| [Framework Migration](FRAMEWORK_MIGRATION.md) | Migrating to Framework API |
| [Releasing](RELEASING.md) | Release process for maintainers |

## Documentation Structure

```
docs/
├── USER_GUIDE.md              # Comprehensive user guide
├── TOOL_CATALOG.md            # All 55 tools with descriptions
├── MODEL_COMPARISON.md        # Ollama model benchmarks
├── DEVELOPER_GUIDE.md         # Contributing and development
├── RELEASING.md               # Release process
├── ENTERPRISE.md              # Enterprise deployment
├── VERTICALS.md               # Domain-specific assistants
├── STATE_MACHINE.md           # Conversation state tracking
├── FRAMEWORK_MIGRATION.md     # Framework API migration guide
├── CORE_PATTERNS.md           # Design patterns used
│
├── guides/                    # How-to guides
│   ├── INSTALLATION.md        # Installation methods
│   ├── QUICKSTART.md          # Step-by-step quickstart
│   ├── PROVIDER_SETUP.md      # 21 provider configurations
│   ├── MCP_GUIDE.md           # MCP server setup
│   ├── PLUGIN_GUIDE.md        # Writing plugins
│   ├── OLLAMA_TOOL_SUPPORT.md # Ollama tool calling
│   └── MODEL_SHARING.md       # Sharing models
│
├── reference/                 # Technical reference
│   ├── PROVIDERS.md           # Provider API reference
│   ├── TOOL_CALLING.md        # Tool calling internals
│   └── TOOL_MIGRATION.md      # Migrating tools
│
├── embeddings/                # Embedding system docs
│   ├── README.md              # Overview
│   ├── ARCHITECTURE.md        # Embedding architecture
│   ├── SETUP.md               # Setup guide
│   ├── AIRGAPPED.md           # Air-gapped operation
│   ├── TOOL_SELECTION.md      # Semantic tool selection
│   └── MODELS.md              # Embedding model options
│
├── design/                    # Active design documents
│   ├── ANTHROPIC_PROVIDER_IMPROVEMENTS.md
│   ├── ARGUMENT_NORMALIZATION_DESIGN.md
│   └── AUTHENTICATION_DESIGN_SPEC.md
│
├── assets/                    # SVG infographics
│   ├── architecture-overview.svg
│   ├── provider-comparison.svg
│   └── provider-quickstart.svg
│
├── archive/                   # Historical/completed docs
│
├── ARCHITECTURE_DEEP_DIVE.md  # System architecture
├── CODEBASE_ANALYSIS_REPORT.md # Current state analysis
└── TESTING_STRATEGY.md        # Testing approach
```

## By Topic

### Getting Started
- [Installation Guide](guides/INSTALLATION.md) - All installation methods
- [Quick Start Guide](guides/QUICKSTART.md) - First steps
- [User Guide](USER_GUIDE.md) - Complete usage
- [Configuration](guides/QUICKSTART.md#configuration)

### Benchmarking & Evaluation
- [Benchmark Evaluation](BENCHMARK_EVALUATION.md) - SWE-bench, HumanEval, MBPP
- [Evaluation Flow Diagram](assets/evaluation-flow.svg) - Token tracking architecture
- Running benchmarks: `victor benchmark run swe-bench --profile <profile>`

### Providers & Models
- [Provider Setup](guides/PROVIDER_SETUP.md) - Configure 21 providers
- [Local Models](guides/LOCAL_MODELS.md) - Ollama, LM Studio, vLLM, llama.cpp
- [Model Comparison](MODEL_COMPARISON.md) - Tested Ollama benchmarks
- [Provider Reference](reference/PROVIDERS.md) - API reference
- [Tool Calling](reference/TOOL_CALLING.md) - How tool calling works

### Using Victor
- [Interactive Mode](USER_GUIDE.md#interactive-mode)
- [One-shot Commands](USER_GUIDE.md#one-shot-mode)
- [Tool Catalog](TOOL_CATALOG.md) - 55 tools across 8 categories

### Air-gapped & Offline
- [Air-gapped Setup](embeddings/AIRGAPPED.md)
- [Docker Deployment](../docker/README.md)

### Development
- [Developer Guide](DEVELOPER_GUIDE.md)
- [Plugin Development](guides/PLUGIN_GUIDE.md)
- [MCP Integration](guides/MCP_GUIDE.md)
- [Testing Strategy](TESTING_STRATEGY.md)

### Architecture
- [Architecture Roadmap](ARCHITECTURE_ROADMAP.md) - Improvement phases and technical debt
- [Architecture Deep Dive](ARCHITECTURE_DEEP_DIVE.md)
- [State Machine](STATE_MACHINE.md) - Conversation state tracking
- [Verticals](VERTICALS.md) - Domain-specific assistants
- [Framework Migration](FRAMEWORK_MIGRATION.md) - Framework API guide
- [Core Patterns](CORE_PATTERNS.md) - Design patterns used
- [Embedding System](embeddings/ARCHITECTURE.md)
- [Codebase Analysis](CODEBASE_ANALYSIS_REPORT.md)

### Releasing & Distribution
- [Releasing Guide](RELEASING.md) - Creating releases
- [Installation Methods](guides/INSTALLATION.md#installation-methods)

## Docker Documentation

See [docker/README.md](../docker/README.md) for Docker-specific documentation:
- Quick Reference: `docker/QUICKREF.md`
- Advanced Setup: `docker/ADVANCED.md`

## Archive

Historical and completed design documents are in [archive/](archive/).
