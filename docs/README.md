# Victor Documentation

Comprehensive documentation for Victor - Enterprise-Ready AI Coding Assistant.

## Quick Links

| Document | Description |
|----------|-------------|
| [Getting Started](GETTING_STARTED.md) | Installation and first steps |
| [User Guide](USER_GUIDE.md) | Complete usage guide |
| [Tool Catalog](TOOL_CATALOG.md) | All 54 available tools |
| [Model Comparison](MODEL_COMPARISON.md) | Ollama model benchmarks |

## Documentation Structure

```
docs/
├── GETTING_STARTED.md      # Quick start guide
├── USER_GUIDE.md           # Comprehensive user guide
├── TOOL_CATALOG.md         # All 54 tools with examples
├── MODEL_COMPARISON.md     # Ollama model benchmarks
├── DEVELOPER_GUIDE.md      # Contributing and development
├── ENTERPRISE.md           # Enterprise deployment
│
├── guides/                 # How-to guides
│   ├── QUICKSTART.md       # Step-by-step quickstart
│   ├── MCP_GUIDE.md        # MCP server setup
│   ├── PLUGIN_GUIDE.md     # Writing plugins
│   └── MODEL_SHARING.md    # Sharing models
│
├── reference/              # Technical reference
│   ├── PROVIDERS.md        # Provider API docs
│   ├── TOOL_CALLING.md     # Tool calling internals
│   └── TOOL_MIGRATION.md   # Migrating tools
│
├── embeddings/             # Embedding system docs
│   ├── ARCHITECTURE.md     # Embedding architecture
│   ├── SETUP.md            # Setup guide
│   ├── AIRGAPPED.md        # Air-gapped operation
│   ├── AIRGAPPED_GUIDE.md  # Deployment guide
│   ├── TOOL_SELECTION.md   # Semantic tool selection
│   └── MODELS.md           # Embedding model options
│
├── design/                 # Design documents (historical)
│   ├── ARGUMENT_NORMALIZATION_DESIGN.md
│   └── AUTHENTICATION_DESIGN_SPEC.md
│
├── ARCHITECTURE_DEEP_DIVE.md  # System architecture
├── CODEBASE_ANALYSIS_REPORT.md # Technical debt analysis
└── TESTING_STRATEGY.md        # Testing approach
```

## By Topic

### Getting Started
- [Installation](GETTING_STARTED.md#installation)
- [Quick Start Guide](guides/QUICKSTART.md)
- [Configuration](GETTING_STARTED.md#configuration)

### Using Victor
- [Interactive Mode](USER_GUIDE.md#interactive-mode)
- [One-shot Commands](USER_GUIDE.md#one-shot-mode)
- [Tool Catalog](TOOL_CATALOG.md) - 54 tools across 8 categories

### Models & Providers
- [Model Comparison](MODEL_COMPARISON.md) - Tested Ollama benchmarks
- [Provider Reference](reference/PROVIDERS.md)
- [Tool Calling](reference/TOOL_CALLING.md)

### Air-gapped & Offline
- [Air-gapped Setup](embeddings/AIRGAPPED.md)
- [Deployment Guide](embeddings/AIRGAPPED_GUIDE.md)
- [Docker Deployment](../docker/README.md)

### Development
- [Developer Guide](DEVELOPER_GUIDE.md)
- [Plugin Development](guides/PLUGIN_GUIDE.md)
- [MCP Integration](guides/MCP_GUIDE.md)

### Architecture
- [Architecture Deep Dive](ARCHITECTURE_DEEP_DIVE.md)
- [Embedding System](embeddings/ARCHITECTURE.md)
- [Codebase Analysis](CODEBASE_ANALYSIS_REPORT.md)

## Docker Documentation

See [docker/README.md](../docker/README.md) for Docker-specific documentation:
- Quick Reference: `docker/QUICKREF.md`
- Advanced Setup: `docker/ADVANCED.md`
