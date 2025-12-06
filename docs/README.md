# Victor Documentation

Comprehensive documentation for Victor - Enterprise-Ready AI Coding Assistant.

## Quick Links

| Document | Description |
|----------|-------------|
| [Installation](guides/INSTALLATION.md) | Complete installation guide |
| [Quick Start](guides/QUICKSTART.md) | First steps with Victor |
| [User Guide](USER_GUIDE.md) | Complete usage guide |
| [Tool Catalog](TOOL_CATALOG.md) | All 54+ available tools |
| [Model Comparison](MODEL_COMPARISON.md) | Ollama model benchmarks |
| [Releasing](RELEASING.md) | Release process for maintainers |

## Documentation Structure

```
docs/
├── USER_GUIDE.md           # Comprehensive user guide
├── TOOL_CATALOG.md         # All 54+ tools with examples
├── MODEL_COMPARISON.md     # Ollama model benchmarks
├── DEVELOPER_GUIDE.md      # Contributing and development
├── RELEASING.md            # Release process for maintainers
├── ENTERPRISE.md           # Enterprise deployment
│
├── guides/                 # How-to guides
│   ├── INSTALLATION.md     # Complete installation guide
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
│   ├── README.md           # Embeddings overview
│   ├── ARCHITECTURE.md     # Embedding architecture
│   ├── SETUP.md            # Setup guide
│   ├── AIRGAPPED.md        # Air-gapped operation
│   ├── TOOL_SELECTION.md   # Semantic tool selection
│   ├── TOOL_CALLING_FORMATS.md # Tool calling formats
│   └── MODELS.md           # Embedding model options
│
├── design/                 # Design documents
│   ├── ARGUMENT_NORMALIZATION_DESIGN.md
│   └── AUTHENTICATION_DESIGN_SPEC.md
│
├── ARCHITECTURE_DEEP_DIVE.md  # System architecture
└── TESTING_STRATEGY.md        # Testing approach
```

## By Topic

### Getting Started
- [Installation Guide](guides/INSTALLATION.md) - All installation methods
- [Quick Start Guide](guides/QUICKSTART.md) - First steps
- [User Guide](USER_GUIDE.md) - Complete usage
- [Configuration](guides/QUICKSTART.md#configuration)

### Using Victor
- [Interactive Mode](USER_GUIDE.md#interactive-mode)
- [One-shot Commands](USER_GUIDE.md#one-shot-mode)
- [Tool Catalog](TOOL_CATALOG.md) - 54+ tools across 8 categories

### Models & Providers
- [Model Comparison](MODEL_COMPARISON.md) - Tested Ollama benchmarks
- [Provider Reference](reference/PROVIDERS.md)
- [Tool Calling](reference/TOOL_CALLING.md)

### Air-gapped & Offline
- [Air-gapped Setup](embeddings/AIRGAPPED.md)
- [Docker Deployment](../docker/README.md)

### Development
- [Developer Guide](DEVELOPER_GUIDE.md)
- [Plugin Development](guides/PLUGIN_GUIDE.md)
- [MCP Integration](guides/MCP_GUIDE.md)

### Releasing & Distribution
- [Releasing Guide](RELEASING.md) - Creating releases
- [Installation Methods](guides/INSTALLATION.md#installation-methods)

### Architecture
- [Architecture Deep Dive](ARCHITECTURE_DEEP_DIVE.md)
- [Embedding System](embeddings/ARCHITECTURE.md)

## Docker Documentation

See [docker/README.md](../docker/README.md) for Docker-specific documentation:
- Quick Reference: `docker/QUICKREF.md`
- Advanced Setup: `docker/ADVANCED.md`
