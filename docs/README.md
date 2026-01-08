# Victor Documentation

Complete documentation for Victor AI Assistant - the provider-agnostic coding assistant that supports 21 LLM providers (local and cloud) through a unified CLI/TUI interface.

---

## Quick Start

**New to Victor?** Start here:

1. **Install**: `pipx install victor-ai`
2. **Run**: `victor` (TUI mode) or `victor chat "your task"` (CLI mode)
3. **Learn**: Browse documentation below

[Full Getting Started Guide →](getting-started/)

---

## Documentation by Audience

### For Users

Daily usage, configuration, troubleshooting, and reference materials.

| Documentation | Description |
|--------------|-------------|
| [**Getting Started**](getting-started/) | Install, configure, and run Victor for the first time |
| [**User Guide**](user-guide/) | Daily usage patterns, commands, modes, workflows |
| [**Troubleshooting**](user-guide/troubleshooting.md) | Common issues and solutions |

### For Developers

Contributing to Victor, understanding the codebase, extending functionality.

| Documentation | Description |
|--------------|-------------|
| [**Development Guide**](development/) | Setup, testing, patterns, contributing |
| [**Architecture**](development/architecture/) | System design, patterns, data flow |
| [**Extending**](development/extending/) | Add providers, tools, verticals, workflows |

### For Architects

Deep technical evaluation, design decisions, scalability, and operations.

| Documentation | Description |
|--------------|-------------|
| [**Architecture Deep Dive**](development/architecture/deep-dive.md) | Comprehensive system internals |
| [**Design Patterns**](development/architecture/design-patterns.md) | SOLID principles, coordinator pattern |
| [**Operations**](operations/) | Deployment, monitoring, security, performance |

### For Operators

Production deployment, monitoring, and maintenance.

| Documentation | Description |
|--------------|-------------|
| [**Deployment**](operations/deployment/) | Docker, Kubernetes, cloud platforms |
| [**Monitoring**](operations/monitoring/) | Metrics, logging, health checks |
| [**Security**](operations/security/) | Key management, compliance, auditing |

---

## Documentation by Topic

### Core Concepts

Understanding Victor's architecture and capabilities.

| Topic | Documentation |
|-------|--------------|
| **Providers** | [Provider Reference →](reference/providers/) - All 21 LLM providers |
| **Tools** | [Tool Catalog →](reference/tools/) - 55+ tools and composition |
| **Verticals** | [Verticals →](reference/verticals/) - 5 domain assistants |
| **Workflows** | [Workflow DSL →](guides/workflow-development/) - YAML automation |
| **Multi-Agent** | [Teams →](guides/multi-agent/) - Agent coordination |

### Configuration

Setting up Victor for your environment.

| Topic | Documentation |
|-------|--------------|
| **Profiles** | [profiles.yaml →](reference/configuration/profiles.md) - Provider/model configs |
| **API Keys** | [Key Management →](reference/configuration/keys.md) - Secure key storage |
| **MCP** | [MCP Config →](reference/configuration/mcp.md) - Model Context Protocol |
| **Environment** | [Config Reference →](reference/configuration/) - Complete reference |

### Integration

Using Victor with other tools and platforms.

| Topic | Documentation |
|-------|--------------|
| **HTTP API** | [API Reference →](reference/api/http-api.md) - victor serve endpoints |
| **MCP Servers** | [MCP Protocol →](reference/api/mcp-server.md) - victor mcp |
| **CI/CD** | [CI/CD Integration →](guides/integration/ci-cd.md) - GitHub Actions, etc. |
| **VS Code** | [VS Code Setup →](guides/integration/vscode-extension.md) - Extension usage |
| **MCP Clients** | [MCP Clients →](guides/integration/mcp-clients.md) - Using MCP servers |

### Advanced Features

Power user capabilities and workflows.

| Topic | Documentation |
|-------|--------------|
| **Observability** | [EventBus →](guides/observability/event-bus.md) - Events and metrics |
| **Workflows** | [Workflow DSL →](guides/workflow-development/dsl.md) - StateGraph YAML |
| **Scheduling** | [Scheduler →](guides/workflow-development/scheduling.md) - Cron workflows |
| **Teams** | [Formations →](guides/multi-agent/teams.md) - Team patterns |
| **Performance** | [Performance →](guides/performance.md) - Optimization and tuning |
| **Security** | [Security →](guides/security.md) - Best practices |
| **Resilience** | [Resilience →](guides/resilience.md) - Error handling patterns |

---

## Visual Guides

Diagrams, flowcharts, and visual documentation.

| Guide | Description |
|-------|-------------|
| [**Architecture Diagrams**](diagrams/architecture/) | System, data flow, components |
| [**Workflow Diagrams**](diagrams/workflows/) | 55 workflow visualizations (SVG) |
| [**Sequence Diagrams**](diagrams/sequences/) | Tool execution, provider switching |

---

## Resources

Additional learning resources and community content.

| Resource | Description |
|----------|-------------|
| [**Examples**](resources/examples/) | Code examples and snippets |
| [**Blog Posts**](resources/blog/) | Articles and tutorials |
| [**Community**](resources/community/) | Discussions, Discord, Stack Overflow |

---

## Key Reference Documentation

### Providers

Complete reference for all 21 supported LLM providers.

**Local Providers:**
- [Ollama](reference/providers/local/ollama.md) - Local model execution
- [LM Studio](reference/providers/local/lm-studio.md) - Local model UI
- [vLLM](reference/providers/local/vllm.md) - Fast local inference
- [llama.cpp](reference/providers/local/llama-cpp.md) - Lightweight inference

**Cloud Providers:**
- [Anthropic](reference/providers/cloud/anthropic.md) - Claude models
- [OpenAI](reference/providers/cloud/openai.md) - GPT models
- [Google](reference/providers/cloud/google.md) - Gemini models
- [Azure](reference/providers/cloud/azure.md) - Azure OpenAI
- [And 17 more...](reference/providers/)

[Full Provider Reference →](reference/providers/)

### Tools

55+ tools organized by category.

**Categories:**
- [File Operations](reference/tools/) - Read, write, edit files
- [Git](reference/tools/) - Version control operations
- [Testing](reference/tools/) - Test execution and coverage
- [Search](reference/tools/) - Code and semantic search
- [Web](reference/tools/) - Web requests and scraping
- [And 8 more categories...](reference/tools/)

[Full Tool Catalog →](reference/tools/catalog.md)

### Configuration

Complete configuration reference.

| File | Purpose | Reference |
|------|---------|----------|
| `profiles.yaml` | Provider/model profiles | [profiles.md →](reference/configuration/profiles.md) |
| `config.yaml` | Global settings | [config.md →](reference/configuration/) |
| `mcp.yaml` | MCP server configuration | [mcp.md →](reference/configuration/mcp.md) |
| `.victor.md` | Project context | [project-context.md →](user-guide/project-context.md) |

---

## Development Documentation

### Contributing

We welcome contributions! See the [Contributing Guide](CONTRIBUTING.md) for details.

**Quick Start:**
```bash
git clone https://github.com/vjsingh1984/victor.git
cd victor
pip install -e ".[dev]"
make test
```

**Resources:**
- [Development Setup →](development/setup/)
- [Testing Strategy →](development/testing/)
- [Code Style →](development/contributing/code-style.md)
- [Pull Requests →](development/contributing/pull-requests.md)

### Architecture

Understanding Victor's internal design.

| Topic | Documentation |
|-------|--------------|
| **Overview** | [High-level architecture →](development/architecture/overview.md) |
| **Deep Dive** | [System internals →](development/architecture/deep-dive.md) (900+ lines) |
| **Patterns** | [SOLID design →](development/architecture/design-patterns.md) |
| **State Machine** | [Conversation stages →](development/architecture/state-machine.md) |
| **Data Flow** | [Request/response flow →](development/architecture/data-flow.md) |
| **Components** | [Component details →](development/architecture/component-details.md) |

### Extending Victor

Adding custom functionality.

| Topic | Documentation |
|-------|--------------|
| **Providers** | [Add providers →](development/extending/providers.md) |
| **Tools** | [Add tools →](development/extending/tools.md) |
| **Verticals** | [Create verticals →](development/extending/verticals.md) |
| **Workflows** | [Custom workflows →](development/extending/workflows.md) |

---

## Operations Documentation

### Deployment

Running Victor in production.

| Platform | Documentation |
|----------|--------------|
| **Docker** | [Docker deployment →](operations/deployment/docker.md) |
| **Kubernetes** | [K8s deployment →](operations/deployment/kubernetes.md) |
| **Cloud** | [Cloud platforms →](operations/deployment/cloud.md) |
| **Air-Gapped** | [Offline mode →](operations/deployment/air-gapped.md) |

### Monitoring & Security

Operating Victor at scale.

| Topic | Documentation |
|-------|--------------|
| **Metrics** | [Metrics & alerting →](operations/monitoring/metrics.md) |
| **Logging** | [Logging config →](operations/monitoring/logging.md) |
| **Health** | [Health checks →](operations/monitoring/health-checks.md) |
| **Security** | [Security overview →](operations/security/) |
| **Compliance** | [SOC2, GDPR →](operations/security/compliance.md) |
| **Performance** | [Optimization →](operations/performance/) |

---

## Search & Navigation

**Looking for something specific?**

- **New users?** Start with [Getting Started](getting-started/)
- **Daily use?** See [User Guide](user-guide/)
- **Developing?** Check [Development Guide](development/)
- **Architecture?** Read [Architecture Deep Dive](development/architecture/deep-dive.md)
- **Providers?** View [Provider Reference](reference/providers/)
- **Tools?** Browse [Tool Catalog](reference/tools/catalog.md)
- **Config?** Read [Configuration Reference](reference/configuration/)
- **Stuck?** Try [Troubleshooting](user-guide/troubleshooting.md)

---

## Community & Support

- **Discord**: [Join our Discord community](https://discord.gg/...)
- **GitHub Discussions**: [Start a discussion](https://github.com/vjsingh1984/victor/discussions)
- **Issues**: [Report bugs or request features](https://github.com/vjsingh1984/victor/issues)
- **Stack Overflow**: Tag questions with `victor-ai`

---

## Additional Resources

- **Changelog**: See [CHANGELOG.md](../CHANGELOG.md) for version history
- **License**: See [LICENSE](../LICENSE) for Apache 2.0 license
- **Contributing**: See [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guide
- **Security**: See [SECURITY.md](../SECURITY.md) for security policy

---

<div align="center">

**[← Back to Repository](../README.md)**

**Victor Documentation**

*Provider-agnostic coding assistant for developers*

</div>
