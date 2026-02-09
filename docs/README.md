# <div align="center">


# Victor AI Documentation

**Provider-agnostic AI coding assistant with 21 LLM providers, 55+ tools, and 5 domain verticals**

[![Version](https://badge.fury.io/py/victor-ai.svg)](https://pypi.org/project/victor-ai/)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://github.com/vjsingh1984/victor)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/vjsingh1984/victor/blob/main/LICENSE)

</div>

---

## 🎯 Quick Start

Choose your path based on your goals:

| Journey | For | Time | Outcome |
|---------|-----|------|---------|
| [🚀 Beginner](journeys/beginner.md) | New users | 30 min | Ready for daily use |
| [💡 Intermediate](journeys/intermediate.md) | Daily users | 80 min | Power user |
| [👨‍💻 Developer](journeys/developer.md) | Contributors | 2 hours | Active contributor |
| [🔧 Operations](journeys/operations.md) | DevOps/SRE | 80 min | Production deployment |
| [🏗️ Advanced](journeys/advanced.md) | Architects | 2.5 hours | System expert |

→ [View All Journeys](journeys/index.md)

---

## 📚 What is Victor?

**Victor AI** is an open-source AI coding assistant that provides:

### Key Features

- **🔄 Provider-Agnostic:** 21 LLM providers (Anthropic, OpenAI, Google, Ollama, etc.)
- **🔀 Provider Switching:** Switch providers mid-conversation without losing context
- **🛠️ 55+ Tools:** File operations, git, testing, web search, and more
- **📦 5 Verticals:** Coding, DevOps, RAG, Data Analysis, Research
- **⚙️ YAML Workflows:** Multi-step automation with StateGraph execution
- **🎯 Two-Layer Architecture:** Application + Framework layers for extensibility

### Architecture Highlights

```text
┌─────────────────────────────────────────────────┐
│          Clients: CLI/TUI, HTTP, MCP            │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│         AgentOrchestrator (Facade)              │
└──────────────────┬──────────────────────────────┘
                   │
       ┌───────────┼───────────┐
       │           │           │
┌──────▼───┐  ┌───▼────┐  ┌──▼─────┐
│  App Layer │Framework │  Core   │
│10 Coordinators│10 Coordinators│Services│
└──────────┘  └─────────┘  └────────┘
```

**See:** [Architecture Overview](architecture/overview.md) | [Coordinator
  Layers](diagrams/architecture/coordinator-layers.mmd)

---

## 🚀 Getting Started

### Installation

**Quick Install (pipx):**
```bash
pipx install victor-ai
```text

**Docker:**
```bash
docker pull ghcr.io/vjsingh1984/victor:latest
docker run -it ghcr.io/vjsingh1984/victor:latest
```

**Development:**
```bash
git clone https://github.com/vjsingh1984/victor.git
cd victor
pip install -e ".[dev]"
```text

### First Run

**Local (No API Key):**
```bash
# Install Ollama
brew install ollama  # macOS
ollama serve
ollama pull qwen2.5-coder:7b

# Start Victor
victor chat --provider ollama
```

**Cloud (API Key):**
```bash
# Set API key
export ANTHROPIC_API_KEY=sk-ant-your-key

# Start Victor
victor chat --provider anthropic "Help me refactor this code"
```text

**See:** [Installation Guide](getting-started/installation.md) | [First Run](getting-started/first-run.md)

---

## 📖 Documentation

### By Content Type

#### 📝 Tutorials (Learning-Oriented)
- [Beginner Journey](journeys/beginner.md) - 30-minute onboarding
- [Developer Journey](journeys/developer.md) - Contributor training
- [Creating Workflows](guides/tutorials/CREATING_WORKFLOWS.md)
- [Creating Tools](guides/tutorials/CREATING_TOOLS.md)
- [Creating Verticals](guides/tutorials/CREATING_VERTICALS.md)

#### 🛠️ How-to Guides (Problem-Oriented)
- [Workflows Guide](guides/workflows/) - Multi-step automation
- [Multi-Agent Teams](guides/multi-agent-teams.md) - Team collaboration
- [Coordinator Recipes](guides/tutorials/coordinator_recipes.md) - Usage patterns
- [Provider Switching](reference/features/multi_provider_workflows.md) - Switch contexts

#### 📋 Reference (Information-Oriented)
- [API Reference](reference/api/) - Internal and external APIs
- [Configuration Reference](reference/configuration/) - All config options
- [Providers](reference/providers/) - 21 provider implementations
- [Tools Catalog](reference/tools/catalog.md) - 55+ tools
- [Errors](reference/errors.md) - Error handling

#### 💡 Explanation (Understanding-Oriented)
- [Architecture Overview](architecture/overview.md) - System design
- [Design Patterns](architecture/patterns/) - SOLID patterns used
- [Best Practices](architecture/best-practices/) - Usage guidelines
- [Protocols Reference](architecture/PROTOCOLS_REFERENCE.md) - Interface definitions

### By Topic

#### 🏗️ Architecture & Design
- [Architecture Overview](architecture/overview.md) - System architecture
- [Two-Layer Coordinators](architecture/coordinator_separation.md) - Application vs Framework
- [Protocols & Events](architecture/protocols-and-events.mmd) - Communication patterns
- [Dependency Injection](architecture/dependency-injection.md) - DI container

#### 👨‍💻 Development
- [Contributing Index](contributing/) - Contributor hub
- [Development Setup](contributing/setup.md) - Environment setup
- [Testing Guide](contributing/testing.md) - Test strategy
- [Code Style](contributing/code-style.md) - Formatting rules

#### 🔧 Operations
- [Operations Guide](operations/) - DevOps/SRE hub
- [Docker Deployment](operations/deployment/docker.md) - Container deployment
- [Kubernetes Deployment](operations/deployment/kubernetes.md) - K8s deployment
- [Monitoring](operations/observability/) - Observability stack
- [Security](operations/security/) - Security and compliance

#### 📦 Verticals
- [Coding Vertical](reference/verticals/coding.md) - Code generation, refactoring
- [DevOps Vertical](reference/verticals/devops.md) - CI/CD, containers
- [RAG Vertical](reference/verticals/rag.md) - Document retrieval
- [Data Analysis Vertical](reference/verticals/data-analysis.md) - Pandas, visualization
- [Research Vertical](reference/verticals/research.md) - Literature search

---

## 🎨 Visual Documentation

Victor includes 20+ Mermaid diagrams for visual learning:

### Architecture Diagrams
- [Coordinator Layers](diagrams/architecture/coordinator-layers.mmd) - Two-layer architecture
- [Data Flow](diagrams/architecture/data-flow.mmd) - Request/response flow
- [Provider Switching](diagrams/architecture/provider-switch-detailed.mmd) - Context preservation
- [Extension System](diagrams/architecture/extension-system.mmd) - Extensibility
- [Tool Execution](diagrams/architecture/tool-execution-detailed.mmd) - Execution flow

### Operations Diagrams
- [Deployment Patterns](diagrams/operations/deployment.mmd) - Local, cloud, hybrid

### User Journey Diagrams
- [Beginner Onboarding](diagrams/user-journeys/beginner-onboarding.mmd) - 30-minute journey
- [Contributor Workflow](diagrams/user-journeys/contributor-workflow.mmd) - Contribution process

**See:** [Diagram Index](diagrams/README.md)

---

## 🎯 Use Cases

### For Individual Developers

**Code Generation:**
```
You: Create a FastAPI endpoint for user authentication with JWT tokens
[Victor generates code with best practices]
```text

**Refactoring:**
```
You: Refactor this function to use the Strategy pattern
[Victor applies pattern with tests]
```text

**Testing:**
```
You: Write unit tests for the UserService class
[Victor generates comprehensive tests]
```text

### For Teams

**Code Reviews:**
```
You: Review this PR and suggest improvements
[Victor analyzes and provides actionable feedback]
```text

**Multi-Agent Collaboration:**
```bash
victor chat --team code-quality-team
[Multiple specialists collaborate: reviewer, tester, optimizer]
```

**Workflow Automation:**
```yaml
# workflows/pr-check.yaml
name: PR Check
steps:
  - name: Run tests
    tool: pytest
  - name: Review code
    tool: code_review
    depends_on: [Run tests]
```text

### For DevOps/SRE

**Deployment:**
```bash
victor chat "Help me create a Kubernetes deployment for Victor"
[Victor generates YAML manifests with best practices]
```

**Monitoring:**
```bash
victor chat "Set up Prometheus monitoring for Victor API"
[Victor creates dashboards and alerts]
```text

---

## 🌟 Highlights

### Provider Switching

Switch providers mid-conversation without losing context:

```
You: switch to openai
[Victor switches to OpenAI, maintaining conversation history]
```text

**Supported Providers:** 21 providers across cloud and local
- **Cloud:** Anthropic, OpenAI, Google, Azure, AWS Bedrock, Cohere, etc.
- **Local:** Ollama, LM Studio, vLLM, Llama.cpp
- **See:** [Provider Reference](reference/providers/)

### Workflows

Automate multi-step processes with YAML:

```yaml
name: Test and Fix
steps:
  - name: Run tests
    tool: pytest
  - name: Fix failures
    tool: refactor
    depends_on: [Run tests]
```

**See:** [Workflows Guide](guides/workflows/)

### Multi-Agent Teams

Collaborative AI agents for complex tasks:

```yaml
team: code-quality
agents:
  - name: reviewer
    role: Code Reviewer
  - name: tester
    role: QA Engineer
  - name: optimizer
    role: Performance Engineer
```text

**See:** [Multi-Agent Teams](guides/multi-agent-teams.md)

---

## 🤝 Contributing

We welcome contributions! See [Contributing Guide](contributing/):

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run `make test` and `make lint`
6. Submit a pull request

**Good First Issues:**
- [github.com/vjsingh1984/victor/labels/good%20first%20issue](https://github.com/vjsingh1984/victor/labels/good%20first%20issue)

**Feature Proposals:**
- [FEP Process](contributing/FEP_PROCESS.md)

---

## 📊 Statistics

| Metric | Value |
|--------|-------|
| **Providers** | 21 (cloud + local) |
| **Tools** | 55+ specialized tools |
| **Verticals** | 5 domain verticals |
| **Protocols** | 98 protocols |
| **Coordinators** | 20 (10 app + 10 framework) |
| **Test Coverage** | 92.13% |
| **Services** | 55+ in DI container |
| **Documentation Files** | 287 markdown files |
| **Diagrams** | 20+ Mermaid + 60+ SVG |

---

## 🔗 Resources

### Official
- **GitHub:** [github.com/vjsingh1984/victor](https://github.com/vjsingh1984/victor)
- **PyPI:** [pypi.org/project/victor-ai](https://pypi.org/project/victor-ai/)
- **Documentation:** [This site](index.md)
- **Contributing:** [CONTRIBUTING.md](../CONTRIBUTING.md)

### Community
- **Issues:** [github.com/vjsingh1984/victor/issues](https://github.com/vjsingh1984/victor/issues)
- **Discussions:** [github.com/vjsingh1984/victor/discussions](https://github.com/vjsingh1984/victor/discussions)
- **Changelog:** [CHANGELOG.md](../CHANGELOG.md)

### Related
- [Architecture Decision Records](architecture/ADR/)
- [Design Documents](archive/design-docs/)
- [Performance Benchmarks](operations/performance/)

---

## 📖 Reading Guide

### New to Victor?
Start with the [Beginner Journey](journeys/beginner.md) (30 minutes).

### Daily User?
Explore the [Intermediate Journey](journeys/intermediate.md) for advanced features.

### Want to Contribute?
Read the [Developer Journey](journeys/developer.md) and [Contributing Guide](contributing/).

### Deploying to Production?
Follow the [Operations Journey](journeys/operations.md).

### Understanding Architecture?
Dive into the [Advanced Journey](journeys/advanced.md).

---

## 📝 Documentation Standards

All documentation follows the [Victor Documentation Standards](STANDARDS.md):

- **Diátaxis Framework:** Tutorials, How-to, Reference, Explanation
- **File Size Limits:** 300-1000 lines by content type
- **Quality Checklist:** Content, formatting, style, technical requirements
- **CI/CD Checks:** Markdown lint, link validation, spell check

**See:** [Standards](STANDARDS.md) | [Templates](templates/)

---

<div align="center">

**[← Back to Top](#victor-ai-documentation)**

**Victor AI** - Provider-agnostic AI coding assistant

[GitHub](https://github.com/vjsingh1984/victor) •
[PyPI](https://pypi.org/project/victor-ai/) •
[Issues](https://github.com/vjsingh1984/victor/issues) •
[Discussions](https://github.com/vjsingh1984/victor/discussions)

</div>

---

**Reading Time:** 6 min
**Last Updated:** January 31, 2026
**Documentation Version:** 1.0
**Maintained by:** Victor AI Documentation Team
