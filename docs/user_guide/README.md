# Victor AI User Guide

**Comprehensive documentation for Victor AI users**

---

## Welcome to Victor AI!

Victor AI is an open-source AI coding assistant supporting 21 LLM providers with 55+ specialized tools across 5 domain verticals. This user guide provides everything you need to get started and master Victor AI.

---

## Quick Navigation

### New to Victor AI?

Start here:
1. **[Quick Start Guide](QUICK_START.md)** - Get up and running in 5 minutes
2. **[FAQ](FAQ.md)** - 30+ frequently asked questions
3. **[Troubleshooting](TROUBLESHOOTING.md)** - Solve common issues

### Upgrading from Previous Version?

1. **[Migration Guide](MIGRATION_GUIDE.md)** - Upgrade with confidence (zero breaking changes)
2. **[Coordinator Guide](COORDINATOR_GUIDE.md)** - Understand the new architecture
3. **[Quick Start Guide](QUICK_START.md)** - New features to try

### Want to Learn More?

1. **[Coordinator Guide](COORDINATOR_GUIDE.md)** - Deep dive into coordinators
2. **[Video Tutorials](VIDEO_TUTORIALS.md)** - Watch and learn
3. **[FAQ](FAQ.md)** - Advanced questions answered

---

## Guide Overview

### Quick Start Guide (5 minutes)

**What**: Get started with Victor AI in 5 minutes
**Includes**:
- Installation instructions
- First task walkthrough
- Basic usage examples
- Common tasks
- Next steps

**Perfect for**: New users who want to get started quickly

**Read time**: 5 minutes
**[Go to Guide →](QUICK_START.md)**

---

### Coordinator Guide (15 minutes)

**What**: Comprehensive guide to coordinator-based architecture
**Includes**:
- What are coordinators
- Why they matter (93% complexity reduction)
- Available coordinators (ConversationController, ToolPipeline, ProviderManager, etc.)
- Team coordinators and formations
- Configuration options
- Customization examples
- Best practices
- Advanced usage

**Perfect for**: Users who want to understand the architecture and use advanced features

**Read time**: 15 minutes
**[Go to Guide →](COORDINATOR_GUIDE.md)**

---

### Migration Guide (10 minutes)

**What**: Upgrade to Victor 0.5 with confidence
**Includes**:
- What changed (coordinator-based architecture)
- What stayed the same (100% backward compatible)
- Breaking changes (none!)
- Migration steps (optional)
- New features to try
- Common migration issues
- Rollback plan

**Perfect for**: Existing users upgrading to Victor 0.5

**Read time**: 10 minutes
**[Go to Guide →](MIGRATION_GUIDE.md)**

---

### Troubleshooting Guide (10 minutes)

**What**: Solve common issues quickly
**Includes**:
- Quick diagnostics
- Installation issues
- Provider issues
- Performance issues
- Team coordination issues
- Tool execution issues
- Memory and context issues
- Workflow issues
- Debug mode
- Getting help

**Perfect for**: All users encountering problems

**Read time**: 10 minutes (or use as reference)
**[Go to Guide →](TROUBLESHOOTING.md)**

---

### FAQ (15 minutes)

**What**: 30+ frequently asked questions
**Includes**:
- General questions
- Installation and setup
- Providers and models
- Team coordination
- Tools and capabilities
- Performance and scaling
- Configuration and customization
- Troubleshooting
- Advanced usage
- Community and support

**Perfect for**: All users looking for quick answers

**Read time**: 15 minutes (or search for specific questions)
**[Go to Guide →](FAQ.md)**

---

### Video Tutorials (25 minutes total)

**What**: Production-ready video scripts for learning Victor AI
**Includes**:
- Video 1: Getting Started (5 minutes)
- Video 2: Multi-Agent Team Coordination (5 minutes)
- Video 3: Advanced Team Configuration (5 minutes)
- Video 4: Provider Switching and Health (5 minutes)
- Video 5: Troubleshooting Common Issues (5 minutes)

Each video includes:
- Screen-by-screen guidance
- Script narration
- Visual cues
- Production notes
- Alternative formats (YouTube, blog post, workshop)

**Perfect for**: Visual learners and content creators

**Watch time**: 25 minutes
**[Go to Guides →](VIDEO_TUTORIALS.md)**

---

## Key Concepts

### Coordinators

Victor 0.5 uses a coordinator-based architecture for better maintainability and testability. Instead of one monolithic orchestrator, Victor now has specialized coordinators:

- **ConversationController**: Manages conversation state and context
- **ToolPipeline**: Handles tool execution and budgeting
- **ProviderManager**: Manages providers and health checks
- **StreamingController**: Coordinates streaming responses
- **WorkflowCoordinator**: Manages YAML workflows
- **LifecycleManager**: Handles session lifecycle

**Benefits**:
- 93% reduction in core complexity
- 85% test coverage
- Easier to understand and extend
- Better separation of concerns

### Team Formations

Victor supports 5 team formations for multi-agent coordination:

1. **SEQUENTIAL**: Agents work one after another
2. **PARALLEL**: Agents work simultaneously
3. **HIERARCHICAL**: Manager delegates to workers
4. **PIPELINE**: Output flows through agents
5. **CONSENSUS**: Agents discuss until agreement

### Agent Modes

Victor has three modes controlling exploration vs. exploitation:

- **BUILD** (default): Make changes, standard exploration
- **PLAN**: 2.5x exploration, sandbox only, no edits
- **EXPLORE**: 3.0x exploration, deep investigation, no edits

### Providers

Victor supports 21 LLM providers:

**Cloud**: Anthropic, OpenAI, Google, Azure, AWS Bedrock, xAI, DeepSeek, Mistral, Cohere, Groq, Together AI, Fireworks AI, OpenRouter, Replicate, Hugging Face, Moonshot, Cerebras

**Local**: Ollama, LM Studio, vLLM, llama.cpp

---

## Learning Path

### Beginner (New to Victor AI)

1. Read [Quick Start Guide](QUICK_START.md) - 5 minutes
2. Complete your first task
3. Read [FAQ](FAQ.md) - General questions section
4. Watch Video 1: Getting Started
5. Try common tasks (code review, test generation)

### Intermediate (Comfortable with Victor AI)

1. Read [Coordinator Guide](COORDINATOR_GUIDE.md) - 15 minutes
2. Try team coordination
3. Experiment with different formations
4. Read [FAQ](FAQ.md) - Team coordination section
5. Watch Video 2: Multi-Agent Team Coordination

### Advanced (Want to Master Victor AI)

1. Read entire [Coordinator Guide](COORDINATOR_GUIDE.md)
2. Create custom coordinators
3. Implement rich personas with memory
4. Configure observability and RL
5. Watch Videos 3-5
6. Read [Troubleshooting Guide](TROUBLESHOOTING.md)
7. Contribute to the project

### Upgrading (Existing Users)

1. Read [Migration Guide](MIGRATION_GUIDE.md) - 10 minutes
2. Upgrade Victor: `pip install --upgrade victor-ai`
3. Verify installation: `victor --version`
4. Try new features
5. Adopt new imports (optional)
6. Read [Coordinator Guide](COORDINATOR_GUIDE.md) for architecture

---

## Quick Reference

### Essential Commands

```bash
# Installation
pip install victor-ai

# Start chat
victor chat

# Use specific provider
victor chat --provider anthropic

# Check version
victor --version

# Health check
victor doctor

# Configuration
victor config show
```

### Environment Variables

```bash
# Provider
export ANTHROPIC_API_KEY=sk-ant-...
export VICTOR_PROVIDER=anthropic
export VICTOR_MODEL=claude-sonnet-4-5

# Mode
export VICTOR_MODE=build

# Tools
export VICTOR_TOOL_BUDGET=100

# Debug
export VICTOR_LOG_LEVEL=DEBUG
```

### Python API

```python
from victor import Victor

# Simple usage
vic = Victor()
response = vic.chat("Review this code")

# With provider
vic = Victor(provider="anthropic", model="claude-sonnet-4-5")

# Team coordination
from victor.teams import create_coordinator, TeamFormation

coordinator = create_coordinator(vic.orchestrator)
coordinator.set_formation(TeamFormation.PIPELINE)
result = await coordinator.execute_task("Review auth.py", {})
```

---

## Success Stories

### Code Review

```bash
# Comprehensive security review
victor chat "Review auth.py for security vulnerabilities"
```

**Result**: Found 3 critical issues, suggested fixes, completed in 30 seconds

### Test Generation

```bash
# Generate unit tests
victor chat "Generate unit tests for user service with 90% coverage"
```

**Result**: Created 25 tests, achieved 92% coverage, saved 2 hours

### Refactoring

```bash
# Plan and execute refactoring
victor chat --mode plan "Plan refactoring of data pipeline"
victor chat --mode build "Execute refactoring"
```

**Result**: Reduced complexity by 40%, no breaking changes

### Team Coordination

```python
# Multi-agent security review
coordinator.set_formation(TeamFormation.CONSENSUS)
result = await coordinator.execute_task(
    "Review authentication system for security issues",
    {}
)
```

**Result**: 3 agents reached consensus, found 5 vulnerabilities, provided fixes

---

## Common Tasks

### Review Code

```bash
# Review single file
victor chat "Review auth.py for security issues"

# Review entire codebase
victor chat "Review entire codebase for best practices"

# Team review
coordinator.set_formation(TeamFormation.PIPELINE)
await coordinator.execute_task("Review payment system", {})
```

### Generate Tests

```bash
# Unit tests
victor chat "Generate unit tests for user service"

# Integration tests
victor chat "Add integration tests for API endpoints"

# With coverage goal
victor chat "Achieve 90% coverage for auth module"
```

### Refactor Code

```bash
# Plan refactoring
victor chat --mode plan "Plan refactoring of data processor"

# Execute refactoring
victor chat --mode build "Refactor to use dependency injection"

# Review refactoring
victor chat "Review the refactored code"
```

### Generate Documentation

```bash
# API docs
victor chat "Generate OpenAPI documentation for REST endpoints"

# Docstrings
victor chat "Add comprehensive docstrings to all functions"

# README
victor chat "Create a comprehensive README for this project"
```

---

## Getting Help

### Documentation

- **[Quick Start Guide](QUICK_START.md)** - Get started in 5 minutes
- **[Coordinator Guide](COORDINATOR_GUIDE.md)** - Understand the architecture
- **[Migration Guide](MIGRATION_GUIDE.md)** - Upgrade from previous versions
- **[Troubleshooting Guide](TROUBLESHOOTING.md)** - Solve common issues
- **[FAQ](FAQ.md)** - Frequently asked questions

### Community

- **GitHub Issues**: https://github.com/vjsingh1984/victor/issues
- **GitHub Discussions**: https://github.com/vjsingh1984/victor/discussions
- **Email**: singhvjd@gmail.com

### Diagnostics

```bash
# Run health check
victor doctor

# Check version
victor --version

# Test provider
victor provider test

# Validate config
victor config validate
```

---

## Metrics

### User Guide Statistics

| Guide | Lines | Words | Read Time |
|-------|-------|-------|-----------|
| Quick Start | 515 | ~3,500 | 5 min |
| Coordinator Guide | 970 | ~6,500 | 15 min |
| Migration Guide | 771 | ~5,200 | 10 min |
| Troubleshooting | 855 | ~5,800 | 10 min |
| FAQ | 758 | ~5,100 | 15 min |
| Video Tutorials | 728 | ~4,900 | 25 min* |
| **Total** | **4,597** | **~31,000** | **80 min** |

*Watch time

### Coverage

- **Beginner topics**: Installation, basic usage, common tasks
- **Intermediate topics**: Teams, formations, configuration
- **Advanced topics**: Custom coordinators, observability, RL
- **Migration**: Upgrading, breaking changes, rollback
- **Troubleshooting**: 50+ common issues with solutions
- **FAQ**: 30+ questions across 10 categories

---

## Contributing

Found an issue with the documentation? Want to improve it?

1. Fork the repository
2. Edit the documentation
3. Submit a pull request

All documentation is in Markdown format, making it easy to contribute.

---

## Version History

### Version 0.5 (January 2025)

**New**:
- Coordinator-based architecture
- 5 team formations
- Rich personas with memory
- Observability integration
- 93% complexity reduction

**Documentation**:
- 6 comprehensive guides created
- 4,597 lines of documentation
- 30+ FAQ entries
- 5 video tutorials scripts

---

## License

Victor AI is open-source and licensed under the Apache 2.0 License.

---

## Acknowledgments

Victor AI is built by the open-source community. Special thanks to all contributors and users who provide feedback and improvements.

---

**Ready to get started?** [Read the Quick Start Guide →](QUICK_START.md)

**Need help?** [Check the Troubleshooting Guide →](TROUBLESHOOTING.md) or [FAQ →](FAQ.md)

**Want to learn more?** [Read the Coordinator Guide →](COORDINATOR_GUIDE.md)

---

**Happy coding with Victor AI! 🚀**
