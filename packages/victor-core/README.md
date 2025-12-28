# Victor Core

**Multi-provider LLM orchestration framework for building AI agents.**

Victor Core provides the foundation for building AI-powered applications with:

- **25+ LLM Providers**: Anthropic, OpenAI, Google, Groq, DeepSeek, Mistral, Ollama, LMStudio, vLLM, and more
- **Tool System**: Extensible tool framework with semantic selection
- **Protocol-based Architecture**: Clean abstractions for providers, tools, and verticals
- **Caching**: Tiered caching (L1 memory + L2 disk) for tool results
- **MCP Support**: Model Context Protocol client and server

## Installation

```bash
pip install victor-core
```

For full coding assistance capabilities, install the coding vertical:

```bash
pip install victor-coding
```

Or install everything with the meta-package:

```bash
pip install victor-ai
```

## Quick Start

```python
from victor.config.settings import Settings
from victor.agent.orchestrator import AgentOrchestrator

# Initialize with your provider
settings = Settings(provider="anthropic", model="claude-sonnet-4-20250514")
orchestrator = AgentOrchestrator(settings)

# Chat with the agent
response = await orchestrator.chat("Hello, what can you help me with?")
print(response.content)
```

## Documentation

- [Full Documentation](https://github.com/vijayksingh/victor)
- [API Reference](https://github.com/vijayksingh/victor/tree/main/docs)
- [Migration Guide](https://github.com/vijayksingh/victor/blob/main/docs/PACKAGE_MIGRATION.md)

## License

Apache-2.0
