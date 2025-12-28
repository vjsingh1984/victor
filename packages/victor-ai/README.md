# Victor AI

**Enterprise-Ready AI Coding Assistant - Code to Victory with Any AI**

Victor is a terminal-based AI coding assistant providing a unified interface for 25+ LLM providers with 45+ enterprise tools, semantic codebase search, and MCP support.

## Installation

```bash
pip install victor-ai
```

This installs everything:
- **victor-core**: Multi-provider LLM orchestration framework
- **victor-coding**: AI-powered coding assistant vertical

## Quick Start

```bash
# Initialize in your project
victor init

# Start interactive chat
victor

# One-shot commands
victor chat "Explain this codebase"
victor chat --mode explore "Find all API endpoints"
victor chat --mode plan "Add user authentication"
```

## Features

### 25+ LLM Providers
- **Cloud**: Anthropic (Claude), OpenAI (GPT-4), Google (Gemini), Mistral, xAI (Grok)
- **Fast Inference**: Groq, Cerebras, Fireworks, Together AI
- **Local**: Ollama, LMStudio, vLLM, llama.cpp
- **Enterprise**: Azure OpenAI, AWS Bedrock, Google Vertex AI

### 45+ Enterprise Tools
- Code search and semantic search
- Code review and refactoring
- Test generation
- Security analysis
- Documentation generation
- Git operations
- And many more...

### Modes
- **explore**: Read-only codebase exploration
- **plan**: Design implementation strategies
- **build**: Full tool access for code changes

## VS Code Extension

Install the [Victor AI extension](https://marketplace.visualstudio.com/items?itemName=victor-ai.victor-ai) for IDE integration.

## Documentation

- [Full Documentation](https://github.com/vijayksingh/victor)
- [Tool Catalog](https://github.com/vijayksingh/victor/blob/main/docs/TOOL_CATALOG.md)
- [Provider Setup](https://github.com/vijayksingh/victor/blob/main/docs/PROVIDERS.md)

## License

Apache-2.0
