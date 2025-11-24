# CodingAgent Examples

This directory contains example scripts demonstrating how to use CodingAgent with different providers.

## Quick Start

```bash
# Install CodingAgent
pip install -e ".[dev]"

# Set up API keys (for cloud providers)
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."
export XAI_API_KEY="xai-..."

# Run an example
python examples/simple_chat.py
```

## Available Examples

### 1. `simple_chat.py` - Basic Ollama Usage
**Requirements:** Ollama running locally

Simple introduction to CodingAgent with Ollama. Shows:
- Basic chat
- Follow-up questions
- Streaming responses
- Using tools (file operations)

```bash
# Make sure Ollama is running
ollama serve

# Pull a model
ollama pull qwen2.5-coder:7b

# Run example
python examples/simple_chat.py
```

### 2. `claude_example.py` - Anthropic Claude
**Requirements:** ANTHROPIC_API_KEY

Demonstrates Claude's capabilities:
- Complex reasoning
- Code generation with docstrings
- Streaming responses
- Multi-turn conversations

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python examples/claude_example.py
```

### 3. `gpt_example.py` - OpenAI GPT
**Requirements:** OPENAI_API_KEY

Shows both GPT-4 and GPT-3.5:
- Code review
- Creative writing
- Fast responses with GPT-3.5
- Problem solving with GPT-4

```bash
export OPENAI_API_KEY="sk-..."
python examples/gpt_example.py
```

### 4. `grok_example.py` - xAI Grok
**Requirements:** XAI_API_KEY

Explores Grok's features:
- General knowledge
- Technical explanations
- Streaming
- Creative + technical tasks
- Multi-turn debugging

```bash
export XAI_API_KEY="xai-..."
python examples/grok_example.py
```

### 5. `gemini_example.py` - Google Gemini
**Requirements:** GOOGLE_API_KEY

Highlights Gemini's strengths:
- Long context handling (1M tokens!)
- Code generation
- Analysis and comparisons
- Creative coding

```bash
export GOOGLE_API_KEY="..."
python examples/gemini_example.py
```

### 6. `multi_provider_workflow.py` - Strategic Multi-Provider Use
**Requirements:** API keys for cloud providers (optional)

Best practices for using multiple providers:
- Brainstorm with Ollama (FREE)
- Implement with GPT-3.5 (CHEAP)
- Review with Claude (QUALITY)
- Generate tests with Ollama (FREE)

Shows 90% cost savings while maintaining quality!

```bash
# Set keys for providers you have
export ANTHROPIC_API_KEY="..."
export OPENAI_API_KEY="..."

python examples/multi_provider_workflow.py
```

## Provider Profiles Example

Create `~/.codingagent/profiles.yaml`:

```yaml
profiles:
  # Free local development
  dev:
    provider: ollama
    model: qwen2.5-coder:7b
    temperature: 0.7

  # Best reasoning
  claude:
    provider: anthropic
    model: claude-sonnet-4-5
    temperature: 1.0

  # Fast and cheap
  gpt35:
    provider: openai
    model: gpt-3.5-turbo
    temperature: 0.7

  # Latest from xAI
  grok:
    provider: xai
    model: grok-beta
    temperature: 0.8

  # Long context
  gemini:
    provider: google
    model: gemini-1.5-pro
    temperature: 0.9

providers:
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}

  openai:
    api_key: ${OPENAI_API_KEY}

  google:
    api_key: ${GOOGLE_API_KEY}

  xai:
    api_key: ${XAI_API_KEY}

  ollama:
    base_url: http://localhost:11434
```

Then use profiles:

```bash
codingagent --profile dev    # Ollama
codingagent --profile claude # Claude
codingagent --profile gpt35  # GPT-3.5
codingagent --profile grok   # Grok
codingagent --profile gemini # Gemini
```

## Creating Your Own Examples

```python
import asyncio
from codingagent.agent.orchestrator import AgentOrchestrator
from codingagent.providers.ollama import OllamaProvider

async def main():
    # Create provider
    provider = OllamaProvider()

    # Create agent
    agent = AgentOrchestrator(
        provider=provider,
        model="qwen2.5-coder:7b",
        temperature=0.7,
    )

    # Chat
    response = await agent.chat("Your question here")
    print(response.content)

    # Stream
    async for chunk in agent.stream_chat("Another question"):
        print(chunk.content, end="", flush=True)

    # Clean up
    await provider.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## Cost Comparison

Running all examples (approximate costs):

| Example | Provider | Tokens | Cost |
|---------|----------|--------|------|
| simple_chat.py | Ollama | N/A | FREE |
| claude_example.py | Claude | ~10K | $0.15 |
| gpt_example.py | GPT-4/3.5 | ~8K | $0.10 |
| grok_example.py | Grok | ~8K | TBD |
| gemini_example.py | Gemini | ~10K | $0.05 |
| multi_provider_workflow.py | Mixed | ~5K | $0.01 |

**Total: ~$0.31** for all paid examples

**Pro tip:** Use Ollama for development and testing!

## Troubleshooting

### "Connection refused" errors

**Ollama:**
```bash
# Make sure Ollama is running
ollama serve

# Check models
ollama list

# Pull a model if needed
ollama pull qwen2.5-coder:7b
```

**Cloud providers:**
```bash
# Verify API key is set
echo $ANTHROPIC_API_KEY
echo $OPENAI_API_KEY

# Test connectivity
curl -H "x-api-key: $ANTHROPIC_API_KEY" https://api.anthropic.com/v1/messages
```

### "No module named 'codingagent'"

```bash
# Install in development mode
cd /path/to/codingagent
pip install -e ".[dev]"
```

### Rate limit errors

- Wait and retry
- Use Ollama instead (no rate limits!)
- Upgrade your API plan

## Next Steps

1. **Try the examples** in order
2. **Modify examples** for your use case
3. **Mix providers** strategically
4. **Read** [PROVIDERS.md](../PROVIDERS.md) for detailed provider info
5. **Check** [README.md](../README.md) for full documentation

Happy coding! 🚀
