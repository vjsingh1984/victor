# Quick Start Guide

Get up and running with Victor in 5 minutes.

## Prerequisites

- Python 3.10 or higher
- pip (Python package installer)
- An API key for at least one LLM provider (OpenAI, Anthropic, etc.)

## Installation

### Option 1: Install with Default Dependencies

```bash
pip install victor-ai
```

### Option 2: Install with Development Tools

```bash
pip install "victor-ai[dev]"
```

This includes additional tools for development and testing.

### Option 3: Install from Source

```bash
git clone https://github.com/vjsingh1984/victor.git
cd victor
pip install -e .
```

## Configure Your API Key

Victor needs an API key to interact with LLM providers. Choose one of the following methods:

### Method 1: Environment Variable (Recommended)

```bash
# For OpenAI
export OPENAI_API_KEY="your-api-key-here"

# For Anthropic Claude
export ANTHROPIC_API_KEY="your-api-key-here"

# For other providers, see the Installation guide
```

### Method 2: Settings File

Create a file at `~/.victor/settings.yaml`:

```yaml
providers:
  openai:
    api_key: "your-api-key-here"
```

### Method 3: Command Line

```bash
victor config set openai.api_key "your-api-key-here"
```

## Your First Agent

Create a file called `my_first_agent.py`:

```python
import asyncio
from victor import Agent

async def main():
    # Create an agent with default settings
    agent = Agent.create()

    # Run the agent
    result = await agent.run("What is the capital of France?")

    # Print the result
    print(result.content)

if __name__ == "__main__":
    asyncio.run(main())
```

Run it:

```bash
python my_first_agent.py
```

Expected output:

```
The capital of France is Paris.
```

## Your First Agent with Tools

Let's create an agent that can use tools:

```python
import asyncio
from victor import Agent

async def main():
    # Create an agent with filesystem tools
    agent = Agent.create(
        tools=["read", "write", "ls"]
    )

    # Ask the agent to list files
    result = await agent.run("List all Python files in the current directory")

    print(result.content)

if __name__ == "__main__":
    asyncio.run(main())
```

## Streaming Responses

For real-time streaming of responses:

```python
import asyncio
from victor import Agent

async def main():
    agent = Agent.create()

    # Stream the response in real-time
    async for event in agent.stream("Tell me a short joke"):
        if event.type == "content":
            print(event.content, end="", flush=True)

    print()  # New line after completion

if __name__ == "__main__":
    asyncio.run(main())
```

## Multi-turn Conversations

For chat-like interactions:

```python
import asyncio
from victor import Agent

async def main():
    agent = Agent.create()

    # Multi-turn conversation
    response1 = await agent.chat("My name is Alice")
    response2 = await agent.chat("What's my name?")
    print(response2.content)  # Should remember "Alice"

if __name__ == "__main__":
    asyncio.run(main())
```

## Using a Vertical

Victor comes with domain-specific "verticals" for specialized tasks:

```python
import asyncio
from victor import Agent

async def main():
    # Use the coding vertical for code-related tasks
    agent = Agent.create(
        vertical="coding",
        tools=["read", "write", "edit", "grep"]
    )

    result = await agent.run("Review the main.py file and suggest improvements")

    print(result.content)

if __name__ == "__main__":
    asyncio.run(main())
```

Available verticals:
- `coding` - Code generation, review, and debugging
- `devops` - Infrastructure and deployment automation
- `research` - Research and information gathering
- `dataanalysis` - Data exploration and analysis
- `rag` - Retrieval-augmented generation
- `security` - Security analysis and testing

## Next Steps

- 📖 [Installation Guide](installation.md) - Detailed installation instructions
- 🤖 [Building Your First Agent](first-agent.md) - Deep dive into agent creation
- 🔄 [Creating Workflows](first-workflow.md) - Build multi-step workflows
- 🛠️ [Using Tools](../examples/agents/tool_usage.md) - Explore available tools
- 📚 [API Reference](../api/index.md) - Full API documentation

## Troubleshooting

### "No module named 'victor'"

**Solution**: Make sure you installed Victor in the correct Python environment:

```bash
python -m pip show victor-ai
```

If not installed, reinstall:

```bash
pip install victor-ai
```

### "API key not found"

**Solution**: Set your API key as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or use a settings file at `~/.victor/settings.yaml`.

### "ImportError: DLL load failed" (Windows)

**Solution**: Install Microsoft Visual C++ Redistributable:

<https://aka.ms/vs/17/release/vc_redist.x64.exe>

### Connection Errors

**Solution**: Check your internet connection and API key validity:

```bash
# Test your API key
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

## Getting Help

- 📖 [Documentation](https://victor-ai.readthedocs.io)
- 💬 [Discord Community](https://discord.gg/victor-ai)
- 🐛 [Report Issues](https://github.com/vjsingh1984/victor/issues)
- ✉️ [Email Support](mailto:support@victor-ai.dev)

## License

Victor is open-source under the Apache 2.0 License. See [LICENSE](https://github.com/vjsingh1984/victor/blob/main/LICENSE) for details.
