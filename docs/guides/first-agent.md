# Building Your First Agent

Learn how to create and customize Victor agents for your specific use cases.

## What is an Agent?

A **Victor Agent** is an autonomous entity that:
- Uses Large Language Models (LLMs) to understand and generate text
- Can call **tools** to perform actions (read files, execute code, etc.)
- Maintains **conversation state** across interactions
- Can be customized with specific **behaviors** and **capabilities**

---

## Basic Agent Creation

### Minimal Agent

The simplest agent uses default settings:

```python
import asyncio
from victor.framework import Agent

async def main():
    # Create agent with defaults
    agent = Agent()

    # Run a query
    result = await agent.run("Explain quantum computing in simple terms")

    print(result.content)

asyncio.run(main())
```

### Agent with Provider Selection

Choose a specific LLM provider:

```python
import asyncio
from victor.framework import Agent

async def main():
    # Use Anthropic Claude
    agent = Agent(
        provider="anthropic",
        model="claude-sonnet-4-20250514"
    )

    result = await agent.run("What's the best way to learn Python?")
    print(result.content)

asyncio.run(main())
```

**Supported providers** (24 available):
- `openai` - GPT-4, GPT-3.5
- `anthropic` - Claude Opus, Sonnet, Haiku
- `azure` - Azure OpenAI
- `google` - Gemini Pro, Flash
- `ollama` - Local models (Llama 3, Mistral, Qwen)
- `groq` - Fast inference
- `cohere` - Command R+
- And 14 more...

---

## Adding Tools

Tools give agents the ability to perform actions.

### Built-in Tool Presets

```python
from victor.framework import Agent

# Minimal tools (no filesystem access)
agent = Agent(tools="minimal")

# Default tools (safe filesystem operations)
agent = Agent(tools="default")

# Full tools (all available tools)
agent = Agent(tools="full")

# Air-gapped (no external APIs)
agent = Agent(tools="airgapped")
```

### Custom Tool Selection

```python
from victor.framework import Agent

# Select specific tools by name
agent = Agent(
    tools=["read_file", "write_file", "list_directory"]
)

result = await agent.run(
    "Find all Python files and count the lines of code"
)
```

**Available tools** (34 tool modules):

| Category | Tools |
|----------|-------|
| **Filesystem** | `read_file`, `write_file`, `edit_file`, `list_directory` |
| **Git** | `git_status`, `git_commit`, `git_diff`, `git_log` |
| **Web** | `web_search`, `web_fetch` |
| **Execution** | `run_shell`, `run_python` |
| **Analysis** | `code_search`, `repository_overview` |
| **Docker** | `docker_build`, `docker_run` |

See [Tool Catalog](../reference/tools/catalog.md) for complete list.

---

## Streaming Responses

For real-time feedback, use streaming:

```python
import asyncio
from victor.framework import Agent

async def main():
    agent = Agent()

    # Stream responses as they arrive
    async for event in agent.stream("Tell me an interesting story"):
        if event.type == "content":
            print(event.content, end="", flush=True)
        elif event.type == "thinking":
            print("\n[Thinking...]", flush=True)

    print()  # New line after completion

asyncio.run(main())
```

**Event types**:
- `THINKING` - Extended thinking (if enabled)
- `TOOL_CALL` - Agent is calling a tool
- `TOOL_RESULT` - Result from a tool
- `CONTENT` - Text content from the LLM
- `ERROR` - An error occurred
- `STREAM_END` - Response complete

---

## Multi-turn Conversations

Maintain context across multiple messages:

```python
import asyncio
from victor.framework import Agent

async def main():
    agent = Agent()

    # Conversation with context
    messages = [
        "My favorite color is blue",
        "What did I just tell you?",
        "Suggest some blue flowers",
    ]

    for msg in messages:
        response = await agent.chat(msg)
        print(f"You: {msg}")
        print(f"Agent: {response.content}\n")

asyncio.run(main())
```

### Chat Context

```python
agent = Agent()

# First message establishes context
await agent.chat("I'm working on a Python web project")

# Agent remembers context in subsequent messages
response = await agent.chat("What web framework should I use?")
# Agent will suggest Flask/FastAPI/Django based on context
```

---

## Custom System Prompts

Override the default behavior:

```python
from victor.framework import Agent

agent = Agent(
    system_prompt="""You are an expert Python developer.
    Always provide code examples with proper error handling.
    Follow PEP 8 style guidelines.
    """
)

result = await agent.run("How do I read a file safely?")
print(result.content)
```

---

## Using Verticals

Verticals are pre-configured agents for specific domains:

### Coding Vertical

```python
from victor.framework import Agent

agent = Agent(
    vertical="coding",
    tools=["read_file", "write_file", "edit_file", "grep", "git"]
)

result = await agent.run(
    "Review the code in src/main.py and suggest improvements"
)
```

### Research Vertical

```python
agent = Agent(
    vertical="research",
    tools=["web_search", "web_fetch", "read_file"]
)

result = await agent.run(
    "Research the latest developments in quantum computing"
)
```

**Available verticals**:
- `coding` - Code generation, review, debugging
- `devops` - Infrastructure, CI/CD, containers
- `research` - Information gathering, analysis
- `dataanalysis` - Data exploration, visualization
- `rag` - Document retrieval and Q&A
- `security` - Security analysis, testing
- `classification` - Document and task classification
- `benchmark` - Performance benchmarking

---

## Advanced Configuration

### Temperature and Creativity

```python
agent = Agent(
    temperature=0.7,    # 0.0 = focused, 1.0 = creative
    max_tokens=2000,    # Maximum response length
)

# Creative writing
creative_agent = Agent(temperature=1.0)

# Code generation (focused)
code_agent = Agent(temperature=0.2)
```

### Working with Files

```python
agent = Agent()

# Pass file context
result = await agent.run(
    "Explain the bug in this code",
    context={
        "file": "app.py",
        "error": "IndexError: list index out of range"
    }
)
```

---

## Error Handling

Handle errors gracefully:

```python
import asyncio
from victor.framework import Agent

async def main():
    agent = Agent()

    try:
        result = await agent.run("Analyze this file")
        print(result.content)
        print(f"Success: {result.success}")
        print(f"Tool calls: {result.tool_calls}")
    except Exception as e:
        print(f"Error: {e}")

asyncio.run(main())
```

---

## Putting It All Together

### Example: Code Review Agent

```python
import asyncio
from victor.framework import Agent

async def review_code(file_path: str):
    """Create a code review agent."""
    agent = Agent(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        tools=["read_file", "search_code"],
        temperature=0.3,
        system_prompt="""You are a senior code reviewer.
        Focus on:
        - Code quality and readability
        - Potential bugs
        - Security issues
        - Performance improvements
        - Best practices
        """
    )

    result = await agent.run(
        f"Review the code in {file_path}. "
        "Provide specific suggestions for improvement."
    )

    return result

# Usage
if __name__ == "__main__":
    review = asyncio.run(review_code("src/main.py"))
    print(review)
```

### Example: Research Assistant

```python
import asyncio
from victor.framework import Agent

async def research_topic(topic: str):
    """Create a research assistant."""
    agent = Agent(
        provider="anthropic",
        vertical="research",
        tools=["web_search", "web_fetch"],
        temperature=0.5,
    )

    query = f"""
    Research the topic: {topic}

    Please:
    1. Search for recent information
    2. Summarize key findings
    3. Identify consensus views
    4. Note any controversies
    5. Suggest further reading
    """

    result = await agent.run(query)
    return result

# Usage
if __name__ == "__main__":
    report = asyncio.run(research_topic("quantum computing applications"))
    print(report)
```

---

## Best Practices

### 1. Choose the Right Provider

| Use Case | Recommended Provider |
|----------|---------------------|
| Complex reasoning | `anthropic` (Claude Sonnet) |
| Code generation | `openai` (GPT-4) |
| Cost efficiency | `openai` (GPT-3.5-turbo) |
| Privacy | `ollama` (local) |

### 2. Use Appropriate Tools

```python
# Only give tools the agent needs
agent = Agent(tools=["read_file", "search_code"])
```

### 3. Set Clear System Prompts

```python
agent = Agent(
    system_prompt="""You are a helpful assistant.
    Be concise and direct.
    If unsure, ask for clarification.
    """
)
```

### 4. Handle Errors Gracefully

```python
result = await agent.run(query)
if result.success:
    print(result.content)
else:
    print(f"Error: {result.error}")
```

---

## Common Patterns

### Pattern 1: Tool-First Agent

```python
# Agent uses tools before responding
agent = Agent(
    tools=["read_file", "search_code"],
    system_prompt="Always read files before answering questions about code."
)
```

### Pattern 2: Streaming Agent

```python
# Real-time feedback for long tasks
async for event in agent.stream("Analyze the entire codebase"):
    if event.type == "content":
        print(event.content, end="", flush=True)
```

### Pattern 3: Memory-Keeping Agent

```python
# Maintain conversation state
agent = Agent()
await agent.chat("I prefer Python over JavaScript")
# Agent will remember this preference
```

---

## Next Steps

- 🔄 [Create Your First Workflow](first-workflow.md) - Build multi-step workflows
- 🛠️ [Tool Catalog](../reference/tools/catalog.md) - Explore all available tools
- 📚 [API Reference](../api/) - Full API documentation
- 💡 [Examples](../examples/agents/) - More agent examples

---

## Quick Reference

```python
# Basic agent
agent = Agent()

# With provider
agent = Agent(provider="anthropic")

# With tools
agent = Agent(tools=["read_file", "write_file"])

# Run once
result = await agent.run("Your query")

# Multi-turn
response = await agent.chat("Your message")

# Streaming
async for event in agent.stream("Your query"):
    print(event.content)
```
