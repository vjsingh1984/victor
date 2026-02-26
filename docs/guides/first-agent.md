# Building Your First Agent

Learn how to create and customize Victor agents for your specific use cases.

## What is an Agent?

A **Victor Agent** is an autonomous entity that:
- Uses Large Language Models (LLMs) to understand and generate text
- Can call **tools** to perform actions (read files, execute code, etc.)
- Maintains **state** across interactions
- Can be customized with specific **behaviors** and **capabilities**

## Basic Agent Creation

### Minimal Agent

The simplest agent uses default settings:

```python
import asyncio
from victor import Agent

async def main():
    # Create agent with defaults
    agent = Agent.create()

    # Run a query
    result = await agent.run("Explain quantum computing in simple terms")

    print(result.content)

asyncio.run(main())
```

### Agent with Provider Selection

Choose a specific LLM provider:

```python
import asyncio
from victor import Agent

async def main():
    # Use Anthropic Claude
    agent = Agent.create(
        provider="anthropic",
        model="claude-3-opus-20240229"
    )

    result = await agent.run("What's the best way to learn Python?")
    print(result.content)

asyncio.run(main())
```

**Supported providers**:
- `openai` - GPT-4, GPT-3.5 (default)
- `anthropic` - Claude 3 Opus, Sonnet, Haiku
- `azure` - Azure OpenAI
- `google` - Gemini Pro
- `cohere` - Command R+
- `ollama` - Local models (Llama 2, Mistral, etc.)

## Adding Tools

Tools give agents the ability to perform actions.

### Built-in Tool Presets

```python
from victor import Agent

# Minimal tools (no filesystem access)
agent = Agent.create(tools="minimal")

# Default tools (safe filesystem operations)
agent = Agent.create(tools="default")

# Full tools (all available tools)
agent = Agent.create(tools="full")

# Air-gapped (no external APIs)
agent = Agent.create(tools="airgapped")
```

### Custom Tool Selection

```python
from victor import Agent

# Select specific tools
agent = Agent.create(tools=[
    "read",      # Read files
    "write",     # Write files
    "ls",        # List directories
    "grep",      # Search files
    "shell",     # Run shell commands
])

result = await agent.run(
    "Find all Python files and count the lines of code"
)
```

**Available tools**:

| Category | Tools |
|----------|-------|
| **Filesystem** | `read`, `write`, `edit`, `ls`, `grep` |
| **Git** | `git_status`, `git_commit`, `git_push`, `git_diff` |
| **Web** | `web_search`, `web_fetch` |
| **Execution** | `shell`, `python` |
| **Analysis** | `overview`, `code_search`, `graph` |
| **Docker** | `docker_build`, `docker_run`, `docker_exec` |

See [Tool Reference](../api/tools.md) for complete list.

## Streaming Responses

For real-time feedback, use streaming:

```python
import asyncio
from victor import Agent

async def main():
    agent = Agent.create()

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
- `content` - Text content from the LLM
- `thinking` - Extended thinking (if enabled)
- `tool_call` - Agent is calling a tool
- `tool_result` - Result from a tool
- `error` - An error occurred

## Multi-turn Conversations

Maintain context across multiple messages:

```python
import asyncio
from victor import Agent

async def main():
    agent = Agent.create()

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

### Conversation History

```python
# Access conversation history
agent = Agent.create()

# Add initial context
await agent.chat("I'm working on a Python web project")

# ... later ...

# Agent remembers the context
response = await agent.chat("What web framework should I use?")
# Agent will suggest Flask/FastAPI/Django based on context
```

## Custom System Prompts

Override the default behavior:

```python
from victor import Agent

agent = Agent.create(
    system_prompt="""You are an expert Python developer.
    Always provide code examples with proper error handling.
    Follow PEP 8 style guidelines.
    """
)

result = await agent.run("How do I read a file safely?")
print(result.content)
```

## Using Verticals

Verticals are pre-configured agents for specific domains:

### Coding Vertical

```python
from victor import Agent

agent = Agent.create(
    vertical="coding",
    tools=["read", "write", "edit", "grep", "git"]
)

result = await agent.run(
    "Review the code in src/main.py and suggest improvements"
)
```

### DevOps Vertical

```python
agent = Agent.create(
    vertical="devops",
    tools=["read", "write", "shell", "docker"]
)

result = await agent.run(
    "Create a Dockerfile for a Python Flask application"
)
```

### Research Vertical

```python
agent = Agent.create(
    vertical="research",
    tools=["web_search", "web_fetch", "read"]
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

## Advanced Configuration

### Temperature and Creativity

```python
agent = Agent.create(
    temperature=0.7,    # 0.0 = focused, 1.0 = creative
    max_tokens=2000,    # Maximum response length
)

# Creative writing
creative_agent = Agent.create(temperature=1.0)

# Code generation (focused)
code_agent = Agent.create(temperature=0.2)
```

### Memory and Context

```python
agent = Agent.create(
    # Control how much conversation history to keep
    max_context_messages=10,  # Keep last 10 messages
)
```

### Observability

```python
agent = Agent.create(
    enable_observability=True,  # Enable metrics collection
    session_id="my-session-123",  # Track across calls
)

# Access metrics later
metrics = agent.get_metrics()
print(f"Total runs: {metrics.total_runs}")
print(f"Tool calls: {metrics.tool_calls}")
```

## Error Handling

Handle errors gracefully:

```python
import asyncio
from victor import Agent

async def main():
    agent = Agent.create()

    try:
        result = await agent.run("Analyze this file")
        print(result.content)
    except Exception as e:
        print(f"Error: {e}")
        # Agent can still be used after errors
        result = await agent.run("Try again with a simpler task")
        print(result.content)

asyncio.run(main())
```

## Putting It All Together

### Example: Code Review Agent

```python
import asyncio
from victor import Agent

async def review_code(file_path: str):
    """Create a code review agent."""
    agent = Agent.create(
        provider="openai",
        model="gpt-4",
        vertical="coding",
        tools=["read", "grep"],
        temperature=0.3,  # Lower temperature for analysis
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

    return result.content

# Usage
if __name__ == "__main__":
    review = asyncio.run(review_code("src/main.py"))
    print(review)
```

### Example: Research Assistant

```python
import asyncio
from victor import Agent

async def research_topic(topic: str):
    """Create a research assistant."""
    agent = Agent.create(
        provider="anthropic",
        model="claude-3-opus-20240229",
        vertical="research",
        tools=["web_search", "web_fetch"],
        temperature=0.5,
    )

    # Multi-step research
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
    return result.content

# Usage
if __name__ == "__main__":
    report = asyncio.run(research_topic("quantum computing applications"))
    print(report)
```

### Example: File Management Agent

```python
import asyncio
from victor import Agent

async def organize_files(directory: str):
    """Create an agent to organize files."""
    agent = Agent.create(
        tools=["ls", "read", "write", "grep"],
        system_prompt="""You are a file organization assistant.
        Help organize files by type, date, or project.
        Always explain what you're doing before making changes.
        """,
    )

    result = await agent.run(
        f"Analyze the files in {directory} and suggest "
        "a better organization structure."
    )

    return result.content

# Usage
if __name__ == "__main__":
    plan = asyncio.run(organize_files("~/Downloads"))
    print(plan)
```

## Best Practices

### 1. Choose the Right Provider

```python
# For complex reasoning
agent = Agent.create(provider="anthropic", model="claude-3-opus")

# For code generation
agent = Agent.create(provider="openai", model="gpt-4")

# For cost efficiency
agent = Agent.create(provider="openai", model="gpt-3.5-turbo")
```

### 2. Use Appropriate Tools

```python
# Only give tools the agent needs
agent = Agent.create(tools=["read", "grep"])  # Not "write" or "shell"
```

### 3. Set Clear System Prompts

```python
agent = Agent.create(
    system_prompt="""You are a helpful assistant.
    Be concise and direct.
    If you don't know, say so.
    """
)
```

### 4. Handle Streaming for Long Responses

```python
async for event in agent.stream(long_query):
    if event.type == "content":
        print(event.content, end="", flush=True)
```

### 5. Use Verticals for Domain-Specific Tasks

```python
# Instead of custom prompts
agent = Agent.create(vertical="coding")  # Pre-configured for coding
```

## Common Patterns

### Pattern 1: Tool-First Agent

```python
# Agent uses tools before responding
agent = Agent.create(
    tools=["read", "grep"],
    system_prompt="Always read files before answering questions about code."
)
```

### Pattern 2: Streaming Agent

```python
# Real-time feedback for long tasks
async for event in agent.stream("Analyze the entire codebase"):
    print(event.content, end="", flush=True)
```

### Pattern 3: Memory-Keeping Agent

```python
# Maintain conversation state
agent = Agent.create()
await agent.chat("I prefer Python over JavaScript")
# Agent will remember this preference
```

## Troubleshooting

### Agent Not Responding

**Check**: API key is set correctly
```bash
echo $OPENAI_API_KEY
```

### Agent Not Using Tools

**Check**: Tools are specified correctly
```python
agent = Agent.create(tools=["read", "write"])  # Must be a list
```

### Slow Responses

**Solution**: Use faster model
```python
agent = Agent.create(model="gpt-3.5-turbo")  # Faster than GPT-4
```

### Agent Hallucinating

**Solution**: Lower temperature
```python
agent = Agent.create(temperature=0.2)  # More focused
```

## Next Steps

- 🔄 [Creating Workflows](first-workflow.md) - Build multi-step workflows
- 🛠️ [Tool Reference](../api/tools.md) - Explore all available tools
- 📚 [API Reference](../api/index.md) - Full API documentation
- 💡 [Examples](../examples/agents/) - More agent examples

## Quick Reference

```python
# Basic agent
agent = Agent.create()

# With provider
agent = Agent.create(provider="anthropic")

# With tools
agent = Agent.create(tools=["read", "write"])

# Run once
result = await agent.run("Your query")

# Multi-turn
response = await agent.chat("Your message")

# Streaming
async for event in agent.stream("Your query"):
    print(event.content)
```
