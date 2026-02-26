# Victor Examples

This directory contains practical examples for using the Victor AI framework.

## Quick Start Examples

### Agent Examples

The `agents/` directory contains examples of creating and using Victor agents:

| Example | Description | File |
|---------|-------------|------|
| Basic Agent | Simplest agent creation | `basic_agent.py` |
| Agent with Tools | Agent with filesystem tools | `agent_with_tools.py` |
| Streaming Agent | Real-time response streaming | `streaming_agent.py` |
| Code Review Agent | Domain-specific agent for code review | `code_review_agent.py` |

### Workflow Examples

The `workflows/` directory contains YAML workflow definitions:

| Example | Description | File |
|---------|-------------|------|
| Simple Workflow | Basic linear workflow | `simple_workflow.yaml` |
| Parallel Analysis | Multi-perspective analysis in parallel | `parallel_analysis.yaml` |

## Running the Examples

### Prerequisites

1. Install Victor:
   ```bash
   pip install victor-ai
   ```

2. Set your API key:
   ```bash
   export OPENAI_API_KEY="your-key-here"
   ```

### Running Agent Examples

```bash
# Basic agent
cd docs/examples/agents
python basic_agent.py

# Agent with tools
python agent_with_tools.py

# Streaming agent
python streaming_agent.py

# Code review agent
python code_review_agent.py
```

### Running Workflow Examples

```bash
cd docs/examples/workflows

# Run the simple workflow
python run_workflow.py

# Run the parallel analysis workflow
python run_parallel_analysis.py
```

## Example Walkthroughs

### Example 1: Basic Agent

The simplest way to use Victor:

```python
from victor import Agent

agent = Agent.create()
result = await agent.run("What is the capital of France?")
print(result.content)
```

**Key concepts**:
- `Agent.create()` - Creates an agent with default settings
- `agent.run()` - Single-turn interaction
- `result.content` - Access the response

### Example 2: Agent with Tools

Give agents the ability to perform actions:

```python
agent = Agent.create(
    tools=["read", "write", "ls", "grep"]
)

result = await agent.run(
    "Find all Python files and count lines of code"
)
```

**Key concepts**:
- `tools` parameter - Specify which tools the agent can use
- Tools extend agent capabilities beyond just text generation

### Example 3: Streaming

Get real-time feedback:

```python
async for event in agent.stream("Tell me a story"):
    if event.type == "content":
        print(event.content, end="", flush=True)
```

**Key concepts**:
- `agent.stream()` - Stream responses as they arrive
- Event types: `content`, `thinking`, `tool_call`, `error`

### Example 4: Code Review

Domain-specific agents with custom behavior:

```python
agent = Agent.create(
    vertical="coding",
    temperature=0.3,
    system_prompt="You are a senior code reviewer..."
)
```

**Key concepts**:
- `vertical` - Pre-configured domain expertise
- `temperature` - Control creativity (0.0 = focused, 1.0 = creative)
- `system_prompt` - Override default behavior

### Example 5: YAML Workflow

Define multi-step processes:

```yaml
name: "My Workflow"
nodes:
  - id: "step1"
    type: "agent"
    config:
      prompt: "Do something"

edges:
  - from: "start"
    to: "step1"
  - from: "step1"
    to: "complete"
```

**Key concepts**:
- `nodes` - Steps in the workflow
- `edges` - Connections between steps
- Workflow execution is stateful and can include conditionals

## Common Patterns

### Pattern 1: Tool-First Agent

```python
agent = Agent.create(
    tools=["read", "grep"],
    system_prompt="Always read files before answering"
)
```

### Pattern 2: Streaming for Long Responses

```python
async for event in agent.stream(long_query):
    if event.type == "content":
        print(event.content, end="", flush=True)
```

### Pattern 3: Parallel Workflow

```yaml
# Run multiple agents in parallel
nodes:
  - id: "analysis1"
  - id: "analysis2"
edges:
  - from: "start"
    to: "analysis1"
  - from: "start"
    to: "analysis2"
  - from: "analysis1"
    to: "combine"
  - from: "analysis2"
    to: "combine"
```

## Tips for Learning

1. **Start simple**: Begin with `basic_agent.py`
2. **Experiment**: Modify the examples and see what happens
3. **Read the code**: Comments explain what's happening
4. **Check the docs**: Refer to the guides for detailed explanations
5. **Build incrementally**: Add complexity step by step

## Next Steps

After exploring these examples:

- 📖 Read the [Quick Start Guide](../guides/quickstart.md)
- 🤖 Learn about [Building Agents](../guides/first-agent.md)
- 🔄 Learn about [Creating Workflows](../guides/first-workflow.md)
- 📚 Explore the [API Reference](../api/index.md)

## Contributing Examples

Have a great example? We'd love to add it!

1. Place your example in the appropriate directory
2. Include clear comments explaining the code
3. Add a brief description to this README
4. Submit a pull request

## Troubleshooting

### "No module named 'victor'"

Install Victor:
```bash
pip install victor-ai
```

### "API key not found"

Set your API key:
```bash
export OPENAI_API_KEY="your-key-here"
```

### Import errors

Make sure you're in the examples directory:
```bash
cd docs/examples/agents
python basic_agent.py
```

## License

These examples are part of the Victor framework and are licensed under Apache 2.0.
