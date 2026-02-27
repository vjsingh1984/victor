# Victor Examples

This directory contains examples and learning materials for the Victor AI Framework.

## Structure

- **playground/** - Interactive CLI for learning and experimentation
- **basic/** - Simple getting-started examples
- **workflows/** - StateGraph workflow examples

## Quick Start

### Playground CLI

The playground provides an interactive learning environment:

```bash
# Install with examples support
pip install victor-ai[examples]

# Run a demo example
victor-examples demo hello
victor-examples demo joke
victor-examples demo code

# Start interactive mode
victor-examples interactive

# List all examples
victor-examples examples

# Show info
victor-examples info
```

### Running Directly from Source

You can also run examples directly from the source tree:

```bash
cd examples/playground
python cli.py demo hello
```

## Examples

### Playground Demos

| Example | Description |
|---------|-------------|
| `hello` | A simple greeting |
| `joke` | Tell a joke |
| `code` | Generate Python code |
| `explain` | Explain a concept |
| `creative` | Creative writing |

### Interactive Mode

Start an interactive chat session:

```bash
victor-examples interactive --provider openai --tools full
```

**Providers**: `openai`, `anthropic`, `ollama`

**Tools**: `minimal`, `default`, `full`

## Requirements

- Python 3.10+
- Victor AI Framework
- API key for chosen provider (except Ollama)
- Optional dependencies: `typer`, `rich`

## License

Apache License 2.0
