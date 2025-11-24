# Quick Start Guide

Get up and running with CodingAgent in minutes!

## Prerequisites

1. **Python 3.10+** installed
2. **Ollama** running locally (since you already have this!)
   - Verify: `ollama list` should show your installed models
3. A code editor (VS Code, PyCharm, etc.)

## Installation

### Step 1: Create Virtual Environment

```bash
cd /Users/vijaysingh/code/codingagent
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 2: Install CodingAgent

```bash
# Install in editable mode with all dependencies
pip install -e ".[dev]"
```

### Step 3: Initialize Configuration

```bash
# This creates ~/.codingagent/profiles.yaml with default settings
codingagent init
```

## Verify Ollama is Ready

```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Pull a recommended model if you don't have one
ollama pull qwen2.5-coder:7b
# or
ollama pull deepseek-coder:6.7b
# or
ollama pull codellama:13b
```

## First Run

### Option 1: Interactive Mode

```bash
# Start interactive REPL
codingagent

# Now you can chat with the agent!
```

Example session:
```
You: Write a Python function to calculate the factorial of a number
Assistant: [Generates code and can even write it to a file for you]

You: Now write unit tests for that function
Assistant: [Creates test file with pytest tests]
```

### Option 2: One-Shot Command

```bash
# Execute a single command
codingagent "Write a hello world FastAPI application"
```

### Option 3: Run Example Script

```bash
# Test with the example script
python examples/simple_chat.py
```

## Configure Your Profile

Edit `~/.codingagent/profiles.yaml`:

```yaml
profiles:
  default:
    provider: ollama
    model: qwen2.5-coder:7b  # Change to your preferred model
    temperature: 0.7
    max_tokens: 4096

providers:
  ollama:
    base_url: http://localhost:11434
```

## Usage Examples

### 1. Code Generation

```bash
codingagent "Create a REST API endpoint for user authentication with FastAPI"
```

### 2. Code Review

```bash
codingagent "Review the code in src/main.py and suggest improvements"
```

### 3. Debugging

```bash
codingagent "Analyze the error in error.log and suggest fixes"
```

### 4. File Operations

The agent can read and write files:
```
You: Read the contents of README.md and summarize it
You: Create a new file called hello.py with a simple greeting function
You: List all Python files in the current directory
```

### 5. Execute Commands

The agent can run bash commands:
```
You: Run pytest and show me the results
You: Check the git status and tell me what changed
You: Install the requests library using pip
```

## Available Commands in Interactive Mode

- `exit` or `quit` - Exit the REPL
- `clear` - Reset conversation history
- Just type your message and press Enter to chat

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov

# Run specific test
pytest tests/unit/test_ollama_provider.py
```

## Troubleshooting

### "Connection refused" error

**Problem**: Can't connect to Ollama
**Solution**: Make sure Ollama is running:
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if needed
ollama serve
```

### Model not found

**Problem**: "Model not found: qwen2.5-coder:7b"
**Solution**: Pull the model first:
```bash
ollama pull qwen2.5-coder:7b
# or choose a different model you have installed
ollama list
```

### Import errors

**Problem**: `ModuleNotFoundError`
**Solution**: Make sure you installed in editable mode:
```bash
pip install -e ".[dev]"
```

## Next Steps

### 1. Add More Models

```bash
# Pull additional models
ollama pull llama3:8b
ollama pull deepseek-coder:6.7b
ollama pull mistral:7b
```

Then add profiles in `~/.codingagent/profiles.yaml`:
```yaml
profiles:
  llama:
    provider: ollama
    model: llama3:8b

  deepseek:
    provider: ollama
    model: deepseek-coder:6.7b
```

Use with: `codingagent --profile llama`

### 2. Add API Keys for Frontier Models

Create `.env` file:
```bash
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
```

Update profiles.yaml:
```yaml
profiles:
  claude:
    provider: anthropic
    model: claude-sonnet-4-5

providers:
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
```

### 3. Explore Examples

```bash
# Check out the examples directory
ls examples/

# Run the simple chat example
python examples/simple_chat.py
```

### 4. Contribute

Check out [CONTRIBUTING.md](CONTRIBUTING.md) to:
- Add new providers (OpenAI, Google, etc.)
- Create custom tools
- Improve documentation
- Fix bugs

## Common Use Cases

### Create a New Project

```bash
codingagent "Create a new FastAPI project structure with:
- User authentication
- Database models with SQLAlchemy
- API endpoints for CRUD operations
- Pytest test suite
- Docker configuration"
```

### Refactor Code

```bash
codingagent "Refactor the code in src/legacy.py to use modern Python 3.11 features and improve performance"
```

### Generate Documentation

```bash
codingagent "Generate comprehensive docstrings for all functions in src/utils.py"
```

### Debug Issues

```bash
codingagent "Analyze the test failures in test_results.txt and fix the issues"
```

## Tips for Best Results

1. **Be Specific**: The more details you provide, the better the results
2. **Iterate**: Ask follow-up questions to refine the output
3. **Use Tools**: The agent can read/write files and execute commands
4. **Context**: Provide file paths or code snippets for context
5. **Test Locally**: Use Ollama for development to save costs

## Get Help

- **Documentation**: Check [README.md](README.md) and [DESIGN.md](DESIGN.md)
- **Issues**: Open an issue on GitHub
- **Examples**: Look at `examples/` directory

## What's Next?

The project roadmap includes:
- [ ] Anthropic Claude integration
- [ ] OpenAI GPT integration
- [ ] Google Gemini integration
- [ ] LMStudio and vLLM support
- [ ] MCP (Model Context Protocol) support
- [ ] Multi-agent collaboration
- [ ] Context caching
- [ ] Web UI

Contributions are welcome!

---

Happy coding with CodingAgent! 🤖
