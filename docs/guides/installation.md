# Installation Guide

Detailed installation instructions for Victor AI framework.

## System Requirements

### Minimum Requirements

- **Python**: 3.10 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: 2GB RAM minimum, 4GB+ recommended
- **Disk Space**: 500MB for package, 1GB+ for development dependencies

### Recommended

- **Python**: 3.11 or 3.12
- **Memory**: 8GB+ RAM
- **Disk Space**: 2GB+ for full development environment

## Verify Python Version

Before installing, check your Python version:

```bash
python --version
# or
python3 --version
```

You should see `Python 3.10.0` or higher.

**If you have Python < 3.10**:

**macOS** (using Homebrew):
```bash
brew install python@3.11
```

**Ubuntu/Debian**:
```bash
sudo apt update
sudo apt install python3.11
```

**Windows**: Download from [python.org](https://www.python.org/downloads/)

## Installation Methods

### Method 1: pip Install (Recommended)

Install from PyPI:

```bash
pip install victor-ai
```

### Method 2: Install with Extras

Install with development dependencies:

```bash
pip install "victor-ai[dev]"
```

Install with all optional dependencies:

```bash
pip install "victor-ai[all]"
```

Available extras:
- `dev` - Development tools (pytest, black, mypy, etc.)
- `viz` - Visualization tools (matplotlib, graphviz)
- `all` - All optional dependencies

### Method 3: Install from Source

For the latest development version:

```bash
# Clone the repository
git clone https://github.com/vjsingh1984/victor.git
cd victor

# Install in editable mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

## Provider Configuration

Victor supports multiple LLM providers. Configure your preferred provider:

### OpenAI

```bash
export OPENAI_API_KEY="sk-..."
```

Or in `~/.victor/settings.yaml`:

```yaml
providers:
  openai:
    api_key: "sk-..."
```

### Anthropic Claude

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Azure OpenAI

```bash
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
```

In settings.yaml:

```yaml
providers:
  azure:
    api_key: "your-api-key"
    endpoint: "https://your-resource.openai.azure.com"
    api_version: "2023-05-15"
```

### Google AI (Gemini)

```bash
export GOOGLE_API_KEY="..."
```

### Cohere

```bash
export COHERE_API_KEY="..."
```

### Local Models (Ollama)

```bash
# Install Ollama first
curl https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama2

# No API key needed!
```

## Verifying Installation

Test your installation:

```bash
python -c "import victor; print(victor.__version__)"
```

Expected output:
```
0.5.8
```

Run the built-in test:

```bash
python -m victor.tools.cli test
```

## Upgrading

To upgrade to the latest version:

```bash
pip install --upgrade victor-ai
```

To upgrade from source:

```bash
cd victor
git pull
pip install -e .
```

## Uninstalling

To remove Victor:

```bash
pip uninstall victor-ai
```

To also remove configuration files:

```bash
# Remove config directory
rm -rf ~/.victor

# Remove virtual environment (if using one)
rm -rf /path/to/venv
```

## Platform-Specific Notes

### macOS

#### Using Homebrew Python

If you installed Python via Homebrew, you might need to set the path:

```bash
export PATH="/opt/homebrew/bin:$PATH"
```

#### OpenSSL Issues (Apple Silicon)

If you encounter OpenSSL errors:

```bash
brew install openssl
export LDFLAGS="-L/opt/homebrew/opt/openssl/lib"
export CPPFLAGS="-I/opt/homebrew/opt/openssl/include"
pip install victor-ai
```

### Linux

#### Ubuntu/Debian Dependencies

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv build-essential
```

#### CentOS/RHEL Dependencies

```bash
sudo yum install python3 python3-pip python3-venv gcc
```

### Windows

#### Install Visual C++ Build Tools

Some packages require C++ compilation:

<https://visualstudio.microsoft.com/visual-cpp-build-tools/>

#### Use PowerShell for Installation

```powershell
# Install
pip install victor-ai

# Set environment variable
$env:OPENAI_API_KEY="your-key-here"

# Make persistent
[System.Environment]::SetEnvironmentVariable('OPENAI_API_KEY', 'your-key-here', 'User')
```

## Virtual Environments (Recommended)

Using a virtual environment keeps your dependencies isolated:

### venv

```bash
# Create virtual environment
python -m venv victor-env

# Activate
source victor-env/bin/activate  # Linux/macOS
victor-env\Scripts\activate     # Windows

# Install Victor
pip install victor-ai

# Deactivate when done
deactivate
```

### conda

```bash
# Create environment
conda create -n victor python=3.11

# Activate
conda activate victor

# Install Victor
pip install victor-ai

# Deactivate
conda deactivate
```

### poetry

```bash
# Create new project
poetry new my-project
cd my-project

# Add Victor
poetry add victor-ai

# Install dependencies
poetry install
```

## Docker Installation

For containerized deployment:

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install Victor
RUN pip install victor-ai

# Copy your application
COPY . .

# Set API key as environment variable
ENV OPENAI_API_KEY=""

CMD ["python", "app.py"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  victor-app:
    build: .
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - .:/app
```

Build and run:

```bash
docker-compose up
```

## Troubleshooting

### "No module named 'victor'"

**Cause**: Victor not installed or wrong Python environment

**Solutions**:
```bash
# Check which Python you're using
which python

# Install Victor
pip install victor-ai

# Or verify installation
python -m pip show victor-ai
```

### "Permission denied" when installing

**Solution**: Install to user directory:
```bash
pip install --user victor-ai
```

### "SSL certificate verification failed"

**Solution**:
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Reinstall with trusted host
pip install --trusted-host pypi.org --trusted-host pypi.python.org victor-ai
```

### "Failed building wheel for X"

**Cause**: Missing build dependencies

**Solution**:
```bash
# Ubuntu/Debian
sudo apt install build-essential python3-dev

# macOS
xcode-select --install

# Windows
# Install Visual C++ Build Tools
```

### "ImportError: cannot import name 'cached_property'"

**Cause**: Python version too old

**Solution**: Upgrade to Python 3.10+
```bash
python --version  # Should be 3.10+
```

### Slow installation

**Solution**: Use a faster package index:
```bash
pip install --index-url https://pypi.org/simple victor-ai
```

### Version conflicts

**Solution**: Use pip-check to find conflicts:
```bash
pip install pip-check
pip-check
```

## Development Installation

For contributing to Victor:

```bash
# Clone repository
git clone https://github.com/vjsingh1984/victor.git
cd victor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest
```

## Checking Your Installation

Run the diagnostic command:

```bash
victor doctor
```

This checks:
- ✅ Python version
- ✅ Package installation
- ✅ API key configuration
- ✅ Provider connectivity
- ✅ Common issues

Output example:
```
Victor Doctor - Installation Diagnostic
========================================

✅ Python: 3.11.4
✅ victor-ai: 0.5.8
✅ OpenAI API key: Configured
✅ OpenAI connection: Working
✅ Filesystem permissions: OK

All checks passed! Victor is ready to use.
```

## Next Steps

- 📖 [Quick Start Guide](quickstart.md) - Get started in 5 minutes
- 🤖 [First Agent Guide](first-agent.md) - Build your first agent
- 🔄 [First Workflow Guide](first-workflow.md) - Create workflows
- 📚 [API Reference](../api/index.md) - Full API documentation

## Additional Resources

- [PyPI Package](https://pypi.org/project/victor-ai/)
- [GitHub Repository](https://github.com/vjsingh1984/victor)
- [Documentation](https://victor-ai.readthedocs.io)
- [Discord Community](https://discord.gg/victor-ai)
