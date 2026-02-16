# Installation Guide

This guide covers all methods for installing Victor, from the simplest pip install to development setups and Docker containers.

## System Requirements

Before installing Victor, ensure your system meets these requirements:

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **Python** | 3.10 | 3.11+ |
| **RAM** | 4 GB | 8 GB+ |
| **Disk Space** | 500 MB | 1 GB (for local models: 8-32 GB) |
| **OS** | Linux, macOS, Windows (WSL2) | Linux or macOS |

**Note**: Windows users should use WSL2 (Windows Subsystem for Linux) for the best experience.

---

## Installation Methods

Choose the installation method that best fits your use case:

| Method | Command | Best For |
|--------|---------|----------|
| **pipx** (Recommended) | `pipx install victor-ai` | CLI users, isolated environment |
| **pip** | `pip install victor-ai` | Virtual environments, Python projects |
| **Docker** | `docker pull ghcr.io/vjsingh1984/victor` | Containers, isolated deployments |
| **Development** | `pip install -e ".[dev]"` | Contributors, local development |

### Method 1: pipx (Recommended for CLI Users)

pipx installs Victor in an isolated environment, preventing dependency conflicts with other Python packages.

```bash
# Install pipx if not already installed
pip install pipx
pipx ensurepath

# Install Victor
pipx install victor-ai

# Verify installation
victor --version
```

**Why pipx?**
- Isolates Victor's dependencies from your system Python
- Automatically adds `victor` to your PATH
- Easy upgrades with `pipx upgrade victor-ai`
- Clean uninstalls with `pipx uninstall victor-ai`

### Method 2: pip (Virtual Environment)

For use within a Python virtual environment or project:

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install Victor
pip install victor-ai

# Verify installation
victor --version
```

### Method 3: Docker

Run Victor in a container without installing Python dependencies:

```bash
# Pull the latest image
docker pull ghcr.io/vjsingh1984/victor:latest

# Run Victor interactively
docker run -it \
  -v ~/.victor:/root/.victor \
  -v "$(pwd)":/workspace \
  -w /workspace \
  ghcr.io/vjsingh1984/victor:latest chat

# With API keys
docker run -it \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  -v ~/.victor:/root/.victor \
  -v "$(pwd)":/workspace \
  -w /workspace \
  ghcr.io/vjsingh1984/victor:latest chat
```

**Docker Compose example:**
```yaml
# docker-compose.yml
version: '3.8'
services:
  victor:
    image: ghcr.io/vjsingh1984/victor:latest
    volumes:
      - ~/.victor:/root/.victor
      - .:/workspace
    working_dir: /workspace
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    stdin_open: true
    tty: true
```

### Method 4: Development Installation

For contributors or those who want to modify Victor:

```bash
# Clone the repository
git clone https://github.com/vjsingh1984/victor.git
cd victor

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode with all dependencies
pip install -e ".[dev]"

# Verify installation
victor --version
pytest tests/unit -v  # Run tests to verify
```

---

## Optional Dependencies

Victor has optional dependencies for specific features:

| Extra | Command | Features |
|-------|---------|----------|
| **api** | `pip install victor-ai[api]` | FastAPI HTTP server, REST API |
| **google** | `pip install victor-ai[google]` | Google Gemini provider |
| **checkpoints** | `pip install victor-ai[checkpoints]` | SQLite checkpoint persistence for workflows |
| **lang-all** | `pip install victor-ai[lang-all]` | All tree-sitter language grammars |
| **native** | `pip install victor-ai[native]` | Maturin for Rust extensions |
| **all** | `pip install victor-ai[all]` | All optional dependencies |

**Install multiple extras:**
```bash
pip install "victor-ai[api,google,checkpoints]"
```

---

## Post-Installation Setup

### Step 1: Initialize Victor

After installation, initialize Victor to create configuration files:

```bash
victor init
```

This creates:
- `~/.victor/profiles.yaml` - Provider and model profiles
- `~/.victor/config.yaml` - Global settings

### Step 2: Verify Installation

Run these commands to verify Victor is working:

```bash
# Check version
victor --version

# List available providers
victor providers

# List available tools
victor tools
```

**Expected output:**
```
Victor AI v0.x.x
22 providers available
33 tools available
```

### Step 3: Choose Your Model

Victor needs an LLM to function. You have two options:

#### Option A: Local Model (No API Key Required)

**Ollama** is the easiest way to run local models:

```bash
# Install Ollama
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama server
ollama serve

# In another terminal, pull a model
ollama pull qwen2.5-coder:7b

# Test with Victor
victor chat "Hello, Victor!"
```

**Other local options:**
- **LM Studio**: GUI-based, download from [lmstudio.ai](https://lmstudio.ai)
- **vLLM**: High-throughput, requires GPU (`pip install vllm`)
- **llama.cpp**: CPU-friendly, minimal footprint

#### Option B: Cloud Provider (API Key Required)

Set up a cloud provider for more powerful models:

```bash
# Anthropic (Claude)
export ANTHROPIC_API_KEY=sk-ant-your-key-here
victor chat --provider anthropic "Hello, Victor!"

# OpenAI (GPT-4)
export OPENAI_API_KEY=sk-proj-your-key-here
victor chat --provider openai "Hello, Victor!"

# Google (Gemini)
export GOOGLE_API_KEY=your-key-here
victor chat --provider google "Hello, Victor!"
```

**Secure key storage (recommended):**
```bash
# Store in system keyring
victor keys --set anthropic --keyring
victor keys --set openai --keyring
victor keys --set google --keyring
```

---

## Platform-Specific Instructions

### macOS

```bash
# Install Python 3.11+ if needed
brew install python@3.11

# Install pipx
brew install pipx
pipx ensurepath

# Install Victor
pipx install victor-ai

# Optional: Install Ollama for local models
brew install ollama
```

### Linux (Ubuntu/Debian)

```bash
# Install Python 3.11+ if needed
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip

# Install pipx
python3.11 -m pip install --user pipx
python3.11 -m pipx ensurepath

# Restart terminal, then install Victor
pipx install victor-ai

# Optional: Install Ollama for local models
curl -fsSL https://ollama.com/install.sh | sh
```

### Windows (WSL2)

Victor works best on Windows through WSL2:

```bash
# Install WSL2 (from PowerShell as Administrator)
wsl --install

# After restart, open Ubuntu terminal and update
sudo apt update && sudo apt upgrade

# Install Python and pipx
sudo apt install python3.11 python3.11-venv python3-pip
python3.11 -m pip install --user pipx
python3.11 -m pipx ensurepath

# Restart terminal, then install Victor
pipx install victor-ai
```

---

## Troubleshooting

### "command not found: victor"

**Cause**: Victor is not in your PATH.

**Solutions:**

1. **If using pipx:**
   ```bash
   pipx ensurepath
   source ~/.bashrc  # or ~/.zshrc
   ```

2. **If using pip:**
   ```bash
   # Add pip user bin to PATH
   export PATH="$HOME/.local/bin:$PATH"
   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
   source ~/.bashrc
   ```

### "Package requires Python >= 3.10"

**Cause**: Your Python version is too old.

**Solution:**
```bash
# Check current version
python --version

# Install Python 3.11
# macOS: brew install python@3.11
# Ubuntu: sudo apt install python3.11

# Use specific version
python3.11 -m pip install victor-ai
```

### "Could not build wheels"

**Cause**: Missing build tools.

**Solutions:**

```bash
# macOS
xcode-select --install

# Ubuntu/Debian
sudo apt install build-essential python3-dev

# Then retry installation
pip install victor-ai
```

### "Permission denied"

**Cause**: Trying to install globally without sudo.

**Solution**: Use pipx or a virtual environment (never use sudo with pip):
```bash
# Recommended
pipx install victor-ai

# Alternative
python -m venv .venv
source .venv/bin/activate
pip install victor-ai
```

### Ollama Connection Errors

**Cause**: Ollama server not running or wrong port.

**Solutions:**
```bash
# Start Ollama server
ollama serve

# Check if running
curl http://localhost:11434/api/version

# Pull a model if needed
ollama pull qwen2.5-coder:7b

# Set custom host if using non-default port
export OLLAMA_HOST=127.0.0.1:11434
```

---

## Upgrading Victor

### pipx
```bash
pipx upgrade victor-ai
```

### pip
```bash
pip install --upgrade victor-ai
```

### Docker
```bash
docker pull ghcr.io/vjsingh1984/victor:latest
```

### Development
```bash
git pull origin main
pip install -e ".[dev]"
```

---

## Uninstalling Victor

### pipx
```bash
pipx uninstall victor-ai
```

### pip
```bash
pip uninstall victor-ai
```

### Docker
```bash
docker rmi ghcr.io/vjsingh1984/victor:latest
```

### Remove configuration
```bash
rm -rf ~/.victor
```

---

## Next Steps

After installation:

1. **[Quickstart](quickstart.md)** - Run your first conversation
2. **[Configuration](configuration.md)** - Customize settings and profiles
3. **[User Guide](../user-guide/)** - Learn daily usage patterns

---

**Having issues?** See the [Troubleshooting Guide](../user-guide/troubleshooting.md) or [open an issue](https://github.com/vjsingh1984/victor/issues).
