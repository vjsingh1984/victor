# Installation Guide

Complete guide for installing Victor on different platforms and environments.

## Table of Contents

- [Quick Install](#quick-install)
- [Installation Methods](#installation-methods)
  - [pip (Recommended)](#pip-recommended)
  - [pipx (Isolated)](#pipx-isolated)
  - [From Source](#from-source)
  - [Standalone Binary](#standalone-binary)
  - [Docker](#docker)
  - [Homebrew (macOS)](#homebrew-macos)
- [Platform-Specific Instructions](#platform-specific-instructions)
- [Post-Installation Setup](#post-installation-setup)
- [Troubleshooting](#troubleshooting)

## Quick Install

### One-Line Install (Recommended)

**macOS / Linux:**
```bash
curl -fsSL https://raw.githubusercontent.com/vijayksingh/victor/main/scripts/install/install.sh | bash
```

**Windows (PowerShell):**
```powershell
iwr -useb https://raw.githubusercontent.com/vijayksingh/victor/main/scripts/install/install.ps1 | iex
```

### pip Install

```bash
pip install victor-ai
```

## Installation Methods

### pip (Recommended)

The simplest way to install Victor if you have Python 3.10+ installed.

```bash
# Basic installation
pip install victor-ai

# With development tools
pip install "victor-ai[dev]"

# With all optional dependencies
pip install "victor-ai[all]"
```

**Requirements:**
- Python 3.10 or higher
- pip 21.0 or higher

**Verify installation:**
```bash
victor --version
```

### pipx (Isolated)

For users who want Victor installed in an isolated environment that doesn't affect system Python.

```bash
# Install pipx if not already installed
pip install pipx
pipx ensurepath

# Install Victor
pipx install victor-ai
```

**Benefits:**
- Isolated from system Python packages
- No dependency conflicts
- Easy to upgrade or uninstall

### From Source

For developers or users who want the latest unreleased features.

```bash
# Clone the repository
git clone https://github.com/vijayksingh/victor.git
cd victor

# Install in development mode
pip install -e ".[dev]"

# Or using make
make install-dev
```

**From a specific branch:**
```bash
pip install git+https://github.com/vijayksingh/victor.git@develop
```

### Standalone Binary

Pre-built binaries that don't require Python installation.

**Download from GitHub Releases:**
1. Go to [GitHub Releases](https://github.com/vjsingh1984/victor/releases/latest)
2. Download the appropriate binary for your platform:
   - `victor-macos-arm64.tar.gz` - macOS Apple Silicon
   - `victor-macos-x64.tar.gz` - macOS Intel
   - `victor-windows-x64.zip` - Windows x64
   - **Linux**: Use `pip install victor-ai` (binaries not available)

**Install on macOS/Linux:**
```bash
# Download and extract
curl -LO https://github.com/vijayksingh/victor/releases/latest/download/victor-macos-arm64.tar.gz
tar -xzf victor-macos-arm64.tar.gz

# Move to a directory in PATH
sudo mv victor /usr/local/bin/

# Verify
victor --version
```

**Install on Windows:**
1. Download `victor-windows-x64.zip`
2. Extract to a folder (e.g., `C:\Program Files\Victor`)
3. Add the folder to your system PATH
4. Open a new terminal and run `victor --version`

### Docker

For containerized environments or when you want complete isolation.

```bash
# Pull the latest image
docker pull vjsingh1984/victor:latest

# Run interactively
docker run -it --rm \
  -v $(pwd):/workspace \
  -v ~/.victor:/home/victor/.victor \
  vjsingh1984/victor:latest \
  victor chat

# Run as a service
docker run -d --name victor-server \
  -p 8765:8765 \
  -v $(pwd):/workspace \
  vjsingh1984/victor:latest \
  victor serve
```

**With Docker Compose:**
```yaml
# docker-compose.yml
version: '3.8'
services:
  victor:
    image: vjsingh1984/victor:latest
    volumes:
      - .:/workspace
      - victor-config:/home/victor/.victor
    ports:
      - "8765:8765"
    command: victor serve

volumes:
  victor-config:
```

```bash
docker-compose up -d
```

**Air-Gapped / Offline Docker:**

The Docker image includes pre-downloaded embedding models for offline use:
```bash
# Build with all models cached
docker build -t victor:airgapped .

# Run completely offline
docker run --network none -it victor:airgapped victor chat
```

### Homebrew (macOS/Linux)

For macOS and Linux users who prefer Homebrew.

```bash
# Add the tap (first time only)
brew tap vjsingh1984/tap

# Install
brew install victor

# Upgrade
brew upgrade victor
```

**Note**: The Homebrew formula auto-updates within 6 hours of new PyPI releases.

## Platform-Specific Instructions

### macOS

**Prerequisites:**
- macOS 11 (Big Sur) or later
- Xcode Command Line Tools: `xcode-select --install`

**Recommended method:** pip or Homebrew

```bash
# Using pip with Homebrew Python
brew install python@3.12
pip3 install victor-ai

# Or using Homebrew tap
brew tap vjsingh1984/tap
brew install victor
```

### Linux (Ubuntu/Debian)

**Prerequisites:**
```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3-pip git
```

**Install:**
```bash
pip3 install victor-ai
```

### Linux (Fedora/RHEL)

**Prerequisites:**
```bash
sudo dnf install -y python3.12 python3-pip git
```

**Install:**
```bash
pip3 install victor-ai
```

### Windows

**Prerequisites:**
1. Install Python 3.10+ from [python.org](https://www.python.org/downloads/)
2. During installation, check "Add Python to PATH"
3. Install Git from [git-scm.com](https://git-scm.com/download/win)

**Install via PowerShell:**
```powershell
pip install victor-ai
```

**Or use the standalone binary** (no Python required):
1. Download from [GitHub Releases](https://github.com/vijayksingh/victor/releases)
2. Extract and add to PATH

### WSL (Windows Subsystem for Linux)

Follow the Linux instructions within your WSL distribution:
```bash
# In WSL terminal
pip3 install victor-ai
```

## Post-Installation Setup

### 1. Initialize Configuration

```bash
# Initialize Victor in your project
cd /path/to/your/project
victor init
```

This creates:
- `.victor/init.md` - Project-specific instructions
- `.victor/` directory for local configuration

### 2. Configure API Keys

Create or edit `~/.victor/profiles.yaml`:

```yaml
default_profile: claude

profiles:
  claude:
    provider: anthropic
    model: claude-sonnet-4-20250514
    # API key can also be set via ANTHROPIC_API_KEY env var

  gpt4:
    provider: openai
    model: gpt-4-turbo
    # API key can also be set via OPENAI_API_KEY env var

  local:
    provider: ollama
    model: qwen2.5-coder:32b
    # No API key needed for local models
```

**Environment variables:**
```bash
# Add to ~/.bashrc or ~/.zshrc
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
```

### 3. Verify Installation

```bash
# Check version
victor --version

# Run health check
victor doctor

# Start interactive mode
victor chat
```

## Upgrading

### pip
```bash
pip install --upgrade victor-ai
```

### pipx
```bash
pipx upgrade victor-ai
```

### Homebrew
```bash
brew upgrade victor
```

### Docker
```bash
docker pull vjsingh1984/victor:latest
```

## Uninstalling

### pip
```bash
pip uninstall victor-ai
```

### pipx
```bash
pipx uninstall victor-ai
```

### Homebrew
```bash
brew uninstall victor
```

### Clean up configuration (optional)
```bash
rm -rf ~/.victor
```

## Troubleshooting

### Common Issues

#### "command not found: victor"

**Cause:** Python scripts directory not in PATH

**Solution:**
```bash
# Find where pip installs scripts
python3 -m site --user-base

# Add to PATH (add to ~/.bashrc or ~/.zshrc)
export PATH="$HOME/.local/bin:$PATH"
```

#### "ModuleNotFoundError: No module named 'victor'"

**Cause:** Victor installed in different Python environment

**Solution:**
```bash
# Check which Python is being used
which python3
python3 -m pip show victor

# Install in correct environment
/path/to/python3 -m pip install victor
```

#### "sentence-transformers" installation fails

**Cause:** Missing build dependencies

**Solution:**
```bash
# macOS
xcode-select --install

# Ubuntu/Debian
sudo apt install -y build-essential python3-dev

# Then reinstall
pip install --upgrade --force-reinstall sentence-transformers
```

#### Docker permission denied

**Solution:**
```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Log out and back in, or run:
newgrp docker
```

### Getting Help

- [GitHub Issues](https://github.com/vijayksingh/victor/issues)
- [Documentation](https://github.com/vijayksingh/victor/tree/main/docs)
- Run `victor --help` for command-line help

## Next Steps

- [Quick Start Guide](QUICKSTART.md) - Your first conversation with Victor
- [User Guide](../USER_GUIDE.md) - Complete usage documentation
- [Configuration](QUICKSTART.md#configuration) - Advanced configuration options
