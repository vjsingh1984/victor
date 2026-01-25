# Troubleshooting

Solve common issues quickly.

## Installation Issues

### Python Version Too Old
```
Error: Python 3.10 or higher required
```

**Solution:**
```bash
# Check Python version
python --version

# Install Python 3.11+
# macOS
brew install python@3.11

# Ubuntu/Debian
sudo apt-get install python3.11

# Windows
# Download from python.org
```

### pip Not Found
```bash
# Ensure pip is installed
python -m ensurepip --upgrade

# Or use pipx
pip install pipx
pipx ensurepath
```

### Permission Denied
```bash
# Use user install
pip install --user victor-ai

# Or use virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install victor-ai
```

## Provider Issues

### API Key Not Working
```bash
# Verify key is set
echo $ANTHROPIC_API_KEY

# Test connection
victor chat --provider anthropic --test

# Check key format (should start with sk-)
```

### Provider Not Found
```
Error: Provider 'xyz' not found
```

**Solution:** Check available providers:
```bash
victor list providers
```

### Ollama Not Running
```bash
# Check Ollama status
ollama ps

# Start Ollama
ollama serve

# Pull a model
ollama pull qwen2.5-coder:7b
```

## Performance Issues

### Slow Response Time
**Possible causes:**
- Network latency
- Model size too large
- insufficient RAM

**Solutions:**
```bash
# Use faster provider (Groq: 300+ tok/s)
victor chat --provider groq --model llama3.1-70b

# Use smaller model
ollama pull qwen2.5-coder:3b

# Check RAM usage
ollama ps
```

### Out of Memory
```
Error: Out of memory
```

**Solutions:**
```bash
# Use smaller model (3B instead of 7B)
ollama pull qwen2.5-coder:3b

# Check available RAM
free -h  # Linux
vm_stat   # macOS
```

## Tool Issues

### Tool Not Found
```
Error: Tool 'xyz' not available
```

**Solution:** Check available tools:
```bash
victor list tools
```

### Tool Execution Failed
```bash
# Check tool is enabled
victor chat --tools all

# Check tool cost tier
victor chat --tool-budget 100
```

## Configuration Issues

### Profile Not Loading
```bash
# Check profile location
ls ~/.victor/profiles.yaml

# Verify YAML syntax
cat ~/.victor/profiles.yaml

# Reinitialize
victor init
```

### Environment Variables Not Set
```bash
# Check environment
env | grep VICTOR

# Set in shell profile
export ANTHROPIC_API_KEY=sk-...
echo 'export ANTHROPIC_API_KEY=sk-...' >> ~/.bashrc
source ~/.bashrc
```

## Docker Issues

### Volume Mount Permission Denied
```bash
# Fix permissions
docker run -it -v ~/.victor:/root/.victor --user $(id -u):$(id -g) ghcr.io/vjsingh1984/victor:latest
```

### Can't Connect to Ollama from Docker
```bash
# Use host networking
docker run --network host ghcr.io/vjsingh1984/victor:latest
```

## Getting Help

### Check Logs
```bash
# Enable verbose logging
export VICTOR_LOG_LEVEL=DEBUG
victor chat

# Check log location
ls ~/.victor/logs/
```

### Report Issues
1. Check [GitHub Issues](https://github.com/vjsingh1984/victor/issues)
2. Search existing issues first
3. Create new issue with:
   - Victor version: `victor --version`
   - Python version: `python --version`
   - OS: `uname -a`
   - Error message
   - Steps to reproduce

### Community
- [GitHub Discussions](https://github.com/vjsingh1984/victor/discussions)
- [Discord](https://discord.gg/...)

## Next Steps

- [Installation](installation.md) - Install Victor
- [First Run](first-run.md) - Get started
- [Configuration](./configuration.md) - Advanced setup
