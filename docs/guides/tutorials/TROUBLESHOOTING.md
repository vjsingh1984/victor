# Victor AI - Troubleshooting Guide

This guide helps you diagnose and fix common issues with Victor AI.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Provider Connection Issues](#provider-connection-issues)
3. [Performance Issues](#performance-issues)
4. [Tool Issues](#tool-issues)
5. [Memory Issues](#memory-issues)
6. [Configuration Issues](#configuration-issues)
7. [Debugging Tips](#debugging-tips)

## Installation Issues

### Issue: "No module named 'victor'"

**Symptoms:**
```
ModuleNotFoundError: No module named 'victor'
```

**Solutions:**

1. **Install in development mode:**
```bash
cd /path/to/victor
pip install -e ".[dev]"
```

2. **Check Python path:**
```bash
which python
python -c "import sys; print(sys.path)"
```

3. **Reinstall Victor:**
```bash
pip uninstall victor-ai
pip install -e ".[dev]"
```

### Issue: "Command not found: victor"

**Symptoms:**
```
bash: victor: command not found
```

**Solutions:**

1. **Check installation:**
```bash
pip show victor-ai
```

2. **Verify PATH:**
```bash
echo $PATH | grep -o '[^:]*bin[^:]*'
```

3. **Reinstall:**
```bash
pip install --force-reinstall victor-ai
```

### Issue: Missing dependencies

**Symptoms:**
```
ImportError: cannot import name 'something' from 'victor'
```

**Solutions:**

1. **Install with all dependencies:**
```bash
pip install -e ".[dev]"
```

2. **Install specific extras:**
```bash
pip install victor-ai[google,api,checkpoints]
```

3. **Update dependencies:**
```bash
pip install --upgrade -e ".[dev]"
```

## Provider Connection Issues

### Issue: "Connection refused" - Ollama

**Symptoms:**
```
ConnectionError: Error connecting to http://localhost:11434
```

**Solutions:**

1. **Start Ollama:**
```bash
ollama serve
```

2. **Check Ollama status:**
```bash
ps aux | grep ollama
curl http://localhost:11434/api/tags
```

3. **Restart Ollama:**
```bash
pkill ollama
ollama serve
```

4. **Verify Ollama installation:**
```bash
which ollama
ollama --version
```

### Issue: "Connection refused" - Local Providers (LMStudio, vLLM)

**Symptoms:**
```
ConnectionError: Error connecting to http://localhost:1234
```

**Solutions:**

1. **LMStudio:**
```bash
# Check if LMStudio is running
# Start LMStudio application
# Verify the API server is enabled
curl http://localhost:1234/v1/models
```

2. **vLLM:**
```bash
# Start vLLM server
python -m vllm.entrypoints.openai.api_server --model /path/to/model

# Check status
curl http://localhost:8000/v1/models
```

### Issue: API Key Errors - Cloud Providers

**Symptoms:**
```
AuthenticationError: Invalid API key
```

**Solutions:**

1. **Verify API key is set:**
```bash
echo $ANTHROPIC_API_KEY
echo $OPENAI_API_KEY
echo $GOOGLE_API_KEY
```

2. **Set API key:**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."
```

3. **Add to `.env` file:**
```bash
# Create .env file in project root
cat > .env << EOF
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
EOF

# Load .env
source .env
```

4. **Test API key:**
```bash
# Anthropic
curl -H "x-api-key: $ANTHROPIC_API_KEY" \
     https://api.anthropic.com/v1/messages

# OpenAI
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models
```

### Issue: Rate Limiting

**Symptoms:**
```
RateLimitError: Rate limit exceeded
```

**Solutions:**

1. **Wait and retry:**
```bash
# Victor has built-in retry logic
# Just wait a few minutes and try again
```

2. **Use different provider:**
```bash
# Switch to Ollama (no rate limits)
victor chat --provider ollama
```

3. **Upgrade API plan:**
```bash
# Upgrade your API tier for higher limits
# Check provider dashboard for details
```

4. **Reduce request rate:**
```python
# Add delay between requests
import time
time.sleep(1)  # 1 second delay
```

### Issue: Timeout Errors

**Symptoms:**
```
TimeoutError: Request timed out
```

**Solutions:**

1. **Increase timeout:**
```yaml
# In .victor/config.yaml
timeout: 120  # seconds
```

2. **Check network connection:**
```bash
ping api.anthropic.com
ping api.openai.com
```

3. **Use VPN if needed:**
```bash
# Some regions may block certain APIs
# Consider using a VPN
```

4. **Try different provider:**
```bash
victor chat --provider google  # Often has better connectivity
```

## Performance Issues

### Issue: Slow Response Times

**Symptoms:**
- Responses take >30 seconds
- High latency

**Solutions:**

1. **Use faster model:**
```bash
# Use cheaper/faster models
victor chat --provider openai --model gpt-3.5-turbo
victor chat --provider ollama --model qwen2.5-coder:7b
```

2. **Reduce context:**
```bash
# Clear conversation history
victor chat --clear-history

# Limit file analysis
victor chat "Only analyze src/main.py"
```

3. **Enable caching:**
```yaml
# In .victor/config.yaml
cache:
  enabled: true
  ttl: 3600  # 1 hour
```

4. **Use native extensions:**
```bash
pip install victor-ai[native]
```

5. **Reduce tool usage:**
```yaml
# In .victor/config.yaml
tools:
  enabled:
    - read_file
    - search_code
  disabled:
    - web_search
    - run_command
```

### Issue: High Memory Usage

**Symptoms:**
- Victor using >4GB RAM
- System slowdown

**Solutions:**

1. **Clear cache:**
```bash
rm -rf ~/.victor/cache/*
```

2. **Reduce context window:**
```yaml
# In .victor/config.yaml
max_tokens: 4096  # Reduce from default
```

3. **Use smaller model:**
```bash
# Use 7B model instead of 70B
victor chat --provider ollama --model qwen2.5-coder:7b
```

4. **Disable features:**
```yaml
# In .victor/config.yaml
features:
  semantic_search: false
  embeddings: false
```

### Issue: Slow Code Analysis

**Symptoms:**
- Codebase search takes >60 seconds
- File indexing is slow

**Solutions:**

1. **Use native extensions:**
```bash
pip install victor-ai[native]
```

2. **Index specific directories:**
```bash
# Only index src/, not all files
victor chat --index-path src/
```

3. **Exclude large files:**
```yaml
# In .victor/config.yaml
index:
  exclude:
    - "*.log"
    - "*.db"
    - "node_modules/"
    - "__pycache__/"
```

4. **Use parallel processing:**
```yaml
# In .victor/config.yaml
parallel:
  enabled: true
  max_workers: 4
```

## Tool Issues

### Issue: Tool Execution Failed

**Symptoms:**
```
ToolExecutionError: Tool 'run_command' failed
```

**Solutions:**

1. **Check tool permissions:**
```yaml
# In .victor/config.yaml
tools:
  run_command:
    enabled: true
    allowed_commands:
      - pytest
      - mypy
      - black
```

2. **Verify tool dependencies:**
```bash
# Install required tools
pip install pytest mypy black
```

3. **Check tool logs:**
```bash
# View tool execution logs
cat ~/.victor/logs/tools.log
```

4. **Test tool manually:**
```bash
# Test command outside Victor
pytest tests/
```

### Issue: Web Search Not Working

**Symptoms:**
```
ToolExecutionError: Web search failed
```

**Solutions:**

1. **Check internet connection:**
```bash
ping duckduckgo.com
curl https://duckduckgo.com
```

2. **Disable if offline:**
```yaml
# In .victor/config.yaml
tools:
  web_search:
    enabled: false
```

3. **Use air-gapped mode:**
```bash
export VICTOR_AIRGAPPED=true
victor chat
```

### Issue: File Access Denied

**Symptoms:**
```
PermissionError: Permission denied: '/etc/hosts'
```

**Solutions:**

1. **Check file permissions:**
```bash
ls -la /path/to/file
```

2. **Run with appropriate permissions:**
```bash
# Use sudo for system files (use with caution)
sudo victor chat
```

3. **Configure allowed paths:**
```yaml
# In .victor/config.yaml
tools:
  read_file:
    allowed_paths:
      - /home/user/project
      - /tmp
    denied_paths:
      - /etc
      - /root
```

## Memory Issues

### Issue: Context Window Exceeded

**Symptoms:**
```
ValidationError: Context window exceeded (120000 tokens > 100000 limit)
```

**Solutions:**

1. **Reduce input size:**
```bash
# Analyze specific files, not entire codebase
victor chat "Analyze only src/main.py"
```

2. **Clear conversation history:**
```bash
victor chat --clear-history "New question"
```

3. **Use model with larger context:**
```bash
# Gemini supports 1M tokens
victor chat --provider google --model gemini-1.5-pro
```

4. **Split into smaller tasks:**
```bash
# Instead of:
victor chat "Analyze entire codebase"

# Use:
victor chat "List all Python files"
victor chat "Analyze file1.py"
victor chat "Analyze file2.py"
```

### Issue: Embedding Cache Too Large

**Symptoms:**
- `~/.victor/cache/` using >1GB disk space

**Solutions:**

1. **Clear cache:**
```bash
rm -rf ~/.victor/cache/*
```

2. **Set cache limit:**
```yaml
# In .victor/config.yaml
cache:
  max_size_mb: 500
  ttl: 3600
```

3. **Disable embeddings:**
```yaml
# In .victor/config.yaml
features:
  embeddings: false
```

## Configuration Issues

### Issue: Configuration Not Loading

**Symptoms:**
- Custom config not being used
- Default settings applied

**Solutions:**

1. **Check config location:**
```bash
# Victor looks for config in:
# 1. .victor/config.yaml (project)
# 2. ~/.victor/config.yaml (user)
# 3. Default config

# Verify config exists
ls -la .victor/config.yaml
ls -la ~/.victor/config.yaml
```

2. **Validate YAML syntax:**
```bash
# Check YAML is valid
python -c "import yaml; yaml.safe_load(open('.victor/config.yaml'))"
```

3. **Check for merge conflicts:**
```bash
# Victor merges project and user configs
# User config overrides project config
cat .victor/config.yaml
cat ~/.victor/config.yaml
```

4. **Use --config flag:**
```bash
victor chat --config /path/to/config.yaml
```

### Issue: Profiles Not Working

**Symptoms:**
- Profile not found
- Profile settings not applied

**Solutions:**

1. **Check profiles file:**
```bash
ls -la ~/.victor/profiles.yaml
```

2. **Validate profile YAML:**
```yaml
# ~/.victor/profiles.yaml
profiles:
  development:
    provider: ollama
    model: qwen2.5-coder:7b
    temperature: 0.7

providers:
  ollama:
    base_url: http://localhost:11434
```

3. **List available profiles:**
```bash
victor chat --list-profiles
```

4. **Test profile:**
```bash
victor chat --profile development
```

## Debugging Tips

### Enable Debug Logging

```bash
# Set environment variable
export VICTOR_DEBUG=true

# Or in Python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run Victor
victor chat
```

### Check Logs

```bash
# Victor logs location
ls -la ~/.victor/logs/

# View latest logs
tail -f ~/.victor/logs/victor.log
tail -f ~/.victor/logs/tools.log
tail -f ~/.victor/logs/providers.log
```

### Verbose Mode

```bash
# Run with verbose output
victor chat --verbose

# Very verbose
victor chat --verbose --verbose
```

### Test Components Individually

```python
# Test provider
from victor.providers import create_provider

provider = create_provider("ollama")
response = provider.chat("Test message")
print(response)
```

### Health Check

```bash
# Check Victor health
victor health-check

# Output:
# ✓ Provider: ollama (connected)
# ✓ Tools: 55 loaded
# ✓ Cache: working
# ✓ Configuration: valid
```

## Common Error Messages

### "Module not found"

**Cause:** Missing dependency
**Fix:** `pip install -e ".[dev]"`

### "Connection refused"

**Cause:** Provider not running
**Fix:** Start provider (e.g., `ollama serve`)

### "Invalid API key"

**Cause:** Wrong or missing API key
**Fix:** Set correct API key in environment

### "Context window exceeded"

**Cause:** Input too large for model
**Fix:** Reduce input size or use model with larger context

### "Tool execution failed"

**Cause:** Tool error or permission denied
**Fix:** Check tool permissions and dependencies

### "Rate limit exceeded"

**Cause:** Too many API requests
**Fix:** Wait, use different provider, or upgrade plan

## Getting Additional Help

If you can't resolve your issue:

1. **Check documentation:** `docs/`
2. **Search GitHub Issues:** Existing solutions
3. **Create minimal reproduction:** Isolate the problem
4. **File GitHub Issue:** Include logs and config
5. **Join community:** Ask for help

## Issue Template

When filing an issue, include:

```markdown
## Environment
- Victor version: `victor --version`
- Python version: `python --version`
- OS: `uname -a`
- Provider: ollama/openai/etc

## Configuration
```yaml
# .victor/config.yaml
provider: ...
model: ...
```

## Error Message
```
Paste error here
```

## Logs
```
Paste relevant logs from ~/.victor/logs/
```

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What should happen

## Actual Behavior
What actually happens
```

Happy troubleshooting! 🛠️

---

**Last Updated:** February 01, 2026
**Reading Time:** 3 minutes
