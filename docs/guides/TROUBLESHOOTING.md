# Victor Troubleshooting Guide

This guide helps you diagnose and fix common issues with Victor. For interactive diagnostics, use `victor doctor`.

## Quick Reference

| Command | Purpose |
|---------|---------|
| `victor doctor` | Run comprehensive diagnostics |
| `victor config validate` | Validate configuration |
| `victor profile list` | List available profiles |
| `victor profile current` | Show current profile |

---

## Installation Issues

### Python Version Errors

**Problem**: `ModuleNotFoundError: No module named 'victor'`

**Symptoms**:
- Import errors when running `victor` commands
- Package not found after installation

**Diagnosis**:
```bash
python --version  # Should be 3.10+
which python
pip show victor-ai
```

**Solutions**:
1. Verify Python version (requires 3.10+):
   ```bash
   python3 --version
   # If < 3.10, install newer Python from python.org or use pyenv/conda
   ```

2. Reinstall Victor:
   ```bash
   pip uninstall victor-ai -y
   pip install victor-ai
   ```

3. Use pip with explicit Python:
   ```bash
   python3.11 -m pip install victor-ai
   ```

**Prevention**:
- Always use Python 3.10 or higher
- Use virtual environments: `python -m venv .venv`

### Dependency Installation Errors

**Problem**: `ERROR: Could not build wheels for...`

**Symptoms**:
- Installation fails during dependency build
- Missing system dependencies

**Diagnosis**:
```bash
# Check system dependencies
victor doctor --verbose
```

**Solutions**:
1. Install system dependencies (macOS):
   ```bash
   brew install python@3.11 rust cmake
   ```

2. Install system dependencies (Ubuntu/Debian):
   ```bash
   sudo apt-get update
   sudo apt-get install -y python3-dev python3-venv build-essential
   ```

3. Install system dependencies (Fedora/RHEL):
   ```bash
   sudo dnf install -y python3-devel gcc cmake rust
   ```

**Prevention**:
- Use pre-built wheels when available
- Keep system packages updated

---

## Configuration Issues

### API Key Not Found

**Problem**: `Error: API key not found for provider`

**Symptoms**:
- Provider authentication failures
- "Unauthorized" or "401" errors

**Diagnosis**:
```bash
# Check environment variables
env | grep API_KEY

# Or use doctor
victor doctor
```

**Solutions**:
1. Set API key (Anthropic):
   ```bash
   export ANTHROPIC_API_KEY=your_key_here
   ```

2. Set API key (OpenAI):
   ```bash
   export OPENAI_API_KEY=your_key_here
   ```

3. Add to `~/.bashrc` or `~/.zshrc`:
   ```bash
   echo 'export ANTHROPIC_API_KEY=your_key_here' >> ~/.bashrc
   source ~/.bashrc
   ```

4. Configure in profile YAML (`~/.victor/profiles.yaml`):
   ```yaml
   providers:
     anthropic:
       api_key: your_key_here
   ```

**Prevention**:
- Use `victor init --wizard` for guided setup
- Run `victor doctor` to verify configuration

### Provider Not Available

**Problem**: `Provider 'xyz' is not available`

**Symptoms**:
- Provider not recognized
- Import errors for provider module

**Diagnosis**:
```bash
# List available providers
victor providers list

# Check provider installation
pip show victor-ai
```

**Solutions**:
1. Use a supported provider:
   ```bash
   # Available: anthropic, openai, google, ollama, lmstudio, vllm, etc.
   victor chat --provider anthropic
   ```

2. Install additional dependencies if needed:
   ```bash
   pip install victor-ai[all]  # Includes all optional dependencies
   ```

**Prevention**:
- Check provider compatibility before setup
- Use `victor providers list` to see available providers

### Invalid Configuration

**Problem**: `Configuration validation failed`

**Symptoms**:
- Settings errors at startup
- YAML parsing errors

**Diagnosis**:
```bash
# Validate configuration
victor config validate --verbose

# Check YAML syntax
python -c "import yaml; yaml.safe_load(open('~/.victor/profiles.yaml'))"
```

**Solutions**:
1. Fix YAML syntax errors:
   ```yaml
   # Correct:
   profiles:
     default:
       provider: ollama

   # Incorrect (missing colon):
   profiles
     default
       provider ollama
   ```

2. Validate with `victor config validate`:
   ```bash
   victor config validate --verbose
   ```

3. Use profile presets for quick setup:
   ```bash
   victor profile apply basic
   ```

**Prevention**:
- Use YAML linters in your editor
- Run `victor config validate` after changes
- Use configuration profiles as starting points

---

## Runtime Issues

### Ollama Connection Failed

**Problem**: `Failed to connect to Ollama`

**Symptoms**:
- "Ollama is not running" errors
- Connection refused

**Diagnosis**:
```bash
# Check Ollama status
ollama list

# Check if service is running
ps aux | grep ollama
```

**Solutions**:
1. Start Ollama:
   ```bash
   # macOS/Linux
   ollama serve

   # Or use service (macOS)
   brew services start ollama

   # Or use systemd (Linux)
   systemctl start ollama
   ```

2. Install Ollama if missing:
   ```bash
   # macOS/Linux
   curl -fsSL https://ollama.com/install.sh | sh

   # Or use Homebrew (macOS)
   brew install ollama
   ```

3. Verify connection:
   ```bash
   ollama list
   # Should show installed models
   ```

**Prevention**:
- Start Ollama on system boot (brew services, systemd)
- Use `victor doctor` to verify Ollama status

### Timeout Errors

**Problem**: `Request timeout after N seconds`

**Symptoms**:
- Requests timing out
- Slow responses from provider

**Diagnosis**:
```bash
# Check timeout settings
victor profile current

# Test provider connectivity
victor providers test <provider>
```

**Solutions**:
1. Increase timeout in profile:
   ```yaml
   profiles:
     default:
       provider: anthropic
       timeout: 120  # Increase timeout (seconds)
   ```

2. Check network connectivity:
   ```bash
   ping example.com
   curl -I https://api.anthropic.com
   ```

3. Use local provider for faster responses:
   ```bash
   victor chat --provider ollama
   ```

**Prevention**:
- Use appropriate timeouts for your network
- Consider local providers for faster responses
- Enable HTTP connection pooling

### Rate Limiting

**Problem**: `Rate limit exceeded` or `429` errors

**Symptoms**:
- Request throttling
- "Too many requests" errors

**Diagnosis**:
```bash
# Check provider status
victor providers test <provider>
```

**Solutions**:
1. Wait and retry:
   ```bash
   # Most rate limits reset after 1 minute
   sleep 60
   victor chat
   ```

2. Use different provider:
   ```bash
   # Switch to local provider temporarily
   victor chat --provider ollama
   ```

3. Reduce request frequency:
   - Fewer concurrent requests
   - Larger batch sizes instead of multiple small requests

**Prevention**:
- Monitor usage limits
- Use appropriate tier for your workload
- Implement caching to reduce API calls

---

## Performance Issues

### Slow First Response

**Problem**: First request takes very long

**Symptoms**:
- 10+ second delays on first message
- Slow startup

**Diagnosis**:
```bash
# Check if preloading is enabled
victor profile current

# Run diagnostics
victor doctor
```

**Solutions**:
1. Enable framework preloading:
   ```yaml
   settings:
     framework_preload_enabled: true
   ```

2. Enable tool selection cache:
   ```yaml
   settings:
     tool_selection_cache_enabled: true
   ```

3. Use advanced or expert profile:
   ```bash
   victor profile apply advanced
   ```

**Expected Improvements**:
- Preloading: 50-70% faster first requests
- Tool cache: 20-40% faster conversations
- HTTP pooling: 20-30% faster HTTP requests

### Slow Subsequent Responses

**Problem**: Responses are slow even after initial setup

**Symptoms**:
- Consistent delays on each message
- Poor performance throughout

**Diagnosis**:
```bash
# Check cache settings
victor profile current

# View performance metrics
victor observability metrics cache
```

**Solutions**:
1. Verify caching is enabled:
   ```yaml
   settings:
     tool_selection_cache_enabled: true
     http_connection_pool_enabled: true
   ```

2. Use smaller/faster models:
   ```yaml
   profiles:
     default:
       model: qwen2.5-coder:7b  # Faster than 14b/30b
   ```

3. Check provider performance:
   - Cloud providers may have latency
   - Local providers (Ollama) depend on hardware

**Prevention**:
- Use appropriate model size for your hardware
- Enable all performance optimizations
- Monitor with `victor observability metrics`

### High Memory Usage

**Problem**: Victor using too much memory

**Symptoms**:
- System becomes slow
- Out of memory errors

**Diagnosis**:
```bash
# Check memory usage
victor observability metrics

# System monitoring
top -p $(pgrep -f victor)
```

**Solutions**:
1. Reduce context window:
   ```yaml
   profiles:
     default:
       max_tokens: 4096  # Reduce from 8192+
   ```

2. Clear cache periodically:
   ```bash
   victor cache clear --all
   ```

3. Use lighter models:
   - 7B models instead of 30B+
   - Quantized models when available

**Prevention**:
- Monitor memory usage with `victor observability`
- Set appropriate max_tokens for your hardware
- Regular cache maintenance

---

## Tool Execution Issues

### Docker Errors

**Problem**: `Docker not running` or `Docker not installed`

**Symptoms**:
- Code execution fails
- Container creation errors

**Diagnosis**:
```bash
# Check Docker status
docker ps

# Check Docker installation
docker --version
```

**Solutions**:
1. Start Docker Desktop:
   - Launch Docker Desktop application
   - Wait for "Docker is running" status

2. Start Docker daemon:
   ```bash
   # macOS
   brew services start docker

   # Linux
   systemctl start docker
   ```

3. Install Docker if missing:
   ```bash
   # macOS
   brew install --cask docker

   # Linux
   curl -fsSL https://get.docker.com | sh
   ```

4. Verify permissions:
   ```bash
   # Add user to docker group (Linux)
   sudo usermod -aG docker $USER

   # Log out and back in
   ```

**Prevention**:
- Start Docker on system boot
- Ensure proper user permissions
- Run `victor doctor` to check Docker status

### File Permission Errors

**Problem**: `Permission denied` when reading/writing files

**Symptoms**:
- Cannot read project files
- Cannot write changes

**Diagnosis**:
```bash
# Check file permissions
ls -la <file>

# Check directory permissions
ls -ld $(dirname <file>)
```

**Solutions**:
1. Fix file permissions:
   ```bash
   chmod +r <file>
   ```

2. Fix directory permissions:
   ```bash
   chmod +x $(dirname <file>)
   ```

3. Run Victor with appropriate permissions:
   ```bash
   # If running as different user
   sudo -u victor victor chat
   ```

**Prevention**:
- Ensure project files are readable
- Run `victor doctor` to check permissions
- Avoid running as root unless necessary

### Git Operation Errors

**Problem**: Git operations fail with errors

**Symptoms**:
- Cannot commit changes
- Push/pull failures

**Diagnosis**:
```bash
# Check git status
git status

# Check remote configuration
git remote -v
```

**Solutions**:
1. Configure git identity:
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   ```

2. Fix force push protection:
   ```yaml
   settings:
     safety:
       allow_force_push: false  # Default: protected
   ```

3. Resolve merge conflicts:
   ```bash
   git pull --rebase
   # Resolve conflicts
   git rebase --continue
   ```

**Prevention**:
- Never use `git push --force` to main/master
- Run `victor doctor` to check git configuration
- Use feature branches instead of main

---

## Error Messages Reference

### Common Error Codes

| Error Code | Description | Solution |
|------------|-------------|----------|
| `NO_API_KEY` | No API key configured | Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` |
| `PROVIDER_CONNECTION_FAILED` | Cannot connect to provider | Check provider status, network connectivity |
| `CONFIGURATION_INVALID` | Configuration has errors | Run `victor config validate` |
| `TOOL_EXECUTION_FAILED` | Tool execution failed | Check Docker, file permissions |
| `TIMEOUT` | Request timeout | Increase timeout, check network |

### Provider-Specific Errors

#### Anthropic
- **401 Unauthorized**: Invalid API key
- **429 Rate Limit**: Too many requests
- **500 Internal Server Error**: Anthropic service issue

#### OpenAI
- **401 Unauthorized**: Invalid API key
- **429 Rate Limit**: Quota exceeded
- **503 Service Unavailable**: OpenAI service issue

#### Ollama
- **Connection refused**: Ollama not running
- **Model not found**: Model not downloaded

---

## Getting Help

### Diagnostic Commands

```bash
# Comprehensive diagnostics
victor doctor

# Configuration validation
victor config validate --verbose

# Test provider connectivity
victor providers test <provider>

# View current profile
victor profile current

# List available profiles
victor profile list
```

### Debug Mode

Enable verbose logging:

```bash
# Set environment variable
export VICTOR_DEBUG=1

# Or use --verbose flag
victor chat --verbose
```

### Where to Get Help

1. **Documentation**: https://github.com/victor-ai/victor
2. **Issues**: https://github.com/victor-ai/victor/issues
3. **Discussions**: https://github.com/victor-ai/victor/discussions
4. **Doctor Command**: `victor doctor --verbose`

### Reporting Issues

When reporting issues, include:

1. Victor version: `victor --version`
2. Python version: `python --version`
3. OS and version
4. Error message
5. Steps to reproduce
6. Doctor output: `victor doctor > doctor-output.txt`

```bash
# Collect diagnostic information
victor --version > issue-info.txt
python --version >> issue-info.txt
victor doctor --verbose >> issue-info.txt
```

---

## Quick Fix Checklist

When something doesn't work:

1. **Run diagnostics**: `victor doctor`
2. **Validate configuration**: `victor config validate`
3. **Check provider**: `victor providers test <provider>`
4. **Review profile**: `victor profile current`
5. **Enable debug**: `VICTOR_DEBUG=1 victor chat`

---

## Advanced Troubleshooting

### Enable Verbose Logging

```python
# In your Python code
import logging
logging.basicConfig(level=logging.DEBUG)

# Or via environment variable
import os
os.environ["VICTOR_DEBUG"] = "1"
```

### Check Database Status

```bash
# View database location
victor doctor --verbose | grep database

# Check database integrity
python -c "from victor.core.database import get_database; print(get_database().db_path)"
```

### Clear Cache

```bash
# Clear all caches
victor cache clear --all

# Clear specific cache
victor cache clear --tool-selection
victor cache clear --embeddings
```

### Reset Configuration

```bash
# Backup current configuration
cp ~/.victor/profiles.yaml ~/.victor/profiles.yaml.backup

# Apply default profile
victor profile apply basic

# Or manually edit
nano ~/.victor/profiles.yaml
```

---

## Preventive Maintenance

### Regular Tasks

**Weekly**:
- Run `victor doctor`
- Check for updates: `pip install --upgrade victor-ai`

**Monthly**:
- Clear old caches: `victor cache clear --old`
- Review logs for errors

**As Needed**:
- Update dependencies
- Review and update profiles
- Clean up old sessions

### Monitoring

```bash
# View performance metrics
victor observability metrics

# Check system resources
victor doctor --verbose | grep -E "memory|disk|cpu"
```

---

## Related Documentation

- **Configuration Guide**: See `victor profile list` for available profiles
- **Provider Setup**: See provider-specific documentation
- **API Reference**: See `victor docs --api`
- **Architecture Guide**: See `docs/architecture/` directory
