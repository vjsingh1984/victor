<div align="center">

# Troubleshooting Hub

**Solutions to common Victor AI issues**

[![Support](https://img.shields.io/badge/support-active-green)](https://github.com/vjsingh1984/victor/issues)
[![Documentation](https://img.shields.io/badge/docs-troubleshooting-blue)](./user-guide/troubleshooting.md)

</div>

---

## Welcome to the Troubleshooting Hub

This hub consolidates all troubleshooting resources for Victor AI. Find quick solutions to common issues, diagnostic procedures, and where to get help.

### Quick Diagnostics

**Start here** if you're experiencing problems:

```bash
# 1. Check Victor version
victor --version

# 2. Check installed providers
victor providers

# 3. Test basic functionality
victor chat "Hello, Victor!" --provider ollama

# 4. Check configuration
victor config show

# 5. View logs
victor logs --tail 50
```

**Expected Output**:
- Version should be latest: `victor 0.5.1` or higher
- Providers should list 21 providers
- Chat should respond successfully
- Config should show your profiles
- Logs should show recent activity

---

## Common Issues by Category

### Installation Issues

| Issue | Solution | Link |
|-------|----------|------|
| Command not found | Check PATH, reinstall with pipx | [Solution](#installation-issues) |
| Python version incompatible | Upgrade to Python 3.10+ | [Solution](#installation-issues) |
| Dependency conflicts | Use virtual environment | [Solution](#installation-issues) |
| Permission denied | Use pipx or user install | [Solution](#installation-issues) |

### Provider Issues

| Issue | Solution | Link |
|-------|----------|------|
| API key not found | Set environment variable | [Solution](./user-guide/providers.md) |
| Provider connection failed | Check API key and network | [Solution](#provider-issues) |
| Model not found | Check model name spelling | [Solution](#provider-issues) |
| Rate limit exceeded | Wait or upgrade plan | [Solution](#provider-issues) |

### Tool Issues

| Issue | Solution | Link |
|-------|----------|------|
| Tool not found | Check tool name spelling | [Solution](#tool-issues) |
| Tool execution failed | Check tool permissions | [Solution](#tool-issues) |
| Tool timeout | Increase timeout setting | [Solution](#tool-issues) |
| Tool not available in air-gapped mode | Use local tools only | [Solution](#tool-issues) |

### Workflow Issues

| Issue | Solution | Link |
|-------|----------|------|
| YAML syntax error | Validate YAML syntax | [Solution](#workflow-issues) |
| Workflow not found | Check workflow path | [Solution](#workflow-issues) |
| Step handler error | Check handler implementation | [Solution](./extensions/step_handler_guide.md) |
| Cache invalidation | Clear workflow cache | [Solution](#workflow-issues) |

### Performance Issues

| Issue | Solution | Link |
|-------|----------|------|
| Slow response time | Check caching configuration | [Solution](./performance/README.md) |
| High memory usage | Adjust cache sizes | [Solution](./performance/optimization_guide.md) |
| CPU usage high | Check provider settings | [Solution](./performance/README.md) |
| Slow tool selection | Enable tool selection cache | [Solution](./performance/tool_selection_caching.md) |

---

## Detailed Troubleshooting Guides

### User Troubleshooting

Comprehensive troubleshooting for end users:

- [User Troubleshooting Guide](./user-guide/troubleshooting.md) - Common user issues
- [Detailed Troubleshooting](./user-guide/troubleshooting-detailed.md) - In-depth solutions
- [Legacy Troubleshooting](./user-guide/troubleshooting-legacy.md) - Archive of legacy issues

### Development Troubleshooting

Troubleshooting for developers:

- [Development Issues](./development/troubleshooting.md) - Development-specific issues
- [Testing Issues](./testing/README.md) - Test failures and debugging
- [CI/CD Issues](./ci_cd/README.md) - CI/CD pipeline problems

### Performance Troubleshooting

Performance-specific troubleshooting:

- [Performance Guide](./performance/optimization_guide.md) - Optimization techniques
- [Caching Issues](./performance/caching_strategy.md) - Cache problems
- [Benchmark Issues](./performance/BENCHMARK_SUMMARY.md) - Benchmark debugging

---

## Diagnostic Procedures

### Health Check

Run a comprehensive health check:

```bash
# 1. Check Victor installation
victor --version

# 2. Check Python environment
python --version
pip list | grep victor

# 3. Check configuration
victor config show

# 4. Test provider connection
victor chat --provider anthropic --model claude-sonnet-4-5 "test"

# 5. Check tools
victor tools list

# 6. Check workflows
victor workflow list

# 7. View logs
victor logs --tail 100
```

### Provider Diagnostics

Diagnose provider-specific issues:

```bash
# Test specific provider
victor chat --provider <provider> --model <model> "test"

# Check provider capabilities
victor providers --show-capabilities

# Test with verbose output
victor chat --provider <provider> --verbose "test"

# Check API key
echo $PROVIDER_API_KEY
```

### Tool Diagnostics

Diagnose tool-specific issues:

```bash
# List available tools
victor tools list

# Test specific tool
victor tools test <tool-name>

# Check tool permissions
victor tools check-permissions

# View tool details
victor tools info <tool-name>
```

### Workflow Diagnostics

Diagnose workflow issues:

```bash
# Validate workflow
victor workflow validate <workflow-path>

# Test workflow
victor workflow test <workflow-path>

# Clear workflow cache
victor workflow cache-clear

# View workflow logs
victor workflow logs <workflow-name>
```

---

## Error Messages

### Common Error Messages

| Error | Meaning | Solution |
|-------|---------|----------|
| `Command not found: victor` | Victor not in PATH | [Solution](#installation-issues) |
| `API key not found` | Missing API key | [Solution](./user-guide/providers.md) |
| `Model not found` | Invalid model name | [Solution](#provider-issues) |
| `Tool execution failed` | Tool error | [Solution](#tool-issues) |
| `YAML syntax error` | Invalid YAML | [Solution](#workflow-issues) |
| `Rate limit exceeded` | Too many requests | [Solution](#provider-issues) |
| `Connection timeout` | Network issue | [Solution](#provider-issues) |
| `Permission denied` | File permission issue | [Solution](#installation-issues) |
| `Module not found` | Missing dependency | [Solution](#installation-issues) |
| `Type error` | Code issue | [Solution](#development/troubleshooting.md) |

---

## Solutions by Issue Type

### Installation Issues

**Problem**: Command not found after installation

**Solution**:
```bash
# Check if Victor is installed
pip list | grep victor

# If not installed, install it
pipx install victor-ai

# Verify installation
which victor
```

**Problem**: Python version incompatible

**Solution**:
```bash
# Check Python version
python --version  # Must be 3.10+

# If too old, upgrade Python
# macOS: brew install python@3.11
# Ubuntu: sudo apt-get install python3.11
```

**Problem**: Permission denied during installation

**Solution**:
```bash
# Use pipx (recommended)
pipx install victor-ai

# Or use --user flag with pip
pip install --user victor-ai

# Or use virtual environment
python -m venv .venv
source .venv/bin/activate
pip install victor-ai
```

### Provider Issues

**Problem**: API key not found

**Solution**:
```bash
# Set API key as environment variable
export ANTHROPIC_API_KEY=sk-your-key-here

# Or add to ~/.bashrc or ~/.zshrc
echo 'export ANTHROPIC_API_KEY=sk-your-key-here' >> ~/.bashrc
source ~/.bashrc

# Or add to ~/.victor/config.yml
victor config set api_keys.anthropic sk-your-key-here
```

**Problem**: Provider connection failed

**Solution**:
```bash
# Check internet connection
ping api.anthropic.com

# Check API key is valid
echo $ANTHROPIC_API_KEY

# Test provider explicitly
victor chat --provider anthropic --model claude-sonnet-4-5 "test"

# Check provider status
victor providers --status
```

**Problem**: Model not found

**Solution**:
```bash
# List available models for provider
victor providers --models anthropic

# Use exact model name
victor chat --provider anthropic --model claude-sonnet-4-5-20250114 "test"

# Check model capabilities
victor providers --show-capabilities
```

### Tool Issues

**Problem**: Tool not found

**Solution**:
```bash
# List available tools
victor tools list

# Check tool name spelling
victor tools info read-file  # Use hyphens, not underscores

# Search for tools
victor tools search file
```

**Problem**: Tool execution failed

**Solution**:
```bash
# Check tool permissions
victor tools check-permissions

# Test tool
victor tools test read-file

# View tool details
victor tools info read-file

# Check if tool is available in current mode
victor config show
```

**Problem**: Tool not available in air-gapped mode

**Solution**:
```bash
# List air-gapped compatible tools
victor tools list --air-gapped

# Disable air-gapped mode if needed
victor config set airgapped_mode false

# Or use local tools only
victor chat --air-gapped-mode=false "Use web search tool"
```

### Workflow Issues

**Problem**: YAML syntax error

**Solution**:
```bash
# Validate workflow
victor workflow validate workflow.yaml

# Check YAML syntax
python -c "import yaml; yaml.safe_load(open('workflow.yaml'))"

# Use YAML linter
yamllint workflow.yaml
```

**Problem**: Workflow not found

**Solution**:
```bash
# Check workflow path
ls -la workflows/

# Use absolute path
victor workflow run /full/path/to/workflow.yaml

# List available workflows
victor workflow list
```

**Problem**: Step handler error

**Solution**:
```bash
# Check handler implementation
victor workflow validate workflow.yaml

# Test handler
victor workflow test workflow.yaml

# View handler documentation
# See: Step Handler Guide
# https://github.com/vjsingh1984/victor/blob/main/docs/extensions/step_handler_guide.md
```

---

## Getting Help

### Self-Service Resources

- [FAQ](./user-guide/faq.md) - Frequently asked questions
- [User Guide](./user-guide/index.md) - Complete user documentation
- [Architecture Docs](./architecture/README.md) - Architecture documentation
- [API Reference](./api/README.md) - API documentation

### Community Support

- [GitHub Issues](https://github.com/vjsingh1984/victor/issues) - Bug reports and feature requests
- [GitHub Discussions](https://github.com/vjsingh1984/victor/discussions) - Community discussions
- [Stack Overflow](https://stackoverflow.com/questions/tagged/victor-ai) - Q&A (coming soon)

### Reporting Issues

When reporting issues, include:

1. **Victor version**: `victor --version`
2. **Python version**: `python --version`
3. **Operating system**: `uname -a` (Linux/macOS) or `ver` (Windows)
4. **Error message**: Full error traceback
5. **Steps to reproduce**: Minimal reproduction case
6. **Configuration**: `victor config show` (sanitize API keys)

**Issue Template**:

```markdown
**Description**
Brief description of the issue

**Steps to Reproduce**
1. Step one
2. Step two
3. Step three

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- Victor version: 0.5.1
- Python version: 3.11.0
- OS: Ubuntu 22.04

**Error Message**
```
Full error traceback here
```

**Configuration**
```yaml
Sanitized configuration here
```
```

---

## Debug Mode

Enable debug mode for detailed logging:

```bash
# Enable debug logging
victor chat --debug "your message"

# Set debug level in config
victor config set logging.level DEBUG

# View debug logs
victor logs --tail 100 --level DEBUG
```

### Debug Output

Debug mode includes:
- HTTP requests and responses
- Tool execution details
- Provider communication
- Cache hits/misses
- Performance metrics
- Error stack traces

---

## Log Files

### Log Locations

- **Linux/macOS**: `~/.victor/logs/`
- **Windows**: `%APPDATA%\Victor\logs\`

### Log Files

- `victor.log` - Main application log
- `provider.log` - Provider-specific logs
- `tool.log` - Tool execution logs
- `workflow.log` - Workflow execution logs
- `error.log` - Error-specific logs

### Viewing Logs

```bash
# View recent logs
victor logs --tail 50

# Follow logs in real-time
victor logs --follow

# View specific log file
victor logs --file provider.log

# View logs for specific level
victor logs --level ERROR
```

---

## Performance Troubleshooting

### Slow Response Times

**Diagnose**:
```bash
# Check caching configuration
victor config show | grep cache

# Run performance benchmark
victor benchmark run --provider anthropic

# Check tool selection cache stats
victor stats --tool-selection
```

**Solutions**:
- Enable tool selection caching
- Increase cache sizes
- Use faster provider/model
- Reduce tool selection complexity
- Enable lazy loading

### High Memory Usage

**Diagnose**:
```bash
# Check memory usage
victor stats --memory

# Check cache sizes
victor config show | grep cache
```

**Solutions**:
- Reduce cache sizes
- Clear caches regularly
- Use streaming responses
- Disable unused tools

### High CPU Usage

**Diagnose**:
```bash
# Check CPU usage
victor stats --cpu

# Check provider settings
victor config show | grep provider
```

**Solutions**:
- Use fewer concurrent requests
- Adjust provider settings
- Use connection pooling
- Enable caching

---

## Known Issues

### Current Known Issues

Check [GitHub Issues](https://github.com/vjsingh1984/victor/issues) for current known issues.

### Common Workarounds

| Issue | Workaround |
|-------|------------|
| Provider timeout | Increase timeout in config |
| Tool not found | Use full tool name with namespace |
| Workflow cache stale | Clear cache with `victor workflow cache-clear` |
| Session not persisting | Check file permissions for `~/.victor/` |
| TUI rendering issues | Use `--no-tui` flag |

---

## Preventive Measures

### Best Practices

1. **Keep Victor Updated**:
   ```bash
   pipx upgrade victor-ai
   ```

2. **Use Virtual Environments**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install victor-ai
   ```

3. **Secure API Keys**:
   ```bash
   # Use environment variables
   export ANTHROPIC_API_KEY=sk-xxx

   # Or use Victor's secure storage
   victor config set api_keys.anthropic sk-xxx
   ```

4. **Regular Maintenance**:
   ```bash
   # Clear caches periodically
   victor cache clear-all

   # Check for updates
   pipx upgrade victor-ai

   # Review logs
   victor logs --tail 100
   ```

---

## Additional Resources

### Documentation

- [Documentation Index](./INDEX.md) - Complete documentation hub
- [Quick Start](./QUICKSTART.md) - Get started quickly
- [User Guide](./user-guide/index.md) - User documentation
- [Developer Guide](./DEVELOPER_ONBOARDING.md) - Developer documentation

### Architecture

- [Architecture Overview](./architecture/overview.md) - System architecture
- [Component Reference](./architecture/COMPONENT_REFERENCE.md) - Component docs
- [Design Patterns](./architecture/DESIGN_PATTERNS.md) - Design patterns

### Performance

- [Performance Guide](./performance/optimization_guide.md) - Optimization
- [Caching Strategy](./performance/caching_strategy.md) - Caching
- [Benchmarking](./performance/BENCHMARK_SUMMARY.md) - Benchmarks

---

<div align="center">

**Still having issues?**

[Open an Issue](https://github.com/vjsingh1984/victor/issues/new) •
[Join Discussions](https://github.com/vjsingh1984/victor/discussions) •
[Check FAQ](./user-guide/faq.md)

**[Back to Documentation Index](./INDEX.md)**

</div>
