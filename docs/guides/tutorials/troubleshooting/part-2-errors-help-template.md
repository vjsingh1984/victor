# Victor AI - Troubleshooting Guide - Part 2

**Part 2 of 2:** Common Error Messages, Getting Additional Help, and Issue Template

---

## Navigation

- [Part 1: Common Issues](part-1-common-issues.md)
- **[Part 2: Errors, Help, Template](#)** (Current)
- [**Complete Guide](../TROUBLESHOOTING.md)**

---

## Common Error Messages

### "API key not found"

**Cause:** Missing or invalid API key

**Solution:**
```bash
# Set API key
export ANTHROPIC_API_KEY="sk-ant-..."
```

### "Connection timeout"

**Cause:** Network issues or provider unavailable

**Solution:**
```bash
# Check connection
ping api.anthropic.com

# Try different provider
export VICTOR_DEFAULT_PROVIDER="openai"
```

### "Tool execution failed"

**Cause:** Tool unavailable or misconfigured

**Solution:**
```bash
# List available tools
victor tools list

# Check tool configuration
victor tools info <tool_name>
```

---

## Getting Additional Help

### Resources

1. **Documentation:**
   - [Architecture Overview](../../architecture/README.md)
   - [Configuration Reference](../reference/api/CONFIGURATION_REFERENCE.md)
   - [Provider Guide](../reference/providers/)

2. **Community:**
   - GitHub Issues: [github.com/vjsingh1984/codingagent/issues](https://github.com/vjsingh1984/codingagent/issues)
   - Discord: [discord.gg/victor](https://discord.gg/victor)
   - Discussions: [github.com/vjsingh1984/codingagent/discussions](https://github.com/vjsingh1984/codingagent/discussions)

3. **Support:**
   - Email: support@victor.ai
   - Slack: #victor-support
   - Twitter: @victor_ai

---

## Issue Template

When reporting issues, please include:

### Environment

**Victor Version:**
```bash
victor --version
```

**Python Version:**
```bash
python --version
```

**Operating System:**
```bash
uname -a
```

### Configuration

**Settings File:**
```bash
cat ~/.victor/settings.yaml
```

**Environment Variables:**
```bash
env | grep VICTOR
```

### Error Message

**Full Error:**
```
[Paste full error message here]
```

**Steps to Reproduce:**
1. Step 1: ...
2. Step 2: ...
3. Step 3: ...

### Expected Behavior

**What should happen:**
```
[Describe expected behavior]
```

### Actual Behavior

**What actually happened:**
```
[Describe actual behavior]
```

---

## See Also

- [Documentation Home](../../README.md)


**Reading Time:** 1 min
**Last Updated:** February 01, 2026
