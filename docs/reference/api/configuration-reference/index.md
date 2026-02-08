# Victor AI 0.5.0 Configuration Reference

Complete reference for all configuration options in Victor AI.

---

## Quick Summary

Victor AI uses a hierarchical configuration system with multiple sources:

**Priority Order (highest to lowest):**
1. **Environment variables** - Runtime overrides
2. **.env file** - Project-specific settings
3. **~/.victor/profiles.yaml** - Global profile configuration
4. **Default values** - Built-in defaults

---

## Reference Parts

### [Part 1: Core Configuration](part-1-settings-security.md)
- Overview
- Settings
- Environment Variables
- Profiles
- Path Configuration
- Provider Configuration
- Tool Selection
- Budget Management
- Logging and Observability
- Security Settings

### [Part 2: Performance, Verticals, Code Execution](part-2-performance-verticals-code-execution.md)
- Performance Tuning
- Vertical Configuration
- Code Execution Settings
- Complete Settings Reference

---

## Quick Start

```bash
# Set provider (environment variable)
export VICTOR_DEFAULT_PROVIDER="anthropic"
export ANTHROPIC_API_KEY="sk-ant-..."

# Or use .env file
cat > .env << EOF
VICTOR_DEFAULT_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
VICTOR_TOOL_CALL_BUDGET=50
EOF

# Or use profiles.yaml
cat > ~/.victor/profiles.yaml << EOF
default_provider: anthropic
anthropic_api_key: sk-ant-...
tool_call_budget: 50
EOF
```

---

## Related Documentation

- [Getting Started](../../../getting-started/README.md)
- [Provider Configuration](../PROVIDER_REFERENCE.md)
- [Vertical Guide](../../guides/tutorials/creating-verticals/)

---

**Last Updated:** February 01, 2026
**Reading Time:** 12 min (all parts)
