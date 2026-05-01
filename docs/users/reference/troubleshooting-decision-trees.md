# Troubleshooting Decision Trees

**Last Updated**: 2026-04-30 | **Format**: Visual Decision Trees

## Overview

This guide provides visual decision trees for troubleshooting common Victor issues. Follow the trees to diagnose and resolve problems quickly.

## Issue Categories

```mermaid
mindmap
  root((Victor Issues))
    Installation
      Python version
      Dependencies
      Build errors
    Configuration
      API keys
      Providers
      Models
    Runtime
      Connection errors
      Timeouts
      Rate limits
    Tools
      Tool not found
      Permission denied
      Tool failures
    Performance
      Slow responses
      High memory
      CPU usage
```

## Decision Tree 1: Installation Issues

```mermaid
flowchart TD
    A[Installation Problem] --> B{Victor installed?}
    B -->|No| C[Check Python version]
    B -->|Yes| D[Check dependencies]

    C --> E{Python >= 3.10?}
    E -->|No| F[Upgrade Python]
    E -->|Yes| G[Install Victor]

    D --> H{Dependencies OK?}
    H -->|No| I[Install missing deps]
    H -->|Yes| Z[Problem solved]

    F --> G
    G --> Z
    I --> Z

    style A fill:#ef4444,color:#fff
    style Z fill:#10b981,color:#fff
```

### Quick Reference

| Symptom | Cause | Solution |
|---------|-------|----------|
| **Module not found** | Not installed | `pip install victor-ai` |
| **Python version error** | Python < 3.10 | Upgrade to 3.10+ |
| **Build failed** | Missing build tools | `pip install --upgrade pip` |

## Decision Tree 2: Provider Issues

```mermaid
flowchart TD
    A[Provider Problem] --> B{Provider available?}

    B -->|No| C[victor provider list]
    B -->|Yes| D{API key set?}

    C --> E{Provider in list?}
    E -->|No| F[Install provider]
    E -->|Yes| G[Check provider status]

    D -->|No| H[Set API key]
    D -->|Yes| I{API key valid?}

    H --> J{Key format OK?}
    J -->|No| K[Fix key format]
    J -->|Yes| L[Generate new key]

    I -->|No| M[Regenerate key]
    I -->|Yes| N{Connection working?}

    F --> N
    G --> N
    K --> L
    L --> N
    M --> N

    N -->|No| O[Check network]
    N -->|Yes| P[Test with simple command]
    O --> Q{Network OK?}
    Q -->|No| R[Fix network/firewall]
    Q -->|Yes| S[Contact provider]

    P --> T{Works?}
    T -->|Yes| U[Problem solved]
    T -->|No| V[Check model name]

    R --> P
    S --> P
    V --> U

    style A fill:#ef4444,color:#fff
    style U fill:#10b981,color:#fff
```

### Quick Reference

| Symptom | Cause | Solution |
|---------|-------|----------|
| **Provider not found** | Not installed | `victor provider list` |
| **Invalid API key** | Wrong key format | Check key format |
| **Connection timeout** | Network/firewall | Check internet |
| **Model not found** | Wrong model name | `victor model list --provider <name>` |

## Decision Tree 3: Tool Issues

```mermaid
flowchart TD
    A[Tool Problem] --> B{Tool enabled?}

    B -->|No| C[Check tool config]
    B -->|Yes| D{Tool available?}

    C --> E{Tool in preset?}
    E -->|No| F[Add tool to config]
    E -->|Yes| G[Enable tool_execution]

    D -->|No| H[victor tool list]
    D -->|Yes| I{Permission OK?}

    H --> J{Tool exists?}
    J -->|No| K[Install tool module]
    J -->|Yes| L[Check tool category]

    I -->|No| M[Check file permissions]
    I -->|Yes| N{Tool working?}

    F --> N
    G --> N
    K --> N
    L --> N
    M --> N

    N -->|No| O[victor tool test]
    N -->|Yes| P[Problem solved]

    O --> Q{Test passes?}
    Q -->|No| R[Check tool dependencies]
    Q -->|Yes| S[Report issue]

    R --> T{Dependencies OK?}
    T -->|No| U[Install dependencies]
    T -->|Yes| V[Check tool config]

    U --> V
    V --> S

    style A fill:#ef4444,color:#fff
    style P fill:#10b981,color:#fff
```

### Quick Reference

| Symptom | Cause | Solution |
|---------|-------|----------|
| **Tool not found** | Not in preset | Add tool to config |
| **Permission denied** | File permissions | Check file access |
| **Tool timeout** | Long execution | Increase timeout |
| **Tool failed** | Missing dependencies | Install deps |

## Decision Tree 4: Performance Issues

```mermaid
flowchart TD
    A[Performance Problem] --> B{Slow responses?}

    B -->|Yes| C{Provider local?}
    B -->|No| D{High memory?}

    C -->|Yes| E[Check model size]
    C -->|No| F[Check network]

    D -->|Yes| G[Check context size]
    D -->|No| H{High CPU?}

    E --> I{Model too large?}
    I -->|Yes| J[Use smaller model]
    I -->|No| K[Check hardware]

    F --> L{Network slow?}
    L -->|Yes| M[Switch provider]
    L -->|No| N[Check API rate limits]

    G --> O{Context too large?}
    O -->|Yes| P[Reduce context]
    O -->|No| Q[Check memory leaks]

    H --> R{CPU high?}
    R -->|Yes| S[Reduce tool usage]
    R -->|No| T[Profile code]

    J --> U[Optimized]
    K --> U
    M --> U
    N --> U
    P --> U
    Q --> U
    S --> U
    T --> V[Investigate further]

    style A fill:#ef4444,color:#fff
    style U fill:#10b981,color:#fff
```

### Quick Reference

| Symptom | Cause | Solution |
|---------|-------|----------|
| **Slow local model** | Model too large | Use smaller model |
| **Slow cloud model** | Network/rate limits | Switch provider |
| **High memory** | Large context | Reduce context size |
| **High CPU** | Tool execution | Reduce tool usage |

## Decision Tree 5: Workflow Issues

```mermaid
flowchart TD
    A[Workflow Problem] --> B{Workflow valid?}

    B -->|No| C[victor workflow validate]
    B -->|Yes| D{Nodes executing?}

    C --> E{Validation errors?}
    E -->|Yes| F[Fix syntax errors]
    E -->|No| G[Check workflow structure]

    D -->|No| H{First node running?}
    D -->|Yes| I{State passing?}

    H -->|No| J[Check start node]
    H -->|Yes| K[Check node configuration]

    I -->|No| L[Check state schema]
    I -->|Yes| M{Edges correct?}

    J --> N{Start node OK?}
    N -->|No| O[Fix start node]
    N -->|Yes| K

    K --> P{Node config OK?}
    P -->|No| Q[Fix node config]
    P -->|Yes| L

    L --> R{State schema OK?}
    R -->|No| S[Fix state schema]
    R -->|Yes| M

    M -->|No| T[Fix edges]
    M -->|Yes| U[Problem solved]

    F --> U
    G --> U
    O --> U
    Q --> U
    S --> U
    T --> U

    style A fill:#ef4444,color:#fff
    style U fill:#10b981,color:#fff
```

### Quick Reference

| Symptom | Cause | Solution |
|---------|-------|----------|
| **Validation error** | Syntax error | Fix YAML syntax |
| **Node not running** | Missing config | Add node config |
| **State not passing** | Schema mismatch | Fix state schema |
| **Wrong path** | Edge condition | Fix edge logic |

## Common Error Messages

### Provider Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `Provider not found: xyz` | Provider doesn't exist | Use valid provider |
| `Invalid API key` | Wrong or missing key | Set correct API key |
| `Connection timeout` | Network/firewall | Check connection |
| `Rate limit exceeded` | Too many requests | Wait or switch provider |

### Tool Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `Tool not found: xyz` | Tool doesn't exist | Use valid tool |
| `Permission denied` | File access denied | Check permissions |
| `Tool timeout` | Execution too long | Increase timeout |
| `Tool execution failed` | Tool error | Check tool output |

### Workflow Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `Invalid YAML` | Syntax error | Fix YAML syntax |
| `Node not found: xyz` | Missing node | Define node |
| `State validation failed` | Schema mismatch | Fix state schema |
| `Edge condition failed` | Logic error | Fix condition |

## Diagnostic Commands

```bash
# Full diagnostics
victor doctor --verbose

# Check provider
victor provider check anthropic

# Check tool
victor tool test read_file

# Validate workflow
victor workflow validate my-workflow.yaml

# List providers
victor provider list

# List tools
victor tool list

# List models
victor model list --provider ollama
```

## Quick Fixes

### Installation

```bash
# Reinstall Victor
pip uninstall victor-ai
pip install victor-ai

# Upgrade dependencies
pip install --upgrade -r requirements.txt
```

### Configuration

```bash
# Reset config
victor config reset

# Reinitialize
victor init

# Check config
victor config list
```

### Providers

```bash
# Test provider
victor provider check anthropic

# Switch provider
victor chat --provider ollama

# Test with simple command
victor chat --provider ollama "Say hello"
```

### Tools

```bash
# List tools
victor tool list

# Enable tools
victor config set tool_execution enabled

# Test tool
victor tool test read_file
```

## When to Ask for Help

```mermaid
flowchart TD
    A[Issue persists] --> B{Tried all fixes?}
    B -->|No| C[Continue troubleshooting]
    B -->|Yes| D{Blocking work?}

    D -->|No| E[Work around issue]
    D -->|Yes| F[Check documentation]

    F --> G{Solution found?}
    G -->|Yes| H[Apply fix]
    G -->|No| I[Search GitHub issues]

    I --> J{Issue reported?}
    J -->|Yes| K[Add comment]
    J -->|No| L[Create new issue]

    C --> A
    E --> A
    H --> M[Resolved]
    K --> N[Wait for response]
    L --> N

    style A fill:#ef4444,color:#fff
    style M fill:#10b981,color:#fff
```

### Getting Help

1. **Documentation** - [docs/](../)
2. **FAQ** - [faq.md](faq.md)
3. **GitHub Issues** - [Report issue](https://github.com/vjsingh1984/victor/issues)
4. **Discussions** - [Ask question](https://github.com/vjsingh1984/victor/discussions)

## Best Practices

✅ **DO**:
- Run `victor doctor` first
- Check error messages carefully
- Try simple commands first
- Test with local providers
- Use verbose mode for debugging

❌ **DON'T**:
- Ignore error messages
- Skip diagnostic steps
- Use production keys for testing
- Exceed rate limits
- Forget to check logs

---

**See Also**: [FAQ](faq.md) | [CLI Cheatsheet](cli-cheatsheet.md) | [Providers Quick Reference](providers-quickref.md)

**Decision Trees**: 5 | **Common Errors**: 15+ | **Diagnostic Commands**: 7
