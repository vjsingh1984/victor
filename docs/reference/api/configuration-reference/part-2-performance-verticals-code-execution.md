# Victor AI 0.5.0 Configuration Reference - Part 2

**Part 2 of 2:** Performance Tuning, Vertical Configuration, Code Execution Settings, Complete Reference

---

## Navigation

- [Part 1: Core Configuration](part-1-settings-security.md)
- **[Part 2: Performance, Verticals, Code Execution](#)** (Current)
- [**Complete Reference**](../CONFIGURATION_REFERENCE.md)

---

## Performance Tuning

### Caching

```yaml
# Enable/disable caching
cache_enabled: true
cache_ttl: 3600  # Cache TTL in seconds
```

### Optimization

```yaml
# Performance optimizations
lazy_loading: true
parallel_execution: true
batch_size: 10
```

### Memory Management

```yaml
# Memory limits
max_memory_mb: 4096
context_window_tokens: 200000
```

---

## Vertical Configuration

### Enabling Verticals

```yaml
# Enable specific verticals
enabled_verticals:
  - coding
  - devops
  - rag
```

### Vertical-Specific Settings

```yaml
# Vertical configurations
verticals:
  coding:
    tools:
      - read
      - write
      - grep
    mode: build

  devops:
    tools:
      - docker
      - kubectl
    mode: explore
```

---

## Code Execution Settings

### Docker Execution

```yaml
# Docker code execution
code_execution:
  enabled: true
  backend: docker
  docker_image: "python:3.11-slim"
  timeout: 30
  memory_limit: "512m"
```

### Local Execution

```yaml
# Local code execution
code_execution:
  enabled: true
  backend: local
  timeout: 10
```

---

## Complete Settings Reference

### All Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `default_provider` | string | `anthropic` | Default LLM provider |
| `model` | string | `claude-sonnet-4-5` | Default model |
| `api_key` | string | - | API key for provider |
| `tool_call_budget` | int | `100` | Max tool calls per request |
| `temperature` | float | `0.7` | LLM temperature |
| `max_tokens` | int | `4096` | Max tokens per response |
| `cache_enabled` | bool | `true` | Enable response caching |
| `lazy_loading` | bool | `true` | Enable lazy loading |
| `airgapped_mode` | bool | `false` | Air-gapped mode |
| `log_level` | string | `INFO` | Logging level |

---

## Related Documentation

- [Getting Started](../../../getting-started/README.md)
- [Provider Configuration](../PROVIDER_REFERENCE.md)
- [Vertical Guide](../../guides/tutorials/creating-verticals/)

---

**Last Updated:** February 01, 2026
