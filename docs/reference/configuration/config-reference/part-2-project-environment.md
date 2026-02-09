# Configuration Reference - Part 2

**Part 2 of 2:** Environment Variables, Project Context Files, and Configuration Examples

---

## Navigation

- [Part 1: Core Configuration](part-1-core-configuration.md)
- **[Part 2: Project & Environment](#)** (Current)
- [**Complete Guide](../index.md)**

---

## Environment Variables

Environment variables for configuration and secrets.

### Provider API Keys

```bash
# Cloud providers
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-proj-...
export GOOGLE_API_KEY=...
export XAI_API_KEY=...
export DEEPSEEK_API_KEY=...
export MISTRAL_API_KEY=...
export TOGETHER_API_KEY=...
export FIREWORKS_API_KEY=...
export OPENROUTER_API_KEY=...
export GROQ_API_KEY=...
export MOONSHOT_API_KEY=...
export CEREBRAS_API_KEY=...

# Enterprise
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_ENDPOINT=https://...
export AWS_PROFILE=default
export GCP_PROJECT=...

# Platforms
export HF_API_KEY=...
export REPLICATE_API_TOKEN=...
```text

### Local Providers

```bash
# Optional: Override default hosts
export OLLAMA_HOST=127.0.0.1:11434
export VICTOR_LM_STUDIO_HOST=127.0.0.1:1234
export VICTOR_VLLM_HOST=127.0.0.1:8000
export VICTOR_LLAMACPP_HOST=127.0.0.1:8080
```

### Victor Configuration

```bash
# General
export VICTOR_CONFIG_DIR=~/.victor
export VICTOR_PROFILE=development
export VICTOR_LOG_LEVEL=INFO
export VICTOR_LOG_FILE=/var/log/victor.log

# Cache
export VICTOR_CACHE_ENABLED=true
export VICTOR_CACHE_TTL=3600

# Providers
export VICTOR_TIMEOUT=30
export VICTOR_RETRY_ATTEMPTS=3

# Tools
export VICTOR_TOOL_TIMEOUT=30
export VICTOR_MAX_CONCURRENT_TOOLS=5
```text

### Proxy Settings

```bash
# HTTP proxy
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
export NO_PROXY=localhost,127.0.0.1

# SOCKS proxy
export ALL_PROXY=socks5://proxy.example.com:1080
```

[Full Environment Variables →](keys.md)

---

## Project Context Files

Project-specific configuration for AI behavior.

### .victor.md

**Location**: `<project>/.victor.md`
**Purpose**: Project context and instructions for Victor

**Example**:
```markdown
# Project Context

## Project Overview
This is a Django web application with PostgreSQL backend.

## Tech Stack
- Backend: Django 5.0, Python 3.11
- Frontend: React 18, TypeScript
- Database: PostgreSQL 15
- Cache: Redis 7

## Code Style
- Follow PEP 8 for Python
- Use type hints (strict mode)
- Max line length: 100 characters
- Write docstrings (Google style)

## Testing
- Use pytest for unit tests
- Minimum coverage: 80%
- Run tests before committing

## Git Workflow
- Feature branches: `feature/xxx`
- Main branch: `main`
- PR required for all changes

## Victor Instructions
When refactoring code:
1. Add type hints
2. Update docstrings
3. Add unit tests
4. Run pytest and fix failures
```text

**Usage**: Victor automatically reads `.victor.md` when working in a project directory.

[Full .victor.md Guide →](../../user-guide/index.md#5-project-context)

### CLAUDE.md

**Location**: `<project>/CLAUDE.md`
**Purpose**: Instructions for Claude AI (used by Claude Code and Victor)

**Example**:
```markdown
# Claude Instructions

## Project Structure
- `src/`: Source code
- `tests/`: Test files
- `docs/`: Documentation

## Development Commands
- `make test`: Run tests
- `make lint`: Run linter
- `make format`: Format code

## Conventions
- Use dependency injection
- Write async/await for I/O
- Add error handling
- Log important events

## Preferences
- Prefer composition over inheritance
- Use type hints
- Keep functions small (<50 lines)
- Write self-documenting code
```

See `CLAUDE.md` in the repo root for the full reference.

---

## Configuration Examples

### Example 1: Local Development

**~/.victor/profiles.yaml**:
```yaml
profiles:
  default:
    provider: ollama
    model: qwen2.5-coder:7b
```text

**~/.victor/config.yaml**:
```yaml
logging:
  level: DEBUG

cache:
  enabled: true
  backend: memory

tools:
  lazy_load: true
```

**.victor.md**:
```markdown
# Local Development
Use Ollama for privacy and speed.
```text

### Example 2: Team Development

**~/.victor/profiles.yaml**:
```yaml
profiles:
  development:
    provider: anthropic
    model: claude-sonnet-4-20250514
    temperature: 0.7

  production:
    provider: azure
    deployment: gpt-4o
    temperature: 0.3
```

**~/.victor/config.yaml**:
```yaml
logging:
  level: INFO
  file: /var/log/victor/team.log

cache:
  enabled: true
  backend: redis
  redis:
    host: redis.example.com
    port: 6379

tools:
  parallel_execution: true
  max_concurrent: 10
```text

### Example 3: Enterprise

**~/.victor/profiles.yaml**:
```yaml
profiles:
  default:
    provider: bedrock
    model_id: anthropic.claude-3-sonnet-20240229-v1:0
    region_name: us-east-1

  fallback:
    provider: vertex
    model: gemini-2.0-flash-exp
    project: enterprise-project
```

**~/.victor/config.yaml**:
```yaml
logging:
  level: INFO
  format: json
  file: /var/log/victor/enterprise.log

providers:
  timeout: 60
  retry_attempts: 5
  circuit_breaker:
    enabled: true
    failure_threshold: 3

cache:
  enabled: true
  backend: redis
  redis:
    host: redis.internal.example.com
    port: 6379
    password: ${REDIS_PASSWORD}
```text

---

## Validation and Testing

### Validate Configuration

```bash
# Validate all configuration files
victor config validate

# Check current configuration
victor config show

# Test profile
victor --profile development chat "Test"

# Test provider
victor chat --provider anthropic "Test"
```

### Debug Configuration

```bash
# Enable debug logging
export VICTOR_LOG_LEVEL=DEBUG

# View configuration
victor config show

# Check logs
victor logs --tail 100

# Trace configuration loading
victor --debug chat "Test"
```text

---

## Troubleshooting

### Common Issues

**Configuration not loading**:
```bash
# Check file location
ls -la ~/.victor/profiles.yaml

# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('~/.victor/profiles.yaml'))"

# Check permissions
chmod 600 ~/.victor/profiles.yaml
```

**Profile not found**:
```bash
# List available profiles
victor config profiles

# Check profile name in profiles.yaml
grep -A 5 "profiles:" ~/.victor/profiles.yaml
```text

**API key not found**:
```bash
# Check environment variable
echo $ANTHROPIC_API_KEY

# Set temporarily
export ANTHROPIC_API_KEY=sk-ant-...

# Add to shell profile
echo 'export ANTHROPIC_API_KEY=sk-ant-...' >> ~/.bashrc
source ~/.bashrc
```

[Full Troubleshooting Guide →](../../user-guide/troubleshooting.md)

---

## Best Practices

### 1. Use Profiles

**DO**:
```yaml
# Reusable profiles
profiles:
  development:
    provider: anthropic
    model: claude-sonnet-4-20250514

  production:
    provider: azure
    deployment: gpt-4o
```text

**DON'T**:
```bash
# Hard-coded values
victor chat --provider anthropic --model claude-sonnet-4-20250514 --api-key sk-...
```

### 2. Secure API Keys

**DO**:
```bash
# Environment variables
export ANTHROPIC_API_KEY=sk-ant-...
```text

**DON'T**:
```yaml
# API keys in config files
api_key: sk-ant-...  # NEVER commit this!
```

### 3. Version Control

**DO**:
```bash
# Add to .gitignore
echo ".victor/" >> .gitignore
echo "*.yaml" >> .gitignore
echo ".env" >> .gitignore
```text

**DON'T**:
```bash
# Commit configuration files
git add ~/.victor/profiles.yaml  # May contain secrets!
```

### 4. Document Configuration

**DO**:
```yaml
# ~/.victor/profiles.yaml
profiles:
  # Development profile for local testing
  development:
    provider: anthropic
    model: claude-sonnet-4-20250514
    temperature: 0.7  # Higher temp for creativity
```text

**DON'T**:
```yaml
# No comments
profiles:
  development:
    provider: anthropic
    model: claude-sonnet-4-20250514
    temperature: 0.7
```

### 5. Test Configuration

**DO**:
```bash
# Test before committing
victor config validate
victor --profile development chat "Test"
```text

**DON'T**:
```bash
# Assume it works
victor chat "Important task"  # May fail!
```

---

## Additional Resources

- **profiles.yaml**: [Full Reference →](#profilesyaml)
- **config.yaml**: [Full Reference →](#configyaml)
- **mcp.yaml**: [Full Reference →](#mcpyaml)
- **API Keys**: [Key Management →](keys.md)
- **Troubleshooting**: [Troubleshooting Guide →](../../user-guide/troubleshooting.md)

---

**Next**: [profiles.yaml →](#profilesyaml) | [config.yaml →](#configyaml) | [API Keys →](keys.md) | [Troubleshooting
  →](../../user-guide/troubleshooting.md)

---

## See Also

- [Documentation Home](../../README.md)


**Last Updated:** February 01, 2026
**Reading Time:** 4 minutes
