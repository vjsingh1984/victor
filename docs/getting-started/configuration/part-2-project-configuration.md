# Configuration Guide - Part 2

**Part 2 of 3:** Project Context Files, Modes Configuration, MCP Configuration, and Example Configurations

---

## Navigation

- [Part 1: Configuration Basics](part-1-configuration-basics.md)
- **[Part 2: Project Configuration](#)** (Current)
- [Part 3: Validation & Best Practices](part-3-validation-best-practices.md)
- [**Complete Guide](../configuration.md)**

---

## Table of Contents

1. [Configuration Overview](#configuration-overview) *(in Part 1)*
2. [Quick Setup](#quick-setup) *(in Part 1)*
3. [Configuration Locations](#configuration-locations) *(in Part 1)*
4. [Environment Variables](#environment-variables) *(in Part 1)*
5. [Profiles Configuration](#profiles-configuration) *(in Part 1)*
6. [Global Settings](#global-settings) *(in Part 1)*
7. [Project Context Files](#project-context-files)
8. [Modes Configuration](#modes-configuration)
9. [MCP Configuration](#mcp-configuration)
10. [Example Configurations](#example-configurations)
11. [Validation and Testing](#validation-and-testing) *(in Part 3)*
12. [Troubleshooting](#troubleshooting) *(in Part 3)*
13. [Best Practices](#best-practices) *(in Part 3)*

---

## Global Settings

### Basic Settings

**File**: `~/.victor/config.yaml`

```yaml
# Logging configuration
logging:
  level: INFO              # DEBUG, INFO, WARNING, ERROR, CRITICAL
  format: text             # text or json
  file: ~/.victor/logs/victor.log
  rotation: daily          # daily, weekly, or size
  retention: 30            # Days to keep logs

# Cache configuration
cache:
  enabled: true
  ttl: 3600                # Time-to-live in seconds
  max_size: 1000           # Maximum cached items
  backend: memory          # memory, disk, or redis

# Tool configuration
tools:
  lazy_load: true          # Load tools on-demand
  parallel_execution: true # Execute tools in parallel
  timeout: 30              # Tool execution timeout

# Conversation settings
conversation:
  max_history: 100         # Maximum messages in history
  save_history: true       # Save conversations to disk
  history_file: ~/.victor/conversations/

# Provider settings
providers:
  timeout: 30              # Request timeout
  retry_attempts: 3        # Retry on failure
  circuit_breaker:
    enabled: true          # Enable circuit breaker
    failure_threshold: 5   # Failures before opening
    recovery_timeout: 60   # Seconds before retry

# UI settings
ui:
  theme: dark              # dark or light
  syntax_highlighting: true
  line_numbers: true
```

---

## Project Context Files

Project context files teach Victor about your specific codebase.

### .victor.md

**Location**: `<project>/.victor.md`
**Purpose**: Project-specific context and instructions

```markdown
# Project Context

## Overview
This is a Django web application for e-commerce.

## Tech Stack
- Backend: Django 5.0, Python 3.11
- Frontend: React 18, TypeScript
- Database: PostgreSQL 15
- Cache: Redis 7

## Code Style
- Follow PEP 8 for Python
- Use type hints everywhere
- Max line length: 100 characters
- Google-style docstrings

## Testing
- Use pytest for all tests
- Minimum coverage: 80%
- Run `make test` before committing

## Important Directories
- `src/api/`: REST API endpoints
- `src/models/`: Database models
- `src/services/`: Business logic
- `tests/`: Test files

## Conventions
- All API endpoints use snake_case
- Database tables are singular (user, not users)
- Use environment variables for all secrets

## Victor Instructions
When modifying code:
1. Always add type hints
2. Update docstrings
3. Add or update tests
4. Run linter before finishing
```

### CLAUDE.md

**Location**: `<project>/CLAUDE.md`
**Purpose**: Instructions for AI assistants (Claude Code, Victor)

This file follows the same format as `.victor.md` and is automatically read by Victor.

### Loading Priority

Victor loads project context in this order (first found wins):
1. `.victor.md` in current directory
2. `CLAUDE.md` in current directory
3. `.victor/init.md` in current directory

---

## Modes Configuration

Victor supports three execution modes:

| Mode | File Edits | Use Case |
|------|------------|----------|
| **BUILD** | Yes | Implementation, refactoring, real changes |
| **PLAN** | Sandbox only | Analysis, planning, code review |
| **EXPLORE** | No | Understanding code, learning codebase |

### Using Modes

```bash
# Via CLI flag
victor chat --mode plan "Analyze this code"
victor chat --mode explore "How does auth work?"
victor chat --mode build "Implement feature X"

# Via in-chat command
/mode plan
/mode explore
/mode build
```

### Mode Budgets

Each mode has different tool exploration budgets:
- **BUILD**: 1x (default)
- **PLAN**: 2.5x (more exploration, sandbox edits)
- **EXPLORE**: 3x (maximum exploration, read-only)

---

## MCP Configuration

Configure Victor as an MCP server or connect to MCP servers.

### Victor as MCP Server

**File**: `~/.victor/mcp.yaml`

```yaml
server:
  host: 127.0.0.1
  port: 8080
  transport: stdio    # stdio or sse

  # Authentication (optional)
  auth:
    enabled: false
    api_key: your-api-key

# Expose specific tools
tools:
  expose_all: false
  allowed_tools:
    - read_file
    - write_file
    - search_files
    - run_tests

# Expose prompts
prompts:
  directory: ~/.victor/prompts/
  expose_prompts:
    - code_review
    - refactor
```

### Connecting to MCP Servers

```yaml
# ~/.victor/mcp.yaml
servers:
  - name: filesystem
    command: npx
    args: ["-y", "@anthropic/mcp-server-filesystem"]

  - name: github
    command: npx
    args: ["-y", "@anthropic/mcp-server-github"]
    env:
      GITHUB_TOKEN: ${GITHUB_TOKEN}
```

---

