# Configuration Reference

Complete reference for Victor AI configuration options.

---

## Quick Summary

Comprehensive configuration reference covering:

- **Configuration Priority** - How different config sources interact
- **profiles.yaml** - Provider and model profiles
- **config.yaml** - Global settings
- **mcp.yaml** - Model Context Protocol configuration
- **Environment Variables** - All VICTOR_* environment variables
- **Project Context** - .victor.md project configuration
- **Configuration Examples** - Common scenarios

---

## Guide Parts

### [Part 1: Core Configuration](part-1-core-configuration.md)
- Configuration Directory Structure
- Configuration Priority
- Quick Configuration
- profiles.yaml (Provider/Model Profiles)
- config.yaml (Global Settings)
- mcp.yaml (MCP Configuration)

### [Part 2: Project & Environment](part-2-project-environment.md)
- Environment Variables
- Project Context Files
  - Project Overview, Tech Stack, Code Style
  - Testing, Git Workflow, Victor Instructions
  - Project Structure, Development Commands
  - Conventions, Preferences
- Configuration Examples

---

## Quick Start

**Basic Configuration:**
```yaml
# ~/.victor/profiles.yaml
profiles:
  default:
    provider: ollama
    model: qwen2.5-coder:7b
```

**Full Documentation:**
- Part 1: Core system configuration
- Part 2: Project and environment setup

---

**Last Updated:** February 01, 2026
**Reading Time:** 25 min (all parts)
