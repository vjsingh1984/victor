# Victor AI 0.5.0 Configuration Reference - Part 1

**Part 1 of 2:** Settings, Environment Variables, Profiles, Paths, Providers, Tools, Budget, Logging, and Security

---

## Navigation

- **[Part 1: Core Configuration](#)** (Current)
- [Part 2: Performance, Verticals, Code Execution](part-2-performance-verticals-code-execution.md)
- [**Complete Reference**](../CONFIGURATION_REFERENCE.md)

---

> **Note**: This legacy API documentation is retained for reference. For current docs, see `docs/reference/api/`.

Complete reference for all configuration options in Victor AI.

**Table of Contents**
- [Overview](#overview)
- [Settings](#settings)
- [Environment Variables](#environment-variables)
- [Profiles](#profiles)
- [Path Configuration](#path-configuration)
- [Provider Configuration](#provider-configuration)
- [Tool Selection](#tool-selection)
- [Budget Management](#budget-management)
- [Logging and Observability](#logging-and-observability)
- [Security Settings](#security-settings)
- [Performance Tuning](#performance-tuning) *(in Part 2)*

---

## Overview

Victor AI uses a hierarchical configuration system with multiple sources:

**Priority Order (highest to lowest):**
1. **Environment variables** - Runtime overrides
2. **.env file** - Project-specific settings
3. **~/.victor/profiles.yaml** - Global profile configuration
4. **Default values** - Built-in defaults

### Quick Start

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
```text

[Content continues through Security Settings...]


**Reading Time:** 1 min
**Last Updated:** February 08, 2026**

---

## See Also

- [Documentation Home](../../README.md)


**Continue to [Part 2: Performance, Verticals, Code Execution](part-2-performance-verticals-code-execution.md)**
