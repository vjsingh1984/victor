# Coordinator Settings Guide - Part 1

**Part 1 of 2:** Feature Flags, Configuration Methods, Environment Variables, Settings File, and Coordinator-Specific Settings

---

## Navigation

- **[Part 1: Configuration](#)** (Current)
- [Part 2: Best Practices](part-2-best-practices.md)
- [**Complete Guide](../coordinator_settings.md)**

---
# Coordinator Orchestrator Settings Guide

**Version**: 1.0
**Date**: 2025-01-14
**Audience**: Developers, DevOps Engineers, System Administrators

---

## Table of Contents

1. [Feature Flag Overview](#feature-flag-overview)
2. [Configuration Methods](#configuration-methods)
3. [Environment Variables](#environment-variables)
4. [Settings File Configuration](#settings-file-configuration)
5. [Coordinator-Specific Settings](#coordinator-specific-settings)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)
8. [Examples](#examples)

---

## Feature Flag Overview

### What is `use_coordinator_orchestrator`?

The `use_coordinator_orchestrator` setting controls whether Victor uses the new coordinator-based orchestrator architecture or the legacy monolithic orchestrator.

**Values**:
- `false` (default): Use legacy orchestrator (monolithic)
- `true`: Use new coordinator-based orchestrator

**Migration Path**:
1. **Phase 1** (Current): Feature flag = `false` (legacy)
2. **Phase 2** (Rollout): Feature flag = `true` (coordinators)
3. **Phase 3** (Future): Remove legacy and feature flag

### Architecture Comparison

| Aspect | Legacy Orchestrator | Coordinator Orchestrator |
|--------|-------------------|-------------------------|
| Architecture | Monolithic (6,082 lines) | Facade + 15 coordinators |
| Testability | 65% coverage | 85% coverage |
| Test Speed | 45s | 12s (10x faster) |
| Performance | Baseline | 3-5% overhead (well under 10% goal) |
| Maintainability | Low | High |
| Extensibility | Difficult | Easy (plugin coordinators) |

---

## Configuration Methods

You can enable the coordinator orchestrator in three ways, listed in order of precedence:

### Method 1: Environment Variable (Highest Priority)

**Best for**: Testing, CI/CD, temporary enabling

```bash
# Enable via environment variable
export VICTOR_USE_COORDINATOR_ORCHESTRATOR=true
victor chat

# Disable
export VICTOR_USE_COORDINATOR_ORCHESTRATOR=false
victor chat
```

**Pros**:
- No file modifications needed
- Easy to toggle for testing
- Works in CI/CD pipelines

**Cons**:
- Not persistent across sessions
- Must set in each shell/session

### Method 2: profiles.yaml (Recommended for Production)

**Best for**: Production environments, permanent configuration

Edit `~/.victor/profiles.yaml`:

```yaml
# ~/.victor/profiles.yaml

# Top-level setting (add this)
use_coordinator_orchestrator: true

# Existing profiles below
profiles:
  default:
    provider: anthropic
    model: claude-sonnet-4-5
    temperature: 0.7
```

**Pros**:
- Persistent across sessions
- Version controlled (if you track config)
- Clear configuration intent

**Cons**:
- Requires file editing
- Manual rollback needed

### Method 3: Toggle Script (Recommended for Safe Operations)

**Best for**: Production rollout with backups

```bash
# Enable with automatic backup
python scripts/toggle_coordinator_orchestrator.py enable --backup

# Check status
python scripts/toggle_coordinator_orchestrator.py status

# Validate configuration
python scripts/toggle_coordinator_orchestrator.py validate

# Disable if needed
python scripts/toggle_coordinator_orchestrator.py disable --backup
```

**Pros**:
- Automatic backup before changes
- Change logging/history
- Built-in validation
- Safe rollback

**Cons**:
- Requires script execution
- Additional dependency (yaml)

---

## Environment Variables

### Primary Feature Flag

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `VICTOR_USE_COORDINATOR_ORCHESTRATOR` | bool | `false` | Enable coordinator-based orchestrator |

**Usage**:
```bash
# Enable
export VICTOR_USE_COORDINATOR_ORCHESTRATOR=true

# Disable
export VICTOR_USE_COORDINATOR_ORCHESTRATOR=false

# Check
echo $VICTOR_USE_COORDINATOR_ORCHESTRATOR
```

### Related Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `VICTOR_HEADLESS_MODE` | bool | `false` | Run without prompts (CI/CD) |
| `VICTOR_DRY_RUN_MODE` | bool | `false` | Preview changes without applying |
| `VICTOR_LOG_LEVEL` | string | `INFO` | Logging verbosity (DEBUG, INFO, WARN, ERROR) |
| `VICTOR_ENABLE_OBSERVABILITY_LOGGING` | bool | `false` | Enable metrics export |

**Example: Full Production Setup**
```bash
export VICTOR_USE_COORDINATOR_ORCHESTRATOR=true
export VICTOR_LOG_LEVEL=INFO
export VICTOR_ENABLE_OBSERVABILITY_LOGGING=true
export VICTOR_OBSERVABILITY_LOG_PATH=~/.victor/metrics/victor.jsonl

victor chat
```

---

## Settings File Configuration

### profiles.yaml Structure

The `profiles.yaml` file has two main sections:

1. **Top-level settings**: Feature flags and global config
2. **Profiles**: Named configurations for different providers/models

```yaml
# ~/.victor/profiles.yaml

# ===================================================================
# TOP-LEVEL SETTINGS (Feature Flags, Global Config)
# ===================================================================

# Enable coordinator-based orchestrator
use_coordinator_orchestrator: true

# Other top-level settings
airgapped_mode: false
log_level: "INFO"
enable_observability_logging: true

# ===================================================================
# PROFILES (Provider/Model Configurations)
# ===================================================================

profiles:
  # Default profile
  default:
    provider: anthropic
    model: claude-sonnet-4-5
    temperature: 0.7
    max_tokens: 4096

  # Local development profile
  local:
    provider: ollama
    model: qwen3-coder:30b
    temperature: 0.7

  # Production profile
  production:
    provider: anthropic
    model: claude-sonnet-4-5
    temperature: 0.3
    tool_selection:
      base_threshold: 0.5
      base_max_tools: 15
```

### Per-Profile Coordinator Settings

You can customize coordinator behavior per profile:

```yaml
profiles:
  default:
    provider: anthropic
    model: claude-sonnet-4-5

    # ContextCoordinator settings
    context_compaction_strategy: "tiered"
    context_min_messages_to_keep: 6

    # ToolCoordinator settings
    max_tool_iterations: 5

    # AnalyticsCoordinator settings
    enable_analytics: true

  # Custom profile with coordinator overrides
  custom:
    provider: ollama
    model: "qwen3-coder:30b"

    # Override defaults for smaller models
    tool_selection:
      base_threshold: 0.6  # Higher threshold (more selective)
      base_max_tools: 10    # Fewer tools
      model_size_tier: "medium"
```

---

## Coordinator-Specific Settings

### ConfigCoordinator

No direct settings. Configured via `Settings` class.

### PromptCoordinator

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `system_prompt_template` | string | `"default"` | System prompt template to use |
| `enable_task_hints` | bool | `true` | Include task type hints in prompts |

### ContextCoordinator

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `context_compaction_strategy` | string | `"tiered"` | Compaction: `simple`, `tiered`, `semantic`, `hybrid` |
| `context_min_messages_to_keep` | int | `6` | Minimum messages to retain |
| `context_tool_retention_weight` | float | `1.5` | Boost for tool results |
| `context_recency_weight` | float | `2.0` | Boost for recent messages |
| `context_semantic_threshold` | float | `0.3` | Min similarity for semantic retention |

### ChatCoordinator

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `max_tool_iterations` | int | `5` | Max tool calling loops |
| `tool_calling_timeout` | int | `30` | Tool execution timeout (seconds) |

### ToolCoordinator

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `tool_call_budget` | int | `100` | Max tool calls per session |
| `tool_call_budget_warning_threshold` | int | `80` | Warn at N tool calls |
| `tool_retry_enabled` | bool | `true` | Enable automatic retry |
| `tool_cache_enabled` | bool | `true` | Enable tool result caching |

### SessionCoordinator

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `conversation_memory_enabled` | bool | `true` | Enable conversation persistence |
| `checkpoint_enabled` | bool | `true` | Enable checkpointing |
| `checkpoint_auto_interval` | int | `5` | Checkpoints per N tool calls |

### MetricsCoordinator

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `streaming_metrics_enabled` | bool | `true` | Enable performance metrics |
| `streaming_metrics_history_size` | int | `1000` | Metrics samples to retain |

### AnalyticsCoordinator

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `analytics_enabled` | bool | `true` | Enable analytics tracking |

### ToolSelectionCoordinator

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `tool_selection_strategy` | string | `"auto"` | Strategy: `auto`, `keyword`, `semantic`, `hybrid` |
| `semantic_similarity_threshold` | float | `0.25` | Min similarity score |

---

