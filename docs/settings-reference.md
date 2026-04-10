# Victor Settings Reference

Comprehensive reference for all configurable settings. Settings are loaded from `~/.victor/profiles.yaml`, environment variables, or CLI flags.

## Quick Reference

| Setting | Default | Env Var | CLI Flag |
|---------|---------|---------|----------|
| `default_provider` | `ollama` | `VICTOR_DEFAULT_PROVIDER` | `--provider` |
| `default_model` | `qwen3-coder:30b` | `VICTOR_DEFAULT_MODEL` | `--model` |
| `default_temperature` | `0.7` | `VICTOR_DEFAULT_TEMPERATURE` | — |
| `default_max_tokens` | `4096` | `VICTOR_DEFAULT_MAX_TOKENS` | — |
| `tool_call_budget` | `2000` | — | `--tool-budget` |
| `enable_planning` | `false` | — | `--planning/--no-planning` |
| `skill_auto_select_enabled` | `true` | — | `--auto-skill/--no-auto-skill` |
| `airgapped_mode` | `false` | `AIRGAPPED_MODE` | `--airgapped` |

## Provider & Model

```yaml
# ~/.victor/profiles.yaml
profiles:
  default:
    provider: ollama
    model: qwen3-coder:30b
    temperature: 0.7
    max_tokens: 4096

  cloud:
    provider: anthropic
    model: claude-sonnet-4-20250514
    temperature: 0.3
    max_tokens: 8192

  fast:
    provider: xai
    model: grok-3-mini-fast
    temperature: 0.5
```

**Supported providers**: `ollama`, `anthropic`, `openai`, `deepseek`, `xai`, `google`, `lmstudio`, `vllm`, `moonshot`

**Usage**:
```bash
victor chat --profile cloud         # Use cloud profile
victor chat --provider anthropic    # Override provider
victor chat --model gpt-4.1         # Override model
```

### Local Server URLs

```yaml
# Environment variables
OLLAMA_BASE_URL=http://localhost:11434
LMSTUDIO_BASE_URLS=http://192.168.1.100:1234,http://localhost:1234
VLLM_BASE_URL=http://localhost:8000
```

## API Keys

Set via environment variables (never commit to config files):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export GOOGLE_API_KEY=AIza...
export DEEPSEEK_API_KEY=sk-...
export XAI_API_KEY=xai-...
```

Check configured keys:
```bash
victor keys check
```

## Tools & Budgets

| Setting | Default | Description |
|---------|---------|-------------|
| `tool_call_budget` | `2000` | Max tool calls per session |
| `fallback_max_tools` | `8` | Cap tool list when stage pruning removes all (1-20, CI-guarded) |
| `tool_retry_enabled` | `true` | Auto-retry failed tool executions |
| `tool_retry_max_attempts` | `3` | Max retry attempts per tool call |
| `tool_retry_base_delay` | `1.0` | Base delay (seconds) for exponential backoff |
| `tool_retry_max_delay` | `10.0` | Max delay between retries |

```bash
victor chat --tool-budget 50 "quick fix"   # Limit tool calls
```

## Skills & Auto-Selection

Skills are composable expertise units that inject focused prompts + tools based on the user's message. Auto-selection uses embedding similarity to match messages to skills.

| Setting | Default | Description |
|---------|---------|-------------|
| `skill_auto_select_enabled` | `true` | Enable embedding-based skill matching |
| `skill_auto_select_high_threshold` | `0.65` | Above: use skill directly (high confidence) |
| `skill_auto_select_low_threshold` | `0.45` | Below: no skill injected |
| `skill_auto_select_use_edge_fallback` | `true` | Use edge LLM for ambiguous matches (0.45-0.65) |
| `skill_auto_select_log_selections` | `true` | Log which skill was selected |

```bash
victor chat "fix the test" --auto-skill     # Force skill selection on
victor chat "hello" --no-auto-skill         # Disable for this session
victor skill list                           # See available skills
victor skill create my_skill ...            # Create custom skill (YAML)
```

User skills are stored at `~/.victor/skills/*.yaml`. See `victor skill --help` for management commands.

## Prompt Optimization (GEPA)

Runtime evolution of system prompt sections using execution trace analysis.

| Setting | Default | Description |
|---------|---------|-------------|
| `prompt_optimization.enabled` | `false` | Master switch for all prompt optimization |
| `prompt_optimization.default_strategies` | `["gepa"]` | Strategies applied in order |
| `prompt_optimization.gepa.enabled` | `true` | Enable GEPA v2 (Pareto + tiered service) |
| `prompt_optimization.gepa.default_tier` | `balanced` | Default tier: economic/balanced/performance |

### GEPA Model Tiers

| Tier | Provider/Model | Use Case |
|------|---------------|----------|
| economic | `ollama/gemma4` | Post-convergence maintenance |
| balanced | `openai/gpt-4.1-mini` | Default — active optimization |
| performance | `anthropic/claude-sonnet-4-20250514` | Initial convergence, regressions |

Auto-switches based on convergence metrics. Three evolvable sections:
- `ASI_TOOL_EFFECTIVENESS_GUIDANCE`
- `GROUNDING_RULES`
- `COMPLETION_GUIDANCE`

## Decision Service (Tiered)

Routes different decision types to different model tiers. Fallback always goes DOWN in cost.

| Setting | Default | Description |
|---------|---------|-------------|
| `DecisionServiceSettings.enabled` | `true` | Enable tiered routing |
| `DecisionServiceSettings.edge` | `ollama/qwen3.5:2b` | Fast local micro-decisions |
| `DecisionServiceSettings.balanced` | `deepseek/deepseek-chat` | Mid-tier cloud decisions |
| `DecisionServiceSettings.performance` | `anthropic/claude-sonnet` | Frontier (opt-in only) |

### Default Routing

| Decision Type | Default Tier |
|--------------|-------------|
| `tool_selection` | edge |
| `skill_selection` | edge |
| `intent_classification` | edge |
| `task_completion` | edge |
| `error_classification` | edge |
| `stage_detection` | edge |
| `task_type_classification` | balanced |
| `multi_skill_decomposition` | balanced |

Performance tier is never used by default — opt-in via custom `tier_routing` config.

## Planning

Structured task decomposition for complex multi-step requests.

| Setting | Default | Description |
|---------|---------|-------------|
| `enable_planning` | `false` | Auto-detect and use planning |
| `planning_min_complexity` | `moderate` | Minimum complexity to trigger: simple/moderate/complex |

```bash
victor chat --planning "Analyze the codebase and refactor auth module"
victor chat --no-planning "quick question"
```

## Observability

| Setting | Default | Description |
|---------|---------|-------------|
| `enable_observability_logging` | `false` | Write events to JSONL for dashboard |
| `observability_log_path` | `~/.victor/metrics/victor.jsonl` | Log file path |
| `log_level` | `INFO` | Logging level: DEBUG/INFO/WARN/ERROR |

```bash
victor chat --log-events "task"       # Enable JSONL logging
victor chat --log-level DEBUG         # Verbose logging
```

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `use_edge_model` | `true` | Enable edge model for micro-decisions |
| `use_semantic_tool_selection` | `true` | Embedding-based tool selection |
| `use_mcp_tools` | `false` | Enable MCP (Model Context Protocol) tools |
| `use_emojis` | `true` (not in CI) | Emoji indicators in output |
| `use_provider_pooling` | `false` | Provider connection pooling |

Feature flags are checked via `FeatureFlag.USE_EDGE_MODEL.is_enabled()`.

## Edge Model

| Setting | Default | Description |
|---------|---------|-------------|
| `EdgeModelConfig.enabled` | `true` | Enable edge model |
| `EdgeModelConfig.provider` | `ollama` | Edge model provider |
| `EdgeModelConfig.model` | `qwen3.5:2b` | Edge model name |
| `EdgeModelConfig.timeout_ms` | `4000` | Hard timeout per call |
| `EdgeModelConfig.max_tokens` | `50` | Max response tokens |
| `EdgeModelConfig.cache_ttl` | `120` | Cache TTL in seconds |

## MCP (Model Context Protocol)

| Setting | Default | Description |
|---------|---------|-------------|
| `use_mcp_tools` | `false` | Enable MCP support |
| `mcp_command` | `null` | MCP server command (e.g., `python mcp_server.py`) |
| `mcp_prefix` | `mcp` | Tool name prefix for MCP tools |

## Advanced Settings

These are internal settings that most users won't need to change.

| Module | Class | Key Settings |
|--------|-------|-------------|
| `automation_settings` | `AutomationSettings` | `one_shot_mode`, `max_follow_ups` |
| `checkpoint_settings` | `CheckpointSettings` | `enabled`, `auto_save_interval` |
| `compaction_settings` | `CompactionSettings` | `enabled`, `max_context_tokens` |
| `conversation_settings` | `ConversationSettings` | `max_history_tokens` |
| `context_settings` | `ContextSettings` | `max_context_files` |
| `event_settings` | `EventSettings` | `backend_type`, `buffer_size` |
| `exploration_settings` | `ExplorationSettings` | `max_parallel_searches` |
| `hitl_settings` | `HITLSettings` | `approval_timeout`, `auto_approve` |
| `resilience_settings` | `ResilienceSettings` | `circuit_breaker_threshold` |
| `sandbox_settings` | `SandboxSettings` | `enabled`, `timeout` |
| `search_settings` | `SearchSettings` | `max_results`, `semantic_threshold` |
| `security_settings` | `SecuritySettings` | `block_dangerous_commands` |

All settings modules live in `victor/config/*_settings.py`.

## Environment Variables

Victor reads these environment variables (prefix `VICTOR_` for most):

```bash
# Provider
VICTOR_DEFAULT_PROVIDER=anthropic
VICTOR_DEFAULT_MODEL=claude-sonnet-4-20250514

# API keys
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Local servers
OLLAMA_BASE_URL=http://localhost:11434
LMSTUDIO_BASE_URLS=http://localhost:1234

# Modes
AIRGAPPED_MODE=true        # Disable network tools
CI=true                    # Auto-disable emojis

# Logging
VICTOR_LOG_LEVEL=DEBUG
```

## Configuration Precedence

Settings are resolved in this order (highest priority first):

1. **CLI flags** (`--provider anthropic --model claude-sonnet`)
2. **Environment variables** (`VICTOR_DEFAULT_PROVIDER=anthropic`)
3. **Profile overrides** (`~/.victor/profiles.yaml` → selected profile)
4. **Settings defaults** (hardcoded in `victor/config/settings.py`)
