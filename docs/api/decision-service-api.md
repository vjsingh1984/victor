# Decision Service API Reference

**Provider-Agnostic Tier System**

The Decision Service provides intelligent decision-making capabilities through a tiered model system that automatically adapts to your active provider.

---

## Overview

The decision service routes different decision types to appropriate model tiers:
- **Edge**: Fast models for micro-decisions (~5-50ms, ~50 tokens)
- **Balanced**: Mid-tier models for moderate decisions (~500ms, ~200 tokens)
- **Performance**: Frontier models for complex decisions (~2s, ~500 tokens)

### Provider-Agnostic Design

✅ **Auto-detects** your active provider from the orchestrator
✅ **Resolves** appropriate models from `provider_model_tiers` mapping
✅ **Ensures** decision service uses same provider as main LLM
✅ **Supports** explicit overrides via `tier_overrides`

---

## Configuration

### DecisionServiceSettings

```python
from victor.config.decision_settings import DecisionServiceSettings

config = DecisionServiceSettings(
    enabled=True,
    # Tiers use "auto" by default for provider-agnostic behavior
    edge=DecisionModelSpec(
        provider="auto",  # Auto-detect from active provider
        model="auto",     # Use provider's edge model
        timeout_ms=4000,
        max_tokens=50,
    ),
    balanced=DecisionModelSpec(
        provider="auto",
        model="auto",
        timeout_ms=8000,
        max_tokens=200,
    ),
    performance=DecisionModelSpec(
        provider="auto",
        model="auto",
        timeout_ms=15000,
        max_tokens=500,
    ),
    # Provider-specific model mappings
    provider_model_tiers={
        "anthropic": {
            "edge": "claude-3-5-haiku-20241022",
            "balanced": "claude-sonnet-4-20250514",
            "performance": "claude-opus-4-5-20251101",
        },
        "openai": {
            "edge": "gpt-5.4-mini",
            "balanced": "gpt-5.4",
            "performance": "gpt-5.4-pro",
        },
        # ... other providers
    },
    # Explicit overrides (optional)
    tier_overrides={
        "edge": {"provider": "ollama", "model": "phi3:mini"}
    }
)
```

### Provider Model Mappings

Each provider defines their own edge/balanced/performance models:

| Provider | Edge | Balanced | Performance |
|----------|------|----------|-------------|
| **Anthropic** | claude-3-5-haiku-20241022 | claude-sonnet-4-20250514 | claude-opus-4-5-20251101 |
| **OpenAI** | gpt-5.4-mini | gpt-5.4 | gpt-5.4-pro |
| **Google** | gemini-3.1-lite | gemini-3.1-flash | gemini-3.1-pro |
| **xAI** | grok-4.1-fast | grok-4.1-fast | grok-4.20 |
| **DeepSeek** | deepseek-chat | deepseek-chat | deepseek-chat |
| **Ollama** | qwen3.5:2b | qwen2.5-coder:14b | qwen3.5:27b-q4_K_M |

See `provider_model_tiers` in `DecisionServiceSettings` for complete list.

---

## Usage

### Basic Usage

```python
from victor.agent.services.tiered_decision_service import TieredDecisionService
from victor.config.decision_settings import DecisionServiceSettings
from victor.agent.decisions.schemas import DecisionType

# Create service with default settings
config = DecisionServiceSettings()
service = TieredDecisionService(config)

# Make a decision
result = service.decide_sync(
    DecisionType.TOOL_SELECTION,
    {"message": "fix the failing test", "tools": ["code_search", "file_read"]}
)

print(f"Decision: {result.result}")
print(f"Source: {result.source}")  # "llm" or "heuristic"
print(f"Confidence: {result.confidence}")
```

### Decision Types

| Decision Type | Default Tier | Description |
|--------------|--------------|-------------|
| `TOOL_SELECTION` | edge | Select appropriate tool for task |
| `SKILL_SELECTION` | edge | Choose skill to activate |
| `STAGE_DETECTION` | edge | Detect conversation stage |
| `INTENT_CLASSIFICATION` | edge | Classify user intent |
| `TASK_COMPLETION` | edge | Determine if task is complete |
| `ERROR_CLASSIFICATION` | edge | Classify error type |
| `CONTINUATION_ACTION` | edge | Decide next action |
| `LOOP_DETECTION` | edge | Detect stuck loops |
| `PROMPT_FOCUS` | edge | Determine prompt focus area |
| `QUESTION_CLASSIFICATION` | edge | Classify question type |
| `TASK_TYPE_CLASSIFICATION` | balanced | Classify task complexity |
| `MULTI_SKILL_DECOMPOSITION` | balanced | Decompose into skills |
| `COMPACTION` | auto | Auto-select based on complexity |

### Tier Override

```python
# Force edge tier to use specific provider/model
config = DecisionServiceSettings(
    tier_overrides={
        "edge": {
            "provider": "ollama",
            "model": "phi3:mini"
        }
    }
)
service = TieredDecisionService(config)

# Edge tier will always use Ollama phi3:mini
result = service.decide_sync(DecisionType.TOOL_SELECTION, {...})
```

### Custom Routing

```python
# Route specific decision types to different tiers
config = DecisionServiceSettings(
    tier_routing={
        "tool_selection": "balanced",  # Use balanced instead of edge
        "compaction": "performance",    # Always use performance
    }
)
service = TieredDecisionService(config)
```

---

## API Reference

### TieredDecisionService

#### `__init__(config: DecisionServiceSettings)`

Create a new tiered decision service.

**Parameters**:
- `config`: Decision service configuration

**Example**:
```python
service = TieredDecisionService(config)
```

---

#### `decide_sync(decision_type, context, *, heuristic_result=None, heuristic_confidence=0.0) -> DecisionResult`

Make a decision using the appropriate tier.

**Parameters**:
- `decision_type` (DecisionType): Type of decision to make
- `context` (Dict[str, Any]): Decision context
- `heuristic_result` (Any, optional): Fallback result if no LLM available
- `heuristic_confidence` (float, optional): Confidence of heuristic result (default: 0.0)

**Returns**: `DecisionResult`
- `result` (Any): The decision result
- `source` (str): "llm", "heuristic", or "edge_fallback"
- `confidence` (float): Confidence score (0.0-1.0)
- `latency_ms` (float): Decision latency in milliseconds

**Example**:
```python
result = service.decide_sync(
    DecisionType.TOOL_SELECTION,
    {"message": "search code", "tools": ["code_search"]}
)
```

---

#### `_detect_active_provider() -> Optional[str]`

Auto-detect the active provider from orchestrator.

**Returns**: Provider name (e.g., "anthropic", "openai") or None

**Detection Order**:
1. Service container (ProviderService/ProviderManager)
2. Settings.default_provider
3. Environment variable `VICTOR_PROVIDER`

**Example**:
```python
provider = service._detect_active_provider()
print(f"Active provider: {provider}")  # e.g., "anthropic"
```

---

### DecisionResult

```python
class DecisionResult:
    result: Any              # The decision result
    source: str              # "llm", "heuristic", "edge_fallback"
    confidence: float        # 0.0 to 1.0
    latency_ms: float        # Decision latency
    decision_type: DecisionType
```

---

## Fallback Chain

When a tier is unavailable, the service falls back through the chain:

```
performance → balanced → edge → heuristic
```

**Example**:
```python
# If performance tier unavailable:
service._failed_tiers.add("performance")
result = service.decide_sync(DecisionType.TASK_TYPE_CLASSIFICATION, {...})
# Falls back to balanced tier
```

---

## Validation

### `validate_provider_tiers()`

Validate provider tier configurations.

**Returns**: Dict with validation results
```python
{
    "valid": bool,
    "warnings": List[str],  # Missing tiers, unknown tiers
    "errors": List[str]      # Invalid overrides
}
```

**Example**:
```python
result = config.validate_provider_tiers()
if not result["valid"]:
    for warning in result["warnings"]:
        print(f"Warning: {warning}")
    for error in result["errors"]:
        print(f"Error: {error}")
```

---

## Best Practices

### 1. Use Provider-Agnostic Defaults

```python
# GOOD - Auto-detects provider
config = DecisionServiceSettings()

# BAD - Hardcoded provider
config = DecisionServiceSettings(
    edge=DecisionModelSpec(provider="anthropic", model="claude-haiku")
)
```

### 2. Override Only When Necessary

```python
# GOOD - Override for specific use case
config = DecisionServiceSettings(
    tier_overrides={
        "edge": {"provider": "ollama", "model": "phi3:mini"}
    }
)

# BAD - Override all tiers
config = DecisionServiceSettings(
    tier_overrides={
        "edge": {...},
        "balanced": {...},
        "performance": {...}
    }
)
```

### 3. Validate Configuration

```python
config = DecisionServiceSettings()
validation = config.validate_provider_tiers()

if not validation["valid"]:
    # Handle validation errors
    for error in validation["errors"]:
        logger.error(f"Config error: {error}")
```

---

## Migration Guide

### From Hardcoded Tiers

**Before**:
```python
# Hardcoded to specific providers
config = DecisionServiceSettings(
    edge=DecisionModelSpec(provider="ollama", model="qwen3.5:2b"),
    balanced=DecisionModelSpec(provider="deepseek", model="deepseek-chat"),
    performance=DecisionModelSpec(provider="anthropic", model="claude-sonnet-4")
)
```

**After**:
```python
# Provider-agnostic - auto-detects from active provider
config = DecisionServiceSettings()
# Tiers automatically use active provider's models
```

### With Explicit Overrides

```python
# Override specific tier while keeping others auto-detected
config = DecisionServiceSettings(
    tier_overrides={
        "performance": {
            "provider": "anthropic",
            "model": "claude-opus-4-5"
        }
    }
)
# Edge and balanced auto-detect, performance uses Anthropic Opus
```

---

## Troubleshooting

### Issue: "Could not auto-detect active provider"

**Solution**: Ensure provider is configured in one of these ways:
1. ProviderService is registered in container
2. Settings.default_provider is set
3. Environment variable `VICTOR_PROVIDER` is set

### Issue: "No edge model defined for provider 'xxx'"

**Solution**: Add tier mappings to `provider_model_tiers`:
```python
config = DecisionServiceSettings(
    provider_model_tiers={
        "xxx": {
            "edge": "xxx-edge-model",
            "balanced": "xxx-balanced-model",
            "performance": "xxx-performance-model",
        }
    }
)
```

### Issue: Tier always uses fallback

**Solution**: Check tier availability:
```python
# Check if tier failed
if "edge" in service._failed_tiers:
    print("Edge tier failed to initialize")

# Check provider detection
provider = service._detect_active_provider()
print(f"Detected provider: {provider}")
```

---

## See Also

- [Provider-Agnostic Architecture](../architecture/provider-agnostic-tiers.md)
- [Decision Settings](../../victor/config/decision_settings.py)
- [Tiered Decision Service](../../victor/agent/services/tiered_decision_service.py)
