# Compaction Enhancement API Reference

## New Decision Types

### DecisionType.COMPACTION

```python
from victor.agent.decisions.schemas import DecisionType, CompactionDecision

# Decision type
decision_type = DecisionType.COMPACTION  # "compaction"

# Decision schema
decision = CompactionDecision(
    complexity="simple",  # or "complex"
    recommended_tier="edge",  # or "balanced", "performance"
    estimated_tokens=1000,
    confidence=0.9,
    reason="Simple compaction with few messages",
)
```

### SystemPromptOptimizationDecision

```python
from victor.agent.decisions.schemas import SystemPromptOptimizationDecision

optimization = SystemPromptOptimizationDecision(
    include_sections=["task_guidance", "completion"],
    add_context_reminder=True,
    add_failure_hints=False,
    adjust_for_complexity=True,
    confidence=0.85,
    reason="Optimization based on task complexity",
)
```

## ContextCompactor API

### Constructor

```python
from victor.agent.context_compactor import ContextCompactor

compactor = ContextCompactor(
    controller: Optional["ConversationController"] = None,
    config: Optional[CompactorConfig] = None,
    pruning_learner: Optional["ContextPruningLearner"] = None,
    provider_type: str = "cloud",
    event_bus: Optional[Any] = None,
    decision_service: Optional[Any] = None,  # NEW: TieredDecisionService
)
```

### Methods

#### check_and_compact

```python
action = compactor.check_and_compact(
    current_query: Optional[str] = None,
    force: bool = False,
    tool_call_count: int = 0,
    task_complexity: str = "medium",
) -> CompactionAction
```

Automatically routes compaction through decision service:
- Simple (≤8 messages) → Edge tier
- Complex (>8 messages) → Performance tier
- Falls back to main LLM if decision service unavailable

#### get_prompt_optimization_decision (NEW)

```python
decision = compactor.get_prompt_optimization_decision(
    current_query: Optional[str] = None,
    recent_failures: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]
```

Returns optimization decision for dynamic system prompts:
- `include_sections`: List of prompt sections to include
- `add_context_reminder`: Whether to add compaction reminder
- `add_failure_hints`: Whether to include failure hints
- `adjust_for_complexity`: Whether to tailor for complexity
- `confidence`: Decision confidence (0.0-1.0)
- `reason`: Explanation of the decision

#### _get_provider_for_tier (NEW)

```python
provider = compactor._get_provider_for_tier(
    tier: str,  # "edge", "balanced", or "performance"
) -> Optional["BaseProvider"]
```

Creates a provider instance for the specified tier.

## SystemPromptBuilder API

### Constructor (Updated)

```python
from victor.agent.prompt_builder import SystemPromptBuilder

builder = SystemPromptBuilder(
    provider_name: str = "anthropic",
    model: str = "claude-3-5-sonnet-20241022",
    tool_adapter: Optional[BaseToolCallingAdapter] = None,
    capabilities: Optional[ToolCallingCapabilities] = None,
    prompt_contributors: Optional[list] = None,
    tool_guidance_strategy: Optional[ToolGuidanceStrategy] = None,
    task_type: str = "medium",
    available_tools: Optional[List[str]] = None,
    enrichment_service: Optional["PromptEnrichmentService"] = None,
    vertical: Optional[str] = None,
    concise_mode: bool = False,
    query_classification: Optional["QueryClassification"] = None,
    provider_caches: bool = False,
    provider_has_kv_cache: bool = False,
    system_prompt_strategy: str = "static",  # NEW: "static", "dynamic", or "hybrid"
)
```

### Methods (Updated)

#### build (Updated)

```python
prompt = builder.build() -> str
```

Builds system prompt with strategy-based caching:
- **static**: Cache after first build, reuse for all turns
- **dynamic**: Rebuild every turn based on context
- **hybrid**: Static for API providers, dynamic for local

#### _get_effective_strategy (NEW)

```python
strategy = builder._get_effective_strategy() -> str
```

Returns the effective strategy based on configuration and provider type:
- Resolves "hybrid" to "static" or "dynamic"
- Considers provider caching capabilities

## Configuration API

### ContextSettings (Updated)

```python
from victor.config.context_settings import ContextSettings

settings = ContextSettings(
    # ... existing settings ...
    system_prompt_strategy: str = "static",  # NEW
)
```

**Values:**
- `"static"`: Freeze at session start (default, 50-90% cache discount)
- `"dynamic"`: Rebuild every turn (no cache benefit)
- `"hybrid"`: Static for API, dynamic for local

### DecisionServiceSettings (Updated)

```python
from victor.config.decision_settings import DecisionServiceSettings

settings = DecisionServiceSettings(
    enabled: bool = True,
    edge: DecisionModelSpec = ...,
    balanced: DecisionModelSpec = ...,
    performance: DecisionModelSpec = ...,
    tier_routing: Dict[str, str] = {
        # ... existing routes ...
        "compaction": "auto",  # NEW: Auto-select based on complexity
    },
)
```

## Usage Examples

### Example 1: Basic Tiered Compaction

```python
from victor.agent.context_compactor import ContextCompactor
from victor.agent.services.tiered_decision_service import create_tiered_decision_service

# Create decision service
decision_service = create_tiered_decision_service()

# Create compactor with tier-based routing
compactor = ContextCompactor(
    controller=conversation_controller,
    decision_service=decision_service,
)

# Compaction will use edge tier for simple, performance for complex
action = compactor.check_and_compact(current_query)
print(f"Trigger: {action.trigger}, Messages removed: {action.messages_removed}")
```

### Example 2: Dynamic System Prompts

```python
from victor.agent.prompt_builder import SystemPromptBuilder

# Create builder with dynamic strategy
builder = SystemPromptBuilder(
    provider_name="anthropic",
    model="claude-3-5-sonnet-20241022",
    system_prompt_strategy="dynamic",
)

# Prompt will rebuild every turn
prompt1 = builder.build()
# ... context changes ...
prompt2 = builder.build()  # Rebuilt with new context
```

### Example 3: Hybrid Strategy

```python
# Create builder with hybrid strategy
builder = SystemPromptBuilder(
    provider_name="ollama",  # Local provider
    model="qwen2.5:7b",
    system_prompt_strategy="hybrid",
    provider_caches=False,  # Local provider doesn't cache
)

# Will behave as dynamic (no cache benefit)
strategy = builder._get_effective_strategy()
assert strategy == "dynamic"
```

### Example 4: Prompt Optimization

```python
# Get optimization decision for dynamic prompts
optimization = compactor.get_prompt_optimization_decision(
    current_query="fix authentication bug",
    recent_failures=["Connection timeout", "Invalid token"],
)

if optimization["add_failure_hints"]:
    # Add failure hints to prompt
    prompt += "\n\nRecent failures to avoid:\n"
    for failure in recent_failures:
        prompt += f"- {failure}\n"
```

## Return Types

### CompactionAction

```python
@dataclass
class CompactionAction:
    trigger: CompactionTrigger
    messages_removed: int = 0
    chars_freed: int = 0
    tokens_freed: int = 0
    truncations_applied: int = 0
    new_utilization: float = 0.0
    details: List[str] = field(default_factory=list)

    @property
    def action_taken(self) -> bool:
        return self.messages_removed > 0 or self.truncations_applied > 0
```

### DecisionResult

```python
@dataclass
class DecisionResult:
    decision_type: DecisionType
    result: Any  # CompactionDecision or other schema
    source: str  # "llm", "heuristic", or "cache"
    confidence: float
    cached: bool = False
```

## Error Handling

### Graceful Degradation

All new features include graceful fallback:

1. **Decision Service Unavailable**: Falls back to main LLM
2. **Provider Creation Failed**: Falls back to default provider
3. **Invalid Strategy**: Defaults to "static"
4. **Optimization Failed**: Returns None, continues without optimization

### Example

```python
# Decision service unavailable - falls back gracefully
compactor = ContextCompactor(
    controller=controller,
    decision_service=None,  # No decision service
)

# Still works, uses main LLM for all compaction
action = compactor.check_and_compact(query)
```

## Performance Considerations

### Caching Behavior

| Strategy | Cache Hit Rate | Token Savings | Latency |
|----------|---------------|---------------|----------|
| static   | 90%+          | 50-90%        | Lowest   |
| dynamic  | 0%            | 0%            | Higher   |
| hybrid   | Variable      | Variable      | Adaptive |

### Tier Selection

| Tier    | Use Case           | Cost   | Quality | Latency |
|---------|-------------------|--------|---------|---------|
| edge    | Simple (≤8 msgs)   | Free   | Good    | ~5ms    |
| balanced| Medium complexity  | Low    | Good    | ~500ms  |
| performance| Complex (>8 msgs)| High   | Best    | ~2s     |

## Migration Guide

### Minimal Changes

```python
# Before
compactor = ContextCompactor(controller)

# After (add decision service)
decision_service = create_tiered_decision_service()
compactor = ContextCompactor(
    controller=controller,
    decision_service=decision_service,
)
```

### Full Migration

```python
# Before
builder = SystemPromptBuilder(
    provider_name="anthropic",
    model="claude-3-5-sonnet-20241022",
)

# After (add strategy)
builder = SystemPromptBuilder(
    provider_name="anthropic",
    model="claude-3-5-sonnet-20241022",
    system_prompt_strategy="hybrid",
    provider_caches=True,
)
```

## See Also

- [Architecture Documentation](../architecture/compaction-enhancements.md)
- [Decision Service Guide](../decision-service.md)
- [Context Management](../context-management.md)
