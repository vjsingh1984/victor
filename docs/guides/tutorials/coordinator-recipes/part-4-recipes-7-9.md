# Coordinator Recipes - Part 4

**Part 4 of 4:** Recipes 7-9 (Analytics Dashboard, Tool Selection, A/B Testing)

---

## Navigation

- [Part 1: Recipes 1-2](part-1-recipes-1-2.md)
- [Part 2: Recipe 3](part-2-recipe-3.md)
- [Part 3: Recipes 4-6](part-3-recipes-4-6.md)
- **[Part 4: Recipes 7-9](#)** (Current)
- [**Complete Guide**](../coordinator_recipes.md)

---
## Recipe 8: Custom Tool Selection Strategy

### Problem Statement

You want to customize how tools are selected based on your specific use case.

### Solution Overview

Create a custom `IToolSelectionStrategy` that implements your selection logic.

### Step-by-Step Instructions

#### Step 1: Create Custom Selection Strategy

```python
# tool_selection.py
from typing import List
from victor.protocols import IToolSelectionStrategy, Tool
from victor.agent.coordinators.tool_selection_coordinator import IToolSelectionCoordinator

class CostAwareToolSelectionStrategy(IToolSelectionStrategy):
    """Select tools based on cost constraints."""

    def __init__(self, max_cost_per_call: float = 0.01):
        self.max_cost_per_call = max_cost_per_call

    async def select_tools(
        self,
        query: str,
        available_tools: List[Tool],
        selection_coordinator: IToolSelectionCoordinator
    ) -> List[Tool]:
        """Select tools within cost budget."""
        # Filter tools by cost
        affordable_tools = [
            tool for tool in available_tools
            if tool.cost_tier in ["free", "low"]
        ]

        # Use semantic selection on affordable tools
        return await selection_coordinator._semantic_selection(
            query,
            affordable_tools
        )

class DomainSpecificToolSelectionStrategy(IToolSelectionStrategy):
    """Select tools based on domain classification."""

    def __init__(self, domain_classifier):
        self.domain_classifier = domain_classifier

        self.domain_tool_mapping = {
            "coding": ["read", "write", "search", "bash"],
            "data_analysis": ["pandas_query", "visualization", "statistics"],
            "web": ["web_search", "web_scrape", "browse"],
        }

    async def select_tools(
        self,
        query: str,
        available_tools: List[Tool],
        selection_coordinator: IToolSelectionCoordinator
    ) -> List[Tool]:
        """Select tools based on domain."""
        # Classify query domain
        domain = await self.domain_classifier.classify(query)

        # Get allowed tools for domain
        allowed_tool_names = self.domain_tool_mapping.get(domain, [])

        # Filter available tools
        domain_tools = [
            tool for tool in available_tools
            if tool.name in allowed_tool_names
        ]

        return domain_tools
```text

#### Step 2: Register Strategy

```python
# main.py
from victor.agent.coordinators import ToolSelectionCoordinator
from victor.agent.orchestrator import AgentOrchestrator

# Create custom strategy
strategy = CostAwareToolSelectionStrategy(max_cost_per_call=0.01)

# Create tool selection coordinator
tool_selection_coordinator = ToolSelectionCoordinator(
    default_strategy=strategy
)

# Create orchestrator
orchestrator = AgentOrchestrator(
    settings=Settings(),
    provider=provider,
    model=model,
    _tool_selection_coordinator=tool_selection_coordinator
)
```

### Production Considerations

- **Caching**: Cache classification results
- **Fallback**: Provide fallback to default strategy
- **Monitoring**: Track strategy performance
- **A/B testing**: Test multiple strategies

---

## Recipe 9: A/B Testing Coordinator

### Problem Statement

You want to A/B test different configurations or prompts.

### Solution Overview

Create a coordinator that randomly assigns sessions to different configurations.

### Step-by-Step Instructions

```python
# ab_testing_coordinator.py
import random
from typing import Dict
from victor.agent.coordinators import ConfigCoordinator

class ABTestingConfigCoordinator(ConfigCoordinator):
    """A/B test different configurations."""

    def __init__(self, variants: Dict[str, dict], weights: Dict[str, float] = None):
        """
        Args:
            variants: Dict of variant_name -> config
            weights: Dict of variant_name -> weight (for weighted sampling)
        """
        self.variants = variants
        self.weights = weights or {v: 1.0 for v in variants}
        self.session_assignments = {}

    async def load_config(self, session_id: str, config_override: dict = None) -> dict:
        """Load config for session (assigning to A/B variant)."""
        # Assign session to variant
        if session_id not in self.session_assignments:
            variant = self._assign_variant(session_id)
        else:
            variant = self.session_assignments[session_id]

        # Load variant config
        variant_config = self.variants[variant].copy()

        # Apply overrides
        if config_override:
            variant_config.update(config_override)

        # Track assignment in analytics
        await self._track_assignment(session_id, variant)

        return variant_config

    def _assign_variant(self, session_id: str) -> str:
        """Assign session to variant."""
        # Weighted random selection
        variants = list(self.variants.keys())
        weights = [self.weights.get(v, 1.0) for v in variants]

        variant = random.choices(variants, weights=weights, k=1)[0]
        self.session_assignments[session_id] = variant

        return variant

    async def _track_assignment(self, session_id: str, variant: str):
        """Track A/B assignment in analytics."""
        # Emit analytics event
        pass
```text

### Usage

```python
# main.py
coordinator = ABTestingConfigCoordinator(
    variants={
        "control": {"temperature": 0.7, "model": "claude-sonnet-4-5"},
        "variant_a": {"temperature": 0.5, "model": "claude-sonnet-4-5"},
        "variant_b": {"temperature": 0.9, "model": "claude-opus-4-5"},
    },
    weights={
        "control": 0.5,
        "variant_a": 0.25,
        "variant_b": 0.25,
    }
)

orchestrator = AgentOrchestrator(
    settings=Settings(),
    provider=provider,
    model="claude-sonnet-4-5",
    _config_coordinator=coordinator
)

# Sessions are automatically assigned to variants
response = await orchestrator.chat("Hello!", session_id="session-123")
```

---

## Summary

This recipes guide provided 9 practical solutions:

1. **File-based configuration** - Load config from YAML/JSON files
2. **Project-specific prompts** - Add coding standards, compliance
3. **Database analytics** - Export analytics to PostgreSQL
4. **Smart context compaction** - Semantic compaction strategies
5. **Custom middleware** - Add logging, auth, rate limiting
6. **Multi-tenant configuration** - Serve multiple tenants
7. **Real-time dashboard** - WebSocket-based analytics
8. **Custom tool selection** - Cost-aware, domain-specific
9. **A/B testing** - Test different configurations

### Next Steps

- [Quick Start Guide](coordinator_quickstart.md) - Get started
- [Usage Examples](../examples/coordinator_examples.md) - More examples
- [Migration Guide](../migration/orchestrator_refactoring_guide.md) - Migrate from legacy

---

**Document Version**: 1.0
**Last Updated**: 2025-01-13
**Next Review**: 2025-04-13

---

**End of Recipes**

---

**Last Updated:** February 01, 2026
**Reading Time:** 6 minutes
