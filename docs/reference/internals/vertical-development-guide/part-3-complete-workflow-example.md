# Vertical Development Guide - Part 3

**Part 3 of 4:** Complete Workflow Example

---

## Navigation

- [Part 1: Capability & Middleware](part-1-capability-middleware.md)
- [Part 2: Chain Registry & Personas](part-2-chain-registry-personas.md)
- **[Part 3: Complete Workflow Example](#)** (Current)
- [Part 4: Appendix & Conclusion](part-4-appendix-conclusion.md)
- [**Complete Guide**](../VERTICAL_DEVELOPMENT_GUIDE.md)

---

## 5. Complete Workflow Example

This section provides a complete, step-by-step guide to creating a new vertical from scratch.

### 5.1 Vertical Structure

```text
victor/
  myvertical/
    __init__.py                 # Package initialization
    assistant.py                # VerticalBase subclass
    capabilities.py             # Capability provider
    middleware.py               # Middleware implementations
    personas.py                 # Persona definitions
    teams.py                    # Team configurations
    chains.py                   # LCEL chain definitions
    handlers.py                 # Workflow handlers
    escape_hatches.py           # YAML workflow escape hatches
    workflows/
      __init__.py               # Workflow provider
      workflow1.yaml            # YAML workflow definition
      workflow2.yaml
```

### 5.2 Step-by-Step Implementation

#### Step 1: Create Vertical Base

```python
# victor/myvertical/__init__.py
"""My custom vertical for domain-specific tasks."""

from victor.myvertical.assistant import MyVerticalAssistant

__all__ = ["MyVerticalAssistant"]
```text

```python
# victor/myvertical/assistant.py
"""My vertical assistant."""

from victor.core.verticals.base import VerticalBase
from typing import List, Dict, Any

class MyVerticalAssistant(VerticalBase):
    """Domain-specific assistant for my use case."""

    name = "my_vertical"
    description = "Specialized assistant for X domain"
    version = "0.5.0"

    @classmethod
    def get_tools(cls) -> List[str]:
        """Return tool names for this vertical."""
        return [
            "read",
            "write",
            "search",
            # Add vertical-specific tools
            "my_tool",
        ]

    @classmethod
    def get_system_prompt(cls) -> str:
        """Return system prompt."""
        return """You are an expert in X domain.

Your role is to:
- Task 1
- Task 2
- Task 3

Use the available tools to accomplish these goals efficiently."""

    @classmethod
    def get_middleware(cls) -> List[Any]:
        """Return middleware for this vertical."""
        from victor.myvertical.middleware import MyVerticalMiddleware
        return [MyVerticalMiddleware()]

    @classmethod
    def get_workflow_provider(cls) -> Optional[Any]:
        """Return workflow provider."""
        from victor.myvertical.workflows import MyVerticalWorkflowProvider
        return MyVerticalWorkflowProvider()
```

#### Step 2: Create Capability Provider

```python
# victor/myvertical/capabilities.py
"""Capability definitions for my vertical."""

from typing import Any, Callable, Dict, List, Set
from victor.framework.capabilities import BaseCapabilityProvider, CapabilityMetadata

# Configuration functions
def configure_my_capability(
    orchestrator: Any,
    *,
    param1: str = "default",
    param2: int = 100,
) -> None:
    """Configure capability on orchestrator."""
    if hasattr(orchestrator, "my_capability_config"):
        orchestrator.my_capability_config = {
            "param1": param1,
            "param2": param2,
        }

def get_my_capability(orchestrator: Any) -> Dict[str, Any]:
    """Get current capability configuration."""
    return getattr(
        orchestrator,
        "my_capability_config",
        {"param1": "default", "param2": 100},
    )

# Capability provider
class MyVerticalCapabilityProvider(BaseCapabilityProvider[Callable[..., None]]):
    """Capability provider for my vertical."""

    def __init__(self):
        self._applied: Set[str] = set()
        self._capabilities: Dict[str, Callable[..., None]] = {
            "my_capability": configure_my_capability,
        }
        self._metadata: Dict[str, CapabilityMetadata] = {
            "my_capability": CapabilityMetadata(
                name="my_capability",
                description="Description of capability",
                version="1.0",
                tags=["config", "domain"],
            ),
        }

    def get_capabilities(self) -> Dict[str, Callable[..., None]]:
        return self._capabilities.copy()

    def get_capability_metadata(self) -> Dict[str, CapabilityMetadata]:
        return self._metadata.copy()

    def apply_my_capability(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply capability."""
        configure_my_capability(orchestrator, **kwargs)
        self._applied.add("my_capability")

__all__ = [
    "configure_my_capability",
    "get_my_capability",
    "MyVerticalCapabilityProvider",
]
```text

#### Step 3: Create Middleware

```python
# victor/myvertical/middleware.py
"""Middleware for my vertical."""

from typing import Any, Dict, Set
from victor.core.verticals.protocols import (
    MiddlewareProtocol,
    MiddlewareResult,
    MiddlewarePriority,
)

class MyVerticalMiddleware(MiddlewareProtocol):
    """Custom middleware for my vertical."""

    def __init__(self, enabled: bool = True):
        self._enabled = enabled
        self._sensitive_tools = {"sensitive_tool", "dangerous_operation"}

    @property
    def priority(self) -> MiddlewarePriority:
        return MiddlewarePriority.MEDIUM

    @property
    def name(self) -> str:
        return "my_vertical_middleware"

    async def before_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> MiddlewareResult:
        """Validate before tool execution."""
        if not self._enabled:
            return MiddlewareResult(should_proceed=True)

        # Check sensitive operations
        if tool_name in self._sensitive_tools:
            if not self._is_safe_operation(arguments):
                return MiddlewareResult(
                    should_proceed=False,
                    error="Operation blocked by safety policy",
                )

        return MiddlewareResult(should_proceed=True)

    async def after_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
    ) -> Any:
        """Post-process result."""
        # Add metadata, logging, etc.
        return result

    def _is_safe_operation(self, arguments: Dict[str, Any]) -> bool:
        """Custom safety logic."""
        # Implement safety checks
        return True

__all__ = ["MyVerticalMiddleware"]
```

#### Step 4: Define Personas and Teams

```python
# victor/myvertical/personas.py
"""Persona definitions for my vertical."""

from victor.framework.multi_agent import (
    PersonaTraits,
    CommunicationStyle,
    ExpertiseLevel,
)

# Define personas
SPECIALIST = PersonaTraits(
    name="X Specialist",
    role="specialist",
    description="Expert in X domain with deep knowledge",
    communication_style=CommunicationStyle.TECHNICAL,
    expertise_level=ExpertiseLevel.SPECIALIST,
    strengths=["analysis", "optimization"],
    preferred_tools=["tool1", "tool2"],
    risk_tolerance=0.3,
)

REVIEWER = PersonaTraits(
    name="X Reviewer",
    role="reviewer",
    description="Reviews work for quality and correctness",
    communication_style=CommunicationStyle.FORMAL,
    expertise_level=ExpertiseLevel.EXPERT,
    strengths=["validation", "quality_assurance"],
    risk_tolerance=0.2,
)

__all__ = ["SPECIALIST", "REVIEWER"]
```text

```python
# victor/myvertical/teams.py
"""Team configurations for my vertical."""

from victor.framework.multi_agent import (
    TeamTemplate,
    TeamTopology,
    TaskAssignmentStrategy,
)
from victor.myvertical.personas import SPECIALIST, REVIEWER

# Define team template
ANALYSIS_TEAM = TeamTemplate(
    name="Analysis Team",
    description="Analyzes X domain problems",
    topology=TeamTopology.HUB_SPOKE,
    assignment_strategy=TaskAssignmentStrategy.SKILL_MATCH,
    member_slots={
        "specialist": 2,
        "reviewer": 1,
    },
    shared_context_keys=["problem", "context", "constraints"],
    escalation_threshold=0.75,
    max_iterations=10,
)

__all__ = ["ANALYSIS_TEAM"]
```

#### Step 5: Create LCEL Chains

```python
# victor/myvertical/chains.py
"""LCEL chains for my vertical."""

from victor.tools.composition import (
    RunnableSequence,
    RunnableParallel,
    RunnableLambda,
)

# Example: Analysis chain
analysis_chain = (
    RunnableParallel({
        "data": read_data,
        "context": gather_context,
    })
    | RunnableLambda(lambda x: analyze(x["data"], x["context"]))
    | RunnableLambda(lambda x: {
        "results": x,
        "confidence": calculate_confidence(x),
    })
)

# Register chains
from victor.framework.chains.registry import get_chain_registry

def register_myvertical_chains():
    """Register all chains for my vertical."""
    registry = get_chain_registry()

    registry.register_chain(
        name="myvertical.analysis",
        version="0.5.0",
        chain=analysis_chain,
        category="analysis",
        description="Analyze data in X domain",
        tags=["analysis", "domain"],
    )

__all__ = ["analysis_chain", "register_myvertical_chains"]
```text

#### Step 6: Create Workflow Handlers

```python
# victor/myvertical/handlers.py
"""Workflow handlers for my vertical."""

from typing import Any, Dict

class AnalysisHandler:
    """Handler for analysis operations."""

    def __init__(self):
        pass

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute analysis."""
        data = context.get("data", {})
        # Perform analysis
        results = self._analyze(data)
        return {"results": results}

    def _analyze(self, data: Any) -> Dict[str, Any]:
        # Analysis logic
        return {"status": "analyzed", "data": data}

HANDLERS = {
    "analysis": AnalysisHandler(),
}

def register_handlers():
    """Register handlers with global registry."""
    from victor.workflows.handlers import HandlerRegistry
    registry = HandlerRegistry()
    for name, handler in HANDLERS.items():
        registry.register(name, handler)

__all__ = ["HANDLERS", "register_handlers"]
```

#### Step 7: Define Escape Hatches

```python
# victor/myvertical/escape_hatches.py
"""Escape hatches for YAML workflows."""

from typing import Any, Dict

# Condition functions
def check_quality(ctx: Dict[str, Any]) -> str:
    """Check if results meet quality threshold."""
    quality_score = ctx.get("quality_score", 0.0)
    if quality_score >= 0.8:
        return "high_quality"
    return "needs_improvement"

def has_errors(ctx: Dict[str, Any]) -> str:
    """Check if there are errors."""
    errors = ctx.get("errors", [])
    if errors:
        return "has_errors"
    return "no_errors"

# Transform functions
def normalize_results(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize results for output."""
    raw = ctx.get("raw_results", {})
    return {
        "normalized_results": {
            "summary": raw.get("summary"),
            "metrics": raw.get("metrics"),
        }
    }

def format_output(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Format output for display."""
    results = ctx.get("results", {})
    return {
        "output": f"Analysis complete: {results.get('status')}",
        "details": results,
    }

# Condition registry
CONDITIONS = {
    "quality_check": check_quality,
    "error_check": has_errors,
}

# Transform registry
TRANSFORMS = {
    "normalize": normalize_results,
    "format": format_output,
}

__all__ = ["CONDITIONS", "TRANSFORMS"]
```text

#### Step 8: Create YAML Workflow

```yaml
# victor/myvertical/workflows/analysis.yaml
workflows:
  analysis:
    description: "Comprehensive analysis workflow"
    metadata:
      version: "1.0"
      vertical: "my_vertical"

    nodes:
      - id: gather_data
        type: agent
        role: specialist
        goal: "Gather relevant data for analysis"
        tool_budget: 10
        output: raw_data
        next: [analyze]

      - id: analyze
        type: compute
        handler: analysis
        inputs:
          data: $ctx.raw_data
        output: results
        next: [check_quality]

      - id: check_quality
        type: condition
        condition: "quality_check"
        branches:
          "high_quality": format_output
          "needs_improvement": improve

      - id: improve
        type: agent
        role: specialist
        goal: "Improve analysis results"
        tool_budget: 5
        output: improved_results
        next: [format_output]

      - id: format_output
        type: transform
        handler: "format"
        inputs:
          results: $ctx.results
        output: final_output
        next: []
```

#### Step 9: Create Workflow Provider

```python
# victor/myvertical/workflows/__init__.py
"""Workflow provider for my vertical."""

from typing import List, Optional, Tuple
from victor.framework.workflows import BaseYAMLWorkflowProvider

class MyVerticalWorkflowProvider(BaseYAMLWorkflowProvider):
    """Provides my vertical workflows."""

    def _get_escape_hatches_module(self) -> str:
        return "victor.myvertical.escape_hatches"

    def _get_capability_provider_module(self) -> Optional[str]:
        return "victor.myvertical.capabilities"

    def get_auto_workflows(self) -> List[Tuple[str, str]]:
        """Return automatic workflow triggers."""
        return [
            (r"analyze\s+.*", "analysis"),
            (r"comprehensive\s+analysis", "analysis"),
        ]

    def get_workflow_for_task_type(self, task_type: str) -> Optional[str]:
        """Map task types to workflows."""
        mapping = {
            "analysis": "analysis",
            "investigation": "analysis",
        }
        return mapping.get(task_type.lower())

# Register handlers when module loads
from victor.myvertical.handlers import register_handlers
register_handlers()

__all__ = ["MyVerticalWorkflowProvider"]
```text

#### Step 10: Register Vertical

```python
# victor/core/verticals/__init__.py (add this section)

def _register_builtin_verticals():
    """Register all built-in verticals."""
    # ... existing registrations ...

    # Register my vertical
    from victor.myvertical.assistant import MyVerticalAssistant
    VerticalRegistry.register(MyVerticalAssistant)
```

### 5.3 Testing Your Vertical

```python
# tests/unit/myvertical/test_assistant.py
"""Tests for my vertical."""

import pytest
from victor.myvertical.assistant import MyVerticalAssistant

def test_vertical_metadata():
    """Test vertical metadata."""
    assert MyVerticalAssistant.name == "my_vertical"
    assert MyVerticalAssistant.description != ""
    assert MyVerticalAssistant.version == "0.5.0"

def test_get_tools():
    """Test tool list."""
    tools = MyVerticalAssistant.get_tools()
    assert isinstance(tools, list)
    assert "read" in tools
    assert "write" in tools

def test_get_system_prompt():
    """Test system prompt."""
    prompt = MyVerticalAssistant.get_system_prompt()
    assert isinstance(prompt, str)
    assert len(prompt) > 0

def test_get_config():
    """Test configuration."""
    config = MyVerticalAssistant.get_config()
    assert config is not None
    assert config.tools is not None
    assert config.system_prompt is not None

@pytest.mark.asyncio
async def test_create_agent():
    """Test agent creation."""
    # This test requires proper provider setup
    agent = await MyVerticalAssistant.create_agent(
        provider="anthropic",
        model="claude-sonnet-4-5",
    )
    assert agent is not None
```text

```python
# tests/unit/myvertical/test_capabilities.py
"""Tests for my vertical capabilities."""

import pytest
from victor.myvertical.capabilities import MyVerticalCapabilityProvider

def test_capability_provider():
    """Test capability provider."""
    provider = MyVerticalCapabilityProvider()

    # Test capabilities
    capabilities = provider.get_capabilities()
    assert isinstance(capabilities, dict)
    assert "my_capability" in capabilities

    # Test metadata
    metadata = provider.get_capability_metadata()
    assert "my_capability" in metadata
    assert metadata["my_capability"].version == "1.0"

def test_apply_capability():
    """Test applying capability."""
    provider = MyVerticalCapabilityProvider()

    # Mock orchestrator
    class MockOrchestrator:
        pass

    orchestrator = MockOrchestrator()
    provider.apply_my_capability(orchestrator, param1="test")

    # Verify applied
    assert "my_capability" in provider.get_applied()
```

```python
# tests/unit/myvertical/test_workflows.py
"""Tests for my vertical workflows."""

import pytest
from victor.myvertical.workflows import MyVerticalWorkflowProvider

def test_workflow_provider():
    """Test workflow provider."""
    provider = MyVerticalWorkflowProvider()

    # Test workflow loading
    workflows = provider.get_workflows()
    assert isinstance(workflows, dict)
    assert "analysis" in workflows

    # Test auto-workflows
    auto_triggers = provider.get_auto_workflows()
    assert len(auto_triggers) > 0

def test_task_type_mapping():
    """Test task type to workflow mapping."""
    provider = MyVerticalWorkflowProvider()

    workflow = provider.get_workflow_for_task_type("analysis")
    assert workflow == "analysis"

@pytest.mark.asyncio
async def test_workflow_execution():
    """Test workflow execution."""
    provider = MyVerticalWorkflowProvider()

    # Note: This requires a proper orchestrator
    # or compute-only workflow
    result = await provider.run_compiled_workflow(
        "analysis",
        {"query": "test"},
    )
    assert result is not None
```text

### 5.4 Documentation

Create a README for your vertical:

```markdown
# My Vertical

Domain-specific assistant for X.

---

## See Also

- [Documentation Home](../../README.md)


**Reading Time:** 7 min
**Last Updated:** February 08, 2026**
