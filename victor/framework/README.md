# Victor Framework

The Victor Framework provides a simplified API for building AI coding agents with support for 21 LLM providers, 55+ specialized tools, and 5 domain verticals (Coding, DevOps, RAG, Data Analysis, Research).

## Quick Start

```python
from victor.framework import Agent, Task, Tools, State

# Create an agent
agent = await Agent.create(provider="anthropic")

# Run a task
result = await agent.run("Write a hello world function in Python")
print(result.content)

# Stream events
async for event in agent.stream("Refactor this code"):
    if event.type == EventType.CONTENT:
        print(event.content, end="")
```

## Core Concepts

Victor Framework is built around 5 core concepts:

1. **Agent** - Single entry point for creating agents
2. **Task** - What the agent should accomplish (TaskResult)
3. **Tools** - Available capabilities (ToolSet with presets)
4. **State** - Observable conversation state (Stage enum)
5. **Event** - Stream of observations (EventType enum)

---

## Table of Contents

- [Agent Creation](#agent-creation)
- [Tools Configuration](#tools-configuration)
- [Task Execution](#task-execution)
- [State Management](#state-management)
- [Event Streaming](#event-streaming)
- [Framework Capabilities](#framework-capabilities)
- [Workflow Coordinators](#workflow-coordinators)
- [Multi-Agent Teams](#multi-agent-teams)
- [Graph Workflows](#graph-workflows)
- [Validation Framework](#validation-framework)
- [Resilience Patterns](#resilience-patterns)
- [Service Lifecycle](#service-lifecycle)
- [Health & Metrics](#health--metrics)
- [API Reference](#api-reference)

---

## Agent Creation

### Basic Agent

```python
from victor.framework import Agent

# Simple agent with default settings
agent = await Agent.create(provider="anthropic")

# Run a query
result = await agent.run("Explain async/await in Python")
print(result.content)
```

### Agent with Configuration

```python
from victor.framework import Agent, AgentConfig, ToolSet

# Create with configuration
config = AgentConfig(
    provider="anthropic",
    model="claude-3-5-sonnet-20241022",
    tools=ToolSet.default(),
    system_prompt="You are a helpful Python expert.",
)

agent = await Agent.create(config=config)
```

### Agent with Builder

```python
from victor.framework import create_builder

builder = create_builder()
agent = (
    builder
    .with_provider("openai")
    .with_model("gpt-4")
    .with_tools(ToolSet.default())
    .with_system_prompt("You are a coding assistant.")
    .build()
)
```

---

## Tools Configuration

### Tool Presets

```python
from victor.framework import ToolSet

# Minimal tools (read, write, edit)
agent = await Agent.create(tools=ToolSet.minimal())

# Default tools (recommended balance)
agent = await Agent.create(tools=ToolSet.default())

# Full access to all 45+ tools
agent = await Agent.create(tools=ToolSet.full())

# Custom tool selection
agent = await Agent.create(
    tools=ToolSet.custom(["read", "write", "grep", "run_tests"])
)
```

### Tool Categories

```python
from victor.framework import ToolCategory, ToolSet

# Select by category
tools = ToolSet.builder() \
    .with_category(ToolCategory.FILE_OPS) \
    .with_category(ToolCategory.SEARCH) \
    .build()

agent = await Agent.create(tools=tools)
```

### Tool Configuration

```python
from victor.framework import configure_tools, ToolConfigMode

# Configure tools with filters
configured_tools = configure_tools(
    tools=ToolSet.full(),
    mode=ToolConfigMode.AIRGAPPED,  # Only local tools
    cost_tier="low",  # Only low-cost tools
    security_level="high",
)
```

---

## Task Execution

### Simple Tasks

```python
from victor.framework import Task

# Run a task
result = await agent.run("Write a function to calculate fibonacci")

# Check result
if result.success:
    print(result.content)
else:
    print(f"Error: {result.error}")
```

### Task with Options

```python
from victor.framework import Task, FrameworkTaskType

# Specify task type
result = await agent.run(
    "Add unit tests for this function",
    task_type=FrameworkTaskType.TEST_GENERATION,
    tool_budget=20,
    max_iterations=5,
)
```

### Streaming Results

```python
from victor.framework import EventType

# Stream events
async for event in agent.stream("Analyze this codebase"):
    if event.type == EventType.CONTENT:
        print(event.content, end="")
    elif event.type == EventType.TOOL_CALL:
        print(f"\n[Using {event.tool_name}]")
    elif event.type == EventType.STAGE_CHANGE:
        print(f"\n[Stage: {event.new_stage}]")
```

---

## State Management

### Observing State

```python
from victor.framework import Stage

# Check current stage
print(f"Stage: {agent.state.stage}")

# Check tool usage
print(f"Tools used: {agent.state.tool_calls_used}/{agent.state.tool_budget}")

# Get conversation history
messages = agent.state.get_messages()
```

### State Change Hooks

```python
# Register state change callback
def on_stage_change(old_stage: Stage, new_stage: Stage):
    print(f"Stage changed: {old_stage} -> {new_stage}")

agent.on_state_change(on_stage_change)
```

### Stage Definitions

```python
from victor.framework import Stage

# Common stages
Stage.INITIAL     # Starting point
Stage.PLANNING    # Planning approach
Stage.READING     # Reading files
Stage.ANALYZING   # Analyzing code
Stage.EXECUTING   # Making changes
Stage.VERIFICATION  # Checking results
Stage.COMPLETION  # Task complete
```

---

## Event Streaming

### Event Types

```python
from victor.framework import EventType

# Content events
EventType.CONTENT          # LLM response content
EventType.THINKING        # Thinking/reasoning content

# Tool events
EventType.TOOL_CALL       # Tool being called
EventType.TOOL_RESULT     # Tool result received
EventType.TOOL_ERROR      # Tool execution failed

# Workflow events
EventType.STAGE_CHANGE    # Stage transition
EventType.PROGRESS        # Progress update
EventType.MILESTONE       # Milestone reached

# Stream events
EventType.STREAM_START    # Stream started
EventType.STREAM_END      # Stream ended

# Error events
EventType.ERROR           # Error occurred
```

### Event Filtering

```python
# Filter specific event types
async for event in agent.stream(task):
    if event.type in (EventType.CONTENT, EventType.THINKING):
        print(event.content, end="")
```

---

## Framework Capabilities

### Retry with Exponential Backoff

```python
from victor.framework.resilience import with_exponential_backoff

# Decorator for automatic retry
@with_exponential_backoff(max_retries=5, base_delay=2.0)
async def fetch_data(url: str):
    return await http_client.get(url)

# Or use standalone function
from victor.framework.resilience import retry_with_backoff

result = await retry_with_backoff(
    api_client.call,
    "https://api.example.com/data",
    max_retries=5,
)
```

### Circuit Breaker

```python
from victor.framework.resilience import CircuitBreaker

@CircuitBreaker(failure_threshold=5, recovery_timeout=60)
async def call_external_api():
    return await api.request()
```

### Validation Pipeline

```python
from victor.framework.validation import (
    ValidationPipeline,
    ThresholdValidator,
    PatternValidator,
)

# Create validation pipeline
pipeline = ValidationPipeline(
    validators=[
        ThresholdValidator(field="score", min_value=0, max_value=100),
        PatternValidator(field="email", pattern=r"^[a-zA-Z0-9_.+-]+@"),
    ],
    halt_on_error=True,
)

# Validate data
result = pipeline.validate({"score": 85, "email": "user@example.com"})
if result.is_valid:
    print("Valid!")
else:
    for error in result.errors:
        print(f"Error: {error}")
```

---

## Workflow Coordinators

### YAML Workflow Execution

```python
from victor.framework.coordinators import YAMLWorkflowCoordinator

coordinator = YAMLWorkflowCoordinator()
result = await coordinator.execute(
    "workflow.yaml",
    initial_state={"query": "analyze this code"}
)
```

### StateGraph Execution

```python
from victor.framework.coordinators import GraphExecutionCoordinator
from victor.framework import StateGraph, CompiledGraph

# Create graph
graph = StateGraph(state_schema)
graph.add_node("analyze", analyze_node)
graph.add_node("execute", execute_node)
graph.add_edge("analyze", "execute")
compiled = graph.compile()

# Execute
coordinator = GraphExecutionCoordinator()
result = await coordinator.execute(compiled)
```

### Human-in-the-Loop

```python
from victor.framework.coordinators import HITLCoordinator

coordinator = HITLCoordinator(timeout_seconds=300)

# Set custom handler
class ApprovalHandler:
    def request_approval(self, request):
        print(f"Approve: {request.description}?")
        return input("y/n: ").lower() == "y"

coordinator.set_handler(ApprovalHandler())

# Execute with HITL
result = await coordinator.execute("workflow_with_approval.yaml")
```

### Cache Management

```python
from victor.framework.coordinators import CacheCoordinator

cache_coordinator = CacheCoordinator()
cache_coordinator.enable_caching(ttl_seconds=3600)

# Clear cache
cache_coordinator.clear_cache()
```

---

## Multi-Agent Teams

### Team Formation

```python
from victor.teams import TeamFormation, create_coordinator

# Create parallel team
coordinator = create_coordinator(formation=TeamFormation.PARALLEL)

# Add members
coordinator.add_member("security_reviewer", role="security")
coordinator.add_member("quality_reviewer", role="quality")

# Execute
result = await coordinator.execute("Review this code")
```

### Team from YAML

```yaml
teams:
  code_review_team:
    formation: parallel
    roles:
      - name: security_reviewer
        persona: "You are a security expert..."
        tools: [security_scan, vulnerability_check]
      - name: quality_reviewer
        persona: "You are a code quality expert..."
        tools: [complexity_check, style_linter]
```

```python
from victor.teams import create_coordinator

coordinator = create_coordinator.from_yaml("team.yaml")
result = await coordinator.execute("Review this code")
```

---

## Graph Workflows

### Creating a StateGraph

```python
from victor.framework import StateGraph, START, END

# Define state schema
class AgentState(TypedDict):
    messages: List[Message]
    next_action: str

# Create graph
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_node("supervisor", supervisor_node)

# Add edges
graph.add_edge(START, "supervisor")
graph.add_conditional_edges(
    "supervisor",
    route_action,
    {
        "agent": "agent",
        "tools": "tools",
        "end": END,
    }
)

# Compile and run
compiled = graph.compile()
result = await compiled.invoke({"messages": []})
```

### Checkpointing

```python
from victor.framework import MemoryCheckpointer

# Add checkpointing for state persistence
checkpointer = MemoryCheckpointer()
compiled = graph.compile(checkpointer=checkpointer)

# Run with thread_id for checkpointing
result = await compiled.invoke(
    {"messages": []},
    config={"configurable": {"thread_id": "session-1"}}
)
```

---

## Validation Framework

### Built-in Validators

```python
from victor.framework.validation import (
    ThresholdValidator,
    RangeValidator,
    PresenceValidator,
    PatternValidator,
    TypeValidator,
    LengthValidator,
)

# Threshold (min/max value)
ThresholdValidator(field="score", min_value=0, max_value=100)

# Range (inclusive)
RangeValidator(field="age", min_value=18, max_value=120)

# Presence (required field)
PresenceValidator(field="email")

# Pattern (regex)
PatternValidator(field="email", pattern=r"^[a-zA-Z0-9_.+-]+@")

# Type checking
TypeValidator(field="count", expected_type=int)

# Length checking
LengthValidator(field="name", min_length=1, max_length=100)
```

### Composite Validators

```python
from victor.framework.validation import CompositeValidator, CompositeLogic

# All must pass (AND)
composite = CompositeValidator(
    validators=[validator1, validator2],
    logic=CompositeLogic.ALL,
)

# Any must pass (OR)
composite = CompositeValidator(
    validators=[validator1, validator2],
    logic=CompositeLogic.ANY,
)
```

### YAML Validation

```yaml
workflows:
  validate_user_input:
    nodes:
      - id: validate
        type: validate_pipeline
        validators:
          - type: threshold
            field: score
            min: 0
            max: 100
          - type: pattern
            field: email
            pattern: "^[a-zA-Z0-9_.+-]+@"
        halt_on_error: true
```

---

## Resilience Patterns

### Retry Strategies

```python
from victor.framework.resilience import (
    ExponentialBackoffStrategy,
    FixedDelayStrategy,
    LinearBackoffStrategy,
    NoRetryStrategy,
)

# Exponential backoff (default)
strategy = ExponentialBackoffStrategy(
    max_attempts=5,
    base_delay=1.0,
    max_delay=60.0,
    multiplier=2.0,
    jitter=0.1,
)

# Fixed delay
strategy = FixedDelayStrategy(max_attempts=3, delay=2.0)

# Linear backoff
strategy = LinearBackoffStrategy(
    max_attempts=5,
    base_delay=1.0,
    increment=1.0,
)
```

### Specialized Retry Handlers

```python
# Network operations
from victor.framework.resilience import NetworkRetryHandler

# Rate-limited APIs
from victor.framework.resilience import RateLimitRetryHandler

# Database operations
from victor.framework.resilience import DatabaseRetryHandler
```

### Circuit Breaker

```python
from victor.framework.resilience import CircuitBreaker, CircuitState

@CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60,
    expected_exception=ConnectionError,
)
async def external_api_call():
    return await api.get_data()

# Check circuit state
breaker = CircuitBreaker.get_breaker("external_api_call")
print(f"State: {breaker.state}")  # CLOSED, OPEN, HALF_OPEN
```

---

## Service Lifecycle

### Creating Services

```python
from victor.framework.services import (
    create_sqlite_service,
    create_http_service,
    ServiceManager,
)

# Create SQLite service
db = await create_sqlite_service(
    name="project_db",
    db_path="./data/project.db",
    enable_wal=True,
)

# Create HTTP service
api = await create_http_service(
    name="external_api",
    base_url="https://api.example.com",
    timeout=30,
)

# Register with manager
manager = ServiceManager()
await manager.register(db)
await manager.register(api)

# Start all services
await manager.start_all()
```

### Service Health

```python
# Check individual service health
health = await db.get_health()
print(f"Status: {health.status}")
print(f"Message: {health.message}")

# Check all services
report = await manager.health_check()
for service, health in report.items():
    print(f"{service}: {health.status}")
```

### YAML Service Configuration

```yaml
services:
  - name: database
    type: sqlite
    db_path: ./data/project.db
    enable_wal: true

  - name: api_client
    type: http
    base_url: https://api.example.com
    timeout: 30
    headers:
      Authorization: "Bearer ${API_TOKEN}"
```

---

## Health & Metrics

### Health Checking

```python
from victor.framework.health import (
    HealthChecker,
    HealthStatus,
    create_default_health_checker,
)

# Create checker
checker = create_default_health_checker()

# Add custom check
async def check_custom_service():
    try:
        await service.ping()
        return HealthStatus.HEALTHY
    except:
        return HealthStatus.UNHEALTHY

checker.register("custom", check_custom_service)

# Run health checks
report = await checker.check_all()
print(f"Overall: {report.overall_status}")
```

### Metrics Collection

```python
from victor.framework.metrics import (
    Counter,
    Gauge,
    Histogram,
    Timer,
    MetricsRegistry,
)

# Get registry
registry = MetricsRegistry()

# Create metrics
counter = registry.create_counter("tool_calls_total")
gauge = registry.create_gauge("active_sessions")
histogram = registry.create_histogram("request_duration")

# Use metrics
counter.inc(labels={"tool": "read"})
gauge.set(5)

with Timer(histogram, labels={"endpoint": "api"}):
    # ... do work ...
    pass
```

### OpenTelemetry

```python
from victor.framework.metrics import setup_opentelemetry

# Enable telemetry
setup_opentelemetry(
    service_name="victor-agent",
    endpoint="http://localhost:4317",
)

# Traces are now automatically collected
```

---

## API Reference

### Agent

```python
class Agent:
    """Main agent class for AI interactions."""

    async def create(
        provider: str = "anthropic",
        model: Optional[str] = None,
        tools: Optional[ToolSet] = None,
        config: Optional[AgentConfig] = None,
    ) -> Agent:
        """Create a new agent instance."""

    async def run(
        self,
        prompt: str,
        task_type: Optional[FrameworkTaskType] = None,
        tool_budget: Optional[int] = None,
        max_iterations: Optional[int] = None,
    ) -> TaskResult:
        """Run a task synchronously."""

    async def stream(
        self,
        prompt: str,
        **kwargs
    ) -> AsyncIterator[AgentExecutionEvent]:
        """Stream task execution events."""

    def get_orchestrator(self) -> OrchestratorProtocol:
        """Get the underlying orchestrator for advanced use cases."""
```

### TaskResult

```python
@dataclass
class TaskResult:
    """Result of a task execution."""

    success: bool
    content: str
    error: Optional[str] = None
    tool_calls_used: int = 0
    tokens_used: int = 0
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### ToolSet

```python
class ToolSet:
    """Collection of tools for agent configuration."""

    @staticmethod
    def minimal() -> ToolSet:
        """Minimal tool set (read, write, edit)."""

    @staticmethod
    def default() -> ToolSet:
        """Default balanced tool set."""

    @staticmethod
    def full() -> ToolSet:
        """All available tools."""

    @staticmethod
    def custom(tools: List[str]) -> ToolSet:
        """Custom tool selection."""

    @staticmethod
    def builder() -> ToolSetBuilder:
        """Builder for custom tool sets."""
```

### AgentExecutionEvent

```python
@dataclass
class AgentExecutionEvent:
    """Event during agent execution."""

    type: EventType
    timestamp: float
    content: Optional[str] = None
    tool_name: Optional[str] = None
    tool_result: Optional[Any] = None
    old_stage: Optional[Stage] = None
    new_stage: Optional[Stage] = None
    error: Optional[str] = None
```

---

## Configuration

### Environment Variables

```bash
# Provider selection
VICTOR_PROVIDER=anthropic
VICTOR_MODEL=claude-3-5-sonnet-20241022

# API keys (or use .env file)
ANTHROPIC_API_KEY=your-key
OPENAI_API_KEY=your-key

# Tool selection
VICTOR_TOOL_SELECTION_STRATEGY=hybrid

# Caching
VICTOR_ENABLE_CACHE=true
VICTOR_CACHE_TTL=3600

# Telemetry
VICTOR_OTEL_ENDPOINT=http://localhost:4317
```

### Settings File

```yaml
# config/settings.yaml
agent:
  provider: anthropic
  model: claude-3-5-sonnet-20241022
  temperature: 0.7
  max_tokens: 4096

tools:
  selection_strategy: hybrid
  enable_cache: true
  cache_ttl: 3600

workflows:
  enable_checkpointing: true
  checkpoint_path: ./checkpoints

observability:
  enable_metrics: true
  enable_tracing: true
  otel_endpoint: http://localhost:4317
```

---

## Advanced Topics

### Custom Verticals

```python
from victor.core.verticals.base import VerticalBase

class MyVertical(VerticalBase):
    """Custom vertical implementation."""

    name = "my_vertical"

    @classmethod
    def get_tools(cls) -> List[BaseTool]:
        return [custom_tool1, custom_tool2]

    @classmethod
    def get_system_prompt(cls) -> str:
        return "You are a specialized assistant..."
```

### Custom Step Handlers

**Step Handlers** are the primary extension mechanism for vertical development. They provide a clean, SOLID-compliant way to customize how verticals are integrated with orchestrators.

```python
from victor.framework.step_handlers import BaseStepHandler, StepHandlerRegistry

class CustomToolsHandler(BaseStepHandler):
    """Handle custom tool registration with validation."""

    @property
    def name(self) -> str:
        return "custom_tools"

    @property
    def order(self) -> int:
        return 12  # After default tools (10), before tiered config (15)

    def _do_apply(self, orchestrator, vertical, context, result):
        tools = vertical.get_tools()
        validated = self._validate_tools(tools)
        context.apply_enabled_tools(validated)
        result.add_info(f"Applied {len(validated)} tools")

# Register and use
registry = StepHandlerRegistry.default()
registry.add_handler(CustomToolsHandler())

# Use with pipeline
from victor.framework.vertical_integration import VerticalIntegrationPipeline

pipeline = VerticalIntegrationPipeline(step_registry=registry)
result = pipeline.apply(orchestrator, MyVertical)
```

**Benefits:**
- **Testable**: Each handler can be unit tested independently
- **Reusable**: Share handlers across verticals
- **Maintainable**: Clear separation of concerns
- **Observable**: Per-step status tracking and metrics

**Built-in Handlers:**
- `CapabilityConfigStepHandler` (order=5) - Capability config storage
- `ToolStepHandler` (order=10) - Tool filter application
- `TieredConfigStepHandler` (order=15) - Tiered tool config
- `PromptStepHandler` (order=20) - System prompt and contributors
- `ConfigStepHandler` (order=40) - Stages, modes, tool dependencies
- `ExtensionsStepHandler` (order=45) - Coordinated extension application
- `MiddlewareStepHandler` (order=50) - Middleware chain application
- `FrameworkStepHandler` (order=60) - Workflows, RL, teams, chains, personas
- `ContextStepHandler` (order=100) - Attach context to orchestrator

**See Also:**
- [Step Handler Guide](../../docs/extensions/step_handler_guide.md) - Complete guide to step handlers
- [Step Handler Examples](../../docs/extensions/step_handler_examples.md) - Practical examples
- [Migration Guide](../../docs/extensions/step_handler_migration.md) - Migrating from direct extension

### Custom Personas

```python
from victor.framework.personas import Persona, register_persona

analyst = Persona(
    name="analyst",
    system_prompt="You are a data analyst...",
    tools=["query_db", "visualize"],
    temperature=0.3,
)

register_persona(analyst)

# Use persona
agent = await Agent.create(persona="analyst")
```

### Custom Validators

```python
from victor.framework.validation import ValidatorProtocol, ValidationResult

class CustomValidator(ValidatorProtocol):
    def validate(self, data: Dict, context: ValidationContext) -> ValidationResult:
        result = ValidationResult()
        # Custom validation logic
        if not self.is_valid(data):
            result.add_error("field", "Custom error message")
        return result
```

---

## See Also

- [Migration Guide](../../MIGRATION_GUIDE.md) - Migrating from previous versions
- [CLAUDE.md](../../CLAUDE.md) - Architecture and design patterns
- [roadmap.md](../../roadmap.md) - Planned features and improvements
