# Victor AI 0.5.0 API Reference

Complete API reference documentation for Victor AI's public interfaces.

**Table of Contents**
- [AgentOrchestrator](#agentorchestrator)
- [Coordinators](#coordinators)
- [Intelligent Pipeline](#intelligent-pipeline)
- [Provider Management](#provider-management)
- [Tool System](#tool-system)
- [Conversation Management](#conversation-management)
- [Streaming](#streaming)
- [Vertical Base](#vertical-base)

---

## AgentOrchestrator

The main facade for Victor AI's agent capabilities.

### Overview

`AgentOrchestrator` is a facade pattern implementation that coordinates all agent operations through specialized coordinators. It provides a unified interface for:

- Multi-turn conversations
- Tool execution and budgeting
- Provider management and switching
- State tracking and persistence
- Streaming and batch responses
- Multi-agent team coordination

### Architecture

```
AgentOrchestrator (Facade)
├── ConversationCoordinator     - Message history, context tracking
├── ToolExecutionCoordinator    - Tool validation, execution, budgeting
├── PromptCoordinator           - System prompt assembly
├── StateCoordinator            - Conversation stage management
├── ProviderCoordinator         - Provider lifecycle and switching
├── StreamingCoordinator        - Response processing for streaming
├── SearchCoordinator           - Semantic and keyword search
├── TeamCoordinator             - Multi-agent coordination
├── CheckpointCoordinator       - State persistence
├── MetricsCoordinator          - Observability and metrics
├── EvaluationCoordinator       - Response validation
├── ResponseCoordinator         - Response formatting
└── IntelligentFeatureCoordinator - RL-based optimization
```

### Basic Usage

```python
from victor.agent.orchestrator import AgentOrchestrator
from victor.config.settings import Settings

# Initialize with settings
settings = Settings()
orchestrator = AgentOrchestrator(
    provider_name="anthropic",
    model="claude-sonnet-4-5",
    settings=settings
)

# Simple chat
response = await orchestrator.chat(
    message="Explain how decorators work in Python",
    stream=False
)
print(response.content)

# Streaming chat
async for chunk in orchestrator.chat(
    message="Implement a binary search tree",
    stream=True
):
    if chunk.type == "content":
        print(chunk.content, end="", flush=True)
```

### Main API

#### `__init__`

```python
def __init__(
    self,
    provider_name: str,
    model: str,
    settings: Optional[Settings] = None,
    project_root: Optional[Path] = None,
    session_id: Optional[str] = None,
    mode: str = "build",
    tool_budget: Optional[int] = None,
    **kwargs
)
```

Initialize a new orchestrator instance.

**Parameters:**
- `provider_name` (str): LLM provider name ("anthropic", "openai", "ollama", etc.)
- `model` (str): Model identifier
- `settings` (Settings, optional): Configuration settings
- `project_root` (Path, optional): Project root directory
- `session_id` (str, optional): Session identifier for recovery
- `mode` (str): Agent mode ("build", "plan", "explore")
- `tool_budget` (int, optional): Maximum tool calls per session
- `**kwargs`: Additional provider-specific parameters

**Returns:** `AgentOrchestrator` instance

---

#### `chat`

```python
async def chat(
    self,
    message: str,
    stream: bool = False,
    tool_budget: Optional[int] = None,
    session_id: Optional[str] = None,
    **kwargs
) -> Union[ChatResponse, AsyncIterator[StreamChunk]]
```

Primary method for conversational interactions.

**Parameters:**
- `message` (str): User message
- `stream` (bool): Enable streaming response (default: False)
- `tool_budget` (int, optional): Override default tool budget
- `session_id` (str, optional): Session identifier
- `**kwargs`: Additional parameters

**Returns:**
- Non-streaming: `ChatResponse` with `content`, `tool_calls`, `metadata`
- Streaming: `AsyncIterator[StreamChunk]` with incremental updates

**Example:**

```python
# Batch response
response = await orchestrator.chat("What files are in this project?")
print(response.content)

# Streaming response
async for chunk in orchestrator.chat("Analyze the codebase", stream=True):
    if chunk.type == "content":
        print(chunk.content, end="")
    elif chunk.type == "tool_start":
        print(f"\n[Tool: {chunk.tool_name}]")
```

---

#### `switch_provider`

```python
async def switch_provider(
    self,
    provider_name: str,
    model: Optional[str] = None,
    reason: str = "manual"
) -> bool
```

Switch to a different LLM provider during conversation.

**Parameters:**
- `provider_name` (str): Target provider name
- `model` (str, optional): Target model (defaults to current model)
- `reason` (str): Reason for switch ("manual", "health", "cost", etc.)

**Returns:** `bool` - True if switch succeeded

**Example:**

```python
# Switch to local model for cost savings
success = await orchestrator.switch_provider(
    provider_name="ollama",
    model="qwen2.5:32b",
    reason="cost"
)
```

---

#### `get_session_state`

```python
def get_session_state(self) -> Dict[str, Any]
```

Get current session state for persistence or debugging.

**Returns:** Dictionary with:
- `messages`: Conversation history
- `stage`: Current conversation stage
- `tool_calls_used`: Tool budget consumed
- `provider`: Current provider info
- `metadata`: Additional session metadata

**Example:**

```python
state = orchestrator.get_session_state()
print(f"Stage: {state['stage']}")
print(f"Tools used: {state['tool_calls_used']}/{state['tool_budget']}")
```

---

#### `set_mode`

```python
def set_mode(self, mode: str) -> bool
```

Change agent behavior mode.

**Parameters:**
- `mode` (str): One of "build", "plan", "explore"
  - "build": Full edit permissions, standard exploration
  - "plan": 2.5x exploration budget, sandbox-only edits
  - "explore": 3.0x exploration budget, no edits allowed

**Returns:** `bool` - True if mode changed successfully

**Example:**

```python
# Switch to planning mode for analysis
orchestrator.set_mode("plan")

# Switch back to build mode for implementation
orchestrator.set_mode("build")
```

---

### Advanced Usage

#### Tool Execution

```python
# Manual tool execution
result = await orchestrator.execute_tool(
    tool_name="read_file",
    arguments={"file_path": "src/main.py"}
)

# Batch tool execution
tool_calls = [
    {"name": "read_file", "arguments": {"file_path": "a.py"}},
    {"name": "read_file", "arguments": {"file_path": "b.py"}},
]

results = await orchestrator.execute_tools_batch(tool_calls, parallel=True)
```

#### Semantic Search

```python
# Semantic code search
results = await orchestrator.semantic_search(
    query="database connection initialization",
    limit=10
)

for result in results:
    print(f"{result.file_path}:{result.line_number}")
    print(f"  Similarity: {result.similarity:.2f}")
    print(f"  {result.code_snippet[:100]}...")
```

#### Team Coordination

```python
# Create a multi-agent team
team_spec = {
    "name": "code_review_team",
    "formation": "parallel",
    "roles": [
        {"name": "security_reviewer", "persona": "..."},
        {"name": "quality_reviewer", "persona": "..."},
    ]
}

# Execute with team
response = await orchestrator.chat_with_team(
    message="Review this PR for security issues",
    team_spec=team_spec
)
```

---

## Coordinators

Specialized coordinators handle specific aspects of agent behavior following SOLID principles.

### ConversationCoordinator

Manages message history, context tracking, and compaction.

```python
from victor.agent.coordinators import ConversationCoordinator

coordinator = ConversationCoordinator(
    max_context_tokens=100000,
    compaction_threshold=0.8
)

# Add messages
coordinator.add_user_message("Hello")
coordinator.add_assistant_message("Hi! How can I help?")

# Get formatted messages for provider
messages = coordinator.get_messages_for_provider()

# Compact if needed
if coordinator.should_compact():
    coordinator.compact()
```

**Key Methods:**
- `add_user_message(content)`: Add user message to history
- `add_assistant_message(content)`: Add assistant response
- `add_tool_result(tool_name, result)`: Add tool execution result
- `get_messages_for_provider()`: Get formatted message list
- `get_context_metrics()`: Get token usage statistics
- `compact_if_needed()`: Auto-compact when context nearly full

---

### ToolExecutionCoordinator

Handles tool validation, execution, and budget enforcement.

```python
from victor.agent.coordinators import ToolExecutionCoordinator

coordinator = ToolExecutionCoordinator(
    tool_registry=registry,
    budget=30,
    enable_caching=True
)

# Execute single tool
result = await coordinator.execute_tool(
    tool_name="write_file",
    arguments={"file_path": "test.py", "content": "print('hello')"}
)

# Execute batch
results = await coordinator.execute_tools_batch([
    {"name": "read_file", "arguments": {...}},
    {"name": "grep", "arguments": {...}},
], parallel=True)

# Check budget
if coordinator.is_budget_exhausted():
    print("Tool budget exhausted")
```

**Key Methods:**
- `execute_tool(tool_name, arguments)`: Execute single tool
- `execute_tools_batch(tool_calls, parallel)`: Execute multiple tools
- `is_budget_exhausted()`: Check if budget exhausted
- `get_remaining_budget()`: Get remaining tool calls
- `reset_budget(new_budget)`: Reset budget counter

---

### PromptCoordinator

Assembles system prompts from multiple components.

```python
from victor.agent.coordinators import PromptCoordinator

coordinator = PromptCoordinator(
    base_prompt="You are a coding assistant...",
    task_hints={"edit": "Focus on making minimal changes"}
)

# Build complete system prompt
system_prompt = coordinator.build(
    context=context,
    include_hints=True,
    include_grounding=True
)

# Add custom section
coordinator.add_section(
    name="coding_standards",
    content="- Use type hints\n- Write docstrings",
    priority=100
)
```

**Key Methods:**
- `build(context, include_hints)`: Build complete system prompt
- `add_section(name, content, priority)`: Add custom prompt section
- `add_task_hint(task_type, hint)`: Add task-specific guidance
- `set_grounding_mode(mode)`: Set grounding strictness ("minimal", "extended")

---

### StateCoordinator

Tracks conversation stage and manages state transitions.

```python
from victor.agent.coordinators import StateCoordinator

coordinator = StateCoordinator()

# Get current stage
stage = coordinator.get_current_stage()  # ConversationStage.PLANNING

# Transition to new stage
coordinator.transition_to(
    stage=ConversationStage.EXECUTING,
    reason="Starting implementation",
    tool_name="write_file"
)

# Check phase
if coordinator.is_in_exploration_phase():
    print("Reading and analyzing code")
elif coordinator.is_in_execution_phase():
    print("Making changes")
```

**Stages:**
- `INITIAL`: Conversation start
- `PLANNING`: Understanding requirements
- `READING`: Reading code files
- `ANALYZING`: Analyzing codebase
- `EXECUTING`: Making changes
- `VERIFICATION`: Checking results
- `COMPLETION`: Task finished

---

### ProviderCoordinator

Manages provider lifecycle, switching, and health monitoring.

```python
from victor.agent.coordinators import ProviderCoordinator

coordinator = ProviderCoordinator(
    settings=settings,
    initial_provider="anthropic",
    initial_model="claude-sonnet-4-5"
)

# Switch provider
success = await coordinator.switch_provider(
    provider_name="ollama",
    model="qwen2.5:32b",
    reason="cost"
)

# Check health
is_healthy = await coordinator.check_health()

# Get current provider
provider = coordinator.get_current_provider()
model = coordinator.get_current_model()
```

**Key Methods:**
- `switch_provider(provider_name, model)`: Switch provider
- `switch_model(model)`: Switch model on current provider
- `check_health()`: Check provider health
- `get_current_provider()`: Get provider instance
- `get_switch_history()`: Get history of provider switches

---

### SearchCoordinator

Unified interface for semantic and keyword search.

```python
from victor.agent.coordinators import SearchCoordinator

coordinator = SearchCoordinator(
    project_root="/path/to/project"
)

# Semantic search
results = await coordinator.search_semantic(
    query="authentication middleware",
    limit=10
)

# Keyword/grep search
results = await coordinator.search_keyword(
    pattern="class.*Auth",
    file_pattern="*.py"
)

# Combined search
results = await coordinator.search(
    query="find OAuth implementation",
    strategy="hybrid"  # "semantic", "keyword", or "hybrid"
)
```

**Key Methods:**
- `search_semantic(query, limit)`: Semantic search via embeddings
- `search_keyword(pattern, file_pattern)`: Grep-style search
- `search(query, strategy)`: Unified search interface
- `get_search_stats()`: Get search performance metrics

---

### TeamCoordinator

Multi-agent team coordination and orchestration.

```python
from victor.agent.coordinators import TeamCoordinator

coordinator = TeamCoordinator(orchestrator)

# Define team
team_spec = {
    "name": "review_team",
    "formation": "parallel",  # "pipeline", "parallel", "sequential", etc.
    "roles": [
        {
            "name": "security_expert",
            "persona": "You are a security specialist...",
            "tool_categories": ["security", "analysis"]
        },
        {
            "name": "performance_expert",
            "persona": "You optimize for performance...",
            "tool_categories": ["profiling", "metrics"]
        }
    ]
}

# Execute with team
response = await coordinator.execute_with_team(
    message="Review this code",
    team_spec=team_spec
)

# Get team results
for member_id, result in response.member_results.items():
    print(f"{member_id}: {result.content}")
```

**Team Formations:**
- `pipeline`: Sequential stages
- `parallel`: Concurrent execution with aggregation
- `sequential`: Step-by-step execution
- `hierarchical`: Manager-worker coordination
- `consensus`: Vote-based decision making

---

### CheckpointCoordinator

State persistence and recovery.

```python
from victor.agent.coordinators import CheckpointCoordinator

coordinator = CheckpointCoordinator(
    checkpoint_dir="/path/to/sessions"
)

# Save checkpoint
await coordinator.create_checkpoint(
    session_id="session_123",
    state=orchestrator.get_session_state()
)

# List checkpoints
checkpoints = await coordinator.list_checkpoints()

# Restore checkpoint
state = await coordinator.restore_checkpoint(
    session_id="session_123",
    checkpoint_id="ckpt_456"
)

orchestrator.load_session_state(state)
```

**Key Methods:**
- `create_checkpoint(session_id, state)`: Save state
- `restore_checkpoint(session_id, checkpoint_id)`: Load state
- `list_checkpoints()`: Get available checkpoints
- `delete_checkpoint(checkpoint_id)`: Remove checkpoint

---

### MetricsCoordinator

Observability and metrics collection.

```python
from victor.agent.coordinators import MetricsCoordinator

coordinator = MetricsCoordinator()

# Record metrics
coordinator.record_tool_call(
    tool_name="read_file",
    duration_ms=150,
    success=True
)

coordinator.record_provider_call(
    provider="anthropic",
    model="claude-sonnet-4-5",
    tokens_used=1000,
    duration_ms=2000
)

# Get metrics
metrics = coordinator.get_metrics()
print(f"Total tool calls: {metrics['tool_calls']['total']}")
print(f"Avg duration: {metrics['tool_calls']['avg_duration_ms']}ms")

# Export metrics
await coordinator.export_to_prometheus()
await coordinator.export_to_file("metrics.json")
```

---

## Intelligent Pipeline

The `IntelligentAgentPipeline` integrates learning components for adaptive behavior.

### Overview

```python
from victor.agent.intelligent_pipeline import IntelligentAgentPipeline

pipeline = await IntelligentAgentPipeline.create(
    provider_name="ollama",
    model="qwen2.5:32b",
    profile_name="local-qwen",
    project_root="/path/to/project"
)

# Prepare request
context = await pipeline.prepare_request(
    task="Implement user authentication",
    task_type="creation",
    current_mode="build",
    tool_calls_made=5,
    tool_budget=30
)

print(f"Recommended mode: {context.recommended_mode}")
print(f"Recommended budget: {context.recommended_tool_budget}")

# Execute with resilience
response = await pipeline.execute_with_resilience(
    provider=provider,
    messages=messages,
    circuit_name="ollama:qwen2.5"
)

# Process response
result = await pipeline.process_response(
    response=response.content,
    query="Implement user authentication",
    tool_calls=5,
    success=True,
    task_type="creation"
)

print(f"Quality score: {result.quality_score:.2f}")
print(f"Grounding: {result.is_grounded}")
print(f"Issues: {result.grounding_issues}")
```

### Components

1. **IntelligentPromptBuilder**: Embedding-based context selection
2. **AdaptiveModeController**: Q-learning for mode transitions
3. **ResponseQualityScorer**: Multi-dimensional quality assessment
4. **GroundingVerifier**: Hallucination detection
5. **ResilientExecutor**: Circuit breaker, retry, rate limiting

### Pipeline Statistics

```python
stats = pipeline.get_stats()
print(f"Total requests: {stats.total_requests}")
print(f"Success rate: {stats.successful_requests / stats.total_requests:.1%}")
print(f"Avg quality: {stats.avg_quality_score:.2f}")
print(f"Avg grounding: {stats.avg_grounding_score:.2f}")
print(f"Circuit trips: {stats.circuit_breaker_trips}")
```

---

## Provider Management

### Provider Registry

```python
from victor.providers.registry import ProviderRegistry

# List available providers
providers = ProviderRegistry.list_providers()
print(providers)
# ['ollama', 'lmstudio', 'vllm', 'anthropic', 'openai', ...]

# Get provider class
AnthropicProvider = ProviderRegistry.get("anthropic")

# Create provider instance
provider = ProviderRegistry.create(
    name="anthropic",
    api_key="sk-...",
    model="claude-sonnet-4-5"
)

# Register custom provider
ProviderRegistry.register(
    name="my_provider",
    provider_class=MyCustomProvider
)
```

### Base Provider Interface

All providers inherit from `BaseProvider`:

```python
from victor.providers.base import BaseProvider

class MyProvider(BaseProvider):
    @property
    def name(self) -> str:
        return "my_provider"

    async def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> ChatResponse:
        # Implement chat
        pass

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncIterator[StreamChunk]:
        # Implement streaming
        pass

    def supports_tools(self) -> bool:
        return True
```

### Provider Pool

Load balancing across multiple providers:

```python
from victor.providers.pool import ProviderPool

pool = ProviderPool(
    providers=[
        {"name": "anthropic", "model": "claude-sonnet-4-5"},
        {"name": "openai", "model": "gpt-4o"},
        {"name": "ollama", "model": "qwen2.5:32b"},
    ],
    strategy="round_robin"  # or "least_latency", "least_cost"
)

# Auto-select provider
provider = pool.get_provider()

# Execute with automatic failover
response = await pool.execute_with_fallback(
    messages=messages,
    max_retries=2
)
```

---

## Tool System

### Tool Registration

```python
from victor.tools.base import BaseTool, CostTier
from victor.agent.protocols import ToolRegistryProtocol

class MyTool(BaseTool):
    name = "my_tool"
    description = "Does something useful"

    parameters = {
        "type": "object",
        "properties": {
            "input": {"type": "string", "description": "Input text"}
        },
        "required": ["input"]
    }

    cost_tier = CostTier.FREE

    async def execute(self, **kwargs):
        input_text = kwargs["input"]
        return f"Processed: {input_text}"

# Register tool
tool_registry: ToolRegistryProtocol = ...
tool_registry.register(MyTool())
```

### Tool Execution

```python
from victor.agent.tool_pipeline import ToolPipeline

pipeline = ToolPipeline(
    tool_registry=registry,
    budget=30,
    enable_caching=True
)

# Execute tool
result = await pipeline.execute(
    tool_name="my_tool",
    arguments={"input": "hello"}
)

print(result.content)
print(result.success)
print(result.duration_ms)
print(result.cached)  # True if served from cache
```

### Tool Selection

```python
from victor.agent.tool_selector import SemanticToolSelector

selector = SemanticToolSelector(
    tool_registry=registry,
    embeddings_service=embeddings
)

# Select tools for task
context = ToolSelectionContext(
    task_type="analysis",
    stage="exploration",
    max_tools=10
)

selected_tools = await selector.select_tools(
    query="find authentication bugs",
    context=context
)

for tool in selected_tools:
    print(f"{tool.name}: {tool.description}")
```

---

## Conversation Management

### Message History

```python
from victor.agent.conversation_state import MessageHistory

history = MessageHistory()

# Add messages
history.add_message(
    role="user",
    content="Hello"
)

history.add_message(
    role="assistant",
    content="Hi! How can I help?",
    tool_calls=[...]
)

history.add_message(
    role="tool",
    content="Result",
    tool_call_id="call_123"
)

# Get messages
messages = history.get_messages()
```

### Context Manager

```python
from victor.agent.context_manager import ContextManager

manager = ContextManager(
    max_tokens=100000,
    compaction_threshold=0.8
)

# Check context
metrics = manager.get_metrics()
print(f"Usage: {metrics.utilization:.1%}")

# Compact if needed
if manager.should_compact():
    manager.compact()
```

---

## Streaming

### Streaming Controller

```python
from victor.agent.streaming_controller import StreamingController

controller = StreamingController()

# Start session
session = controller.start_session(
    session_id="session_123"
)

# Process chunks
async for chunk in stream:
    controller.record_chunk(chunk)

# End session
metrics = controller.end_session("session_123")
print(f"Duration: {metrics.duration_ms}ms")
print(f"Tokens: {metrics.total_tokens}")
```

### Stream Types

```python
class StreamChunk:
    type: str  # "content", "tool_start", "tool_result", "error"
    content: str
    metadata: Dict[str, Any]
    is_final: bool
```

---

## Vertical Base

Base class for domain-specific verticals.

```python
from victor.core.verticals.base import VerticalBase

class MyVertical(VerticalBase):
    name = "my_vertical"

    @classmethod
    def get_mode_config(cls, mode: str) -> ModeConfig:
        """Get mode configuration."""
        return ModeConfigRegistry.get_instance().get_config(
            cls.name,
            mode
        )

    @classmethod
    def list_capabilities_by_type(cls, capability_type: str):
        """List capabilities by type."""
        loader = CapabilityLoader.from_vertical(cls.name)
        return loader.list_capabilities(cls.name, capability_type)

    @classmethod
    def get_team(cls, team_name: str):
        """Get team specification."""
        provider = BaseYAMLTeamProvider(cls.name)
        return provider.get_team(team_name)
```

---

**See Also:**
- [Protocol Reference](PROTOCOL_REFERENCE.md) - Protocol interfaces
- [Provider Reference](PROVIDER_REFERENCE.md) - LLM provider details
- [Configuration Reference](CONFIGURATION_REFERENCE.md) - Settings and environment variables
