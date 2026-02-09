# Victor AI 0.5.0 API Documentation

> **Note**: This legacy API documentation is retained for reference. For current docs, see `docs/reference/api/`.


Complete API reference documentation for Victor AI's public interfaces.

## Documentation Files

| File | Size | Lines | Description |
|------|------|-------|-------------|
| [API_REFERENCE.md](API_REFERENCE.md) | 23 KB | 1,009 | Main APIs: AgentOrchestrator, Coordinators, Pipeline, Providers, Tools |
| [PROTOCOL_REFERENCE.md](PROTOCOL_REFERENCE.md) | 32 KB | 1,388 | All protocol interfaces for dependency injection and testing |
| [PROVIDER_REFERENCE.md](PROVIDER_REFERENCE.md) | 21 KB | 1,021 | All 21 LLM providers with configuration and usage |
| [CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md) | 23 KB | 1,009 | Settings, environment variables, profiles, and tuning |

## Quick Navigation

### By Topic

#### Getting Started
- [Configuration Reference](CONFIGURATION_REFERENCE.md#overview) - Setting up Victor AI
- [Provider Reference](PROVIDER_REFERENCE.md#overview) - Choosing and configuring providers

#### Core APIs
- [AgentOrchestrator](API_REFERENCE.md#agentorchestrator) - Main facade for agent operations
- [Provider Management](API_REFERENCE.md#provider-management) - Provider lifecycle and switching
- [Tool System](API_REFERENCE.md#tool-system) - Tool registration and execution

#### Advanced Features
- [Coordinators](API_REFERENCE.md#coordinators) - Specialized coordinators for different aspects
- [Intelligent Pipeline](API_REFERENCE.md#intelligent-pipeline) - Adaptive behavior with RL
- [Conversation Management](API_REFERENCE.md#conversation-management) - Message history and context
- [Streaming](API_REFERENCE.md#streaming) - Streaming responses

#### Reference
- [Protocols](PROTOCOL_REFERENCE.md#overview) - Protocol interfaces
- [Configuration](CONFIGURATION_REFERENCE.md) - Complete configuration reference
- [Providers](PROVIDER_REFERENCE.md) - Provider details and capabilities

## Quick Start

### 1. Basic Usage

```python
from victor.agent.orchestrator import AgentOrchestrator

# Initialize with default provider
orchestrator = AgentOrchestrator(
    provider_name="anthropic",
    model="claude-sonnet-4-5"
)

# Simple chat
response = await orchestrator.chat("Explain decorators in Python")
print(response.content)
```

### 2. Configuration

```bash
# Set provider and API key
export VICTOR_DEFAULT_PROVIDER="anthropic"
export ANTHROPIC_API_KEY="sk-ant-..."

# Or use profiles.yaml
cat > ~/.victor/profiles.yaml << EOF
default_provider: anthropic
default_model: claude-sonnet-4-5
anthropic_api_key: sk-ant-...
EOF
```

### 3. Provider Selection

- **Local (Offline)**: [Ollama](PROVIDER_REFERENCE.md#ollama), [vLLM](PROVIDER_REFERENCE.md#vllm)
- **Cloud (Production)**: [Anthropic](PROVIDER_REFERENCE.md#anthropic), [OpenAI](PROVIDER_REFERENCE.md#openai)
- **Cost-Optimized**: [Groq](PROVIDER_REFERENCE.md#groq), [Together](PROVIDER_REFERENCE.md#together)

See [Provider Reference](PROVIDER_REFERENCE.md) for all 21 providers.

## Architecture Overview

Victor AI uses a **facade pattern** with specialized coordinators following SOLID principles:

```
AgentOrchestrator (Facade)
├── ConversationCoordinator    - Message history, context tracking
├── ToolExecutionCoordinator   - Tool validation, execution, budgeting
├── PromptCoordinator          - System prompt assembly
├── StateCoordinator           - Conversation stage management
├── ProviderCoordinator        - Provider lifecycle and switching
├── SearchCoordinator          - Semantic and keyword search
├── TeamCoordinator            - Multi-agent coordination
├── CheckpointCoordinator      - State persistence
├── MetricsCoordinator         - Observability and metrics
└── IntelligentFeatureCoordinator - RL-based optimization
```

**Key Design Principles:**
- **Protocol-Based**: All components use protocols for loose coupling
- **Dependency Injection**: ServiceContainer manages 55+ services
- **Event-Driven**: Pluggable event backends for scalability
- **Provider Agnostic**: Unified interface across 21 LLM providers

## Key Features

### Provider Management
- **21 Providers**: Ollama, Anthropic, OpenAI, Google, Azure, AWS, xAI, etc.
- **Runtime Switching**: Switch providers mid-conversation
- **Load Balancing**: Provider pool with adaptive strategies
- **Health Monitoring**: Automatic failover on failures

### Tool System
- **55+ Tools**: Code search, editing, testing, DevOps, RAG, etc.
- **Intelligent Selection**: Semantic and keyword-based selection
- **Budget Management**: Configurable tool call budgets
- **Result Caching**: Optional tool result caching

### Intelligent Features
- **Adaptive Mode Control**: Q-learning for mode transitions
- **Response Quality Scoring**: Multi-dimensional quality assessment
- **Grounding Verification**: Hallucination detection
- **Resilient Execution**: Circuit breaker, retry, rate limiting

### Multi-Agent Coordination
- **Team Formations**: Pipeline, parallel, sequential, hierarchical, consensus
- **Specialized Roles**: Customizable agent personas
- **Communication Styles**: Structured and flexible coordination

## Usage Patterns

### Streaming Chat

```python
# Streaming response
async for chunk in orchestrator.chat("Implement a binary search tree", stream=True):
    if chunk.type == "content":
        print(chunk.content, end="", flush=True)
    elif chunk.type == "tool_start":
        print(f"\n[Tool: {chunk.tool_name}]")
```

### Tool Execution

```python
# Manual tool execution
result = await orchestrator.execute_tool(
    tool_name="read_file",
    arguments={"file_path": "src/main.py"}
)

# Batch execution
results = await orchestrator.execute_tools_batch([
    {"name": "read_file", "arguments": {"file_path": "a.py"}},
    {"name": "read_file", "arguments": {"file_path": "b.py"}},
], parallel=True)
```

### Provider Switching

```python
# Switch to local model
await orchestrator.switch_provider(
    provider_name="ollama",
    model="qwen2.5:32b",
    reason="cost"
)

# Switch back
await orchestrator.switch_provider(
    provider_name="anthropic",
    model="claude-sonnet-4-5",
    reason="quality"
)
```

### Semantic Search

```python
# Semantic code search
results = await orchestrator.semantic_search(
    query="database connection initialization",
    limit=10
)

for result in results:
    print(f"{result.file_path}:{result.line_number}")
    print(f"  Similarity: {result.similarity:.2f}")
```

### Team Coordination

```python
# Multi-agent team
team_spec = {
    "name": "code_review_team",
    "formation": "parallel",
    "roles": [
        {"name": "security_reviewer", "persona": "..."},
        {"name": "quality_reviewer", "persona": "..."},
    ]
}

response = await orchestrator.chat_with_team(
    message="Review this PR for security issues",
    team_spec=team_spec
)
```

## Configuration Examples

### Development Profile

```yaml
# ~/.victor/profiles.yaml
default_provider: ollama
default_model: qwen2.5:32b
default_temperature: 0.8
tool_call_budget: 50
tool_selection_strategy: keyword
log_level: DEBUG
enable_observability_logging: true
```

### Production Profile

```yaml
# ~/.victor/profiles.yaml
default_provider: anthropic
default_model: claude-sonnet-4-5
default_temperature: 0.3
tool_call_budget: 200
tool_selection_strategy: semantic
log_level: WARNING
show_cost_metrics: true
write_approval_mode: all_writes
```

### Air-Gapped Profile

```yaml
# ~/.victor/profiles.yaml
default_provider: ollama
default_model: qwen2.5:32b
airgapped_mode: true
embedding_provider: sentence-transformers
embedding_model: BAAI/bge-small-en-v1.5
codebase_vector_store: lancedb
```

## Performance Tuning

### Tool Selection Optimization

```yaml
# For large models (30B+)
tool_selection:
  model_size_tier: large
  base_threshold: 0.4
  base_max_tools: 15

# For small models (< 30B)
tool_selection:
  model_size_tier: small
  base_threshold: 0.3
  base_max_tools: 10
```

### Provider Pool Configuration

```yaml
enable_provider_pool: true
pool_size: 3
pool_load_balancer: adaptive
pool_enable_warmup: true
pool_health_check_interval: 30
```

### Embedding Optimization

```bash
# Use fast local embeddings
export VICTOR_EMBEDDING_PROVIDER=sentence-transformers
export VICTOR_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# Adjust similarity threshold
export VICTOR_SEMANTIC_SIMILARITY_THRESHOLD=0.25
```

## See Also

- `CLAUDE.md` - Project instructions and architecture (repo root)
- [Architecture](../architecture/overview.md) - Detailed architecture documentation
- [Best Practices](../architecture/BEST_PRACTICES.md) - Usage patterns and guidelines
- [Migration Guides](../architecture/MIGRATION_GUIDES.md) - How to migrate to new patterns

## Support

- **Issues**: https://github.com/yourusername/victor/issues
- **Discussions**: https://github.com/yourusername/victor/discussions
- **Documentation**: https://docs.victor.ai

---

**Version:** 0.5.0
**Reading Time:** 4 min
**Last Updated:** January 18, 2026
**License:** Apache 2.0
