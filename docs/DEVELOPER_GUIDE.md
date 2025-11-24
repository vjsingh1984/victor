# Victor Developer Guide

> Comprehensive guide for developers contributing to or extending Victor

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Core Concepts](#core-concepts)
- [System Design](#system-design)
- [Module Deep Dives](#module-deep-dives)
- [Adding New Features](#adding-new-features)
- [Development Workflow](#development-workflow)
- [Best Practices](#best-practices)

## Architecture Overview

Victor follows a layered architecture with clear separation of concerns:

```mermaid
graph TB
    subgraph "Presentation Layer"
        CLI[CLI Interface]
        MCP[MCP Server]
    end

    subgraph "Application Layer"
        Agent[Agent Orchestrator]
        Context[Context Manager]
        Editor[Multi-File Editor]
    end

    subgraph "Domain Layer"
        Tools[Tool Registry]
        Providers[Provider Registry]
        Embeddings[Embedding Registry]
    end

    subgraph "Infrastructure Layer"
        Config[Configuration]
        Storage[File System]
        Network[HTTP Client]
    end

    CLI --> Agent
    MCP --> Agent
    Agent --> Context
    Agent --> Tools
    Agent --> Providers
    Editor --> Storage
    Tools --> Storage
    Tools --> Network
    Context --> Embeddings

    style Agent fill:#f9f,stroke:#333,stroke-width:4px
    style Tools fill:#bbf,stroke:#333,stroke-width:2px
    style Providers fill:#bbf,stroke:#333,stroke-width:2px
```

### Design Principles

1. **Provider Abstraction**: Unified interface for all LLM providers
2. **Tool Registry**: Dynamic tool discovery and registration
3. **Plugin Architecture**: Easy extension without core modifications
4. **Type Safety**: Pydantic models throughout
5. **Async First**: All I/O operations are async
6. **Transaction-Based**: Atomic operations with rollback support

## Core Concepts

### 1. Provider System

```mermaid
classDiagram
    class BaseProvider {
        <<abstract>>
        +generate(messages, **kwargs) ChatResponse
        +generate_stream(messages, **kwargs) AsyncIterator
        #_normalize_response(response) ChatResponse
        #_normalize_tool_calls(tools) List~ToolCall~
    }

    class OllamaProvider {
        -base_url: str
        -client: httpx.AsyncClient
        +generate() ChatResponse
        +generate_stream() AsyncIterator
        -_convert_tools() List~Dict~
    }

    class AnthropicProvider {
        -api_key: str
        -client: Anthropic
        +generate() ChatResponse
        +generate_stream() AsyncIterator
    }

    class OpenAIProvider {
        -api_key: str
        -client: OpenAI
        +generate() ChatResponse
        +generate_stream() AsyncIterator
    }

    BaseProvider <|-- OllamaProvider
    BaseProvider <|-- AnthropicProvider
    BaseProvider <|-- OpenAIProvider

    class ProviderRegistry {
        -providers: Dict
        +register(name, provider)
        +get(name) BaseProvider
        +list() List~str~
    }

    ProviderRegistry --> BaseProvider
```

**Key Points**:
- All providers implement `BaseProvider` interface
- Response normalization ensures consistent output
- Tool call translation handles provider-specific formats
- Async streaming for real-time responses

### 2. Tool System

```mermaid
classDiagram
    class BaseTool {
        <<abstract>>
        +name: str
        +description: str
        +parameters: List~ToolParameter~
        +execute(**kwargs) ToolResult
    }

    class ToolParameter {
        +name: str
        +type: str
        +description: str
        +required: bool
    }

    class ToolResult {
        +success: bool
        +output: str
        +error: str
    }

    class FileSystemTool {
        +execute(**kwargs) ToolResult
        -_read_file(path) str
        -_write_file(path, content)
    }

    class DatabaseTool {
        -connections: Dict
        +execute(**kwargs) ToolResult
        -_connect(params) str
        -_query(sql) ToolResult
    }

    class DockerTool {
        +execute(**kwargs) ToolResult
        -_run_docker_command(args) tuple
    }

    BaseTool <|-- FileSystemTool
    BaseTool <|-- DatabaseTool
    BaseTool <|-- DockerTool
    BaseTool --> ToolParameter
    BaseTool --> ToolResult

    class ToolRegistry {
        -_tools: Dict
        +register(tool: BaseTool)
        +get(name: str) BaseTool
        +list_tools() List~BaseTool~
    }

    ToolRegistry --> BaseTool
```

**Key Points**:
- Standardized tool interface
- Self-describing with parameters
- Consistent result format
- Registry pattern for discovery

### 3. Agent Orchestrator

```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant Context
    participant Provider
    participant Tool

    User->>Agent: chat(message)
    Agent->>Context: add_message(user, message)
    Agent->>Context: get_messages()
    Context-->>Agent: conversation_history

    Agent->>Provider: generate(messages, tools)
    Provider-->>Agent: response

    alt Has Tool Calls
        Agent->>Tool: execute(**params)
        Tool-->>Agent: ToolResult
        Agent->>Context: add_message(tool, result)
        Agent->>Provider: generate(updated_messages)
        Provider-->>Agent: final_response
    end

    Agent->>Context: add_message(assistant, response)
    Agent-->>User: ChatResponse
```

**Key Points**:
- Manages conversation flow
- Coordinates tool execution
- Handles multi-turn interactions
- Context window management

## System Design

### Request Flow

```mermaid
flowchart TD
    Start([User Request]) --> Parse{Parse<br/>Command}

    Parse -->|Chat| Agent[Agent Orchestrator]
    Parse -->|MCP| MCP[MCP Server]
    Parse -->|Direct| Tool[Tool Execution]

    Agent --> Context[Get Context]
    Context --> Provider[Select Provider]
    Provider --> Generate[Generate Response]

    Generate --> HasTools{Has Tool<br/>Calls?}
    HasTools -->|Yes| ExecTools[Execute Tools]
    ExecTools --> UpdateContext[Update Context]
    UpdateContext --> Provider

    HasTools -->|No| Format[Format Response]
    Format --> Return([Return to User])

    MCP --> Discovery[Tool Discovery]
    Discovery --> Execute[Execute via MCP]
    Execute --> Return

    Tool --> DirectExec[Direct Execution]
    DirectExec --> Return

    style Agent fill:#f96,stroke:#333,stroke-width:3px
    style Provider fill:#6f9,stroke:#333,stroke-width:2px
    style ExecTools fill:#69f,stroke:#333,stroke-width:2px
```

### Multi-File Editing System

```mermaid
stateDiagram-v2
    [*] --> Idle

    Idle --> Planning: start_transaction()
    Planning --> Editing: add_edit()
    Editing --> Editing: add_edit()
    Editing --> Validating: commit()

    Validating --> Applying: validation_success
    Validating --> Failed: validation_error

    Applying --> Backup: create_backups()
    Backup --> Writing: apply_changes()
    Writing --> Success: all_writes_ok
    Writing --> Rollback: write_error

    Rollback --> Failed: restore_backups()
    Success --> [*]
    Failed --> [*]

    Editing --> Idle: cancel()
    Failed --> Idle: reset()
```

**Transaction Properties**:
- **Atomic**: All edits succeed or none do
- **Consistent**: Files remain in valid state
- **Isolated**: No partial state visible
- **Durable**: Changes persist after commit

### MCP Protocol Integration

```mermaid
sequenceDiagram
    participant Claude as Claude Desktop
    participant MCP as Victor MCP Server
    participant Registry as Tool Registry
    participant Tool as Tool Implementation

    Claude->>MCP: initialize()
    MCP-->>Claude: {capabilities, version}

    Claude->>MCP: list_tools()
    MCP->>Registry: list_tools()
    Registry-->>MCP: [tools...]
    MCP-->>Claude: {tools: [...]}

    Claude->>MCP: call_tool(name, params)
    MCP->>Registry: get(name)
    Registry-->>MCP: tool
    MCP->>Tool: execute(**params)
    Tool-->>MCP: ToolResult
    MCP-->>Claude: {result: ...}
```

## Module Deep Dives

### Provider Module (`victor/providers/`)

**Purpose**: Abstract LLM provider differences

```python
# Base Provider Interface
class BaseProvider:
    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> ChatResponse:
        """Generate completion from messages."""
        raise NotImplementedError

    async def generate_stream(
        self,
        messages: List[Message],
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> AsyncIterator[ChatResponse]:
        """Stream completion chunks."""
        raise NotImplementedError
```

**Provider Lifecycle**:

```mermaid
flowchart LR
    Init[Initialize] --> Config[Load Config]
    Config --> Client[Create Client]
    Client --> Ready[Ready]
    Ready --> Generate[Generate]
    Generate --> Parse[Parse Response]
    Parse --> Normalize[Normalize]
    Normalize --> Return[Return Response]
    Return --> Ready
    Ready --> Close[Close]
    Close --> End([End])
```

### Tool Module (`victor/tools/`)

**Tool Execution Pipeline**:

```mermaid
flowchart TD
    Receive[Receive Tool Call] --> Validate{Validate<br/>Parameters}
    Validate -->|Invalid| Error[Return Error]
    Validate -->|Valid| Safety{Safety<br/>Checks}

    Safety -->|Blocked| Deny[Deny Execution]
    Safety -->|Allowed| Execute[Execute Operation]

    Execute --> Success{Success?}
    Success -->|Yes| Format[Format Output]
    Success -->|No| CatchError[Catch Error]

    Format --> Result[Return ToolResult]
    CatchError --> Result
    Deny --> Result
    Error --> Result

    Result --> Log[Log Execution]
    Log --> End([End])
```

**Database Tool Architecture**:

```mermaid
graph TB
    subgraph "DatabaseTool"
        Connect[Connect Operation]
        Query[Query Operation]
        Schema[Schema Operations]
        Disconnect[Disconnect]
    end

    subgraph "Connection Pool"
        SQLite[SQLite Connections]
        Postgres[PostgreSQL Connections]
        MySQL[MySQL Connections]
    end

    subgraph "Safety Layer"
        Validate[SQL Validation]
        Dangerous[Dangerous Pattern Detection]
        ReadOnly[Read-Only Mode]
    end

    Connect --> SQLite
    Connect --> Postgres
    Connect --> MySQL

    Query --> Validate
    Validate --> Dangerous
    Dangerous --> ReadOnly
    ReadOnly --> Execution[Execute Query]

    Schema --> Inspection[Schema Inspection]
```

### Context Management (`victor/context/`)

**Context Window Strategy**:

```mermaid
flowchart TD
    Start([New Message]) --> Count{Token<br/>Count}
    Count -->|Under Limit| Add[Add to Context]
    Count -->|Over Limit| Strategy{Pruning<br/>Strategy}

    Strategy -->|Sliding| Remove[Remove Oldest]
    Strategy -->|Semantic| Compress[Semantic Compression]
    Strategy -->|Summary| Summarize[Summarize Old Messages]

    Remove --> Add
    Compress --> Add
    Summarize --> Add

    Add --> Index[Update Search Index]
    Index --> Save[Persist to Disk]
    Save --> End([Ready])
```

## Adding New Features

### Adding a New Provider

```mermaid
flowchart TD
    Start([New Provider]) --> Inherit[Inherit BaseProvider]
    Inherit --> Implement[Implement Methods]

    Implement --> Gen[generate()]
    Implement --> Stream[generate_stream()]
    Implement --> Norm[_normalize_response()]

    Gen --> Client[Create HTTP Client]
    Stream --> Client
    Norm --> Format[Format to ChatResponse]

    Client --> Test[Write Unit Tests]
    Format --> Test

    Test --> Register[Register in Registry]
    Register --> Config[Add to Config Schema]
    Config --> Docs[Update Documentation]
    Docs --> Example[Create Example]
    Example --> End([Complete])

    style Start fill:#9f9
    style End fill:#9f9
```

**Step-by-Step**:

1. **Create provider file**: `victor/providers/my_provider.py`

```python
from typing import List, Optional, AsyncIterator
from victor.providers.base import BaseProvider, ChatResponse, Message
import httpx

class MyProvider(BaseProvider):
    def __init__(self, api_key: str, base_url: str):
        super().__init__()
        self.api_key = api_key
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"}
        )

    async def generate(
        self,
        messages: List[Message],
        **kwargs
    ) -> ChatResponse:
        # Implementation
        response = await self.client.post(
            "/chat",
            json={"messages": messages, **kwargs}
        )
        return self._normalize_response(response.json())

    def _normalize_response(self, response: dict) -> ChatResponse:
        # Convert provider format to ChatResponse
        return ChatResponse(
            content=response["output"],
            role="assistant",
            model=response["model"]
        )
```

2. **Register provider**: Update `victor/providers/registry.py`

3. **Add tests**: `tests/unit/providers/test_my_provider.py`

4. **Update config**: Add to `profiles.yaml.example`

5. **Document**: Add to README and PROVIDERS.md

### Adding a New Tool

```mermaid
flowchart TD
    Start([New Tool Idea]) --> Design[Design Interface]
    Design --> Params[Define Parameters]
    Params --> Impl[Implement execute()]

    Impl --> Safety{Needs<br/>Safety?}
    Safety -->|Yes| Checks[Add Safety Checks]
    Safety -->|No| Test

    Checks --> Test[Write Tests]
    Test --> Register[Register in Registry]
    Register --> MCP[Verify MCP Exposure]
    MCP --> Docs[Write Documentation]
    Docs --> Example[Create Example]
    Example --> End([Complete])

    style Start fill:#9f9
    style End fill:#9f9
```

**Example: Custom Tool**:

```python
from typing import Any, List
from victor.tools.base import BaseTool, ToolParameter, ToolResult

class MyCustomTool(BaseTool):
    """Custom tool for specific operations."""

    @property
    def name(self) -> str:
        return "my_custom_tool"

    @property
    def description(self) -> str:
        return """Perform custom operations.

        Operations:
        - operation1: Does X
        - operation2: Does Y

        Example:
        my_custom_tool(operation="operation1", param="value")
        """

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="operation",
                type="string",
                description="Operation to perform",
                required=True
            ),
            ToolParameter(
                name="param",
                type="string",
                description="Operation parameter",
                required=False
            )
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        operation = kwargs.get("operation")

        if operation == "operation1":
            result = self._operation1(kwargs)
            return ToolResult(
                success=True,
                output=result,
                error=""
            )

        return ToolResult(
            success=False,
            output="",
            error=f"Unknown operation: {operation}"
        )
```

## Development Workflow

### Feature Development Lifecycle

```mermaid
gitGraph
    commit id: "main"
    branch feature/new-tool
    checkout feature/new-tool
    commit id: "implement tool"
    commit id: "add tests"
    commit id: "update docs"
    checkout main
    merge feature/new-tool tag: "v0.2.0"
    branch fix/bug-123
    checkout fix/bug-123
    commit id: "fix bug"
    commit id: "add test"
    checkout main
    merge fix/bug-123 tag: "v0.2.1"
```

### Code Review Checklist

```mermaid
flowchart TD
    PR[Pull Request] --> Tests{Tests<br/>Pass?}
    Tests -->|No| FixTests[Fix Tests]
    FixTests --> Tests

    Tests -->|Yes| Coverage{Coverage<br/>OK?}
    Coverage -->|No| AddTests[Add Tests]
    AddTests --> Coverage

    Coverage -->|Yes| Lint{Linting<br/>Pass?}
    Lint -->|No| FixLint[Fix Lint]
    FixLint --> Lint

    Lint -->|Yes| Types{Types<br/>OK?}
    Types -->|No| FixTypes[Fix Types]
    FixTypes --> Types

    Types -->|Yes| Docs{Docs<br/>Updated?}
    Docs -->|No| UpdateDocs[Update Docs]
    UpdateDocs --> Docs

    Docs -->|Yes| Review[Manual Review]
    Review --> Approve{Approved?}
    Approve -->|No| Changes[Request Changes]
    Changes --> PR

    Approve -->|Yes| Merge[Merge]
    Merge --> End([Complete])

    style Merge fill:#9f9
```

## Best Practices

### 1. Error Handling

```python
# Good: Specific error handling
async def execute(self, **kwargs):
    try:
        result = await self._perform_operation(kwargs)
        return ToolResult(success=True, output=result, error="")
    except ValidationError as e:
        logger.error("Validation failed: %s", e)
        return ToolResult(success=False, output="", error=f"Validation: {e}")
    except ConnectionError as e:
        logger.error("Connection failed: %s", e)
        return ToolResult(success=False, output="", error=f"Connection: {e}")
    except Exception as e:
        logger.exception("Unexpected error")
        return ToolResult(success=False, output="", error=str(e))
```

### 2. Async Best Practices

```python
# Good: Use async context managers
async with self.client as client:
    response = await client.post(url, json=data)

# Good: Batch async operations
results = await asyncio.gather(
    self.operation1(),
    self.operation2(),
    self.operation3()
)

# Good: Stream processing
async for chunk in provider.generate_stream(messages):
    yield chunk
```

### 3. Type Hints

```python
# Good: Complete type hints
from typing import List, Optional, Dict, Any

async def process(
    self,
    items: List[str],
    options: Optional[Dict[str, Any]] = None
) -> List[Dict[str, str]]:
    """Process items with options."""
    results: List[Dict[str, str]] = []
    for item in items:
        result = await self._process_item(item, options or {})
        results.append(result)
    return results
```

### 4. Logging

```python
import logging

logger = logging.getLogger(__name__)

# Good: Structured logging with context
logger.info("Processing request", extra={
    "operation": operation,
    "user_id": user_id,
    "request_id": request_id
})

# Good: Log levels
logger.debug("Detailed debugging info")
logger.info("General information")
logger.warning("Warning condition")
logger.error("Error occurred", exc_info=True)
```

### 5. Testing

```python
# Good: Comprehensive test coverage
class TestMyTool:
    @pytest.fixture
    def tool(self):
        return MyTool(config={"key": "value"})

    async def test_success_case(self, tool):
        """Test successful execution."""
        result = await tool.execute(operation="test")
        assert result.success
        assert "expected" in result.output

    async def test_error_handling(self, tool):
        """Test error handling."""
        result = await tool.execute(operation="invalid")
        assert not result.success
        assert "error" in result.error.lower()

    async def test_edge_cases(self, tool):
        """Test edge cases."""
        # Empty input
        result = await tool.execute()
        assert not result.success

        # Large input
        result = await tool.execute(data="x" * 10000)
        assert result.success
```

---

**Next**: See [USER_GUIDE.md](USER_GUIDE.md) for end-user documentation

**Related**:
- [TESTING_STRATEGY.md](TESTING_STRATEGY.md) - Testing approach
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Contribution guidelines
- [ARCHITECTURE.md](ARCHITECTURE.md) - Detailed architecture

*Last Updated: 2025-11-24*
