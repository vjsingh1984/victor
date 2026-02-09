# Protocols API Reference - Part 2

**Part 2 of 4:** Team and LSP Protocols

---

## Navigation

- [Part 1: Core & Search](part-1-core-search.md)
- **[Part 2: Team & LSP](#)** (Current)
- [Part 3: Tool Selection](part-3-tool-selection.md)
- [Part 4: Implementation & Examples](part-4-implementation-examples.md)
- [**Complete Reference**](../protocols-api.md)

---

### ISemanticSearch

**Location**: `victor/protocols/search.py`

**Import**: `from victor.protocols import ISemanticSearch`

The `ISemanticSearch` protocol defines the interface for semantic search implementations across all verticals.

#### Protocol Definition

```python
@runtime_checkable
class ISemanticSearch(Protocol):
    """Protocol for semantic search implementations."""

    @property
    def is_indexed(self) -> bool:
        """Whether the search provider has indexed content."""
        ...

    async def search(
        self,
        query: str,
        max_results: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchHit]:
        """Execute semantic search on indexed content.

        Args:
            query: Natural language search query
            max_results: Maximum number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of SearchHit objects ordered by relevance
        """
        ...
```text

#### Usage

```python
from victor.protocols import ISemanticSearch

async def find_relevant_code(searcher: ISemanticSearch, query: str):
    if not searcher.is_indexed:
        return []

    results = await searcher.search(
        query="authentication error handling",
        max_results=5,
        filter_metadata={"file_type": "py"}
    )

    for hit in results:
        print(f"{hit.file_path}:{hit.line_number} - {hit.score:.2f}")
```

---

### IIndexable

**Location**: `victor/protocols/search.py`

**Import**: `from victor.protocols import IIndexable`

The `IIndexable` protocol defines the interface for content that can be indexed for semantic search.

#### Protocol Definition

```python
@runtime_checkable
class IIndexable(Protocol):
    """Protocol for indexable content sources."""

    async def index_document(
        self,
        file_path: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Index a document for semantic search.

        Args:
            file_path: Path or identifier for the document
            content: Text content to index
            metadata: Optional metadata to store
        """
        ...

    async def remove_document(self, file_path: str) -> bool:
        """Remove a document from the index.

        Args:
            file_path: Path or identifier for the document

        Returns:
            True if document was removed
        """
        ...

    async def clear_index(self) -> None:
        """Clear all indexed content."""
        ...
```text

#### Combined Protocol

```python
@runtime_checkable
class ISemanticSearchWithIndexing(ISemanticSearch, IIndexable, Protocol):
    """Combined protocol for searchable and indexable implementations."""
    pass
```

---

## Team Protocols

### IAgent

**Location**: `victor/protocols/team.py`

**Import**: `from victor.protocols import IAgent`

The `IAgent` protocol is the unified base protocol for all agent implementations in Victor.

#### Protocol Definition

```python
@runtime_checkable
class IAgent(Protocol):
    """Unified protocol for all agent implementations."""

    @property
    def id(self) -> str:
        """Unique identifier for this agent."""
        ...

    @property
    def role(self) -> Any:
        """Role of this agent."""
        ...

    async def execute_task(self, task: str, context: Dict[str, Any]) -> Any:
        """Execute a task and return the result.

        Args:
            task: Task description
            context: Execution context with shared state

        Returns:
            Result of task execution
        """
        ...
```text

---

### ITeamMember

**Location**: `victor/protocols/team.py`

**Import**: `from victor.protocols import ITeamMember`

The `ITeamMember` protocol extends `IAgent` with team coordination capabilities.

#### Protocol Definition

```python
@runtime_checkable
class ITeamMember(IAgent, Protocol):
    """Protocol for team members."""

    @property
    def persona(self) -> Optional[Any]:
        """Persona of this member."""
        ...

    async def execute_task(self, task: str, context: Dict[str, Any]) -> Any:
        """Execute a task and return the result."""
        ...

    async def receive_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Receive and optionally respond to a message.

        Args:
            message: Incoming message

        Returns:
            Optional response message
        """
        ...
```

---

### ITeamCoordinator

**Location**: `victor/protocols/team.py`

**Import**: `from victor.protocols import ITeamCoordinator`

The `ITeamCoordinator` protocol defines the base interface for team coordinators.

#### Protocol Definition

```python
@runtime_checkable
class ITeamCoordinator(Protocol):
    """Base protocol for team coordinators."""

    def add_member(self, member: ITeamMember) -> "ITeamCoordinator":
        """Add a member to the team.

        Args:
            member: Team member to add

        Returns:
            Self for fluent chaining
        """
        ...

    def set_formation(self, formation: TeamFormation) -> "ITeamCoordinator":
        """Set the team formation pattern.

        Args:
            formation: Formation to use

        Returns:
            Self for fluent chaining
        """
        ...

    async def execute_task(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task with the team.

        Args:
            task: Task description
            context: Execution context

        Returns:
            Result dictionary with success, member_results, final_output
        """
        ...

    async def broadcast(self, message: AgentMessage) -> List[Optional[AgentMessage]]:
        """Broadcast a message to all team members.

        Args:
            message: Message to broadcast

        Returns:
            List of responses from members
        """
        ...
```text

#### Extended Protocols

Victor provides several extended coordinator protocols for additional capabilities:

```python
# Observability integration
class IObservableCoordinator(Protocol):
    def set_execution_context(self, task_type, complexity, vertical, trigger): ...
    def set_progress_callback(self, callback): ...

# Reinforcement learning integration
class IRLCoordinator(Protocol):
    def set_rl_coordinator(self, rl_coordinator): ...

# Message bus provider
class IMessageBusProvider(Protocol):
    @property
    def message_bus(self) -> Any: ...

# Shared memory provider
class ISharedMemoryProvider(Protocol):
    @property
    def shared_memory(self) -> Any: ...

# Combined enhanced coordinator
class IEnhancedTeamCoordinator(
    ITeamCoordinator,
    IObservableCoordinator,
    IRLCoordinator,
    IMessageBusProvider,
    ISharedMemoryProvider,
    Protocol,
): ...
```


**Reading Time:** 3 min
**Last Updated:** February 08, 2026**

---

## See Also

- [Documentation Home](../../README.md)


## LSP Types
