# Protocols API Reference - Part 3

**Part 3 of 4:** Tool Selection Protocols

---

## Navigation

- [Part 1: Core & Search](part-1-core-search.md)
- [Part 2: Team & LSP](part-2-team-lsp.md)
- **[Part 3: Tool Selection](#)** (Current)
- [Part 4: Implementation & Examples](part-4-implementation-examples.md)
- [**Complete Reference**](../protocols-api.md)

---

**Location**: `victor/protocols/lsp_types.py`

**Import**: `from victor.protocols import Position, Range, Diagnostic, DocumentSymbol`

Victor provides Language Server Protocol (LSP) standard types for cross-vertical document operations.

### Position

Represents a cursor position in a document (0-indexed).

```python
@dataclass
class Position:
    line: int       # Line position (0-indexed)
    character: int  # Character offset in line (0-indexed)

    def to_dict(self) -> Dict[str, int]: ...

    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> "Position": ...
```

### Range

Represents a text span from start to end position.

```python
@dataclass
class Range:
    start: Position  # Start position (inclusive)
    end: Position    # End position (exclusive)

    def contains(self, position: Position) -> bool: ...
    def overlaps(self, other: "Range") -> bool: ...

    @property
    def is_empty(self) -> bool: ...
```

### Diagnostic

Represents a problem or suggestion in a document.

```python
@dataclass
class Diagnostic:
    range: Range
    message: str
    severity: DiagnosticSeverity = DiagnosticSeverity.ERROR
    source: Optional[str] = None
    code: Optional[Union[str, int]] = None
    tags: List[DiagnosticTag] = field(default_factory=list)
    related_information: List[DiagnosticRelatedInformation] = field(default_factory=list)

    @property
    def is_error(self) -> bool: ...

    @property
    def is_warning(self) -> bool: ...
```

### DocumentSymbol

Hierarchical representation of symbols (classes, functions, etc.).

```python
@dataclass
class DocumentSymbol:
    name: str
    kind: SymbolKind
    range: Range
    selection_range: Range
    detail: Optional[str] = None
    children: List["DocumentSymbol"] = field(default_factory=list)
    deprecated: bool = False
```

### Enumerations

```python
class DiagnosticSeverity(IntEnum):
    ERROR = 1
    WARNING = 2
    INFORMATION = 3
    HINT = 4

class SymbolKind(IntEnum):
    FILE = 1
    MODULE = 2
    CLASS = 5
    METHOD = 6
    FUNCTION = 12
    VARIABLE = 13
    # ... and more
```

### Usage in Code Analysis

```python
from victor.protocols import Position, Range, Diagnostic, DiagnosticSeverity

# Create a diagnostic for an undefined variable
diagnostic = Diagnostic(
    range=Range(
        start=Position(line=10, character=4),
        end=Position(line=10, character=15),
    ),
    message="Variable 'undefined_var' is not defined",
    severity=DiagnosticSeverity.ERROR,
    source="pylint",
    code="E0602",
)

# Check severity
if diagnostic.is_error:
    print(f"Error at line {diagnostic.range.start.line + 1}")
```

---

## Tool Selection Protocols

**Location**: `victor/protocols/tool_selector.py`

**Import**: `from victor.protocols import IToolSelector, ToolSelectionResult`

### IToolSelector

```python
@runtime_checkable
class IToolSelector(Protocol):
    """Protocol for tool selection implementations."""

    def select_tools(
        self,
        task: str,
        *,
        limit: int = 10,
        min_score: float = 0.0,
        context: Optional[ToolSelectionContext] = None,
    ) -> ToolSelectionResult:
        """Select relevant tools for a task.

        Args:
            task: Task description or query
            limit: Maximum number of tools to return
            min_score: Minimum relevance score threshold
            context: Optional additional context

        Returns:
            ToolSelectionResult with ranked tool names and scores
        """
        ...

    def get_tool_score(
        self,
        tool_name: str,
        task: str,
        *,
        context: Optional[ToolSelectionContext] = None,
    ) -> float:
        """Get relevance score for a specific tool."""
        ...

    @property
    def strategy(self) -> ToolSelectionStrategy:
        """Get the selection strategy used."""
        ...
```

### Supporting Types

```python
class ToolSelectionStrategy(Enum):
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"

@dataclass
class ToolSelectionResult:
    tool_names: List[str]
    scores: Dict[str, float]
    strategy_used: ToolSelectionStrategy
    metadata: Dict[str, Any]

    @property
    def top_tool(self) -> Optional[str]: ...

    def filter_by_score(self, min_score: float) -> "ToolSelectionResult": ...

@dataclass
class ToolSelectionContext:
    task_description: str
    conversation_stage: Optional[str] = None
    previous_tools: List[str] = field(default_factory=list)
    failed_tools: Set[str] = field(default_factory=set)
    model_name: str = ""
    provider_name: str = ""
```

---

