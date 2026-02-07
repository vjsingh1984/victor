# Protocols API Reference - Part 1

**Part 1 of 4:** Core and Search Protocols

---

## Navigation

- **[Part 1: Core & Search](#)** (Current)
- [Part 2: Team & LSP](part-2-team-lsp.md)
- [Part 3: Tool Selection](part-3-tool-selection.md)
- [Part 4: Implementation & Examples](part-4-implementation-examples.md)
- [**Complete Reference**](../protocols-api.md)

---
# Victor Protocols API Reference

This document provides comprehensive API reference documentation for Victor's protocol interfaces. Protocols in Victor follow Python's `Protocol` pattern (PEP 544) to enable structural subtyping, dependency inversion, and interface segregation.

## Table of Contents

- [Protocol Overview](#protocol-overview)
- [Core Protocols](#core-protocols)
  - [IProviderAdapter](#iprovideradapter)
  - [IGroundingStrategy](#igroundingstrategy)
  - [IQualityAssessor](#iqualityassessor)
  - [IModeController](#imodecontroller)
  - [IPathResolver](#ipathresolver)
- [Search Protocols](#search-protocols)
  - [ISemanticSearch](#isemanticsearch)
  - [IIndexable](#iindexable)
- [Team Protocols](#team-protocols)
  - [IAgent](#iagent)
  - [ITeamMember](#iteammember)
  - [ITeamCoordinator](#iteamcoordinator)
- [LSP Types](#lsp-types)
- [Tool Selection Protocols](#tool-selection-protocols)
- [Implementation Examples](#implementation-examples)

---

## Protocol Overview

### What are Protocols in Victor?

Victor uses Python's `Protocol` pattern (from `typing`) to define interfaces without requiring inheritance. This approach, known as structural subtyping or "duck typing with type hints," allows any class that implements the required methods to satisfy a protocol, regardless of its inheritance hierarchy.

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class IExample(Protocol):
    """Protocol interface definition."""

    def required_method(self, arg: str) -> int:
        """Method that implementations must provide."""
        ...
```

### ISP Compliance Design

Victor's protocols follow the Interface Segregation Principle (ISP):

1. **Small, Focused Interfaces**: Each protocol defines the minimum interface needed for a specific capability
2. **Composable Protocols**: Complex behaviors are achieved by implementing multiple simple protocols
3. **No Fat Interfaces**: Clients depend only on the methods they use

Example of protocol composition:
```python
# Small, focused protocols
class ITeamCoordinator(Protocol): ...
class IObservableCoordinator(Protocol): ...
class IRLCoordinator(Protocol): ...

# Composed protocol for full capabilities
class IEnhancedTeamCoordinator(
    ITeamCoordinator,
    IObservableCoordinator,
    IRLCoordinator,
    Protocol
): ...
```

### How to Implement Protocols

To implement a Victor protocol:

1. **Implicit Implementation**: Simply implement all required methods with matching signatures
2. **No Inheritance Required**: Your class doesn't need to explicitly inherit from the protocol
3. **Runtime Checking**: Use `isinstance()` with `@runtime_checkable` protocols

```python
from victor.protocols import IProviderAdapter, ProviderCapabilities

class MyProviderAdapter:
    """Custom provider adapter - implicitly implements IProviderAdapter."""

    @property
    def name(self) -> str:
        return "my_provider"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(quality_threshold=0.75)

    def detect_continuation_needed(self, response: str) -> bool:
        return not response.strip()

    # ... implement other required methods
```

---

## Core Protocols

### IProviderAdapter

**Location**: `victor/protocols/provider_adapter.py`

**Import**: `from victor.protocols import IProviderAdapter`

The `IProviderAdapter` protocol defines the interface for adapting provider-specific behaviors. Each LLM provider (OpenAI, Anthropic, DeepSeek, etc.) has different response formats, tool calling conventions, and quality thresholds.

#### Protocol Definition

```python
@runtime_checkable
class IProviderAdapter(Protocol):
    """Interface for provider-specific behavior adaptation."""

    @property
    def name(self) -> str:
        """Return the provider name."""
        ...

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Return provider capabilities configuration."""
        ...

    def detect_continuation_needed(self, response: str) -> bool:
        """Detect if response indicates continuation is needed.

        Args:
            response: The LLM response text

        Returns:
            True if the response appears incomplete
        """
        ...

    def extract_thinking_content(self, response: str) -> Tuple[str, str]:
        """Extract thinking tags and content separately.

        Args:
            response: The LLM response text

        Returns:
            Tuple of (thinking_content, main_content)
        """
        ...

    def normalize_tool_calls(self, raw_calls: List[Any]) -> List[ToolCall]:
        """Normalize tool calls to standard format.

        Args:
            raw_calls: Provider-specific tool call data

        Returns:
            List of normalized ToolCall objects
        """
        ...

    def should_retry(self, error: Exception) -> Tuple[bool, float]:
        """Determine if error is retryable and backoff time.

        Args:
            error: The exception that occurred

        Returns:
            Tuple of (is_retryable, backoff_seconds)
        """
        ...
```

#### Supporting Types

```python
class ToolCallFormat(Enum):
    """Tool call format variants across providers."""
    OPENAI = "openai"      # Standard OpenAI format
    ANTHROPIC = "anthropic" # Anthropic's content blocks
    NATIVE = "native"       # Provider's native format
    FALLBACK = "fallback"   # Text-based parsing fallback

@dataclass
class ProviderCapabilities:
    """Provider-specific capabilities and thresholds."""
    quality_threshold: float = 0.80
    supports_thinking_tags: bool = False
    thinking_tag_format: str = ""
    continuation_markers: List[str] = field(default_factory=list)
    max_continuation_attempts: int = 5
    tool_call_format: ToolCallFormat = ToolCallFormat.OPENAI
    output_deduplication: bool = False
    streaming_chunk_size: int = 1024
    supports_parallel_tools: bool = True
    grounding_required: bool = True
    grounding_strictness: float = 0.8
```

#### Usage

```python
from victor.protocols import get_provider_adapter

# Get adapter for a specific provider
adapter = get_provider_adapter("deepseek")

# Check capabilities
if adapter.capabilities.supports_thinking_tags:
    thinking, content = adapter.extract_thinking_content(response)

# Detect if continuation needed
if adapter.detect_continuation_needed(response):
    # Request continuation from LLM
    pass
```

---

### IGroundingStrategy

**Location**: `victor/protocols/grounding.py`

**Import**: `from victor.protocols import IGroundingStrategy`

The `IGroundingStrategy` protocol defines the interface for verifying claims in LLM responses against verifiable sources (files, symbols, content).

#### Protocol Definition

```python
@runtime_checkable
class IGroundingStrategy(Protocol):
    """Strategy interface for grounding verification."""

    @property
    def name(self) -> str:
        """Return strategy name."""
        ...

    @property
    def claim_types(self) -> List[GroundingClaimType]:
        """Return claim types this strategy can verify."""
        ...

    async def verify(
        self,
        claim: GroundingClaim,
        context: Dict[str, Any],
    ) -> VerificationResult:
        """Verify a claim against context.

        Args:
            claim: The claim to verify
            context: Additional context for verification

        Returns:
            Verification result with grounding status
        """
        ...

    def extract_claims(
        self,
        response: str,
        context: Dict[str, Any],
    ) -> List[GroundingClaim]:
        """Extract claims of this type from a response.

        Args:
            response: The response text to analyze
            context: Additional context

        Returns:
            List of claims found in the response
        """
        ...
```

#### Supporting Types

```python
class GroundingClaimType(str, Enum):
    """Types of claims that can be grounded."""
    FILE_EXISTS = "file_exists"
    FILE_NOT_EXISTS = "file_not_exists"
    SYMBOL_EXISTS = "symbol_exists"
    CONTENT_MATCH = "content_match"
    LINE_NUMBER = "line_number"
    DIRECTORY_EXISTS = "directory_exists"

@dataclass
class GroundingClaim:
    """A claim extracted from a response."""
    claim_type: GroundingClaimType
    value: str
    context: Dict[str, Any] = field(default_factory=dict)
    source_text: str = ""
    confidence: float = 1.0

@dataclass
class VerificationResult:
    """Result of verifying a single claim."""
    is_grounded: bool
    confidence: float = 0.0
    claim: Optional[GroundingClaim] = None
    evidence: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
```

#### Built-in Strategies

- **FileExistenceStrategy**: Verifies file path references
- **SymbolReferenceStrategy**: Verifies code symbol references
- **ContentMatchStrategy**: Verifies quoted content matches source

#### Usage

```python
from victor.protocols import (
    CompositeGroundingVerifier,
    FileExistenceStrategy,
    SymbolReferenceStrategy,
)

# Create composite verifier
verifier = CompositeGroundingVerifier([
    FileExistenceStrategy(project_root),
    SymbolReferenceStrategy(symbol_table),
])

# Verify response
result = await verifier.verify(response, context)
if result.is_grounded:
    print(f"Verified {result.verified_claims}/{result.total_claims} claims")
```

---

### IQualityAssessor

**Location**: `victor/protocols/quality.py`

**Import**: `from victor.protocols import IQualityAssessor`

The `IQualityAssessor` protocol defines the interface for assessing response quality across multiple dimensions.

#### Protocol Definition

```python
@runtime_checkable
class IQualityAssessor(Protocol):
    """Interface for quality assessment."""

    def assess(
        self,
        response: str,
        context: Dict[str, Any],
    ) -> QualityScore:
        """Assess response quality.

        Args:
            response: The response text to assess
            context: Additional context (query, provider, etc.)

        Returns:
            Quality score with dimension breakdown
        """
        ...

    @property
    def dimensions(self) -> List[ProtocolQualityDimension]:
        """Return dimensions this assessor evaluates."""
        ...
```

#### Supporting Types

```python
class ProtocolQualityDimension(str, Enum):
    """Quality dimensions for response assessment."""
    GROUNDING = "grounding"      # Factual accuracy
    COVERAGE = "coverage"        # Query coverage
    CLARITY = "clarity"          # Response clarity
    CORRECTNESS = "correctness"  # Code correctness
    CONCISENESS = "conciseness"  # Appropriate brevity
    HELPFULNESS = "helpfulness"  # Task helpfulness
    SAFETY = "safety"            # Safety considerations

@dataclass
class DimensionScore:
    """Score for a single quality dimension."""
    dimension: ProtocolQualityDimension
    score: float  # 0.0-1.0
    weight: float = 1.0
    reason: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QualityScore:
    """Overall quality assessment result."""
    score: float  # 0.0-1.0
    is_acceptable: bool
    threshold: float = 0.80
    provider: str = ""
    dimension_scores: Dict[ProtocolQualityDimension, DimensionScore]
    feedback: str = ""
    suggestions: List[str] = field(default_factory=list)
```

#### Built-in Assessors

- **SimpleQualityAssessor**: Heuristic-based assessment
- **ProviderAwareQualityAssessor**: Provider-specific adjustments
- **CompositeQualityAssessor**: Combines multiple assessors

#### Usage

```python
from victor.protocols import ProviderAwareQualityAssessor

assessor = ProviderAwareQualityAssessor(
    provider_name="anthropic",
    provider_threshold=0.85,
)

score = assessor.assess(response, {"query": user_query})
if score.is_acceptable:
    print(f"Quality: {score.score:.2%}")
else:
    print(f"Quality below threshold: {score.feedback}")
```

---

### IModeController

**Location**: `victor/protocols/mode_aware.py`

**Import**: `from victor.protocols import IModeController`

The `IModeController` protocol defines the interface for mode management (BUILD/PLAN/EXPLORE modes).

#### Protocol Definition

```python
@runtime_checkable
class IModeController(Protocol):
    """Protocol for mode controller access."""

    @property
    def current_mode(self) -> Any:
        """Get the current agent mode."""
        ...

    @property
    def config(self) -> Any:
        """Get the current mode configuration."""
        ...

    def is_tool_allowed(self, tool_name: str) -> bool:
        """Check if a tool is allowed in the current mode."""
        ...

    def get_tool_priority(self, tool_name: str) -> float:
        """Get priority adjustment for a tool in current mode."""
        ...
```

#### Supporting Types

```python
@dataclass
class ModeInfo:
    """Information about the current mode."""
    name: str = "BUILD"  # BUILD, PLAN, or EXPLORE
    allow_all_tools: bool = True
    exploration_multiplier: float = 1.0
    sandbox_dir: Optional[str] = None
    allowed_tools: Set[str] = None
    disallowed_tools: Set[str] = None
```

#### ModeAwareMixin

The `ModeAwareMixin` provides convenient mode-aware functionality:

```python
from victor.protocols import ModeAwareMixin

class MyComponent(ModeAwareMixin):
    def process(self) -> None:
        if self.is_build_mode:
            # Full capabilities
            pass
        elif self.is_plan_mode:
            # Read-only with sandbox
            pass
        elif self.is_explore_mode:
            # Read-only exploration
            pass
```

---

### IPathResolver

**Location**: `victor/protocols/path_resolver.py`

**Import**: `from victor.protocols import IPathResolver`

The `IPathResolver` protocol defines the interface for centralized path resolution and normalization.

#### Protocol Definition

```python
@runtime_checkable
class IPathResolver(Protocol):
    """Protocol for path resolution."""

    def resolve(self, path: str, must_exist: bool = True) -> PathResolution:
        """Resolve a path with normalization.

        Args:
            path: Path to resolve (relative or absolute)
            must_exist: If True, raises error if path doesn't exist

        Returns:
            PathResolution with resolved path and metadata
        """
        ...

    def resolve_file(self, path: str) -> PathResolution:
        """Resolve a file path.

        Args:
            path: File path to resolve

        Returns:
            PathResolution, raises if not a file
        """
        ...

    def resolve_directory(self, path: str) -> PathResolution:
        """Resolve a directory path.

        Args:
            path: Directory path to resolve

        Returns:
            PathResolution, raises if not a directory
        """
        ...

    def suggest_similar(self, path: str, limit: int = 5) -> List[str]:
        """Suggest similar paths that exist.

        Args:
            path: Non-existent path to find matches for
            limit: Maximum suggestions to return

        Returns:
            List of similar existing paths
        """
        ...
```

#### Supporting Types

```python
@dataclass
class PathResolution:
    """Result of path resolution."""
    original_path: str
    resolved_path: Path
    was_normalized: bool = False
    normalization_applied: Optional[str] = None
    exists: bool = False
    is_file: bool = False
    is_directory: bool = False
```

#### Usage

```python
from victor.protocols import create_path_resolver

resolver = create_path_resolver()

# Resolve with automatic normalization
result = resolver.resolve_file("project/utils/helper.py")
if result.was_normalized:
    print(f"Normalized: {result.original_path} -> {result.resolved_path}")

# Get suggestions for typos
suggestions = resolver.suggest_similar("modls/news.py")
```

---

## Search Protocols
