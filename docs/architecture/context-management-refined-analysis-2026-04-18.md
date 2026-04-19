# Victor Context Management - Refined Analysis

**Date**: 2026-04-18
**Status**: Actual Code Examination Complete
**Scope**: Comprehensive analysis of Victor's existing context management capabilities vs. perceived gaps

---

## Executive Summary

After comprehensive examination of Victor's actual codebase, the initial enhancement plan was based on **assumptions rather than reality**. Victor already has **sophisticated context management** with most proposed features already implemented:

**Already Implemented** ✅:
- Vector-based retrieval via LanceDB with lazy embedding
- Tree-sitter AST-based code parsing for entity extraction
- Multiple compaction strategies (SIMPLE, TIERED, SEMANTIC, HYBRID)
- Hierarchical compaction with epoch-level compression
- Topic extraction via heuristics (CamelCase, snake_case, keywords)
- File modification tracking (in task_completion and compaction_summarizer)
- Semantic message scoring with configurable weights
- Rust-accelerated batch scoring (4.8-9.9x speedup)
- Focus phase detection (DACS-inspired) for exploration/mutation/execution
- Semantic deduplication (DCE-inspired) for near-duplicate removal
- Session ledger integration for context preservation
- Context reminder framework for compaction injection

**Actual Gaps** ❌:
- No topic-aware segmentation (messages in single flat list per session)
- No file timestamp-based cache invalidation (files tracked but not used for invalidation)
- Vector retrieval exists but not integrated for cross-session learning
- Components exist but are disconnected (need integration wiring)

**Real Opportunity**: Integration and enhancement of existing components, not building new ones.

---

## Part 1: Victor's Actual Capabilities (By Component)

### 1.1 Context Compaction (`victor/agent/context_compactor.py`)

**Capability**: Proactive compaction with multiple sophisticated strategies

**Key Features**:
```python
# Token-accurate estimation by content type
TOKEN_FACTORS = {
    "code": 3.0,      # Code is more token-dense
    "json": 2.8,      # JSON has structure overhead
    "prose": 4.0,     # English prose ~4 chars/token
    "mixed": 3.5,     # Default mixed content
}

# Priority levels
class MessagePriority(Enum):
    CRITICAL = 100    # System prompts, current task
    HIGH = 75         # Recent tool results, code context
    MEDIUM = 50       # Previous exchanges
    LOW = 25          # Old context, summaries
    EPHEMERAL = 0     # Can be dropped immediately

# Pinned output requirements (never compacted)
PINNED_REQUIREMENT_PATTERNS = [
    re.compile(r"\bmust\s+output\b", re.IGNORECASE),
    re.compile(r"\brequired\s+format\b", re.IGNORECASE),
    # ... 8 total patterns
]
```

**Compaction Strategies**:
- **SIMPLE**: Keep N recent messages
- **TIERED**: Prioritize tool results over discussion
- **SEMANTIC**: Use embeddings to keep task-relevant messages
- **HYBRID**: Combine tiered and semantic approaches

**Sophistication Level**: ⭐⭐⭐⭐⭐ (Very High)
- Content-aware pruning with regex patterns
- Tool result truncation strategies (HEAD, TAIL, BOTH, SMART)
- Parallel read budget calculation
- Proactive compaction before overflow (at configurable threshold)

---

### 1.2 Conversation Storage (`victor/agent/conversation/store.py`)

**Capability**: ML/RL-friendly SQLite storage with normalized schema

**Schema Design**:
```sql
-- Lookup tables for efficient aggregation
CREATE TABLE model_families (id INTEGER PRIMARY KEY, name TEXT);
CREATE TABLE model_sizes (id INTEGER PRIMARY KEY, context_window INTEGER);
CREATE TABLE context_sizes (id INTEGER PRIMARY KEY, max_tokens INTEGER);
CREATE TABLE providers (id INTEGER PRIMARY KEY, name TEXT);

-- Sessions table uses INTEGER FKs for fast GROUP BY
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    model_family INTEGER REFERENCES model_families(id),
    model_size INTEGER REFERENCES model_sizes(id),
    -- ... other FK columns
);
```

**Key Features**:
- Schema versioning (v0.2.0)
- WAL mode for performance
- Token-aware context window management
- Priority-based message pruning
- Semantic relevance scoring
- Compaction summary persistence
- Cross-session semantic retrieval

**Sophistication Level**: ⭐⭐⭐⭐⭐ (Very High)
- Normalized schema for ML/RL aggregation
- Performance optimizations (2MB cache, synchronous=NORMAL)
- Schema migrations with version tracking

---

### 1.3 Vector Storage (`victor/agent/conversation_embedding_store.py`)

**Capability**: Lazy LanceDB vector store with lean schema

**Schema Design**:
```python
# Lean schema - no content duplication
# - message_id: str (FK -> SQLite messages.id)
# - session_id: str (for filtering)
# - vector: list[float] (384-dim embedding)
# - timestamp: str (ISO format, for pruning)
```

**Key Features**:
- **Lazy embedding**: Compute on search, not on add (saves 90%+ compute)
- SQLite as source of truth (no content duplication)
- Auto-compact after batch operations
- Semantic search with similarity scores

**Design Philosophy**:
> "Lean schema: Only message_id (FK), session_id, vector, timestamp. Content lives in SQLite, not duplicated."

**Sophistication Level**: ⭐⭐⭐⭐⭐ (Very High)
- Zero-copy reads via Apache Arrow
- Disk-based storage with mmap
- Scales to billions of vectors

---

### 1.4 Tree-Sitter Integration (`victor/storage/memory/extractors/tree_sitter_extractor.py`)

**Capability**: AST-based code parsing for entity extraction

**Entity Types**:
```python
class EntityType(Enum):
    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    FILE = "file"
    VARIABLE = "variable"
```

**Key Features**:
- Extracts FUNCTION, CLASS, MODULE, FILE, VARIABLE entities
- EntityRelation support (call graphs, inheritance)
- Integration with TreeSitterExtractorProtocol
- High accuracy through AST parsing (not regex)

**Sophistication Level**: ⭐⭐⭐⭐⭐ (Very High)
- Language-agnostic AST parsing
- Accurate entity extraction (vs fragile regex)

---

### 1.5 Hierarchical Compaction (`victor/agent/compaction_hierarchy.py`)

**Capability**: Epoch-level compression to prevent linear accumulation

**Key Components**:
```python
class CompactionEpoch:
    """Groups summaries by time window for epoch-level compression."""
    start_index: int
    end_index: int
    summaries: List[str]
    timestamp: datetime

class HierarchicalCompactionManager:
    """Manages compaction summaries with hierarchical compression.

    When individual summaries exceed epoch_threshold, older summaries
    are compressed into epoch-level summaries. Only the most recent
    max_individual summaries are kept as individual entries.
    """
```

**Sophistication Level**: ⭐⭐⭐⭐⭐ (Very High)
- Prevents linear accumulation of summaries
- Keeps max_individual most recent summaries
- Compresses older summaries into epochs

---

### 1.6 Message Scoring (`victor/agent/conversation/scoring.py`)

**Capability**: Canonical configurable scorer with Rust acceleration

**Scoring Factors**:
```python
@dataclass(frozen=True)
class ScoringWeights:
    priority: float = 0.2    # Message priority
    recency: float = 0.4     # Time-based relevance
    role: float = 0.2        # Role importance
    length: float = 0.1      # Content length
    semantic: float = 0.1    # Semantic similarity

# Presets
STORE_WEIGHTS = ScoringWeights(priority=0.4, recency=0.6, role=0.0, length=0.0, semantic=0.0)
CONTROLLER_WEIGHTS = ScoringWeights(priority=0.0, recency=0.3, role=0.3, length=0.1, semantic=0.3)
DEFAULT_WEIGHTS = ScoringWeights()
```

**Rust Acceleration**:
- Fast path for pure priority+recency scoring (4.8-9.9x speedup)
- Batch scoring via `batch_score_messages()` from Rust
- Graceful fallback to Python

**Role Importance Scores**:
```python
_ROLE_SCORES = {
    "system": 0.7,      # High but not absolute (was 1.0, reduced to prevent hoarding)
    "user": 0.8,        # Higher than system to ensure user intent survives
    "assistant": 0.6,
    "tool": 0.9,        # Tool results are critical for task state
    "tool_call": 0.7,   # Internal role for assistant tool requests
}
```

**Sophistication Level**: ⭐⭐⭐⭐⭐ (Very High)
- Configurable weights for different use cases
- Rust acceleration for performance
- Semantic similarity support via embedding_fn

---

### 1.7 Context Assembly (`victor/agent/conversation/assembler.py`)

**Capability**: Turn-boundary context assembly with advanced optimizations

**Key Features**:

1. **Semantic Deduplication (DCE-inspired)**:
```python
def _deduplicate_semantic(self, messages: list) -> list:
    """Remove semantically redundant messages (DCE-inspired).

    Uses content normalization + hashing for fast near-duplicate detection.
    Short messages (<50 chars) are never deduplicated. Keeps first occurrence.
    """
    # Normalize: strip whitespace, collapse runs, lowercase
    normalized = " ".join(content.lower().split())
    content_hash = hashlib.md5(normalized.encode()).hexdigest()
```

2. **Focus Phase Detection (DACS-inspired)**:
```python
def _detect_focus_phase(recent_messages: list) -> str:
    """Detect current activity phase from recent tool calls.

    Returns: 'exploration' | 'mutation' | 'execution' | 'mixed'
    """
    # exploration: read_file, code_search, grep, ls
    # mutation: edit, write_file, create
    # execution: bash, shell, test, git
```

3. **Focus Scoring with Predictive Pruning**:
```python
def _apply_focus_scoring(self, messages, scores, focus_phase, predicted_phase=None):
    """Adjust message scores based on current AND predicted phase.

    Boosts active-phase messages (1.5x), compresses stale-phase (0.3x).
    With prediction: double-compresses messages irrelevant to BOTH
    current and predicted phases (0.2x) — AutonAgenticAI-inspired.
    """
```

4. **Semantic Augmentation**:
```python
# Lines 296-314: Retrieve relevant compacted context
if self._conversation_controller and current_query:
    retrieve_fn = getattr(self._conversation_controller, "retrieve_relevant_history", None)
    if retrieve_fn:
        relevant = retrieve_fn(query=current_query, limit=3)
        # Inject as [Historical context: ...] messages
```

**Sophistication Level**: ⭐⭐⭐⭐⭐ (Very High)
- Paper-inspired optimizations (DCE, DACS, AutonAgenticAI)
- Session ledger integration
- Smart context selection with Rust fallback

---

### 1.8 Conversation Control (`victor/agent/conversation/controller.py`)

**Capability**: Sophisticated conversation management with multiple integrations

**Key Features**:

1. **Topic Extraction** (lines 780-807):
```python
def _extract_key_topics(self, text: str) -> List[str]:
    """Extract key topics from text using simple heuristics."""
    # CamelCase detection
    words = re.findall(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b", text)
    # snake_case detection
    words += re.findall(r"\b[a-z]+_[a-z_]+\b", text)
    # Keyword patterns
    words += re.findall(r"\b(?:def|class|function|file|error|test|api)\s+\w+", text.lower())
```

2. **Semantic Retrieval** (lines 884-937):
```python
def retrieve_relevant_history(self, query: str, limit: int = 5) -> List[str]:
    """Retrieve relevant historical context from persistent store.

    Uses semantic similarity to find relevant messages and summaries
    from the full conversation history in SQLite.
    """
    # Get semantically similar historical messages
    relevant_msgs = self._conversation_store.get_semantically_relevant_messages(
        self._session_id, query, limit=limit,
        min_similarity=self.config.semantic_relevance_threshold,
    )
    # Get relevant summaries
    relevant_summaries = self._conversation_store.get_relevant_summaries(
        self._session_id, query, limit=2,
    )
```

3. **Compaction Summary Persistence** (lines 939-958):
```python
def persist_compaction_summary(self, summary: str, message_ids: List[str]) -> None:
    """Persist a compaction summary to the SQLite store."""
    self._conversation_store.store_compaction_summary(
        self._session_id, summary, message_ids,
    )
```

4. **Smart Compaction** (lines 390-561):
- Normalizes non-Message objects before compaction
- ALWAYS keeps root system message (for caching stability)
- Preserves tool-call pairs (prevents HTTP 400 errors)
- Generates summary of removed messages
- Feeds into hierarchical manager

**Sophistication Level**: ⭐⭐⭐⭐⭐ (Very High)
- Multiple compaction strategies
- Semantic retrieval from full history
- Hierarchical compaction integration
- Context reminder framework integration

---

### 1.9 Compaction Summarizers (`victor/agent/compaction_summarizer.py`)

**Capability**: Multiple compaction summary strategies

**Strategies**:

1. **KeywordCompactionSummarizer**:
```python
"""Topic extraction with CamelCase, snake_case, keyword patterns."""
```

2. **LedgerAwareCompactionSummarizer**:
```python
"""Uses SessionLedger for structured summaries.

Tracks:
- files_read
- files_modified
- decisions
- recommendations
- pending actions
"""
```

**Sophistication Level**: ⭐⭐⭐⭐ (High)
- Multiple strategies for different use cases
- Ledger-aware with structured summaries

---

### 1.10 LLM-Based Summarization (`victor/agent/llm_compaction_summarizer.py`)

**Capability**: LLM-based abstractive summarization (vs claudecode's deterministic approach)

**Key Features**:
- Fast LLM call for rich summaries
- Fallback to other strategies on failure
- Timeout handling
- Sync/async compatibility

**Comparison with Claudecode**:
- **Claudecode**: Deterministic, rule-based, no LLM, ~23KB Rust
- **Victor**: LLM-based, multiple strategies, sophisticated, complex

**Trade-offs**:
- **Claudecode**: Simpler, faster, no API cost, deterministic
- **Victor**: Richer summaries, more expensive, non-deterministic

**Sophistication Level**: ⭐⭐⭐⭐ (High)
- LLM-based for richer summaries
- Graceful fallback to deterministic strategies

---

## Part 2: Claudecode vs Victor Compaction Comparison

### 2.1 Architecture

| Aspect | Claudecode | Victor |
|--------|-----------|--------|
| **Implementation** | Rust (~23KB) | Python (multiple modules) |
| **Approach** | Deterministic rule-based | LLM-based + multiple strategies |
| **Summarization** | Structured XML `<summary>` tags | Natural language + structured sections |
| **Idempotency** | Yes (merges old + new summaries) | Yes (incremental compaction) |
| **Threshold** | Dual: message count AND token estimate | Token budget + priority scoring |
| **Re-compaction Safety** | Excludes existing summary from decision | Separate summary layer + hierarchical compression |

### 2.2 Summary Format

**Claudecode** (Structured XML):
```xml
<summary>
  <scope>42 messages (15 user, 20 assistant, 7 tool)</scope>
  <tools_mentioned>read_file, grep, edit</tools_mentioned>
  <recent_user_requests>
    <request>Fix authentication bug in login flow</request>
    <request>Update error handling</request>
  </recent_user_requests>
  <pending_work>
    <item>Need to add retry logic for failed login attempts</item>
  </pending_work>
  <key_files_referenced>
    <file>src/auth/login.py</file>
    <file>src/auth/oauth.py</file>
  </key_files_referenced>
  <current_work>
    Currently debugging the authentication flow where users
    are getting logged out immediately after login.
  </current_work>
  <key_timeline>
    <message role="user" index="0">Fix the login bug</message>
    <message role="assistant" index="1">I'll help fix the login bug</message>
    <!-- ... more timeline entries ... -->
  </key_timeline>
</summary>
```

**Victor** (Natural Language + Structured):
```
[Earlier conversation: 15 user messages; 20 tool results; topics: authentication, login, oauth, error_handling]

---

**LedgerAwareCompactionSummarizer** (Structured):
- Files read: src/auth/login.py, src/auth/oauth.py
- Files modified: src/auth/login.py
- Decisions: Add retry logic for failed login attempts
- Recommendations: Update error handling in oauth flow
- Pending: Test login flow with retry logic
```

### 2.3 What Victor Can Learn from Claudecode

1. **Deterministic Fallback**:
   - Claudecode's rule-based approach is faster and cheaper
   - Victor has LLM-based summarization but could add deterministic fallback
   - **Opportunity**: Add `RuleBasedCompactionSummarizer` for resource-constrained environments

2. **Structured XML Format**:
   - Claudecode's XML format is more machine-readable
   - Victor's natural language summaries are richer but less structured
   - **Opportunity**: Add optional XML/JSON output format for summaries

3. **File Path Extraction**:
   - Claudecode uses regex to extract file paths from messages
   - Victor tracks files in LedgerAwareCompactionSummarizer but not in all strategies
   - **Opportunity**: Add file path extraction to all summarization strategies

4. **Re-compaction Safety**:
   - Claudecode excludes existing summary from compaction decision
   - Victor uses hierarchical compression to prevent re-compaction
   - **Both approaches are valid** - different solutions to the same problem

### 2.4 What Claudecode Can Learn from Victor

1. **Hierarchical Compression**:
   - Victor's epoch-level compression prevents linear accumulation
   - Claudecode merges summaries but doesn't have epoch-level compression
   - **Opportunity for claudecode**: Add hierarchical compression

2. **Semantic Scoring**:
   - Victor uses embeddings for semantic relevance
   - Claudecode uses only rule-based heuristics
   - **Opportunity for claudecode**: Add optional semantic scoring

3. **Multiple Strategies**:
   - Victor has SIMPLE, TIERED, SEMANTIC, HYBRID strategies
   - Claudecode has only one deterministic strategy
   - **Opportunity for claudecode**: Add tiered strategy for tool result prioritization

4. **Paper-Inspired Optimizations**:
   - Victor implements DCE (semantic deduplication), DACS (focus phase detection)
   - Claudecode doesn't have these optimizations
   - **Opportunity for claudecode**: Add DCE and DACS optimizations

---

## Part 3: Real Gaps and Integration Opportunities

### 3.1 Actual Gap: No Topic-Aware Segmentation

**Current State**:
- Topic extraction exists (`_extract_key_topics()` in ConversationController)
- But all messages are in a single flat list per session
- No segmentation by topic for multi-topic conversations

**Impact**:
- Multi-topic conversations pollute each other's context
- Subagents can't focus on specific topics without noise
- Context relevance degrades with topic switches

**Solution Design**:
```python
@dataclass
class TopicSegment:
    """A segment of conversation focused on a single topic."""
    topic_id: str
    topic_name: str
    start_message_id: str
    end_message_id: Optional[str]
    messages: List[ConversationMessage]
    metadata: Dict[str, Any]  # files mentioned, tools used, etc.

class TopicAwareContextAssembler:
    """Build context focused on current topic."""

    def build_context(self, current_topic: str) -> List[ConversationMessage]:
        """Build context with messages relevant to current topic."""
        # Get current topic segment
        # Get related historical segments (via embeddings)
        # Build context with topic-aware message selection
```

**Implementation Effort**: 5-7 days
- Create `TopicDetector` with LLM-based shift detection
- Create `TopicSegment` data structure
- Implement `TopicAwareContextAssembler`
- Add topic metadata to `ConversationStore`
- Write tests

**Integration Points**:
- Enhance existing `_extract_key_topics()` to detect shifts
- Use existing `ConversationEmbeddingStore` for semantic topic similarity
- Add topic_id FK to messages table in `ConversationStore`

---

### 3.2 Actual Gap: No File Timestamp Invalidation

**Current State**:
- File modification tracking exists (in `task_completion` and `compaction_summarizer`)
- But timestamps are not used for cache invalidation
- Stale file context remains in conversation until manually removed

**Impact**:
- Agent may use outdated file content after files change
- Tool calls may operate on stale context
- No automatic refresh when files are modified

**Solution Design**:
```python
@dataclass
class FileReference:
    """Track file references in messages."""
    file_path: str
    message_id: str
    timestamp: float  # When file was last modified
    hash: str  # File content hash for change detection

class FileContextInvalidator:
    """Invalidate context when files change."""

    def check_files_changed(self, messages: List[ConversationMessage]) -> List[str]:
        """Check if referenced files have changed."""
        # Compare current timestamps with stored timestamps
        # Return list of changed file paths

    def invalidate_context(self, changed_files: List[str]) -> None:
        """Invalidate context referencing changed files."""
        # Mark messages referencing changed files as stale
        # Trigger context refresh

    def refresh_file_context(self, file_path: str) -> Optional[ConversationMessage]:
        """Refresh context for changed file."""
        # Re-read file and create new context message
```

**Implementation Effort**: 4-5 days
- Create `FileReference` tracking
- Implement `FileContextInvalidator`
- Integrate with tool executor
- Add file watching for real-time updates
- Write tests

**Integration Points**:
- Enhance existing file tracking in `LedgerAwareCompactionSummarizer`
- Add file_references table to `ConversationStore`
- Integrate with tool execution pipeline

---

### 3.3 Integration Opportunity: Cross-Session Vector Retrieval

**Current State**:
- `ConversationEmbeddingStore` exists with LanceDB
- `ConversationController.retrieve_relevant_history()` exists
- `TurnBoundaryContextAssembler` calls `retrieve_relevant_history()` (lines 296-314)
- But not consistently used across all context building paths

**Impact**:
- Cold starts for new sessions (no historical context)
- Cross-session learning not fully utilized
- Vector retrieval capability underutilized

**Solution Design**:
```python
# Already exists in ConversationController (lines 884-937)
def retrieve_relevant_history(
    self,
    query: str,
    limit: int = 5,
) -> List[str]:
    """Retrieve relevant historical context from persistent store."""
    # Get semantically similar historical messages
    relevant_msgs = self._conversation_store.get_semantically_relevant_messages(
        self._session_id, query, limit=limit,
        min_similarity=self.config.semantic_relevance_threshold,
    )
    # Get relevant summaries
    relevant_summaries = self._conversation_store.get_relevant_summaries(
        self._session_id, query, limit=2,
    )

# Need to wire this into all context assembly paths consistently
```

**Implementation Effort**: 2-3 days
- Ensure `retrieve_relevant_history()` is called in all context assembly paths
- Add cross-session retrieval (current implementation is session-scoped)
- Add settings to control cross-session retrieval behavior
- Write tests

**Integration Points**:
- Wire into `TurnBoundaryContextAssembler.assemble()` (already has placeholder)
- Add to `ContextCompactor` for cross-session semantic compaction
- Add to session initialization for cold start avoidance

---

### 3.4 Integration Opportunity: Code Graph Integration

**Current State**:
- `TreeSitterEntityExtractor` exists with AST-based parsing
- EntityRelation support for call graphs, inheritance
- But not used for context building

**Impact**:
- Code relationships not leveraged for context relevance
- Function/class context not automatically included
- Missing opportunity for code-aware context

**Solution Design**:
```python
class GraphAwareContextSelector:
    """Select context based on code graph."""

    def select_relevant_functions(
        self,
        current_function: str,
        file_path: str,
    ) -> List[FunctionDefinition]:
        """Select functions relevant to current context."""
        # Use code graph to find callers/callees
        # Select related functions
        # Build context with function signatures + bodies

class CodeAwareContextAssembler:
    """Build context with code graph awareness."""

    def build_context(
        self,
        current_file: str,
        current_function: str,
    ) -> List[ConversationMessage]:
        """Build context with relevant code."""
        # Extract code graph
        # Find related functions
        # Build context with function definitions
```

**Implementation Effort**: 3-4 days
- Create `GraphAwareContextSelector`
- Create `CodeAwareContextAssembler`
- Integrate with existing `TreeSitterEntityExtractor`
- Add code graph caching
- Write tests

**Integration Points**:
- Use existing `TreeSitterEntityExtractor`
- Add to `TurnBoundaryContextAssembler` for code-aware context
- Add file-level metadata to `ConversationStore`

---

## Part 4: Implementation Priority (Revised)

### Phase 1: Integration Quick Wins (Week 1) - LOW RISK
**Goal**: Wire existing components together

**Tasks**:
1. Wire `retrieve_relevant_history()` into all context assembly paths (2-3 days)
2. Add cross-session retrieval to avoid cold starts (1-2 days)
3. Ensure vector retrieval is consistent across all paths (1 day)

**Deliverables**:
- Cross-session vector retrieval working
- No cold starts for new sessions
- Tests passing

**Effort**: 4-6 days
**Risk**: Low (wiring existing components)

---

### Phase 2: Topic-Aware Segmentation (Week 2-3) - MEDIUM RISK
**Goal**: Enable multi-topic conversations with segmented context

**Tasks**:
1. Create `TopicDetector` with LLM-based shift detection (2-3 days)
2. Create `TopicSegment` data structure (1 day)
3. Implement `TopicAwareContextAssembler` (2-3 days)
4. Add topic metadata to `ConversationStore` (1-2 days)
5. Write tests (1-2 days)

**Deliverables**:
- Multi-topic conversations supported
- Topic-aware context building
- Tests passing

**Effort**: 7-11 days
**Risk**: Medium (new feature, requires LLM calls)

---

### Phase 3: File Timestamp Invalidation (Week 4) - MEDIUM RISK
**Goal**: Automatic cache invalidation when files change

**Tasks**:
1. Create `FileReference` tracking (1 day)
2. Implement `FileContextInvalidator` (2-3 days)
3. Integrate with tool executor (1 day)
4. Add file watching for real-time updates (1-2 days)
5. Write tests (1 day)

**Deliverables**:
- File timestamp tracking
- Automatic cache invalidation
- Real-time file updates
- Tests passing

**Effort**: 6-8 days
**Risk**: Medium (file watching complexity)

---

### Phase 4: Code Graph Integration (Week 5) - MEDIUM RISK
**Goal**: Use tree-sitter graphs for code-aware context

**Tasks**:
1. Create `GraphAwareContextSelector` (2 days)
2. Create `CodeAwareContextAssembler` (2 days)
3. Integrate with existing `TreeSitterEntityExtractor` (1 day)
4. Add code graph caching (1 day)
5. Write tests (1 day)

**Deliverables**:
- Tree-sitter integration
- Code graph extraction
- Graph-based context selection
- Tests passing

**Effort**: 7 days
**Risk**: Medium (graph complexity)

---

### Phase 5: Optional - Deterministic Compaction (Week 6) - LOW RISK
**Goal**: Add claudecode-style deterministic compaction as fallback

**Tasks**:
1. Create `RuleBasedCompactionSummarizer` (2-3 days)
2. Add structured XML/JSON output format (1-2 days)
3. Add file path extraction to all strategies (1 day)
4. Write tests (1 day)

**Deliverables**:
- Deterministic compaction option
- Structured summary formats
- Tests passing

**Effort**: 5-7 days
**Risk**: Low (optional feature)

---

## Part 5: Success Criteria

### 5.1 Functional Requirements
- ✅ Cross-session vector retrieval working
- ✅ Topic-aware segmentation implemented
- ✅ File timestamp invalidation working
- ✅ Code graph integration working
- ✅ All tests passing
- ✅ Documentation complete

### 5.2 Performance Requirements
- Context building < 500ms for 1000 messages
- Vector retrieval < 100ms for top-10 results
- File invalidation < 50ms per file
- Code graph extraction < 200ms per file

### 5.3 Quality Requirements
- 90%+ test coverage
- No regressions in existing functionality
- Memory usage < 2x current usage
- Backward compatibility maintained

---

## Part 6: Comparison with Original Plan

### Original Plan (Assumptions)

| Feature | Original Assessment | Actual Reality |
|---------|-------------------|----------------|
| Vector-based retrieval | ❌ Not implemented | ✅ Fully implemented (LanceDB + lazy embedding) |
| Tree-sitter integration | ❌ Not implemented | ✅ Fully implemented (AST-based entity extraction) |
| Topic-aware segmentation | ❌ Not implemented | ⚠️ Partial (extraction exists, no segmentation) |
| File timestamp invalidation | ❌ Not implemented | ⚠️ Partial (tracking exists, no invalidation) |
| Sophisticated compaction | ❌ Not implemented | ✅ Fully implemented (multiple strategies) |
| Hierarchical compression | ❌ Not implemented | ✅ Fully implemented (epoch-level) |
| Semantic scoring | ❌ Not implemented | ✅ Fully implemented (configurable weights) |
| Rust acceleration | ❌ Not implemented | ✅ Fully implemented (4.8-9.9x speedup) |

### Revised Plan (Reality-Based)

| Feature | Status | Action Required |
|---------|--------|-----------------|
| Vector-based retrieval | ✅ Exists | Integration work only |
| Tree-sitter integration | ✅ Exists | Integration work only |
| Topic-aware segmentation | ⚠️ Partial | Build segmentation layer |
| File timestamp invalidation | ⚠️ Partial | Build invalidation layer |
| Sophisticated compaction | ✅ Exists | None (already excellent) |
| Hierarchical compression | ✅ Exists | None (already excellent) |
| Semantic scoring | ✅ Exists | None (already excellent) |
| Rust acceleration | ✅ Exists | None (already excellent) |

**Key Insight**: The original plan was based on assumptions without code examination. The real need is **integration and enhancement**, not building new features from scratch.

---

## Conclusion

Victor already has a **sophisticated context management system** that rivals or exceeds claudecode's capabilities in most areas:

**Victor Advantages**:
- Multiple compaction strategies (SIMPLE, TIERED, SEMANTIC, HYBRID)
- LLM-based rich summaries (vs claudecode's deterministic approach)
- Hierarchical compression (prevents linear accumulation)
- Rust-accelerated scoring (4.8-9.9x speedup)
- Paper-inspired optimizations (DCE, DACS, AutonAgenticAI)
- Semantic deduplication and focus phase detection
- Lazy embedding strategy (saves 90%+ compute)
- ML/RL-friendly SQLite schema

**Claudecode Advantages**:
- Deterministic rule-based (faster, cheaper, reproducible)
- Structured XML format (more machine-readable)
- Simpler implementation (~23KB Rust vs complex Python)

**Real Gaps**:
1. No topic-aware segmentation (only extraction exists)
2. No file timestamp invalidation (only tracking exists)
3. Cross-session vector retrieval exists but not consistently integrated
4. Code graph integration exists but not used for context

**Recommended Approach**:
1. Focus on **integration** of existing components (Phases 1-4)
2. Add **missing layers** (segmentation, invalidation) on top of existing foundation
3. Consider **optional deterministic compaction** (Phase 5) for resource-constrained environments
4. **Avoid rebuilding** what already exists

**Estimated Timeline**: 4-6 weeks for Phases 1-4 (vs 8 weeks in original plan)

**Risk Level**: Low-Medium (mostly integration work, fewer new features)

---

**Next Steps**:
1. Review this refined analysis
2. Prioritize phases based on business value
3. Begin Phase 1 (Integration Quick Wins)
4. Monitor performance and regressions

**Ready to proceed upon approval!**
