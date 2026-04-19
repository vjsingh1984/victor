# Victor Context Management & Compaction Enhancement Plan

**Date**: 2026-04-18
**Status**: Analysis Phase
**Scope**: Learn from claudecode compaction, enhance Victor's context management with topic-aware compaction, vector-based context retrieval, and file timestamp invalidation

---

## Executive Summary

After analyzing claudecode's compaction implementation, this document proposes enhancements to Victor's context management system. The goal is to create a more intelligent, topic-aware context system that:

1. **Learns from claudecode's deterministic rule-based compaction** (Rust-based, no LLM)
2. **Adds topic-aware context segmentation** for multi-topic conversations
3. **Implements vector-based context retrieval** using LanceDB embeddings
4. **Adds file timestamp-based cache invalidation** for dynamic context
5. **Enhances tree-sitter graph integration** for code-aware context

---

## Part 1: Claudecode Compaction Analysis

### 1.1 Architecture Overview

Claudecode's compaction is **entirely in Rust** at `/Users/vijaysingh/code/claudecode/rust/crates/runtime/src/compact.rs` (~23KB).

**Key Characteristics:**
- **No LLM-based summarization** - deterministic, rule-based summarizer
- **Runs locally** - no API calls
- **Three-layer architecture**: Data Model → Configuration → Core Algorithm

### 1.2 Data Model

```rust
// Session structure
struct Session {
    version: u32,
    messages: Vec<ConversationMessage>,
}

// Message structure
struct ConversationMessage {
    role: Role,  // System/User/Assistant/Tool
    blocks: Vec<ContentBlock>,  // Text/ToolUse/ToolResult
    usage: Option<TokenUsage>,
}

// Content block (tagged union)
enum ContentBlock {
    Text(String),
    ToolUse(ToolUse),
    ToolResult(ToolResult),
}
```

**Victor Comparison:**
- Victor has similar structures in `victor/agent/conversation/types.py`
- Victor's `ConversationMessage` is more feature-rich (priority, scoring, metadata)
- Both use role-based message structure

### 1.3 Configuration

```rust
pub struct CompactionConfig {
    pub preserve_recent_messages: usize,  // default: 4
    pub max_estimated_tokens: usize,      // default: 10,000
}
```

**Trigger Conditions (BOTH must be true):**
1. Number of compactable messages > `preserve_recent_messages`
2. Estimated token count ≥ `max_estimated_tokens`

**Token Estimation:**
- Uses `estimate_message_tokens()` - heuristic, not exact tokenizer
- ~4 chars per token estimate

**Victor Comparison:**
- Victor has more sophisticated token tracking via `StreamMetrics`
- Victor has actual token counts from provider API when available
- Victor's `ConversationStore` has token-aware pruning with `ScoringWeights`

### 1.4 Core Compaction Algorithm

When triggered, the algorithm does:

1. **Detect prior compaction** - checks if first message is already a summary
2. **Split messages into zones:**
   - `[0..compacted_prefix_len]` - existing summary (0 or 1 message)
   - `[compacted_prefix_len..keep_from]` - messages to compact (removed)
   - `[keep_from..end]` - last N messages preserved verbatim (default: 4)
3. **Generate structured summary** via `summarize_messages()`:
   - **Scope** - count of compacted messages by role
   - **Tools mentioned** - unique tool names extracted
   - **Recent user requests** - last 3 user messages (~160 chars each)
   - **Pending work** - inferred from "Next:" or "TODO:" patterns
   - **Key files referenced** - extracted via regex matching file paths
   - **Current work** - inferred from last assistant message
   - **Key timeline** - each message gets one-line entry with role + truncated content
4. **Merge with prior summary** - combines old + new summaries
5. **Format continuation message** - system message with:
   - Preamble: "This session is being continued from a previous conversation..."
   - Merged summary
   - Note: recent messages preserved verbatim
   - Instruction: "Continue the conversation from where it left off..."
6. **Construct new session** - replaces message list with:
   - One System message containing continuation
   - Preserved recent messages

**Victor Comparison:**
- Victor has `ContextCompactor` in `victor/agent/context_compactor.py`
- Victor's compaction is **more sophisticated** with semantic scoring
- Victor uses **priority-based message pruning** (not just recent messages)
- Victor has **ML/RL-friendly aggregation** via `ConversationStore`

### 1.5 Key Design Decisions

| Aspect | Claudecode | Victor |
|--------|-----------|--------|
| **Summarization** | Deterministic rule-based (no LLM) | LLM-based with prompt optimization |
| **Summary Format** | Structured XML `<summary>` tags | Natural language with structured sections |
| **Idempotency** | Yes - merges old + new summaries | Yes - incremental compaction |
| **Threshold** | Dual: message count AND token estimate | Token budget + priority scoring |
| **Re-compaction Safety** | Excludes existing summary from decision | Separate summary layer |
| **Block Truncation** | ~160 chars with "…" suffix | Priority-based full/summarized/dropped |
| **Result** | Returns `CompactionResult` with metadata | Returns pruned message list |

**Victor Advantages:**
- More sophisticated scoring (semantic, priority, recency)
- ML/RL-friendly with `ConversationStore`
- Token-aware with actual counts from API
- Structured sections for different message types

**Claudecode Advantages:**
- Simpler, faster (no LLM calls)
- Deterministic (reproducible)
- Lower cost (no API usage)
- Better for resource-constrained environments

---

## Part 2: Victor's Current Context Management

### 2.1 Existing Components

**Context Storage:**
- `ConversationStore` (victor/agent/conversation/store.py) - SQLite persistence
- `MessageHistory` (victor/agent/message_history.py) - In-memory message list
- `ConversationController` (victor/agent/conversation/controller.py) - State management

**Context Building:**
- `TurnBoundaryContextAssembler` (victor/agent/conversation/assembler.py) - Context budget management
- `SystemPromptBuilder` (victor/agent/prompt_builder.py) - Prompt construction

**Compaction:**
- `ContextCompactor` (victor/agent/context_compactor.py) - Message compaction logic
- `LLMCompactionSummarizer` (victor/agent/llm_compaction_summarizer.py) - LLM-based summarization

**Scoring:**
- `score_messages()` (victor/agent/conversation/scoring.py) - Message relevance scoring
- `ScoringWeights` presets (DEFAULT, CONTROLLER, STORE)

### 2.2 Current Limitations

1. **No topic-aware segmentation** - all messages in single session
2. **No vector-based retrieval** - only keyword/semantic search
3. **No file timestamp invalidation** - static context once compacted
4. **Limited tree-sitter integration** - code graphs not used for context
5. **Session-based isolation** - no cross-session learning

---

## Part 3: Proposed Enhancements

### 3.1 Topic-Aware Context Segmentation

**Goal:** Enable multiple topics within a single conversation, with context segmented by topic.

**Approach:**

1. **Topic Detection:**
   ```python
   class TopicDetector:
       """Detect conversation topics using LLM + embeddings."""

       def detect_topic_shift(self, messages: List[ConversationMessage]) -> Optional[str]:
           """Detect if topic has shifted based on message analysis."""
           # Use embeddings to detect semantic shifts
           # Use LLM to categorize topic (coding, debugging, analysis, etc.)
           # Return new topic name if shift detected
   ```

2. **Topic Segments:**
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
   ```

3. **Topic-Aware Context Building:**
   ```python
   class TopicAwareContextAssembler:
       """Build context focused on current topic."""

       def build_context(self, current_topic: str) -> List[ConversationMessage]:
           """Build context with messages relevant to current topic."""
           # Get current topic segment
           # Get related historical segments (via embeddings)
           # Build context with topic-aware message selection
   ```

**Benefits:**
- Multi-topic conversations don't pollute each other's context
- Subagent handling per topic
- Better context relevance

**Implementation Effort:** 5-7 days

### 3.2 Vector-Based Context Retrieval

**Goal:** Use LanceDB embeddings to retrieve relevant historical context.

**Approach:**

1. **Message Embedding Storage:**
   ```python
   class MessageEmbeddingStore:
       """Store message embeddings in LanceDB for similarity search."""

       def store_message(self, message: ConversationMessage, embedding: np.ndarray):
           """Store message with embedding."""

       def find_similar_messages(self, query: str, top_k: int = 5) -> List[ConversationMessage]:
           """Find similar messages using vector similarity."""
   ```

2. **Context Retrieval:**
   ```python
   class VectorContextRetriever:
       """Retrieve relevant context from historical conversations."""

       def retrieve_context(self, current_message: str, topic: str) -> List[ConversationMessage]:
           """Retrieve relevant historical messages."""
           # Embed current message
           # Search LanceDB for similar messages
           # Filter by topic
           # Return top-K relevant messages
   ```

3. **Integration with ConversationStore:**
   ```python
   class EnhancedConversationStore(ConversationStore):
       """Enhanced conversation store with vector retrieval."""

       def get_relevant_history(self, session_id: str, current_context: str) -> List[ConversationMessage]:
           """Get relevant historical messages across sessions."""
   ```

**Benefits:**
- Avoid cold starts with relevant historical context
- Cross-session learning
- Better context for recurring themes

**Implementation Effort:** 7-10 days

### 3.3 File Timestamp Invalidation

**Goal:** Use file timestamps to invalidate stale context and refresh relevant files.

**Approach:**

1. **File Reference Tracking:**
   ```python
   @dataclass
   class FileReference:
       """Track file references in messages."""
       file_path: str
       message_id: str
       timestamp: float  # When file was last modified
       hash: str  # File content hash for change detection
   ```

2. **Timestamp-Based Invalidation:**
   ```python
   class FileContextInvalidator:
       """Invalidate context when files change."""

       def check_files_changed(self, messages: List[ConversationMessage]) -> List[str]:
           """Check if referenced files have changed."""

       def invalidate_context(self, changed_files: List[str]) -> None:
           """Invalidate context referencing changed files."""

       def refresh_file_context(self, file_path: str) -> Optional[ConversationMessage]:
           """Refresh context for changed file."""
   ```

3. **Integration with Tool Calls:**
   ```python
   class ContextAwareToolExecutor:
       """Tool executor that updates file references."""

       def after_tool_call(self, tool: str, result: Any):
           """Update file references after tool execution."""
           if tool in ("read_file", "write_file", "edit_file"):
               # Track file references
               # Check timestamps
               # Invalidate stale context
   ```

**Benefits:**
- Always use up-to-date file context
- Automatic cache invalidation
- Better tool call accuracy

**Implementation Effort:** 4-5 days

### 3.4 Tree-Sitter Graph Integration

**Goal:** Use tree-sitter code graphs for intelligent code-aware context.

**Approach:**

1. **Code Graph Extraction:**
   ```python
   class CodeGraphExtractor:
       """Extract code structure using tree-sitter."""

       def extract_graph(self, file_path: str) -> CodeGraph:
           """Extract code graph from file."""
           # Use tree-sitter to parse code
           # Extract function/class definitions
           # Build call graph
           # Return structured graph
   ```

2. **Graph-Based Context Selection:**
   ```python
   class GraphAwareContextSelector:
       """Select context based on code graph."""

       def select_relevant_functions(self, current_function: str, file_path: str) -> List[FunctionDefinition]:
           """Select functions relevant to current context."""
           # Use code graph to find callers/callees
           # Select related functions
           # Build context with function signatures + bodies
   ```

3. **Integration with Context Building:**
   ```python
   class CodeAwareContextAssembler:
       """Build context with code graph awareness."""

       def build_context(self, current_file: str, current_function: str) -> List[ConversationMessage]:
           """Build context with relevant code."""
           # Extract code graph
           # Find related functions
           # Build context with function definitions
   ```

**Benefits:**
- More relevant code context
- Better understanding of code relationships
- Improved tool call accuracy

**Implementation Effort:** 8-10 days

---

## Part 4: Integration Architecture

### 4.1 Proposed Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ContextManager (NEW)                      │
│  - Coordinates all context-related operations               │
│  - Manages topic segmentation                               │
│  - Orchestrates vector retrieval, file invalidation, etc.   │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼─────────┐   ┌───────▼─────────┐   ┌──────▼──────────┐
│  TopicSegmenter  │   │ VectorRetriever │   │ FileInvalidator │
│  - Detect shifts │   │ - LanceDB search│   │ - Timestamps    │
│  - Manage topics │   │ - Embeddings    │   │ - Hash checks   │
└─────────────────┘   └─────────────────┘   └─────────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Enhanced         │
                    │  ConversationStore│
                    │  - SQLite + LanceDB│
                    │  - Topic metadata  │
                    │  - File references │
                    └───────────────────┘
```

### 4.2 Data Flow

1. **New Message Arrives:**
   ```
   New Message → TopicDetector → Topic Shift?
                                 ├─ Yes → Create new segment, store previous
                                 └─ No  → Add to current segment
   ```

2. **Context Building:**
   ```
   Build Context → TopicAwareAssembler → Get current segment
                                     → VectorRetriever → Get relevant history
                                     → FileInvalidator → Check file changes
                                     → CodeGraphExtractor → Get relevant code
                                     → Merge all sources
                                     → Return context
   ```

3. **Tool Call Execution:**
   ```
   Tool Call → FileInvalidator → Track file references
                                 → Check timestamps
                                 → Invalidate if changed
                                 → Refresh context
   ```

---

## Part 5: Implementation Phases

### Phase 1: Foundation (Week 1-2)
**Goal:** Add topic-aware segmentation

**Tasks:**
1. Create `TopicDetector` with LLM-based shift detection
2. Create `TopicSegment` data structure
3. Implement `TopicAwareContextAssembler`
4. Add topic metadata to `ConversationStore`
5. Write tests for topic detection and segmentation

**Deliverables:**
- Multi-topic conversations supported
- Topic-aware context building
- Tests passing

### Phase 2: Vector Retrieval (Week 3-4)
**Goal:** Add LanceDB-based context retrieval

**Tasks:**
1. Set up LanceDB with message embeddings
2. Create `MessageEmbeddingStore`
3. Implement `VectorContextRetriever`
4. Integrate with `ConversationStore`
5. Add embedding pipeline for new messages
6. Write tests for vector retrieval

**Deliverables:**
- LanceDB integration
- Vector-based context retrieval
- Cross-session learning
- Tests passing

### Phase 3: File Invalidation (Week 5)
**Goal:** Add timestamp-based cache invalidation

**Tasks:**
1. Create `FileReference` tracking
2. Implement `FileContextInvalidator`
3. Integrate with tool executor
4. Add file watching for real-time updates
5. Write tests for invalidation logic

**Deliverables:**
- File timestamp tracking
- Automatic cache invalidation
- Real-time file updates
- Tests passing

### Phase 4: Code Graph Integration (Week 6-7)
**Goal:** Add tree-sitter graph-based context

**Tasks:**
1. Create `CodeGraphExtractor` with tree-sitter
2. Implement `GraphAwareContextSelector`
3. Create `CodeAwareContextAssembler`
4. Add code graph caching
5. Write tests for graph-based context

**Deliverables:**
- Tree-sitter integration
- Code graph extraction
- Graph-based context selection
- Tests passing

### Phase 5: Integration & Testing (Week 8)
**Goal:** Integrate all components and comprehensive testing

**Tasks:**
1. Create `ContextManager` orchestrator
2. Integrate all components
3. End-to-end testing
4. Performance optimization
5. Documentation

**Deliverables:**
- Full integration
- Comprehensive tests
- Documentation
- Performance benchmarks

---

## Part 6: Success Criteria

### 6.1 Functional Requirements
- ✅ Multi-topic conversations supported
- ✅ Vector-based context retrieval working
- ✅ File timestamp invalidation working
- ✅ Code graph integration working
- ✅ All tests passing
- ✅ Documentation complete

### 6.2 Performance Requirements
- Context building < 500ms for 1000 messages
- Vector retrieval < 100ms for top-10 results
- File invalidation < 50ms per file
- Code graph extraction < 200ms per file

### 6.3 Quality Requirements
- 90%+ test coverage
- No regressions in existing functionality
- Memory usage < 2x current usage
- Backward compatibility maintained

---

## Part 7: Open Questions & Risks

### 7.1 Open Questions

1. **Topic Detection Granularity:** How fine-grained should topics be? (file-level? function-level? feature-level?)
2. **Vector Database:** Should we use LanceDB or another vector DB? (Milvus? Qdrant? Weaviate?)
3. **File Watching:** Should we use inotify (Linux) or FSEvents (macOS) or cross-platform library?
4. **Code Graph Storage:** Should we store code graphs in SQLite or separate graph DB?
5. **Backward Compatibility:** How to handle existing conversations without topic segmentation?

### 7.2 Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Topic detection accuracy | High | Use LLM + embeddings hybrid approach |
| Vector DB performance | Medium | Benchmark multiple solutions, use caching |
| File watching reliability | Medium | Use polling fallback, handle failures gracefully |
| Code graph extraction speed | High | Cache graphs, use incremental updates |
| Memory usage | Medium | Implement LRU cache, monitor usage |
| Backward compatibility | High | Add feature flags, gradual rollout |

---

## Part 8: Next Steps

1. **Review this analysis** with team
2. **Prioritize phases** based on business value
3. **Set up proof-of-concept** for highest-risk components (topic detection, vector retrieval)
4. **Define success metrics** for each phase
5. **Begin implementation** starting with Phase 1

---

## Conclusion

This plan proposes significant enhancements to Victor's context management system, learning from claudecode's deterministic compaction while adding advanced features like topic-aware segmentation, vector-based retrieval, file timestamp invalidation, and code graph integration.

The phased approach allows for incremental delivery and risk mitigation, with each phase building on the previous one. The estimated timeline is 8 weeks for full implementation, with potential to deliver value earlier through phased releases.

**Key Success Factors:**
- Maintain backward compatibility
- Keep performance overhead minimal
- Ensure test coverage > 90%
- Document all changes
- Monitor performance in production

**Ready to proceed upon approval!**
