# Claudecode vs Victor: Direct Compaction Comparison

**Date**: 2026-04-18
**Scope**: Direct source code comparison between claudecode (compact.rs) and Victor (context management modules)

---

## Executive Summary

| Aspect | Claudecode | Victor | Winner |
|--------|-----------|--------|--------|
| **Implementation** | 703 lines Rust | ~2,000+ lines Python (multiple modules) | Claudecode (simplicity) |
| **Approach** | Deterministic rule-based | Multi-strategy (LLM + deterministic) | Victor (flexibility) |
| **Performance** | Fast (Rust, no API calls) | Variable (LLM calls, Rust acceleration) | Claudecode (speed) |
| **Cost** | Zero (no API usage) | Non-zero (LLM calls for SEMANTIC/HYBRID) | Claudecode (cost) |
| **Summary Quality** | Structured, machine-readable | Rich, abstractive, natural language | Victor (quality) |
| **Configurability** | 2 parameters | 5 strategies + configurable weights | Victor (flexibility) |
| **Token Estimation** | Rough (len/4 for all) | Accurate (content-type aware) | Victor (accuracy) |

**Key Insight**: These are **different design philosophies**, not one better than the other:
- **Claudecode**: Fast, cheap, deterministic, simple
- **Victor**: Rich, flexible, intelligent, complex

---

## Part 1: Source Code Comparison

### 1.1 File Structure

**Claudecode** (`rust/crates/runtime/src/compact.rs`):
```
703 lines total
├── Data structures (lines 8-30): CompactionConfig, CompactionResult
├── Public API (lines 32-86): estimate_tokens, should_compact, format_summary, compact_session
├── Core algorithm (lines 89-131): compact_session
├── Summary generation (lines 143-228): summarize_messages
├── Merge logic (lines 230-263): merge_compact_summaries
├── Helper functions (lines 265-500): extract_file_candidates, truncate_summary, etc.
└── Tests (lines 502-703): 8 comprehensive tests
```

**Victor** (multiple Python modules):
```
~2,000+ lines total
├── victor/agent/context_compactor.py (600+ lines)
│   ├── MessagePriority enum (CRITICAL=100, HIGH=75, MEDIUM=50, LOW=25, EPHEMERAL=0)
│   ├── CompactionStrategy enum (SIMPLE, TIERED, SEMANTIC, HYBRID)
│   ├── TruncationStrategy (HEAD, TAIL, BOTH, SMART)
│   ├── Token estimation by content type (code=3.0, json=2.8, prose=4.0, mixed=3.5)
│   └── Pinned requirement patterns (8 regex patterns)
├── victor/agent/conversation/scoring.py (195 lines)
│   ├── ScoringWeights (priority, recency, role, length, semantic)
│   ├── 3 presets (STORE, CONTROLLER, DEFAULT)
│   └── Rust-accelerated batch scoring (4.8-9.9x speedup)
├── victor/agent/conversation/controller.py (1038 lines)
│   ├── smart_compact_history() with 4 strategies
│   ├── _extract_key_topics() (CamelCase, snake_case, keywords)
│   └── retrieve_relevant_history() for semantic retrieval
├── victor/agent/compaction_hierarchy.py (200+ lines)
│   ├── HierarchicalCompactionManager
│   └── Epoch-level compression to prevent linear accumulation
├── victor/agent/compaction_summarizer.py (300+ lines)
│   ├── KeywordCompactionSummarizer
│   └── LedgerAwareCompactionSummarizer
└── victor/agent/llm_compaction_summarizer.py (150+ lines)
    └── LLM-based abstractive summarization
```

**Comparison**:
- **Claudecode**: Single file, focused, simple (703 lines)
- **Victor**: Multiple modules, sophisticated, complex (~2,000+ lines)

---

### 1.2 Configuration

**Claudecode** (lines 8-21):
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompactionConfig {
    pub preserve_recent_messages: usize,  // default: 4
    pub max_estimated_tokens: usize,      // default: 10,000
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            preserve_recent_messages: 4,
            max_estimated_tokens: 10_000,
        }
    }
}
```

**Victor** (from context_compactor.py and conversation/controller.py):
```python
@dataclass
class ConversationConfig:
    max_context_chars: int = CONTEXT_LIMITS.max_context_chars  # ~200,000
    chars_per_token_estimate: int = CONTEXT_LIMITS.chars_per_token_estimate  # ~2.8
    enable_stage_tracking: bool = True
    enable_context_monitoring: bool = True
    compaction_strategy: CompactionStrategy = CompactionStrategy.TIERED
    min_messages_to_keep: int = 6
    tool_result_retention_weight: float = COMPACTION_CONFIG.tool_result_retention_weight
    recent_message_weight: float = COMPACTION_CONFIG.recent_message_weight
    semantic_relevance_threshold: float = SEMANTIC_THRESHOLDS.compaction_relevance

class CompactionStrategy(Enum):
    SIMPLE = "simple"      # Keep N recent
    TIERED = "tiered"      # Prioritize tool results
    SEMANTIC = "semantic"  # Use embeddings
    HYBRID = "hybrid"      # Combine tiered + semantic

class ScoringWeights:
    priority: float = 0.2
    recency: float = 0.4
    role: float = 0.2
    length: float = 0.1
    semantic: float = 0.1
```

**Comparison**:
- **Claudecode**: 2 parameters (preserve_recent_messages, max_estimated_tokens)
- **Victor**: 10+ parameters across multiple configs (strategy, weights, thresholds)

**Trade-off**:
- Claudecode: Simple, easy to understand, less configurable
- Victor: Complex, highly configurable, adaptable to different scenarios

---

### 1.3 Trigger Conditions

**Claudecode** (lines 37-47):
```rust
pub fn should_compact(session: &Session, config: CompactionConfig) -> bool {
    let start = compacted_summary_prefix_len(session);
    let compactable = &session.messages[start..];

    compactable.len() > config.preserve_recent_messages
        && compactable
            .iter()
            .map(estimate_message_tokens)
            .sum::<usize>()
            >= config.max_estimated_tokens
}
```

**Key Points**:
- BOTH conditions must be true (AND logic)
- Excludes existing summary from compactable count
- Checks message count AND token estimate

**Victor** (from conversation/controller.py lines 322-342):
```python
def get_context_metrics(self) -> ContextMetrics:
    """Calculate current context metrics."""
    total_chars = sum(len(m.content) for m in self.messages)
    estimated_tokens = total_chars // self.config.chars_per_token_estimate

    return ContextMetrics(
        char_count=total_chars,
        estimated_tokens=estimated_tokens,
        message_count=len(self.messages),
        is_overflow_risk=total_chars
            > self.config.max_context_chars * CONTEXT_LIMITS.overflow_threshold,  # 0.75
        max_context_chars=self.config.max_context_chars,
    )

def check_context_overflow(self) -> bool:
    """Check if context is at risk of overflow."""
    metrics = self.get_context_metrics()
    return metrics.is_overflow_risk
```

**Key Points**:
- Single threshold check (overflow_threshold = 0.75 of max_context_chars)
- Proactive compaction before overflow (at 75% capacity)
- More sophisticated metrics (char_count, message_count, utilization)

**Comparison**:
- **Claudecode**: Dual threshold (message count AND tokens), reactive
- **Victor**: Single threshold (75% of max), proactive

**Trade-off**:
- Claudecode: More precise (two conditions), reactive (compact when full)
- Victor: More proactive (compact at 75%), simpler threshold

---

### 1.4 Token Estimation

**Claudecode** (lines 391-403):
```rust
fn estimate_message_tokens(message: &ConversationMessage) -> usize {
    message
        .blocks
        .iter()
        .map(|block| match block {
            ContentBlock::Text { text } => text.len() / 4 + 1,
            ContentBlock::ToolUse { name, input, .. } => (name.len() + input.len()) / 4 + 1,
            ContentBlock::ToolResult {
                tool_name, output, ..
            } => (tool_name.len() + output.len()) / 4 + 1,
        })
        .sum()
}
```

**Key Points**:
- Simple heuristic: `len / 4` for all content types
- Same factor for text, tool_use, tool_result
- Very rough estimate (assumes 4 chars per token)

**Victor** (from context_compactor.py):
```python
class ContextCompactor:
    """Proactive context compaction and optimization."""

    TOKEN_FACTORS = {
        "code": 3.0,      # Code is more token-dense
        "json": 2.8,      # JSON has structure overhead
        "prose": 4.0,     # English prose ~4 chars/token
        "mixed": 3.5,     # Default mixed content
    }

    def estimate_tokens(self, content: str, content_type: str = "mixed") -> int:
        """Estimate token count for content."""
        factor = self.TOKEN_FACTORS.get(content_type, self.TOKEN_FACTORS["mixed"])
        return int(len(content) / factor)
```

**Key Points**:
- Content-type aware estimation
- Different factors for code (3.0), json (2.8), prose (4.0), mixed (3.5)
- More accurate than simple len/4

**Comparison**:
- **Claudecode**: Simple (len/4 for all), less accurate
- **Victor**: Content-type aware (3.0-4.0), more accurate

**Trade-off**:
- Claudecode: Simpler, faster, less accurate
- Victor: More complex, slower, more accurate

---

### 1.5 Summary Format

**Claudecode** (lines 169-228):
```rust
fn summarize_messages(messages: &[ConversationMessage]) -> String {
    let user_messages = /* count user messages */;
    let assistant_messages = /* count assistant messages */;
    let tool_messages = /* count tool messages */;
    let mut tool_names = /* extract unique tool names */;

    let mut lines = vec![
        "<summary>".to_string(),
        "Conversation summary:".to_string(),
        format!(
            "- Scope: {} earlier messages compacted (user={}, assistant={}, tool={}).",
            messages.len(), user_messages, assistant_messages, tool_messages
        ),
    ];

    if !tool_names.is_empty() {
        lines.push(format!("- Tools mentioned: {}.", tool_names.join(", ")));
    }

    let recent_user_requests = collect_recent_role_summaries(messages, MessageRole::User, 3);
    if !recent_user_requests.is_empty() {
        lines.push("- Recent user requests:".to_string());
        lines.extend(recent_user_requests.into_iter().map(|request| format!("  - {request}")));
    }

    let pending_work = infer_pending_work(messages);
    if !pending_work.is_empty() {
        lines.push("- Pending work:".to_string());
        lines.extend(pending_work.into_iter().map(|item| format!("  - {item}")));
    }

    let key_files = collect_key_files(messages);
    if !key_files.is_empty() {
        lines.push(format!("- Key files referenced: {}.", key_files.join(", ")));
    }

    if let Some(current_work) = infer_current_work(messages) {
        lines.push(format!("- Current work: {current_work}"));
    }

    lines.push("- Key timeline:".to_string());
    for message in messages {
        let role = /* convert role to string */;
        let content = /* truncate to 160 chars */;
        lines.push(format!("  - {role}: {content}"));
    }
    lines.push("</summary>".to_string());
    lines.join("\n")
}
```

**Example Output**:
```xml
<summary>
Conversation summary:
- Scope: 42 earlier messages compacted (user=15, assistant=20, tool=7).
- Tools mentioned: bash, code_search, grep, edit_file.
- Recent user requests:
  - Fix authentication bug in login flow
  - Update error handling
  - Add retry logic
- Pending work:
  - Need to add retry logic for failed login attempts
  - Update OAuth token refresh
- Key files referenced: src/auth/login.py, src/auth/oauth.py, src/auth/token.py.
- Current work: Currently debugging the authentication flow where users are getting logged out immediately after login.
- Key timeline:
  - user: Fix the login bug
  - assistant: I'll help fix the login bug
  - tool: tool_result bash: error Authentication failed
  - assistant: Found the issue in the login flow
  - user: Update error handling
  - assistant: Updated error handling in login.py
</summary>
```

**Victor** (from compaction_summarizer.py and llm_compaction_summarizer.py):

**KeywordCompactionSummarizer**:
```python
def summarize(self, messages: List[Message]) -> str:
    """Generate summary using keyword extraction."""
    topics = self._extract_key_topics(messages)
    files = self._extract_files_mentioned(messages)
    tools = self._extract_tools_used(messages)

    parts = []
    if topics:
        parts.append(f"topics: {', '.join(topics)}")
    if files:
        parts.append(f"files: {', '.join(files)}")
    if tools:
        parts.append(f"tools: {', '.join(tools)}")

    return f"[Earlier conversation: {'; '.join(parts)}]"
```

**LedgerAwareCompactionSummarizer**:
```python
def summarize(self, messages: List[Message]) -> str:
    """Generate summary using session ledger."""
    ledger = self._get_ledger()

    parts = []
    if ledger.files_read:
        parts.append(f"Files read: {', '.join(ledger.files_read)}")
    if ledger.files_modified:
        parts.append(f"Files modified: {', '.join(ledger.files_modified)}")
    if ledger.decisions:
        parts.append(f"Decisions: {'; '.join(ledger.decisions)}")
    if ledger.recommendations:
        parts.append(f"Recommendations: {'; '.join(ledger.recommendations)}")
    if ledger.pending_actions:
        parts.append(f"Pending: {'; '.join(ledger.pending_actions)}")

    return "\n".join(parts)
```

**LLMCompactionSummarizer**:
```python
async def summarize(self, messages: List[Message]) -> str:
    """Generate abstractive summary using LLM."""
    prompt = self._build_summary_prompt(messages)
    summary = await self._llm_service.generate(prompt, timeout=5.0)

    if not summary:
        # Fallback to keyword summarizer
        return self._fallback_summarizer.summarize(messages)

    return summary
```

**Example Outputs**:

*Keyword*:
```
[Earlier conversation: topics: authentication, login, oauth, error_handling; files: src/auth/login.py, src/auth/oauth.py; tools: bash, code_search, grep, edit_file]
```

*Ledger*:
```
Files read: src/auth/login.py, src/auth/oauth.py
Files modified: src/auth/login.py
Decisions: Add retry logic for failed login attempts
Recommendations: Update error handling in oauth flow
Pending: Test login flow with retry logic
```

*LLM*:
```
The conversation focused on debugging an authentication issue where users were being
logged out immediately after login. We identified the problem in the login flow
(src/auth/login.py), updated error handling, and decided to add retry logic for
failed authentication attempts. The OAuth token refresh mechanism in src/auth/oauth.py
also needs to be updated to handle the new error cases.
```

**Comparison**:
- **Claudecode**: Structured XML format, machine-readable, consistent structure
- **Victor**: Multiple formats (keyword, ledger, LLM), natural language, variable structure

**Trade-off**:
- Claudecode: Machine-readable, parseable, consistent, less rich
- Victor: Human-readable, rich, flexible, less parseable

---

### 1.6 Compaction Algorithm

**Claudecode** (lines 89-131):
```rust
pub fn compact_session(session: &Session, config: CompactionConfig) -> CompactionResult {
    if !should_compact(session, config) {
        return CompactionResult {
            summary: String::new(),
            formatted_summary: String::new(),
            compacted_session: session.clone(),
            removed_message_count: 0,
        };
    }

    let existing_summary = session
        .messages
        .first()
        .and_then(extract_existing_compacted_summary);
    let compacted_prefix_len = usize::from(existing_summary.is_some());
    let keep_from = session
        .messages
        .len()
        .saturating_sub(config.preserve_recent_messages);
    let removed = &session.messages[compacted_prefix_len..keep_from];
    let preserved = session.messages[keep_from..].to_vec();
    let summary = merge_compact_summaries(existing_summary.as_deref(), &summarize_messages(removed));
    let formatted_summary = format_compact_summary(&summary);
    let continuation = get_compact_continuation_message(&summary, true, !preserved.is_empty());

    let mut compacted_messages = vec![ConversationMessage {
        role: MessageRole::System,
        blocks: vec![ContentBlock::Text { text: continuation }],
        usage: None,
    }];
    compacted_messages.extend(preserved);

    CompactionResult {
        summary,
        formatted_summary,
        compacted_session: Session {
            version: session.version,
            messages: compacted_messages,
        },
        removed_message_count: removed.len(),
    }
}
```

**Key Steps**:
1. Check if compaction is needed (should_compact)
2. Detect existing summary (if any)
3. Split messages into zones:
   - `[0..compacted_prefix_len]`: existing summary (0 or 1 message)
   - `[compacted_prefix_len..keep_from]`: messages to compact
   - `[keep_from..end]`: recent messages to preserve
4. Generate new summary via `summarize_messages()`
5. Merge with existing summary via `merge_compact_summaries()`
6. Format continuation message
7. Construct new session with summary + preserved messages

**Victor** (from conversation/controller.py lines 390-561):
```python
def smart_compact_history(
    self,
    target_messages: Optional[int] = None,
    current_query: Optional[str] = None,
) -> int:
    """Smart context compaction using configured strategy."""
    target = target_messages or self.config.min_messages_to_keep
    strategy = self.config.compaction_strategy

    if len(self.messages) <= target + 1:
        return 0

    logger.info(f"Smart compacting with strategy: {strategy.value}")

    if strategy == CompactionStrategy.SIMPLE:
        return self.compact_history(keep_recent=target)

    # Normalize non-Message objects IN-PLACE
    # ... (normalization code)

    # Score all messages for importance
    scored_messages = self._score_messages(current_query=current_query)

    # Sort by score (descending) and keep top N
    scored_messages.sort(key=lambda x: x.score, reverse=True)

    # ALWAYS keep root system message (for caching stability)
    system_msgs_to_keep = []
    if self.messages and self.messages[0].role == "system":
        system_msgs_to_keep.append(self.messages[0])
        scored_messages = [sm for sm in scored_messages if sm.index != 0]

    # Keep top messages (up to target)
    messages_to_keep = scored_messages[:target]

    # Ensure balanced representation: Guaranteed Role Minimums
    # Ensure at least 1 USER and 1 ASSISTANT message
    if "user" not in kept_roles:
        # Add highest-scoring user message
    if "assistant" not in kept_roles:
        # Add most recent assistant message

    # Preserve tool-call pairs (prevent orphaned tool_calls)
    # ... (pair preservation code)

    # Sort by original index to maintain conversation order
    messages_to_keep.sort(key=lambda x: x.index)

    # Generate summary of removed messages
    removed_indices = {sm.index for sm in scored_messages[target:]}
    removed_messages = [m for i, m in enumerate(self.messages) if i in removed_indices]
    if removed_messages:
        summary = self._generate_compaction_summary(removed_messages)
        if summary:
            self._compaction_summaries.append(summary)
            # Feed into hierarchical manager if available
            if self._hierarchical_manager:
                turn_index = len(self.messages)
                self._hierarchical_manager.add_summary(summary, turn_index)
            # Persist to SQLite for future semantic retrieval
            self.persist_compaction_summary(summary, message_ids)

    # Rebuild message list
    self._history.clear()
    for msg in system_msgs_to_keep:
        self._history.append_message(msg)
    for scored_msg in messages_to_keep:
        self._history.append_message(scored_msg.message)

    return removed_count
```

**Key Steps**:
1. Check if compaction is needed (message count vs target)
2. Normalize non-Message objects (defensive programming)
3. Score all messages using canonical scorer (priority, recency, role, length, semantic)
4. Sort by score (descending)
5. Keep root system message (for caching stability)
6. Keep top N messages by score
7. Ensure balanced representation (at least 1 USER, 1 ASSISTANT)
8. Preserve tool-call pairs (prevent orphaned tool_calls)
9. Sort by original index (maintain conversation order)
10. Generate summary via compaction summarizer strategy
11. Feed into hierarchical manager (epoch-level compression)
12. Persist summary to SQLite (for semantic retrieval)
13. Rebuild message list

**Comparison**:
- **Claudecode**: Simple, deterministic, 7 steps, no scoring
- **Victor**: Complex, intelligent, 13+ steps, multi-factor scoring

**Trade-off**:
- Claudecode: Simple, fast, deterministic, less intelligent
- Victor: Complex, slower, intelligent, more context-aware

---

### 1.7 Merge Logic

**Claudecode** (lines 230-263):
```rust
fn merge_compact_summaries(existing_summary: Option<&str>, new_summary: &str) -> String {
    let Some(existing_summary) = existing_summary else {
        return new_summary.to_string();
    };

    let previous_highlights = extract_summary_highlights(existing_summary);
    let new_formatted_summary = format_compact_summary(new_summary);
    let new_highlights = extract_summary_highlights(&new_formatted_summary);
    let new_timeline = extract_summary_timeline(&new_formatted_summary);

    let mut lines = vec!["<summary>".to_string(), "Conversation summary:".to_string()];

    if !previous_highlights.is_empty() {
        lines.push("- Previously compacted context:".to_string());
        lines.extend(previous_highlights.into_iter().map(|line| format!("  {line}")));
    }

    if !new_highlights.is_empty() {
        lines.push("- Newly compacted context:".to_string());
        lines.extend(new_highlights.into_iter().map(|line| format!("  - {line}")));
    }

    if !new_timeline.is_empty() {
        lines.push("- Key timeline:".to_string());
        lines.extend(new_timeline.into_iter().map(|line| format!("  {line}")));
    }

    lines.push("</summary>".to_string());
    lines.join("\n")
}
```

**Key Points**:
- Separates "Previously compacted context" from "Newly compacted context"
- Keeps only the latest timeline (drops old timeline)
- Maintains chronological structure

**Victor** (from compaction_hierarchy.py):
```python
class HierarchicalCompactionManager:
    """Manages compaction summaries with hierarchical compression."""

    def add_summary(self, summary: str, turn_index: int) -> None:
        """Add a summary to the manager."""
        self._summaries.append(CompactionEpoch(
            start_index=self._last_index,
            end_index=turn_index,
            summary=summary,
            timestamp=datetime.now(tz=timezone.utc),
        ))
        self._last_index = turn_index

        # Check if we need epoch-level compression
        if len(self._summaries) > self._max_individual:
            self._compress_to_epochs()

    def _compress_to_epochs(self) -> None:
        """Compress older summaries into epoch-level summaries."""
        # Keep max_individual most recent summaries
        # Compress older summaries into epochs
        # Use LLM to generate epoch summaries
        pass

    def get_active_context(self) -> str:
        """Get the active context for injection."""
        # Combine recent summaries + epoch summaries
        # Return as formatted string
        pass
```

**Key Points**:
- Hierarchical compression (individual summaries → epoch summaries)
- Keeps max_individual most recent summaries
- Uses LLM to generate epoch summaries
- Prevents linear accumulation of summaries

**Comparison**:
- **Claudecode**: Flat merge (old + new), linear accumulation
- **Victor**: Hierarchical compression (individual → epochs), prevents accumulation

**Trade-off**:
- Claudecode: Simpler, linear accumulation (can get large over time)
- Victor: More complex, bounded size (hierarchical compression)

---

### 1.8 File Path Extraction

**Claudecode** (lines 321-335, 355-380):
```rust
fn collect_key_files(messages: &[ConversationMessage]) -> Vec<String> {
    let mut files = messages
        .iter()
        .flat_map(|message| message.blocks.iter())
        .map(|block| match block {
            ContentBlock::Text { text } => text.as_str(),
            ContentBlock::ToolUse { input, .. } => input.as_str(),
            ContentBlock::ToolResult { output, .. } => output.as_str(),
        })
        .flat_map(extract_file_candidates)
        .collect::<Vec<_>>();
    files.sort();
    files.dedup();
    files.into_iter().take(8).collect()
}

fn has_interesting_extension(candidate: &str) -> bool {
    std::path::Path::new(candidate)
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            ["rs", "ts", "tsx", "js", "json", "md"]
                .iter()
                .any(|expected| extension.eq_ignore_ascii_case(expected))
        })
}

fn extract_file_candidates(content: &str) -> Vec<String> {
    content
        .split_whitespace()
        .filter_map(|token| {
            let candidate = token.trim_matches(|char: char| {
                matches!(char, ',' | '.' | ':' | ';' | ')' | '(' | '"' | '\'' | '`')
            });
            if candidate.contains('/') && has_interesting_extension(candidate) {
                Some(candidate.to_string())
            } else {
                None
            }
        })
        .collect()
}
```

**Key Points**:
- Extracts files from text, tool_use input, tool_result output
- Filters by extension (rs, ts, tsx, js, json, md)
- Must contain '/' to be considered a file path
- Strips punctuation from tokens
- Limits to 8 files

**Victor** (from compaction_summarizer.py):
```python
class LedgerAwareCompactionSummarizer:
    """Uses SessionLedger for structured summaries."""

    def summarize(self, messages: List[Message]) -> str:
        """Generate summary using session ledger."""
        ledger = self._get_ledger()

        parts = []
        if ledger.files_read:
            parts.append(f"Files read: {', '.join(ledger.files_read)}")
        if ledger.files_modified:
            parts.append(f"Files modified: {', '.join(ledger.files_modified)}")

        return "\n".join(parts)
```

**Key Points**:
- Uses SessionLedger to track files
- Separates files_read from files_modified
- No limit on file count
- No extension filtering

**Comparison**:
- **Claudecode**: Regex-based extraction, extension filtering, limit 8
- **Victor**: Ledger-based tracking, no filtering, no limit

**Trade-off**:
- Claudecode: More precise (extension filtering), bounded (limit 8)
- Victor: More comprehensive (ledger tracking), unbounded

---

### 1.9 Pending Work Inference

**Claudecode** (lines 300-319):
```rust
fn infer_pending_work(messages: &[ConversationMessage]) -> Vec<String> {
    messages
        .iter()
        .rev()
        .filter_map(first_text_block)
        .filter(|text| {
            let lowered = text.to_ascii_lowercase();
            lowered.contains("todo")
                || lowered.contains("next")
                || lowered.contains("pending")
                || lowered.contains("follow up")
                || lowered.contains("remaining")
        })
        .take(3)
        .map(|text| truncate_summary(text, 160))
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect()
}
```

**Key Points**:
- Scans messages in reverse (most recent first)
- Looks for keywords: todo, next, pending, follow up, remaining
- Takes up to 3 matches
- Truncates to 160 chars

**Victor** (from compaction_summarizer.py):
```python
class LedgerAwareCompactionSummarizer:
    """Uses SessionLedger for structured summaries."""

    def summarize(self, messages: List[Message]) -> str:
        """Generate summary using session ledger."""
        ledger = self._get_ledger()

        parts = []
        if ledger.pending_actions:
            parts.append(f"Pending: {'; '.join(ledger.pending_actions)}")

        return "\n".join(parts)
```

**Key Points**:
- Uses SessionLedger to track pending actions
- Ledger is populated during conversation execution
- No keyword inference (explicit tracking)

**Comparison**:
- **Claudecode**: Keyword inference (todo, next, pending, etc.)
- **Victor**: Explicit tracking (ledger-based)

**Trade-off**:
- Claudecode: Works without explicit tracking, heuristic-based
- Victor: More accurate (explicit tracking), requires ledger population

---

### 1.10 Tests

**Claudecode** (lines 502-703, 8 tests):
```rust
#[cfg(test)]
mod tests {
    #[test]
    fn formats_compact_summary_like_upstream() { /* ... */ }

    #[test]
    fn leaves_small_sessions_unchanged() { /* ... */ }

    #[test]
    fn compacts_older_messages_into_a_system_summary() { /* ... */ }

    #[test]
    fn keeps_previous_compacted_context_when_compacting_again() { /* ... */ }

    #[test]
    fn ignores_existing_compacted_summary_when_deciding_to_recompact() { /* ... */ }

    #[test]
    fn truncates_long_blocks_in_summary() { /* ... */ }

    #[test]
    fn extracts_key_files_from_message_content() { /* ... */ }

    #[test]
    fn infers_pending_work_from_recent_messages() { /* ... */ }
}
```

**Test Coverage**:
- Summary formatting
- Small sessions (no compaction)
- Basic compaction
- Re-compaction with existing summary
- Summary exclusion from re-compaction decision
- Block truncation
- File extraction
- Pending work inference

**Victor** (estimated from test files):
- Multiple test files across modules
- Integration tests for compaction strategies
- Scoring tests
- Semantic retrieval tests
- Hierarchical compaction tests

**Comparison**:
- **Claudecode**: 8 comprehensive tests in one file
- **Victor**: 20+ tests across multiple files

---

## Part 2: What Victor Can Learn from Claudecode

### 2.1 Deterministic Fallback Strategy

**Current State**: Victor's default compaction is LLM-based (SEMANTIC/HYBRID strategies)

**Claudecode Advantage**: Deterministic, rule-based approach is:
- Faster (no API calls)
- Cheaper (no cost)
- Reproducible (same input → same output)
- Works offline

**Recommendation for Victor**:
```python
class RuleBasedCompactionSummarizer(CompactionSummaryStrategy):
    """Deterministic rule-based summarization (claudecode-style).

    Fallback strategy for resource-constrained environments or when
    LLM services are unavailable.
    """

    def summarize(self, messages: List[Message]) -> str:
        """Generate deterministic summary using rules."""
        # Count messages by role
        user_count = sum(1 for m in messages if m.role == "user")
        assistant_count = sum(1 for m in messages if m.role == "assistant")
        tool_count = sum(1 for m in messages if m.role == "tool")

        # Extract unique tool names
        tools = self._extract_tool_names(messages)

        # Extract file paths
        files = self._extract_file_paths(messages)

        # Infer pending work
        pending = self._infer_pending_work(messages)

        # Build structured summary (XML format for parseability)
        lines = ["<summary>", "Conversation summary:"]
        lines.append(f"- Scope: {len(messages)} earlier messages compacted "
                   f"(user={user_count}, assistant={assistant_count}, tool={tool_count}).")

        if tools:
            lines.append(f"- Tools mentioned: {', '.join(tools)}.")
        if files:
            lines.append(f"- Key files referenced: {', '.join(files)}.")
        if pending:
            lines.append("- Pending work:")
            lines.extend(f"  - {p}" for p in pending)

        lines.append("- Key timeline:")
        for msg in messages:
            role = msg.role
            content = self._truncate(msg.content, 160)
            lines.append(f"  - {role}: {content}")

        lines.append("</summary>")
        return "\n".join(lines)
```

**Benefits**:
- Fast, cheap, deterministic
- Claudecode-compatible format
- Fallback for LLM failures

**Implementation Effort**: 2-3 days

---

### 2.2 Structured XML Format

**Current State**: Victor's summaries are natural language (variable structure)

**Claudecode Advantage**: XML format is:
- Machine-readable
- Parseable
- Consistent structure
- Easy to extract specific fields

**Recommendation for Victor**:
```python
def format_summary_xml(summary: Dict[str, Any]) -> str:
    """Format summary as XML (claudecode-compatible)."""
    lines = ["<summary>", "Conversation summary:"]

    if "scope" in summary:
        lines.append(f"- Scope: {summary['scope']}.")
    if "tools" in summary:
        lines.append(f"- Tools mentioned: {', '.join(summary['tools'])}.")
    if "files" in summary:
        lines.append(f"- Key files referenced: {', '.join(summary['files'])}.")
    if "pending" in summary:
        lines.append("- Pending work:")
        lines.extend(f"  - {p}" for p in summary["pending"])
    if "timeline" in summary:
        lines.append("- Key timeline:")
        lines.extend(f"  - {e}" for e in summary["timeline"])

    lines.append("</summary>")
    return "\n".join(lines)
```

**Benefits**:
- Parseable by external tools
- Consistent structure
- Claudecode-compatible

**Implementation Effort**: 1 day

---

### 2.3 File Path Extraction

**Current State**: Victor uses ledger-based tracking (no extraction from text)

**Claudecode Advantage**: Regex-based extraction:
- Works without explicit tracking
- No need to populate ledger
- Extension filtering reduces false positives

**Recommendation for Victor**:
```python
def extract_file_paths(messages: List[Message]) -> List[str]:
    """Extract file paths from messages (claudecode-style)."""
    import re

    FILE_EXTENSIONS = {"rs", "ts", "tsx", "js", "json", "md", "py", "go", "java"}

    files = set()
    for msg in messages:
        # Extract from content
        tokens = msg.content.split()
        for token in tokens:
            # Strip punctuation
            clean = token.strip(",.:;()\"'`")
            # Check if it looks like a file path
            if "/" in clean:
                ext = clean.split(".")[-1].lower()
                if ext in FILE_EXTENSIONS:
                    files.add(clean)

    return sorted(files)[:8]  # Limit to 8 files
```

**Benefits**:
- Works without ledger
- Extension filtering reduces noise
- Compatible with claudecode approach

**Implementation Effort**: 1 day

---

### 2.4 Pending Work Inference

**Current State**: Victor uses explicit ledger tracking

**Claudecode Advantage**: Keyword-based inference:
- Works without explicit tracking
- Covers cases where ledger isn't populated

**Recommendation for Victor**:
```python
def infer_pending_work(messages: List[Message]) -> List[str]:
    """Infer pending work from recent messages (claudecode-style)."""
    keywords = ["todo", "next", "pending", "follow up", "remaining"]

    pending = []
    for msg in reversed(messages):  # Most recent first
        content_lower = msg.content.lower()
        if any(kw in content_lower for kw in keywords):
            # Extract the sentence containing the keyword
            pending.append(msg.content[:160])  # Truncate to 160 chars
            if len(pending) >= 3:  # Limit to 3 items
                break

    return list(reversed(pending))  # Return in chronological order
```

**Benefits**:
- Works without ledger
- Complements explicit tracking
- Claudecode-compatible

**Implementation Effort**: 1 day

---

## Part 3: What Claudecode Could Learn from Victor

### 3.1 Hierarchical Compression

**Current State**: Claudecode merges summaries linearly (old + new)

**Victor Advantage**: Hierarchical compression:
- Prevents linear accumulation
- Bounded memory usage
- Epoch-level summaries

**Recommendation for Claudecode**:
```rust
struct HierarchicalCompactor {
    max_individual_summaries: usize,
    epochs: Vec<EpochSummary>,
}

impl HierarchicalCompactor {
    fn add_summary(&mut self, summary: String, turn_index: usize) {
        self.summaries.push(IndividualSummary { summary, turn_index });

        // Compress to epochs if too many individual summaries
        if self.summaries.len() > self.max_individual_summaries {
            self.compress_to_epochs();
        }
    }

    fn compress_to_epochs(&mut self) {
        // Compress older summaries into epoch-level summaries
        // Use LLM or rule-based approach for epoch compression
    }
}
```

**Benefits**:
- Bounded memory usage
- Better for long conversations
- Prevents summary bloat

---

### 3.2 Content-Type-Aware Token Estimation

**Current State**: Claudecode uses simple len/4 for all content

**Victor Advantage**: Content-type-aware estimation:
- More accurate
- Different factors for code, json, prose

**Recommendation for Claudecode**:
```rust
fn estimate_message_tokens(message: &ConversationMessage) -> usize {
    message.blocks.iter().map(|block| {
        let (text, content_type) = match block {
            ContentBlock::Text { text } => (text, detect_content_type(text)),
            ContentBlock::ToolUse { name, input, .. } => {
                (&format!("{name} {input}"), "code")
            }
            ContentBlock::ToolResult { tool_name, output, .. } => {
                (&format!("{tool_name} {output}"), "mixed")
            }
        };

        let factor = match content_type {
            "code" => 3.0,
            "json" => 2.8,
            "prose" => 4.0,
            _ => 3.5,
        };

        (text.len() as f64 / factor).ceil() as usize
    }).sum()
}

fn detect_content_type(text: &str) -> &str {
    if text.contains('{') && text.contains('}') {
        "json"
    } else if text.contains("fn ") || text.contains("def ") || text.contains("class ") {
        "code"
    } else {
        "prose"
    }
}
```

**Benefits**:
- More accurate token estimation
- Better compaction decisions

---

### 3.3 Multiple Compaction Strategies

**Current State**: Claudecode has one deterministic strategy

**Victor Advantage**: Multiple strategies (SIMPLE, TIERED, SEMANTIC, HYBRID)

**Recommendation for Claudecode**:
```rust
enum CompactionStrategy {
    Simple,
    Tiered,  // Prioritize tool results
    Semantic,  // Use embeddings (if available)
}

fn compact_session_with_strategy(
    session: &Session,
    config: CompactionConfig,
    strategy: CompactionStrategy,
) -> CompactionResult {
    match strategy {
        CompactionStrategy::Simple => compact_session_simple(session, config),
        CompactionStrategy::Tiered => compact_session_tiered(session, config),
        CompactionStrategy::Semantic => compact_session_semantic(session, config),
    }
}

fn compact_session_tiered(session: &Session, config: CompactionConfig) -> CompactionResult {
    // Prioritize tool results over discussion messages
    // Keep all tool results, drop low-priority assistant messages
}

fn compact_session_semantic(session: &Session, config: CompactionConfig) -> CompactionResult {
    // Use embeddings to keep task-relevant messages
    // Requires embedding service
}
```

**Benefits**:
- More flexible compaction
- Adaptable to different scenarios
- User-selectable strategy

---

### 3.4 Rust-Accelerated Scoring

**Current State**: Claudecode has no scoring (deterministic selection)

**Victor Advantage**: Rust-accelerated batch scoring (4.8-9.9x speedup)

**Recommendation for Claudecode**:
```rust
// Add optional scoring for more intelligent compaction
#[cfg(feature = "semantic")]
fn score_messages(messages: &[ConversationMessage], query: &str) -> Vec<f32> {
    // Use Rust-native embeddings library (e.g., candle)
    // Return similarity scores
}

#[cfg(feature = "semantic")]
fn compact_session_semantic(session: &Session, config: CompactionConfig, query: &str) -> CompactionResult {
    let scores = score_messages(&session.messages, query);
    // Keep messages with highest scores
    // ...
}
```

**Benefits**:
- More intelligent compaction
- Optional feature (can be disabled)
- Fast (Rust-native)

---

## Part 4: Conclusion

### Key Findings

1. **Different Design Philosophies**:
   - **Claudecode**: Simple, fast, cheap, deterministic
   - **Victor**: Complex, intelligent, flexible, expensive

2. **Claudecode Strengths**:
   - Deterministic (reproducible)
   - Fast (no API calls)
   - Cheap (zero cost)
   - Simple (703 lines Rust)
   - Machine-readable (XML format)

3. **Victor Strengths**:
   - Intelligent (multi-factor scoring)
   - Flexible (multiple strategies)
   - Rich (LLM-based summaries)
   - Sophisticated (hierarchical compression)
   - Accurate (content-type-aware estimation)

4. **Best of Both Worlds**:
   - **Victor should add**: Deterministic fallback, XML format option
   - **Claudecode could add**: Hierarchical compression, content-type-aware estimation

### Recommendations for Victor

1. **Add `RuleBasedCompactionSummarizer`** (2-3 days):
   - Deterministic fallback for resource-constrained environments
   - Claudecode-compatible XML format
   - Fast, cheap, reproducible

2. **Add XML format option** (1 day):
   - Machine-readable output
   - Parseable by external tools
   - Consistent structure

3. **Enhance file extraction** (1 day):
   - Regex-based extraction from text
   - Extension filtering
   - Works without ledger

4. **Add pending work inference** (1 day):
   - Keyword-based inference
   - Complements explicit tracking
   - Works without ledger

**Total Effort**: 5-7 days for all enhancements

**Risk**: Low (additive changes, no breaking changes)

**Benefits**:
- Claudecode compatibility
- Deterministic fallback
- Faster compaction (no API calls)
- Zero-cost option

---

**Status**: ✅ **Direct comparison complete. Ready to proceed with enhancements.**
