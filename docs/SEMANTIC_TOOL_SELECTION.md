# Semantic Tool Selection

Victor supports **embedding-based semantic tool selection** as an alternative to keyword matching.

## Overview

### Keyword-Based (Default)
```python
# Hardcoded keyword matching
if "test" in message or "pytest" in message:
    add_testing_tools()
```

**Problems:**
- Misses synonyms ("verify", "validate", "check")
- Brittle, requires maintenance
- No semantic understanding
- Language-dependent

### Embedding-Based (Semantic)
```python
# Compute semantic similarity
query_embedding = embed(user_message)
tool_embeddings = [embed(tool) for tool in tools]
similar_tools = top_k_similar(query_embedding, tool_embeddings)
```

**Benefits:**
- ✅ Handles synonyms automatically
- ✅ Semantic understanding of intent
- ✅ No hardcoded keywords
- ✅ Language-agnostic
- ✅ Self-improving with better tool descriptions

## How It Works

### 1. Initialization (One-Time)

On first use, Victor pre-computes embeddings for all tools:

```python
for tool in tools:
    tool_text = f"{tool.name} {tool.description} {tool.parameters}"
    embedding = embedding_model.embed(tool_text)
    cache[tool.name] = embedding  # Store for fast lookup
```

**Example Tool Embeddings:**
```
read_file → [0.23, -0.41, 0.89, ..., 0.12]  (768-dim vector)
git_commit → [-0.12, 0.67, -0.34, ..., 0.45]
security_scan → [0.56, -0.23, 0.78, ..., -0.11]
```

### 2. Query Time

When user makes a request:

```python
# 1. Embed user message
query = "Write tests for authentication module"
query_embedding = embedding_model.embed(query)  # [0.12, -0.67, 0.34, ...]

# 2. Calculate cosine similarity with all tools
similarities = {
    tool_name: cosine_similarity(query_embedding, tool_embedding)
    for tool_name, tool_embedding in cache.items()
}

# 3. Rank by similarity
ranked_tools = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

# 4. Return top-K above threshold
selected = [tool for tool, score in ranked_tools if score > 0.3][:max_tools]
```

### 3. Results

**Query:** "Write tests for authentication module"

**Selected Tools (with similarity scores):**
```
run_tests (0.87) ← High similarity to "tests"
write_file (0.72) ← Writing test files
read_file (0.68) ← Reading auth module
security_scan (0.65) ← Authentication security
code_review (0.61) ← Reviewing auth code
...
```

## Configuration

### Enable Semantic Selection

**Option 1: Environment Variable**
```bash
export USE_SEMANTIC_TOOL_SELECTION=true
victor
```

**Option 2: Configuration File**
```yaml
# ~/.victor/config.yaml
use_semantic_tool_selection: true
```

**Option 3: Programmatic**
```python
from victor.config.settings import Settings

settings = Settings()
settings.use_semantic_tool_selection = True
```

### Embedding Model Selection

Default: `nomic-embed-text` (768-dim, optimized for semantic search)

**Change Model:**
```python
from victor.tools.semantic_selector import SemanticToolSelector

selector = SemanticToolSelector(
    embedding_model="nomic-embed-text",  # or "all-minilm-l6-v2", "qwen3-embedding:8b"
    embedding_provider="ollama",  # or "openai", "sentence-transformers"
    cache_embeddings=True
)
```

## Comparison

### Example 1: Testing

**Query:** "verify the authentication module works correctly"

**Keyword-Based:**
- Looks for: "test", "pytest", "unittest"
- ❌ Misses query (no keyword match)
- Result: No testing tools selected

**Semantic-Based:**
- Understands: "verify" ≈ "test", "validate"
- ✅ Matches testing tools semantically
- Result: `run_tests` (0.82 similarity)

### Example 2: Security

**Query:** "scan for vulnerabilities and secrets"

**Keyword-Based:**
- Looks for: "security", "vulnerability", "secret", "scan"
- ✅ Matches all keywords
- Result: All 5 security tools

**Semantic-Based:**
- Understands: "vulnerabilities" ≈ "security issues", "scan" ≈ "check", "analyze"
- ✅ Matches security tools semantically
- Result: `security_scan` (0.91 similarity) - consolidated tool

### Example 3: Git Operations

**Query:** "save my changes with a good commit message"

**Keyword-Based:**
- Looks for: "git", "commit", "branch"
- ✅ Matches "commit"
- Result: git_commit, git_stage

**Semantic-Based:**
- Understands: "save changes" ≈ "commit", "good message" ≈ "AI suggest"
- ✅ Better matching
- Result: git_commit (0.88), git_suggest_commit (0.85), git_stage (0.79)

## Performance

### Initialization Cost (One-Time)

| Tools | Embedding Time | Cache Size |
|-------|----------------|------------|
| 86 tools (original) | ~3-5 seconds | ~260KB |
| 31 tools (consolidated) | <1 second | ~95KB |

**Note:** Only happens once per session, embeddings cached in memory. Tool consolidation from 86 → 31 significantly reduced initialization time and memory usage.

### Query Time

| Operation | Time |
|-----------|------|
| Embed query | ~50-100ms |
| Compute similarities (31 tools) | <1ms |
| Sort & select | <1ms |
| **Total** | **~50-100ms overhead** |

**Impact:** Negligible compared to LLM inference (5-30+ seconds). Tool consolidation reduced similarity computation overhead by 64%.

## Advanced Features

### Custom Similarity Threshold

```python
# More selective (higher threshold)
tools = await selector.select_relevant_tools(
    user_message,
    similarity_threshold=0.5  # Only very similar tools
)

# More permissive (lower threshold)
tools = await selector.select_relevant_tools(
    user_message,
    similarity_threshold=0.2  # Include loosely related tools
)
```

### Hybrid Selection

Combine semantic + keyword for best results:

```python
semantic_tools = await semantic_selector.select_relevant_tools(message)
keyword_tools = keyword_selector.select_relevant_tools(message)

# Union: tools from either method
all_tools = set(semantic_tools) | set(keyword_tools)

# Intersection: tools from both methods (high confidence)
confident_tools = set(semantic_tools) & set(keyword_tools)
```

### Tool Ranking by Confidence

```python
tools_with_scores = await selector.select_relevant_tools_with_scores(message)

for tool, similarity in tools_with_scores[:10]:
    print(f"{tool.name}: {similarity:.3f}")

# Output:
# run_tests: 0.872
# write_file: 0.721
# read_file: 0.684
# ...
```

## Migration Guide

### From Keyword to Semantic

**Before (Keyword):**
```python
def _select_relevant_tools(self, message: str):
    if "test" in message.lower():
        return testing_tools
    if "git" in message.lower():
        return git_tools
    # ... 20 more categories
```

**After (Semantic):**
```python
async def _select_relevant_tools(self, message: str):
    return await self.semantic_selector.select_relevant_tools(
        message,
        max_tools=10,
        similarity_threshold=0.3
    )
```

**Benefits:**
- 80% less code
- No keyword maintenance
- Better accuracy
- Automatic improvement as models improve

## Embedding Models

### Recommended Models

| Model | Provider | Dimensions | Use Case |
|-------|----------|------------|----------|
| nomic-embed-text | Ollama | 768 | General purpose, fast |
| qwen3-embedding:8b | Ollama | 4096 | High accuracy, slower |
| all-MiniLM-L6-v2 | Sentence-Transformers | 384 | Fast, good enough |
| text-embedding-3-small | OpenAI | 1536 | Cloud, high quality |

### Performance vs Quality

```
nomic-embed-text (768-dim)    ████████░░ 80% quality, 100% speed
qwen3-embedding:8b (4096-dim) ██████████ 100% quality, 60% speed
all-MiniLM-L6-v2 (384-dim)    ███████░░░ 70% quality, 150% speed
```

## Debugging

### Enable Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("victor.tools.semantic_selector")
```

**Log Output:**
```
2025-11-26 20:00:00 - INFO - Initializing tool embeddings for 31 tools
2025-11-26 20:00:01 - INFO - Tool embeddings initialized
2025-11-26 20:00:03 - INFO - Selected 8 tools by semantic similarity:
  run_tests(0.872), write_file(0.721), read_file(0.684), ...
```

### Similarity Matrix Visualization

```python
import matplotlib.pyplot as plt

# Get all similarities
similarities = await selector.get_all_similarities(user_message, tools)

# Plot heatmap
plt.imshow(similarities, cmap='viridis')
plt.colorbar(label='Similarity')
plt.xlabel('Tools')
plt.ylabel('Query')
plt.show()
```

## Limitations

1. **Requires embedding model**: Needs Ollama/OpenAI/etc running
2. **Initialization overhead**: 1-5 seconds on first use
3. **Memory usage**: ~125-260KB for cached embeddings
4. **Not deterministic**: Embeddings can vary slightly
5. **Model dependency**: Quality depends on embedding model

## Future Enhancements

1. **Multi-language embeddings**: Support for non-English queries
2. **Fine-tuned embeddings**: Train custom embeddings on Victor's tool usage
3. **Dynamic re-ranking**: Use LLM to re-rank selected tools
4. **User feedback loop**: Learn from which tools users actually use
5. **Hierarchical clustering**: Group similar tools for better selection

---

**Conclusion:**

Embedding-based semantic tool selection is **more robust, accurate, and maintainable** than keyword matching. While it adds ~100ms overhead and requires an embedding model, the benefits far outweigh the costs for production deployments.

**Recommended:** Enable for production. Keep keyword-based as fallback.
