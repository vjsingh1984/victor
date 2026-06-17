# Graph-Based Code Intelligence: Research Quick Reference

## Core Papers at a Glance

### 1. GraphCoder (2024)
**arXiv:2406.07003** | Liu et al., Peking University | Score: 0.708

**Key Innovation**: Code Context Graph (CCG)
- Superimposes 3 graphs: CFG + CDG + DDG
- Statement-level granularity (not just function-level)
- Coarse-to-fine retrieval (sequence → structure)
- Decay-with-distance weighting

**Results**: +6.06 code match, +6.23 identifier match vs baseline

**Code Snippet**:
```python
# CCG Edge Types
EDGE_CF = "control_flow"   # Sequential execution (CFG)
EDGE_CD = "control_dep"    # Control dependence (CDG)
EDGE_DD = "data_dep"       # Data dependence (DDG)

# CCG Slicing Algorithm
def ccg_slice(graph, target, max_hops=3, max_nodes=50):
    """Extract h-hop subgraph with context statements."""
    # BFS on CF edges + include DD/CD neighbors
```

**Applicable to Victor**: HIGH
- Enhances code completion with statement-level context
- Builds on existing Tree-sitter infrastructure

---

### 2. CodexGraph (2024)
**arXiv:2408.03910** | Liu et al., Alibaba/NUS | Score: 0.703

**Key Innovation**: Graph Database Interface for LLMs
- Neo4j-based code graph storage
- "Write then translate" pattern (NL → Cypher)
- Unified schema for all repo-level tasks
- Iterative pipeline (multi-round retrieval)

**Schema**:
```cypher
// Node Types
MODULE, CLASS, METHOD, FUNCTION, FIELD, GLOBAL_VARIABLE

// Edge Types
CONTAINS, INHERITS, HAS_METHOD, HAS_FIELD, USES

// Example Query
MATCH (c:Class {name: 'LinearClassifier'})<-[:CONTAINS]-(m:Module)
MATCH (c)-[:CONTAINS]->(f:Function)
RETURN m.name, collect(f.name)
```

**Results**: 47.99% on SWE-bench Lite (GPT-4o)

**Applicable to Victor**: HIGH
- Natural language to graph query translation
- Graph database integration (ProximaDB/Neo4j)

---

### 3. GraphCodeAgent (2025)
**arXiv:2504.10046** | Li et al., Wuhan/Peking University | Score: 0.691

**Key Innovation**: Dual Graph Architecture
- **Requirement Graph (RG)**: Models requirement relations
- **Structural-Semantic Code Graph (SSCG)**: Code dependencies
- Mapping between RG and SSCG
- Multi-hop reasoning for implicit dependencies

**Workflow**:
```
User Requirement
    ↓
Requirement Graph (find subrequirements, similar requirements)
    ↓
Map to Code Symbols (predefined APIs)
    ↓
Multi-hop Traversal (find dependencies of dependencies)
    ↓
Web Search (up-to-date knowledge)
    ↓
Generate Code
```

**Results**: +43.81% on DevEval (GPT-4o)

**Applicable to Victor**: HIGH
- Bridges NL requirements to code implementation
- Multi-hop dependency retrieval

---

### 4. CGM - Code Graph Model (2025)
**arXiv:2505.16901** | Tao et al., Ant Group | Score: 0.697

**Key Innovation**: Graph-Integrated LLM
- **Semantic Integration**: Node attributes → LLM input space via adapter
- **Structural Integration**: Graph structure → attention mask (GNN-style)
- Agentless Graph RAG: Rewriter → Retriever → Reranker → Reader

**Architecture**:
```
Code Graph (Nodes + Edges)
    ↓
Text Encoder (Frozen) → Adapter → LLM Input Space
    ↓
Attention Mask (Graph-aware)
    ↓
LoRA Fine-tuning
```

**Results**: 43.00% on SWE-bench Lite (Qwen2.5-72B, open-source SOTA)

**Applicable to Victor**: MEDIUM
- Requires model training/fine-tuning
- Graph-aware attention is complex

---

### 5. GraphRAG Survey (2024)
**arXiv:2408.08921** | Peng et al., Peking University | Score: 0.692

**Key Innovation**: Systematic Framework
- **G-Indexing**: Build graph from data
- **G-Retrieval**: Query → Graph elements
- **G-Generation**: Graph → Response

**Comparison**:
| Aspect | RAG | GraphRAG |
|--------|-----|----------|
| Data | Text snippets | Graph (nodes, edges, paths) |
| Retrieval | Semantic similarity | Structure + semantics |
| Context | Linear | Relational |
| Length | Can be verbose | Abstracted/compact |

**Applicable to Victor**: HIGH
- Provides systematic framework for implementation

---

## Technique Comparison Matrix

| Technique | Paper | Complexity | Impact | Victor Fit |
|-----------|-------|-----------|--------|------------|
| Code Context Graph (CCG) | GraphCoder | Medium | High | ✅ Existing Tree-sitter |
| Graph Query Translation | CodexGraph | Low | High | ✅ Add query templates |
| Multi-hop Traversal | GraphCodeAgent | Medium | High | ✅ Extend get_neighbors |
| Requirement Graph | GraphCodeAgent | High | Medium | ⚠️ Requires mapping |
| Graph-Aware Attention | CGM | Very High | Medium | ❌ Model training |
| Graph RAG Framework | Survey | Low | High | ✅ Architectural |

---

## Quick Implementation Checklist

### Immediate Wins (Low-Hanging Fruit)

- [ ] **Extend edge types** in `GraphEdge`: Add `control_flow`, `control_dep`, `data_dep`
- [ ] **Add query templates** for common graph patterns (find callers, trace dependencies)
- [ ] **Implement multi-hop** wrapper around existing `get_neighbors()`
- [ ] **Add subgraph extraction** with node limits and edge filtering
- [ ] **Enhance init.md** with graph context (call chains, dependencies)

### Medium-Term Enhancements

- [ ] **Build CCG** for Python (CFG from AST)
- [ ] **Implement CDG/DDG** computation
- [ ] **Create NL→Query translator** using LLM
- [ ] **Add graph embeddings** (structure-aware)
- [ ] **Integrate ProximaDB** ORION engine

### Long-Term Research

- [ ] **Requirement Graph** construction and mapping
- [ ] **Graph-aware LLM** fine-tuning (LoRA)
- [ ] **Multi-modal graph** (code + docs + issues)

---

## Key Algorithms

### CCG Slice (GraphCoder)
```python
def ccg_slice(graph, target, max_hops=3, max_nodes=50):
    """
    Extract context subgraph for code completion.

    1. Start from target statement
    2. BFS on control_flow edges (max_hops)
    3. Include data_dep and control_dep neighbors
    4. Limit to max_nodes
    5. Return ordered by line number
    """
    visited = set()
    frontier = {target}
    result = []

    for hop in range(max_hops):
        # Add CF neighbors
        cf_neighbors = get_neighbors(frontier, "control_flow")
        # Add their DD/CD dependencies
        for node in cf_neighbors:
            dd_neighbors = get_neighbors(node, "data_dep", direction="in")
            cd_neighbors = get_neighbors(node, "control_dep", direction="in")
            result.extend(dd_neighbors + cd_neighbors)

        frontier = set(cf_neighbors)
        if len(result) > max_nodes:
            break

    return result
```

### Multi-Hop Retrieval (GraphCodeAgent)
```python
async def multi_hop_retrieve(graph, start, query, max_hops=3):
    """
    Retrieve implicit dependencies via multi-hop traversal.

    1. Hop 1: Direct dependencies (CALLS, REFERENCES)
    2. Hop 2+: Transitive dependencies
    3. Re-rank by semantic similarity to query
    4. Apply decay-with-distance weighting
    """
    results = {0: [start]}
    scores = {start: 1.0}

    for hop in range(1, max_hops + 1):
        frontier = results[hop - 1]
        hop_results = []

        for node in frontier:
            neighbors = await graph.get_neighbors(
                node,
                edge_types=["CALLS", "REFERENCES", "USES"],
                direction="out"
            )
            for edge in neighbors:
                neighbor = edge.dst
                # Decay score
                score = scores[node] * 0.5
                # Boost by semantic similarity
                score *= semantic_similarity(neighbor, query)
                hop_results.append((neighbor, score))

        # Re-rank and top-k
        hop_results.sort(key=lambda x: x[1], reverse=True)
        results[hop] = [n for n, s in hop_results[:10]]

    return results
```

---

## Citation Summary

For implementation reference:
- **GraphCoder**: Statement-level graph for completion context
- **CodexGraph**: Graph database + query translation
- **GraphCodeAgent**: Dual graphs + multi-hop reasoning
- **CGM**: Graph-integrated LLM (advanced, requires training)
- **GraphRAG Survey**: Systematic framework

---

**Last Updated**: 2025-04-28
**Maintained by**: Victor Architecture Team
