# Graph RAG API Reference

## victor.core.graph_rag

### GraphIndexingPipeline

```python
class GraphIndexingPipeline:
    """Build graph with embeddings for RAG."""

    def __init__(
        self,
        graph_store: GraphStore,
        config: GraphIndexConfig,
    ) -> None: ...

    async def index_repository(
        self,
    ) -> GraphIndexStats: ...

    async def index_file(
        self,
        file_path: Path,
    ) -> List[GraphNode]: ...

    async def build_embeddings(
        self,
        nodes: List[GraphNode],
    ) -> None: ...
```

### GraphIndexConfig

```python
@dataclass
class GraphIndexConfig:
    """Configuration for graph indexing."""

    root_path: Path
    enable_ccg: bool = True
    enable_embeddings: bool = True
    ccg_languages: List[str] = field(
        default_factory=lambda: ["python", "javascript", "typescript"]
    )
    embedding_batch_size: int = 32
    chunk_size: int = 500
    chunk_overlap: int = 50
```

### GraphIndexStats

```python
@dataclass
class GraphIndexStats:
    """Statistics from graph indexing."""

    files_processed: int
    nodes_created: int
    edges_created: int
    ccg_nodes_created: int
    ccg_edges_created: int
    error_count: int
    duration_seconds: float
```

### MultiHopRetriever

```python
class MultiHopRetriever:
    """Multi-hop traversal for context retrieval."""

    def __init__(
        self,
        graph_store: GraphStore,
        config: RetrievalConfig,
    ) -> None: ...

    async def retrieve(
        self,
        query: str,
        config: RetrievalConfig | None = None,
    ) -> GraphQueryResult: ...

    async def retrieve_by_seed(
        self,
        seed_ids: List[str],
        config: RetrievalConfig | None = None,
    ) -> GraphQueryResult: ...

    async def find_shortest_path(
        self,
        source: str,
        target: str,
    ) -> List[str] | None: ...
```

### RetrievalConfig

```python
@dataclass
class RetrievalConfig:
    """Configuration for multi-hop retrieval."""

    seed_count: int = 5
    max_hops: int = 2
    top_k: int = 10
    edge_types: List[str] | None = None
    centrality_weight: float = 0.2
    size_penalty_weight: float = 0.01
    use_subgraph_cache: bool = True
```

### GraphAwarePromptBuilder

```python
class GraphAwarePromptBuilder:
    """Construct prompts with graph context."""

    def build_prompt(
        self,
        query: str,
        subgraphs: List[Subgraph],
        config: PromptConfig,
    ) -> str: ...

    def format_subgraph_hierarchical(
        self,
        subgraph: Subgraph,
    ) -> str: ...

    def format_subgraph_flat(
        self,
        subgraph: Subgraph,
    ) -> str: ...

    def format_subgraph_compact(
        self,
        subgraph: Subgraph,
    ) -> str: ...
```

### PromptConfig

```python
@dataclass
class PromptConfig:
    """Configuration for prompt building."""

    format: str = "hierarchical"  # hierarchical, flat, compact
    include_edges: bool = True
    include_metadata: bool = False
    max_nodes_per_subgraph: int = 50
    context_budget: int = 4000  # tokens
```

## victor.core.indexing.ccg_builder

### CodeContextGraphBuilder

```python
class CodeContextGraphBuilder:
    """Build CFG, CDG, and DDG from source code."""

    def __init__(
        self,
        graph_store: GraphStore,
        language: str = "python",
    ) -> None: ...

    async def build_ccg_for_file(
        self,
        file_path: Path,
    ) -> Tuple[List[GraphNode], List[GraphEdge]]: ...

    async def build_cfg(
        self,
        ast_root: Any,
        file_path: Path,
    ) -> Tuple[List[GraphNode], List[GraphEdge]]: ...

    async def build_cdg(
        self,
        cfg_nodes: List[GraphNode],
        cfg_edges: List[GraphEdge],
    ) -> List[GraphEdge]: ...

    async def build_ddg(
        self,
        ast_root: Any,
        cfg_nodes: List[GraphNode],
        file_path: Path,
    ) -> List[GraphEdge]: ...
```

### StatementType

```python
class StatementType(str, Enum):
    """Statement type categories."""

    CONDITION = "condition"
    LOOP = "loop"
    TRY = "try"
    CATCH = "catch"
    FINALLY = "finally"
    SWITCH = "switch"
    CASE = "case"
    DEFAULT = "default"
    ASSIGNMENT = "assignment"
    CALL = "call"
    RETURN = "return"
    YIELD = "yield"
    AWAIT = "await"
    THROW = "throw"
    FUNCTION_DEF = "function_def"
    CLASS_DEF = "class_def"
    VARIABLE_DEF = "variable_def"
    BLOCK = "block"
    EXPRESSION = "expression"
    UNKNOWN = "unknown"
```

### BasicBlock

```python
@dataclass
class BasicBlock:
    """Basic block in CFG."""

    block_id: str
    entry_line: int
    exit_line: int
    statements: List[str] = field(default_factory=list)
    successors: List[str] = field(default_factory=list)
    predecessors: List[str] = field(default_factory=list)
    is_loop_body: bool = False
    is_conditional: bool = False
    loop_type: str | None = None
    condition: str | None = None
```

### VariableInfo

```python
@dataclass
class VariableInfo:
    """Variable information for DDG."""

    name: str
    defining_node: str
    scope: str
    type: str | None = None
    use_sites: List[str] = field(default_factory=list)
```

## victor.storage.graph

### GraphStore Protocol

```python
class GraphStore(Protocol):
    """Graph storage interface."""

    async def initialize(self) -> None: ...

    async def close(self) -> None: ...

    async def upsert_nodes(
        self,
        nodes: List[GraphNode],
    ) -> None: ...

    async def upsert_edges(
        self,
        edges: List[GraphEdge],
    ) -> None: ...

    async def get_node_by_id(
        self,
        node_id: str,
    ) -> GraphNode | None: ...

    async def get_nodes_by_id(
        self,
        node_ids: List[str],
    ) -> List[GraphNode]: ...

    async def get_nodes_by_file(
        self,
        file_path: str,
    ) -> List[GraphNode]: ...

    async def get_neighbors(
        self,
        node_id: str,
        direction: str = "out",  # out, in, both
        edge_types: List[str] | None = None,
        max_depth: int = 1,
    ) -> List[GraphEdge]: ...

    async def search_symbols(
        self,
        query: str,
        limit: int = 10,
    ) -> List[GraphNode]: ...

    async def find_nodes(
        self,
        name: str | None = None,
        type: str | None = None,
        file: str | None = None,
    ) -> List[GraphNode]: ...

    async def get_all_nodes(self) -> List[GraphNode]: ...

    async def get_all_edges(self) -> List[GraphEdge]: ...

    async def stats(self) -> Dict[str, int]: ...
```

### GraphNode

```python
@dataclass
class GraphNode:
    """Node in the code graph."""

    # Core fields
    node_id: str
    type: str
    name: str
    file: str
    line: int | None = None
    end_line: int | None = None

    # Optional metadata
    lang: str | None = None
    signature: str | None = None
    docstring: str | None = None
    parent_id: str | None = None
    embedding_ref: str | None = None

    # CCG fields (v5)
    ast_kind: str | None = None
    scope_id: str | None = None
    statement_type: str | None = None
    requirement_id: str | None = None
    visibility: str | None = None

    # Extended metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### GraphEdge

```python
@dataclass
class GraphEdge:
    """Edge in the code graph."""

    src: str
    dst: str
    type: str
    weight: float | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

## victor.storage.graph.edge_types

### EdgeType

```python
class EdgeType(str, Enum):
    """Edge type constants."""

    # Legacy edges
    CALLS = "CALLS"
    REFERENCES = "REFERENCES"
    CONTAINS = "CONTAINS"
    INHERITS = "INHERITS"
    IMPLEMENTS = "IMPLEMENTS"

    # CFG edges
    CFG_SUCCESSOR = "CFG_SUCCESSOR"
    CFG_TRUE_BRANCH = "CFG_TRUE"
    CFG_FALSE_BRANCH = "CFG_FALSE"
    CFG_CASE = "CFG_CASE"
    CFG_DEFAULT = "CFG_DEFAULT"

    # CDG edges
    CDG = "CDG"

    # DDG edges
    DDG_DEF_USE = "DDG_DEF_USE"
    DDG_RAW = "DDG_RAW"
    DDG_WAR = "DDG_WAR"
    DDG_WAW = "DDG_WAW"

    # Requirement edges
    SATISFIES = "SATISFIES"
    TESTS = "TESTS"
    DERIVES_FROM = "DERIVES_FROM"

    # Semantic edges
    SEMANTIC_SIMILAR = "SEMANTIC_SIM"

    @classmethod
    def is_cfg_edge(cls, edge_type: str) -> bool: ...

    @classmethod
    def is_cdg_edge(cls, edge_type: str) -> bool: ...

    @classmethod
    def is_ddg_edge(cls, edge_type: str) -> bool: ...

    @classmethod
    def is_requirement_edge(cls, edge_type: str) -> bool: ...
```

## victor.tools.graph_query_tool

### graph_query

```python
@tool
async def graph_query(
    query: str,
    path: str = ".",
    mode: str = "semantic",
    max_hops: int = 2,
    max_results: int = 10,
) -> Dict[str, Any]:
    """Query codebase graph using natural language.

    Args:
        query: Natural language query
        path: Path to search within
        mode: Query mode (semantic, structural, hybrid)
        max_hops: Maximum hops for graph traversal
        max_results: Maximum results to return

    Returns:
        Dict with nodes, edges, query, and metadata
    """
```

### impact_analysis

```python
@tool
async def impact_analysis(
    target: str,
    analysis_type: str = "forward",
    max_depth: int = 3,
    path: str = ".",
) -> Dict[str, Any]:
    """Analyze impact of code changes using CCG.

    Args:
        target: Target symbol or file:line
        analysis_type: forward (downstream) or backward (upstream)
        max_depth: Maximum depth for traversal
        path: Path to codebase

    Returns:
        Dict with target, analysis_type, impacted_symbols
    """
```

## victor.context.graph_context_builder

### GraphEnhancedContextBuilder

```python
class GraphEnhancedContextBuilder:
    """Build init.md with graph context."""

    async def build_context(
        self,
        task: str,
        max_symbols: int = 50,
        max_hops: int = 2,
    ) -> str: ...

    async def build_data_flow_graph(
        self,
        symbol_id: str,
    ) -> Dict[str, Any]: ...

    async def find_similar_code(
        self,
        code_snippet: str,
        limit: int = 5,
    ) -> List[GraphNode]: ...
```

## victor.processing.graph_embeddings

### GraphAwareEmbedder

```python
class GraphAwareEmbedder:
    """Generate embeddings that capture structure."""

    async def embed_with_context(
        self,
        node: GraphNode,
        graph: CodeContextGraph,
    ) -> List[float]: ...

    async def embed_subgraph(
        self,
        subgraph: Subgraph,
    ) -> List[float]: ...
```

## victor.processing.graph_algorithms

### Graph Algorithms

```python
def build_networkx_graph(
    nodes: List[GraphNode],
    edges: List[GraphEdge],
) -> nx.DiGraph | None: ...

def compute_centrality(
    graph: nx.DiGraph,
    alpha: float = 0.85,
) -> Dict[str, float]: ...

def compute_betweenness(
    graph: nx.DiGraph,
) -> Dict[str, float]: ...

def compute_closeness(
    graph: nx.DiGraph,
) -> Dict[str, float]: ...

def detect_communities(
    graph: nx.DiGraph,
    method: str = "louvain",
) -> Dict[str, int]: ...

def find_shortest_path(
    graph: nx.DiGraph,
    source: str,
    target: str,
) -> List[str] | None: ...

def extract_subgraph(
    graph: nx.DiGraph,
    center_node: str,
    radius: int = 2,
) -> nx.DiGraph | None: ...

def compute_all_metrics(
    nodes: List[GraphNode],
    edges: List[GraphEdge],
) -> GraphMetrics: ...
```

### GraphMetrics

```python
@dataclass
class GraphMetrics:
    """Metrics computed for a code graph."""

    pagerank: Dict[str, float]
    betweenness: Dict[str, float]
    closeness: Dict[str, float]
    in_degree: Dict[str, int]
    out_degree: Dict[str, int]
    communities: Dict[str, int]

    def get_top_nodes(
        self,
        metric: str = "pagerank",
        n: int = 10,
    ) -> List[Tuple[str, float]]: ...
```

## victor.core.graph_rag.requirement_graph

### RequirementGraphBuilder

```python
class RequirementGraphBuilder:
    """Map requirements to code symbols."""

    async def map_requirement(
        self,
        requirement: str,
    ) -> List[str]: ...

    async def create_requirement_node(
        self,
        requirement: str,
        type: str,
        source: str,
    ) -> RequirementNode: ...

    async def link_requirement_to_code(
        self,
        requirement_id: str,
        symbol_ids: List[str],
    ) -> None: ...
```

### RequirementNode

```python
@dataclass
class RequirementNode:
    """Requirement node in the graph."""

    requirement_id: str
    type: str  # feature, bug, task, etc.
    source: str  # github_issue, jira, etc.
    title: str
    description: str | None = None
    priority: float = 0.5
    status: str = "open"
    metadata: Dict[str, Any] = field(default_factory=dict)
```

## victor.storage.graph.protocol

### Subgraph

```python
@dataclass
class Subgraph:
    """A subgraph extracted from the main graph."""

    subgraph_id: str
    anchor_node_id: str
    radius: int
    edge_types: List[str]
    node_ids: List[str]
    edges: List[GraphEdge]
    node_count: int
    computed_at: str | None = None
```

### GraphQueryResult

```python
@dataclass
class GraphQueryResult:
    """Result from a graph query."""

    nodes: List[GraphNode]
    edges: List[GraphEdge]
    subgraphs: List[Subgraph]
    query: str
    execution_time_ms: float
    scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]: ...
```
