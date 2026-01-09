# Experiment Tracking and Metadata System Design

**Author:** Victor AI Team
**Date:** 2025-01-09
**Status:** Design Document
**Version:** 1.0

## Executive Summary

This document outlines the design for a comprehensive experiment tracking and metadata system for Victor's adaptive orchestration. The system is inspired by MLflow and Weights & Biases but tailored specifically for workflow optimization experiments, A/B testing of RL policies, and hyperparameter tuning for LLM-based agents.

**Key Objectives:**
- Track hundreds/thousands of workflow experiments with reproducible metadata
- Support both A/B tests and optimization runs
- Integrate seamlessly with existing RL framework and event systems
- Provide excellent developer experience with simple APIs
- Enable experiment comparison and visualization

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Experiment Metadata Schema](#2-experiment-metadata-schema)
3. [Run Tracking System](#3-run-tracking-system)
4. [Metrics Logging API](#4-metrics-logging-api)
5. [Query and Comparison Interface](#5-query-and-comparison-interface)
6. [Reproducibility Strategy](#6-reproducibility-strategy)
7. [Storage Architecture](#7-storage-architecture)
8. [CLI and Web UI](#8-cli-and-web-ui)
9. [Implementation Plan](#9-implementation-plan)
10. [MVP Feature List](#10-mvp-feature-list)

---

## 1. Architecture Overview

### 1.1 System Design

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         EXPERIMENT TRACKING                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │   CLI/UI     │───▶│   API Layer  │───▶│   Storage    │              │
│  │              │    │              │    │   Layer      │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│         │                   │                    │                       │
│         │                   ▼                    ▼                       │
│         │            ┌──────────────┐    ┌──────────────┐              │
│         │            │  Experiment  │    │   SQLite     │              │
│         │            │  Manager     │    │   Backend    │              │
│         │            └──────────────┘    └──────────────┘              │
│         │                   │                                            │
│         ▼                   ▼                                            │
│  ┌─────────────────────────────────────────────────┐                  │
│  │            Integration Points                    │                  │
│  ├─────────────────────────────────────────────────┤                  │
│  │ • RL Framework (RLOutcome, learners)           │                  │
│  │ • EventBus (MetricsCollector)                  │                  │
│  │ • WorkflowEngine (workflow runs)               │                  │
│  │ • AdaptiveModeController (mode transitions)     │                  │
│  │ • ExperimentCoordinator (A/B tests)            │                  │
│  └─────────────────────────────────────────────────┘                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Design Principles

1. **MLflow-like Simplicity**: Minimal API surface area with intuitive methods
2. **Automatic Tracking**: Capture metadata without explicit logging
3. **Hierarchical Organization**: Experiments → Runs → Metrics → Artifacts
4. **Reproducibility First**: Capture full environment state automatically
5. **Query Performance**: Optimized for common query patterns (filtering, sorting, comparison)
6. **Extensibility**: Plugin architecture for custom metrics and visualizations

### 1.3 Integration with Existing Systems

| Existing Component | Integration Point |
|--------------------|-------------------|
| `RLOutcome` | Auto-logged as experiment run metrics |
| `ExperimentCoordinator` | Experiments tracked as experiment groups |
| `MetricsCollector` | Stream metrics logged to active run |
| `WorkflowEngine` | Each workflow execution becomes a run |
| `EventBus` | Subscribe to events for automatic logging |
| `AdaptiveModeController` | Mode transitions logged as metadata |

---

## 2. Experiment Metadata Schema

### 2.1 Experiment Entity

**Core Experiment Metadata:**

```python
@dataclass
class Experiment:
    """A collection of runs with a common hypothesis."""

    # Identification
    experiment_id: str  # UUID
    name: str  # Human-readable name
    description: str  # Detailed description

    # Hypothesis & Goals
    hypothesis: str  # What we're testing
    success_criteria: Dict[str, Any]  # Metrics for success
    baseline_value: Optional[float]  # Current baseline to compare against

    # Organization
    tags: List[str]  # For categorization/search
    parent_id: Optional[str]  # Parent experiment (iterative improvements)
    group_id: Optional[str]  # Experiment group (batch runs)

    # Configuration
    parameters: Dict[str, Any]  # Hyperparameters, workflow config
    workflow_name: Optional[str]  # Associated workflow
    vertical: str  # Domain vertical (coding, devops, etc.)

    # Reproducibility
    git_commit_sha: str  # Git commit for code reproducibility
    git_branch: str  # Git branch name
    git_dirty: bool  # Whether there are uncommitted changes

    # Timestamps
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]

    # Status
    status: ExperimentStatus  # DRAFT, RUNNING, COMPLETED, FAILED, ARCHIVED

    # Relationships
    production_workflow_id: Optional[str]  # Link to production workflow
    linked_experiments: List[str]  # Related experiments
```

**Experiment Status Enum:**

```python
class ExperimentStatus(str, Enum):
    DRAFT = "draft"  # Not yet started
    RUNNING = "running"  # Actively collecting runs
    PAUSED = "paused"  # Temporarily halted
    COMPLETED = "completed"  # Finished successfully
    FAILED = "failed"  # Failed to complete
    ARCHIVED = "archived"  # No longer active
```

### 2.2 Run Entity

**Core Run Metadata:**

```python
@dataclass
class Run:
    """A single execution within an experiment."""

    # Identification
    run_id: str  # UUID
    experiment_id: str  # Parent experiment
    name: str  # Human-readable name

    # Status
    status: RunStatus  # QUEUED, RUNNING, COMPLETED, FAILED, KILLED

    # Timing
    started_at: datetime
    completed_at: Optional[datetime]
    duration_seconds: Optional[float]

    # Input/Output
    input_data: Dict[str, Any]  # Input parameters for this run
    output_data: Dict[str, Any]  # Output results
    error_message: Optional[str]  # Error if failed

    # Environment (captured automatically)
    python_version: str
    os_info: str
    victor_version: str
    dependencies: Dict[str, str]  # pip freeze output

    # Random Seeds (for reproducibility)
    python_seed: int
    numpy_seed: int
    llm_temperature: float
    llm_top_p: float

    # Provider/Model
    provider: str
    model: str
    task_type: str

    # Metrics Summary
    metrics_summary: Dict[str, float]  # Final metric values

    # Artifacts
    artifact_count: int  # Number of artifacts
    artifact_size_bytes: int  # Total artifact size
```

**Run Status Enum:**

```python
class RunStatus(str, Enum):
    QUEUED = "queued"  # Waiting to start
    RUNNING = "running"  # Currently executing
    COMPLETED = "completed"  # Finished successfully
    FAILED = "failed"  # Failed with error
    KILLED = "killed"  # Terminated by user
```

### 2.3 Experiment Relationships

**Parent-Child Relationships:**

```python
@dataclass
class ExperimentRelation:
    """Relationship between experiments."""

    parent_id: str
    child_id: str
    relation_type: RelationType
    created_at: datetime

class RelationType(str, Enum):
    ITERATION = "iteration"  # Child improves on parent
    ABDUCTION = "abduction"  # Child investigates parent failure
    BRANCH = "branch"  # Child explores different direction
    REPRODUCTION = "reproduction"  # Child reproduces parent
```

**Experiment Groups:**

```python
@dataclass
class ExperimentGroup:
    """A collection of related experiments."""

    group_id: str
    name: str
    description: str
    experiment_ids: List[str]
    created_at: datetime
    metadata: Dict[str, Any]
```

---

## 3. Run Tracking System

### 3.1 Run Lifecycle

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  QUEUED  │───▶│ RUNNING  │───▶│COMPLETED │     │  FAILED  │
└──────────┘     └──────────┘     └──────────┘     └──────────┘
                     │                                  │
                     │                                  ▼
                     │                             ┌──────────┐
                     └────────────────────────────▶│  KILLED  │
                                                     └──────────┘
```

### 3.2 Artifact Management

**Artifact Types:**

1. **Workflow Configuration** - JSON/YAML workflow definitions
2. **Trained Models** - RL policy checkpoints (pickle/pytorch)
3. **Logs** - Debug logs and error traces
4. **Visualizations** - Charts, graphs, plots
5. **Data** - Input/output data snapshots
6. **Code** - Source code snapshots

**Artifact Schema:**

```python
@dataclass
class Artifact:
    """A file or data associated with a run."""

    artifact_id: str
    run_id: str
    artifact_type: ArtifactType
    filename: str
    file_path: str  # Local or S3/GCS path
    file_size_bytes: int
    created_at: datetime
    metadata: Dict[str, Any]  # Custom metadata
    is_cached: bool  # Whether artifact is in local cache

class ArtifactType(str, Enum):
    WORKFLOW_CONFIG = "workflow_config"
    MODEL = "model"
    LOG = "log"
    VISUALIZATION = "visualization"
    DATA = "data"
    CODE = "code"
    CUSTOM = "custom"
```

**Artifact Storage Backends:**

```python
class ArtifactBackend(ABC):
    """Abstract base for artifact storage."""

    @abstractmethod
    def log_artifact(self, run_id: str, local_path: str, artifact_path: str) -> Artifact:
        """Upload artifact to storage."""
        pass

    @abstractmethod
    def download_artifact(self, artifact: Artifact, local_path: str) -> None:
        """Download artifact from storage."""
        pass

    @abstractmethod
    def get_uri(self, artifact: Artifact) -> str:
        """Get URI for artifact (e.g., s3://...)."""
        pass
```

### 3.3 Run Metadata Capture

**Automatic Capture:**

```python
def capture_run_environment() -> RunEnvironment:
    """Capture full environment state for reproducibility."""

    # System info
    os_info = platform.platform()
    python_version = sys.version

    # Dependencies
    dependencies = subprocess.check_output(["pip", "freeze"]).decode()

    # Git info
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    git_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
    git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"]).decode().strip())

    # Victor version
    victor_version = importlib.metadata.version("victor-ai")

    return RunEnvironment(
        os_info=os_info,
        python_version=python_version,
        dependencies=dependencies,
        git_commit=git_commit,
        git_branch=git_branch,
        git_dirty=git_dirty,
        victor_version=victor_version,
    )
```

---

## 4. Metrics Logging API

### 4.1 Public API

**Simple and Intuitive:**

```python
from victor.experiments import ExperimentTracker

# Start a new experiment
experiment = ExperimentTracker.create_experiment(
    name="tool-selector-optimization",
    description="Optimize tool selection thresholds",
    hypothesis="Lower semantic threshold improves quality",
    tags=["tool-selection", "optimization"],
)

# Start a run
with experiment.start_run(
    name="run-1",
    parameters={"semantic_threshold": 0.7, "keyword_threshold": 0.5},
) as run:
    # Log metrics
    run.log_metric("quality_score", 0.85)
    run.log_metric("success_rate", 0.92)
    run.log_metric("latency_ms", 150.0)

    # Log parameters
    run.log_param("tool_budget", 10)
    run.log_param("max_iterations", 20)

    # Log artifacts
    run.log_artifact("workflow_config.json", "path/to/config.json")
    run.log_model("policy_checkpoint.pkl")

    # Log text
    run.log_text("debug_log", "Tool selection: semantic=5, keyword=2")

# Query experiments
experiments = ExperimentTracker.search_experiments(
    tags=["optimization"],
    status="completed"
)

# Compare experiments
result = ExperimentTracker.compare_experiments(
    experiment_ids=["exp-1", "exp-2"],
    metrics=["quality_score", "success_rate"]
)
```

### 4.2 Metrics Storage

**Time-Series Database Schema:**

```sql
CREATE TABLE metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    step INTEGER,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE INDEX idx_metrics_run_step ON metrics(run_id, step);
CREATE INDEX idx_metrics_name ON metrics(metric_name);
```

**Metric Types:**

```python
@dataclass
class Metric:
    """A single metric data point."""

    run_id: str
    metric_name: str
    metric_value: float
    step: Optional[int]  # Iteration/step number
    timestamp: datetime
    is_final: bool  # Whether this is the final value

class MetricType:
    SCALAR = "scalar"  # Single value (time, cost, accuracy)
    TIME_SERIES = "time_series"  # Multiple values over time
    HISTOGRAM = "histogram"  # Distribution of values
    TEXT = "text"  # Textual information
```

### 4.3 Downsampling Strategy

**For Long-Running Experiments:**

```python
class MetricsDownsampler:
    """Downsample metrics for long-running experiments."""

    def __init__(self, max_points: int = 10000):
        self.max_points = max_points

    def should_downsample(self, run_id: str, metric_name: str) -> bool:
        """Check if metric should be downsampled."""
        count = db.execute(
            "SELECT COUNT(*) FROM metrics WHERE run_id=? AND metric_name=?",
            (run_id, metric_name)
        ).fetchone()[0]
        return count >= self.max_points

    def downsample(self, run_id: str, metric_name: str) -> None:
        """Downsample metric using LTTB algorithm."""
        # Implementation of Largest Triangle Three Buckets
        # Preserves visual fidelity while reducing data points
        pass
```

---

## 5. Query and Comparison Interface

### 5.1 Query API

**Search Experiments:**

```python
@dataclass
class ExperimentQuery:
    """Query for filtering experiments."""

    # Text search
    name_contains: Optional[str] = None
    description_contains: Optional[str] = None

    # Tags
    tags_any: List[str] = None  # Experiments with any of these tags
    tags_all: List[str] = None  # Experiments with all of these tags

    # Status
    status: Optional[ExperimentStatus] = None

    # Date range
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None

    # Performance
    metric_name: Optional[str] = None
    metric_min: Optional[float] = None
    metric_max: Optional[float] = None

    # Relations
    parent_id: Optional[str] = None
    group_id: Optional[str] = None

    # Provider/Model
    provider: Optional[str] = None
    model: Optional[str] = None

    # Pagination
    limit: int = 100
    offset: int = 0

    # Sorting
    sort_by: str = "created_at"  # created_at, name, metric, duration
    sort_order: str = "desc"  # asc, desc
```

**Usage:**

```python
# Find successful tool selection experiments
experiments = ExperimentTracker.search(
    ExperimentQuery(
        tags_any=["tool-selection"],
        status=ExperimentStatus.COMPLETED,
        metric_name="quality_score",
        metric_min=0.8,
        sort_by="created_at",
        sort_order="desc",
        limit=10
    )
)
```

### 5.2 Comparison Tools

**Side-by-Side Comparison:**

```python
@dataclass
class ExperimentComparison:
    """Result of comparing two or more experiments."""

    experiment_ids: List[str]
    comparison_date: datetime

    # Metric comparison
    metric_diffs: Dict[str, MetricDiff]
    overall_winner: str  # Experiment ID with best overall performance

    # Statistical significance
    significant_metrics: List[str]  # Metrics with significant differences
    confidence_level: float  # e.g., 0.95

    # Recommendations
    recommendation: str  # Text recommendation
    should_rollback: bool

@dataclass
class MetricDiff:
    """Difference in a metric between experiments."""

    metric_name: str
    control_value: float
    treatment_value: float
    absolute_diff: float
    relative_diff: float  # Percentage
    p_value: float
    is_significant: bool
    confidence_interval: Tuple[float, float]
```

**Usage:**

```python
# Compare two experiments
comparison = ExperimentTracker.compare(
    experiment_ids=["exp-1", "exp-2"],
    metrics=["quality_score", "success_rate", "latency_ms"],
    confidence_level=0.95
)

print(comparison.recommendation)
# "Experiment exp-2 shows significant improvement in quality_score (+5.2%, p<0.01).
#  Recommend rollout."
```

### 5.3 Leaderboards

**Best Performing Experiments:**

```python
@dataclass
class LeaderboardEntry:
    """Entry in performance leaderboard."""

    experiment_id: str
    experiment_name: str
    metric_value: float
    metric_name: str
    run_count: int
    created_at: datetime

def get_leaderboard(
    metric_name: str,
    tags: Optional[List[str]] = None,
    limit: int = 10
) -> List[LeaderboardEntry]:
    """Get top-performing experiments for a metric."""

    query = """
    SELECT
        e.experiment_id,
        e.name,
        AVG(m.metric_value) as avg_metric_value,
        COUNT(DISTINCT m.run_id) as run_count,
        e.created_at
    FROM experiments e
    JOIN runs r ON e.experiment_id = r.experiment_id
    JOIN metrics m ON r.run_id = m.run_id
    WHERE m.metric_name = ?
    AND e.status = 'completed'
    GROUP BY e.experiment_id
    ORDER BY avg_metric_value DESC
    LIMIT ?
    """

    return db.execute(query, (metric_name, limit)).fetchall()
```

**Usage:**

```python
# Get top 10 experiments by quality score
leaderboard = ExperimentTracker.get_leaderboard(
    metric_name="quality_score",
    tags=["tool-selection"],
    limit=10
)

for entry in leaderboard:
    print(f"{entry.experiment_name}: {entry.metric_value:.3f}")
```

### 5.4 Visualization

**Chart Types:**

1. **Line Charts** - Metric values over time/steps
2. **Bar Charts** - Compare metrics across experiments
3. **Scatter Plots** - Metric correlations
4. **Histograms** - Value distributions
5. **Heatmaps** - Parameter importance

**Visualization API:**

```python
def plot_metrics(
    run_id: str,
    metrics: List[str],
    chart_type: str = "line"
) -> Figure:
    """Generate matplotlib/plotly figure."""

    if chart_type == "line":
        # Time series plot
        fig = go.Figure()
        for metric in metrics:
            data = get_metric_data(run_id, metric)
            fig.add_trace(go.Scatter(
                x=data["step"],
                y=data["value"],
                name=metric
            ))
        return fig

    elif chart_type == "bar":
        # Comparison bar chart
        pass
```

---

## 6. Reproducibility Strategy

### 6.1 Environment Capture

**Full State Snapshot:**

```python
@dataclass
class ReproducibilitySnapshot:
    """Complete snapshot for reproducibility."""

    # Code
    git_commit_sha: str
    git_branch: str
    git_dirty: bool
    git_diff: Optional[str]  # Diff if dirty

    # Environment
    python_version: str
    os_info: str
    cpu_info: str
    memory_info: str

    # Dependencies
    pip_freeze: str
    conda_env_export: Optional[str]

    # Configuration
    victor_config: Dict[str, Any]
    environment_variables: Dict[str, str]

    # Random Seeds
    python_seed: int
    numpy_seed: int
    torch_seed: Optional[int]

    # Provider/Model
    provider: str
    model: str
    api_version: Optional[str]

    # Timestamp
    captured_at: datetime
```

### 6.2 Input Data Versioning

**Data Hashing:**

```python
def hash_input_data(data: Any) -> str:
    """Generate hash for input data."""

    # Serialize to JSON
    serialized = json.dumps(data, sort_keys=True, default=str)

    # Generate SHA-256 hash
    return hashlib.sha256(serialized.encode()).hexdigest()

# Store with run
run.input_hash = hash_input_data(input_data)
run.input_data = input_data  # Optionally store full data
```

### 6.3 Reproducibility Verification

**Check if Run is Reproducible:**

```python
def verify_reproducibility(run_id: str) -> ReproducibilityReport:
    """Verify if run can be reproduced."""

    run = get_run(run_id)
    snapshot = get_snapshot(run_id)

    issues = []

    # Check git state
    current_commit = get_git_commit()
    if current_commit != snapshot.git_commit_sha:
        issues.append("Git commit mismatch")

    if snapshot.git_dirty:
        issues.append("Git working directory was dirty")

    # Check dependencies
    current_deps = get_pip_freeze()
    if current_deps != snapshot.pip_freeze:
        issues.append("Dependencies changed")

    # Check Victor version
    current_version = get_victor_version()
    if current_version != snapshot.victor_version:
        issues.append(f"Victor version mismatch: {current_version} vs {snapshot.victor_version}")

    return ReproducibilityReport(
        run_id=run_id,
        is_reproducible=len(issues) == 0,
        issues=issues,
        recommendations=_generate_recommendations(issues)
    )
```

### 6.4 Reproduction Command

**Generate Command to Reproduce:**

```python
def generate_reproduction_command(run_id: str) -> str:
    """Generate shell command to reproduce a run."""

    run = get_run(run_id)
    snapshot = get_snapshot(run_id)

    # Checkout git commit
    cmd = f"git checkout {snapshot.git_commit_sha}\n"

    # Install dependencies
    cmd += f"pip install -r requirements.txt\n"

    # Set environment variables
    for key, value in snapshot.environment_variables.items():
        cmd += f"export {key}='{value}'\n"

    # Run with same parameters
    cmd += f"victor experiment run {run.experiment_id} --params '{json.dumps(run.input_data)}'"

    return cmd
```

---

## 7. Storage Architecture

### 7.1 Storage Backends

**SQLite Backend (Default):**

```python
class SQLiteExperimentStore:
    """SQLite-based storage for experiments."""

    def __init__(self, db_path: str = "~/.victor/experiments.db"):
        self.db_path = Path(db_path).expanduser()
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        """Create tables if not exist."""
        # Uses Schema.get_all_schemas() + experiment-specific tables
        pass

    # CRUD operations
    def create_experiment(self, experiment: Experiment) -> str:
        pass

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        pass

    def update_experiment(self, experiment: Experiment) -> bool:
        pass

    def delete_experiment(self, experiment_id: str) -> bool:
        pass

    # Query operations
    def search_experiments(self, query: ExperimentQuery) -> List[Experiment]:
        pass

    # Run operations
    def create_run(self, run: Run) -> str:
        pass

    def get_run(self, run_id: str) -> Optional[Run]:
        pass

    def update_run(self, run: Run) -> bool:
        pass

    # Metric operations
    def log_metric(self, metric: Metric) -> None:
        pass

    def get_metrics(self, run_id: str) -> List[Metric]:
        pass

    # Artifact operations
    def log_artifact(self, artifact: Artifact) -> None:
        pass

    def get_artifacts(self, run_id: str) -> List[Artifact]:
        pass
```

**PostgreSQL Backend (Production):**

```python
class PostgresExperimentStore(ExperimentStore):
    """PostgreSQL-based storage for multi-user production."""

    def __init__(self, connection_string: str):
        self.pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            dsn=connection_string
        )
        self._ensure_tables()

    # Same interface as SQLite, but with PostgreSQL-specific optimizations
    # - Connection pooling
    # - Better concurrent access
    # - Full-text search on descriptions
    # - JSONB for metadata
```

### 7.2 Data Model

**Core Tables:**

```sql
-- Experiments table
CREATE TABLE experiments (
    experiment_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    hypothesis TEXT,
    success_criteria TEXT,  -- JSON
    baseline_value REAL,

    tags TEXT,  -- JSON array
    parent_id TEXT,
    group_id TEXT,

    parameters TEXT,  -- JSON
    workflow_name TEXT,
    vertical TEXT DEFAULT 'coding',

    git_commit_sha TEXT NOT NULL,
    git_branch TEXT NOT NULL,
    git_dirty INTEGER DEFAULT 0,

    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,

    status TEXT NOT NULL,
    production_workflow_id TEXT,
    linked_experiments TEXT,  -- JSON array

    FOREIGN KEY (parent_id) REFERENCES experiments(experiment_id)
);

-- Runs table
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    name TEXT NOT NULL,

    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    duration_seconds REAL,

    input_data TEXT,  -- JSON
    output_data TEXT,  -- JSON
    error_message TEXT,

    python_version TEXT NOT NULL,
    os_info TEXT NOT NULL,
    victor_version TEXT NOT NULL,
    dependencies TEXT,  -- JSON

    python_seed INTEGER NOT NULL,
    numpy_seed INTEGER NOT NULL,
    llm_temperature REAL,
    llm_top_p REAL,

    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    task_type TEXT NOT NULL,

    metrics_summary TEXT,  -- JSON
    artifact_count INTEGER DEFAULT 0,
    artifact_size_bytes INTEGER DEFAULT 0,

    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id) ON DELETE CASCADE
);

-- Metrics table (time-series)
CREATE TABLE metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    step INTEGER,
    timestamp TEXT NOT NULL,
    is_final INTEGER DEFAULT 0,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

-- Artifacts table
CREATE TABLE artifacts (
    artifact_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    metadata TEXT,  -- JSON
    is_cached INTEGER DEFAULT 0,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

-- Experiment groups
CREATE TABLE experiment_groups (
    group_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    experiment_ids TEXT NOT NULL,  -- JSON array
    created_at TEXT NOT NULL,
    metadata TEXT  -- JSON
);

-- Indexes for performance
CREATE INDEX idx_experiments_status ON experiments(status);
CREATE INDEX idx_experiments_tags ON experiments(tags);
CREATE INDEX idx_experiments_created ON experiments(created_at);
CREATE INDEX idx_runs_experiment ON runs(experiment_id);
CREATE INDEX idx_runs_status ON runs(status);
CREATE INDEX idx_metrics_run_step ON metrics(run_id, step);
CREATE INDEX idx_metrics_name ON metrics(metric_name);
CREATE INDEX idx_artifacts_run ON artifacts(run_id);
```

### 7.3 Schema Migrations

**Version Control:**

```python
class SchemaVersion:
    """Experiment tracking schema version."""

    CURRENT_VERSION = 1

MIGRATIONS = {
    1: [
        # Initial schema
        "CREATE TABLE experiments (...)",
        "CREATE TABLE runs (...)",
        "CREATE TABLE metrics (...)",
        "CREATE TABLE artifacts (...)",
    ],
    2: [
        # Add experiment groups
        "CREATE TABLE experiment_groups (...)",
        "ALTER TABLE experiments ADD COLUMN group_id TEXT",
    ],
    # Future migrations
}

def migrate_schema(from_version: int, to_version: int) -> None:
    """Apply schema migrations."""

    for version in range(from_version + 1, to_version + 1):
        if version in MIGRATIONS:
            for sql in MIGRATIONS[version]:
                db.execute(sql)
            logger.info(f"Migrated to schema version {version}")

    # Update version
    db.execute("INSERT INTO sys_schema_version VALUES (?)", (to_version,))
    db.commit()
```

---

## 8. CLI and Web UI

### 8.1 CLI Commands

**Primary Commands:**

```bash
# List experiments
victor experiment list [--status STATUS] [--tags TAGS] [--limit N]

# Show experiment details
victor experiment show <experiment_id>

# Create new experiment
victor experiment create \
    --name "Tool Selector V2" \
    --description "Optimize tool selection thresholds" \
    --tags tool-selection,optimization \
    --params '{"semantic_threshold": 0.7}'

# Start a run
victor experiment run <experiment_id> \
    --name "run-1" \
    --params '{"semantic_threshold": 0.7}'

# Compare experiments
victor experiment compare <exp_id_1> <exp_id_2> \
    --metrics quality_score,success_rate

# Delete experiment
victor experiment delete <experiment_id> [--force]

# Export experiment data
victor experiment export <experiment_id> --format json --output exp.json

# Import experiment data
victor experiment import exp.json

# Get leaderboard
victor experiment leaderboard --metric quality_score --tags optimization

# Verify reproducibility
victor experiment verify <run_id>

# Generate reproduction command
victor experiment reproduce <run_id>
```

**Example Usage:**

```bash
# List all completed optimization experiments
$ victor experiment list --status completed --tags optimization

Experiment ID          Name                    Status      Quality Score
exp-123               Tool Selector V2        completed   0.85
exp-124               Mode Transition Tuning  completed   0.82
exp-125               Budget Optimization     completed   0.88

# Show experiment details
$ victor experiment show exp-123

Name: Tool Selector V2
Description: Optimize tool selection thresholds
Status: completed
Created: 2025-01-09 10:30:00
Git Commit: a1b2c3d4

Parameters:
  semantic_threshold: 0.7
  keyword_threshold: 0.5

Best Run: run-456
  Quality Score: 0.85
  Success Rate: 0.92
  Latency: 150ms

# Compare two experiments
$ victor experiment compare exp-123 exp-124 --metrics quality_score,success_rate

Metric          Control (exp-123)    Treatment (exp-124)    Diff    P-value
quality_score   0.85                0.82                  -3.5%   0.12
success_rate    0.92                0.89                  -3.3%   0.08

Recommendation: No significant difference detected. Continue experiment.
```

### 8.2 Web UI (FastAPI Extension)

**Routes:**

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI()

# Experiment CRUD
@app.get("/api/experiments")
def list_experiments(
    status: Optional[str] = None,
    tags: Optional[str] = None,
    limit: int = 100
):
    """List experiments with filtering."""
    pass

@app.get("/api/experiments/{experiment_id}")
def get_experiment(experiment_id: str):
    """Get experiment details."""
    pass

@app.post("/api/experiments")
def create_experiment(experiment: ExperimentCreate):
    """Create new experiment."""
    pass

# Run operations
@app.get("/api/experiments/{experiment_id}/runs")
def list_runs(experiment_id: str):
    """List runs for an experiment."""
    pass

@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    """Get run details."""
    pass

@app.get("/api/runs/{run_id}/metrics")
def get_run_metrics(run_id: str):
    """Get metrics for a run."""
    pass

@app.get("/api/runs/{run_id}/artifacts")
def get_run_artifacts(run_id: str):
    """Get artifacts for a run."""
    pass

# Comparison
@app.post("/api/experiments/compare")
def compare_experiments(request: ComparisonRequest):
    """Compare multiple experiments."""
    pass

# Leaderboard
@app.get("/api/leaderboard")
def get_leaderboard(metric: str, tags: Optional[str] = None):
    """Get performance leaderboard."""
    pass

# Visualization
@app.get("/api/runs/{run_id}/plots/{metric}")
def get_metric_plot(run_id: str, metric: str):
    """Generate plot for a metric."""
    pass

# Dashboard
@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Serve experiment tracking dashboard."""
    return html_content
```

**Dashboard UI:**

- **Experiment List** - Searchable, filterable table of experiments
- **Experiment Details** - Run metrics, parameters, artifacts
- **Run Comparison** - Side-by-side comparison with statistical tests
- **Leaderboards** - Top experiments by metric
- **Charts** - Time-series plots, bar charts, scatter plots
- **Artifact Browser** - View and download artifacts
- **Reproducibility Report** - Verify if run can be reproduced

---

## 9. Implementation Plan

### 9.1 Module Structure

```
victor/experiments/
├── __init__.py                 # Public API exports
├── tracking.py                 # ExperimentTracker public API
├── entities.py                 # Experiment, Run, Metric dataclasses
├── storage.py                  # ExperimentStore ABC
├── sqlite_store.py             # SQLite implementation
├── postgres_store.py           # PostgreSQL implementation
├── query.py                    # ExperimentQuery, comparison logic
├── artifacts.py                # Artifact management
├── reproducibility.py          # Environment capture, verification
├── cli.py                      # CLI commands
├── api.py                      # FastAPI routes
└── utils/
    ├── logging.py              # Metric logging utilities
    ├── hashing.py              # Data hashing
    └── visualization.py        # Plot generation
```

### 9.2 Implementation Phases

**Phase 1: Core Tracking (Week 1-2)**
- `entities.py` - Data models (300 LOC)
- `tracking.py` - Public API (400 LOC)
- `storage.py` - Storage abstraction (200 LOC)
- `sqlite_store.py` - SQLite backend (600 LOC)

**Phase 2: Metrics & Artifacts (Week 3)**
- `utils/logging.py` - Metric logging (300 LOC)
- `artifacts.py` - Artifact management (500 LOC)
- Schema updates for metrics/artifacts (included in sqlite_store)

**Phase 3: Query & Comparison (Week 4)**
- `query.py` - Query and comparison (600 LOC)
- Statistical tests (200 LOC)
- Leaderboards (150 LOC)

**Phase 4: Reproducibility (Week 5)**
- `reproducibility.py` - Environment capture (400 LOC)
- `utils/hashing.py` - Data hashing (100 LOC)
- Verification logic (200 LOC)

**Phase 5: CLI & UI (Week 6)**
- `cli.py` - CLI commands (500 LOC)
- `api.py` - FastAPI routes (400 LOC)
- Web dashboard templates (800 LOC HTML/JS)

**Phase 6: Integration & Testing (Week 7-8)**
- Integration with RL framework
- Integration with EventBus
- Integration with WorkflowEngine
- Comprehensive tests
- Documentation

### 9.3 Estimated LOC

| Module | Estimated LOC | Complexity |
|--------|--------------|------------|
| `entities.py` | 300 | Low |
| `tracking.py` | 400 | Medium |
| `storage.py` | 200 | Low |
| `sqlite_store.py` | 600 | Medium |
| `postgres_store.py` | 500 | Medium |
| `artifacts.py` | 500 | Medium |
| `query.py` | 600 | High |
| `reproducibility.py` | 400 | Medium |
| `cli.py` | 500 | Medium |
| `api.py` | 400 | Medium |
| `utils/*` | 500 | Low |
| **Tests** | 2000 | High |
| **Documentation** | 800 | Low |
| **Total** | **7700** | - |

**Estimated Implementation Time:** 8 weeks (1 developer)

---

## 10. MVP Feature List

### 10.1 Minimum Viable Tracking System

**Core Features (Must Have):**

1. **Experiment Creation & Management**
   - Create experiments with metadata
   - Start/run/complete experiments
   - Tag and search experiments
   - Experiment relationships (parent-child)

2. **Run Tracking**
   - Start runs with parameters
   - Log metrics (scalar, time-series)
   - Log parameters
   - Run status tracking

3. **Artifacts**
   - Log files as artifacts
   - Download artifacts
   - Artifact metadata

4. **Query Interface**
   - Search experiments by tags, status, date
   - Filter by metrics
   - Sort by performance

5. **Comparison**
   - Compare two experiments
   - Metric diffing (percent change)
   - Basic statistical tests

6. **CLI**
   - `victor experiment list`
   - `victor experiment show`
   - `victor experiment create`
   - `victor experiment run`
   - `victor experiment compare`

7. **Reproducibility**
   - Capture git commit
   - Capture pip dependencies
   - Store environment info
   - Generate reproduction command

8. **SQLite Storage**
   - Local SQLite database
   - Efficient queries
   - Automatic schema migrations

### 10.2 Future Enhancements (Should Have)

**Post-MVP Features:**

1. **PostgreSQL Backend** - Multi-user production deployment
2. **Advanced Visualization** - Rich charts and dashboards
3. **Artifact Storage** - S3/GCS integration
4. **Experiment Groups** - Batch runs and hyperparameter sweeps
5. **Real-time Monitoring** - Live experiment tracking
6. **Notifications** - Alerts on experiment completion
7. **Export/Import** - Share experiments across teams
8. **Web Dashboard** - Full-featured web UI
9. **API Keys** - Secure multi-user access
10. **Collaboration** - Comments, notes, sharing

### 10.3 Experimental Features (Nice to Have)

**Long-term Vision:**

1. **Auto-ML Integration** - Automatic hyperparameter optimization
2. **Multi-objective Optimization** - Pareto frontiers
3. **Causal Inference** - Understand experiment impacts
4. **Transfer Learning** - Apply learnings across experiments
5. **Experiment Templates** - Reusable experiment configs
6. **A/B Testing Framework** - Integration with production
7. **Canary Deployments** - Gradual rollouts
8. **Analytics Dashboard** - Team-level insights

---

## Appendices

### A. Integration with Existing RL Framework

**RLOutcome Auto-Logging:**

```python
# In victor/agent/rl/base.py
def record_outcome(self, outcome: RLOutcome) -> None:
    """Record outcome and auto-log to active experiment."""

    # Existing logic
    ...

    # Auto-log to experiment if active
    if ExperimentTracker.active_run():
        ExperimentTracker.log_metric("success", 1.0 if outcome.success else 0.0)
        ExperimentTracker.log_metric("quality_score", outcome.quality_score)
        ExperimentTracker.log_param("provider", outcome.provider)
        ExperimentTracker.log_param("model", outcome.model)
        ExperimentTracker.log_param("task_type", outcome.task_type)
```

**ExperimentCoordinator Integration:**

```python
# In victor/agent/rl/experiment_coordinator.py
class ExperimentCoordinator:
    def create_experiment(self, config: ExperimentConfig) -> bool:
        """Create experiment and register with ExperimentTracker."""

        # Create A/B test experiment
        experiment = ExperimentTracker.create_experiment(
            name=config.name,
            description=config.description,
            hypothesis=f"A/B test: {config.control.name} vs {config.treatment.name}",
            tags=["ab_test", config.control.name, config.treatment.name],
        )

        # Link to coordinator
        config.experiment_tracking_id = experiment.experiment_id

        return True
```

### B. Example Workflows

**Workflow 1: Hyperparameter Optimization**

```python
from victor.experiments import ExperimentTracker

# Create experiment
experiment = ExperimentTracker.create_experiment(
    name="semantic-threshold-optimization",
    description="Find optimal semantic similarity threshold for tool selection",
    hypothesis="Lower threshold improves quality but reduces efficiency",
    tags=["optimization", "tool-selection"],
)

# Grid search over thresholds
for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
    with experiment.start_run(
        name=f"threshold-{threshold}",
        parameters={"semantic_threshold": threshold}
    ) as run:
        # Run workflow
        result = run_workflow(tool_selector_config)

        # Log metrics
        run.log_metric("quality_score", result.quality)
        run.log_metric("success_rate", result.success_rate)
        run.log_metric("avg_tool_calls", result.avg_tools)

# Analyze results
leaderboard = experiment.get_leaderboard(metric="quality_score")
print(f"Best threshold: {leaderboard[0].parameters['semantic_threshold']}")
```

**Workflow 2: A/B Testing RL Policy**

```python
# Create A/B test experiment
experiment = ExperimentTracker.create_experiment(
    name="rl-policy-v2-vs-baseline",
    description="Test new RL policy against baseline",
    hypothesis="RL policy improves tool selection by 10%",
    tags=["ab_test", "rl", "tool-selection"],
)

# Control group (baseline)
for i in range(50):
    with experiment.start_run(
        name=f"control-{i}",
        parameters={"policy": "baseline"}
    ) as run:
        result = run_workflow(baseline_policy)
        run.log_metric("quality_score", result.quality)

# Treatment group (RL policy)
for i in range(50):
    with experiment.start_run(
        name=f"treatment-{i}",
        parameters={"policy": "rl_v2"}
    ) as run:
        result = run_workflow(rl_policy_v2)
        run.log_metric("quality_score", result.quality)

# Compare
comparison = experiment.compare_runs(
    control_filter={"policy": "baseline"},
    treatment_filter={"policy": "rl_v2"},
    metrics=["quality_score"]
)

if comparison.is_significant and comparison.treatment_better:
    print("Roll out RL policy!")
else:
    print("Keep baseline")
```

### C. Performance Considerations

**Query Optimization:**

1. **Indexes** - Create indexes on commonly filtered columns
2. **Pagination** - Limit result sets for large queries
3. **Caching** - Cache frequently accessed experiments
4. **Lazy Loading** - Load artifacts on demand
5. **Downsampling** - Downsample time-series metrics for long runs

**Storage Optimization:**

1. **Compression** - Compress artifact files
2. **Pruning** - Delete old/failed experiments
3. **Archiving** - Move old experiments to cold storage
4. **Partitioning** - Partition tables by date (PostgreSQL)

---

## Conclusion

This experiment tracking system provides a comprehensive solution for tracking workflow experiments in Victor. The design emphasizes:

- **Simplicity** - MLflow-like API with minimal boilerplate
- **Reproducibility** - Automatic capture of environment state
- **Performance** - Optimized queries and storage
- **Integration** - Seamless integration with existing RL framework
- **Extensibility** - Plugin architecture for custom metrics and visualizations

The MVP can be implemented in 8 weeks and provides immediate value for tracking and optimizing workflow experiments. Future enhancements will add multi-user support, advanced visualizations, and production deployment features.

---

**Document Version:** 1.0
**Last Updated:** 2025-01-09
**Next Review:** After MVP implementation
