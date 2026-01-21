# Enhanced Proficiency Tracker Documentation

## Overview

The `ProficiencyTracker` is a comprehensive system for tracking and analyzing agent performance metrics over time. It provides moving averages, trend detection, trajectory tracking, and RL training data export capabilities.

**File**: `/Users/vijaysingh/code/codingagent/victor/agent/improvement/proficiency_tracker.py`
**Lines**: 1,494 lines
**New Methods**: 12 advanced analytical methods

## Key Features

### 1. Moving Averages
Calculate moving averages for performance metrics with configurable window sizes.

```python
tracker = ProficiencyTracker(moving_avg_window=20)

# Get moving average metrics
ma = tracker.get_moving_average_metrics("code_review")
print(f"Success rate MA: {ma.success_rate_ma:.1%}")
print(f"Std deviation: {ma.std_dev:.3f}")
print(f"Variance: {ma.variance:.3f}")
```

### 2. Improvement Trajectory Tracking
Track historical improvement over time with automatic snapshots.

```python
# Record a snapshot
tracker.record_trajectory_snapshot("code_review")

# Get trajectory
trajectory = tracker.get_improvement_trajectory("code_review", limit=100)
for point in trajectory:
    print(f"{point.timestamp}: {point.success_rate:.1%} ({point.trend})")
```

### 3. Trend Detection
Automatically detect performance trends (improving, declining, stable).

```python
values = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
trend = tracker.detect_trend_direction(values)
# Returns: TrendDirection.IMPROVING
```

### 4. Top Proficiencies & Weaknesses
Identify strongest and weakest areas.

```python
# Get top N tools
top_tools = tracker.get_top_proficiencies(n=10)
for tool, score in top_tools:
    print(f"{tool}: {score.success_rate:.1%}")

# Find weaknesses
weaknesses = tracker.get_weaknesses(threshold=0.7)
for tool in weaknesses:
    print(f"Needs improvement: {tool}")
```

### 5. Manual Proficiency Updates
Adjust proficiency scores for RL feedback or external adjustments.

```python
# Increase proficiency by 10%
tracker.update_proficiency("code_review", 0.1)
```

### 6. RL Training Data Export
Export data for reinforcement learning model training.

```python
# Export all historical outcomes
df = tracker.export_training_data()
print(df.shape)  # (n_samples, 7)
print(df.columns)
# ['task', 'tool', 'success', 'duration', 'cost', 'quality_score', 'timestamp']

# Export specific task trajectory
history_df = tracker.export_proficiency_history("code_review")
```

### 7. Statistical Analysis
Get comprehensive statistics and pattern analysis.

```python
# Get summary statistics
stats = tracker.get_statistics_summary()
print(f"Total outcomes: {stats['total_outcomes']}")
print(f"Success rate: {stats['success_rate']['average']:.1%}")

# Analyze performance patterns
patterns = tracker.analyze_performance_patterns()
print(f"Improving tools: {len(patterns['improving_tools'])}")
print(f"Declining tools: {len(patterns['declining_tools'])}")
```

## Data Structures

### TaskOutcome
```python
@dataclass
class TaskOutcome:
    success: bool
    duration: float
    cost: float
    errors: List[str] = field(default_factory=list)
    quality_score: float = 1.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### ProficiencyScore
```python
@dataclass
class ProficiencyScore:
    success_rate: float
    avg_execution_time: float
    avg_cost: float
    total_executions: int
    trend: TrendDirection
    last_updated: str
    quality_score: float = 1.0
```

### ImprovementTrajectory
```python
@dataclass
class ImprovementTrajectory:
    task_type: str
    timestamp: str
    success_rate: float
    avg_time: float
    avg_quality: float
    sample_count: int
    moving_avg_success: float
    moving_avg_time: float
    moving_avg_quality: float
    trend: TrendDirection
```

### MovingAverageMetrics
```python
@dataclass
class MovingAverageMetrics:
    window_size: int
    success_rate_ma: float
    execution_time_ma: float
    quality_score_ma: float
    cost_ma: float
    variance: float
    std_dev: float
    min_value: float
    max_value: float
```

## Database Schema

The tracker uses SQLite with the following tables:

### tool_proficiency
- `tool`: Primary key
- `success_count`: Number of successful executions
- `total_count`: Total executions
- `total_duration`: Cumulative execution time
- `total_cost`: Cumulative cost
- `total_quality`: Cumulative quality score
- `last_updated`: ISO timestamp
- `trend`: Current trend direction

### task_outcomes
- `id`: Auto-increment primary key
- `task`: Task type
- `tool`: Tool name
- `success`: Boolean (0/1)
- `duration`: Execution time in seconds
- `cost`: Estimated cost in USD
- `errors`: Comma-separated error messages
- `quality_score`: Quality 0.0-1.0
- `timestamp`: ISO timestamp
- `metadata`: Additional metadata (stringified)

### task_success_rates
- `task`: Primary key
- `success_count`: Number of successes
- `total_count`: Total attempts
- `last_updated`: ISO timestamp

### improvement_trajectory
- `id`: Auto-increment primary key
- `task_type`: Task type name
- `timestamp`: ISO timestamp
- `success_rate`: Success rate at snapshot
- `avg_time`: Average execution time
- `avg_quality`: Average quality score
- `sample_count`: Number of samples
- `moving_avg_success`: Moving average of success rate
- `moving_avg_time`: Moving average of execution time
- `moving_avg_quality`: Moving average of quality score
- `trend`: Trend direction

## Integration with RL System

The proficiency tracker integrates with the RL coordinator for reward shaping and policy optimization:

```python
from victor.agent.improvement import ProficiencyTracker, EnhancedRLCoordinator

tracker = ProficiencyTracker()
rl_coordinator = EnhancedRLCoordinator()

# Record outcome
outcome = TaskOutcome(
    success=True,
    duration=1.5,
    cost=0.001,
    quality_score=0.9
)
tracker.record_outcome("code_review", "ast_analyzer", outcome)

# Get proficiency for reward shaping
score = tracker.get_proficiency("ast_analyzer")
reward = rl_coordinator.reward_shaping(outcome, score)

# Export for training
df = tracker.export_training_data()
# Use df for RL model training
```

## Usage Examples

### Basic Recording
```python
from victor.agent.improvement import ProficiencyTracker, TaskOutcome

tracker = ProficiencyTracker()

# Record an outcome
outcome = TaskOutcome(
    success=True,
    duration=2.5,
    cost=0.002,
    quality_score=0.85
)
tracker.record_outcome("code_review", "ast_analyzer", outcome)

# Get proficiency
score = tracker.get_proficiency("ast_analyzer")
print(f"Success rate: {score.success_rate:.1%}")
print(f"Trend: {score.trend}")
```

### Advanced Analytics
```python
# Moving averages
ma = tracker.get_moving_average_metrics("code_review", window_size=20)
print(f"Success rate MA: {ma.success_rate_ma:.1%} ± {ma.std_dev:.3f}")

# Trajectory tracking
tracker.record_trajectory_snapshot("code_review")
trajectory = tracker.get_improvement_trajectory("code_review")
print(f"Recorded {len(trajectory)} trajectory points")

# Pattern analysis
patterns = tracker.analyze_performance_patterns()
print(f"Improving tools: {len(patterns['improving_tools'])}")
print(f"Fastest tools: {[t['tool'] for t in patterns['fastest_tools']]}")
```

### RL Training Pipeline
```python
import pandas as pd

# Export training data
df = tracker.export_training_data()

# Feature engineering
df['success_rate'] = df.groupby('task')['success'].transform('mean')
df['avg_duration'] = df.groupby('tool')['duration'].transform('mean')

# Split for training
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

# Train RL model
# model.fit(train_df[['task', 'tool', 'success_rate', 'avg_duration']],
#          train_df['success'])
```

## Performance Considerations

- **Moving Average Window**: Default 20 samples. Larger windows = smoother trends but slower response.
- **Database Indexing**: Automatic indexing on task, tool, and timestamp for fast queries.
- **Caching**: Proficiency scores cached in memory to reduce database queries.
- **Batch Recording**: For high-volume scenarios, consider batch recording with transactions.

## Dependencies

### Required
- `sqlite3`: Built-in Python library
- `dataclasses`: Built-in (Python 3.7+)
- `typing`: Built-in

### Optional
- `pandas`: Required for `export_training_data()` and `export_proficiency_history()`
  ```bash
  pip install pandas
  ```

## Testing

Run the test suite:

```bash
# Unit tests
pytest tests/unit/agent/improvement/test_proficiency_tracker.py -v

# Integration tests
pytest tests/integration/agent/improvement/test_rl_integration.py -v

# Demo
python examples/proficiency_tracker_demo.py
```

## File Structure

```
victor/agent/improvement/
├── __init__.py                    # Public API exports
├── proficiency_tracker.py         # Main tracker (1,494 lines)
└── rl_coordinator.py              # RL integration

examples/
└── proficiency_tracker_demo.py    # Usage demo

tests/
├── unit/agent/improvement/
│   └── test_proficiency_tracker.py
└── integration/agent/improvement/
    └── test_rl_integration.py
```

## API Reference

See the complete API reference in the module docstring of `proficiency_tracker.py`.

## Contributing

When adding new features:
1. Add method to `ProficiencyTracker` class
2. Update `__init__.py` to export new types
3. Add unit tests in `tests/unit/agent/improvement/`
4. Update this documentation
5. Run the demo to verify functionality

## License

Apache License 2.0 - See LICENSE file for details.
