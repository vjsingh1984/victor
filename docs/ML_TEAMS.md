# ML-Powered Teams Guide

Victor's machine learning capabilities enable intelligent team optimization, member selection, formation prediction, and performance estimation. This guide covers all ML-powered features for teams.

## Table of Contents

- [Overview](#overview)
- [Team Performance Prediction](#team-performance-prediction)
- [Team Member Selection](#team-member-selection)
- [Formation Prediction](#formation-prediction)
- [Team Optimization](#team-optimization)
- [Team Learning](#team-learning)
- [Training Your Own Models](#training-your-own-models)
- [Best Practices](#best-practices)

## Overview

Victor provides ML-powered capabilities for:

1. **Performance Prediction** - Predict execution time, success rate, quality score
2. **Member Selection** - Select optimal team members using ML
3. **Formation Prediction** - Predict best formation for task
4. **Team Optimization** - Optimize team configuration automatically
5. **Team Learning** - Learn from experience and adapt

### Key Benefits

- **Data-Driven Decisions**: Use historical data to guide choices
- **Continuous Improvement**: Learn from each execution
- **Adaptive Teams**: Automatically adapt based on performance
- **Resource Optimization**: Maximize efficiency with predictions

## Team Performance Prediction

The `TeamPredictor` provides predictions for various team performance metrics.

### Basic Usage

```python
from victor.teams import TeamPredictor, PredictionMetric

predictor = TeamPredictor()

# Load historical data for better predictions
predictor.load_historical_data("team_executions.jsonl")

# Predict execution time
time_result = predictor.predict(
    metric=PredictionMetric.EXECUTION_TIME,
    team_config=team_config,
    task="Implement OAuth authentication",
    context={"file_count": 5, "complexity": 0.7}
)

print(f"Predicted time: {time_result.predicted_value}s")
print(f"Confidence: {time_result.confidence:.1%}")
print(f"Reasoning: {time_result.explanation}")
```

### Supported Predictions

#### Execution Time

```python
result = predictor.predict(
    metric=PredictionMetric.EXECUTION_TIME,
    team_config=team_config,
    task="Task description"
)
```

#### Success Probability

```python
result = predictor.predict(
    metric=PredictionMetric.SUCCESS_PROBABILITY,
    team_config=team_config,
    task="Task description"
)
```

#### Tool Call Usage

```python
result = predictor.predict(
    metric=PredictionMetric.TOOL_CALLS,
    team_config=team_config,
    task="Task description"
)
```

#### Quality Score

```python
result = predictor.predict(
    metric=PredictionMetric.QUALITY_SCORE,
    team_config=team_config,
    task="Task description"
)
```

#### Formation Suitability

```python
result = predictor.predict(
    metric=PredictionMetric.FORMATION_SUITABILITY,
    team_config=team_config,
    task="Task description"
)

print(f"Recommended formation: {result.predicted_value}")
print(f"Alternatives: {result.metadata['formation_scores']}")
```

### Recording Executions

```python
# Track execution for future predictions
predictor.record_execution(
    team_config=team_config,
    task="Implement feature",
    context={"domain": "security", "urgency": 0.8},
    execution_result={
        "success": True,
        "total_duration": 145.5,
        "total_tool_calls": 87,
        "quality_score": 0.9,
        "iteration_count": 7
    }
)

# Save historical data periodically
predictor.save_historical_data(Path("team_executions.jsonl"))
```

### Historical Data Format

```json
{
  "team_config_hash": "abc123",
  "task_features": {
    "complexity": 0.7,
    "estimated_lines_of_code": 500,
    "file_count": 5,
    "domain": "security",
    "required_expertise": ["oauth", "jwt"],
    "urgency": 0.8
  },
  "team_features": {
    "member_count": 3,
    "formation": "parallel",
    "total_tool_budget": 150,
    "expertise_coverage": 0.8,
    "diversity": 0.6
  },
  "formation": "parallel",
  "execution_time": 145.5,
  "success": true,
  "tool_calls_used": 87,
  "quality_score": 0.9,
  "iteration_count": 7
}
```

## Team Member Selection

The `TeamMemberSelector` uses ML to score and rank team members based on task fit.

### Basic Usage

```python
from victor.teams import TeamMemberSelector
from victor.teams.team_predictor import TaskFeatures

selector = TeamMemberSelector()

# Define task features
task_features = TaskFeatures(
    complexity=0.7,
    estimated_lines_of_code=500,
    file_count=5,
    domain="security",
    required_expertise=["oauth", "jwt", "authentication"],
    urgency=0.8
)

# Select optimal members
selected = selector.select_members(
    task="Implement OAuth authentication",
    available_members=member_pool,
    task_features=task_features,
    top_k=5
)

# View results
for score in selected:
    print(f"{score.member.name}: {score.score:.3f}")
    print(f"  Confidence: {score.confidence:.1%}")
    print(f"  Reasons: {', '.join(score.reasons)}")
```

### Training Custom Models

```python
# Prepare training data
training_data = []

for historical_execution in history:
    training_data.append({
        "member": member,
        "task_features": task_features,
        "success": execution_success  # True/False
    })

# Train model
selector.train(training_data)

# Save model
selector.save_model(Path("member_selector.pkl"))
```

### Using Trained Models

```python
# Load trained model
selector = TeamMemberSelector(model_path="member_selector.pkl")

# Use for predictions
selected = selector.select_members(
    task="Task description",
    available_members=members,
    task_features=features,
    top_k=3
)
```

### Feature Importance

```python
# Understand what drives selection
importance = selector.get_feature_importance()

for feature, score in list(importance.items())[:10]:
    print(f"{feature}: {score:.3f}")
```

## Formation Prediction

The `FormationPredictor` recommends optimal team formations based on task and team characteristics.

### Basic Usage

```python
from victor.teams import FormationPredictor
from victor.teams.team_predictor import TaskFeatures, TeamFeatures

predictor = FormationPredictor()

# Extract features
task_features = TaskFeatures.from_task(task, context)
team_features = TeamFeatures.from_team_config(team_config, task_features)

# Predict formation
prediction = predictor.predict_formation(
    task_features=task_features,
    team_features=team_features
)

print(f"Recommended: {prediction.formation.value}")
print(f"Confidence: {prediction.confidence:.1%}")
print(f"Reasoning: {prediction.reasoning}")
print(f"All probabilities:")
for formation, prob in prediction.probabilities.items():
    print(f"  {formation}: {prob:.1%}")
```

### Training Custom Models

```python
# Prepare training data
training_data = []

for execution in history:
    training_data.append({
        "task_features": task_features,
        "team_features": team_features,
        "formation": formation,  # Label
        "success": success  # For weighting
    })

# Train model
predictor.train(training_data)

# Save model
predictor.save_model(Path("formation_predictor.pkl"))
```

## Team Optimization

The `TeamOptimizer` automatically finds optimal team configurations using various algorithms.

### Basic Usage

```python
from victor.teams import TeamOptimizer, OptimizationObjective, OptimizationConstraints

optimizer = TeamOptimizer(
    optimizer_type="genetic",  # or "greedy", "bayesian", "random"
    population_size=50,
    generations=20
)

# Define constraints
constraints = OptimizationConstraints(
    max_members=5,
    min_members=2,
    max_budget=200,
    min_budget=50,
    required_expertise=["security", "testing"],
    allowed_formations=["parallel", "hierarchical"],
    max_execution_time=300,
    min_success_probability=0.7
)

# Optimize team
result = optimizer.optimize_team(
    task="Implement secure authentication",
    available_members=member_pool,
    objective=OptimizationObjective.BALANCED,
    constraints=constraints,
    context={"urgency": 0.8}
)

print(f"Optimal configuration: {result.optimal_config.name}")
print(f"Score: {result.score:.3f}")
print(f"Predicted metrics:")
for metric, value in result.predicted_metrics.items():
    print(f"  {metric}: {value}")

print(f"\nAlternatives:")
for config, score in result.alternative_configs[:3]:
    print(f"  {config.name}: {score:.3f}")
```

### Optimization Objectives

```python
# Minimize execution time
result = optimizer.optimize_team(
    task=task,
    available_members=members,
    objective=OptimizationObjective.MINIMIZE_TIME,
    constraints=constraints
)

# Maximize success rate
result = optimizer.optimize_team(
    task=task,
    available_members=members,
    objective=OptimizationObjective.MAXIMIZE_SUCCESS,
    constraints=constraints
)

# Balance all objectives
result = optimizer.optimize_team(
    task=task,
    available_members=members,
    objective=OptimizationObjective.BALANCED,
    constraints=constraints
)
```

### Optimization Algorithms

#### Greedy
Fast, good for small search spaces:

```python
optimizer = TeamOptimizer(optimizer_type="greedy")
```

#### Genetic Algorithm
Global optimization for complex spaces:

```python
optimizer = TeamOptimizer(
    optimizer_type="genetic",
    population_size=50,
    generations=20,
    mutation_rate=0.1,
    crossover_rate=0.7
)
```

#### Random Search
Baseline comparison:

```python
optimizer = TeamOptimizer(
    optimizer_type="random",
    max_iterations=100
)
```

## Team Learning

The `TeamLearningSystem` enables teams to learn from experience and improve over time.

### Basic Usage

```python
from victor.teams import TeamLearningSystem

learner = TeamLearningSystem(
    learning_strategy=LearningStrategy.REINFORCEMENT,
    learning_rate=0.1,
    discount_factor=0.95
)

# Record experience
learner.record_experience(
    team_config=team_config,
    task="Implement feature",
    result=execution_result,
    team_id="my_team",
    reward=1.0,  # or compute automatically
    metadata={"domain": "security"}
)

# Get learning progress
progress = learner.get_progress(team_id="my_team")
print(f"Total experiences: {progress.total_experiences}")
print(f"Average reward: {progress.avg_reward:.3f}")
print(f"Skill level: {progress.skill_level:.1%}")
print(f"Trend: {progress.reward_trend}")
```

### Getting Recommendations

```python
# Get improvement recommendations
recommendations = learner.get_recommendations(
    team_id="my_team",
    top_k=5
)

for rec in recommendations:
    print(f"{rec.recommendation_type}: {rec.description}")
    print(f"  Expected improvement: {rec.expected_improvement:.1%}")
    print(f"  Confidence: {rec.confidence:.1%}")
    print(f"  Changes: {rec.changes}")
    print(f"  Rationale: {rec.rationale}")
```

### Optimal Formation Selection

```python
# Get learned optimal formation
formation = learner.get_optimal_formation(
    task="Implement authentication",
    team_id="my_team"
)

if formation:
    print(f"Optimal formation: {formation}")
```

### Learning Strategies

#### Reinforcement Learning
Learn from rewards/penalties:

```python
learner = TeamLearningSystem(
    learning_strategy=LearningStrategy.REINFORCEMENT
)
```

#### Supervised Learning
Learn from labeled examples:

```python
learner = TeamLearningSystem(
    learning_strategy=LearningStrategy.SUPERVISED
)
```

### Reward Computation

Rewards are computed automatically if not provided:

```python
# Default reward computation
reward = learner._compute_reward(result)

# Based on:
# - Success: +0.5 for success, -0.5 for failure
# - Quality: (quality - 0.5) * 0.5
# - Speed: (100/time - 1) * 0.1
# - Efficiency: (1 - tool_calls/budget) * 0.2
```

## Training Your Own Models

### Data Collection

```python
from victor.teams import TeamAnalytics

# Collect execution data
analytics = TeamAnalytics()

# Track executions
for execution in executions:
    analytics.track_execution(
        team_config=config,
        task=task,
        result=result,
        team_id="team_name"
    )

# Export for training
analytics.export_report(
    team_id="team_name",
    output_path=Path("training_data.json")
)
```

### Model Training Pipeline

```python
# 1. Collect historical data
historical_data = collect_team_executions()

# 2. Prepare training datasets
member_data = prepare_member_training_data(historical_data)
formation_data = prepare_formation_training_data(historical_data)
performance_data = prepare_performance_training_data(historical_data)

# 3. Train models
member_selector = TeamMemberSelector()
member_selector.train(member_data)
member_selector.save_model("member_selector.pkl")

formation_predictor = FormationPredictor()
formation_predictor.train(formation_data)
formation_predictor.save_model("formation_predictor.pkl")

performance_predictor = PerformancePredictor()
performance_predictor.train_execution_time_model(performance_data)
performance_predictor.train_success_model(performance_data)
performance_predictor.save_models("performance_predictor.pkl")
```

### Model Evaluation

```python
# Split data
train_data, test_data = split_data(data, train_ratio=0.8)

# Train on training set
selector.train(train_data)

# Evaluate on test set
correct = 0
total = len(test_data)

for example in test_data:
    selected = selector.select_members(
        task=example["task"],
        available_members=example["available_members"],
        task_features=example["task_features"],
        top_k=1
    )

    if selected[0].member.id == example["best_member"].id:
        correct += 1

accuracy = correct / total
print(f"Accuracy: {accuracy:.1%}")
```

## Best Practices

### 1. Collect Quality Data

```python
# Ensure consistent data collection
analytics.track_execution(
    team_config=config,
    task=task,
    result=result,
    team_id="team_name"  # Use consistent team IDs
)
```

### 2. Use Appropriate Features

```python
# Good: Specific, measurable features
task_features = TaskFeatures(
    complexity=0.7,
    estimated_lines_of_code=500,
    file_count=5,
    domain="security"
)

# Avoid: Vague features
task_features = TaskFeatures(
    complexity=0.5,  # Default
    estimated_lines_of_code=100,  # Default
    file_count=1  # Default
)
```

### 3. Train with Sufficient Data

```python
# Minimum recommended training data sizes:
# - Member selection: 100+ examples
# - Formation prediction: 200+ examples
# - Performance prediction: 500+ examples

if len(training_data) < 100:
    logger.warning("Insufficient training data")
    # Use heuristic instead
    predictor = TeamPredictor(use_heuristic=True)
```

### 4. Validate Models

```python
# Use cross-validation
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, features, labels, cv=5)
print(f"Mean accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
```

### 5. Update Models Regularly

```python
# Retrain periodically with new data
if should_retrain(last_training_time, new_data_count):
    logger.info("Retraining models...")
    selector.train(new_training_data)
    selector.save_model("member_selector.pkl")
```

### 6. Monitor Predictions

```python
# Track prediction accuracy
predictions = []
actuals = []

for execution in test_executions:
    prediction = predictor.predict(..., team_config, task)
    predictions.append(prediction.predicted_value)
    actuals.append(execution.actual_value)

# Compute metrics
mae = mean_absolute_error(actuals, predictions)
print(f"Mean Absolute Error: {mae:.2f}")
```

## Troubleshooting

### Poor Predictions

```python
# Check historical data quality
stats = predictor.get_historical_stats()
print(f"Total records: {stats['total_records']}")
print(f"Success rate: {stats['success_rate']:.1%}")

# Need more data?
if stats['total_records'] < 100:
    print("Collect more training data")
```

### Model Not Loading

```python
# Use heuristic fallback
predictor = TeamPredictor(
    model_path=None,  # Don't load model
    use_heuristic=True  # Use heuristics
)
```

### Overfitting

```python
# Use regularization
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,  # Limit depth
    min_samples_split=5,  # Require more samples
    random_state=42
)
```

## Further Reading

- [Team Formations Guide](TEAM_FORMATIONS.md) - Using advanced formations
- [Team Analytics Guide](TEAM_ANALYTICS.md) - Tracking team performance
- [Team Optimization Guide](../workflows/README.md) - Workflow optimization
