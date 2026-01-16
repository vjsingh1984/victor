# Team Analytics Guide

Victor's team analytics system provides comprehensive tracking, analysis, and insights for team performance. This guide covers all analytics capabilities.

## Table of Contents

- [Overview](#overview)
- [Tracking Executions](#tracking-executions)
- [Performance Metrics](#performance-metrics)
- [Bottleneck Detection](#bottleneck-detection)
- [Team Comparison](#team-comparison)
- [Member Analysis](#member-analysis)
- [Formation Analysis](#formation-analysis)
- [Exporting Reports](#exporting-reports)
- [Best Practices](#best-practices)

## Overview

The `TeamAnalytics` system provides:

- **Execution Tracking**: Record all team executions
- **Performance Metrics**: Compute key performance indicators
- **Bottleneck Detection**: Identify performance issues
- **Team Comparison**: Compare different team configurations
- **Member Ranking**: Rank members by performance
- **Formation Effectiveness**: Analyze formation performance
- **Report Generation**: Export detailed analytics reports

### Key Metrics Tracked

- Execution time (total, per-member)
- Success rate
- Tool call usage
- Quality scores
- Iteration counts
- Member utilization
- Formation effectiveness

## Tracking Executions

### Basic Tracking

```python
from victor.teams import TeamAnalytics
from victor.teams.team_analytics import ExecutionEvent

analytics = TeamAnalytics(storage_path=Path("team_analytics.json"))

# Track execution
execution_id = analytics.track_execution(
    team_config=team_config,
    task="Implement OAuth authentication",
    result=execution_result,
    team_id="auth_team",
    events=[
        ExecutionEvent(
            timestamp=datetime.now(),
            event_type="formation_switch",
            data={"from": "sequential", "to": "parallel"}
        )
    ]
)

print(f"Tracked execution: {execution_id}")
```

### Execution Events

Track specific events during execution:

```python
events = [
    ExecutionEvent(
        timestamp=datetime.now(),
        event_type="member_started",
        member_id="researcher",
        data={"phase": "planning"}
    ),
    ExecutionEvent(
        timestamp=datetime.now(),
        event_type="formation_switch",
        data={"from": "sequential", "to": "parallel"}
    ),
    ExecutionEvent(
        timestamp=datetime.now(),
        event_type="bottleneck_detected",
        member_id="executor",
        data={"issue": "slow_execution"}
    )
]

analytics.track_execution(
    team_config=team_config,
    task=task,
    result=result,
    team_id="team_name",
    events=events
)
```

### Auto-Tracking with Coordinator

```python
from victor.teams import UnifiedTeamCoordinator

coordinator = UnifiedTeamCoordinator(orchestrator)
coordinator.add_member(member1).add_member(member2)

# Execute with analytics tracking
result = await coordinator.execute_task(task, context)

# Track result
analytics.track_execution(
    team_config=config_from_coordinator(coordinator),
    task=task,
    result=result,
    team_id="my_team"
)
```

## Performance Metrics

### Team Statistics

```python
# Get comprehensive team stats
stats = analytics.get_team_stats("auth_team")

print(f"Total executions: {stats['total_executions']}")
print(f"Success rate: {stats['success_rate']:.1%}")
print(f"Avg execution time: {stats['avg_execution_time']:.2f}s")

print(f"\nMember Performance:")
for member_id, member_stats in stats['member_performance'].items():
    print(f"  {member_id}:")
    print(f"    Executions: {member_stats['executions']}")
    print(f"    Success rate: {member_stats['success_rate']:.1%}")
    print(f"    Avg time: {member_stats['avg_time']:.2f}s")
    print(f"    Avg tool calls: {member_stats['avg_tool_calls']:.1f}")
```

### Member-Level Metrics

```python
stats = analytics.get_team_stats("my_team")

for member_id, member_stats in stats['member_performance'].items():
    print(f"\n{member_id}:")
    print(f"  Success Rate: {member_stats['success_rate']:.1%}")
    print(f"  Avg Time: {member_stats['avg_time']:.2f}s")
    print(f"  Avg Tool Calls: {member_stats['avg_tool_calls']:.1f}")
    print(f"  Executions: {member_stats['executions']}")
```

### Metric Types Available

```python
from victor.teams.team_analytics import MetricType

# Tracked metrics:
- MetricType.EXECUTION_TIME
- MetricType.SUCCESS_RATE
- MetricType.TOOL_CALLS
- MetricType.QUALITY_SCORE
- MetricType.ITERATION_COUNT
- MetricType.MEMBER_UTILIZATION
- MetricType.FORMATION_EFFECTIVENESS
```

## Bottleneck Detection

### Detecting Bottlenecks

```python
# Identify performance bottlenecks
bottlenecks = analytics.detect_bottlenecks("auth_team")

print(f"Found {len(bottlenecks)} bottlenecks:\n")

for bottleneck in bottlenecks:
    print(f"{bottleneck.bottleneck_type}: {bottleneck.description}")
    print(f"  Severity: {bottleneck.severity:.1%}")
    print(f"  Affected members: {', '.join(bottleneck.affected_members)}")
    print(f"  Suggested fixes:")
    for fix in bottleneck.suggested_fixes:
        print(f"    - {fix}")
    print()
```

### Bottleneck Types

#### Slow Members

```python
BottleneckInfo(
    bottleneck_type="slow_member",
    severity=0.8,
    affected_members=["executor"],
    description="Member executor takes 145s on average",
    suggested_fixes=[
        "Increase member tool budget",
        "Optimize member's assigned tasks",
        "Consider parallelizing member's work"
    ]
)
```

#### High Failure Rate

```python
BottleneckInfo(
    bottleneck_type="high_failure_rate",
    severity=0.9,
    affected_members=["tester"],
    description="Member tester fails 45% of the time",
    suggested_fixes=[
        "Review member's expertise alignment",
        "Provide member with better context",
        "Consider replacing member"
    ]
)
```

#### Tool Budget Exhaustion

```python
BottleneckInfo(
    bottleneck_type="tool_budget_exhaustion",
    severity=0.7,
    affected_members=["all"],
    description="Team frequently exhausts tool budget",
    suggested_fixes=[
        "Increase total tool budget",
        "Optimize tool usage",
        "Use more efficient tools"
    ]
)
```

### Proactive Bottleneck Monitoring

```python
# Set up monitoring
def monitor_team_performance(team_id: str):
    bottlenecks = analytics.detect_bottlenecks(team_id)

    critical_bottlenecks = [
        b for b in bottlenecks if b.severity > 0.7
    ]

    if critical_bottlenecks:
        logger.warning(f"Found {len(critical_bottlenecks)} critical bottlenecks")
        for b in critical_bottlenecks:
            logger.error(f"{b.bottleneck_type}: {b.description}")

        return critical_bottlenecks
    return []

# Run periodically
critical_issues = monitor_team_performance("auth_team")
```

## Team Comparison

### Comparing Two Teams

```python
# Compare team configurations
comparison = analytics.compare_teams("team_a", "team_b")

if comparison:
    print(f"Overall winner: {comparison.overall_winner}")
    print(f"Confidence: {comparison.confidence:.1%}\n")

    print("Metric Comparisons:")
    for metric, comp in comparison.metric_comparisons.items():
        print(f"\n{metric}:")
        print(f"  Team A: {comp['team1']:.3f}")
        print(f"  Team B: {comp['team2']:.3f}")
        print(f"  Difference: {comp['difference']:.3f}")
        print(f"  Winner: {comp['winner']}")

    print("\nKey Insights:")
    for insight in comparison.insights:
        print(f"  - {insight}")
```

### Batch Comparison

```python
# Compare multiple team configurations
teams = ["team_a", "team_b", "team_c"]
comparisons = {}

for i, team1 in enumerate(teams):
    for team2 in teams[i+1:]:
        comparison = analytics.compare_teams(team1, team2)
        if comparison:
            comparisons[f"{team1}_vs_{team2}"] = comparison

# Find best team
win_counts = {team: 0 for team in teams}

for comp in comparisons.values():
    if comp.overall_winner:
        win_counts[comp.overall_winner] += 1

best_team = max(win_counts, key=win_counts.get)
print(f"Best overall team: {best_team}")
```

## Member Analysis

### Member Ranking

```python
# Rank members by performance
rankings = analytics.get_member_ranking(team_id="auth_team")

print("Member Performance Rankings:\n")
for rank, (member_id, score) in enumerate(rankings[:10], 1):
    print(f"{rank}. {member_id}: {score:.3f}")
```

### Top Performers

```python
rankings = analytics.get_member_ranking()

top_performers = rankings[:5]
print("Top 5 Performers:")
for member_id, score in top_performers:
    print(f"  {member_id}: {score:.3f}")

bottom_performers = rankings[-5:]
print("\nBottom 5 Performers:")
for member_id, score in reversed(bottom_performers):
    print(f"  {member_id}: {score:.3f}")
```

### Member Progress Tracking

```python
def track_member_progress(member_id: str, team_id: str):
    stats = analytics.get_team_stats(team_id)

    if member_id in stats['member_performance']:
        member_stats = stats['member_performance'][member_id]

        print(f"{member_id} Performance:")
        print(f"  Total executions: {member_stats['executions']}")
        print(f"  Success rate: {member_stats['success_rate']:.1%}")
        print(f"  Avg time: {member_stats['avg_time']:.2f}s")

        # Compare to team average
        team_avg_time = stats['avg_execution_time']
        if member_stats['avg_time'] > team_avg_time * 1.2:
            print(f"  ⚠️  20% slower than team average")
        elif member_stats['avg_time'] < team_avg_time * 0.8:
            print(f"  ✓  20% faster than team average")

track_member_progress("researcher", "auth_team")
```

## Formation Analysis

### Formation Effectiveness

```python
# Get formation performance metrics
formation_metrics = analytics.get_formation_effectiveness()

print("Formation Performance:\n")
for formation, metrics in sorted(
    formation_metrics.items(),
    key=lambda x: x[1]['avg_time']
):
    print(f"{formation}:")
    print(f"  Avg time: {metrics['avg_time']:.2f}s")
    print(f"  Min time: {metrics['min_time']:.2f}s")
    print(f"  Max time: {metrics['max_time']:.2f}s")
    print(f"  Std dev: {metrics['std_time']:.2f}s")
    print(f"  Executions: {metrics['count']}")
    print()
```

### Formation Selection Insights

```python
# Find best formation for your use case
formation_metrics = analytics.get_formation_effectiveness()

# Fastest formation
fastest = min(
    formation_metrics.items(),
    key=lambda x: x[1]['avg_time']
)
print(f"Fastest: {fastest[0]} ({fastest[1]['avg_time']:.2f}s)")

# Most consistent
most_consistent = min(
    formation_metrics.items(),
    key=lambda x: x[1]['std_time']
)
print(f"Most consistent: {most_consistent[0]} ({most_consistent[1]['std_time']:.2f}s std)")
```

## Exporting Reports

### Generate Team Report

```python
# Export comprehensive report
analytics.export_report(
    team_id="auth_team",
    output_path=Path("auth_team_report.json")
)

print("Report exported to auth_team_report.json")
```

### Report Contents

```json
{
  "team_id": "auth_team",
  "generated_at": "2025-01-15T10:30:00",
  "statistics": {
    "team_id": "auth_team",
    "total_executions": 50,
    "success_rate": 0.85,
    "avg_execution_time": 123.5,
    "member_performance": {
      "researcher": {
        "executions": 50,
        "successes": 45,
        "success_rate": 0.9,
        "avg_time": 45.2,
        "avg_tool_calls": 23
      }
    }
  },
  "bottlenecks": [
    {
      "bottleneck_type": "slow_member",
      "severity": 0.75,
      "affected_members": ["executor"],
      "description": "Member executor takes 145s on average",
      "suggested_fixes": [
        "Increase member tool budget",
        "Optimize member's assigned tasks"
      ]
    }
  ],
  "recommendations": [
    "Increase total tool budget",
    "Optimize member's assigned tasks",
    "Consider replacing member with high failure rate"
  ]
}
```

### Custom Reports

```python
def generate_custom_report(team_id: str, output_path: Path):
    stats = analytics.get_team_stats(team_id)
    bottlenecks = analytics.detect_bottlenecks(team_id)
    rankings = analytics.get_member_ranking(team_id)

    report = {
        "team_id": team_id,
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_executions": stats['total_executions'],
            "success_rate": stats['success_rate'],
            "avg_time": stats['avg_execution_time']
        },
        "top_performers": [
            {"member_id": m, "score": s}
            for m, s in rankings[:5]
        ],
        "critical_bottlenecks": [
            b.to_dict() for b in bottlenecks if b.severity > 0.7
        ],
        "recommendations": generate_recommendations(bottlenecks)
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

generate_custom_report("auth_team", Path("custom_report.json"))
```

## Best Practices

### 1. Consistent Team IDs

```python
# Good: Consistent team IDs
analytics.track_execution(..., team_id="auth_team_v1")

# Avoid: Inconsistent team IDs
analytics.track_execution(..., team_id="auth")
analytics.track_execution(..., team_id="authentication")
analytics.track_execution(..., team_id="AuthTeam")
```

### 2. Track All Executions

```python
# Track both successful and failed executions
analytics.track_execution(
    team_config=config,
    task=task,
    result=result,  # Include failed results
    team_id=team_id
)
```

### 3. Use Events for Context

```python
# Track important events
events = [
    ExecutionEvent(..., event_type="formation_switch"),
    ExecutionEvent(..., event_type="bottleneck_detected"),
    ExecutionEvent(..., event_type="member_timeout")
]
```

### 4. Regular Analysis

```python
# Schedule periodic analysis
import schedule

def weekly_analysis():
    teams = ["auth_team", "code_review_team", "testing_team"]

    for team_id in teams:
        bottlenecks = analytics.detect_bottlenecks(team_id)
        if any(b.severity > 0.7 for b in bottlenecks):
            alert_team(team_id, bottlenecks)

schedule.every().monday.at(9am).do(weekly_analysis)
```

### 5. Data Retention

```python
# Archive old data periodically
def archive_old_data(days_to_keep=90):
    cutoff = datetime.now() - timedelta(days=days_to_keep)

    # Save old data
    analytics.save_data(Path("archive.json"))

    # Clear old data (implement based on your needs)
    # analytics._clear_old_data(cutoff)
```

### 6. Monitor Trends

```python
def track_trends(team_id: str, window=10):
    stats = analytics.get_team_stats(team_id)

    # Compute trends over time
    recent_executions = get_recent_executions(team_id, window)

    success_trend = compute_trend([
        e.success for e in recent_executions
    ])

    time_trend = compute_trend([
        e.duration for e in recent_executions
    ])

    print(f"Success rate trend: {success_trend}")
    print(f"Execution time trend: {time_trend}")
```

## Integrations

### With Team Learning

```python
# Combine analytics with learning
from victor.teams import TeamLearningSystem

learner = TeamLearningSystem()

# Use analytics to identify improvements
bottlenecks = analytics.detect_bottlenecks("my_team")

# Get recommendations
recommendations = learner.get_recommendations("my_team")

# Apply changes
for rec in recommendations:
    if rec.recommendation_type == "formation_change":
        team.formation = rec.changes["formation"]
```

### With Optimization

```python
# Use analytics data for optimization
from victor.teams import TeamOptimizer

optimizer = TeamOptimizer()

# Use analytics to inform constraints
stats = analytics.get_team_stats("current_team")

constraints = OptimizationConstraints(
    max_members=5,
    max_budget=int(stats['avg_execution_time'] * 1.2),
    min_success_probability=stats['success_rate']
)

result = optimizer.optimize_team(
    task=task,
    available_members=members,
    constraints=constraints
)
```

## Troubleshooting

### No Data Available

```python
# Check if data exists
stats = analytics.get_team_stats("team_id")

if stats['total_executions'] == 0:
    print("No execution data found")
    print("Track some executions first with analytics.track_execution()")
```

### Inconsistent Metrics

```python
# Ensure consistent tracking
# Always track with same team_id
# Always include required fields in result
required_fields = ["success", "total_duration", "total_tool_calls"]
```

### Large Data Files

```python
# Archive old data
if analytics.storage_path.stat().st_size > 10_000_000:  # 10MB
    backup_path = Path(f"backup_{datetime.now():%Y%m%d}.json")
    analytics.save_data(backup_path)

    # Clear old records
    analytics._executions.clear()
```

## Further Reading

- [Team Formations Guide](TEAM_FORMATIONS.md) - Using advanced formations
- [ML-Powered Teams](ML_TEAMS.md) - ML for team optimization
- [Performance Benchmarking](../benchmarking/README.md) - Benchmarking teams
