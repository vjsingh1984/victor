# Observability Dashboards Guide - Part 1

**Part 1 of 2:** Overview, Installation, Dashboard Guide, Alerting, Alert Tuning, Query Examples, and Integration with Existing Monitoring

---

## Navigation

- **[Part 1: Dashboard Setup & Usage](#)** (Current)
- [Part 2: Advanced & Maintenance](part-2-advanced-maintenance.md)
- [**Complete Guide](../dashboards.md)**

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Dashboard Guide](#dashboard-guide)
4. [Alerting](#alerting)
5. [Alert Tuning](#alert-tuning)
6. [Query Examples](#query-examples)
7. [Integration with Existing Monitoring](#integration-with-existing-monitoring)
8. [Troubleshooting](#troubleshooting) *(in Part 2)*
9. [Best Practices](#best-practices) *(in Part 2)*
10. [Advanced Usage](#advanced-usage) *(in Part 2)*

---

# Victor Team Observability Dashboards

Comprehensive monitoring and alerting for Victor team workflow execution.

## Overview

Victor provides a complete observability solution for team workflows with four Grafana dashboards and Prometheus alerting rules. These dashboards provide real-time insights into team execution, performance, member activity, and recursion depth.

### Dashboards

1. **Team Overview** (`victor-team-overview`) - High-level metrics and KPIs
2. **Team Performance** (`victor-team-performance`) - Formation performance comparison
3. **Team Recursion** (`victor-team-recursion`) - Recursion depth monitoring
4. **Team Members** (`victor-team-members`) - Member-level performance metrics

## Installation

### Prerequisites

- Grafana 9.0+ installed and running
- Prometheus 2.30+ installed and running
- Victor metrics collection enabled (see [team_metrics.md](team_metrics.md))

### Quick Start

1. **Install dependencies:**
   ```bash
   # Install Grafana (example for Ubuntu)
   sudo apt-get install -y grafana

   # Install Prometheus (example for Ubuntu)
   sudo apt-get install -y prometheus
   ```

2. **Configure Prometheus to scrape Victor metrics:**
   ```yaml
   # /etc/prometheus/prometheus.yml
   scrape_configs:
     - job_name: 'victor'
       static_configs:
         - targets: ['localhost:8000']  # Victor metrics endpoint
       scrape_interval: 15s
   ```

3. **Run the setup script:**
   ```bash
   python scripts/observability/setup_dashboards.py
   ```

4. **Access dashboards:**
   - Open Grafana: http://localhost:3000
   - Navigate to Dashboards → Victor

### Manual Installation

#### 1. Create Grafana Datasource

In Grafana UI:
- Go to Configuration → Data Sources
- Add Prometheus data source
- URL: `http://localhost:9090`
- Click "Save & Test"

#### 2. Import Dashboards

Via Grafana UI:
- Go to Dashboards → Import
- Upload dashboard JSON from `observability/dashboards/`
- Select Prometheus datasource
- Click "Import"

Via API:
```bash
export GRAFANA_URL="http://localhost:3000"
export GRAFANA_API_KEY="your-api-key"

python scripts/observability/setup_dashboards.py
```

#### 3. Configure Prometheus Alerting Rules

Copy alerting rules to Prometheus configuration:
```bash
sudo cp observability/alerts/team_alerts.yml /etc/prometheus/rules/

# Add to prometheus.yml
cat >> /etc/prometheus/prometheus.yml << EOF
rule_files:
  - "/etc/prometheus/rules/team_alerts.yml"
EOF

# Reload Prometheus
sudo killall -HUP prometheus
```

### Environment Configuration

Configure via environment variables:

```bash
# Grafana configuration
export GRAFANA_URL="http://localhost:3000"
export GRAFANA_API_KEY="eyJrIjoi...your-api-key"

# Prometheus configuration
export PROMETHEUS_URL="http://localhost:9090"
export PROMETHEUS_CONFIG_PATH="/etc/prometheus/prometheus.yml"

# Run setup
python scripts/observability/setup_dashboards.py
```

## Dashboard Guide

### 1. Team Overview Dashboard

**Purpose:** High-level view of team execution health and throughput.

**Key Panels:**

- **Team Execution Rate**: Teams executed per minute (success vs failure)
- **Success Rate**: Overall success percentage (color-coded: green >95%, yellow >80%, red <80%)
- **Average Duration**: Mean team execution time
- **Formation Distribution**: Pie chart of formation types
- **Execution Rate by Formation**: Time series of execution rate per formation
- **Duration Percentiles**: p50, p95, p99 execution times
- **Tool Calls per Team**: Average tool usage
- **Team Size Distribution**: Histogram of member counts
- **Active Teams**: Current active team count
- **Tool Call Rate**: Total tool call rate

**Variables:**
- `vertical`: Filter by vertical (coding, devops, rag, etc.)
- `rate_interval`: Time interval for rate calculations (5m, 10m, 1h)

**Common Queries:**

```promql
# Overall success rate
sum(victor_teams_executed_total) - sum(victor_teams_failed_total)
/ sum(victor_teams_executed_total) * 100

# Average team duration
rate(victor_teams_duration_seconds_sum[5m])
/ rate(victor_teams_duration_seconds_count[5m])

# Formation distribution
sum by (formation) (victor_teams_executed_total)
```

### 2. Team Performance Dashboard

**Purpose:** Compare performance across different formation types.

**Key Panels:**

- **Success Rate by Formation**: Line graph comparing formation success rates
- **Average Duration by Formation**: Execution time comparison
- **95th Percentile Duration**: High-water mark performance
- **Avg Tool Calls per Team**: Efficiency comparison
- **Avg Member Count**: Team size comparison
- **Formation Performance Summary**: Table with key metrics

**Use Cases:**

- Identify which formations perform best for your workload
- Compare efficiency (tool calls per team)
- Spot performance regressions in specific formations
- Optimize formation selection in workflows

**Common Queries:**

```promql
# Success rate by formation
(
  sum by (formation) (rate(victor_teams_formation_{formation}_total[5m])) -
  sum by (formation) (rate(victor_teams_failed_total[5m]))
) / sum by (formation) (rate(victor_teams_formation_{formation}_total[5m])) * 100

# Duration by formation
rate(victor_teams_duration_seconds_sum[5m]) by (formation)
/ rate(victor_teams_duration_seconds_count[5m]) by (formation)
```

### 3. Team Recursion Dashboard

**Purpose:** Monitor recursion depth to prevent infinite loops and stack overflow.

**Key Panels:**

- **Current Recursion Depth**: Real-time depth per team
- **Max Recursion Depth**: Maximum depth observed
- **Recursion Depth Distribution**: Histogram of depth values
- **Active Teams by Recursion Depth**: Table of teams with depth > 0
- **Average Recursion Depth by Formation**: Mean depth per formation type
- **Recursion Depth Exceeded Rate**: Rate of limit violations
- **Recursion Depth Heatmap**: Time vs depth heatmap
- **Recursion Limit Utilization**: Percentage of limit used
- **Deepest Team Executions**: Top teams by depth

**Alerts:**

- **Critical**: Depth ≥ 90% of limit (9 of 10)
- **Warning**: Depth ≥ 70% of limit (7 of 10)

**Common Queries:**

```promql
# Current recursion depth
victor_teams_recursion_depth

# Recursion limit utilization
(victor_teams_recursion_depth / 10) * 100

# Depth exceeded events
rate(victor_teams_recursion_depth_exceeded_total[5m])
```

### 4. Team Members Dashboard

**Purpose:** Monitor individual member performance and role-based metrics.

**Key Panels:**

- **Member Success Rate by Role**: Success percentage per role
- **Average Member Duration by Role**: Execution time per role
- **Avg Tool Calls per Member**: Tool usage efficiency
- **Top 10 Most Active Members**: Most executed members
- **Member Failure Rate by Role**: Error rate per role
- **Error Distribution by Type**: Breakdown of error types
- **Member Execution Summary**: Detailed table of member activity
- **Member Duration Percentiles**: p50, p95, p99 per role
- **Recent Member Executions**: Latest executions with status

**Use Cases:**

- Identify problematic roles (high failure rate)
- Spot performance bottlenecks in specific roles
- Optimize tool usage per role
- Track member-level health

**Common Queries:**

```promql
# Member success rate by role
sum by (role) (rate(victor_team_members_completed_total{success="true"}[5m]))
/ sum by (role) (rate(victor_team_members_completed_total[5m])) * 100

# Member duration by role
rate(victor_team_members_duration_seconds_sum[5m]) by (role)
/ rate(victor_team_members_duration_seconds_count[5m]) by (role)

# Tool calls per member
sum by (role) (rate(victor_team_members_tool_calls_total[5m]))
/ sum by (role) (rate(victor_team_members_completed_total[5m]))
```

## Alerting

### Alert Severities

| Severity | Description | Action Required |
|----------|-------------|-----------------|
| **Critical** | Service impact | Immediate action required |
| **Warning** | Performance degradation | Investigate within 15 minutes |
| **Info** | Informational | Review during business hours |

### Critical Alerts

#### VictorTeamRecursionDepthCritical
**Condition:** Recursion depth ≥ 90% of limit (9 of 10)

**Impact:** Risk of infinite loop or stack overflow

**Action:**
1. Review workflow for recursive team calls
2. Check for circular dependencies
3. Increase recursion limit if needed (with caution)
4. Kill affected team execution

#### VictorTeamExecutionTimeout
**Condition:** 90th percentile duration > 5 minutes

**Impact:** Teams running too long

**Action:**
1. Identify slow formations (use Performance dashboard)
2. Review tool usage for inefficiencies
3. Consider timeout configuration
4. Optimize team composition

#### VictorTeamCatastrophicFailureRate
**Condition:** Failure rate > 50%

**Impact:** Major service disruption

**Action:**
1. Check provider status (API health)
2. Verify tool availability
3. Review recent configuration changes
4. Check error logs for patterns

### Warning Alerts

#### VictorTeamHighFailureRate
**Condition:** Failure rate > 10%

**Action:** Review failed executions and identify patterns

#### VictorTeamPerformanceRegression
**Condition:** 20% slower than baseline (1h ago)

**Action:** Investigate recent changes affecting performance

#### VictorTeamRecursionDepthWarning
**Condition:** Recursion depth ≥ 70% of limit

**Action:** Review workflow for optimization opportunities

#### VictorTeamMemberHighFailureRate
**Condition:** Role-specific failure rate > 20%

**Action:** Investigate role-specific issues and tool availability

### Info Alerts

#### VictorTeamFormationSlowExecution
**Condition:** Formation average > 60 seconds

**Action:** Consider formation optimization

#### VictorTeamLowToolUsage
**Condition:** < 1 tool call per team

**Action:** Review if teams are effectively utilizing tools

#### VictorTeamLargeSize
**Condition:** Average > 10 members per team

**Action:** Consider team size optimization

#### VictorTeamNoExecutions
**Condition:** No executions in 15 minutes

**Action:** Verify workflow configuration and team node availability

## Alert Tuning

### Adjusting Thresholds

Edit `observability/alerts/team_alerts.yml`:

```yaml
# Example: Increase failure rate threshold to 15%
- alert: VictorTeamHighFailureRate
  expr: |
    (
      sum(rate(victor_teams_failed_total[5m])) /
      sum(rate(victor_teams_executed_total[5m]))
    ) > 0.15  # Changed from 0.10 to 0.15
```

### Adjusting Durations

```yaml
# Example: Require 10 minutes of violations before alerting
- alert: VictorTeamRecursionDepthCritical
  expr: (victor_teams_recursion_depth / 10) >= 0.9
  for: 10m  # Changed from 1m to 10m
```

### Adding Notification Channels

Configure in Prometheus (`alertmanager.yml`):

```yaml
receivers:
  - name: 'victor-alerts'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
        channel: '#victor-alerts'
        title: 'Victor Alert: {{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

    email_configs:
      - to: 'victor-ops@example.com'
        subject: 'Victor Alert: {{ .GroupLabels.alertname }}'
        body: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

route:
  receiver: 'victor-alerts'
  group_by: ['alertname', 'severity']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
```

### Testing Alerts

```bash
# Validate alert syntax
promtool check rules observability/alerts/team_alerts.yml

# Check active alerts
curl http://localhost:9090/api/v1/alerts | jq

# Test alert firing
# Temporarily adjust threshold in team_alerts.yml
# Reload Prometheus: killall -HUP prometheus
# Check alerts UI: http://localhost:9090/alerts
```

## Query Examples

### Finding Slow Teams

```promql
# Top 10 slowest teams
topk(10,
  rate(victor_teams_duration_seconds_sum[5m]) by (team_id)
  / rate(victor_teams_duration_seconds_count[5m]) by (team_id)
)

# Slowest by formation
avg by (formation) (
  rate(victor_teams_duration_seconds_sum[5m]) by (team_id, formation)
  / rate(victor_teams_duration_seconds_count[5m]) by (team_id, formation)
)
```

### Failure Analysis

```promql
# Failure rate by formation
sum by (formation) (rate(victor_teams_failed_total[5m]))
/ sum by (formation) (rate(victor_teams_executed_total[5m])) * 100

# Recent failures (last 5 minutes)
rate(victor_teams_failed_total[5m]) > 0

# Failure spike detection
rate(victor_teams_failed_total[5m])
> rate(victor_teams_failed_total[1h] offset 1h) * 2
```

### Tool Usage Analysis

```promql
# Most used tools by teams
topk(20,
  sum by (tool) (rate(victor_teams_tool_calls_total[5m]))
)

# Tool usage per formation
sum by (formation, tool) (
  rate(victor_teams_tool_calls_total[5m])
)

# Tool efficiency (calls per successful team)
sum by (formation) (rate(victor_teams_tool_calls_total[5m]))
/ sum by (formation) (rate(victor_teams_executed_total[5m]))
```

### Recursion Analysis

```promql
# Teams approaching recursion limit
victor_teams_recursion_depth > 7

# Recursion depth trend over time
avg_over_time(victor_teams_recursion_depth[1h])

# Deep recursion by formation
avg by (formation) (victor_teams_recursion_depth)
```

### Member Performance

```promql
# Most successful members
topk(10,
  sum by (member_id) (victor_team_members_completed_total{success="true"})
)

# Member duration outliers
histogram_quantile(0.95,
  sum by (le, member_id) (
    rate(victor_team_members_duration_seconds_bucket[5m])
  )
)

# Role-specific failure patterns
sum by (role, error_type) (
  rate(victor_team_members_errors_total[5m])
)
```

