# Victor Observability Dashboards and Alerting

This directory contains Grafana dashboards and Prometheus alerting rules for comprehensive monitoring of Victor team workflows.

## Contents

### Dashboards (`dashboards/`)

- **team_overview.json** - High-level team execution metrics and KPIs
- **team_performance.json** - Formation performance comparison
- **team_recursion.json** - Recursion depth monitoring
- **team_members.json** - Member-level performance metrics

### Alerting Rules (`alerts/`)

- **team_alerts.yml** - Prometheus alerting rules with severity levels
  - Critical alerts (immediate action required)
  - Warning alerts (investigate within 15 minutes)
  - Info alerts (review during business hours)

## Quick Start

### 1. Install Prerequisites

```bash
# Grafana
sudo apt-get install -y grafana

# Prometheus
sudo apt-get install -y prometheus

# Start services
sudo systemctl start grafana-server
sudo systemctl start prometheus
```

### 2. Configure Prometheus

Add to `/etc/prometheus/prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'victor'
    static_configs:
      - targets: ['localhost:8000']  # Victor metrics endpoint
    scrape_interval: 15s
```

Reload Prometheus:
```bash
sudo killall -HUP prometheus
```

### 3. Setup Dashboards

```bash
# Option 1: Use setup script (recommended)
export GRAFANA_URL="http://localhost:3000"
export GRAFANA_API_KEY="your-api-key"
python scripts/observability/setup_dashboards.py

# Option 2: Use example script with health checks
python examples/observability/dashboard_setup.py

# Option 3: Import manually via Grafana UI
# 1. Open http://localhost:3000
# 2. Go to Dashboards → Import
# 3. Upload dashboard JSON files from observability/dashboards/
```

### 4. Configure Alerting

```bash
# Copy alerting rules
sudo cp observability/alerts/team_alerts.yml /etc/prometheus/rules/

# Add to prometheus.yml
echo 'rule_files:
  - "/etc/prometheus/rules/team_alerts.yml"' | sudo tee -a /etc/prometheus/prometheus.yml

# Reload Prometheus
sudo killall -HUP prometheus
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GRAFANA_URL` | Grafana server URL | `http://localhost:3000` |
| `GRAFANA_API_KEY` | Grafana API key for authentication | - |
| `PROMETHEUS_URL` | Prometheus server URL | `http://localhost:9090` |
| `PROMETHEUS_CONFIG_PATH` | Path to prometheus.yml | `/etc/prometheus/prometheus.yml` |
| `VICTOR_METRICS_URL` | Victor metrics endpoint | `http://localhost:8000` |

## Dashboards Overview

### Team Overview (`victor-team-overview`)
**Purpose:** High-level view of team execution health

**Key Metrics:**
- Team execution rate (success vs failure)
- Overall success rate (color-coded)
- Average duration
- Formation distribution
- Duration percentiles (p50, p95, p99)
- Tool calls per team
- Active teams

**Use Cases:**
- Real-time monitoring of team health
- Quick assessment of system status
- Identifying trends and anomalies

### Team Performance (`victor-team-performance`)
**Purpose:** Compare performance across formation types

**Key Metrics:**
- Success rate by formation
- Average duration by formation
- 95th percentile duration
- Tool calls per team by formation
- Member count by formation

**Use Cases:**
- Identify which formations perform best
- Optimize formation selection in workflows
- Spot performance regressions

### Team Recursion (`victor-team-recursion`)
**Purpose:** Monitor recursion depth to prevent infinite loops

**Key Metrics:**
- Current recursion depth per team
- Maximum depth observed
- Depth distribution histogram
- Recursion limit utilization
- Depth exceeded events

**Use Cases:**
- Detect infinite loops early
- Identify workflows with deep nesting
- Optimize recursion usage

### Team Members (`victor-team-members`)
**Purpose:** Monitor individual member performance

**Key Metrics:**
- Member success rate by role
- Member duration by role
- Tool calls per member
- Top active members
- Error distribution by type

**Use Cases:**
- Identify problematic roles
- Optimize tool usage per role
- Track member-level health

## Alerting

### Critical Alerts (Immediate Action Required)

- **VictorTeamRecursionDepthCritical** - Recursion depth ≥ 90% of limit
- **VictorTeamExecutionTimeout** - 90th percentile > 5 minutes
- **VictorTeamCatastrophicFailureRate** - Failure rate > 50%

### Warning Alerts (Investigate Within 15 Minutes)

- **VictorTeamHighFailureRate** - Failure rate > 10%
- **VictorTeamPerformanceRegression** - 20% slower than baseline
- **VictorTeamRecursionDepthWarning** - Recursion depth ≥ 70%
- **VictorTeamMemberHighFailureRate** - Role failure rate > 20%

### Info Alerts (Review During Business Hours)

- **VictorTeamFormationSlowExecution** - Formation > 60 seconds
- **VictorTeamLowToolUsage** - < 1 tool call per team
- **VictorTeamLargeSize** - Average > 10 members
- **VictorTeamNoExecutions** - No executions in 15 minutes

## Documentation

For detailed documentation, see:
- **[dashboards.md](../../docs/observability/dashboards.md)** - Complete dashboard guide
- **[team_metrics.md](../../docs/observability/team_metrics.md)** - Metrics reference

## Troubleshooting

### Dashboards Show No Data

1. Verify Prometheus datasource is configured in Grafana
2. Check Prometheus is scraping Victor: `curl http://localhost:9090/api/v1/targets`
3. Verify metrics exist: `curl http://localhost:8000/metrics | grep victor_teams`

### Alerts Not Firing

1. Validate alert syntax: `promtool check rules observability/alerts/team_alerts.yml`
2. Check Prometheus loaded rules: `curl http://localhost:9090/api/v1/rules`
3. Verify Alertmanager is configured: `curl http://localhost:9093/api/v1/status`

### Health Checks Fail

1. Ensure Grafana is running: `sudo systemctl status grafana-server`
2. Ensure Prometheus is running: `sudo systemctl status prometheus`
3. Ensure Victor is running with metrics enabled
4. Check firewall rules and port accessibility

## Maintenance

### Weekly
- Review alert firing history
- Check dashboard performance
- Validate alert thresholds

### Monthly
- Review and update dashboard panels
- Clean up unused alerts
- Optimize slow queries

### Quarterly
- Review observability strategy
- Update documentation
- Capacity planning for metrics storage

## Support

For issues and questions:
- **Documentation**: [docs/observability/dashboards.md](../../docs/observability/dashboards.md)
- **GitHub Issues**: https://github.com/victor-ai/victor/issues
- **Example Script**: [examples/observability/dashboard_setup.py](../../examples/observability/dashboard_setup.py)
