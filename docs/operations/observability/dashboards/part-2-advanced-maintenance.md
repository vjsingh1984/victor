# Observability Dashboards Guide - Part 2

**Part 2 of 2:** Integration with Existing Monitoring, Troubleshooting, Best Practices, Advanced Usage, Maintenance, Support, and Changelog

---

## Navigation

- [Part 1: Dashboard Setup & Usage](part-1-dashboard-setup-usage.md)
- **[Part 2: Advanced & Maintenance](#)** (Current)
- [**Complete Guide](../dashboards.md)**

---

## Table of Contents

1. [Overview](#overview) *(in Part 1)*
2. [Installation](#installation) *(in Part 1)*
3. [Dashboard Guide](#dashboard-guide) *(in Part 1)*
4. [Alerting](#alerting) *(in Part 1)*
5. [Alert Tuning](#alert-tuning) *(in Part 1)*
6. [Query Examples](#query-examples) *(in Part 1)*
7. [Integration with Existing Monitoring](#integration-with-existing-monitoring)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)
10. [Advanced Usage](#advanced-usage)
11. [Maintenance](#maintenance)
12. [Support](#support)
13. [Changelog](#changelog)

---

## Integration with Existing Monitoring

### Grafana

**Add to existing dashboard:**

1. Edit existing dashboard
2. Add new panel
3. Select Prometheus datasource
4. Use example queries above

**Link dashboards:**

Dashboard JSON includes links to other Victor dashboards. Update URLs to match your Grafana instance:

```json
{
  "links": [
    {
      "title": "Team Overview",
      "type": "dashboards",
      "tags": ["victor-team-overview"],
      "url": "https://grafana.yourcompany.com/d/victor-team-overview"
    }
  ]
}
```

### Prometheus

**Add to existing scrape config:**

```yaml
scrape_configs:
  - job_name: 'victor'
    static_configs:
      - targets: ['localhost:8000']
    scrape_interval: 15s
    metrics_path: '/metrics'
```

**Combine with existing alerting:**

Add team alerting rules to your existing `rule_files` list in `prometheus.yml`.

### Alertmanager

**Add Victor-specific routing:**

```yaml
route:
  receiver: 'default'
  routes:
    - match:
        component: "teams"
      receiver: 'victor-team-alerts'
      group_by: ['alertname', 'formation']
      group_wait: 30s
      repeat_interval: 2h

receivers:
  - name: 'victor-team-alerts'
    slack_configs:
      - api_url: '${SLACK_WEBHOOK_URL}'
        channel: '#victor-teams'
```

### Custom Metrics

**Add custom team metrics:**

```python
from victor.observability.metrics import MetricsRegistry

registry = MetricsRegistry.get_instance()

# Custom metric
custom_duration = registry.histogram(
    "victor_teams_custom_duration_seconds",
    "Custom team duration",
    labels=["vertical", "formation"]
)
```

Then query in Grafana:
```promql
rate(victor_teams_custom_duration_seconds_sum[5m])
/ rate(victor_teams_custom_duration_seconds_count[5m])
```

## Troubleshooting

### Dashboards Not Loading

**Issue:** Dashboards show "No data"

**Solutions:**
1. Verify Prometheus datasource is configured
2. Check Prometheus is scraping Victor: `curl http://localhost:9090/api/v1/targets`
3. Verify metrics exist: `curl http://localhost:9090/api/v1/label/__name__/values | grep victor_teams`
4. Check time range in dashboard (should be "now-1h" to "now")

### Metrics Missing

**Issue:** Expected metrics not appearing

**Solutions:**
1. Verify metrics collection is enabled:
   ```python
   from victor.workflows.team_metrics import get_team_metrics_collector
   collector = get_team_metrics_collector()
   print(collector.is_enabled())  # Should be True
   ```

2. Check metrics endpoint: `curl http://localhost:8000/metrics`

3. Verify team execution: Check logs for team execution activity

### Alerts Not Firing

**Issue:** Alerts not triggering when expected

**Solutions:**
1. Validate alert syntax: `promtool check rules observability/alerts/team_alerts.yml`
2. Check Prometheus loaded rules: `curl http://localhost:9090/api/v1/rules`
3. Verify Alertmanager is configured: `curl http://localhost:9093/api/v1/status`
4. Test alert manually: Adjust threshold temporarily

### High Recursion Depth False Positives

**Issue:** Alerts firing for normal deep workflows

**Solutions:**
1. Increase recursion limit in `team_alerts.yml`
2. Increase `for` duration to avoid transient spikes
3. Exclude specific formations from alert

### Performance Impact

**Issue:** Metrics collection affecting performance

**Solutions:**
1. Use metric priority filtering:
   ```python
   from victor.workflows.team_metrics import TeamMetricsCollector, MetricPriority
   collector = TeamMetricsCollector(
       enabled=True,
       priority_threshold=MetricPriority.HIGH  # Only collect HIGH and CRITICAL
   )
   ```

2. Reduce scrape interval in Prometheus (30s → 60s)

3. Disable detailed member metrics for high-throughput scenarios

## Best Practices

### 1. Dashboard Time Ranges

- **Real-time monitoring**: 5-15 minutes refresh
- **Performance analysis**: 1-6 hours
- **Trend analysis**: 24-48 hours
- **Capacity planning**: 7-30 days

### 2. Alert Thresholds

- **Critical alerts**: Set thresholds based on SLA requirements
- **Warning alerts**: Set at 70-80% of critical threshold
- **Info alerts**: Use for informational/optimization insights

### 3. Alert Routing

- **Critical**: PagerDuty/On-call SMS
- **Warning**: Slack channel
- **Info**: Email digest

### 4. Dashboard Organization

- Use folders to group dashboards by environment (dev, staging, prod)
- Tag dashboards for easy discovery
- Use consistent naming convention

### 5. Query Optimization

- Use `rate()` for counters
- Use appropriate time windows (5m for real-time, 1h for trends)
- Label filtering for better performance

## Advanced Usage

### Custom Dashboards

Create custom dashboards using example queries. Dashboard JSON files can be used as templates.

### Grafana API

Provision dashboards programmatically:

```python
import requests

GRAFANA_URL = "http://localhost:3000"
API_KEY = "your-api-key"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# List dashboards
response = requests.get(
    f"{GRAFANA_URL}/api/search?tag=victor",
    headers=headers
)

# Delete dashboard
dashboard_uid = "victor-team-overview"
requests.delete(
    f"{GRAFANA_URL}/api/dashboards/uid/{dashboard_uid}",
    headers=headers
)
```

### Prometheus Recording Rules

Add recording rules for complex queries:

```yaml
# /etc/prometheus/rules/recording_rules.yml
groups:
  - name: victor_teams_recording
    interval: 30s
    rules:
      - record: victor_teams_health_score
        expr: |
          (
            (1 - (sum(rate(victor_teams_failed_total[5m])) / sum(rate(victor_teams_executed_total[5m])))) * 0.5
            +
            (1 - (histogram_quantile(0.90, sum by (le) (rate(victor_teams_duration_seconds_bucket[5m]))) / 300)) * 0.3
            +
            (1 - (max(victor_teams_recursion_depth) / 10)) * 0.2
          ) * 100
```

## Maintenance

### Regular Tasks

**Weekly:**
- Review alert firing history
- Check dashboard performance
- Validate alert thresholds

**Monthly:**
- Review and update dashboard panels
- Clean up unused alerts
- Optimize slow queries

**Quarterly:**
- Review observability strategy
- Update documentation
- Capacity planning for metrics storage

### Backup and Restore

**Backup dashboards:**

```bash
# Export all dashboards
curl -H "Authorization: Bearer $API_KEY" \
  http://localhost:3000/api/search?query=victor | \
  jq -r '.[] | .uid' | \
  while read uid; do
    curl -H "Authorization: Bearer $API_KEY" \
      "http://localhost:3000/api/dashboards/uid/$uid" | \
      jq '.dashboard' > "backup_${uid}.json"
  done
```

**Restore dashboards:**

```bash
for file in backup_*.json; do
  curl -X POST \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"dashboard\": $(cat $file), \"overwrite\": true}" \
    http://localhost:3000/api/dashboards/db
done
```

## Support

For issues and questions:

- **Documentation**: See [team_metrics.md](team_metrics.md)
- **GitHub Issues**: https://github.com/victor-ai/victor/issues
- **Slack**: #victor-observability

## Changelog

### Version 0.5.0 (2025-01-15)

- Initial release of team observability dashboards
- 4 dashboards: Overview, Performance, Recursion, Members
- 15+ alerting rules across 3 severity levels
- Comprehensive documentation and examples

---

**Last Updated:** February 01, 2026
**Reading Time:** 8 minutes
