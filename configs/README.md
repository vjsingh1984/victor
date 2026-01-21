# Victor AI Monitoring Configuration

This directory contains configuration files for Victor AI's production monitoring stack.

## Directory Structure

```
configs/
├── prometheus/
│   ├── prometheus.yml          # Prometheus main configuration
│   ├── alerting_rules.yml      # Alert rules
│   └── recording_rules.yml     # Recording rules
├── grafana/
│   ├── dashboard_overview.json     # Overview dashboard
│   ├── dashboard_performance.json  # Performance dashboard
│   ├── dashboard_verticals.json    # Verticals dashboard
│   └── dashboard_errors.json       # Errors dashboard
└── alertmanager/
    └── alertmanager.yml        # AlertManager configuration
```

## Usage

### Quick Start

```bash
# Start monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d

# Access services
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
# AlertManager: http://localhost:9093
```

### Prometheus Configuration

**File**: `prometheus/prometheus.yml`

Key settings:
- Scrape interval: 15 seconds
- Evaluation interval: 15 seconds
- Data retention: 15 days
- Targets: Victor AI (localhost:9091), Prometheus, Node Exporter, cAdvisor

### Alert Rules

**File**: `prometheus/alerting_rules.yml`

Alert categories:
- **Critical**: Error rate > 5%, response time p95 > 30s, memory > 90%
- **Warning**: Response time p95 > 10s, memory > 75%, tool failures > 5%

### Recording Rules

**File**: `prometheus/recording_rules.yml`

Pre-computed queries:
- Request rate, error rate, duration percentiles
- Tool execution rate, success rate
- Provider request rate, success rate, latency
- Vertical usage, success rate
- Cache hit rates

### Grafana Dashboards

Four pre-configured dashboards:

1. **Overview** (`dashboard_overview.json`)
   - System health, request rate, error rate
   - Response time percentiles, vertical usage
   - Memory and CPU usage

2. **Performance** (`dashboard_performance.json`)
   - Response times (p50, p95, p99)
   - Tool execution time, provider latency
   - Memory, CPU, request rate, cache hit rates

3. **Verticals** (`dashboard_verticals.json`)
   - Vertical usage distribution
   - Vertical-specific metrics:
     - Coding: files analyzed, LOC reviewed, issues found
     - RAG: documents ingested, search accuracy
     - DevOps: deployments, containers managed
     - DataAnalysis: queries, visualizations
     - Research: searches, citations

4. **Errors** (`dashboard_errors.json`)
   - Error rate by endpoint
   - Tool failures, provider failures
   - Security events, rate limits

## Importing Grafana Dashboards

### Via UI

1. Navigate to Grafana: http://localhost:3000
2. Dashboards → Import
3. Upload JSON file or paste JSON content
4. Select Prometheus datasource
5. Click Import

### Via API

```bash
# Import a dashboard
curl -X POST \
  -H "Content-Type: application/json" \
  -d @configs/grafana/dashboard_overview.json \
  http://admin:admin@localhost:3000/api/dashboards/import

# Import all dashboards
for dashboard in configs/grafana/*.json; do
  curl -X POST \
    -H "Content-Type: application/json" \
    -d @"$dashboard" \
    http://admin:admin@localhost:3000/api/dashboards/import
done
```

## Customization

### Modify Scrape Interval

Edit `prometheus/prometheus.yml`:

```yaml
global:
  scrape_interval: 30s  # Change from 15s
```

### Add New Alert Rules

Edit `prometheus/alerting_rules.yml`:

```yaml
- alert: MyCustomAlert
  expr: my_metric > threshold
  for: 5m
  annotations:
    summary: "Custom alert description"
```

### Add Recording Rules

Edit `prometheus/recording_rules.yml`:

```yaml
- record: job:my_metric:rate:5m
  expr: rate(my_metric[5m])
```

### Customize Dashboards

1. Export dashboard from Grafana (JSON)
2. Edit JSON file
3. Re-import or place in `configs/grafana/`

## Validation

### Validate Prometheus Configuration

```bash
# Using docker
docker run --rm -v $(pwd)/configs/prometheus:/etc/prometheus \
  prom/prometheus:latest \
  promtool check config /etc/prometheus/prometheus.yml

# Using promtool (native)
promtool check config configs/prometheus/prometheus.yml
```

### Validate Alert Rules

```bash
# Using docker
docker run --rm -v $(pwd)/configs/prometheus:/etc/prometheus \
  prom/prometheus:latest \
  promtool check rules /etc/prometheus/alerting_rules.yml

# Using promtool (native)
promtool check rules configs/prometheus/alerting_rules.yml
```

### Test Prometheus Queries

```bash
# Query metric
curl 'http://localhost:9090/api/v1/query?query=victor_request_duration_seconds_count'

# Query range
curl 'http://localhost:9090/api/v1/query_range?query=victor_request_duration_seconds_count&start=2024-01-01T00:00:00Z&end=2024-01-01T01:00:00Z&step=15s'
```

## Troubleshooting

### Prometheus Not Scraping Victor AI

1. Check Victor AI metrics endpoint:
   ```bash
   curl http://localhost:9091/metrics
   ```

2. Verify Prometheus target:
   ```bash
   curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | select(.labels.job=="victor-ai")'
   ```

3. Check Prometheus logs:
   ```bash
   docker logs prometheus
   ```

### Grafana Dashboards Show No Data

1. Verify Prometheus datasource:
   - Configuration → Data Sources → Prometheus
   - Click "Test" button

2. Check dashboard queries:
   - Edit panel → Query inspector
   - Verify query syntax

3. Check Grafana logs:
   ```bash
   docker logs grafana
   ```

### Alerts Not Firing

1. Verify AlertManager is running:
   ```bash
   curl http://localhost:9093/-/healthy
   ```

2. Check alert rules:
   ```bash
   curl http://localhost:9090/api/v1/rules | jq '.data.groups[] | .rules[] | select(.type=="alerting")'
   ```

3. Check AlertManager logs:
   ```bash
   docker logs alertmanager
   ```

## Security Considerations

### Production Deployment

1. **Change Default Passwords**
   ```yaml
   # docker-compose.monitoring.yml
   environment:
     - GF_SECURITY_ADMIN_PASSWORD=your_secure_password
   ```

2. **Enable Authentication**
   ```yaml
   # prometheus.yml
   basic_auth:
     username: prometheus
     password: your_password
   ```

3. **Use TLS**
   - Configure HTTPS for all services
   - Use valid SSL certificates

4. **Restrict Network Access**
   - Use firewall rules
   - Bind to localhost only if possible
   - Use VPN for remote access

5. **Secret Management**
   - Use environment variables for secrets
   - Consider using Docker secrets or Kubernetes secrets

## Backup and Recovery

### Backup Grafana Dashboards

```bash
# Export all dashboards
curl http://admin:admin@localhost:3000/api/search?query=\
  | jq -r '.[] | .uri' \
  | while read dashboard; do
      curl "http://admin:admin@localhost:3000/api/dashboards/db/$dashboard" \
        | jq '. > "$(echo $dashboard | sed 's/\//-/g').json"
    done
```

### Backup Prometheus Configuration

```bash
# Backup configs
tar -czf prometheus-configs-$(date +%Y%m%d).tar.gz configs/prometheus/

# Backup Prometheus data
docker exec prometheus tar -czf /tmp/prometheus-data.tar.gz /prometheus
docker cp prometheus:/tmp/prometheus-data.tar.gz ./backups/
```

### Restore Prometheus Data

```bash
# Stop Prometheus
docker-compose -f docker-compose.monitoring.yml stop prometheus

# Restore data
docker cp ./backups/prometheus-data.tar.gz prometheus:/tmp/
docker exec prometheus tar -xzf /tmp/prometheus-data.tar.gz -C /

# Start Prometheus
docker-compose -f docker-compose.monitoring.yml start prometheus
```

## Performance Tuning

### Prometheus Memory Usage

Add to `prometheus.yml`:

```yaml
storage:
  tsdb:
    retention.time: 7d  # Reduce retention
```

### Reduce Scrape Overhead

```yaml
scrape_configs:
  - job_name: 'victor-ai'
    scrape_interval: 30s  # Increase from 15s
    sample_limit: 10000   # Limit samples per scrape
```

### Optimize Recording Rules

- Use efficient PromQL
- Avoid high-cardinality labels
- Pre-aggregate where possible

## Related Documentation

- [Production Metrics Guide](../docs/observability/PRODUCTION_METRICS.md) - Complete metrics reference
- [Monitoring Setup Guide](../docs/observability/MONITORING_SETUP.md) - Detailed setup instructions
- [Quick Reference](../docs/observability/QUICK_REFERENCE.md) - Quick metrics overview
