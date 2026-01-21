# Production Monitoring Setup Guide

This guide provides step-by-step instructions for setting up comprehensive monitoring for Victor AI in production environments.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
- [Configuration](#configuration)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

## Overview

Victor AI's production monitoring stack consists of:

1. **Prometheus** - Metrics collection and storage
2. **Grafana** - Visualization and dashboards
3. **AlertManager** - Alert routing and notification
4. **Victor AI Metrics Collector** - Built-in metrics collection

## Prerequisites

### System Requirements

- **Operating System**: Linux, macOS, or Windows with WSL2
- **Python**: 3.10 or higher
- **Memory**: 4GB minimum, 8GB recommended
- **Disk**: 20GB free space for metrics storage
- **Network**: Port access for 9090 (Prometheus), 9091 (Victor metrics), 3000 (Grafana), 9093 (AlertManager)

### Software Requirements

```bash
# Docker (recommended for production)
docker --version  # 20.10 or higher

# Or install natively
# Python dependencies
pip install prometheus-client psutil requests

# Optional: For metrics report generation
pip install pandas
```

## Quick Start

### 1. Start Monitoring Stack (Docker)

```bash
# Navigate to project root
cd /path/to/victor

# Start all services
docker-compose -f docker-compose.monitoring.yml up -d
```

### 2. Enable Metrics in Victor AI

```bash
# Set environment variable
export VICTOR_PROMETHEUS_ENABLED=true
export VICTOR_PROMETHEUS_PORT=9091

# Start Victor AI
victor chat --no-tui
```

### 3. Access Dashboards

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Victor Metrics**: http://localhost:9091/metrics

## Detailed Setup

### Step 1: Install Prometheus

#### Option A: Docker (Recommended)

```bash
# Pull Prometheus image
docker pull prom/prometheus

# Create configuration directory
mkdir -p ~/victor-monitoring/prometheus
cp configs/prometheus/*.yml ~/victor-monitoring/prometheus/

# Start Prometheus
docker run -d \
  --name prometheus \
  --restart unless-stopped \
  -p 9090:9090 \
  -v ~/victor-monitoring/prometheus:/etc/prometheus \
  -v prometheus-data:/prometheus \
  prom/prometheus \
  --config.file=/etc/prometheus/prometheus.yml \
  --storage.tsdb.path=/prometheus \
  --web.enable-lifecycle
```

#### Option B: Native Installation

```bash
# Download Prometheus
wget https://github.com/prometheus/prometheus/releases/download/v2.45.0/prometheus-2.45.0.linux-amd64.tar.gz
tar xvfz prometheus-2.45.0.linux-amd64.tar.gz
cd prometheus-2.45.0.linux-amd64

# Copy configuration
cp /path/to/victor/configs/prometheus/*.yml .

# Start Prometheus
./prometheus --config.file=prometheus.yml
```

### Step 2: Install Grafana

#### Option A: Docker (Recommended)

```bash
# Pull Grafana image
docker pull grafana/grafana

# Create configuration directory
mkdir -p ~/victor-monitoring/grafana/provisioning/dashboards
mkdir -p ~/victor-monitoring/grafana/provisioning/datasources

# Copy dashboards
cp configs/grafana/*.json ~/victor-monitoring/grafana/provisioning/dashboards/

# Create datasource configuration
cat > ~/victor-monitoring/grafana/provisioning/datasources/prometheus.yml <<EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

# Start Grafana
docker run -d \
  --name grafana \
  --restart unless-stopped \
  -p 3000:3000 \
  -v ~/victor-monitoring/grafana:/var/lib/grafana \
  -e "GF_SECURITY_ADMIN_PASSWORD=admin" \
  -e "GF_INSTALL_PLUGINS=" \
  grafana/grafana
```

#### Option B: Native Installation

```bash
# Add Grafana repository
wget -q -O - https://packages.grafana.com/gpg.key | sudo apt-key add -
sudo add-apt-repository "deb https://packages.grafana.com/oss/deb stable main"

# Install Grafana
sudo apt update
sudo apt install grafana

# Start Grafana
sudo systemctl start grafana-server
sudo systemctl enable grafana-server
```

### Step 3: Configure AlertManager

```bash
# Pull AlertManager image
docker pull prom/alertmanager

# Create configuration
mkdir -p ~/victor-monitoring/alertmanager

cat > ~/victor-monitoring/alertmanager/alertmanager.yml <<EOF
global:
  resolve_timeout: 5m

route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'default'

  routes:
    - match:
        severity: critical
      receiver: 'critical'
      continue: true

    - match:
        severity: warning
      receiver: 'warning'

receivers:
  - name: 'default'
    webhook_configs:
      - url: 'http://localhost:5001/webhook'

  - name: 'critical'
    webhook_configs:
      - url: 'http://localhost:5001/critical'
    email_configs:
      - to: 'oncall@example.com'
        from: 'alertmanager@example.com'
        smarthost: 'smtp.example.com:587'

  - name: 'warning'
    webhook_configs:
      - url: 'http://localhost:5001/warning'
EOF

# Start AlertManager
docker run -d \
  --name alertmanager \
  --restart unless-stopped \
  -p 9093:9093 \
  -v ~/victor-monitoring/alertmanager:/etc/alertmanager \
  prom/alertmanager \
  --config.file=/etc/alertmanager/alertmanager.yml
```

### Step 4: Enable Victor AI Metrics

#### In Python Code

```python
from victor.observability.metrics_collector import ProductionMetricsCollector

# Initialize metrics collector
collector = ProductionMetricsCollector()

# Start metrics server on port 9091
collector.start(port=9091)

# Record metrics
collector.record_tool_execution(
    tool="read_file",
    vertical="coding",
    status="success",
    duration=0.5,
    mode="build"
)

# Use decorators
@collector.track_request("/api/chat", "POST")
async def chat_handler(request):
    # Handler implementation
    pass

# Use context managers
with collector.track_provider_request("anthropic", "claude-sonnet-4-5"):
    response = await provider.chat(messages)
```

#### Environment Variables

```bash
# Enable Prometheus metrics
export VICTOR_PROMETHEUS_ENABLED=true
export VICTOR_PROMETHEUS_PORT=9091
export VICTOR_PROMETHEUS_HOST=0.0.0.0
```

### Step 5: Import Grafana Dashboards

#### Via UI

1. Navigate to Grafana: http://localhost:3000
2. Login (admin/admin)
3. Go to Dashboards → Import
4. Upload dashboard JSON files:
   - `configs/grafana/dashboard_overview.json`
   - `configs/grafana/dashboard_performance.json`
   - `configs/grafana/dashboard_verticals.json`
   - `configs/grafana/dashboard_errors.json`

#### Via CLI

```bash
# Import all dashboards
for dashboard in configs/grafana/*.json; do
  curl -X POST \
    -H "Content-Type: application/json" \
    -d @"$dashboard" \
    http://admin:admin@localhost:3000/api/dashboards/import
done
```

## Configuration

### Prometheus Configuration

Edit `configs/prometheus/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'victor-ai'
    static_configs:
      - targets: ['localhost:9091']
    labels:
      service: 'victor-ai'
      environment: 'production'
```

### Alert Rules

Alert rules are defined in `configs/prometheus/alerting_rules.yml`:

```yaml
groups:
  - name: victor_critical
    rules:
      - alert: HighErrorRate
        expr: rate(victor_request_errors_total[5m]) > 0.05
        for: 2m
        annotations:
          summary: "High error rate detected"
```

### Grafana Datasource

Configure Prometheus datasource in Grafana:

1. Go to Configuration → Data Sources
2. Add new data source → Prometheus
3. URL: `http://prometheus:9090`
4. Access: Server (default)
5. Click "Save & Test"

## Verification

### 1. Check Prometheus Targets

```bash
# Check if Victor AI is being scraped
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | select(.labels.job=="victor-ai")'
```

Expected output:
```json
{
  "scrapeUrl": "http://localhost:9091/metrics",
  "health": "up",
  "lastError": ""
}
```

### 2. Query Metrics

```bash
# Query a metric
curl 'http://localhost:9090/api/v1/query?query=victor_request_duration_seconds_count' | jq

# Expected: Metric data
```

### 3. Test Alerts

```bash
# Trigger a test alert
curl 'http://localhost:9090/api/v1/alerts' | jq '.data.alerts[] | select(.state=="firing")'
```

### 4. View Metrics in Grafana

1. Open Grafana: http://localhost:3000
2. Navigate to "Victor AI - Overview" dashboard
3. Verify metrics are displaying

### 5. Generate Metrics Report

```bash
# Generate a comprehensive metrics report
python scripts/generate_metrics_report.py \
  --format markdown \
  --output metrics_report.md \
  --prometheus http://localhost:9090

# Generate JSON report
python scripts/generate_metrics_report.py \
  --format json \
  --output metrics_report.json

# Generate CSV report
python scripts/generate_metrics_report.py \
  --format csv \
  --output metrics_report.csv
```

## Docker Compose Setup

Create `docker-compose.monitoring.yml`:

```yaml
version: '3.8'

services:
  victor:
    image: victor-ai:latest
    ports:
      - "8000:8000"
      - "9091:9091"
    environment:
      - VICTOR_PROMETHEUS_ENABLED=true
      - VICTOR_PROMETHEUS_PORT=9091
    restart: unless-stopped

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./configs/prometheus:/etc/prometheus
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.enable-lifecycle'
    restart: unless-stopped
    depends_on:
      - victor

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    volumes:
      - ./configs/grafana:/var/lib/grafana/dashboards
      - grafana-data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_INSTALL_PLUGINS=
    restart: unless-stopped
    depends_on:
      - prometheus

  alertmanager:
    image: prom/alertmanager
    ports:
      - "9093:9093"
    volumes:
      - ./configs/alertmanager:/etc/alertmanager
    restart: unless-stopped
    depends_on:
      - prometheus

volumes:
  prometheus-data:
  grafana-data:
```

Start the stack:

```bash
docker-compose -f docker-compose.monitoring.yml up -d
```

## Kubernetes Deployment

### Prometheus Operator

```bash
# Install Prometheus Operator
kubectl apply -f https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/main/bundle.yaml

# Create ServiceMonitor for Victor AI
cat > victor-servicemonitor.yaml <<EOF
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: victor-ai
  labels:
    app: victor-ai
spec:
  selector:
    matchLabels:
      app: victor-ai
  endpoints:
  - port: metrics
    interval: 15s
    path: /metrics
EOF

kubectl apply -f victor-servicemonitor.yaml
```

### Grafana Helm Chart

```bash
# Add Grafana Helm repository
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

# Install Grafana
helm install grafana grafana/grafana \
  --set persistence.storageClassName="standard" \
  --set persistence.size="10Gi" \
  --set adminPassword="admin" \
  --set service.type=LoadBalancer
```

## Troubleshooting

### Victor Metrics Not Appearing in Prometheus

**Symptoms**: Prometheus shows target as down or no metrics

**Solutions**:

1. Check if Victor AI metrics server is running:
   ```bash
   curl http://localhost:9091/metrics
   ```

2. Verify Prometheus configuration:
   ```bash
   # Check scrape config
   curl http://localhost:9090/api/v1/status/config | jq '.data.yml'
   ```

3. Check Prometheus logs:
   ```bash
   docker logs prometheus
   ```

4. Verify network connectivity:
   ```bash
   # From Prometheus container
   docker exec prometheus wget -O- http://victor:9091/metrics
   ```

### High Memory Usage

**Symptoms**: Grafana or Prometheus using excessive memory

**Solutions**:

1. Reduce Prometheus retention:
   ```yaml
   storage:
     tsdb:
       retention.time: 7d  # Reduce from 15d
   ```

2. Adjust scrape interval:
   ```yaml
   scrape_interval: 30s  # Increase from 15s
   ```

3. Add recording rules to reduce query complexity (already configured)

### Alerts Not Firing

**Symptoms**: Alert conditions met but no alerts triggered

**Solutions**:

1. Check AlertManager is running:
   ```bash
   curl http://localhost:9093/-/healthy
   ```

2. Verify alert rules are loaded:
   ```bash
   curl http://localhost:9090/api/v1/rules | jq '.data.groups[] | .rules[] | select(.type=="alerting")'
   ```

3. Check AlertManager logs:
   ```bash
   docker logs alertmanager
   ```

### Grafana Dashboards Not Loading

**Symptoms**: Dashboards show "No data"

**Solutions**:

1. Verify Prometheus datasource is configured:
   - Go to Configuration → Data Sources
   - Test connection

2. Check dashboard queries:
   - Click panel title → Edit
   - Verify query syntax

3. Check Grafana logs:
   ```bash
   docker logs grafana
   ```

## Best Practices

### 1. Metrics Collection

- **Scrape Interval**: 15 seconds for production
- **Retention**: 15 days for metrics data
- **Cardinality**: Keep label cardinality low (< 10,000 unique label combinations)

### 2. Alert Rules

- **Severity Levels**: Use critical, warning, info
- **Thresholds**: Base on historical baselines
- **For Duration**: Set appropriate for durations (2-5 minutes)
- **Documentation**: Include runbooks in alert annotations

### 3. Dashboards

- **Refresh Rate**: 30 seconds for overview, 15 seconds for performance
- **Time Range**: Default to last 1 hour
- **Annotations**: Mark deployments and incidents
- **Folders**: Organize dashboards by team/function

### 4. Storage

- **Prometheus**: Use SSD for better performance
- **Grafana**: Use persistent volumes
- **Backup**: Regular backups of Grafana dashboards and Prometheus configuration

### 5. Security

- **Authentication**: Enable authentication in production
- **TLS**: Use HTTPS for all endpoints
- **Firewall**: Restrict access to monitoring ports
- **Secrets**: Use environment variables or secret management

### 6. Performance

- **Recording Rules**: Pre-compute complex queries
- **Query Optimization**: Use efficient PromQL
- **Rate Limiting**: Set appropriate rate limits
- **Caching**: Leverage Grafana query caching

## Monitoring the Monitoring Stack

### Health Checks

```bash
# Prometheus health
curl http://localhost:9090/-/healthy

# Grafana health
curl http://localhost:3000/api/health

# AlertManager health
curl http://localhost:9093/-/healthy
```

### Metrics About Monitoring

Monitor the monitoring stack itself:

```promql
# Prometheus target health
up{job="prometheus"}

# Prometheus ingestion rate
rate(prometheus_tsdb_head_samples_appended_total[5m])

# Grafana request rate
rate(grafana_http_request_total[5m])
```

## Related Documentation

- [Production Metrics Guide](PRODUCTION_METRICS.md) - Complete metrics reference
- [Quick Reference](QUICK_REFERENCE.md) - Quick metrics overview
- [Structured Logging](STRUCTURED_LOGGING.md) - Logging setup
- [Health Checks](HEALTH_CHECKS.md) - Health monitoring
