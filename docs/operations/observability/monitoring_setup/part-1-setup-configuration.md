# Monitoring Setup Guide - Part 1

**Part 1 of 2:** Overview, Prerequisites, Quick Start, Detailed Setup, Configuration, and Verification

---

## Navigation

- **[Part 1: Setup & Configuration](#)** (Current)
- [Part 2: Deployment](part-2-deployment.md)
- [**Complete Guide](../MONITORING_SETUP.md)**

---
# Production Monitoring Setup Guide

This guide provides step-by-step instructions for setting up comprehensive monitoring for Victor AI in production
  environments.

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

---

## See Also

- [Documentation Home](../../README.md)


**Reading Time:** 4 min
**Last Updated:** February 08, 2026**
