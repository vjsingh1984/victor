# Monitoring Setup Guide - Part 2

**Part 2 of 2:** Docker Compose Setup, Kubernetes Deployment, Troubleshooting, Best Practices, and Monitoring the Stack

---

## Navigation

- [Part 1: Setup & Configuration](part-1-setup-configuration.md)
- **[Part 2: Deployment](#)** (Current)
- [**Complete Guide](../MONITORING_SETUP.md)**

---
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

---

**Last Updated:** February 01, 2026
**Reading Time:** 4 minutes
