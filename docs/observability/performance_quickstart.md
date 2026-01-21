# Performance Monitoring Quick Start

Quick guide to get started with Victor AI performance monitoring.

## 1-Minute Setup

```bash
# Run the setup script
./scripts/setup_performance_monitoring.sh

# Open Grafana
kubectl port-forward -n victor-monitoring svc/grafana 3000:3000
open http://localhost:3000
```

## 5-Minute Setup (Manual)

### Step 1: Deploy Performance Alerts (1 min)

```bash
kubectl apply -f deployment/kubernetes/monitoring/performance-alerts.yaml
kubectl rollout restart deployment/prometheus -n victor-monitoring
```

### Step 2: Deploy Grafana Dashboard (1 min)

```bash
kubectl port-forward -n victor-monitoring svc/grafana 3000:3000
# Open http://localhost:3000
# Go to Dashboards -> Import
# Upload: deployment/kubernetes/monitoring/dashboards/victor-performance.json
```

### Step 3: Verify Metrics (1 min)

```bash
# Check Prometheus is scraping
kubectl port-forward -n victor-monitoring svc/prometheus 9090:9090
# Open http://localhost:9090/targets

# Check metrics endpoint
kubectl port-forward -n victor-production svc/victor-api 8000:8000
curl http://localhost:8000/api/performance/summary
```

### Step 4: Test Alerts (1 min)

```bash
# Check AlertManager
kubectl port-forward -n victor-monitoring svc/alertmanager 9093:9093
# Open http://localhost:9093/#/alerts

# View Prometheus alerts
# Open http://localhost:9090/alerts
```

### Step 5: Monitor Dashboard (1 min)

Open Grafana dashboard and monitor:
- Cache hit rate (should be > 40%)
- Selection latency (P95 should be < 1ms)
- Memory usage (should be < 1GB)
- Tool error rate (should be < 5%)

## Key Metrics

| Metric | Good | Warning | Critical |
|--------|------|---------|----------|
| Cache Hit Rate | > 60% | < 40% | < 20% |
| Selection Latency P95 | < 1ms | > 1ms | > 5ms |
| Memory Usage | < 1GB | > 1GB | > 2GB |
| Tool Error Rate | < 1% | > 5% | > 15% |
| CPU Usage | < 70% | > 80% | > 95% |

## Common Commands

```bash
# View all performance metrics
curl http://localhost:8000/api/performance/summary | jq

# View cache metrics only
curl http://localhost:8000/api/performance/cache | jq

# View Prometheus format
curl http://localhost:8000/api/performance/prometheus

# Port-forward services
kubectl port-forward -n victor-monitoring svc/grafana 3000:3000 &
kubectl port-forward -n victor-monitoring svc/prometheus 9090:9090 &
kubectl port-forward -n victor-monitoring svc/alertmanager 9093:9093 &

# Check Prometheus targets
open http://localhost:9090/targets

# Check Prometheus alerts
open http://localhost:9090/alerts

# Check AlertManager
open http://localhost:9093
```

## Troubleshooting

### No Data in Dashboard

```bash
# Check if metrics endpoint is accessible
curl http://localhost:8000/api/performance/prometheus

# Check Prometheus is scraping
open http://localhost:9090/targets

# Restart Prometheus
kubectl rollout restart deployment/prometheus -n victor-monitoring
```

### Alerts Not Firing

```bash
# Check alert rules are loaded
open http://localhost:9090/rules

# Reload Prometheus config
kubectl rollout restart deployment/prometheus -n victor-monitoring
```

### High Memory Usage

```bash
# Check current usage
curl http://localhost:8000/api/performance/system | jq '.memory'

# Reduce cache size
kubectl set env deployment/victor-api VICTOR_CACHE_SIZE=500

# Scale deployment
kubectl scale deployment/victor-api --replicas=3
```

## Dashboard Panels

### Overview Row
- **Tool Selection Latency**: P95 latency in ms
- **Cache Hit Rate**: Overall cache effectiveness
- **Memory Usage**: System memory consumption
- **System Uptime**: How long Victor has been running

### Cache Performance Row
- **Cache Entries**: Number of entries by namespace
- **Hit/Miss Ratio**: Visual breakdown of cache operations
- **Cache Evictions**: Rate of entry evictions
- **Cache Utilization**: Cache fullness percentage

### Tool Selection Row
- **Selection Latency**: P50/P95/P99 percentiles
- **Hit Rate by Type**: Query, context, RL breakdown
- **Selection Errors**: Miss rate over time

### Provider Pool Row
- **Provider Health**: Status of each provider
- **Provider Latency**: P95 latency by provider
- **Active Providers**: Count of healthy providers

### Tool Execution Row
- **Execution Duration**: P95/P99 tool execution times
- **Execution Errors**: Error rate over time
- **Top Tools**: Most frequently used tools

## Next Steps

1. **Read full documentation**: `docs/observability/performance_monitoring.md`
2. **Customize alerts**: Edit `deployment/kubernetes/monitoring/performance-alerts.yaml`
3. **Customize dashboard**: Edit in Grafana UI and export
4. **Set up notifications**: Configure AlertManager routing
5. **Create custom metrics**: Extend `PerformanceMetricsCollector`

## Support

- **Documentation**: `docs/observability/performance_monitoring.md`
- **Issues**: https://github.com/yourusername/victor-ai/issues
- **Community**: https://discord.gg/victor-ai
