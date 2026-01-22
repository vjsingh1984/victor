# Monitoring Stack - Quick Start Guide

## One-Command Deployment

```bash
./deployment/scripts/deploy_monitoring_complete.sh
```

## Access Your Monitoring Stack

```bash
# Forward all services
kubectl port-forward -n victor-monitoring svc/grafana 3000:3000 &
kubectl port-forward -n victor-monitoring svc/prometheus 9090:9090 &
kubectl port-forward -n victor-monitoring svc/alertmanager 9093:9093 &

# Open in browser
open http://localhost:3000  # Grafana (admin/changeme123)
open http://localhost:9090  # Prometheus
open http://localhost:9093  # AlertManager
```

## Common Commands

### Deploy with Email Notifications

```bash
./deployment/scripts/deploy_monitoring_complete.sh \
  --email admin@example.com
```

### Deploy with Slack Notifications

```bash
./deployment/scripts/deploy_monitoring_complete.sh \
  --slack-webhook https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

### Generate Secure Password

```bash
./deployment/scripts/setup_monitoring_credentials.sh --generate-password
```

### Verify Deployment

```bash
./deployment/scripts/verify_monitoring_complete.sh
```

### Run Tests

```bash
./deployment/scripts/test_monitoring_deployment.sh
```

## Dashboard List

1. **Overview** - System-wide overview
2. **Features** - Feature flag metrics
3. **Performance** - Performance metrics
4. **Team Overview** - Multi-agent teams
5. **Team Members** - Individual agents
6. **Team Performance** - Team performance
7. **Team Recursion** - Recursion tracking

## Key Metrics

### System Health
- `victor_health_status` - Overall health
- `victor_active_sessions` - Active sessions
- `victor_errors_total` - Error count

### Feature Performance
- `victor_feature_duration_ms_bucket` - Feature latency
- `victor_feature_executions_total` - Feature executions
- `victor_feature_errors_total` - Feature errors

### Model Performance
- `victor_model_requests_total` - Model requests
- `victor_model_latency_ms_bucket` - Model latency
- `victor_provider_circuit_state` - Circuit breaker state

### Tool Execution
- `victor_tool_calls_total` - Tool calls
- `victor_tool_errors_total` - Tool errors
- `victor_tool_timeouts_total` - Tool timeouts

## Troubleshooting

### Something's Not Working?

```bash
# Check all pods
kubectl get pods -n victor-monitoring

# Check pod logs
kubectl logs -n victor-monitoring deployment/prometheus --tail=50

# Verify everything
./deployment/scripts/verify_monitoring_complete.sh --detailed

# Run tests
./deployment/scripts/test_monitoring_deployment.sh
```

### Reset Password

```bash
./deployment/scripts/setup_monitoring_credentials.sh --generate-password
```

### Restart Everything

```bash
kubectl rollout restart deployment/prometheus -n victor-monitoring
kubectl rollout restart deployment/grafana -n victor-monitoring
kubectl rollout restart deployment/alertmanager -n victor-monitoring
```

## Cleanup

```bash
# Delete entire monitoring stack
kubectl delete namespace victor-monitoring

# Or just restart deployments
kubectl rollout restart deployment -n victor-monitoring
```

## Next Steps

1. ✅ Change default Grafana password
2. ✅ Configure notification channels (email/Slack)
3. ✅ Review alerting rules
4. ✅ Set up custom dashboards
5. ✅ Configure persistent storage
6. ✅ Enable TLS/SSL for production

## Documentation

- Full Guide: `deployment/monitoring/MONITORING_DEPLOYMENT_README.md`
- Deployment Guide: `deployment/monitoring/DEPLOYMENT_GUIDE.md`
- Runbook: `deployment/monitoring/runbook.md`
- Troubleshooting: `deployment/monitoring/troubleshooting.md`
