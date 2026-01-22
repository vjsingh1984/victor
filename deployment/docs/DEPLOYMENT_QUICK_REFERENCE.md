# Production Deployment Quick Reference

Quick reference guide for the master deployment orchestration script.

## Quick Start Commands

### Complete Production Deployment
```bash
./deployment/scripts/deploy_production_complete.sh \
  --environment production \
  --image-tag v1.0.0
```

### Staging Deployment (Fast)
```bash
./deployment/scripts/deploy_production_complete.sh \
  --environment staging \
  --skip-monitoring \
  --skip-backups
```

### Dry Run (No Changes)
```bash
./deployment/scripts/deploy_production_complete.sh \
  --environment production \
  --dry-run
```

### Application Update Only
```bash
./deployment/scripts/deploy_production_complete.sh \
  --environment production \
  --image-tag v1.1.0 \
  --skip-infrastructure \
  --skip-monitoring \
  --skip-backups
```

### Blue-Green Deployment
```bash
./deployment/scripts/deploy_production_complete.sh \
  --environment production \
  --strategy blue-green
```

### Rolling Update
```bash
./deployment/scripts/deploy_production_complete.sh \
  --environment production \
  --strategy rolling
```

## Common Options

| Option | Description |
|--------|-------------|
| `--environment ENV` | Target environment (production/staging/development) |
| `--image-tag TAG` | Docker image tag |
| `--strategy STRAT` | Deployment strategy (blue-green/rolling) |
| `--dry-run` | Validate without applying changes |
| `--skip-infrastructure` | Skip PostgreSQL, Redis, Ingress deployment |
| `--skip-monitoring` | Skip Prometheus, Grafana deployment |
| `--skip-backups` | Skip backup setup |
| `--skip-validation` | Skip pre-deployment validation |
| `--continue-on-failure` | Continue deployment on stage failure |
| `--no-rollback` | Disable automatic rollback |

## Deployment Stages

| # | Stage | Time | Skip Flag |
|---|-------|------|------------|
| 1 | Environment Validation | ~5 min | `--skip-validation` |
| 2 | Infrastructure Setup | ~10 min | `--skip-infrastructure` |
| 3 | Monitoring Stack | ~5 min | `--skip-monitoring` |
| 4 | Secrets Configuration | ~2 min | `--skip-secrets` |
| 5 | Backup Setup | ~3 min | `--skip-backups` |
| 6 | Application Deployment | ~10 min | - |
| 7 | Post-Deployment Verification | ~5 min | - |

**Total Time:** ~40 minutes

## Secrets Setup

### Quick Secrets Setup

```bash
# Database secret
kubectl create secret generic victor-database-secret \
  --from-literal=POSTGRES_USER=victor \
  --from-literal=POSTGRES_PASSWORD=changeme \
  --from-literal=POSTGRES_DB=victordb \
  -n victor

# Provider secrets
kubectl create secret generic victor-provider-secrets \
  --from-literal=ANTHROPIC_API_KEY=your-key \
  --from-literal=OPENAI_API_KEY=your-key \
  -n victor
```

## Rollback Commands

### Quick Rollback
```bash
# Rollback to previous version
./deployment/scripts/rollback_production.sh production

# Rollback to specific revision
./deployment/scripts/rollback_production.sh production --revision 5

# Force rollback without confirmation
./deployment/scripts/rollback_production.sh production --force
```

## Verification Commands

### Check Deployment Status
```bash
# Check pods
kubectl get pods -n victor

# Check services
kubectl get svc -n victor

# Check deployment status
kubectl rollout status deployment/victor -n victor

# Check logs
kubectl logs -n victor -l app=victor --tail=50
```

### Health Check
```bash
# Get service endpoint
kubectl get svc victor -n victor

# Test health endpoint
curl http://<service-ip>/health
```

## Monitoring Access

### Port Forwarding
```bash
# Grafana
kubectl port-forward -n victor-monitoring svc/grafana 3000:3000

# Prometheus
kubectl port-forward -n victor-monitoring svc/prometheus 9090:9090
```

### Access URLs
- Grafana: http://localhost:3000 (admin/changeme123)
- Prometheus: http://localhost:9090

## Artifact Locations

| Artifact | Location |
|----------|----------|
| Deployment Log | `logs/deployment_<ID>.log` |
| Deployment Report | `reports/deployment_<ID>.json` |
| Backups | `backups/<DEPLOYMENT_ID>/` |
| State File | `logs/.deploy_state` |

## Troubleshooting Quick Commands

```bash
# Check what's failing
kubectl describe pod <pod-name> -n victor

# View logs
kubectl logs <pod-name> -n victor --tail=100 -f

# Check events
kubectl get events -n victor --sort-by='.lastTimestamp'

# Get all resources
kubectl get all -n victor

# Check resource usage
kubectl top pods -n victor
kubectl top nodes
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Cluster connectivity failed |
| 3 | Validation failed |
| 4 | Infrastructure deployment failed |
| 5 | Application deployment failed |
| 6 | Verification failed |
| 7 | Rollback failed |

## Resource Requirements

**Minimum:**
- CPU: 3.5 cores
- Memory: 4.5 GB
- Storage: 71 GB

**Recommended:**
- CPU: 7.5 cores
- Memory: 10 GB
- Storage: 310 GB

## Pre-Deployment Checklist

- [ ] Cluster connectivity verified
- [ ] Sufficient resources available
- [ ] All secrets configured
- [ ] Tested in staging
- [ ] Monitoring deployed
- [ ] Backups configured
- [ ] Rollback plan ready
- [ ] Team notified

## Post-Deployment Verification

- [ ] All pods running
- [ ] All services accessible
- [ ] Health checks passing
- [ ] Logs error-free
- [ ] Metrics collecting
- [ ] Alerts configured
- [ ] Backups scheduled
- [ ] Documentation updated

## Emergency Procedures

```bash
# Immediate rollback
./deployment/scripts/rollback_production.sh production --force

# Scale down
kubectl scale deployment victor -n victor --replicas=0

# Pause rollout
kubectl rollout pause deployment victor -n victor

# Resume rollout
kubectl rollout resume deployment victor -n victor
```

## Useful Aliases

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
# Deployment aliases
alias victor-deploy-prod='./deployment/scripts/deploy_production_complete.sh --environment production'
alias victor-deploy-staging='./deployment/scripts/deploy_production_complete.sh --environment staging'
alias victor-rollback-prod='./deployment/scripts/rollback_production.sh production'

# Monitoring aliases
alias victor-grafana='kubectl port-forward -n victor-monitoring svc/grafana 3000:3000'
alias victor-prometheus='kubectl port-forward -n victor-monitoring svc/prometheus 9090:9090'

# Status aliases
alias victor-pods='kubectl get pods -n victor'
alias victor-logs='kubectl logs -n victor -l app=victor --tail=50 -f'
alias victor-status='kubectl rollout status deployment/victor -n victor'
```

## CI/CD Integration Examples

### GitLab CI
```yaml
deploy_production:
  script:
    - ./deployment/scripts/deploy_production_complete.sh
        --environment production
        --image-tag ${CI_COMMIT_TAG}
  only:
    - tags
```

### GitHub Actions
```yaml
- name: Deploy to production
  run: |
    ./deployment/scripts/deploy_production_complete.sh \
      --environment production \
      --image-tag ${{ github.ref_name }}
```

## Getting Help

```bash
# Show help
./deployment/scripts/deploy_production_complete.sh --help

# Check logs
cat logs/deployment_*.log

# View report
cat reports/deployment_*.json | jq .

# Full documentation
cat deployment/docs/DEPLOYMENT_ORCHESTRATION_GUIDE.md
```

---

**For detailed documentation, see:**
[Deployment Orchestration Guide](./DEPLOYMENT_ORCHESTRATION_GUIDE.md)

**Last Updated:** 2024-01-21
