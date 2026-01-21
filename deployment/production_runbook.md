# Production Deployment Runbook

This runbook provides step-by-step procedures for deploying Victor to production environments, including troubleshooting guidance and rollback procedures.

## Table of Contents

- [Pre-Deployment](#pre-deployment)
- [Deployment Procedures](#deployment-procedures)
- [Post-Deployment Verification](#post-deployment-verification)
- [Rollback Procedures](#rollback-procedures)
- [Common Issues and Solutions](#common-issues-and-solutions)
- [Emergency Contacts](#emergency-contacts)

---

## Pre-Deployment

### 1. Pre-Deployment Checklist

Before deploying to production, ensure all items in the checklist are complete:

- [ ] Code reviewed and approved
- [ ] All tests passing (unit, integration, smoke)
- [ ] Security scan completed with no critical vulnerabilities
- [ ] Performance benchmarks acceptable
- [ ] Documentation updated
- [ ] Change request approved
- [ ] Stakeholders notified
- [ ] Maintenance window scheduled (if required)
- [ ] Backup of current deployment completed
- [ ] Rollback plan documented

### 2. Pre-Deployment Validation

Run the pre-deployment validation script:

```bash
./deployment/scripts/validate_deployment.sh production
```

This script validates:
- Cluster connectivity
- Configuration correctness
- Security settings
- Monitoring setup
- Resource availability
- Backup status
- Dependencies

**Expected Output**: Score >= 80/100

**If Score < 80**:
1. Review validation report
2. Address failed checks
3. Re-run validation
4. Do not proceed until score is acceptable

### 3. Create Backup

Ensure a fresh backup exists before deploying:

```bash
# Check latest backup
velero backup get -o json | jq -r '.items[] | select(.status.phase == "Completed") | .metadata.name' | sort -r | head -n1

# Create manual backup if needed
velero backup create pre-production-$(date +%Y%m%d-%H%M%S) --namespace victor --wait
```

---

## Deployment Procedures

### 1. Standard Deployment (Blue-Green)

For most deployments, use the blue-green strategy for zero downtime:

```bash
./deployment/scripts/deploy_production.sh \
  --environment production \
  --image-tag v1.0.0 \
  --strategy blue-green
```

**Process Flow**:

1. **Validation** - Runs pre-deployment checks
2. **Backup** - Creates automatic backup
3. **Deploy New Color** - Deploys to inactive color (blue/green)
4. **Health Checks** - Validates new deployment
5. **Traffic Switch** - Shifts traffic to new color
6. **Verification** - Runs post-deployment tests
7. **Rollback** - Automatic on failure

**Duration**: 10-15 minutes
**Downtime**: Zero

### 2. Rolling Update Deployment

For minor updates with backward compatibility:

```bash
./deployment/scripts/deploy_production.sh \
  --environment production \
  --image-tag v1.0.1 \
  --strategy rolling
```

**Process Flow**:

1. **Validation** - Runs pre-deployment checks
2. **Backup** - Creates automatic backup
3. **Apply Update** - Updates deployment with new image
4. **Rolling Update** - Gradually replaces pods
5. **Health Checks** - Validates during rollout
6. **Verification** - Runs post-deployment tests

**Duration**: 5-10 minutes
**Downtime**: Minimal (pod-level only)

### 3. Dry Run Deployment

Test deployment without making changes:

```bash
./deployment/scripts/deploy_production.sh \
  --environment production \
  --image-tag v1.0.0 \
  --dry-run
```

Use this to:
- Validate deployment scripts
- Test new deployment configurations
- Verify environment setup
- Train new operators

### 4. Interactive Deployment

Use the dashboard for guided deployment:

```bash
./deployment/scripts/deployment_dashboard.sh production
```

**Dashboard Features**:
- Real-time deployment status
- Interactive menu navigation
- Live log viewing
- Health monitoring
- One-click rollback

---

## Post-Deployment Verification

### 1. Automated Verification

Run the post-deployment verification script:

```bash
./deployment/scripts/verify_deployment.sh production
```

**Verification Checks**:
- Deployment health (replicas, pods)
- Service endpoints (HTTP, API)
- Database connectivity
- Redis connectivity
- Smoke tests
- Monitoring dashboards
- Log analysis
- Performance metrics

**Expected Output**: Health >= 80%

### 2. Manual Verification

Perform these manual checks:

#### Application Health

```bash
# Check deployment status
kubectl get deployment victor -n victor

# Check pod status
kubectl get pods -n victor -l app=victor

# Check service endpoints
kubectl get endpoints victor -n victor
```

#### Health Endpoints

```bash
# Get service endpoint
ENDPOINT=$(kubectl get service victor -n victor -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

# Test health endpoint
curl http://${ENDPOINT}/health

# Test API endpoint
curl http://${ENDPOINT}/api/v1/health
```

#### Logs Analysis

```bash
# Check for errors
kubectl logs -n victor -l app=victor --tail=100 | grep -i error

# Check for critical issues
kubectl logs -n victor -l app=victor --tail=100 | grep -i "critical\|fatal"
```

#### Monitoring

Access monitoring dashboards:
- Grafana: http://grafana.your-domain.com
- Prometheus: http://prometheus.your-domain.com

Check:
- Error rate (should be < 1%)
- Response time (should be < 500ms)
- CPU usage (should be < 70%)
- Memory usage (should be < 80%)

### 3. Smoke Tests

Run smoke tests to verify functionality:

```bash
# From dashboard: Menu -> Health Checks
# Or manually:
./deployment/scripts/verify_deployment.sh production
```

---

## Rollback Procedures

### 1. Automatic Rollback

If deployment verification fails, automatic rollback is triggered.

To enable automatic rollback, ensure `ROLLBACK_ON_FAILURE=true` in deployment script.

### 2. Manual Rollback to Previous Version

```bash
./deployment/scripts/rollback_production.sh production
```

This rolls back to the previous deployment revision.

### 3. Rollback to Specific Revision

```bash
# List available revisions
kubectl rollout history deployment/victor -n victor

# Rollback to specific revision
./deployment/scripts/rollback_production.sh production --revision 5
```

### 4. Rollback from Backup

```bash
# List available backups
velero backup get

# Rollback from backup
./deployment/scripts/rollback_production.sh production --backup pre-production-20231201-120000
```

### 5. Manual Rollback Steps

If scripts fail, perform manual rollback:

```bash
# 1. Undo last deployment
kubectl rollout undo deployment/victor -n victor

# 2. Wait for rollout
kubectl rollout status deployment/victor -n victor --timeout=300s

# 3. Verify health
kubectl get pods -n victor -l app=victor

# 4. Test endpoint
ENDPOINT=$(kubectl get service victor -n victor -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
curl http://${ENDPOINT}/health
```

### 6. Blue-Green Rollback

For blue-green deployments, switch traffic back:

```bash
# Get current color
kubectl get service victor -n victor -o jsonpath='{.spec.selector.color}'

# Switch to other color
kubectl patch service victor -n victor --type=json \
  -p='[{"op": "replace", "path": "/spec/selector/color", "value": "blue"}]'
```

---

## Common Issues and Solutions

### Issue 1: Deployment Timeout

**Symptoms**:
- Deployment takes longer than expected
- Health checks timeout
- Pods stuck in pending state

**Solutions**:

1. Check resource availability:
```bash
kubectl top nodes
kubectl describe pod <pod-name> -n victor
```

2. Check image pull:
```bash
kubectl describe pod <pod-name> -n victor | grep -i image
```

3. Increase timeout:
```bash
kubectl rollout status deployment/victor -n victor --timeout=600s
```

### Issue 2: Pods Not Ready

**Symptoms**:
- Pods show as not ready
- Readiness probes failing
- Service not receiving traffic

**Solutions**:

1. Check pod events:
```bash
kubectl describe pod <pod-name> -n victor
```

2. Check pod logs:
```bash
kubectl logs <pod-name> -n victor --all-containers=true
```

3. Check readiness probe:
```bash
kubectl get pod <pod-name> -n victor -o jsonpath='{.spec.containers[*].readinessProbe}'
```

4. Common fixes:
- Fix application startup issues
- Adjust readiness probe parameters
- Check database connectivity
- Verify environment variables

### Issue 3: High Error Rate

**Symptoms**:
- 500 errors in logs
- High error rate in monitoring
- Application crashes

**Solutions**:

1. Check recent logs:
```bash
kubectl logs -n victor -l app=victor --tail=100 | grep -i error
```

2. Check pod restarts:
```bash
kubectl get pods -n victor -l app=victor
```

3. Common causes:
- Database connection issues
- Memory leaks
- Configuration errors
- Dependency failures

4. Immediate actions:
- Rollback deployment
- Scale up replicas
- Restart pods
- Check dependencies

### Issue 4: Database Connection Failures

**Symptoms**:
- Application can't connect to database
- Connection timeout errors
- Authentication failures

**Solutions**:

1. Check database statefulset:
```bash
kubectl get statefulset postgresql -n victor
kubectl get pods -n victor -l app=postgresql
```

2. Test database connection:
```bash
kubectl exec -it <postgres-pod> -n victor -- psql -U victor -d victor
```

3. Check secrets:
```bash
kubectl get secret victor-database-secret -n victor -o json
```

4. Common fixes:
- Restart database pods
- Verify secrets and credentials
- Check network policies
- Increase connection pool size

### Issue 5: Redis Connection Failures

**Symptoms**:
- Cache errors in logs
- Session management failures
- Slow performance

**Solutions**:

1. Check Redis statefulset:
```bash
kubectl get statefulset redis -n victor
kubectl get pods -n victor -l app=redis
```

2. Test Redis connection:
```bash
kubectl exec -it <redis-pod> -n victor -- redis-cli ping
```

3. Common fixes:
- Restart Redis pods
- Check Redis memory settings
- Verify connection settings
- Clear cache if corrupted

### Issue 6: Performance Degradation

**Symptoms**:
- Slow response times
- High CPU/memory usage
- Increased latency

**Solutions**:

1. Check resource usage:
```bash
kubectl top pods -n victor -l app=victor
kubectl top nodes
```

2. Check horizontal pod autoscaler:
```bash
kubectl get hpa -n victor
```

3. Common fixes:
- Scale up deployment
- Adjust resource limits
- Optimize application code
- Review database queries

### Issue 7: Image Pull Errors

**Symptoms**:
- ImagePullBackOff or ErrImagePull
- Pods can't start
- Authentication errors

**Solutions**:

1. Check image name:
```bash
kubectl get deployment victor -n victor -o jsonpath='{.spec.template.spec.containers[*].image}'
```

2. Create image pull secret:
```bash
kubectl create secret docker-registry regcred \
  --docker-server=<registry-url> \
  --docker-username=<username> \
  --docker-password=<password> \
  -n victor
```

3. Common fixes:
- Verify image exists in registry
- Check credentials
- Pull image manually to test
- Use correct image tag

### Issue 8: Monitoring Not Working

**Symptoms**:
- No metrics in Grafana
- Prometheus targets down
- Alerts not firing

**Solutions**:

1. Check Prometheus:
```bash
kubectl get statefulset prometheus -n monitoring
kubectl get pods -n monitoring -l app=prometheus
```

2. Check targets:
```bash
kubectl port-forward -n monitoring svc/prometheus 9090:9090
# Access http://localhost:9090/targets
```

3. Common fixes:
- Restart Prometheus
- Check ServiceMonitor configuration
- Verify network policies
- Update Prometheus configuration

---

## Emergency Contacts

### On-Call Team

| Role | Name | Contact | Hours |
|------|------|---------|-------|
| Site Reliability Lead | [Name] | [Email/Phone] | 24/7 |
| DevOps Engineer | [Name] | [Email/Phone] | Business hours |
| Backend Lead | [Name] | [Email/Phone] | Business hours |

### Escalation Path

1. **Level 1**: Deploying engineer
2. **Level 2**: Site Reliability Lead (if issue > 15 min)
3. **Level 3**: Engineering Manager (if issue > 30 min)
4. **Level 4**: CTO (if critical incident)

### Communication Channels

- **Slack**: #production-deployments
- **Incident Channel**: #incident-{date}
- **Email**: production-alerts@your-domain.com

---

## Best Practices

### Before Deployment

1. **Always run validation** - Never skip pre-deployment checks
2. **Create backups** - Ensure recent backups exist
3. **Test in staging** - Validate in staging first
4. **Notify stakeholders** - Communicate deployment windows
5. **Document changes** - Maintain change logs

### During Deployment

1. **Monitor progress** - Watch logs and metrics
2. **Be ready to rollback** - Have rollback plan ready
3. **Communicate status** - Update team on progress
4. **Document issues** - Record any problems encountered

### After Deployment

1. **Run verification** - Ensure deployment is healthy
2. **Monitor metrics** - Watch for anomalies
3. **Check logs** - Look for errors
4. **Update documentation** - Document any changes
5. **Post-mortem** - Conduct review if issues occurred

### Rollback Decisions

Rollback immediately if:
- Error rate > 5%
- Response time > 3x baseline
- Database connection failures
- Authentication failures
- Data corruption detected
- Security breach suspected

---

## Training and Onboarding

### New Operator Checklist

- [ ] Read this runbook completely
- [ ] Practice deployments in staging
- [ ] Complete dry-run deployment
- [ ] Practice rollback procedure
- [ ] Review monitoring dashboards
- [ ] Learn to use deployment dashboard
- [ ] Join deployment training session
- [ ] Shadow experienced operator
- [ ] Complete supervised deployment
- [ ] Approved for independent deployments

### Resources

- **Deployment Scripts**: `/deployment/scripts/`
- **Configuration**: `/deployment/kubernetes/`
- **Documentation**: `/docs/`
- **Monitoring**: http://grafana.your-domain.com
- **Logs**: Accessible via dashboard or `kubectl logs`

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-01-21 | Initial production runbook |

---

## Appendix

### A. Quick Reference Commands

```bash
# Quick deployment
./deployment/scripts/deploy_production.sh --environment production --image-tag v1.0.0

# Quick rollback
./deployment/scripts/rollback_production.sh production

# Check status
kubectl get deployment victor -n victor

# View logs
./deployment/scripts/deployment_dashboard.sh production

# Run diagnostics
kubectl get events -n victor --sort-by='.lastTimestamp'
```

### B. Useful Aliases

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
alias victor-deploy='~/code/codingagent/deployment/scripts/deploy_production.sh'
alias victor-rollback='~/code/codingagent/deployment/scripts/rollback_production.sh'
alias victor-dashboard='~/code/codingagent/deployment/scripts/deployment_dashboard.sh'
alias victor-logs='kubectl logs -n victor -l app=victor -f --all-containers=true'
alias victor-status='kubectl get all -n victor -l app=victor'
```

### C. Environment Variables

```bash
export KUBECONFIG=~/code/codingagent/deployment/kubeconfig_production
export VICTOR_ENV=production
export DEPLOYMENT_TIMEOUT=600
```

---

**Document Owner**: DevOps Team
**Last Updated**: 2024-01-21
**Next Review**: 2024-02-21
