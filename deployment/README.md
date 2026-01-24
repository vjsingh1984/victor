# Victor AI Production Deployment

Complete production deployment infrastructure for Victor AI with orchestration, monitoring, backups, and comprehensive validation.

## Overview

This deployment automation provides enterprise-grade deployment capabilities:

- **Master Orchestration**: Single-command complete infrastructure setup
- **Zero-Downtime Deployment**: Blue-green and rolling update strategies
- **Automatic Rollback**: Health-based rollback with stage gating
- **Progress Tracking**: Time estimates and real-time progress
- **Comprehensive Validation**: Pre and post-deployment verification
- **Monitoring Stack**: Integrated Prometheus, Grafana, AlertManager
- **Backup Automation**: Velero integration with automated backups
- **Deployment Reporting**: JSON reports and detailed logging

## Quick Start

### One-Command Production Deployment

```bash
# Deploy everything (infrastructure + monitoring + application)
./deployment/scripts/deploy_production_complete.sh \
  --environment production \
  --image-tag v0.5.0
```

**Estimated time:** ~40 minutes

### Documentation

| Document | Description |
|----------|-------------|
| **[Deployment Orchestration Guide](./docs/DEPLOYMENT_ORCHESTRATION_GUIDE.md)** | Complete deployment guide with detailed instructions |
| **[Quick Reference](./docs/DEPLOYMENT_QUICK_REFERENCE.md)** | Quick command reference and common scenarios |
| **[Kubernetes Guide](./kubernetes/README.md)** | Kubernetes manifests and configurations |

## Deployment Options

### Option 1: Complete Infrastructure Setup

**Use Case:** First-time production deployment

```bash
./deployment/scripts/deploy_production_complete.sh \
  --environment production \
  --image-tag v0.5.0
```

**What happens:**
1. Validates environment and prerequisites (~5 min)
2. Deploys PostgreSQL, Redis, Ingress (~10 min)
3. Deploys monitoring stack (~5 min)
4. Configures secrets (~2 min)
5. Sets up backups (~3 min)
6. Deploys application (~10 min)
7. Runs verification (~5 min)

### Option 2: Application Update Only

**Use Case:** Update existing deployment

```bash
./deployment/scripts/deploy_production_complete.sh \
  --environment production \
  --image-tag v1.1.0 \
  --skip-infrastructure \
  --skip-monitoring
```

### Option 3: Blue-Green Deployment

**Use Case:** Zero-downtime production deployment

```bash
./deployment/scripts/deploy_production_complete.sh \
  --environment production \
  --strategy blue-green \
  --image-tag v2.0.0
```

### Option 4: Dry Run (Validation)

**Use Case:** Test deployment without changes

```bash
./deployment/scripts/deploy_production_complete.sh \
  --environment production \
  --dry-run
```

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  MASTER ORCHESTRATION SCRIPT                 │
│           deploy_production_complete.sh                      │
│  - Coordinates all stages                                    │
│  - Progress tracking with time estimates                     │
│  - Automatic rollback on failure                             │
│  - Comprehensive reporting                                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┬──────────────┐
        │              │              │              │
        ▼              ▼              ▼              ▼
┌──────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│  Validation  │ │  Infra   │ │ Monitor  │ │  Deploy  │
│              │ │          │ │          │ │          │
│ - Cluster    │ │ - PostgreSQL│-Prometheus│- Application│
│ - Config     │ │ - Redis  │ │ - Grafana│ │ - Health│
│ - Security   │ │ - Ingress│ │ - Alerts│ │ - Verify│
└──────────────┘ └──────────┘ └──────────┘ └──────────┘
```

## Traditional Deployment Workflow

### 1. Pre-Deployment Validation
./deployment/scripts/deploy_production.sh \
  --environment production \
  --image-tag v0.5.0 \
  --dry-run
```

### 3. Verify Deployment

```bash
./deployment/scripts/verify_deployment.sh production
```

Verifies:
- Deployment health
- Service endpoints
- Database connectivity
- Redis connectivity
- Smoke tests
- Monitoring dashboards
- Log analysis
- Performance metrics

### 4. Rollback (if needed)

```bash
# Rollback to previous version
./deployment/scripts/rollback_production.sh production

# Rollback to specific revision
./deployment/scripts/rollback_production.sh production --revision 5

# Rollback from backup
./deployment/scripts/rollback_production.sh production --backup pre-deploy-20231201-120000
```

### 5. Interactive Dashboard

```bash
./deployment/scripts/deployment_dashboard.sh production
```

Features:
- Real-time deployment status
- Live log viewing
- Health monitoring
- Resource usage
- One-click deployment/rollback

## Deployment Automation Scripts

### validate_deployment.sh (530 lines)

Pre-deployment validation script that checks production readiness.

**Usage**:
```bash
./deployment/scripts/validate_deployment.sh [environment]
```

**Checks**:
- Cluster connectivity
- Configuration validation
- Security scans
- Monitoring readiness
- Resource availability
- Backup verification
- Dependencies

**Output**:
- Validation score (0-100)
- JSON report
- Detailed error list

### deploy_production.sh (650 lines)

Main deployment script with blue-green and rolling update support.

**Usage**:
```bash
./deployment/scripts/deploy_production.sh \
  --environment [production|staging|development] \
  --image-tag TAG \
  --strategy [blue-green|rolling] \
  [--skip-validation] \
  [--dry-run]
```

**Features**:
- Pre-deployment validation
- Automatic backup
- Blue-green deployment
- Rolling updates
- Health checks
- Automatic rollback
- Progress monitoring
- JSON reporting

### verify_deployment.sh (580 lines)

Post-deployment verification with comprehensive health checks.

**Usage**:
```bash
./deployment/scripts/verify_deployment.sh [environment]
```

**Verifications**:
- Deployment health
- Service endpoints
- Database connectivity
- Redis connectivity
- Smoke tests
- Monitoring
- Log analysis
- Performance metrics

**Output**:
- Health score (0-100)
- JSON report
- Detailed status

### rollback_production.sh (480 lines)

Automated rollback with health-based verification.

**Usage**:
```bash
./deployment/scripts/rollback_production.sh [environment] \
  [--revision REVISION] \
  [--backup BACKUP_ID] \
  [--force] \
  [--dry-run]
```

**Features**:
- Rollback to previous version
- Rollback to specific revision
- Restore from backup
- Health verification
- Traffic switching
- JSON reporting

### deployment_dashboard.sh (560 lines)

Interactive dashboard for deployment management.

**Usage**:
```bash
./deployment/scripts/deployment_dashboard.sh [environment]
```

**Menu Options**:
1. Deployment Status - Real-time status and metrics
2. Deploy New Version - Interactive deployment wizard
3. Rollback Deployment - Interactive rollback wizard
4. View Logs - Live log tailing
5. Health Checks - Comprehensive health monitoring
6. Monitoring - Resource usage and links
7. Configuration - Environment and configuration viewing
8. Run Diagnostics - Automated diagnostics

## Documentation

### production_runbook.md (680 lines)

Comprehensive deployment runbook with step-by-step procedures.

**Contents**:
- Pre-deployment procedures
- Deployment procedures (blue-green, rolling)
- Post-deployment verification
- Rollback procedures
- Common issues and solutions (8 issues)
- Emergency contacts
- Best practices
- Training and onboarding

### production_checklist.md (390 lines)

Complete checklist for production deployments.

**Sections**:
- Pre-deployment phase (25 items)
- Infrastructure readiness (25 items)
- Pre-deployment validation (3 items)
- Deployment execution (13 items)
- Post-deployment verification (18 items)
- Rollback criteria (10 items)
- Post-deployment actions (5 items)
- Documentation and communication (3 items)
- Sign-off (5 items)

## Deployment Strategies

### Blue-Green Deployment (Recommended)

**Best for**: Major releases, breaking changes

**Advantages**:
- Zero downtime
- Instant rollback
- Easy testing before cutover
- Minimal risk

**Process**:
1. Deploy to inactive color (blue/green)
2. Health check new deployment
3. Switch traffic to new color
4. Verify production traffic
5. Keep old deployment for quick rollback

**Duration**: 10-15 minutes

### Rolling Update Deployment

**Best for**: Minor updates, backward-compatible changes

**Advantages**:
- Gradual rollout
- Resource efficient
- Faster deployment
- Can detect issues early

**Process**:
1. Update deployment with new image
2. Gradually replace pods
3. Health check each pod
4. Continue until all pods updated

**Duration**: 5-10 minutes

## Deployment Workflow

### Standard Deployment Flow

```
1. Pre-Deployment Validation
   └─> ./deployment/scripts/validate_deployment.sh production
       └─> Score >= 80/100 required

2. Deploy (with automatic rollback)
   └─> ./deployment/scripts/deploy_production.sh --environment production
       ├─> Backup current state
       ├─> Deploy new version
       ├─> Health checks
       ├─> Traffic switch
       └─> Automatic rollback on failure

3. Post-Deployment Verification
   └─> ./deployment/scripts/verify_deployment.sh production
       └─> Health >= 80% required

4. Monitor and Validate
   └─> ./deployment/scripts/deployment_dashboard.sh production
       └─> Monitor for 2 hours
```

### Rollback Flow

```
Automatic Rollback (on failure)
   └─> Triggered by health check failure
       ├─> Stop deployment
       ├─> Rollback to previous version
       ├─> Verify rollback
       └─> Generate incident report

Manual Rollback
   └─> ./deployment/scripts/rollback_production.sh production
       ├─> Choose rollback target
       ├─> Execute rollback
       ├─> Verify rollback
       └─> Post-incident review
```

## Safety Features

### Automatic Rollback Triggers

Rollback occurs automatically if:
- Error rate > 5%
- Response time > 3x baseline
- Database connection failures
- Authentication failures
- Health check timeout
- Pod restart threshold exceeded

### Pre-Deployment Checks

Before deployment, the system validates:
- Cluster health and connectivity
- Configuration correctness
- Security vulnerabilities
- Resource availability
- Monitoring readiness
- Backup status
- Dependencies

### Post-Deployment Verification

After deployment, the system verifies:
- All replicas ready
- All pods running
- Endpoints responding
- Database connected
- Redis connected
- Smoke tests passing
- Metrics flowing
- No critical errors

## Monitoring and Alerting

### Key Metrics

- **Deployment Status**: Replicas, pods, rollout status
- **Health**: HTTP 200 rate, error rate, response time
- **Resources**: CPU, memory, disk usage
- **Connections**: Database, Redis, external APIs
- **Logs**: Error rate, critical issues

### Grafana Dashboards

- **Deployment Overview**: Deployment status and metrics
- **Application Health**: Health checks and endpoints
- **Resource Usage**: CPU, memory, network
- **Database Performance**: Queries, connections, locks
- **Error Tracking**: Error rate, types, patterns

### Alerts

- **DeploymentFailed**: Deployment not successful
- **HighErrorRate**: Error rate > 5%
- **SlowResponse**: Response time > 3x baseline
- **PodNotReady**: Pod not ready > 5 minutes
- **DatabaseDown**: Database not accessible
- **RedisDown**: Redis not accessible

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
4. **Document issues** - Record any problems

### After Deployment

1. **Run verification** - Ensure deployment is healthy
2. **Monitor metrics** - Watch for anomalies
3. **Check logs** - Look for errors
4. **Update documentation** - Document any changes
5. **Post-mortem** - Conduct review if issues occurred

## Troubleshooting

### Common Issues

1. **Deployment Timeout**
   - Check: `kubectl top nodes`
   - Fix: Increase timeout, check resources

2. **Pods Not Ready**
   - Check: `kubectl describe pod <pod-name>`
   - Fix: Review logs, check dependencies

3. **High Error Rate**
   - Check: `kubectl logs -l app=victor`
   - Fix: Rollback, investigate logs

4. **Database Connection Failures**
   - Check: `kubectl get statefulset postgresql`
   - Fix: Restart database, verify secrets

5. **Image Pull Errors**
   - Check: Image tag, registry credentials
   - Fix: Verify image, update secrets

See `production_runbook.md` for detailed troubleshooting guide with 8 common issues and solutions.

## Training

### New Operators

1. Read `production_runbook.md` completely
2. Practice deployments in staging
3. Complete dry-run deployment
4. Practice rollback procedure
5. Review monitoring dashboards
6. Learn to use deployment dashboard
7. Shadow experienced operator
8. Complete supervised deployment

### Commands to Learn

```bash
# Basic commands
kubectl get all -n victor
kubectl logs -n victor -l app=victor -f
kubectl describe pod <pod-name> -n victor

# Deployment commands
./deployment/scripts/validate_deployment.sh production
./deployment/scripts/deploy_production.sh --environment production
./deployment/scripts/verify_deployment.sh production
./deployment/scripts/rollback_production.sh production
./deployment/scripts/deployment_dashboard.sh production
```

## Support

### Emergency Contacts

- **On-Call DevOps**: [Phone/Email]
- **Engineering Manager**: [Phone/Email]
- **CTO**: [Phone/Email]
- **Incident Channel**: #incident-{date}

### Documentation

- **Runbook**: `deployment/production_runbook.md`
- **Checklist**: `deployment/production_checklist.md`
- **API Docs**: `/docs/api/`
- **Architecture**: `/docs/architecture/`

### Communication Channels

- **Slack**: #production-deployments
- **Email**: production-alerts@your-domain.com

## Metrics and Reporting

### Deployment Success Rate

Track deployment success rate over time:
- Target: > 95%
- Measure: Successful deployments / Total deployments

### Mean Time to Recovery (MTTR)

Track how quickly issues are resolved:
- Target: < 15 minutes
- Measure: Time from detection to resolution

### Deployment Frequency

Track how often deployments occur:
- Target: As needed (typically weekly)
- Measure: Number of deployments per week

### Change Failure Rate

Track how many deployments cause issues:
- Target: < 15%
- Measure: Failed deployments / Total deployments

## Additional Deployment Options

### Kubernetes Deployment

```bash
# 1. Create namespace
kubectl create namespace victor-prod

# 2. Create secrets
kubectl create secret generic victor-secrets \
  --from-literal=database-url="postgresql://..." \
  --from-literal=anthropic-api-key="sk-ant-..." \
  -n victor-prod

# 3. Deploy
kubectl apply -k deployment/kubernetes/overlays/production

# 4. Verify
kubectl get pods -n victor-prod
```

### Docker Deployment

```bash
# Using Docker Compose
cd deployment/docker
docker-compose -f docker-compose.prod.yml up -d

# Using Docker directly
docker run -d \
  --name victor \
  -p 8000:8000 \
  -e ANTHROPIC_API_KEY="sk-ant-..." \
  victorai/victor:0.5.0
```

### Helm Deployment

```bash
# Install Helm chart
helm install victor ./deployment/helm \
  -f deployment/helm/values-prod.yaml \
  --namespace victor-prod \
  --create-namespace

# Verify
kubectl get pods -n victor-prod
```

### Terraform Deployment (AWS)

```bash
cd deployment/terraform

# Initialize
terraform init

# Plan
terraform plan -var="environment=production"

# Apply
terraform apply -var="environment=production"
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.5.0 | 2024-01-21 | Initial production deployment automation |

---

**Maintained by**: DevOps Team
**Last Updated**: 2024-01-21
**Status**: Production Ready

## License

Apache License 2.0 - see [LICENSE](../LICENSE) for details.
