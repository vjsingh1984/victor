# Victor AI - Production Infrastructure Guide

Complete guide for deploying, scaling, and monitoring Victor AI in production environments.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Deployment Options](#deployment-options)
  - [Docker](#docker-deployment)
  - [Docker Compose](#docker-compose-deployment)
  - [Kubernetes](#kubernetes-deployment)
- [CI/CD Pipeline](#cicd-pipeline)
- [Monitoring & Observability](#monitoring--observability)
- [Scaling Guidelines](#scaling-guidelines)
- [Security Best Practices](#security-best-practices)
- [Backup & Disaster Recovery](#backup--disaster-recovery)
- [Troubleshooting](#troubleshooting)

## Overview

Victor AI supports multiple deployment strategies optimized for different use cases:

| Deployment | Best For | Scalability | Complexity |
|------------|----------|-------------|------------|
| **Docker** | Local development, testing | Low | Simple |
| **Docker Compose** | Small production, single server | Medium | Simple |
| **Kubernetes** | Enterprise, multi-region | High | Complex |

### Key Features

- **Multi-stage Docker builds** for optimized image size (< 1GB)
- **Health checks** and readiness probes
- **Horizontal Pod Autoscaling** for automatic scaling
- **Rolling updates** and zero-downtime deployments
- **Comprehensive monitoring** with Prometheus and Grafana
- **Security scanning** in CI/CD pipeline
- **Performance regression detection** with automated benchmarks

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Load Balancer / Ingress                  │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
┌───────▼──────┐ ┌─────▼──────┐ ┌──────▼───────┐
│ Victor Pod 1 │ │ Victor Pod 2│ │ Victor Pod 3 │
└───────┬──────┘ └─────┬──────┘ └──────┬───────┘
        │               │               │
        └───────────────┼───────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
┌───────▼──────┐ ┌─────▼──────┐ ┌──────▼───────┐
│   Redis      │ │ PostgreSQL  │ │  LanceDB     │
│   (Cache)    │ │ (Database)  │ │  (Vector)    │
└──────────────┘ └────────────┘ └──────────────┘
```

## Deployment Options

### Docker Deployment

#### Build Production Image

```bash
# Build with metadata
docker build \
  -f Dockerfile.production \
  --build-arg VERSION=$(cat pyproject.toml | grep version | head -1 | awk -F= '{print $2}' | tr -d ' "') \
  --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
  --build-arg VCS_REF=$(git rev-parse --short HEAD) \
  -t victor:production .

# Build for specific platform
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f Dockerfile.production \
  -t victor:latest .
```

#### Run Container

```bash
# Interactive CLI mode
docker run -it --rm \
  -v $(pwd)/workspace:/workspace \
  -e ANTHROPIC_API_KEY=xxx \
  victor:production

# API server mode
docker run -d --name victor-api \
  -p 8000:8000 \
  -e ANTHROPIC_API_KEY=xxx \
  -e VICTOR_PROFILE=production \
  victor:production victor serve --port 8000

# Check health
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready
```

### Docker Compose Deployment

#### Prerequisites

```bash
# Create .env file
cat > .env << EOF
# API Keys
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here

# Configuration
VICTOR_PROFILE=production
LOG_LEVEL=INFO
WORKSPACE_DIR=./workspace
EOF
```

#### Deploy Full Stack

```bash
# Start all services
docker-compose -f docker-compose.production.yml up -d

# View logs
docker-compose -f docker-compose.production.yml logs -f victor-api

# Check status
docker-compose -f docker-compose.production.yml ps

# Scale API servers
docker-compose -f docker-compose.production.yml up -d --scale victor-api=3
```

#### Access Services

- **Victor API**: http://localhost:8000
- **Prometheus**: http://localhost:9091
- **Grafana**: http://localhost:3000 (admin/admin)
- **Jaeger**: http://localhost:16686

#### Stop Services

```bash
# Stop all services
docker-compose -f docker-compose.production.yml down

# Stop and remove volumes
docker-compose -f docker-compose.production.yml down -v
```

### Kubernetes Deployment

#### Prerequisites

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/darwin/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/

# Install Helm (optional)
brew install helm

# Configure kubeconfig
export KUBECONFIG=~/.kube/config
```

#### Deploy with Kubectl

```bash
# Create namespace
kubectl create namespace victor

# Create secrets
kubectl create secret generic victor-secrets \
  --from-literal=anthropic-api-key=xxx \
  --from-literal=openai-api-key=xxx \
  -n victor

# Apply manifests
kubectl apply -f config/k8s/namespace.yaml
kubectl apply -f config/k8s/configmap.yaml
kubectl apply -f config/k8s/secret.yaml
kubectl apply -f config/k8s/serviceaccount.yaml
kubectl apply -f config/k8s/pvc.yaml
kubectl apply -f config/k8s/deployment.yaml
kubectl apply -f config/k8s/service.yaml
kubectl apply -f config/k8s/ingress.yaml
kubectl apply -f config/k8s/hpa.yaml
kubectl apply -f config/k8s/pdb.yaml

# Wait for rollout
kubectl rollout status deployment/victor-api -n victor --timeout=5m
```

#### Deploy with Helm

```bash
# Add Helm repository (if applicable)
# helm repo add victor https://charts.victor.ai

# Install/Upgrade
helm upgrade --install victor ./config/helm/victor \
  --namespace victor \
  --create-namespace \
  --set image.repository=ghcr.io/vjsingh1984/victor \
  --set image.tag=0.5.0 \
  --set replicaCount=3 \
  --set autoscaling.enabled=true \
  --set ingress.enabled=true \
  --set ingress.hosts[0].host=victor.example.com

# Check deployment
helm status victor -n victor

# Get values
helm get values victor -n victor
```

#### Upgrade Deployment

```bash
# Using kubectl
kubectl set image deployment/victor-api \
  victor-api=ghcr.io/vjsingh1984/victor:0.5.1 \
  -n victor

# Using Helm
helm upgrade victor ./config/helm/victor \
  --namespace victor \
  --set image.tag=0.5.1
```

#### Rollback

```bash
# Using kubectl
kubectl rollout undo deployment/victor-api -n victor

# View rollout history
kubectl rollout history deployment/victor-api -n victor

# Rollback to specific revision
kubectl rollout undo deployment/victor-api --to-revision=2 -n victor

# Using Helm
helm rollback victor -n victor
```

## CI/CD Pipeline

### Workflow Triggers

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| **infrastructure-ci.yml** | Push/PR to main/develop | Validate infrastructure changes |
| **cd-production.yml** | Push to main | Deploy to staging/production |
| **test-performance.yml** | Schedule/PR | Performance regression detection |
| **release.yml** | Tag push | Build and publish releases |

### Infrastructure CI Pipeline

```yaml
# .github/workflows/infrastructure-ci.yml

Jobs:
  1. Validate Dockerfiles
     - Check syntax
     - Hadolint linting
     - Multi-stage build verification

  2. Build Docker Images
     - Build all Dockerfiles
     - Test functionality
     - Check image size

  3. Validate K8s Manifests
     - kubectl dry-run
     - Resource limits check
     - Health probes verification

  4. Validate Helm Chart
     - Helm lint
     - Template rendering
     - Dry-run install

  5. Security Scanning
     - Trivy vulnerability scan
     - Docker Bench Security
```

### CD Pipeline

```yaml
# .github/workflows/cd-production.yml

Stages:
  1. Get Version
     - From pyproject.toml or input

  2. Run Tests
     - Unit tests (70% coverage required)
     - Smoke tests

  3. Build Image
     - Multi-platform build (amd64, arm64)
     - Push to GHCR

  4. Security Scan
     - Trivy scan
     - Upload results to GitHub Security

  5. Deploy to Staging
     - kubectl set image
     - Wait for rollout
     - Run smoke tests

  6. Deploy to Production
     - Manual approval required
     - Create backup
     - Deploy with monitoring
     - Rollback on failure
```

### Manual Deployment

```bash
# Using deployment script
./scripts/ci/deploy_production.sh staging --dry-run
./scripts/ci/deploy_production.sh staging
./scripts/ci/deploy_production.sh production --version v0.5.0

# Using kubectl directly
kubectl set image deployment/victor-api \
  victor-api=victor:0.5.0 \
  -n victor

# Using Helm directly
helm upgrade victor ./config/helm/victor \
  --namespace victor \
  --set image.tag=0.5.0
```

## Monitoring & Observability

### Prometheus Metrics

Victor exposes metrics on port `9090`:

```bash
# Access metrics endpoint
curl http://localhost:9090/metrics

# Key metrics
# - victor_api_requests_total
# - victor_api_duration_seconds
# - victor_tool_executions_total
# - victor_llm_requests_total
# - victor_cache_hit_ratio
```

### Grafana Dashboards

Import dashboards from `config/grafana/dashboards/`:

- **API Performance**: Request rate, latency, error rate
- **Tool Execution**: Tool usage, execution time
- **LLM Performance**: Token usage, cost tracking
- **System Resources**: CPU, memory, disk I/O

### Distributed Tracing

```bash
# Access Jaeger UI
open http://localhost:16686

# View traces
# - API requests
# - Tool executions
# - LLM calls
# - Workflow compilations
```

### Logging

```bash
# View logs
kubectl logs -f deployment/victor-api -n victor

# View logs for specific pod
kubectl logs -f victor-api-xxx -n victor

# View logs with label selector
kubectl logs -f -l app=victor-api -n victor

# Stream all logs
kubectl logs -f --all-containers=true -l app=victor-api -n victor
```

### Alerts

Configure alerts in `config/prometheus/alerts.yml`:

```yaml
# Example alerts
- alert: HighErrorRate
  expr: rate(victor_api_errors_total[5m]) > 0.05
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "High error rate detected"

- alert: HighMemoryUsage
  expr: container_memory_usage_bytes > 2GB
  for: 10m
  labels:
    severity: warning
```

## Scaling Guidelines

### Horizontal Pod Autoscaler

Victor uses HPA for automatic scaling:

```yaml
# config/k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: victor-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: victor-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Scaling Strategy

| Metric | Target | Scale Up | Scale Down |
|--------|--------|----------|------------|
| CPU | 70% | > 70% for 5m | < 30% for 10m |
| Memory | 80% | > 80% for 5m | < 40% for 10m |
| Requests | 1000 req/min/pod | > 1000 | < 200 |

### Manual Scaling

```bash
# Scale to 5 replicas
kubectl scale deployment victor-api --replicas=5 -n victor

# Check replica count
kubectl get deployment victor-api -n victor

# Check HPA status
kubectl get hpa -n victor
```

### Performance Optimization

#### Resource Limits

```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "2000m"
```

#### Cache Configuration

```yaml
# Redis cache
- name: REDIS_URL
  value: "redis://redis:6379/0"
- name: VICTOR_CACHE_TTL
  value: "3600"  # 1 hour
```

#### Tool Selection Caching

```yaml
# Enable tool selection cache
- name: VICTOR_ENABLE_TOOL_CACHE
  value: "true"
- name: VICTOR_TOOL_CACHE_SIZE
  value: "1000"
```

## Security Best Practices

### Container Security

1. **Non-root user**: Containers run as user `victor` (UID 1000)
2. **Read-only root**: Enable where possible
3. **Drop capabilities**: All capabilities dropped except required
4. **Security contexts**: Enforce in pod specification
5. **Network policies**: Restrict pod-to-pod communication

### Image Security

1. **Multi-stage builds**: Minimal attack surface
2. **Vulnerability scanning**: Trivy in CI/CD
3. **Base image updates**: Regular security patches
4. **Minimal dependencies**: Only required packages

### Secrets Management

```bash
# Never commit secrets to git
# Use Kubernetes secrets
kubectl create secret generic victor-secrets \
  --from-literal=api-key=xxx \
  -n victor

# Use External Secrets Operator for cloud providers
# AWS Secrets Manager
# Azure Key Vault
# Google Secret Manager
```

### Network Security

```yaml
# Network policy example
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: victor-network-policy
spec:
  podSelector:
    matchLabels:
      app: victor-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
```

## Backup & Disaster Recovery

### Backup Strategy

#### Automated Backups

```bash
# Backup Kubernetes resources
kubectl get all -n victor -o yaml > victor-backup-$(date +%Y%m%d).yaml

# Backup secrets
kubectl get secrets -n victor -o yaml > victor-secrets-$(date +%Y%m%d).yaml

# Backup PVCs
kubectl get pvc -n victor
# Use Velero for volume backups
velero backup create victor-backup --namespace victor
```

#### Database Backups

```bash
# PostgreSQL backup
kubectl exec -it victor-postgres-0 -n victor -- \
  pg_dump -U victor victor > victor-db-$(date +%Y%m%d).sql

# Restore
kubectl exec -i victor-postgres-0 -n victor -- \
  psql -U victor victor < victor-db-20250120.sql
```

### Disaster Recovery

#### Recovery Steps

1. **Restore from backup**
   ```bash
   kubectl apply -f victor-backup-20250120.yaml
   ```

2. **Verify deployment**
   ```bash
   kubectl get pods -n victor
   kubectl rollout status deployment/victor-api -n victor
   ```

3. **Run smoke tests**
   ```bash
   ./scripts/ci/smoke_test.sh production
   ```

4. **Monitor metrics**
   - Check error rates
   - Verify latency
   - Monitor resource usage

#### Rollback Procedure

```bash
# Quick rollback
kubectl rollout undo deployment/victor-api -n victor

# Rollback to specific version
kubectl rollout undo deployment/victor-api --to-revision=2 -n victor

# Using Helm
helm rollback victor -n victor

# Verify rollback
kubectl rollout status deployment/victor-api -n victor
```

## Troubleshooting

### Common Issues

#### 1. Pod Not Starting

```bash
# Check pod status
kubectl describe pod victor-api-xxx -n victor

# Common causes:
# - Image pull error
# - Resource limits too low
# - Missing secrets/configmaps
# - Failed health checks
```

#### 2. High Memory Usage

```bash
# Check memory usage
kubectl top pods -n victor

# Increase memory limits
kubectl set resources deployment victor-api \
  --limits=memory=4Gi \
  -n victor

# Check for memory leaks
kubectl logs victor-api-xxx -n victor | grep -i memory
```

#### 3. Slow Response Times

```bash
# Check metrics
curl http://localhost:9090/metrics | grep victor_api_duration

# Check HPA status
kubectl get hpa -n victor

# Scale up manually
kubectl scale deployment victor-api --replicas=5 -n victor

# Check database performance
kubectl exec -it victor-postgres-0 -n victor -- psql -U victor -c "SELECT * FROM pg_stat_activity;"
```

#### 4. High Error Rates

```bash
# Check logs
kubectl logs -f deployment/victor-api -n victor

# Check recent errors
kubectl logs --since=5m deployment/victor-api -n victor | grep -i error

# Check API provider status
curl https://status.anthropic.com
curl https://status.openai.com

# Verify API keys
kubectl get secret victor-secrets -n victor -o yaml
```

### Debug Mode

```yaml
# Enable debug logging
env:
  - name: VICTOR_LOG_LEVEL
    value: "DEBUG"
  - name: VICTOR_DEBUG
    value: "true"
```

### Health Check Endpoints

```bash
# Liveness probe
curl http://localhost:8000/health/live

# Readiness probe
curl http://localhost:8000/health/ready

# Startup probe
curl http://localhost:8000/health/startup

# Detailed health
curl http://localhost:8000/health
```

## Performance Tuning

### Tool Selection Caching

Enable caching for faster tool selection (24-37% latency reduction):

```yaml
env:
  - name: VICTOR_ENABLE_TOOL_CACHE
    value: "true"
  - name: VICTOR_TOOL_CACHE_SIZE
    value: "1000"
  - name: VICTOR_TOOL_CACHE_TTL
    value: "3600"  # 1 hour
```

### Parallel Tool Execution

```yaml
env:
  - name: VICTOR_MAX_CONCURRENT_TOOLS
    value: "10"
  - name: VICTOR_ENABLE_PARALLEL_EXECUTION
    value: "true"
```

### Embedding Model Caching

Pre-download embedding model during build:

```dockerfile
# In Dockerfile
RUN python3 -c "from sentence_transformers import SentenceTransformer; \
    model = SentenceTransformer('BAAI/bge-small-en-v1.5'); \
    print('Model cached')"
```

## Maintenance

### Regular Updates

1. **Weekly**
   - Check security vulnerabilities
   - Review error logs
   - Monitor performance metrics

2. **Monthly**
   - Update base images
   - Review and update dependencies
   - Clean up old backups

3. **Quarterly**
   - Architecture review
   - Cost optimization
   - Capacity planning

### Log Rotation

```yaml
# Configure log retention
env:
  - name: VICTOR_LOG_RETENTION_DAYS
    value: "30"
  - name: VICTOR_LOG_MAX_SIZE
    value: "100M"
```

### Resource Cleanup

```bash
# Clean up old images
docker image prune -a --filter "until=72h"

# Clean up unused resources
kubectl delete pod -n victor --field-selector=status.phase=Succeeded

# Clean up old backups
find ./backups -type f -mtime +30 -delete
```

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Helm Documentation](https://helm.sh/docs/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)

## Support

For issues and questions:
- GitHub Issues: https://github.com/vijayksingh/victor/issues
- Documentation: https://github.com/vijayksingh/victor/tree/main/docs
- Discussions: https://github.com/vijayksingh/victor/discussions
