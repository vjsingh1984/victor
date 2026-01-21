# Victor AI Production Deployment Runbook

This document provides comprehensive guidance for deploying Victor AI to production environments.

## Table of Contents

1. [Deployment Overview](#deployment-overview)
2. [Pre-Deployment Checklist](#pre-deployment-checklist)
3. [Deployment Methods](#deployment-methods)
4. [Health Checks](#health-checks)
5. [Rollback Procedures](#rollback-procedures)
6. [Troubleshooting](#troubleshooting)
7. [Monitoring](#monitoring)

---

## Deployment Overview

Victor AI supports multiple deployment methods:

| Method | Best For | Complexity | Scalability |
|--------|----------|------------|-------------|
| **pip install** | Development, testing | Low | Manual |
| **Docker** | Single-server deployment | Medium | Medium |
| **Kubernetes** | Production, multi-cloud | High | High |
| **Ansible** | Automated server deployment | Medium | High |

---

## Pre-Deployment Checklist

### System Requirements

- **Python**: 3.9 or higher
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Disk**: 10GB free space
- **Network**: Outbound HTTPS access

### Environment Variables

```bash
# Required for production
export VICTOR_PROFILE=production
export VICTOR_LOG_LEVEL=INFO

# Optional
export VICTOR_API_KEY=your_api_key
export VICTOR_MAX_WORKERS=4
export VICTOR_CACHE_TTL=3600
```

### Dependency Checks

```bash
# Check Python version
python --version

# Check Docker (if using Docker)
docker --version

# Check kubectl (if using Kubernetes)
kubectl version --client
```

---

## Deployment Methods

### Method 1: pip Installation

**Best for**: Development, testing, single-server production

```bash
# Install dependencies
pip install -e ".[api]"

# Run application
uvicorn victor.api.server:app --host 0.0.0.0 --port 8000
```

**Pros**:
- Simple, fast setup
- Easy to update
- Low resource overhead

**Cons**:
- Manual process management
- No automatic restarts
- Limited scalability

### Method 2: Docker Deployment

**Best for**: Production servers, container orchestration

```bash
# Build image
cd scripts/docker
./build.sh 1.0.0 --push --scan

# Run container
docker run -d \
  --name victor \
  -p 8000:8000 \
  --env-file .env \
  victorai/victor:1.0.0
```

**Docker Compose Example**:

```yaml
version: '3.8'
services:
  victor:
    image: victorai/victor:latest
    container_name: victor
    ports:
      - "8000:8000"
    environment:
      - VICTOR_PROFILE=production
      - VICTOR_LOG_LEVEL=INFO
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

**Pros**:
- Containerized environment
- Easy scaling
- Consistent deployments

**Cons**:
- Requires Docker knowledge
- Additional overhead

### Method 3: Kubernetes Deployment

**Best for**: Large-scale production, multi-cloud, high availability

```bash
# Deploy to Kubernetes
cd scripts/kubernetes
./deploy.sh production --namespace victor-prod

# Check deployment status
kubectl get pods -n victor-prod
kubectl logs -f deployment/victor-prod -n victor-prod
```

**Helm Chart Installation**:

```bash
# Add Helm repository
helm repo add victor https://charts.victorai.com

# Install
helm install victor-prod victor/victor \
  --namespace production \
  --values values-production.yaml

# Upgrade
helm upgrade victor-prod victor/victor \
  --namespace production \
  --set image.tag=new_version
```

**Pros**:
- Auto-scaling
- Self-healing
- Rolling updates
- Zero-downtime deployments

**Cons**:
- High complexity
- Requires Kubernetes cluster

### Method 4: Ansible Deployment

**Best for**: Automated multi-server deployments

```bash
# Run Ansible playbook
cd deploy/ansible
ansible-playbook -i inventory playbook.yml --extra-vars "env=production"
```

**Inventory File Example**:

```ini
[production]
server1 ansible_host=192.168.1.10 ansible_user=deploy
server2 ansible_host=192.168.1.11 ansible_user=deploy

[production:vars]
ansible_python_interpreter=/usr/bin/python3
deployment_env=production
```

**Pros**:
- Automated server setup
- Configuration management
- Idempotent operations

**Cons**:
- Requires Ansible knowledge
- Server management overhead

---

## Health Checks

### Automated Health Checks

```bash
# Run health check script
./scripts/health_check.sh --endpoint http://localhost:8000/health

# With custom timeout
./scripts/health_check.sh --endpoint http://localhost:8000/health --timeout 30

# Verbose mode
./scripts/health_check.sh --endpoint http://localhost:8000/health --verbose
```

### Health Check Endpoints

| Endpoint | Purpose | Response |
|----------|---------|----------|
| `GET /health` | Basic health | `{"status": "healthy"}` |
| `GET /health/db` | Database check | `{"status": "healthy", "connection": "ok"}` |
| `GET /health/providers` | Provider status | `{"providers": ["anthropic", "openai"]}` |
| `GET /health/resources` | Resource usage | `{"memory_percent": 45, "cpu_percent": 30}` |
| `GET /health/critical` | Critical services | `{"services": {"api": "healthy", "db": "healthy"}}` |

### Critical Health Checks

1. **API Availability**: Service responds on port 8000
2. **Database Connectivity**: SQLite/PostgreSQL accessible
3. **Provider Availability**: At least one LLM provider ready
4. **Memory Usage**: Below 90% capacity
5. **CPU Usage**: Below 90% capacity

---

## Rollback Procedures

### Automatic Rollback

Rollback is automatically triggered on:
- Health check failures
- Smoke test failures
- Critical error detection

### Manual Rollback

**Using Rollback Script**:

```bash
# Interactive rollback to latest backup
./scripts/rollback.sh

# Rollback to specific version
./scripts/rollback.sh --version backup_20240120_153000

# Forced rollback (skip confirmation)
./scripts/rollback.sh --force

# Keep current database
./scripts/rollback.sh --keep-db

# Dry run (simulate)
./scripts/rollback.sh --dry-run
```

**Kubernetes Rollback**:

```bash
# Helm rollback
helm rollback victor-prod -n production

# Rollback to specific revision
helm rollback victor-prod 5 -n production

# Check rollback history
helm history victor-prod -n production
```

**Docker Rollback**:

```bash
# Stop current container
docker stop victor
docker rm victor

# Start previous version
docker run -d \
  --name victor \
  -p 8000:8000 \
  --env-file .env \
  victorai/victor:previous_version
```

### Rollback Verification

After rollback:

```bash
# Check health
./scripts/health_check.sh

# Verify version
curl http://localhost:8000/health

# Check logs
tail -f logs/deployment.log
```

---

## Troubleshooting

### Common Issues

#### 1. Service Won't Start

**Symptoms**: Port already in use, process exits immediately

**Solutions**:
```bash
# Check if port is in use
lsof -i :8000
netstat -tuln | grep 8000

# Kill existing process
pkill -f "victor.api.server"

# Check logs
tail -f logs/victor.log
```

#### 2. High Memory Usage

**Symptoms**: OOM errors, slow performance

**Solutions**:
```bash
# Check memory usage
docker stats victor
kubectl top pod -n production

# Increase memory limit
docker run -m 4g victorai/victor

# Configure in Kubernetes
resources:
  requests:
    memory: "2Gi"
  limits:
    memory: "4Gi"
```

#### 3. Database Connection Errors

**Symptoms**: Can't connect to database, migration failures

**Solutions**:
```bash
# Check database file
ls -la victor.db

# Run migrations
python -c "from victor.core.migrations import run_migrations; run_migrations()"

# Check database health
curl http://localhost:8000/health/db
```

#### 4. Provider API Failures

**Symptoms**: API errors, timeouts

**Solutions**:
```bash
# Check API keys
cat .env | grep API_KEY

# Test provider connectivity
python -c "from victor.providers.registry import ProviderRegistry; print(ProviderRegistry.list_providers())"

# Check provider health
curl http://localhost:8000/health/providers
```

### Log Locations

| Deployment | Log Location |
|------------|--------------|
| pip | `logs/victor.log` |
| Docker | `docker logs victor` |
| Kubernetes | `kubectl logs -f deployment/victor-prod` |

### Debug Mode

```bash
# Enable debug logging
export VICTOR_LOG_LEVEL=DEBUG

# Run with debug output
victor chat --debug

# Check configuration
victor config show
```

---

## Monitoring

### Key Metrics

1. **System Metrics**:
   - CPU usage
   - Memory usage
   - Disk I/O
   - Network traffic

2. **Application Metrics**:
   - Request rate
   - Response time
   - Error rate
   - Provider availability

3. **Business Metrics**:
   - Active users
   - Tools used
   - Task completion rate

### Monitoring Tools

**Prometheus + Grafana**:

```yaml
# Prometheus scraping config
scrape_configs:
  - job_name: 'victor'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

**Health Check Dashboard**:

```bash
# Continuous monitoring
watch -n 10 './scripts/health_check.sh'

# Monitor Kubernetes pods
watch -n 5 'kubectl get pods -n production'
```

### Alerting

**Critical Alerts**:
- Service down (5+ minutes)
- Error rate > 5%
- Response time > 5s
- Memory usage > 90%
- Database connection failures

**Warning Alerts**:
- High CPU usage (> 80%)
- Slow queries (> 2s)
- Provider API errors

---

## Deployment Best Practices

1. **Always test in staging first**
2. **Use version tags for production releases**
3. **Enable health checks and monitoring**
4. **Keep backups before deployments**
5. **Use canary deployments for major versions**
6. **Monitor rollback status**
7. **Document all changes**
8. **Run smoke tests after deployment**

---

## Support

- **Documentation**: https://docs.victorai.com
- **Issues**: https://github.com/victorai/victor/issues
- **Discussions**: https://github.com/victorai/victor/discussions
- **Email**: support@victorai.com

---

**Last Updated**: 2025-01-20
**Version**: 0.5.1
