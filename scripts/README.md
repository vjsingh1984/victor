# Victor AI Deployment Scripts

This directory contains comprehensive deployment automation scripts for Victor AI production environments.

## Available Scripts

### Core Deployment Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `deploy.sh` | Main deployment script | `./deploy.sh [staging\|production]` |
| `rollback.sh` | Rollback to previous version | `./rollback.sh [--version VERSION]` |
| `health_check.sh` | Health check monitoring | `./health_check.sh [--endpoint URL]` |

### Platform-Specific Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `docker/build.sh` | Build and push Docker images | `./build.sh [version] [--push] [--scan]` |
| `kubernetes/deploy.sh` | Deploy to Kubernetes | `./deploy.sh [staging\|production]` |

## Quick Start

### 1. Deploy to Staging (pip)

```bash
cd scripts
./deploy.sh staging
```

### 2. Deploy to Production (Docker)

```bash
# Build Docker image
cd scripts/docker
./build.sh 1.0.0 --push --scan

# Deploy
cd ..
./deploy.sh production
```

### 3. Deploy to Kubernetes

```bash
cd scripts/kubernetes
./deploy.sh production --namespace victor-prod
```

## Script Details

### deploy.sh

**Main deployment script for pip-based installations**

**Features**:
- Environment validation (Python 3.9+ required)
- Configuration validation (checks required env vars)
- Automatic backup of current deployment
- Dependency installation
- Database migration
- Service startup (systemd, screen, or tmux)
- Health checks (5 critical checks)
- Smoke tests
- Automatic rollback on failure

**Usage**:

```bash
# Deploy to staging
./deploy.sh staging

# Deploy to production
./deploy.sh production

# Rollback on failure
./deploy.sh production --rollback
```

**Environment Variables**:

| Variable | Description | Default |
|----------|-------------|---------|
| `VICTOR_PROFILE` | Environment profile | `production` |
| `VICTOR_LOG_LEVEL` | Logging level | `INFO` |
| `VICTOR_MAX_WORKERS` | Number of workers | `4` |

**Exit Codes**:
- `0`: Success
- `1`: Deployment failed

**Backup Strategy**:
- Creates backup before deployment
- Stores last 5 backups
- Includes: venv, .env, database, git state

**Health Checks**:
1. API endpoint response (200 OK)
2. Database connectivity
3. Provider availability
4. Memory usage (< 90%)
5. CPU usage (< 90%)

---

### rollback.sh

**Rollback to previous deployment**

**Features**:
- Interactive backup selection
- Graceful service shutdown
- Virtual environment restoration
- Configuration restoration
- Database rollback (optional)
- Git state restoration
- Health check verification

**Usage**:

```bash
# Interactive rollback
./rollback.sh

# Rollback to specific version
./rollback.sh --version backup_20240120_153000

# Forced rollback (skip confirmation)
./rollback.sh --force

# Keep current database
./rollback.sh --keep-db

# Dry run (simulate)
./rollback.sh --dry-run
```

**Options**:

| Option | Description |
|--------|-------------|
| `--version VERSION` | Rollback to specific backup |
| `--force` | Skip confirmation prompt |
| `--keep-db` | Don't rollback database |
| `--dry-run` | Simulate rollback |
| `--help` | Show help message |

**What Gets Restored**:
- Virtual environment (Python packages)
- Configuration (.env file)
- Database (victor.db)
- Git state (commit SHA)

**Backup Location**:
```bash
/backups/
  ├── backup_20240120_153000/
  │   ├── venv/
  │   ├── .env
  │   ├── victor.db
  │   └── git_commit.txt
  └── backup_20240120_140000/
```

---

### health_check.sh

**Health monitoring script**

**Features**:
- HTTP endpoint checks
- Service availability verification
- Database connectivity
- Provider status
- Resource usage (memory, CPU)
- Critical services check
- Summary report

**Usage**:

```bash
# Check localhost
./health_check.sh

# Check specific endpoint
./health_check.sh --endpoint http://localhost:8000/health

# Custom timeout
./health_check.sh --timeout 30

# Verbose output
./health_check.sh --verbose

# Continue on failure
./health_check.sh --no-exit-on-failure
```

**Health Check Endpoints**:

| Endpoint | Check | Description |
|----------|-------|-------------|
| `/health` | HTTP | API availability |
| `/health/db` | Database | Connectivity status |
| `/health/providers` | Providers | LLM provider status |
| `/health/resources` | Resources | Memory/CPU usage |
| `/health/critical` | Critical | All critical services |

**Exit Codes**:
- `0`: All checks passed
- `1`: One or more checks failed

**Output Example**:

```
[INFO] Starting health checks for Victor AI...
[INFO] Endpoint: http://localhost:8000/health
[INFO] Timeout: 10s

[INFO] Checking service availability...
[✓] Service is listening on localhost:8000

[INFO] Checking HTTP endpoint: http://localhost:8000/health
[✓] HTTP endpoint is healthy (200 OK)
[✓] Application status: healthy

[INFO] Checking database connectivity...
[✓] Database connectivity: OK

[INFO] Checking provider availability...
[✓] Available providers: 21

[INFO] Checking system resources...
[✓] Memory usage: 45%
[✓] CPU usage: 30%

==========================================
Health Check Summary
==========================================
Total Checks:   6
Passed:         6
Failed:         0
==========================================

[✓] All health checks passed!
```

---

### docker/build.sh

**Docker image build and push script**

**Features**:
- Multi-stage builds
- Multi-architecture support (amd64, arm64)
- Tagging strategy (version, git SHA, branch, latest)
- Registry push
- Vulnerability scanning (Trivy)
- Build caching

**Usage**:

```bash
# Build with version tag
./build.sh 1.0.0

# Build and push to registry
./build.sh 1.0.0 --push

# Build, push, and scan
./build.sh 1.0.0 --push --scan

# Use git tag as version
./build.sh $(git describe --tags)

# Custom registry
./build.sh latest --registry ghcr.io/victorai --push
```

**Tags Created**:

```bash
# For version 1.0.0
victorai/victor:1.0.0          # Version tag
victorai/victor:abc1234        # Git SHA
victorai/victor:main           # Branch name
victorai/victor:latest         # Latest (if production)
```

**Build Arguments**:

| Argument | Description | Example |
|----------|-------------|---------|
| `VERSION` | Application version | `1.0.0` |
| `GIT_SHA` | Git commit SHA | `abc1234` |
| `GIT_BRANCH` | Git branch name | `main` |
| `BUILD_DATE` | Build timestamp | `2024-01-20T15:30:00Z` |

**Image Information**:

```bash
# View image details
docker images victorai/victor:1.0.0

# View image layers
docker history victorai/victor:1.0.0

# Inspect image
docker inspect victorai/victor:1.0.0
```

**Vulnerability Scanning**:

```bash
# Requires Trivy
brew install trivy  # macOS
apt install trivy  # Ubuntu

# Scan image
trivy image victorai/victor:1.0.0

# With script
./build.sh 1.0.0 --scan
```

---

### kubernetes/deploy.sh

**Kubernetes deployment script**

**Features**:
- Helm chart installation
- ConfigMap/Secret management
- Rolling updates
- Pod verification
- Health checks

**Usage**:

```bash
# Deploy to staging
./deploy.sh staging --install

# Deploy to production
./deploy.sh production --upgrade

# Dry run
./deploy.sh production --dry-run

# Custom namespace
./deploy.sh production --namespace victor-prod

# Custom values file
./deploy.sh production --values custom-values.yaml
```

**Prerequisites**:

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/darwin/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/

# Install Helm
brew install helm  # macOS
# or
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Configure kubeconfig
mkdir -p ~/.kube
cp /path/to/kubeconfig ~/.kube/config
```

**Deployment Flow**:

1. Validate Kubernetes environment
2. Create namespace (if needed)
3. Create/update secrets
4. Deploy with Helm
5. Wait for rollout
6. Run health checks
7. Verify deployment

**Useful Commands**:

```bash
# Check deployment status
kubectl get pods -n victor-prod
kubectl get deployment -n victor-prod
kubectl get service -n victor-prod

# View logs
kubectl logs -f deployment/victor-prod -n victor-prod

# Port forward
kubectl port-forward service/victor-prod 8000:8000 -n victor-prod

# Scale deployment
kubectl scale deployment/victor-prod --replicas=3 -n victor-prod

# Check rollout status
kubectl rollout status deployment/victor-prod -n victor-prod
```

---

## Testing the Scripts

### Test Deploy Script

```bash
# Test in development environment
cd scripts
./deploy.sh staging

# Check logs
tail -f logs/deploy_*.log

# Verify deployment
./health_check.sh
```

### Test Docker Build

```bash
# Test build locally
cd scripts/docker
./build.sh test --scan

# Run container
docker run -d -p 8000:8000 victorai/victor:test

# Test health endpoint
curl http://localhost:8000/health
```

### Test Kubernetes Deployment

```bash
# Dry run first
cd scripts/kubernetes
./deploy.sh staging --dry-run

# Deploy to staging
./deploy.sh staging --install

# Verify
kubectl get pods -n staging
./health_check.sh --endpoint http://staging.victorai.com/health
```

### Test Rollback

```bash
# Test rollback with dry run
./rollback.sh --dry-run

# List available backups
ls -la ../backups/

# Rollback to latest
./rollback.sh --force --keep-db

# Verify
./health_check.sh
```

---

## CI/CD Integration

### GitHub Actions

See `.github/workflows/deploy-production.yml` for complete CI/CD pipeline.

**Workflow Stages**:

1. **Test**: Run tests (92%+ pass rate required)
2. **Security Scan**: Run Bandit, Safety, Trivy
3. **Build**: Build Docker image
4. **Deploy Staging**: Deploy to staging, run smoke tests
5. **Deploy Production**: Canary deployment, full rollout

**Manual Deployment**:

```yaml
on:
  workflow_dispatch:
    inputs:
      environment:
        type: choice
        options: [staging, production]
```

### Ansible Integration

```bash
# Deploy with Ansible
cd deploy/ansible
ansible-playbook -i inventory playbook.yml --extra-vars "env=production"

# Check deployment
ansible all -m shell -a "systemctl status victor-api"
```

---

## Troubleshooting

### Script Execution Issues

**Permission Denied**:

```bash
chmod +x scripts/*.sh
chmod +x scripts/docker/*.sh
chmod +x scripts/kubernetes/*.sh
```

**Python Not Found**:

```bash
# Install Python 3.11
brew install python@3.11  # macOS
apt install python3.11   # Ubuntu

# Set as default
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
```

### Docker Issues

**Docker Daemon Not Running**:

```bash
# Start Docker
sudo systemctl start docker  # Linux
open -a Docker               # macOS

# Check status
docker info
```

**Permission Denied**:

```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Reload group
newgrp docker
```

### Kubernetes Issues

**Connection Refused**:

```bash
# Check kubeconfig
kubectl config current-context
kubectl config view

# Test connection
kubectl cluster-info

# Switch context
kubectl config use-context production
```

**Helm Chart Not Found**:

```bash
# Add Helm repository
helm repo add victor https://charts.victorai.com
helm repo update

# List charts
helm search repo victor
```

---

## Best Practices

1. **Always Test in Staging First**
   ```bash
   ./deploy.sh staging
   ./health_check.sh --endpoint http://staging.victorai.com/health
   ```

2. **Create Backups Before Deployment**
   - Backups are automatic with deploy.sh
   - Keep at least 5 backups
   - Test restore procedure

3. **Monitor Deployments**
   ```bash
   # Watch logs
   tail -f logs/deploy_*.log

   # Monitor health
   watch -n 10 './health_check.sh'
   ```

4. **Use Version Tags**
   ```bash
   # Use git tags for production
   git tag -a v1.0.0 -m "Release v1.0.0"
   git push origin v1.0.0
   ```

5. **Enable Health Checks**
   - Always run health checks after deployment
   - Monitor critical metrics
   - Set up alerts

6. **Document Changes**
   - Maintain changelog
   - Document deployment steps
   - Track rollback history

---

## Support

For issues or questions:
- Documentation: `docs/DEPLOYMENT_RUNBOOK.md`
- Issues: https://github.com/victorai/victor/issues
- Discussions: https://github.com/victorai/victor/discussions

---

**Last Updated**: 2025-01-20
**Version**: 0.5.1
