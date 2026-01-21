# Victor AI Deployment - Quick Reference

## Deployment Commands

### Pip Deployment
```bash
# Staging
./scripts/deploy.sh staging

# Production
./scripts/deploy.sh production

# Rollback
./scripts/deploy.sh production --rollback
```

### Docker Deployment
```bash
# Build and push
./scripts/docker/build.sh 1.0.0 --push --scan

# Run container
docker run -d -p 8000:8000 --env-file .env victorai/victor:1.0.0
```

### Kubernetes Deployment
```bash
# Deploy
./scripts/kubernetes/deploy.sh production --namespace victor-prod

# Check status
kubectl get pods -n victor-prod
kubectl logs -f deployment/victor-prod -n victor-prod

# Rollback
helm rollback victor-prod -n victor-prod
```

### Ansible Deployment
```bash
# Deploy to servers
ansible-playbook -i inventory playbook.yml --extra-vars "env=production"
```

## Health Checks

```bash
# Basic health check
./scripts/health_check.sh

# Custom endpoint
./scripts/health_check.sh --endpoint http://localhost:8000/health

# Verbose mode
./scripts/health_check.sh --verbose --timeout 30
```

## Rollback

```bash
# Interactive
./scripts/rollback.sh

# Specific version
./scripts/rollback.sh --version backup_20250120_153000

# Forced
./scripts/rollback.sh --force --keep-db

# Dry run
./scripts/rollback.sh --dry-run
```

## Environment Variables

```bash
# Required for production
export VICTOR_PROFILE=production
export VICTOR_LOG_LEVEL=INFO

# Optional
export VICTOR_API_KEY=your_key_here
export VICTOR_MAX_WORKERS=4
export VICTOR_CACHE_TTL=3600
```

## Health Endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Basic health |
| `GET /health/db` | Database status |
| `GET /health/providers` | Provider status |
| `GET /health/resources` | Resource usage |
| `GET /health/critical` | Critical services |

## Useful Commands

```bash
# Check service status
sudo systemctl status victor-api

# View logs
tail -f logs/victor.log

# Docker logs
docker logs -f victor

# Kubernetes logs
kubectl logs -f deployment/victor-prod -n victor-prod

# Restart service
sudo systemctl restart victor-api

# Scale Kubernetes deployment
kubectl scale deployment/victor-prod --replicas=3 -n victor-prod
```

## Troubleshooting

```bash
# Port in use
lsof -i :8000

# Kill existing process
pkill -f "victor.api.server"

# Check Python version
python --version

# Check Docker
docker info

# Check Kubernetes
kubectl cluster-info
```

## CI/CD

```bash
# Trigger workflow manually (GitHub UI)
Actions → Deploy to Production → Run workflow

# Or via git tag
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0
```

## Backup Locations

```bash
# Automatic backups
/backups/backup_YYYYMMDD_HHMMSS/

# Contains:
- venv/           # Virtual environment
- .env            # Configuration
- victor.db       # Database
- git_commit.txt  # Git state
```

## Documentation

- **Full Runbook**: `docs/DEPLOYMENT_RUNBOOK.md`
- **Scripts Guide**: `scripts/README.md`
- **Summary**: `docs/DEPLOYMENT_AUTOMATION_SUMMARY.md`
- **This Guide**: `docs/DEPLOYMENT_QUICK_REFERENCE.md`

## Support

- Issues: https://github.com/victorai/victor/issues
- Discussions: https://github.com/victorai/victor/discussions
- Documentation: https://docs.victorai.com

---

**Quick Links**:
- [Deployment Runbook](./DEPLOYMENT_RUNBOOK.md)
- [Scripts Documentation](../scripts/README.md)
- [Deployment Summary](./DEPLOYMENT_AUTOMATION_SUMMARY.md)
