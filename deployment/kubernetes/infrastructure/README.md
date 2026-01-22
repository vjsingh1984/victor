# Infrastructure Deployment Guide

This directory contains production-ready infrastructure components for Victor AI deployment on Kubernetes.

## Overview

The infrastructure includes:

- **PostgreSQL**: StatefulSet with persistent storage for application data
- **Redis**: StatefulSet with persistent storage for caching and sessions
- **NGINX Ingress Controller**: Layer 7 load balancing and SSL termination
- **cert-manager**: Automatic TLS certificate management with Let's Encrypt

## Quick Start

### 1. Deploy All Infrastructure

```bash
# From the project root
./deployment/scripts/deploy_infrastructure.sh
```

### 2. Deploy Specific Components

```bash
# Deploy only PostgreSQL and Redis
./deployment/scripts/deploy_infrastructure.sh --components postgres,redis

# Deploy ingress and cert-manager
./deployment/scripts/deploy_infrastructure.sh --components ingress,cert-manager
```

### 3. Dry Run (Test Without Deploying)

```bash
./deployment/scripts/deploy_infrastructure.sh --dry-run
```

## Components

### PostgreSQL

**Features:**
- StatefulSet for stable network identity
- Persistent volume claim (10Gi default)
- Automated health checks and failover
- Prometheus metrics integration
- Production-optimized configuration

**Connection Details:**
```
Host: postgres.victor.svc.cluster.local
Port: 5432
Database: victor
User: victor
Password: $(kubectl get secret -n victor postgres-secret -o jsonpath='{.data.POSTGRES_PASSWORD}' | base64 -d)
```

**Get Password:**
```bash
kubectl get secret -n victor postgres-secret \
  -o jsonpath='{.data.POSTGRES_PASSWORD}' | base64 -d
```

**Connect to Database:**
```bash
kubectl exec -it -n victor postgres-0 -- psql -U victor -d victor
```

**Backup Database:**
```bash
kubectl exec -n victor postgres-0 -- pg_dump -U victor victor > backup.sql
```

**Restore Database:**
```bash
cat backup.sql | kubectl exec -i -n victor postgres-0 -- psql -U victor victor
```

### Redis

**Features:**
- StatefulSet for stable network identity
- Persistent volume claim (5Gi default)
- AOF persistence enabled
- Max memory policy: allkeys-lru
- Redis exporter for Prometheus monitoring

**Connection Details:**
```
Host: redis.victor.svc.cluster.local
Port: 6379
Database: 0 (default)
```

**Connect to Redis:**
```bash
kubectl exec -it -n victor redis-0 -- redis-cli
```

**Monitor Redis:**
```bash
kubectl exec -n victor redis-0 -- redis-cli INFO
```

**Flush Cache (Use with Caution):**
```bash
kubectl exec -n victor redis-0 -- redis-cli FLUSHALL
```

### NGINX Ingress Controller

**Features:**
- High availability (2 replicas)
- LoadBalancer service type
- HTTP/HTTPS support
- SSL termination
- Webhook validation
- Prometheus metrics

**Get LoadBalancer IP:**
```bash
kubectl get service ingress-nginx-controller -n ingress-nginx \
  -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
```

**View Ingress Logs:**
```bash
kubectl logs -n ingress-nginx -l app.kubernetes.io/name=ingress-nginx --tail=100 -f
```

**View Ingress Configuration:**
```bash
kubectl exec -n ingress-nginx ingress-nginx-controller-<pod> -- cat /etc/nginx/nginx.conf
```

### cert-manager

**Features:**
- Automatic TLS certificate issuance
- Let's Encrypt integration (staging and production)
- Automatic certificate renewal
- Multiple domain support
- ACME protocol support

**Check Certificate Status:**
```bash
kubectl get certificate -n victor
kubectl describe certificate victor-tls -n victor
```

**View Certificate:**
```bash
kubectl get secret victor-tls -n victor -o jsonpath='{.data.tls\.crt}' | base64 -d | openssl x509 -text
```

**Test Certificate (Staging):**
```bash
# Update Ingress to use staging issuer
kubectl patch certificate victor-tls -n victor \
  --type='json' -p='[{"op": "replace", "path": "/spec/issuerRef/name", "value": "letsencrypt-staging"}]'
```

## Configuration

### Custom Storage Class

```bash
./deployment/scripts/deploy_infrastructure.sh --storage-class fast-ssd
```

### Custom Namespace

```bash
./deployment/scripts/deploy_infrastructure.sh --namespace my-namespace
```

## Operations

### Health Checks

```bash
# Check all infrastructure health
kubectl get pods -n victor
kubectl get pvc -n victor
kubectl get statefulsets -n victor

# PostgreSQL health
kubectl exec -n victor postgres-0 -- pg_isready -U victor -d victor

# Redis health
kubectl exec -n victor redis-0 -- redis-cli ping
```

### Scaling

**Scale PostgreSQL (Note: Requires replication setup):**
```bash
kubectl scale statefulset postgres -n victor --replicas=3
```

**Scale Redis (Note: Requires Redis Cluster setup):**
```bash
kubectl scale statefulset redis -n victor --replicas=3
```

**Scale Ingress Controller:**
```bash
kubectl scale deployment ingress-nginx-controller -n ingress-nginx --replicas=3
```

### Maintenance

**Restart PostgreSQL:**
```bash
kubectl rollout restart statefulset postgres -n victor
```

**Restart Redis:**
```bash
kubectl rollout restart statefulset redis -n victor
```

**Restart Ingress:**
```bash
kubectl rollout restart deployment ingress-nginx-controller -n ingress-nginx
```

### Monitoring

**View Metrics:**
```bash
# PostgreSQL metrics (requires postgres-exporter)
kubectl port-forward -n victor postgres-0 9187:9187
curl http://localhost:9187/metrics

# Redis metrics
kubectl port-forward -n victor redis-0 9121:9121
curl http://localhost:9121/metrics

# Ingress metrics
kubectl port-forward -n ingress-nginx <ingress-pod> 10254:10254
curl http://localhost:10254/metrics
```

### Logs

**PostgreSQL Logs:**
```bash
kubectl logs -n victor statefulset/postgres --tail=100 -f
```

**Redis Logs:**
```bash
kubectl logs -n victor statefulset/redis --tail=100 -f
```

**Ingress Logs:**
```bash
kubectl logs -n ingress-nginx -l app.kubernetes.io/name=ingress-nginx --tail=100 -f
```

## Rollback

### Rollback All Components

```bash
./deployment/scripts/deploy_infrastructure.sh --rollback
```

### Rollback Specific Component

```bash
# Rollback PostgreSQL
kubectl delete statefulset postgres -n victor
kubectl apply -f kubernetes/infrastructure/postgres-statefulset.yaml
```

### Uninstall

```bash
# Uninstall all infrastructure
./deployment/scripts/deploy_infrastructure.sh --uninstall

# Uninstall specific components
./deployment/scripts/deploy_infrastructure.sh --components postgres,redis --uninstall
```

**Warning:** Uninstalling does not delete persistent volumes by default. To delete volumes:

```bash
kubectl delete pvc -n victor --all
```

## Troubleshooting

### PostgreSQL Issues

**Pod Not Starting:**
```bash
# Check events
kubectl describe pod -n victor -l app=postgres

# Check logs
kubectl logs -n victor statefulset/postgres

# Check PVC
kubectl get pvc -n victor -l app=postgres
```

**Connection Issues:**
```bash
# Test from application pod
kubectl run -it --rm debug --image=postgres:15 --restart=Never -- \
  psql postgres://victor:password@postgres.victor.svc.cluster.local:5432/victor
```

### Redis Issues

**Pod Not Starting:**
```bash
kubectl describe pod -n victor -l app=redis
kubectl logs -n victor statefulset/redis
```

**Connection Issues:**
```bash
kubectl run -it --rm debug --image=redis:7 --restart=Never -- \
  redis-cli -h redis.victor.svc.cluster.local ping
```

### Ingress Issues

**No LoadBalancer IP:**
```bash
# Check service
kubectl describe service ingress-nginx-controller -n ingress-nginx

# Check cloud provider load balancer
# (AWS, GCP, Azure consoles)
```

**502 Bad Gateway:**
```bash
# Check backend service exists
kubectl get svc -n victor

# Check ingress rules
kubectl describe ingress -n victor

# Check ingress controller logs
kubectl logs -n ingress-nginx -l app.kubernetes.io/name=ingress-nginx
```

### Certificate Issues

**Certificate Not Issued:**
```bash
# Check certificate status
kubectl describe certificate victor-tls -n victor

# Check certificate request
kubectl get certificaterequest -n victor

# Check cert-manager logs
kubectl logs -n cert-manager -l app.kubernetes.io/name=cert-manager
```

**Rate Limited by Let's Encrypt:**
```bash
# Switch to staging issuer for testing
kubectl patch certificate victor-tls -n victor --type='json' \
  -p='[{"op": "replace", "path": "/spec/issuerRef/name", "value": "letsencrypt-staging"}]'
```

## Security Best Practices

1. **Change Default Passwords**
   ```bash
   kubectl create secret generic postgres-secret -n victor \
     --from-literal=POSTGRES_PASSWORD=$(openssl rand -base64 32)
   ```

2. **Enable Network Policies**
   ```bash
   kubectl apply -f kubernetes/base/networkpolicy.yaml
   ```

3. **Use Secrets Management**
   - Consider using External Secrets Operator
   - Integrate with AWS Secrets Manager, GCP Secret Manager, or Azure Key Vault

4. **Enable Pod Security Policies**
   ```bash
   kubectl apply -f kubernetes/infrastructure/psp.yaml
   ```

5. **Regular Updates**
   ```bash
   # Update images
   kubectl set image statefulset/postgres postgres=postgres:15-alpine -n victor
   kubectl set image statefulset/redis redis=redis:7-alpine -n victor
   ```

## Performance Tuning

### PostgreSQL

**Increase Resources:**
```yaml
resources:
  requests:
    cpu: 1000m
    memory: 2Gi
  limits:
    cpu: 2000m
    memory: 4Gi
```

**Tune Configuration:**
Edit the `postgres.conf` in the ConfigMap to match your workload.

### Redis

**Increase Memory:**
```yaml
- name: redis
  command:
    - redis-server
    args:
    - /etc/redis/redis.conf
    - --maxmemory
    - 2gb
```

**Enable Persistence:**
```yaml
# Append only file
appendonly yes
appendfsync everysec
```

### Ingress

**Increase Replicas:**
```bash
kubectl scale deployment ingress-nginx-controller -n ingress-nginx --replicas=3
```

**Enable PROXY Protocol:**
```yaml
# For preserving client IP behind proxy
use-forwarded-headers: "true"
compute-full-forwarded-for: "true"
```

## Backup and Recovery

### Automated Backups

Use the backup scripts:
```bash
./deployment/scripts/backup/backup_database.sh
./deployment/scripts/backup/backup_volumes.sh
```

### Manual Backup

**PostgreSQL:**
```bash
kubectl exec -n victor postgres-0 -- pg_dump -U victor victor > backup.sql
```

**Redis:**
```bash
kubectl exec -n victor redis-0 -- redis-cli SAVE
kubectl cp victor/redis-0:/data/dump.rdb ./redis-backup.rdb
```

**Volumes:**
```bash
# Get PVC name
PVC=$(kubectl get pvc -n victor -l app=postgres -o jsonpath='{.items[0].metadata.name}')

# Create snapshot (cloud provider specific)
# AWS EBS
aws ec2 create-snapshot --volume-id <volume-id>

# GCP Persistent Disk
gcloud compute disks snapshot <disk-name> --snapshot-name postgres-backup-$(date +%Y%m%d)
```

## Support

For issues or questions:

1. Check logs: `kubectl logs -n victor <pod-name>`
2. Check events: `kubectl describe pod -n victor <pod-name>`
3. Review documentation in `/Users/vijaysingh/code/codingagent/deployment/kubernetes/README.md`
4. Create an issue in the project repository

## Additional Resources

- [PostgreSQL Kubernetes Operator](https://github.com/zalando/postgres-operator)
- [Redis on Kubernetes](https://redis.io/docs/management/scale/)
- [NGINX Ingress Controller](https://kubernetes.github.io/ingress-nginx/)
- [cert-manager Documentation](https://cert-manager.io/docs/)
