# Velero Backup Quick Reference

## Setup Verification

```bash
# Run setup script
cd deployment/scripts
./setup_backups.sh --bucket my-k8s-backups --region us-east-1

# Verify installation
./verify_backup_setup.sh --verbose

# Check Velero version
velero version

# Check Velero pods
kubectl get pods -n velero
```

## Daily Operations

### View Backup Status

```bash
# List all backups
velero backup get -n velero

# List backups with details
velero backup get -n velero -o wide

# Describe specific backup
velero backup describe victor-daily-20250121-020000 -n velero --details

# Check backup storage location
kubectl get backupstoragelocations -n velero
kubectl describe backupstoragelocation default -n velero
```

### Create Manual Backup

```bash
# Quick backup
velero backup create urgent-backup -n velero --wait

# Backup with specific namespaces
velero backup create victor-backup \
  -n velero \
  --include-namespaces victor,victor-database \
  --wait

# Backup with labels
velero backup create labeled-backup \
  -n velero \
  --selector app.kubernetes.io/name=victor \
  --wait

# Backup with TTL (retention)
velero backup create temp-backup \
  -n velero \
  --ttl 24h \
  --wait
```

### Restore Operations

```bash
# List backups for restore
velero backup get -n velero

# Restore (dry run first)
velero restore create test-restore \
  --from-backup victor-daily-20250121-020000 \
  --namespace velero \
  --dry-run \
  --wait

# Actual restore
velero restore create victor-restore \
  --from-backup victor-daily-20250121-020000 \
  --namespace velero \
  --wait

# Restore to different namespace
velero restore create victor-restore \
  --from-backup victor-daily-20250121-020000 \
  --namespace-mappings victor:victor-restored \
  --wait

# List restores
velero restore get -n velero

# Describe restore
velero restore describe victor-restore -n velero --details
```

## Schedule Management

```bash
# List schedules
velero schedule get -n velero

# Create hourly schedule
velero schedule create hourly \
  --schedule "@hourly" \
  -n velero \
  --ttl 24h

# Pause schedule
velero schedule pause daily-backup -n velero

# Resume schedule
velero schedule resume daily-backup -n velero

# Delete schedule
velero schedule delete old-schedule -n velero --confirm
```

## CronJob Management

```bash
# List CronJobs
kubectl get cronjobs -n velero

# View CronJob details
kubectl describe cronjob victor-backup -n velero

# View CronJob history
kubectl get jobs -n velero -l app.kubernetes.io/component=backup

# Manually trigger CronJob
kubectl create job manual-backup --from=cronjob/victor-backup -n velero

# View job logs
kubectl logs -n velero job/manual-backup-xxxxx -f

# Pause CronJob
kubectl patch cronjob victor-backup -n velero -p '{"spec":{"suspend":true}}'

# Resume CronJob
kubectl patch cronjob victor-backup -n velero -p '{"spec":{"suspend":false}}'
```

## Troubleshooting

### Check Velero Logs

```bash
# Velero server logs
kubectl logs -n velero deployment/velero -f

# Backup job logs
kubectl logs -n velero job/victor-backup-xxxxx -f

# Verification job logs
kubectl logs -n velero job/victor-backup-verification-xxxxx -f
```

### Check Backup Status

```bash
# Get backup status
velero backup get my-backup -n velero

# Describe backup with errors
velero backup describe my-backup -n velero --details

# Check backup in S3
aws s3 ls s3://my-k8s-backups/velero/backups/

# Download backup for inspection
aws s3 cp s3://my-k8s-backups/velero/backups/my-backup/velero-backup.json .
```

### Common Issues

**Backup stuck in progress:**
```bash
# Check backup status
velero backup get -n velero

# Cancel long-running backup
velero backup delete stuck-backup -n velero --confirm
```

**Storage location not available:**
```bash
# Check storage location
kubectl get backupstoragelocations -n velero
kubectl describe backupstoragelocation default -n velero

# Test S3 access
aws s3 ls s3://my-k8s-backups
```

**Permission errors:**
```bash
# Check service account
kubectl get serviceaccount -n velero velero-server

# Check cluster role binding
kubectl get clusterrolebinding velero-server

# View role details
kubectl describe clusterrolebinding velero-server
```

## Maintenance

### Cleanup Old Backups

```bash
# Delete specific backup
velero backup delete old-backup -n velero --confirm

# Delete backups older than 30 days
velero backup delete --older-than 30d -n velero --confirm

# Delete all backups (use with caution)
velero backup delete --all -n velero --confirm
```

### Backup Statistics

```bash
# Count backups
velero backup get -n velero -o json | jq '.items | length'

# Total backup size in S3
aws s3 ls s3://my-k8s-backups --recursive --summarize | grep "Total"

# Latest backup
velero backup get -n velero -o json | \
  jq -r '.items | sort_by(.status.startTimestamp) | reverse | [0] | .[0].name'

# Failed backups
velero backup get -n velero -o json | \
  jq -r '.items[] | select(.status.phase != "Completed") | .name'
```

## Testing

### Test Backup and Restore

```bash
# Create test backup
velero backup create test-backup -n velero --wait

# Verify backup
velero backup describe test-backup -n velero --details

# Test restore (dry run)
velero restore create test-restore \
  --from-backup test-backup \
  -n velero \
  --dry-run \
  --wait

# Cleanup
velero restore delete test-restore -n velero --confirm
velero backup delete test-backup -n velero --confirm
```

### Disaster Recovery Drill

```bash
# 1. Create backup
velero backup create dr-test -n velero --wait

# 2. Note backup name
BACKUP=$(velero backup get -n velero -o json | jq -r '.items[-1].name')

# 3. Simulate disaster (delete namespace)
kubectl delete namespace victor

# 4. Restore from backup
velero restore create dr-restore \
  --from-backup $BACKUP \
  -n velero \
  --wait

# 5. Verify restore
kubectl get all -n victor
```

## Monitoring

### Metrics Endpoint

```bash
# Port forward to access metrics
kubectl port-forward -n velero svc/velero 8085:8085

# Access metrics
curl http://localhost:8085/metrics

# Key metrics to check
curl -s http://localhost:8085/metrics | grep velero_backup_last_status
curl -s http://localhost:8085/metrics | grep velero_backup_duration_seconds
```

### Prometheus Queries

```bash
# Last backup status
velero_backup_last_status

# Backup duration
velero_backup_duration_seconds

# Last successful backup timestamp
velero_backup_last_success_timestamp_seconds

# Storage location readiness
velero_backup_storage_location_ready
```

## Configuration Files

### CronJobs Location

```bash
# View CronJob YAML
kubectl get cronjob victor-backup -n velero -o yaml

# Edit CronJob
kubectl edit cronjob victor-backup -n velero

# Apply updated CronJob
kubectl apply -f deployment/kubernetes/base/cronjob.yaml
```

### Update Schedule

```bash
# Edit CronJob schedule
kubectl patch cronjob victor-backup -n velero \
  -p '{"spec":{"schedule":"0 3 * * *"}}'

# Edit multiple CronJobs
kubectl patch cronjob victor-backup-verification -n velero \
  -p '{"spec":{"schedule":"0 6 * * *"}}'
```

## Quick Commands Reference

| Task | Command |
|------|---------|
| List backups | `velero backup get -n velero` |
| Create backup | `velero backup create my-backup -n velero --wait` |
| Describe backup | `velero backup describe my-backup -n velero --details` |
| Restore backup | `velero restore create my-restore --from-backup my-backup -n velero --wait` |
| List schedules | `velero schedule get -n velero` |
| List CronJobs | `kubectl get cronjobs -n velero` |
| View logs | `kubectl logs -n velero deployment/velero -f` |
| Check pods | `kubectl get pods -n velero` |
| Verify setup | `./verify_backup_setup.sh --verbose` |

## Emergency Procedures

### Restore from S3 (if Velero not installed)

```bash
# 1. Install Velero
./setup_backups.sh --bucket my-k8s-backups

# 2. Sync backups from S3
velero backup get -n velero

# 3. Restore
velero restore create emergency-restore \
  --from-backup latest-backup-name \
  -n velero \
  --wait
```

### Partial Restore

```bash
# Restore specific namespace
velero restore create partial-restore \
  --from-backup my-backup \
  --include-namespaces victor \
  -n velero \
  --wait

# Restore specific resources
velero restore create resource-restore \
  --from-backup my-backup \
  --include-resources deployments,configmaps \
  -n velero \
  --wait

# Restore with label selector
velero restore create labeled-restore \
  --from-backup my-backup \
  --selector app.kubernetes.io/name=victor \
  -n velero \
  --wait
```

## Support

```bash
# Check Velero logs
kubectl logs -n velero deployment/velero --tail=100

# Describe problematic backup
velero backup describe failed-backup -n velero --details

# Check storage location
kubectl describe backupstoragelocation default -n velero

# Run verification script
./verify_backup_setup.sh --verbose --fix
```

## Useful Aliases

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
# Velero aliases
alias v-backups='velero backup get -n velero'
alias v-schedules='velero schedule get -n velero'
alias v-logs='kubectl logs -n velero deployment/velero -f'
alias v-status='kubectl get pods -n velero'
alias v-verify='~/code/codingagent/deployment/scripts/verify_backup_setup.sh --verbose'
```
