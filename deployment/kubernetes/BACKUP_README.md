# Victor Backup Automation with Velero

This directory contains comprehensive backup automation setup using Velero for Kubernetes.

## Overview

The backup system provides:

- **Automated daily backups** using Velero
- **Backup verification** to ensure backup integrity
- **Automated cleanup** of old backups based on retention policy
- **Weekly integrity tests** using dry-run restores
- **Monitoring and alerting** integration with Prometheus
- **S3-compatible storage** for backup repository

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                       │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Velero Server (velero namespace)                     │  │
│  │  - Backup controller                                   │  │
│  │  - Restore controller                                  │  │
│  │  - Storage location management                         │  │
│  └──────────────────────────────────────────────────────┘  │
│                            │                                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  CronJobs                                             │  │
│  │  ├─ victor-backup (Daily 2 AM)                       │  │
│  │  ├─ victor-backup-verification (Daily 6 AM)          │  │
│  │  ├─ victor-backup-cleanup (Weekly Sunday 3 AM)       │  │
│  │  └─ victor-backup-test (Weekly Sunday 4 AM)          │  │
│  └──────────────────────────────────────────────────────┘  │
│                            │                                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Monitoring                                           │  │
│  │  ├─ ServiceMonitor (Velero metrics)                   │  │
│  │  └─ PrometheusRule (Alert rules)                     │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  S3-Compatible Storage (e.g., AWS S3, MinIO, Ceph)         │
│  - Backup repository                                       │
│  - Volume snapshots                                        │
│  - Metadata files                                          │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Prerequisites

```bash
# Install Velero CLI
brew install velero

# Or download from releases
curl -L https://github.com/vmware-tanzu/velero/releases/download/v1.13.0/velero-v1.13.0-darwin-amd64.tar.gz | tar xz
sudo mv velero-v1.13.0-darwin-amd64/velero /usr/local/bin/
```

### 2. Setup Backups

```bash
# Basic setup with IAM role
cd deployment/scripts
./setup_backups.sh --bucket my-k8s-backups

# Custom configuration
./setup_backups.sh \
  --bucket my-k8s-backups \
  --region us-west-2 \
  --schedule "0 3 * * *" \
  --retention 60

# Dry run to preview
./setup_backups.sh --bucket my-k8s-backups --dry-run

# With AWS credentials (not recommended, use IAM role instead)
AWS_ACCESS_KEY_ID=xxx AWS_SECRET_ACCESS_KEY=yyy \
  ./setup_backups.sh --bucket my-k8s-backups
```

### 3. Apply CronJobs

```bash
# Apply CronJobs from kubernetes/base
kubectl apply -f deployment/kubernetes/base/cronjob.yaml

# Verify CronJobs
kubectl get cronjobs -n velero
```

## Configuration

### Backup Schedule

Edit the schedule in `cronjob.yaml`:

```yaml
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
```

Cron format: `minute hour day month weekday`

Examples:
- `0 2 * * *` - Daily at 2 AM
- `0 */6 * * *` - Every 6 hours
- `0 2 * * 0` - Weekly on Sunday at 2 AM
- `0 2 1 * *` - Monthly on the 1st at 2 AM

### Retention Policy

Configure retention in `setup_backups.sh`:

```bash
--retention 60  # Keep backups for 60 days
```

Or set via environment variable:

```yaml
env:
- name: BACKUP_RETENTION_DAYS
  value: "30"
```

### Storage Location

```bash
# Create S3 bucket
aws s3api create-bucket \
  --bucket my-k8s-backups \
  --region us-east-1 \
  --create-bucket-configuration LocationConstraint=us-east-1

# Enable versioning
aws s3api put-bucket-versioning \
  --bucket my-k8s-backups \
  --versioning-configuration Status=Enabled
```

## Monitoring

### Prometheus Metrics

Velero exposes metrics at `http://velero.velero:8085/metrics`

Key metrics:
- `velero_backup_last_status` - Last backup status (0=Failed, 1=Completed)
- `velero_backup_last_success_timestamp_seconds` - Last successful backup timestamp
- `velero_backup_duration_seconds` - Backup duration
- `velero_restore_last_status` - Last restore status
- `velero_backup_storage_location_ready` - Storage location readiness

### Alert Rules

The system includes Prometheus alert rules for:

- **VeleroBackupFailed** (Critical) - Backup has failed
- **VeleroBackupOlderThan24h** (Warning) - No successful backup in 24 hours
- **VeleroRestoreFailed** (Critical) - Restore has failed
- **VeleroBackupStorageLocationNotReady** (Critical) - Storage location unavailable
- **VeleroVolumeSnapshotNotReady** (Warning) - Volume snapshot unavailable

View alerts in Prometheus or Grafana:

```bash
# Port forward to Prometheus
kubectl port-forward -n monitoring svc/prometheus 9090:9090

# Access at http://localhost:9090
```

## Operations

### Manual Backup

```bash
# Create on-demand backup
velero backup create urgent-backup \
  --namespace velero \
  --wait

# Create backup with specific namespaces
velero backup create victor-backup \
  --namespace velero \
  --include-namespaces victor,victor-database \
  --wait

# Create backup with label selector
velero backup create labeled-backup \
  --namespace velero \
  --selector app.kubernetes.io/name=victor \
  --wait
```

### Restore from Backup

```bash
# List backups
velero backup get -n velero

# Describe backup
velero backup describe victor-daily-20250121-020000 -n velero --details

# Restore from backup (dry run)
velero restore create test-restore \
  --from-backup victor-daily-20250121-020000 \
  --namespace velero \
  --dry-run \
  --wait

# Perform actual restore
velero restore create victor-restore \
  --from-backup victor-daily-20250121-020000 \
  --namespace velero \
  --wait

# Restore with namespace mapping
velero restore create victor-restore \
  --from-backup victor-daily-20250121-020000 \
  --namespace-mappings victor:victor-restored \
  --wait
```

### Backup Management

```bash
# List all backups
velero backup get -n velero

# List schedules
velero schedule get -n velero

# Delete a backup
velero backup delete old-backup -n velero --confirm

# Delete old backups (CLI)
velero backup delete --older-than 30d -n velero --confirm

# List backup storage locations
kubectl get backupstoragelocations -n velero

# Describe storage location
kubectl describe backupstoragelocation default -n velero
```

### Schedule Management

```bash
# Create schedule
velero schedule create hourly \
  --schedule "@hourly" \
  --namespace velero \
  --ttl 24h

# List schedules
velero schedule get -n velero

# Pause schedule
velero schedule pause hourly -n velero

# Resume schedule
velero schedule resume hourly -n velero

# Delete schedule
velero schedule delete hourly -n velero --confirm
```

## Disaster Recovery

### Full Cluster Restore

```bash
# 1. Install Velero in new cluster
./setup_backups.sh --bucket my-k8s-backups

# 2. Restore all resources
velero restore create --from-backup victor-daily-20250121-020000 \
  --namespace-mappings '*:*' \
  --wait

# 3. Verify restore
kubectl get all --all-namespaces
```

### Selective Restore

```bash
# Restore specific namespace
velero restore create victor-restore \
  --from-backup victor-daily-20250121-020000 \
  --include-namespaces victor \
  --wait

# Restore specific resources
velero restore create partial-restore \
  --from-backup victor-daily-20250121-020000 \
  --include-resources deployments,configmaps,secrets \
  --wait

# Restore with label selector
velero restore create labeled-restore \
  --from-backup victor-daily-20250121-020000 \
  --selector app.kubernetes.io/name=victor \
  --wait
```

## Troubleshooting

### Check Velero Status

```bash
# Check Velero pod
kubectl get pods -n velero

# Check Velero logs
kubectl logs -n velero deployment/velero -f

# Check Velero version
velero version

# Check backup status
velero backup get -n velero
```

### Common Issues

**Issue: Backup fails with "No credentials found"**

```bash
# Check if secret exists
kubectl get secret cloud-credentials -n velero

# Recreate secret
kubectl create secret generic cloud-credentials \
  --namespace velero \
  --from-file=cloud=credentials-velero \
  --dry-run=client -o yaml | kubectl apply -f -
```

**Issue: Storage location not available**

```bash
# Check storage location status
kubectl get backupstoragelocations -n velero
kubectl describe backupstoragelocation default -n velero

# Test S3 access
aws s3 ls s3://my-k8s-backups
```

**Issue: Backup timeout**

```bash
# Increase timeout
velero backup create my-backup \
  --timeout 14400s \
  --wait

# Check backup logs
velero backup describe my-backup --details
```

**Issue: Volume snapshot failed**

```bash
# Check volume snapshot locations
kubectl get volumesnapshotlocations -n velero

# Use file system backup instead
velero backup create my-backup \
  --default-volumes-to-fs-backup=true \
  --wait
```

### Backup Verification

```bash
# Run backup verification manually
kubectl create job backup-verify --from=cronjob/victor-backup-verification -n velero

# Check job logs
kubectl logs -n velero job/backup-verify-xxxxx -f
```

### Performance Tuning

```bash
# Increase Velero resource limits
kubectl set resources deployment velero \
  -n velero \
  --limits=cpu=2,memory=2Gi \
  --requests=cpu=500m,memory=512Mi

# Adjust backup parallelism
velero backup create my-backup \
  --uploader-type=kopia \
  --kopia-pool-size=4
```

## Testing

### Test Backup and Restore

```bash
# 1. Create test backup
velero backup create test-backup --wait

# 2. Verify backup completed
velero backup describe test-backup --details

# 3. Test restore (dry run)
velero restore create test-restore \
  --from-backup test-backup \
  --dry-run \
  --wait

# 4. Clean up
velero backup delete test-backup --confirm
velero restore delete test-restore --confirm
```

### Simulated Disaster Recovery

```bash
# 1. Create backup
velero backup create dr-test-backup --wait

# 2. Delete application (simulate disaster)
kubectl delete namespace victor

# 3. Restore from backup
velero restore create dr-test-restore \
  --from-backup dr-test-backup \
  --wait

# 4. Verify restore
kubectl get all -n victor
```

## Maintenance

### Regular Tasks

```bash
# Review backup storage usage
aws s3 ls s3://my-k8s-backups --recursive --summarize

# Check backup statistics
velero backup get -n velero -o json | jq '.items | length'

# Review old backups
velero backup get -n velero --older-than 30d

# Cleanup old backups
velero backup delete --older-than 30d -n velero --confirm
```

### Monitoring Setup

```bash
# Apply ServiceMonitor
kubectl apply -f deployment/kubernetes/base/cronjob.yaml

# Verify ServiceMonitor
kubectl get servicemonitor -n velero

# Check metrics
kubectl port-forward -n velero svc/velero 8085:8085
curl http://localhost:8085/metrics
```

## Security Best Practices

### IAM Role Configuration

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket",
        "s3:GetBucketVersioning",
        "s3:PutBucketVersioning"
      ],
      "Resource": [
        "arn:aws:s3:::my-k8s-backups",
        "arn:aws:s3:::my-k8s-backups/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "ec2:DescribeVolumes",
        "ec2:DescribeSnapshots",
        "ec2:CreateTags",
        "ec2:CreateSnapshot",
        "ec2:DeleteSnapshot"
      ],
      "Resource": "*"
    }
  ]
}
```

### Encryption at Rest

```bash
# Enable S3 bucket encryption
aws s3api put-bucket-encryption \
  --bucket my-k8s-backups \
  --server-side-encryption-configuration '{
    "Rules": [{
      "ApplyServerSideEncryptionByDefault": {
        "SSEAlgorithm": "AES256"
      }
    }]
  }'
```

## Cost Optimization

### Lifecycle Policies

```bash
# Transition old backups to Glacier
aws s3api put-bucket-lifecycle-configuration \
  --bucket my-k8s-backups \
  --lifecycle-configuration '{
    "Rules": [{
      "Id": "BackupLifecycle",
      "Status": "Enabled",
      "Transitions": [{
        "Days": 30,
        "StorageClass": "GLACIER"
      }],
      "Expiration": {
        "Days": 90
      }
    }]
  }'
```

### Backup Compression

Velero automatically compresses backups. Monitor backup sizes:

```bash
# Check backup sizes
aws s3 ls s3://my-k8s-backups/velero/backups/ --recursive --human-readable
```

## Additional Resources

- [Velero Documentation](https://velero.io/docs/)
- [Velero GitHub](https://github.com/vmware-tanzu/velero)
- [AWS EKS Backup with Velero](https://aws.amazon.com/blogs/containers/backup-and-restore-amazon-eks-cluster-resources-using-velero/)
- [Troubleshooting Guide](https://velero.io/docs/troubleshooting/)

## Support

For issues or questions:
1. Check Velero logs: `kubectl logs -n velero deployment/velero`
2. Check CronJob logs: `kubectl logs -n velero job/<job-name>`
3. Review backup details: `velero backup describe <backup-name> --details`
4. Check storage location: `kubectl describe backupstoragelocation default -n velero`
