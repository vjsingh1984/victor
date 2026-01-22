# Disaster Recovery Runbook for Victor Backups

## Overview

This runbook provides step-by-step procedures for handling disaster recovery scenarios using Velero backups.

## Table of Contents

1. [Pre-Recovery Checklist](#pre-recovery-checklist)
2. [Recovery Scenarios](#recovery-scenarios)
3. [Testing Procedures](#testing-procedures)
4. [Post-Recovery Verification](#post-recovery-verification)
5. [Emergency Contacts](#emergency-contacts)

---

## Pre-Recovery Checklist

Before attempting any recovery operation:

### 1. Assess the Situation

```bash
# Determine the scope of the disaster
# - Single namespace affected?
# - Entire cluster affected?
# - Storage affected?
# - Multiple regions affected?

# Check cluster status
kubectl cluster-info
kubectl get nodes
kubectl get pods --all-namespaces

# Check Velero status
kubectl get pods -n velero
velero version
```

### 2. Identify the Backup

```bash
# List available backups
velero backup get -n velero

# Find the most recent successful backup
LATEST_BACKUP=$(velero backup get -n velero -o json | \
  jq -r '.items | sort_by(.status.startTimestamp) | reverse | [0] | .[0].name')

echo "Latest backup: $LATEST_BACKUP"

# Verify backup integrity
velero backup describe $LATEST_BACKUP -n velero --details
```

### 3. Verify Backup Storage

```bash
# Check storage location availability
kubectl get backupstoragelocations -n velero
kubectl describe backupstoragelocation default -n velero

# Verify S3 access
aws s3 ls s3://my-k8s-backups/velero/backups/

# Check backup files in S3
aws s3 ls s3://my-k8s-backups/velero/backups/$LATEST_BACKUP/
```

### 4. Document the Recovery

Create a recovery log:

```bash
# Create recovery log file
cat > /tmp/recovery-log-$(date +%Y%m%d-%H%M%S).txt << EOF
Recovery Log
Date: $(date)
Operator: $(whoami)
Cluster: $(kubectl config current-context)
Latest Backup: $LATEST_BACKUP
Situation: [Describe the disaster]
Steps Taken:
1.
2.
3.
Result:
EOF
```

---

## Recovery Scenarios

### Scenario 1: Single Namespace Deletion

**Situation**: The `victor` namespace was accidentally deleted.

**Recovery Time**: 15-30 minutes

**Steps**:

1. **Confirm namespace is missing**
   ```bash
   kubectl get namespace victor
   # Error: namespace not found
   ```

2. **Identify the backup**
   ```bash
   # Get latest backup
   BACKUP=$(velero backup get -n velero -o json | \
     jq -r '.items | sort_by(.status.startTimestamp) | reverse | [0] | .[0].name')
   ```

3. **Perform restore**
   ```bash
   # Create restore
   velero restore create victor-restore-$(date +%Y%m%d-%H%M%S) \
     --from-backup $BACKUP \
     --namespace-mappings victor:victor \
     --include-namespaces victor \
     --wait

   # Monitor restore progress
   velero restore get -n velero
   watch kubectl get all -n victor
   ```

4. **Verify restore**
   ```bash
   # Check namespace
   kubectl get namespace victor

   # Check pods
   kubectl get pods -n victor

   # Check services
   kubectl get svc -n victor

   # Check deployments
   kubectl get deployments -n victor
   ```

5. **Test application**
   ```bash
   # Port forward to test
   kubectl port-forward -n victor svc/victor 8080:80

   # Access application
   curl http://localhost:8080/health
   ```

**Rollback if needed**:
```bash
# Delete restore and retry
velero restore delete victor-restore-<timestamp> -n velero --confirm
kubectl delete namespace victor
# Retry restore with different backup
```

---

### Scenario 2: Complete Cluster Loss

**Situation**: Entire Kubernetes cluster is lost.

**Recovery Time**: 2-4 hours

**Steps**:

1. **Create new cluster**
   ```bash
   # Using cloud provider console or CLI
   # Example for EKS:
   aws eks create-cluster \
     --name victor-prod-new \
     --role-arn <role-arn> \
     --resources-vpc-config <vpc-config> \
     --region us-east-1

   # Wait for cluster to be ready
   aws eks wait cluster-active --name victor-prod-new

   # Update kubeconfig
   aws eks update-kubeconfig --name victor-prod-new --region us-east-1
   ```

2. **Install Velero**
   ```bash
   cd deployment/scripts
   ./setup_backups.sh --bucket my-k8s-backups --region us-east-1
   ```

3. **Sync backups from S3**
   ```bash
   # Wait for Velero to sync backups
   sleep 60

   # Verify backups are visible
   velero backup get -n velero
   ```

4. **Restore cluster resources**
   ```bash
   # Restore all cluster-scoped resources
   velero restore create cluster-resources-restore \
     --from-backup <latest-backup> \
     --include-cluster-resources=true \
     --exclude-namespaces kube-system,kube-public,kube-node-lease \
     --wait
   ```

5. **Restore namespaces**
   ```bash
   # Restore application namespaces
   velero restore create app-restore \
     --from-backup <latest-backup> \
     --include-namespaces victor,victor-database,monitoring \
     --wait
   ```

6. **Verify cluster**
   ```bash
   # Check all namespaces
   kubectl get namespaces

   # Check all pods
   kubectl get pods --all-namespaces

   # Check storage classes
   kubectl get storageclass

   # Check PVs and PVCs
   kubectl get pv
   kubectl get pvc --all-namespaces
   ```

7. **Update DNS/LB**
   ```bash
   # Update load balancer endpoints
   kubectl get svc -n victor

   # Update DNS records to point to new LB
   ```

---

### Scenario 3: Persistent Volume Loss

**Situation**: Persistent volumes are corrupted or lost.

**Recovery Time**: 1-2 hours

**Steps**:

1. **Identify affected volumes**
   ```bash
   kubectl get pvc -n victor
   kubectl get pv
   kubectl describe pvc <pvc-name> -n victor
   ```

2. **Check backup for volume data**
   ```bash
   velero backup describe <backup-name> -n velero --details | grep -A 50 "Volumes"
   ```

3. **Delete affected PVCs and PVs**
   ```bash
   # Delete PVCs (PVs will be deleted automatically if reclaim policy is Delete)
   kubectl delete pvc <pvc-name> -n victor
   ```

4. **Restore from backup**
   ```bash
   # Restore with volume data
   velero restore create volume-restore \
     --from-backup <backup-name> \
     --include-resources persistentvolumes,persistentvolumeclaims \
     --wait
   ```

5. **Restart affected pods**
   ```bash
   # Pods will automatically restart when PVCs are ready
   kubectl delete pods -n victor -l app=victor
   ```

---

### Scenario 4: Accidental Configuration Change

**Situation**: Incorrect configuration was applied and needs to be rolled back.

**Recovery Time**: 30-60 minutes

**Steps**:

1. **Identify the bad change**
   ```bash
   # Check recent changes
   kubectl rollout history deployment/victor -n victor
   ```

2. **Find backup before the change**
   ```bash
   # List backups with timestamps
   velero backup get -n velero -o json | \
     jq -r '.items[] | "\(.metadata.name): \(.status.startTimestamp)"' | \
     sort -k2 -r
   ```

3. **Restore specific resources**
   ```bash
   # Option 1: Restore specific configmaps/secrets
   velero restore create config-restore \
     --from-backup <backup-before-change> \
     --include-resources configmaps,secrets \
     --selector app.kubernetes.io/name=victor \
     --wait

   # Option 2: Rollback deployment
   kubectl rollout undo deployment/victor -n victor

   # Option 3: Restore from backup
   velero restore create full-restore \
     --from-backup <backup-before-change> \
     --include-namespaces victor \
     --wait
   ```

---

### Scenario 5: Regional Disaster

**Situation**: Entire AWS region is down.

**Recovery Time**: 4-8 hours

**Steps**:

1. **Activate DR region**
   ```bash
   # Cross-region replication must be configured beforehand
   # Check replicated backups
   aws s3 ls s3://my-k8s-backups-dr --recursive
   ```

2. **Create new cluster in DR region**
   ```bash
   # Update region
   export AWS_REGION=us-west-2

   # Create cluster
   aws eks create-cluster \
     --name victor-prod-dr \
     --region us-west-2 \
     ...

   # Update kubeconfig
   aws eks update-kubeconfig --name victor-prod-dr --region us-west-2
   ```

3. **Install Velero pointing to DR bucket**
   ```bash
   ./setup_backups.sh \
     --bucket my-k8s-backups-dr \
     --region us-west-2
   ```

4. **Restore from DR backups**
   ```bash
   # Follow same restore procedure as Scenario 2
   ```

5. **Cutover traffic**
   ```bash
   # Update DNS to DR region
   # Update CDN origins
   # Update any hardcoded endpoints
   ```

---

## Testing Procedures

### Monthly DR Drill

Perform a monthly disaster recovery test:

```bash
#!/bin/bash
# Monthly DR Drill Script

# 1. Create test backup
BACKUP_NAME="dr-test-$(date +%Y%m%d)"
velero backup create $BACKUP_NAME --wait

# 2. Create test namespace
kubectl create namespace victor-dr-test

# 3. Restore to test namespace
velero restore create dr-test-restore \
  --from-backup $BACKUP_NAME \
  --namespace-mappings victor:victor-dr-test \
  --wait

# 4. Verify restore
kubectl get pods -n victor-dr-test
kubectl get svc -n victor-dr-test

# 5. Run smoke tests
# [Add application-specific tests]

# 6. Cleanup
kubectl delete namespace victor-dr-test
velero restore delete dr-test-restore --confirm
velero backup delete $BACKUP_NAME --confirm

echo "DR drill completed successfully"
```

### Backup Integrity Check

```bash
# Weekly backup verification
velero backup get -n velero -o json | \
  jq -r '.items[] | select(.status.phase != "Completed") | .name' | \
  while read backup; do
    echo "Investigating failed backup: $backup"
    velero backup describe $backup -n velero --details
  done
```

### Restore Time Testing

```bash
# Time your restore operations
time velero restore create test-restore \
  --from-backup <backup-name> \
  --wait

# Document RTO (Recovery Time Objective)
```

---

## Post-Recovery Verification

### Checklist

- [ ] All namespaces restored
- [ ] All pods are running
- [ ] All services are accessible
- [ ] Persistent volumes attached
- [ ] Ingress/routes working
- [ ] Application health checks passing
- [ ] Database connectivity verified
- [ ] External integrations working
- [ ] Monitoring/alerting functional
- [ ] Log collection working

### Verification Commands

```bash
# 1. Check namespaces
kubectl get namespaces

# 2. Check pods
kubectl get pods --all-namespaces | grep -v Running

# 3. Check services
kubectl get svc --all-namespaces

# 4. Check ingress
kubectl get ingress --all-namespaces

# 5. Check storage
kubectl get pv
kubectl get pvc --all-namespaces

# 6. Check application health
kubectl get pods -n victor -o json | \
  jq -r '.items[] | select(.status.containerStatuses[].ready != true) | .metadata.name'

# 7. Check metrics
kubectl top nodes
kubectl top pods -n victor

# 8. Check logs
kubectl logs -n victor deployment/victor --tail=50
```

### Application-Specific Tests

```bash
# Example: HTTP health check
curl http://victor.example.com/health

# Example: Database connectivity
kubectl exec -n victor deployment/victor -- \
  psql postgresql://user:pass@db:5432/victor -c "SELECT 1"

# Example: API test
curl -X POST http://victor.example.com/api/test -d '{"test": true}'
```

---

## Emergency Contacts

| Role | Name | Contact | Responsibilities |
|------|------|---------|------------------|
| Site Lead | | | Overall recovery coordination |
| Kubernetes Admin | | | Cluster operations |
| Database Admin | | | Database recovery |
| Network Engineer | | | DNS/LB configuration |
| Security Officer | | | Security verification |
| Product Owner | | | Business impact assessment |

---

## Recovery Metrics

Track these metrics for continuous improvement:

| Metric | Target | Actual |
|--------|--------|--------|
| RTO (Recovery Time Objective) | 4 hours | ___ |
| RPO (Recovery Point Objective) | 24 hours | ___ |
| Backup success rate | 99% | ___ |
| Restore success rate | 100% | ___ |
| Time to detect failure | 15 minutes | ___ |
| Time to initiate recovery | 30 minutes | ___ |

---

## Appendix

### Useful Scripts

#### Monitor Restore Progress

```bash
watch -n 5 'velero restore get -n velero && echo "---" && kubectl get pods --all-namespaces | grep -v Running'
```

#### Cleanup Failed Restores

```bash
velero restore get -n velero -o json | \
  jq -r '.items[] | select(.status.phase != "Completed") | .name' | \
  xargs -I {} velero restore delete {} -n velero --confirm
```

#### Compare Backups

```bash
# Compare two backups
velero backup describe backup1 -n velero --details | grep "Resources Included"
velero backup describe backup2 -n velero --details | grep "Resources Included"
```

---

## Runbook Maintenance

- **Last Updated**: $(date)
- **Last Tested**: $(date)
- **Next Review**: $(date -d "+3 months")

**Change Log**:
- $(date) - Initial runbook creation
- [Add future changes here]
