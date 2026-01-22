# Secrets Management Guide

This guide covers secure secrets configuration and management for Victor AI deployments.

## Overview

Victor AI uses Kubernetes Secrets to store sensitive configuration data including:
- Database credentials
- Redis credentials
- Provider API keys
- Encryption keys
- JWT secrets
- Alerting service credentials
- Grafana credentials

## Quick Start

### 1. Initial Setup

```bash
# Copy the template
cp deployment/secrets.env.template .secrets

# Edit with your actual values
vim .secrets

# Set secure permissions
chmod 600 .secrets

# Create secrets in Kubernetes
./deployment/scripts/configure_secrets.sh --namespace victor-ai
```

### 2. Verify Secrets

```bash
./deployment/scripts/verify_secrets.sh --namespace victor-ai
```

## Script: configure_secrets.sh

### Features

- **Multiple Sources**: Environment variables, AWS SSM Parameter Store, Azure Key Vault
- **Auto-Generation**: Automatically generates secure passwords and encryption keys
- **Validation**: Verifies all secrets are properly created
- **Dry-Run**: Preview changes without applying them
- **Security**: Never logs secret values

### Usage

```bash
./deployment/scripts/configure_secrets.sh [OPTIONS]

Options:
  -n, --namespace NAMESPACE       Kubernetes namespace (default: victor-ai)
  -s, --source SOURCE             Secret source: env, aws-ssm, azure-keyvault
  -c, --context CONTEXT           Kubernetes context to use
  --dry-run                       Show what would be done without making changes
  --verify-only                   Only validate existing secrets
  --skip-validation               Skip secret validation
  --no-auto-generate              Don't auto-generate missing secrets
  -h, --help                      Show help message
```

### Examples

```bash
# Create secrets from environment variables
./deployment/scripts/configure_secrets.sh --namespace production

# Create secrets from AWS SSM Parameter Store
export AWS_REGION=us-east-1
./deployment/scripts/configure_secrets.sh --source aws-ssm --namespace production

# Dry-run to preview changes
./deployment/scripts/configure_secrets.sh --dry-run

# Validate existing secrets only
./deployment/scripts/configure_secrets.sh --verify-only

# Use custom Kubernetes context
./deployment/scripts/configure_secrets.sh --context production-cluster
```

## Script: verify_secrets.sh

### Features

- **Secret Existence**: Checks if all required secrets exist
- **Required Keys**: Verifies all required keys are present
- **Security Checks**: Detects weak passwords and placeholder values
- **Age Analysis**: Warns about old secrets that need rotation
- **No Value Exposure**: Shows secret status without revealing values

### Usage

```bash
./deployment/scripts/verify_secrets.sh [OPTIONS]

Options:
  -n, --namespace NAMESPACE    Kubernetes namespace (default: victor-ai)
  -d, --detailed               Show detailed secret information
  -h, --help                   Show help message
```

### Examples

```bash
# Basic verification
./deployment/scripts/verify_secrets.sh

# Detailed output
./deployment/scripts/verify_secrets.sh --detailed

# Specific namespace
./deployment/scripts/verify_secrets.sh --namespace production
```

## Environment Variables

### Required

| Variable | Description | Default | Auto-Generated |
|----------|-------------|---------|----------------|
| `DB_PASSWORD` | Database password | - | Yes (32 chars) |
| `ENCRYPTION_KEY` | Data at rest encryption | - | Yes (256-bit) |
| `JWT_SECRET` | JWT signing secret | - | Yes (512-bit) |

### Optional - Provider API Keys

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Anthropic Claude API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `GOOGLE_API_KEY` | Google Gemini API key |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |
| `CEREBRAS_API_KEY` | Cerebras API key |
| `DEEPSEEK_API_KEY` | DeepSeek API key |
| `FIREWORKS_API_KEY` | Fireworks AI API key |
| `GROQ_API_KEY` | Groq API key |
| `MISTRAL_API_KEY` | Mistral AI API key |
| `MOONSHOT_API_KEY` | Moonshot API key |
| `TOGETHER_API_KEY` | Together AI API key |
| `REPLICATE_API_KEY` | Replicate API key |
| `XAI_API_KEY` | X.AI API key |

### Optional - Database

| Variable | Description | Default |
|----------|-------------|---------|
| `DB_HOST` | Database host | victor-postgres |
| `DB_PORT` | Database port | 5432 |
| `DB_NAME` | Database name | victor |
| `DB_USER` | Database user | victor |

### Optional - Redis

| Variable | Description | Default |
|----------|-------------|---------|
| `REDIS_HOST` | Redis host | victor-redis |
| `REDIS_PORT` | Redis port | 6379 |
| `REDIS_PASSWORD` | Redis password | - |

### Optional - Alerting

| Variable | Description |
|----------|-------------|
| `SMTP_HOST` | SMTP server host |
| `SMTP_PORT` | SMTP server port (default: 587) |
| `SMTP_USER` | SMTP username |
| `SMTP_PASSWORD` | SMTP password |
| `SMTP_FROM` | From email address |
| `SLACK_WEBHOOK_URL` | Slack webhook URL |
| `SLACK_API_TOKEN` | Slack API token |
| `PAGERDUTY_INTEGRATION_KEY` | PagerDuty integration key |

### Optional - Monitoring

| Variable | Description | Auto-Generated |
|----------|-------------|----------------|
| `GRAFANA_ADMIN_PASSWORD` | Grafana admin password | Yes (24 chars) |
| `SENTRY_DSN` | Sentry DSN | - |
| `DATADOG_API_KEY` | Datadog API key | - |

## Secret Sources

### Environment Variables (default)

Reads from shell environment or `.secrets` file:

```bash
# From environment
export ANTHROPIC_API_KEY=sk-ant-...
./configure_secrets.sh

# From .secrets file
echo "ANTHROPIC_API_KEY=sk-ant-..." > .secrets
./configure_secrets.sh
```

### AWS SSM Parameter Store

Stores secrets in AWS Systems Manager Parameter Store:

```bash
# Store parameter
aws ssm put-parameter \
  --name "/victor-ai/anthropic-api-key" \
  --value "sk-ant-..." \
  --type "SecureString" \
  --region us-east-1

# Use in script
./configure_secrets.sh --source aws-ssm
```

**Parameter naming convention:**
- `/victor-ai/database-password`
- `/victor-ai/anthropic-api-key`
- `/victor-ai/encryption-key`
- etc.

### Azure Key Vault

Stores secrets in Azure Key Vault:

```bash
# Store secret
az keyvault secret set \
  --vault-name victor-ai-kv \
  --name anthropic-api-key \
  --value sk-ant-...

# Use in script
export AZURE_KEY_VAULT=victor-ai-kv
./configure_secrets.sh --source azure-keyvault
```

## Security Best Practices

### 1. Never Commit Secrets

```bash
# Add .secrets to .gitignore
echo ".secrets" >> .gitignore

# Remove accidentally committed secrets
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch .secrets" \
  --prune-empty --tag-name-filter cat -- --all
```

### 2. Secure File Permissions

```bash
# Restrict .secrets file
chmod 600 .secrets

# Verify
ls -la .secrets
# Should show: -rw------- (600)
```

### 3. Rotate Secrets Regularly

```bash
# Generate new passwords
openssl rand -base64 32

# Update .secrets file
vim .secrets

# Re-create secrets
./configure_secrets.sh

# Restart pods to pick up new secrets
kubectl rollout restart deployment victor-ai -n victor-ai
```

### 4. Use Secrets Managers for Production

- **AWS**: AWS Secrets Manager or SSM Parameter Store
- **Azure**: Azure Key Vault
- **GCP**: Secret Manager
- **Kubernetes**: External Secrets Operator or Sealed Secrets

### 5. Enable Audit Logging

```bash
# Kubernetes audit logging
kubectl logs -n kube-system -l component=kube-apiserver | grep secret

# AWS CloudTrail (for SSM access)
aws cloudtrail lookup-events --lookup-attributes AttributeKey=ResourceName,AttributeValue=/victor-ai/
```

### 6. Limit Secret Access

```bash
# Create RBAC role for secret access
cat <<EOF | kubectl apply -f -
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: secret-reader
  namespace: victor-ai
rules:
- apiGroups: [""]
  resources: ["secrets"]
  verbs: ["get"]
EOF
```

## GitOps Integration

### SealedSecrets (Recommended)

Encrypt secrets for Git storage:

```bash
# Install kubeseal
kubectl kustomize github.com/bitnami-labs/sealed-secrets?ref=v0.24.0 | kubectl apply -f -

# Create sealed secret
kubeseal -f deployment/kubernetes/base/secret.yaml \
  -w deployment/kubernetes/base/sealed-secret.yaml

# Commit sealed secret (safe!)
git add deployment/kubernetes/base/sealed-secret.yaml
git commit -m "Add sealed secrets"
```

### External Secrets Operator

Sync secrets from external stores:

```yaml
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: aws-secrets-manager
  namespace: victor-ai
spec:
  provider:
    aws:
      service: SecretsManager
      region: us-east-1
      auth:
        jwt:
          serviceAccountRef:
            name: external-secrets-sa
---
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: victor-ai-secrets
  namespace: victor-ai
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secrets-manager
    kind: SecretStore
  target:
    name: victor-ai-secrets
    creationPolicy: Owner
  data:
    - secretKey: database-url
      remoteRef:
        key: victor-ai/database-url
```

## Troubleshooting

### Secret Not Found

```bash
# Check if secret exists
kubectl get secrets -n victor-ai

# Describe secret (without showing values)
kubectl describe secret victor-ai-secrets -n victor-ai

# Re-create secret
./configure_secrets.sh --namespace victor-ai
```

### Incorrect Secret Value

```bash
# Decode and check (without logging)
kubectl get secret victor-ai-secrets -n victor-ai \
  -o jsonpath='{.data.database-url}' | base64 -d | less

# Update specific key
kubectl create secret generic victor-ai-secrets \
  --namespace=victor-ai \
  --from-literal=database-url='postgresql://user:pass@host:5432/db' \
  --dry-run=client -o yaml | kubectl apply -f -
```

### Pods Not Picking Up Secrets

```bash
# Check pod environment
kubectl exec -it deployment/victor-ai -n victor-ai -- env | grep -i secret

# Restart pods
kubectl rollout restart deployment victor-ai -n victor-ai

# Check pod events
kubectl describe pod -l app=victor-ai -n victor-ai
```

### Auto-Generated Passwords Lost

```bash
# Re-generate and display (secure!)
./configure_secrets.sh --dry-run | grep -A 1 "Auto-generated"

# Or decode existing secret
kubectl get secret victor-ai-secrets -n victor-ai \
  -o jsonpath='{.data.database-url}' | base64 -d | grep -oP '://\K[^:]*' | head -1
```

## Verification Commands

### List All Secrets

```bash
kubectl get secrets -n victor-ai
```

### Describe Secret

```bash
kubectl describe secret victor-ai-secrets -n victor-ai
```

### Decode Specific Value

```bash
# Database URL
kubectl get secret victor-ai-secrets -n victor-ai \
  -o jsonpath='{.data.database-url}' | base64 -d

# Encryption key
kubectl get secret victor-ai-secrets -n victor-ai \
  -o jsonpath='{.data.encryption-key}' | base64 -d

# Grafana password
kubectl get secret grafana-credentials -n victor-ai \
  -o jsonpath='{.data.admin-password}' | base64 -d
```

### Show All Keys (with lengths)

```bash
kubectl get secret victor-ai-secrets -n victor-ai \
  -o jsonpath='{.data}' | jq -r 'to_entries[] | "\(.key): \(.value | @base64d | length) bytes"'
```

### Export Secrets (Caution!)

```bash
# Export to YAML (DO NOT COMMIT!)
kubectl get secrets -n victor-ai -o yaml > secrets-backup.yaml

# Encrypt backup
gpg --encrypt --recipient you@example.com secrets-backup.yaml
rm secrets-backup.yaml
```

## Automated Secret Rotation

### Monthly Rotation Script

```bash
#!/bin/bash
# rotate_secrets.sh - Monthly secret rotation

NAMESPACE="victor-ai"
SECRETS_FILE=".secrets"

# Generate new passwords
new_db_password=$(openssl rand -base64 32)
new_encryption_key=$(openssl rand -base64 32)
new_jwt_secret=$(openssl rand -base64 64)

# Update .secrets
sed -i "s/^DB_PASSWORD=.*/DB_PASSWORD=$new_db_password/" $SECRETS_FILE
sed -i "s/^ENCRYPTION_KEY=.*/ENCRYPTION_KEY=$new_encryption_key/" $SECRETS_FILE
sed -i "s/^JWT_SECRET=.*/JWT_SECRET=$new_jwt_secret/" $SECRETS_FILE

# Re-create secrets
./deployment/scripts/configure_secrets.sh --namespace $NAMESPACE

# Restart pods
kubectl rollout restart deployment victor-ai -n $NAMESPACE

echo "Secrets rotated successfully"
```

### Schedule with Cron

```bash
# Add to crontab
0 0 1 * * /path/to/rotate_secrets.sh >> /var/log/secret-rotation.log 2>&1
```

## Monitoring and Alerts

### Prometheus Alerts

```yaml
groups:
  - name: secrets
    rules:
      - alert: SecretMissing
        expr: |
          kube_secret_info{namespace="victor-ai",secret="victor-ai-secrets"} == 0
        for: 5m
        annotations:
          summary: "Required secret is missing"

      - alert: SecretStale
        expr: |
          time() - kube_secret_created{namespace="victor-ai"} > 7776000
        for: 1h
        annotations:
          summary: "Secret is older than 90 days"
```

## References

- [Kubernetes Secrets Documentation](https://kubernetes.io/docs/concepts/configuration/secret/)
- [SealedSecrets](https://github.com/bitnami-labs/sealed-secrets)
- [External Secrets Operator](https://external-secrets.io/)
- [AWS Secrets Manager](https://aws.amazon.com/secrets-manager/)
- [Azure Key Vault](https://azure.microsoft.com/services/key-vault/)
