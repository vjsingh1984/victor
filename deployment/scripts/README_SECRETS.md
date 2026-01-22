# Secrets Configuration Scripts

Quick reference for secrets management in Victor AI deployments.

## Quick Start

```bash
# 1. Create environment file from template
cp deployment/secrets.env.template .secrets
vim .secrets

# 2. Set secure permissions
chmod 600 .secrets

# 3. Configure secrets in Kubernetes
./deployment/scripts/configure_secrets.sh

# 4. Verify
./deployment/scripts/verify_secrets.sh
```

## Available Scripts

### configure_secrets.sh

Creates and updates Kubernetes secrets from secure sources.

**Features:**
- Creates victor-ai-secrets (database, API keys, encryption)
- Creates alertmanager-secrets (SMTP, Slack, PagerDuty)
- Creates grafana-credentials (admin password)
- Auto-generates secure passwords
- Validates all secrets
- Never logs secret values

**Usage:**
```bash
./configure_secrets.sh [OPTIONS]

Options:
  -n, --namespace NAMESPACE    Kubernetes namespace (default: victor-ai)
  -s, --source SOURCE          Secret source: env, aws-ssm, azure-keyvault
  --dry-run                    Preview changes without applying
  --verify-only                Validate existing secrets only
  -h, --help                   Show help
```

**Examples:**
```bash
# Create from environment
./configure_secrets.sh

# Create from AWS SSM
./configure_secrets.sh --source aws-ssm

# Dry-run
./configure_secrets.sh --dry-run
```

### verify_secrets.sh

Verifies secrets are properly configured (without showing values).

**Usage:**
```bash
./verify_secrets.sh [OPTIONS]

Options:
  -n, --namespace NAMESPACE    Kubernetes namespace
  -d, --detailed               Show detailed information
```

**Examples:**
```bash
# Basic verification
./verify_secrets.sh

# Detailed output
./verify_secrets.sh --detailed
```

## Required Secrets

### victor-ai-secrets
- `database-url` - PostgreSQL connection string
- `encryption-key` - Data encryption (256-bit)
- `jwt-secret` - JWT signing (512-bit)
- Provider API keys (optional)

### alertmanager-secrets (optional)
- `smtp-host`, `smtp-user`, `smtp-password`
- `slack-webhook-url`
- `pagerduty-integration-key`

### grafana-credentials
- `admin-password` - Grafana admin password

## Verification Commands

```bash
# List all secrets
kubectl get secrets -n victor-ai

# Describe secret (metadata only)
kubectl describe secret victor-ai-secrets -n victor-ai

# Decode specific value
kubectl get secret victor-ai-secrets -n victor-ai \
  -o jsonpath='{.data.database-url}' | base64 -d

# Show all keys with lengths
kubectl get secret victor-ai-secrets -n victor-ai \
  -o jsonpath='{.data}' | jq -r 'to_entries[] | "\(.key): \(.value | @base64d | length) bytes"'
```

## Secret Sources

### Environment Variables (default)
```bash
export ANTHROPIC_API_KEY=sk-ant-...
./configure_secrets.sh
```

### AWS SSM Parameter Store
```bash
aws ssm put-parameter \
  --name "/victor-ai/anthropic-api-key" \
  --value "sk-ant-..." \
  --type "SecureString"

./configure_secrets.sh --source aws-ssm
```

### Azure Key Vault
```bash
az keyvault secret set \
  --vault-name victor-ai-kv \
  --name anthropic-api-key \
  --value sk-ant-...

./configure_secrets.sh --source azure-keyvault
```

## Security Checklist

- [ ] `.secrets` file added to `.gitignore`
- [ ] `.secrets` file has `chmod 600` permissions
- [ ] No secrets committed to git
- [ ] Secrets manager configured for production
- [ ] Secret rotation policy defined
- [ ] Audit logging enabled
- [ ] RBAC limits secret access

## Troubleshooting

**Secret not found:**
```bash
kubectl get secrets -n victor-ai
./configure_secrets.sh
```

**Pods not using new secrets:**
```bash
kubectl rollout restart deployment victor-ai -n victor-ai
```

**Decode secret value:**
```bash
kubectl get secret victor-ai-secrets -n victor-ai \
  -o jsonpath='{.data.database-url}' | base64 -d
```

## Documentation

See [deployment/docs/SECRETS_MANAGEMENT.md](../docs/SECRETS_MANAGEMENT.md) for complete documentation.

## Auto-Generated Secrets

The following secrets are auto-generated if not provided:

| Secret | Length | Generation Method |
|--------|--------|-------------------|
| Database password | 32 chars | OpenSSL base64 |
| Encryption key | 256-bit | OpenSSL rand |
| JWT secret | 512-bit | OpenSSL rand 64 |
| Grafana password | 24 chars | OpenSSL base64 |

**Important:** Save auto-generated passwords securely! They are shown once during creation.
