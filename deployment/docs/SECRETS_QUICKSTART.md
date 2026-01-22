# Secrets Configuration - Quick Start Guide

This guide provides the fastest path to configure secrets for Victor AI.

## Prerequisites

- Kubernetes cluster access
- kubectl configured
- Provider API keys (Anthropic, OpenAI, etc.)

## 30-Second Setup

### 1. Create Environment File

```bash
cp deployment/secrets.env.template .secrets
```

### 2. Edit Required Values

```bash
vim .secrets
```

**Minimum required values:**
- `DB_PASSWORD` - or leave empty to auto-generate
- `ENCRYPTION_KEY` - or leave empty to auto-generate
- `JWT_SECRET` - or leave empty to auto-generate
- At least one provider API key (e.g., `ANTHROPIC_API_KEY`)

### 3. Secure the File

```bash
chmod 600 .secrets
```

### 4. Create Secrets

```bash
./deployment/scripts/configure_secrets.sh
```

### 5. Verify

```bash
./deployment/scripts/verify_secrets.sh
```

That's it! Your secrets are now configured.

## Common Scenarios

### Development Setup

```bash
# Minimal config
cat > .secrets << EOF
DB_PASSWORD=dev123
ANTHROPIC_API_KEY=sk-ant-your-key
EOF

chmod 600 .secrets
./deployment/scripts/configure_secrets.sh
```

### Production with AWS SSM

```bash
# Store in AWS first
aws ssm put-parameter \
  --name "/victor-ai/db-password" \
  --value "secure-password" \
  --type "SecureString"

# Create from SSM
./deployment/scripts/configure_secrets.sh --source aws-ssm
```

### Different Namespaces

```bash
# Staging
./deployment/scripts/configure_secrets.sh --namespace victor-ai-staging

# Production
./deployment/scripts/configure_secrets.sh --namespace victor-ai-prod
```

## Verification

Check that secrets were created:

```bash
# List all secrets
kubectl get secrets -n victor-ai

# Verify specific secret
kubectl describe secret victor-ai-secrets -n victor-ai

# Run verification script
./deployment/scripts/verify_secrets.sh --detailed
```

## Updating Secrets

```bash
# 1. Edit .secrets
vim .secrets

# 2. Re-run configuration
./deployment/scripts/configure_secrets.sh

# 3. Restart pods
kubectl rollout restart deployment victor-ai -n victor-ai
```

## Auto-Generated Secrets

The following are auto-generated if left empty:

| Secret | Length | Example |
|--------|--------|---------|
| DB_PASSWORD | 32 chars | `aB3xK9mP2qL7vN4wR8sT6uY1cF5gHj` |
| ENCRYPTION_KEY | 256-bit | `Base64-encoded-32-bytes` |
| JWT_SECRET | 512-bit | `Base64-encoded-64-bytes` |
| GRAFANA_ADMIN_PASSWORD | 24 chars | `xY7nP2qR9sT4uV8wZ1aB3cD6` |

**Important:** Save auto-generated values when shown! They're only displayed once.

## Security Checklist

Before deploying to production:

- [ ] `.secrets` added to `.gitignore`
- [ ] File permissions set to `chmod 600 .secrets`
- [ ] No secrets committed to git (check: `git log --all --full-history -- "*secrets*"`)
- [ ] Using secrets manager (AWS SSM, Azure Key Vault) for production
- [ ] Passwords are strong (32+ chars, mixed case, numbers, symbols)
- [ ] Different API keys for dev/staging/production
- [ ] Secret rotation policy defined
- [ ] RBAC limits secret access

## Troubleshooting

### "Secret not found"

```bash
# Check if secret exists
kubectl get secrets -n victor-ai

# Re-create
./deployment/scripts/configure_secrets.sh
```

### "Pods not using new secrets"

```bash
# Restart pods
kubectl rollout restart deployment victor-ai -n victor-ai

# Check pod environment
kubectl exec -it deployment/victor-ai -n victor-ai -- env | grep -i secret
```

### "Need to decode secret value"

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

## Next Steps

- **Full Documentation**: `deployment/docs/SECRETS_MANAGEMENT.md`
- **Examples**: `./deployment/scripts/secrets_examples.sh`
- **Script Reference**: `deployment/scripts/README_SECRETS.md`
- **Template**: `deployment/secrets.env.template`

## Quick Reference Commands

```bash
# Create secrets
./deployment/scripts/configure_secrets.sh

# Verify secrets
./deployment/scripts/verify_secrets.sh

# List secrets
kubectl get secrets -n victor-ai

# Describe secret
kubectl describe secret victor-ai-secrets -n victor-ai

# Decode value
kubectl get secret victor-ai-secrets -n victor-ai \
  -o jsonpath='{.data.database-url}' | base64 -d

# Update and restart
vim .secrets
./deployment/scripts/configure_secrets.sh
kubectl rollout restart deployment victor-ai -n victor-ai

# Dry run
./deployment/scripts/configure_secrets.sh --dry-run

# Different namespace
./deployment/scripts/configure_secrets.sh --namespace production

# From AWS SSM
./deployment/scripts/configure_secrets.sh --source aws-ssm

# From Azure Key Vault
./deployment/scripts/configure_secrets.sh --source azure-keyvault
```

## Support

For issues or questions:
1. Check `deployment/docs/SECRETS_MANAGEMENT.md`
2. Run `./deployment/scripts/configure_secrets.sh --help`
3. Run `./deployment/scripts/verify_secrets.sh --detailed`
