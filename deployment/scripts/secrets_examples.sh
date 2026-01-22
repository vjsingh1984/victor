#!/bin/bash
################################################################################
# Examples of using the secrets configuration scripts
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "=== Victor AI Secrets Configuration Examples ==="
echo ""

################################################################################
# Example 1: Basic Setup with Environment Variables
################################################################################

echo "Example 1: Basic Setup"
echo "----------------------"
cat << 'EOF'
# Step 1: Copy template
cp deployment/secrets.env.template .secrets

# Step 2: Edit with your values
vim .secrets

# Step 3: Set secure permissions
chmod 600 .secrets

# Step 4: Create secrets
./deployment/scripts/configure_secrets.sh --namespace victor-ai

# Step 5: Verify
./deployment/scripts/verify_secrets.sh --namespace victor-ai
EOF
echo ""

################################################################################
# Example 2: Using Auto-Generated Secrets
################################################################################

echo "Example 2: Auto-Generated Secrets"
echo "----------------------------------"
cat << 'EOF'
# Only set API keys, let passwords auto-generate
export ANTHROPIC_API_KEY=sk-ant-xxx
export OPENAI_API_KEY=sk-xxx

# Create secrets (passwords will be auto-generated)
./deployment/scripts/configure_secrets.sh

# Save the auto-generated passwords shown in output!
EOF
echo ""

################################################################################
# Example 3: AWS SSM Parameter Store
################################################################################

echo "Example 3: AWS SSM Parameter Store"
echo "-----------------------------------"
cat << 'EOF'
# First, store secrets in SSM
aws ssm put-parameter \
  --name "/victor-ai/database-password" \
  --value "secure-password-here" \
  --type "SecureString" \
  --region us-east-1

aws ssm put-parameter \
  --name "/victor-ai/anthropic-api-key" \
  --value "sk-ant-xxx" \
  --type "SecureString" \
  --region us-east-1

# Create secrets from SSM
export AWS_REGION=us-east-1
./deployment/scripts/configure_secrets.sh \
  --source aws-ssm \
  --namespace production
EOF
echo ""

################################################################################
# Example 4: Azure Key Vault
################################################################################

echo "Example 4: Azure Key Vault"
echo "--------------------------"
cat << 'EOF'
# First, store secrets in Key Vault
az keyvault secret set \
  --vault-name victor-ai-kv \
  --name database-password \
  --value secure-password-here

az keyvault secret set \
  --vault-name victor-ai-kv \
  --name anthropic-api-key \
  --value sk-ant-xxx

# Create secrets from Key Vault
export AZURE_KEY_VAULT=victor-ai-kv
./deployment/scripts/configure_secrets.sh \
  --source azure-keyvault \
  --namespace production
EOF
echo ""

################################################################################
# Example 5: Dry Run (Preview)
################################################################################

echo "Example 5: Dry Run Preview"
echo "--------------------------"
cat << 'EOF'
# Preview what would be created
./deployment/scripts/configure_secrets.sh \
  --dry-run \
  --namespace staging
EOF
echo ""

################################################################################
# Example 6: Update Existing Secrets
################################################################################

echo "Example 6: Update Existing Secrets"
echo "----------------------------------"
cat << 'EOF'
# Update .secrets file with new values
vim .secrets

# Re-run configuration (updates existing secrets)
./deployment/scripts/configure_secrets.sh

# Restart pods to pick up new secrets
kubectl rollout restart deployment victor-ai -n victor-ai
EOF
echo ""

################################################################################
# Example 7: Verification Commands
################################################################################

echo "Example 7: Verification Commands"
echo "--------------------------------"
cat << 'EOF'
# Basic verification
./deployment/scripts/verify_secrets.sh

# Detailed verification
./deployment/scripts/verify_secrets.sh --detailed

# Production namespace
./deployment/scripts/verify_secrets.sh --namespace production

# Check if secret exists
kubectl get secret victor-ai-secrets -n victor-ai

# Describe secret (metadata only)
kubectl describe secret victor-ai-secrets -n victor-ai

# List all secrets
kubectl get secrets -n victor-ai
EOF
echo ""

################################################################################
# Example 8: Decode Specific Secret Values
################################################################################

echo "Example 8: Decode Secret Values"
echo "-------------------------------"
cat << 'EOF'
# Decode database URL
kubectl get secret victor-ai-secrets -n victor-ai \
  -o jsonpath='{.data.database-url}' | base64 -d

# Decode encryption key
kubectl get secret victor-ai-secrets -n victor-ai \
  -o jsonpath='{.data.encryption-key}' | base64 -d

# Decode Grafana password
kubectl get secret grafana-credentials -n victor-ai \
  -o jsonpath='{.data.admin-password}' | base64 -d

# Show all keys with lengths
kubectl get secret victor-ai-secrets -n victor-ai \
  -o jsonpath='{.data}' | jq -r 'to_entries[] | "\(.key): \(.value | @base64d | length) bytes"'
EOF
echo ""

################################################################################
# Example 9: Manual Secret Creation
################################################################################

echo "Example 9: Manual Secret Creation"
echo "---------------------------------"
cat << 'EOF'
# Create individual secret manually
kubectl create secret generic victor-ai-secrets \
  --namespace=victor-ai \
  --from-literal=database-url='postgresql://user:pass@host:5432/db' \
  --from-literal=encryption-key='base64-encoded-key' \
  --from-literal=jwt-secret='base64-encoded-secret'

# Or from file
kubectl create secret generic victor-ai-secrets \
  --namespace=victor-ai \
  --from-env-file=.secrets
EOF
echo ""

################################################################################
# Example 10: Secret Rotation
################################################################################

echo "Example 10: Secret Rotation"
echo "---------------------------"
cat << 'EOF'
# Generate new passwords
NEW_DB_PASS=$(openssl rand -base64 32)
NEW_ENCRYPTION_KEY=$(openssl rand -base64 32)
NEW_JWT_SECRET=$(openssl rand -base64 64)

# Update .secrets
sed -i "s/^DB_PASSWORD=.*/DB_PASSWORD=$NEW_DB_PASS/" .secrets
sed -i "s/^ENCRYPTION_KEY=.*/ENCRYPTION_KEY=$NEW_ENCRYPTION_KEY/" .secrets
sed -i "s/^JWT_SECRET=.*/JWT_SECRET=$NEW_JWT_SECRET/" .secrets

# Re-create secrets
./deployment/scripts/configure_secrets.sh

# Restart pods
kubectl rollout restart deployment victor-ai -n victor-ai

# Verify
./deployment/scripts/verify_secrets.sh
EOF
echo ""

################################################################################
# Example 11: Multi-Environment Setup
################################################################################

echo "Example 11: Multi-Environment"
echo "------------------------------"
cat << 'EOF'
# Development
cp deployment/secrets.env.template .secrets.dev
vim .secrets.dev
export NAMESPACE=victor-ai-dev
./deployment/scripts/configure_secrets.sh \
  --namespace victor-ai-dev

# Staging
cp deployment/secrets.env.template .secrets.staging
vim .secrets.staging
export NAMESPACE=victor-ai-staging
./deployment/scripts/configure_secrets.sh \
  --namespace victor-ai-staging

# Production (use AWS SSM)
export NAMESPACE=victor-ai-prod
./deployment/scripts/configure_secrets.sh \
  --source aws-ssm \
  --namespace victor-ai-prod
EOF
echo ""

################################################################################
# Example 12: Backup and Restore
################################################################################

echo "Example 12: Backup and Restore"
echo "------------------------------"
cat << 'EOF'
# Backup all secrets (encrypt before storing!)
kubectl get secrets -n victor-ai -o yaml > secrets-backup.yaml
gpg --encrypt --recipient you@example.com secrets-backup.yaml
shred -u secrets-backup.yaml

# Restore
gpg --decrypt secrets-backup.yaml.gpg > secrets-backup.yaml
kubectl apply -f secrets-backup.yaml
shred -u secrets-backup.yaml
EOF
echo ""

################################################################################
# Example 13: Troubleshooting
################################################################################

echo "Example 13: Troubleshooting"
echo "---------------------------"
cat << 'EOF'
# Check secret exists
kubectl get secret victor-ai-secrets -n victor-ai

# Check pod has access to secret
kubectl describe pod -l app=victor-ai -n victor-ai

# Check pod environment variables
kubectl exec -it deployment/victor-ai -n victor-ai -- env | grep -i secret

# Restart pods to pick up secrets
kubectl rollout restart deployment victor-ai -n victor-ai

# Check pod logs for secret-related errors
kubectl logs -f deployment/victor-ai -n victor-ai
EOF
echo ""

################################################################################
# Example 14: Production Best Practices
################################################################################

echo "Example 14: Production Best Practices"
echo "-------------------------------------"
cat << 'EOF'
# 1. Use sealed-secrets for GitOps
kubeseal -f deployment/kubernetes/base/secret.yaml \
  -w deployment/kubernetes/base/sealed-secret.yaml

# 2. Enable audit logging
kubectl audit log --namespace=victor-ai

# 3. Limit secret access with RBAC
cat << 'RBAC' | kubectl apply -f -
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: secret-reader
  namespace: victor-ai
rules:
- apiGroups: [""]
  resources: ["secrets"]
  verbs: ["get"]
RBAC

# 4. Use external secrets operator
# (see deployment/docs/SECRETS_MANAGEMENT.md)

# 5. Regular secret rotation (monthly cron)
0 0 1 * * /path/to/rotate_secrets.sh
EOF
echo ""

################################################################################
# Example 15: Quick Start for Development
################################################################################

echo "Example 15: Quick Development Setup"
echo "-----------------------------------"
cat << 'EOF'
# One-command setup for development
cat << 'ENV' > .secrets
DB_PASSWORD=dev-password-123
ANTHROPIC_API_KEY=sk-ant-your-key-here
ENV

chmod 600 .secrets
./deployment/scripts/configure_secrets.sh
kubectl rollout restart deployment victor-ai -n victor-ai
EOF
echo ""

echo "=== Examples Complete ==="
echo ""
echo "For more information, see:"
echo "  - deployment/docs/SECRETS_MANAGEMENT.md"
echo "  - deployment/scripts/README_SECRETS.md"
echo "  - deployment/secrets.env.template"
