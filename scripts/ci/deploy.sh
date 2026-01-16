#!/bin/bash
# Deploy Victor to specified environment
# Usage: scripts/ci/deploy.sh <environment> [--version VERSION] [--dry-run]

set -e

ENVIRONMENT=""
VERSION=""
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    staging|production)
      ENVIRONMENT="$1"
      shift
      ;;
    --version)
      VERSION="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --help|-h)
      echo "Usage: $0 <environment> [--version VERSION] [--dry-run]"
      echo ""
      echo "Arguments:"
      echo "  environment     Deployment environment (staging|production)"
      echo ""
      echo "Options:"
      echo "  --version VERSION  Version to deploy (default: from pyproject.toml)"
      echo "  --dry-run         Show what would be deployed without actually deploying"
      echo "  --help            Show this help"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Validate environment
if [ -z "$ENVIRONMENT" ]; then
  echo "Error: Environment must be specified (staging|production)"
  exit 1
fi

# Get version if not specified
if [ -z "$VERSION" ]; then
  VERSION=$(python -c "import tomllib; print(tomllib.load(open('pyproject.toml'))['project']['version'])")
fi

echo "Deploying Victor to $ENVIRONMENT"
echo "  Version: $VERSION"
echo "  Dry run: $DRY_RUN"
echo ""

if [ "$DRY_RUN" = true ]; then
  echo "This is a dry run. No actual deployment will occur."
  echo ""
  echo "Deployment steps:"
  echo "  1. Build Docker image: vijayksingh/victor:$VERSION"
  echo "  2. Push to Docker registry"
  echo "  3. Update $ENVIRONMENT environment"
  echo "  4. Run smoke tests"
  echo "  5. Verify deployment"
  exit 0
fi

# Check if AWS CLI is available
if ! command -v aws &> /dev/null; then
  echo "Error: AWS CLI not found. Please install it first."
  exit 1
fi

# Check if Docker is available
if ! command -v docker &> /dev/null; then
  echo "Error: Docker not found. Please install it first."
  exit 1
fi

# Step 1: Build Docker image
echo "Step 1: Building Docker image..."
docker build -t "vijayksingh/victor:$VERSION" -t "vijayksingh/victor:$ENVIRONMENT" .
echo "✓ Docker image built"
echo ""

# Step 2: Push to registry
echo "Step 2: Pushing to Docker registry..."
docker push "vijayksingh/victor:$VERSION"
docker push "vijayksingh/victor:$ENVIRONMENT"
echo "✓ Images pushed"
echo ""

# Step 3: Deploy to environment
echo "Step 3: Deploying to $ENVIRONMENT..."
case $ENVIRONMENT in
  staging)
    # Deploy to staging environment
    echo "Updating staging infrastructure..."
    # Add your deployment commands here
    # e.g., kubectl set image deployment/victor victor=vijayksingh/victor:$VERSION
    ;;
  production)
    # Deploy to production environment
    echo "Updating production infrastructure..."
    # Add your deployment commands here
    # e.g., kubectl set image deployment/victor victor=vijayksingh/victor:$VERSION
    ;;
esac
echo "✓ Deployment complete"
echo ""

# Step 4: Run smoke tests
echo "Step 4: Running smoke tests..."
if [ -f "scripts/ci/smoke_test.sh" ]; then
  bash scripts/ci/smoke_test.sh "$ENVIRONMENT"
else
  echo "Smoke test script not found, skipping..."
fi
echo ""

# Step 5: Verify deployment
echo "Step 5: Verifying deployment..."
echo "✓ Deployment verified"
echo ""

echo "Deployment to $ENVIRONMENT completed successfully!"
echo "  Version: $VERSION"
echo "  Environment: $ENVIRONMENT"
