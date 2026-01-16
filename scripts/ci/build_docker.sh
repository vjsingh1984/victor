#!/bin/bash
# Build Docker image for Victor
# Usage: scripts/ci/build_docker.sh [--push] [--tag TAG]

set -e

PUSH=false
TAG="latest"
DOCKERFILE="Dockerfile"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --push)
      PUSH=true
      shift
      ;;
    --tag)
      TAG="$2"
      shift 2
      ;;
    --dockerfile)
      DOCKERFILE="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: $0 [--push] [--tag TAG] [--dockerfile DOCKERFILE]"
      echo ""
      echo "Options:"
      echo "  --push           Push image to registry after building"
      echo "  --tag TAG        Image tag (default: latest)"
      echo "  --dockerfile     Dockerfile path (default: Dockerfile)"
      echo "  --help           Show this help"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Get version from pyproject.toml
VERSION=$(python -c "import tomllib; print(tomllib.load(open('pyproject.toml'))['project']['version'])")

# Image name
IMAGE_NAME="vijayksingh/victor"
FULL_TAG="${IMAGE_NAME}:${TAG}"

echo "Building Docker image..."
echo "  Image: $FULL_TAG"
echo "  Version: $VERSION"
echo "  Dockerfile: $DOCKERFILE"
echo ""

# Build image
docker build \
  --build-arg "VERSION=${VERSION}" \
  --build-arg "BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
  --build-arg "VCS_REF=$(git rev-parse --short HEAD)" \
  -f "$DOCKERFILE" \
  -t "$FULL_TAG" \
  -t "${IMAGE_NAME}:${VERSION}" \
  .

echo ""
echo "✓ Docker image built successfully!"
echo ""
echo "Images:"
echo "  - $FULL_TAG"
echo "  - ${IMAGE_NAME}:${VERSION}"
echo ""

# Push if requested
if [ "$PUSH" = true ]; then
  echo "Pushing images to registry..."
  docker push "$FULL_TAG"
  docker push "${IMAGE_NAME}:${VERSION}"
  echo ""
  echo "✓ Images pushed successfully!"
fi
