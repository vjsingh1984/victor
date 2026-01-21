#!/bin/bash
# Victor AI Release Automation Script
# Usage: ./scripts/release.sh [version] [--dry-run] [--skip-upload]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse arguments
VERSION=""
DRY_RUN=false
SKIP_UPLOAD=false
SKIP_TESTS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-upload)
            SKIP_UPLOAD=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        -*)
            log_error "Unknown option: $1"
            exit 1
            ;;
        *)
            VERSION="$1"
            shift
            ;;
    esac
done

# Validate version
if [ -z "$VERSION" ]; then
    log_error "Version is required. Usage: $0 [version] [--dry-run] [--skip-upload]"
    exit 1
fi

# Validate version format (semantic versioning)
if [[ ! $VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    log_error "Invalid version format: $VERSION (expected X.Y.Z)"
    exit 1
fi

log_info "Starting release process for version $VERSION"
if [ "$DRY_RUN" = true ]; then
    log_warning "DRY RUN MODE - No actual changes will be made"
fi

# Check prerequisites
log_info "Checking prerequisites..."

# Check if git repo is clean
if [ -n "$(git status --porcelain)" ]; then
    log_error "Git repository is not clean. Please commit or stash changes first."
    exit 1
fi

# Check if on main branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" != "main" ]; then
    log_warning "Not on main branch (currently on $CURRENT_BRANCH)"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Aborting release"
        exit 1
    fi
fi

# Check if required tools are installed
if ! command -v python &> /dev/null; then
    log_error "Python is required but not installed"
    exit 1
fi

if ! command -v git &> /dev/null; then
    log_error "Git is required but not installed"
    exit 1
fi

log_success "Prerequisites check passed"

# Run tests if not skipped
if [ "$SKIP_TESTS" = false ]; then
    log_info "Running tests..."

    if [ "$DRY_RUN" = false ]; then
        # Run unit tests only (fast)
        log_info "Running unit tests..."
        pytest tests/unit -v -m "not slow" || {
            log_error "Unit tests failed. Aborting release."
            exit 1
        }

        log_success "Unit tests passed"
    else
        log_warning "Skipping tests in dry-run mode"
    fi
else
    log_warning "Skipping tests (--skip-tests flag)"
fi

# Update version in pyproject.toml
log_info "Updating version in pyproject.toml to $VERSION..."

if [ "$DRY_RUN" = false ]; then
    # Update version using sed
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s/^version = \".*\"/version = \"$VERSION\"/" pyproject.toml
    else
        # Linux
        sed -i "s/^version = \".*\"/version = \"$VERSION\"/" pyproject.toml
    fi

    # Update VERSION file
    cat > VERSION << EOF
$VERSION
build: $(git rev-parse HEAD)
date: $(date +%Y-%m-%d)
stable: true
EOF

    log_success "Version updated"
else
    log_warning "Would update version in dry-run mode"
fi

# Build distribution packages
log_info "Building distribution packages..."

if [ "$DRY_RUN" = false ]; then
    # Clean previous builds
    rm -rf dist/ build/ *.egg-info

    # Build source distribution and wheel
    python -m build

    # Check if build was successful
    if [ ! -d "dist" ] || [ -z "$(ls -A dist)" ]; then
        log_error "Build failed - no distribution files found"
        exit 1
    fi

    log_success "Distribution packages built:"
    ls -lh dist/
else
    log_warning "Would build packages in dry-run mode"
fi

# Generate checksums
log_info "Generating checksums..."

if [ "$DRY_RUN" = false ]; then
    python scripts/create_checksums.py
    log_success "Checksums generated in SHA256SUMS"
else
    log_warning "Would generate checksums in dry-run mode"
fi

# Create git tag
log_info "Creating git tag v$VERSION..."

if [ "$DRY_RUN" = false ]; then
    # Create annotated tag
    git tag -a "v$VERSION" -m "Release $VERSION: Production-ready AI coding assistant

Key features:
- 21 LLM provider support
- 55+ specialized tools across 5 verticals
- SOLID architecture with 98 protocols
- Comprehensive security suite (132 tests)
- 72.8% faster startup with lazy loading
- Full backward compatibility with 0.5.x

See CHANGELOG.md for full release notes."

    log_success "Git tag v$VERSION created"
else
    log_warning "Would create git tag in dry-run mode"
fi

# Upload to PyPI (if not skipped)
if [ "$SKIP_UPLOAD" = false ]; then
    log_info "Uploading to PyPI..."

    if [ "$DRY_RUN" = false ]; then
        # Check if twine is installed
        if ! command -v twine &> /dev/null; then
            log_warning "Twine not installed. Install with: pip install twine"
            log_info "Skipping PyPI upload"
        else
            # Upload to PyPI
            twine upload dist/*

            log_success "Uploaded to PyPI"
        fi
    else
        log_warning "Would upload to PyPI in dry-run mode"
    fi
else
    log_warning "Skipping PyPI upload (--skip-upload flag)"
fi

# Push git tag (if not dry run)
if [ "$DRY_RUN" = false ]; then
    log_info "Pushing git tag to remote..."

    read -p "Push tag to origin? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git push origin "v$VERSION"
        log_success "Git tag pushed to origin"
    else
        log_warning "Skipping git push"
    fi
fi

# Summary
log_info "Release summary:"
echo ""
echo "Version: $VERSION"
echo "Tag: v$VERSION"
echo "Distribution files:"
if [ "$DRY_RUN" = false ]; then
    ls -lh dist/ 2>/dev/null || echo "  No distribution files found"
    if [ -f "SHA256SUMS" ]; then
        echo "Checksums: SHA256SUMS"
        cat SHA256SUMS
    fi
fi
echo ""

if [ "$DRY_RUN" = true ]; then
    log_warning "DRY RUN COMPLETE - No actual changes were made"
else
    log_success "Release $VERSION complete!"

    echo ""
    echo "Next steps:"
    echo "1. Verify the release on GitHub: https://github.com/vijayksingh/victor/releases"
    echo "2. Announce the release"
    echo "3. Update documentation with new version"
    echo "4. Monitor issues and pull requests"
fi
