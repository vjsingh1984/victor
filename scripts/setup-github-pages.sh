#!/bin/bash
# Setup GitHub Pages for Victor Documentation
# This script configures GitHub Pages using gh CLI

set -e

echo "🚀 Setting up GitHub Pages for Victor documentation..."
echo

# Check if gh is installed
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) is not installed."
    echo "Install it from: https://cli.github.com/"
    exit 1
fi

# Check authentication
echo "📋 Checking GitHub authentication..."
if ! gh auth status &> /dev/null; then
    echo "❌ Not authenticated with GitHub."
    echo "Please run: gh auth login"
    exit 1
fi

echo "✅ Authenticated as: $(gh auth status --show-token 2>&1 | head -1)"
echo

# Get repository info
REPO=$(git remote get-url origin | sed 's/.*:\(.*\)\.git/\1/')
echo "📦 Repository: $REPO"
echo

# Check current Pages status
echo "🔍 Checking current Pages status..."
PAGES_STATUS=$(gh api repos/$REPO/pages --jq '.source' 2>/dev/null || echo "null")

if [ "$PAGES_STATUS" = "null" ]; then
    echo "⚠️  GitHub Pages not enabled yet"
    echo
    echo "📝 To enable GitHub Pages, choose an option:"
    echo "  1) Automatic via GitHub Actions (recommended)"
    echo "  2) Manual deployment via mkdocs gh-deploy"
    echo
    read -p "Choose option (1 or 2): " choice

    if [ "$choice" = "1" ]; then
        echo "⚙️  Configuring GitHub Pages with GitHub Actions..."

        # Enable GitHub Actions as source
        gh api \
            --method POST \
            -H "Accept: application/vnd.github+json" \
            repos/$REPO/pages \
            -f build_type='workflow' \
            -f source[branch]=main \
            -f source[path]='/workflow' || {
            echo "⚠️  Could not configure via API. Setting up manually..."
            echo "Please go to: https://github.com/$REPO/settings/pages"
            echo "Select 'GitHub Actions' as the source."
        }

        echo
        echo "✅ GitHub Pages configured!"
        echo "   Documentation will auto-deploy on push to main branch."

    elif [ "$choice" = "2" ]; then
        echo "📝 Manual deployment setup"
        echo "To deploy manually:"
        echo "  1. Build: mkdocs build"
        echo "  2. Deploy: mkdocs gh-deploy"
        echo
        echo "Or create a GitHub Actions workflow:"
        echo "  File: .github/workflows/docs.yml"
        echo "  See docs/DEPLOYMENT.md for details."
    fi

else
    echo "✅ GitHub Pages already enabled!"
    echo "   Source: $PAGES_STATUS"
    echo "   View at: https://$(gh api repos/$REPO --jq '.name').github.com/victor/"
fi

echo
echo "📋 Next steps:"
echo "  1. Push changes to main: git push origin main"
echo "  2. Wait for workflow to complete (check Actions tab)"
echo "  3. Visit: https://$(gh api repos/$REPO --jq '.owner.login').github.io/victor/"
echo

# Check workflow permissions
echo "🔒 Checking workflow permissions..."
WORKFLOW_PERMS=$(gh api repos/$REPO/actions/permissions/workflow --jq '.default_workflow_permissions' 2>/dev/null || echo "null")

if [ "$WORKFLOW_PERMS" != "write" ]; then
    echo "⚠️  Workflow permissions may need updating:"
    echo "   Go to: https://github.com/$REPO/settings/actions"
    echo "   Set 'Workflow permissions' to 'Read and write permissions'"
fi

echo
echo "✅ Setup complete!"
