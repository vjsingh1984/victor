# GitHub Pages Deployment Guide

This guide explains how to set up and deploy Victor documentation to GitHub Pages.

## Initial Setup

### 1. Enable GitHub Pages

1. Go to your repository on GitHub
2. Click **Settings** > **Pages**
3. Under **Source**, select **GitHub Actions** (recommended)
4. Click **Save**

### 2. Configure Workflow Permissions

1. Go to **Settings** > **Actions** > **General**
2. Scroll to **Workflow permissions**
3. Select **Read and write permissions**
4. Check **Allow GitHub Actions to create and approve pull requests**
5. Click **Save**

### 3. Verify the Workflow

The documentation deployment workflow is located at:
```
.github/workflows/docs.yml
```

This workflow will:
- Trigger on push to `main` branch when docs change
- Build the documentation with MkDocs
- Deploy to GitHub Pages

## Deployment Process

### Automatic Deployment

Documentation deploys automatically when you push to the `main` branch:

```bash
# Make changes to documentation
git add docs/
git commit -m "docs: update installation guide"
git push origin main
```

The GitHub Actions workflow will:
1. Detect changes in `docs/` or `mkdocs.yml`
2. Build the documentation
3. Deploy to GitHub Pages
4. Publish to: `https://vjsingh1984.github.io/victor/`

### Manual Deployment

To trigger a manual deployment:

1. Go to **Actions** tab in your GitHub repository
2. Select **Deploy Documentation** workflow
3. Click **Run workflow**
4. Select `main` branch
5. Click **Run workflow** button

## Testing Locally

Before pushing changes, test the documentation locally:

### Option 1: Using the helper script

```bash
# From project root
./scripts/docs-serve.sh
```

### Option 2: Using mkdocs directly

```bash
# Install dependencies
pip install -e ".[docs]"

# Serve documentation
mkdocs serve
```

Then open `http://127.0.0.1:8000` in your browser.

### Option 3: Build and serve with Python

```bash
# Build documentation
./scripts/docs-build.sh

# Or manually
mkdocs build --clean

# Serve with Python's HTTP server
python3 -m http.server 8000 --directory site
```

## Custom Domain Setup

To use a custom domain (e.g., `docs.victor-ai.com`):

### 1. Create CNAME File

Create `docs/CNAME` with your domain:

```bash
echo "docs.victor-ai.com" > docs/CNAME
git add docs/CNAME
git commit -m "docs: add custom domain CNAME"
git push
```

### 2. Configure DNS

Add DNS records with your domain provider:

| Type | Name | Value |
|------|------|-------|
| A | `docs` (or `@`) | `185.199.108.153` |
| A | `docs` (or `@`) | `185.199.109.153` |
| A | `docs` (or `@`) | `185.199.110.153` |
| A | `docs` (or `@`) | `185.199.111.153` |
| CNAME | `www` | `vjsingh1984.github.io` |

### 3. Update mkdocs.yml

Update the `site_url` in `mkdocs.yml`:

```yaml
site_url: https://docs.victor-ai.com/
```

## Troubleshooting

### Workflow Fails

**Problem**: GitHub Actions workflow fails

**Solutions**:
1. Check **Actions** tab for error logs
2. Verify workflow permissions are enabled
3. Ensure Python 3.12 is available (default in GitHub Actions)
4. Check that `mkdocs.yml` syntax is valid

### Build Errors

**Problem**: MkDocs build fails locally

**Solutions**:
1. Clear cache: `mkdocs build --clean`
2. Verify all files referenced in `mkdocs.yml` exist
3. Check for broken links: `mkdocs build --strict`
4. Validate YAML: `python -c "import yaml; yaml.safe_load(open('mkdocs.yml'))"`

### Pages Not Updating

**Problem**: GitHub Pages shows old content

**Solutions**:
1. Check deployment status in **Actions** tab
2. Clear browser cache (Ctrl+Shift+R or Cmd+Shift+R)
3. Wait up to 5 minutes for GitHub Pages cache to clear
4. Verify GitHub Pages is enabled in repository settings

### 404 Errors

**Problem**: Pages return 404 errors

**Solutions**:
1. Check that `site_url` in `mkdocs.yml` matches GitHub Pages URL
2. Verify files exist in the `site/` directory after build
3. Ensure navigation structure in `mkdocs.yml` is correct
4. Check that GitHub Pages source is set to **GitHub Actions**

### Navigation Not Working

**Problem**: Navigation menu doesn't appear

**Solutions**:
1. Verify pages are listed in `nav:` section of `mkdocs.yml`
2. Check file paths are correct relative to `docs/` directory
3. Ensure each page has valid front matter
4. Test locally first: `mkdocs serve`

## Workflow Configuration

The deployment workflow `.github/workflows/docs.yml` includes:

- **Triggers**: Push to main, PRs, manual dispatch
- **Path filtering**: Only runs on docs/ changes
- **Python version**: 3.12
- **Dependencies**: mkdocs-material, mkdocstrings, pymdown-extensions
- **Deployment**: Automatic via GitHub Actions
- **Concurrency**: Cancels in-progress deployments

To customize the workflow:

```yaml
# Change Python version
env:
  PYTHON_VERSION: "3.11"  # or "3.10"

# Change trigger branches
on:
  push:
    branches:
      - main
      - develop  # Add more branches
```

## Versioning Documentation

To version your documentation (e.g., stable, dev, latest):

### 1. Install mike

```bash
pip install mike
```

### 2. Configure mkdocs.yml

```yaml
extra:
  version:
    provider: mike
    default: latest
```

### 3. Deploy versions

```bash
# Deploy version 1.0
mike deploy 1.0

# Deploy version 2.0 and update latest
mike deploy 2.0 latest

# Set default version
mike set-default latest

# Alias
mike alias 2.0 stable
```

## Monitoring Deployment

### Check Deployment Status

1. Go to **Actions** tab
2. Click on **Deploy Documentation** workflow
3. View recent workflow runs
4. Click on a run to see logs

### View Deployment History

1. Go to **Settings** > **Pages**
2. See build and deployment history
3. Check for any deployment errors

### Access Deployed Site

- **GitHub Pages URL**: `https://vjsingh1984.github.io/victor/`
- **Custom domain** (if configured): `https://docs.victor-ai.com/`

## Best Practices

### Documentation Workflow

1. **Write locally**: Use `mkdocs serve` for live preview
2. **Test before push**: Run `mkdocs build` to verify
3. **Commit changes**: Use descriptive commit messages
4. **Push to main**: Triggers automatic deployment
5. **Verify deployment**: Check Actions tab for status

### Commit Message Conventions

Use conventional commits for documentation:

```bash
git commit -m "docs: update installation guide"
git commit -m "docs(api): add provider interface reference"
git commit -m "docs(fix): correct workflow example"
git commit -m "docs(chore): update copyright year"
```

### Branch Strategy

- **main**: Production documentation (auto-deploys)
- **develop**: Development documentation
- **feature/***: Feature branches for new docs

### Review Process

For significant documentation changes:

1. Create a feature branch
2. Make documentation changes
3. Push branch to GitHub
4. Create pull request
5. Review changes in PR
6. Merge to main (triggers deployment)

## Resources

- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

## Support

For issues with documentation deployment:

1. Check GitHub Actions logs
2. Review this guide's troubleshooting section
3. Search existing GitHub issues
4. Create a new issue with the `documentation` label

---

**Last Updated**: January 2025
