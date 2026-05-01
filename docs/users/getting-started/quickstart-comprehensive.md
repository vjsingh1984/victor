# Documentation Quick Start

Quick guide for contributors to build and test documentation locally.

## Prerequisites

```bash
# Install documentation dependencies
pip install -e ".[docs]"
```

Or install manually:

```bash
pip install mkdocs>=1.5
pip install mkdocs-material>=9.5
pip install mkdocstrings[python]>=0.24
pip install mkdocs-git-revision-date-localized-plugin
pip install pymdown-extensions
```

## Local Development

### Start Live Server

```bash
# Option 1: Using helper script (recommended)
./scripts/docs-serve.sh

# Option 2: Direct mkdocs command
mkdocs serve
```

Open `http://127.0.0.1:8000` in your browser.

### Build Static Site

```bash
# Option 1: Using helper script
./scripts/docs-build.sh

# Option 2: Direct mkdocs command
mkdocs build --clean
```

Output will be in the `./site/` directory.

## Making Changes

1. Edit documentation files in `docs/`
2. Save changes - they auto-reload in live server
3. Test navigation and links
4. Commit and push to `main` branch

```bash
git add docs/
git commit -m "docs: describe your changes"
git push origin main
```

## Common Tasks

### Add New Page

1. Create `.md` file in appropriate directory
2. Add to `nav:` section in `mkdocs.yml`
3. Test locally
4. Commit changes

### Add Image

1. Place image in `docs/assets/images/`
2. Reference in markdown:

```markdown
![Alt text](assets/images/filename.png)
```

### Add Code Block

Use fenced code blocks with language:

````markdown
```python
def example():
    pass
```
````

### Add Admonition

```markdown
!!! note
    This is a note block

!!! warning
    This is a warning
```

## Validation

### Check Links

```bash
mkdocs build --strict
```

### Validate Configuration

```bash
python3 -c "import yaml; yaml.safe_load(open('mkdocs.yml'))"
```

### Test Deployment Locally

```bash
# Build
mkdocs build --clean

# Serve built site
python3 -m http.server 8000 --directory site
```

## Deployment

Documentation deploys automatically to GitHub Pages when:

- Changes are pushed to `main` branch
- Files in `docs/` or `mkdocs.yml` are modified

Deployed site: `https://vjsingh1984.github.io/victor/`

For manual deployment or troubleshooting, see [DEPLOYMENT.md](DEPLOYMENT.md).

## Next Steps

- Full documentation guide: [README.md](README.md)
- Deployment guide: [DEPLOYMENT.md](DEPLOYMENT.md)
- MkDocs documentation: https://www.mkdocs.org/
- Material theme: https://squidfunk.github.io/mkdocs-material/

## Quick Reference

| Task | Command |
|------|---------|
| Start server | `./scripts/docs-serve.sh` |
| Build docs | `./scripts/docs-build.sh` |
| Validate links | `mkdocs build --strict` |
| Clean build | `mkdocs build --clean` |
| Help | `mkdocs --help` |

---

Need help? Check the [troubleshooting section](DEPLOYMENT.md#troubleshooting) or create an issue.
