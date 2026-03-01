# Victor Documentation

This directory contains the MkDocs-based documentation for Victor.

## Quick Start

New contributors should start with the [Quick Start Guide](QUICKSTART.md) for fast setup.

For deployment and configuration, see the [Deployment Guide](DEPLOYMENT.md).

## Building Documentation Locally

### Prerequisites

Install the documentation dependencies:

```bash
# Install with pip
pip install -e ".[docs]"

# Or install dependencies manually
pip install mkdocs>=1.5
pip install mkdocs-material>=9.5
pip install mkdocstrings[python]>=0.24
pip install mkdocs-git-revision-date-localized-plugin
pip install pymdown-extensions
```

### Build Commands

```bash
# Serve documentation locally with live reload
mkdocs serve

# Build static site (outputs to ./site directory)
mkdocs build

# Build and deploy to custom directory
mkdocs build -d /path/to/output

# Clean build directory before building
mkdocs build --clean

# Show verbose output during build
mkdocs build -v
```

### Local Development

1. Start the local development server:
   ```bash
   mkdocs serve
   ```

2. Open your browser to `http://127.0.0.1:8000`

3. Edit documentation files - changes will auto-reload

4. Press `Ctrl+C` to stop the server

## Documentation Structure

```
docs/
├── index.md                          # Landing page
├── getting-started/                  # Getting started guide
│   ├── index.md
│   ├── installation.md
│   ├── quickstart.md
│   ├── configuration.md
│   ├── first-run.md
│   └── basic-usage.md
├── user-guide/                       # User guide
│   ├── index.md
│   ├── cli-reference.md
│   ├── tui-mode.md
│   ├── providers.md
│   ├── tools.md
│   ├── workflows.md
│   ├── session-management.md
│   └── troubleshooting.md
├── api-reference/                    # API reference
│   ├── protocols.md
│   ├── providers.md
│   ├── tools.md
│   └── workflows.md
├── tutorials/                        # Tutorials
│   ├── build-custom-tool.md
│   ├── create-workflow.md
│   └── integrate-provider.md
├── reference/                        # Reference documentation
│   ├── index.md
│   ├── cli-commands.md
│   ├── configuration-options.md
│   └── environment-variables.md
├── verticals/                        # Vertical-specific docs
│   ├── coding.md
│   ├── devops.md
│   ├── rag.md
│   ├── data-analysis.md
│   └── research.md
└── development/                      # Development guide
    ├── index.md
    ├── setup.md
    ├── code-style.md
    ├── testing.md
    └── PR_WORKFLOW.md                 # Pull request workflow guide
```

## Deployment to GitHub Pages

### Automated Deployment

Documentation is automatically deployed to GitHub Pages when:

- You push to the `main` branch
- Changes are made to:
  - `docs/**` directory
  - `mkdocs.yml` configuration file
  - `.github/workflows/docs.yml` workflow file

The GitHub Actions workflow (`.github/workflows/docs.yml`) will:
1. Build the documentation using MkDocs
2. Deploy to GitHub Pages
3. Publish to `https://vjsingh1984.github.io/victor/`

### Manual Deployment

To manually trigger a documentation build:

1. Go to the **Actions** tab in your GitHub repository
2. Select **Deploy Documentation** workflow
3. Click **Run workflow**
4. Select the `main` branch
5. Click **Run workflow**

### Deployment Workflow Details

The deployment workflow:
- Uses Python 3.12
- Installs MkDocs Material theme and dependencies
- Builds the documentation site
- Deploys to GitHub Pages using GitHub Actions
- Only runs on the `main` branch
- Cancels in-progress deployments to avoid conflicts

### Custom Domain

To use a custom domain (e.g., `docs.victor-ai.com`):

1. Create a `CNAME` file in the `docs/` directory:
   ```
   docs.victor-ai.com
   ```

2. Commit the CNAME file:
   ```bash
   git add docs/CNAME
   git commit -m "Add custom domain CNAME"
   git push
   ```

3. Configure DNS records with your domain provider

4. GitHub Pages will automatically use the custom domain

## Documentation Features

### Theme Features

The documentation uses the **Material for MkDocs** theme with:
- Navigation instant loading
- Tabbed navigation sections
- Search functionality with suggestions
- Dark/light mode toggle
- Code copying
- Content tabs
- Table of contents integration

### Markdown Extensions

- **Admonitions**: Note, warning, tip, info blocks
- **Code highlighting**: Pygments with line numbers
- **Task lists**: Custom checkboxes
- **Emoji support**: Full emoji set
- **Math support**: LaTeX math equations
- **Tabs**: Alternate style tabbed content
- **SuperFences**: Nested code blocks
- **MagicLink**: Auto-linking to GitHub issues/PRs

### Plugins

- **Search**: Full-text search
- **Git Revision Date**: Display last update dates
- **Minify**: (optional) HTML/CSS/JS minification

## Writing Documentation

### Markdown Style Guide

1. Use ATX-style headings (`#` rather than `=`)
2. Use `$$` for block math, `$` for inline math
3. Use `!!!` for admonitions:
   ```markdown
   !!! note
       This is a note

   !!! warning
       This is a warning
   ```

4. Use code blocks with language specification:
   ````markdown
   ```python
   def hello():
       print("Hello, World!")
   ```
   ````

5. Use task lists:
   ```markdown
   - [x] Completed task
   - [ ] Incomplete task
   ```

### Adding New Pages

1. Create a new `.md` file in the appropriate directory
2. Add it to the navigation structure in `mkdocs.yml`:
   ```yaml
   nav:
     - Section Name:
       - section/page.md
       - section/new-page.md  # Add new page here
   ```

3. Write the content following the markdown style guide
4. Test locally with `mkdocs serve`
5. Commit and push to trigger deployment

### Code Documentation

Use `mkdocstrings` to automatically generate API documentation from docstrings:

```python
def example_function(param1: str, param2: int) -> bool:
    """
    Brief description.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: If param1 is empty
    """
    pass
```

Then reference it in documentation:

````markdown
::: victor.module.example_function
    :members:
    :show-inheritance:
````

## Troubleshooting

### Build Fails

If the documentation build fails:

1. Check the GitHub Actions logs for errors
2. Test locally first: `mkdocs build -v`
3. Verify all referenced files exist
4. Check for broken links: `mkdocs build --strict`
5. Validate YAML syntax: `python -c "import yaml; yaml.safe_load(open('mkdocs.yml'))"`

### Navigation Issues

If navigation doesn't appear:

1. Verify the page is listed in `mkdocs.yml` under the `nav` section
2. Check the file path is correct relative to `docs/`
3. Ensure the file exists and has valid front matter

### Theme Not Loading

If the theme doesn't load correctly:

1. Clear your browser cache
2. Verify `mkdocs-material` is installed: `pip show mkdocs-material`
3. Check the theme configuration in `mkdocs.yml`

## Resources

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [MkDocstrings](https://mkdocstrings.github.io/)
- [PyMdown Extensions](https://facelessuser.github.io/pymdown-extensions/)

## Contributing

When contributing documentation:

1. Follow the existing style and structure
2. Test locally before submitting PRs
3. Use descriptive commit messages
4. Add yourself to contributors if desired
5. Ensure all links are valid
6. Check for spelling and grammar errors

For more information, see the [Development Guide](development/).

---

Back to [README.md](../README.md)
