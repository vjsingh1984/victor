"""Project scaffolding tool for generating project templates and boilerplate.

Features:
- Project templates (FastAPI, Flask, React, etc.)
- File structure generation
- Configuration files
- Best practices setup
- Development tooling
"""

import subprocess
from pathlib import Path
from typing import Any, Dict
import logging

from victor.tools.decorators import tool

logger = logging.getLogger(__name__)

# Project templates
TEMPLATES = {
    "fastapi": {
        "name": "FastAPI Application",
        "description": "Production-ready FastAPI application with best practices",
        "files": [
            ("app/__init__.py", ""),
            ("app/main.py", "fastapi_main"),
            ("app/api/__init__.py", ""),
            ("app/api/routes.py", "fastapi_routes"),
            ("app/core/__init__.py", ""),
            ("app/core/config.py", "fastapi_config"),
            ("app/models/__init__.py", ""),
            ("app/schemas/__init__.py", ""),
            ("tests/__init__.py", ""),
            ("tests/test_api.py", "fastapi_test"),
            (".env.example", "env_example"),
            ("requirements.txt", "fastapi_requirements"),
            ("pyproject.toml", "pyproject_fastapi"),
            ("README.md", "readme_fastapi"),
            (".gitignore", "gitignore_python"),
        ],
    },
    "flask": {
        "name": "Flask Application",
        "description": "Flask web application with blueprints",
        "files": [
            ("app/__init__.py", "flask_init"),
            ("app/routes.py", "flask_routes"),
            ("app/models.py", "flask_models"),
            ("app/config.py", "flask_config"),
            ("tests/__init__.py", ""),
            ("tests/test_routes.py", "flask_test"),
            ("requirements.txt", "flask_requirements"),
            ("run.py", "flask_run"),
            ("README.md", "readme_flask"),
            (".gitignore", "gitignore_python"),
        ],
    },
    "python-cli": {
        "name": "Python CLI Application",
        "description": "Python command-line tool with Click",
        "files": [
            ("cli/__init__.py", ""),
            ("cli/main.py", "cli_main"),
            ("cli/commands/__init__.py", ""),
            ("cli/commands/base.py", "cli_commands"),
            ("tests/__init__.py", ""),
            ("tests/test_cli.py", "cli_test"),
            ("setup.py", "setup_py"),
            ("requirements.txt", "cli_requirements"),
            ("README.md", "readme_cli"),
            (".gitignore", "gitignore_python"),
        ],
    },
    "react-app": {
        "name": "React Application",
        "description": "Modern React app with TypeScript",
        "files": [
            ("src/App.tsx", "react_app"),
            ("src/index.tsx", "react_index"),
            ("src/components/Header.tsx", "react_header"),
            ("public/index.html", "react_html"),
            ("package.json", "react_package"),
            ("tsconfig.json", "tsconfig"),
            (".gitignore", "gitignore_node"),
            ("README.md", "readme_react"),
        ],
    },
    "microservice": {
        "name": "Microservice",
        "description": "Microservice with Docker and CI/CD",
        "files": [
            ("src/__init__.py", ""),
            ("src/main.py", "microservice_main"),
            ("src/api/__init__.py", ""),
            ("src/api/handlers.py", "microservice_handlers"),
            ("src/config.py", "microservice_config"),
            ("tests/__init__.py", ""),
            ("tests/test_api.py", "microservice_test"),
            ("Dockerfile", "dockerfile"),
            ("docker-compose.yml", "docker_compose"),
            (".dockerignore", "dockerignore"),
            ("requirements.txt", "microservice_requirements"),
            ("README.md", "readme_microservice"),
            (".gitignore", "gitignore_python"),
        ],
    },
}

# Template content
TEMPLATE_CONTENT = {
    "fastapi_main": '''"""FastAPI application entry point."""

from fastapi import FastAPI
from app.core.config import settings
from app.api import routes

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

app.include_router(routes.router, prefix=settings.API_V1_STR)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
''',
    "fastapi_routes": '''"""API routes."""

from fastapi import APIRouter, HTTPException

router = APIRouter()

@router.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to the API"}

@router.get("/items/{item_id}")
async def read_item(item_id: int):
    """Get item by ID."""
    if item_id < 0:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"item_id": item_id, "name": f"Item {item_id}"}
''',
    "fastapi_config": '''"""Application configuration."""

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings."""

    PROJECT_NAME: str = "FastAPI Application"
    API_V1_STR: str = "/api/v1"

    class Config:
        env_file = ".env"

settings = Settings()
''',
    "fastapi_test": '''"""API tests."""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_root():
    """Test root endpoint."""
    response = client.get("/api/v1/")
    assert response.status_code == 200
''',
    "fastapi_requirements": '''fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
python-multipart>=0.0.6
pytest>=7.4.0
httpx>=0.25.0
''',
    "pyproject_fastapi": '''[tool.black]
line-length = 100

[tool.ruff]
line-length = 100

[tool.mypy]
python_version = "3.10"
strict = true

[tool.pytest.ini_options]
testpaths = ["tests"]
''',
    "readme_fastapi": '''# FastAPI Application

Production-ready FastAPI application.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --reload
```

## Test

```bash
pytest
```

## API Documentation

Visit http://localhost:8000/docs for interactive API documentation.
''',
    "cli_main": '''"""CLI application."""

import click

@click.group()
def cli():
    """My awesome CLI tool."""
    pass

@cli.command()
@click.option('--name', default='World', help='Name to greet')
def hello(name):
    """Say hello."""
    click.echo(f'Hello, {name}!')

if __name__ == '__main__':
    cli()
''',
    "dockerfile": '''FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
''',
    "docker_compose": '''version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENV=development
    volumes:
      - .:/app
''',
    "gitignore_python": '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
dist/
*.egg-info/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# Environment
.env
.env.local

# Testing
.coverage
htmlcov/
.pytest_cache/
''',
    "gitignore_node": '''# Dependencies
node_modules/

# Production
build/
dist/

# Misc
.DS_Store
.env.local
.env.development.local
.env.test.local
.env.production.local

# Logs
npm-debug.log*
yarn-debug.log*
yarn-error.log*
''',
    "env_example": '''# Application
PROJECT_NAME=My FastAPI App
API_V1_STR=/api/v1

# Database (if needed)
# DATABASE_URL=postgresql://user:password@localhost/dbname
''',
}


@tool
async def scaffold_create(
    template: str,
    name: str,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Create new project from template.

    Generates a complete project structure from predefined templates
    with best practices, configuration files, and boilerplate code.

    Args:
        template: Template name (fastapi, flask, python-cli, react-app, microservice).
        name: Project name (will be used as directory name).
        force: Overwrite existing directory if it exists (default: False).

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - project_dir: Path to created project directory
        - files_created: List of created file paths
        - template_info: Template name and description
        - next_steps: Recommended next steps
        - formatted_report: Human-readable creation report
        - error: Error message if failed
    """
    if not template:
        return {
            "success": False,
            "error": "Missing required parameter: template"
        }

    if not name:
        return {
            "success": False,
            "error": "Missing required parameter: name"
        }

    if template not in TEMPLATES:
        available = ", ".join(TEMPLATES.keys())
        return {
            "success": False,
            "error": f"Unknown template: {template}. Available: {available}"
        }

    # Create project directory
    project_dir = Path(name)

    if project_dir.exists() and not force:
        return {
            "success": False,
            "error": f"Directory '{name}' already exists. Use force=True to overwrite"
        }

    project_dir.mkdir(parents=True, exist_ok=True)

    # Get template
    template_info = TEMPLATES[template]
    created_files = []

    # Create all files
    for file_path, content_key in template_info["files"]:
        full_path = project_dir / file_path

        # Create parent directories
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # Get content
        if content_key:
            content = TEMPLATE_CONTENT.get(content_key, "")
        else:
            content = ""

        # Write file
        try:
            full_path.write_text(content)
            created_files.append(str(file_path))
        except Exception as e:
            logger.error("Failed to create %s: %s", file_path, e)

    # Determine next steps
    next_steps = [f"cd {name}"]

    if template in ["fastapi", "flask", "microservice"]:
        next_steps.extend([
            "pip install -r requirements.txt",
            "# Edit .env.example and save as .env"
        ])
    elif template == "react-app":
        next_steps.extend([
            "npm install",
            "npm start"
        ])
    elif template == "python-cli":
        next_steps.append("pip install -e .")

    next_steps.extend([
        "git init",
        "git add .",
        "git commit -m 'Initial commit'"
    ])

    # Build report
    report = []
    report.append(f"Created project: {name}")
    report.append(f"Template: {template_info['name']}")
    report.append(f"Description: {template_info['description']}")
    report.append("")
    report.append("Files created:")
    for file_path in created_files:
        report.append(f"  ✓ {file_path}")
    report.append("")
    report.append("Next steps:")
    for step in next_steps:
        report.append(f"  {step}")

    return {
        "success": True,
        "project_dir": name,
        "files_created": created_files,
        "template_info": {
            "name": template_info["name"],
            "description": template_info["description"]
        },
        "next_steps": next_steps,
        "formatted_report": "\n".join(report)
    }


@tool
async def scaffold_list_templates() -> Dict[str, Any]:
    """
    List available project templates.

    Returns information about all available project templates,
    including names, descriptions, and file counts.

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - templates: List of template information dicts
        - count: Number of templates available
        - formatted_report: Human-readable template listing
    """
    templates = []

    report = []
    report.append("Available Project Templates:")
    report.append("=" * 70)
    report.append("")

    for template_id, template_info in TEMPLATES.items():
        templates.append({
            "id": template_id,
            "name": template_info["name"],
            "description": template_info["description"],
            "file_count": len(template_info["files"])
        })

        report.append(f"• {template_id}")
        report.append(f"  Name: {template_info['name']}")
        report.append(f"  Description: {template_info['description']}")
        report.append(f"  Files: {len(template_info['files'])}")
        report.append("")

    report.append("Usage:")
    report.append("  scaffold_create(template='fastapi', name='my-project')")

    return {
        "success": True,
        "templates": templates,
        "count": len(templates),
        "formatted_report": "\n".join(report)
    }


@tool
async def scaffold_add_file(
    path: str,
    content: str = "",
) -> Dict[str, Any]:
    """
    Add file to existing project.

    Creates a new file in the current project, creating parent
    directories as needed.

    Args:
        path: File path to create.
        content: File content (default: empty string).

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - file_path: Path to created file
        - message: Success message
        - error: Error message if failed
    """
    if not path:
        return {
            "success": False,
            "error": "Missing required parameter: path"
        }

    file_path = Path(path)

    # Create parent directories
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        file_path.write_text(content)
        return {
            "success": True,
            "file_path": path,
            "message": f"Created file: {path}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to create file: {e}"
        }


@tool
async def scaffold_init_git() -> Dict[str, Any]:
    """
    Initialize git repository.

    Initializes a git repository in the current directory and
    creates an initial commit.

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - message: Success message
        - error: Error message if failed
    """
    try:
        # Initialize git
        subprocess.run(["git", "init"], check=True, capture_output=True)

        # Create initial commit
        subprocess.run(["git", "add", "."], check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"],
            check=True,
            capture_output=True,
        )

        return {
            "success": True,
            "message": "Git repository initialized with initial commit"
        }

    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "error": f"Git initialization failed: {e.stderr.decode()}"
        }
    except FileNotFoundError:
        return {
            "success": False,
            "error": "Git not found. Please install git first."
        }


# Keep class for backward compatibility
class ScaffoldTool:
    """Deprecated: Use individual scaffold_* functions instead."""

    def __init__(self):
        """Initialize - deprecated."""
        import warnings
        warnings.warn(
            "ScaffoldTool class is deprecated. Use scaffold_* functions instead.",
            DeprecationWarning,
            stacklevel=2
        )
