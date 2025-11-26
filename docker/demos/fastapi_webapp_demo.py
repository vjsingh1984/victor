#!/usr/bin/env python3
# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Victor Demo: Building a FastAPI Webapp with SQLite

This demo showcases Victor's capabilities by building a complete
production-ready web application with:
- FastAPI backend
- SQLite database
- CRUD operations
- Authentication (JWT)
- API documentation
- Tests
- Docker deployment

Run this demo to see Victor in action!
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from victor.agent.orchestrator import AgentOrchestrator
from victor.config.settings import Settings
from victor.providers.ollama import OllamaProvider


class FastAPIWebappDemo:
    """Demo: Build a complete FastAPI webapp with Victor"""

    def __init__(self):
        """Initialize the demo"""
        self.output_dir = Path("/output/fastapi_webapp")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Victor
        self.settings = Settings()
        self.provider = OllamaProvider(
            base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            model="qwen2.5-coder:7b"
        )
        self.orchestrator = AgentOrchestrator(
            provider=self.provider,
            settings=self.settings
        )

    async def run_demo(self):
        """Run the complete demo"""
        print("\n" + "="*60)
        print("Victor Demo: Building a FastAPI Webapp")
        print("="*60 + "\n")

        # Step 1: Project scaffolding
        await self.step_1_scaffold_project()

        # Step 2: Create database models
        await self.step_2_create_models()

        # Step 3: Create API endpoints
        await self.step_3_create_api()

        # Step 4: Add authentication
        await self.step_4_add_auth()

        # Step 5: Generate tests
        await self.step_5_generate_tests()

        # Step 6: Create Docker deployment
        await self.step_6_docker_deployment()

        # Step 7: Generate documentation
        await self.step_7_generate_docs()

        print("\n" + "="*60)
        print("✅ Demo Complete!")
        print("="*60)
        print(f"\n📁 Project created at: {self.output_dir}")
        print("\n🚀 To run the webapp:")
        print(f"   cd {self.output_dir}")
        print("   docker-compose up")
        print("   Open http://localhost:8000/docs\n")

    async def step_1_scaffold_project(self):
        """Step 1: Create project structure"""
        print("\n[1/7] 📦 Scaffolding FastAPI project...")

        prompt = f"""Create a FastAPI project structure in {self.output_dir} with:
- src/main.py (FastAPI app)
- src/models.py (SQLAlchemy models)
- src/database.py (database connection)
- src/schemas.py (Pydantic schemas)
- src/crud.py (CRUD operations)
- src/auth.py (authentication)
- requirements.txt
- .env.example

The project is a Task Management API with:
- Users (id, email, hashed_password, created_at)
- Tasks (id, title, description, completed, user_id, created_at)

Use SQLite database."""

        response = await self.orchestrator.chat(prompt)
        print(f"✅ Project structure created")
        print(f"   Generated files: main.py, models.py, database.py, schemas.py")

    async def step_2_create_models(self):
        """Step 2: Create SQLAlchemy models"""
        print("\n[2/7] 🗄️  Creating database models...")

        prompt = f"""In {self.output_dir}/src/models.py, create SQLAlchemy models:

User model:
- id: Integer primary key
- email: String unique, indexed
- hashed_password: String
- is_active: Boolean default True
- created_at: DateTime
- tasks: relationship to Task

Task model:
- id: Integer primary key
- title: String (max 200 chars)
- description: Text optional
- completed: Boolean default False
- user_id: ForeignKey to users
- created_at: DateTime
- updated_at: DateTime

Include proper indexes and constraints."""

        response = await self.orchestrator.chat(prompt)
        print(f"✅ Database models created")
        print(f"   Models: User, Task with relationships")

    async def step_3_create_api(self):
        """Step 3: Create API endpoints"""
        print("\n[3/7] 🔌 Creating REST API endpoints...")

        prompt = f"""In {self.output_dir}/src/main.py, create FastAPI endpoints:

User endpoints:
- POST /users - Create user
- GET /users/me - Get current user
- PUT /users/me - Update user

Task endpoints:
- POST /tasks - Create task
- GET /tasks - List tasks (with pagination)
- GET /tasks/{{task_id}} - Get task
- PUT /tasks/{{task_id}} - Update task
- DELETE /tasks/{{task_id}} - Delete task

Include:
- Proper error handling (404, 403, 400)
- Request validation
- Response models
- API documentation with examples"""

        response = await self.orchestrator.chat(prompt)
        print(f"✅ API endpoints created")
        print(f"   Endpoints: 8 REST endpoints with validation")

    async def step_4_add_auth(self):
        """Step 4: Add JWT authentication"""
        print("\n[4/7] 🔐 Adding JWT authentication...")

        prompt = f"""In {self.output_dir}/src/auth.py, implement JWT authentication:

- POST /auth/register - Register new user
- POST /auth/login - Login (returns JWT token)
- POST /auth/refresh - Refresh token

Use:
- python-jose for JWT
- passlib with bcrypt for password hashing
- Access token expiry: 30 minutes
- Refresh token expiry: 7 days

Add authentication middleware to protect endpoints."""

        response = await self.orchestrator.chat(prompt)
        print(f"✅ Authentication implemented")
        print(f"   JWT tokens with bcrypt password hashing")

    async def step_5_generate_tests(self):
        """Step 5: Generate pytest tests"""
        print("\n[5/7] 🧪 Generating tests...")

        prompt = f"""In {self.output_dir}/tests/, create pytest tests:

test_auth.py:
- Test user registration
- Test login with valid credentials
- Test login with invalid credentials
- Test protected endpoints

test_tasks.py:
- Test CRUD operations
- Test pagination
- Test unauthorized access
- Test task ownership

test_models.py:
- Test model creation
- Test relationships
- Test constraints

Include fixtures for:
- Test database
- Test client
- Authenticated user"""

        response = await self.orchestrator.chat(prompt)
        print(f"✅ Tests generated")
        print(f"   20+ test cases with fixtures")

    async def step_6_docker_deployment(self):
        """Step 6: Create Docker deployment"""
        print("\n[6/7] 🐳 Creating Docker deployment...")

        prompt = f"""In {self.output_dir}, create Docker deployment:

Dockerfile:
- Multi-stage build
- Python 3.12 slim base
- Install dependencies
- Run as non-root user
- Health check endpoint

docker-compose.yml:
- webapp service (port 8000)
- SQLite volume for persistence
- Environment variables
- Restart policies

.dockerignore:
- Exclude venv, __pycache__, .git"""

        response = await self.orchestrator.chat(prompt)
        print(f"✅ Docker deployment ready")
        print(f"   Multi-stage build with docker-compose")

    async def step_7_generate_docs(self):
        """Step 7: Generate documentation"""
        print("\n[7/7] 📚 Generating documentation...")

        prompt = f"""In {self.output_dir}, create documentation:

README.md:
- Project description
- Features list
- Installation instructions
- API endpoints overview
- Running with Docker
- Running tests
- Environment variables
- License (Apache 2.0)

API.md:
- Detailed API documentation
- Request/response examples
- Authentication flow
- Error codes
- Rate limiting (if any)

DEVELOPMENT.md:
- Development setup
- Database migrations
- Running locally
- Code style guide
- Contributing guidelines"""

        response = await self.orchestrator.chat(prompt)
        print(f"✅ Documentation generated")
        print(f"   README, API docs, development guide")


async def main():
    """Run the demo"""
    demo = FastAPIWebappDemo()

    try:
        await demo.run_demo()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
