#!/usr/bin/env python3
"""FastAPI backend for workflow editor with frontend serving."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Any, List
import os

app = FastAPI(title="Victor Workflow Editor", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the directory of this file
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BACKEND_DIR, '..', 'frontend')


class NodeSchema(BaseModel):
    """Schema for a node type."""
    type: str
    description: str
    required_fields: List[str]
    optional_fields: List[str]


@app.get("/")
async def root():
    """Serve the frontend."""
    index_path = os.path.join(FRONTEND_DIR, 'index.html')
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Frontend not found. Please ensure frontend/index.html exists."}


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/api/node-types")
async def get_node_types() -> Dict[str, List[NodeSchema]]:
    """Get available node types."""
    return {
        "basic": [
            NodeSchema(
                type="agent",
                description="LLM-powered agent node",
                required_fields=["id", "role", "goal"],
                optional_fields=["name", "tool_budget", "next"]
            ),
            NodeSchema(
                type="compute",
                description="Execute a handler function",
                required_fields=["id", "handler"],
                optional_fields=["inputs", "next"]
            ),
            NodeSchema(
                type="condition",
                description="Conditional branching",
                required_fields=["id", "condition"],
                optional_fields=["branches"]
            ),
            NodeSchema(
                type="parallel",
                description="Parallel execution",
                required_fields=["id", "nodes"],
                optional_fields=["merge_strategy"]
            ),
            NodeSchema(
                type="transform",
                description="State transformation",
                required_fields=["id", "transform"],
                optional_fields=["inputs"]
            ),
            NodeSchema(
                type="hitl",
                description="Human-in-the-loop",
                required_fields=["id", "approval_type"],
                optional_fields=["message", "timeout"]
            ),
            NodeSchema(
                type="team",
                description="Multi-agent team",
                required_fields=["id", "team_formation", "members"],
                optional_fields=["goal", "timeout_seconds", "tool_budget"]
            ),
        ]
    }


@app.get("/api/formations")
async def get_team_formations() -> Dict[str, Any]:
    """Get available team formations."""
    return {
        "formations": [
            {"value": "sequential", "label": "Sequential", "description": "Execute members one by one"},
            {"value": "parallel", "label": "Parallel", "description": "All members work simultaneously"},
            {"value": "pipeline", "label": "Pipeline", "description": "Stage-wise refinement"},
            {"value": "hierarchical", "label": "Hierarchical", "description": "Manager-coordinated"},
            {"value": "consensus", "label": "Consensus", "description": "Agreement-based"},
            {"value": "dynamic", "label": "Dynamic", "description": "Adaptive switching"},
            {"value": "adaptive", "label": "Adaptive", "description": "ML-powered selection"},
            {"value": "hybrid", "label": "Hybrid", "description": "Multi-phase execution"},
        ]
    }


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Victor Workflow Editor...")
    print("🌐 Editor will be available at: http://localhost:8000")
    print("📚 API docs at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
