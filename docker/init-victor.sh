#!/bin/bash
# Victor Docker Initialization Script
# Prepares tool embeddings cache for air-gapped semantic tool selection

set -e

echo "🚀 Initializing Victor for air-gapped deployment..."
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create Victor home directory
VICTOR_HOME="${VICTOR_HOME:-/home/victor/.victor}"
mkdir -p "$VICTOR_HOME"
mkdir -p "$VICTOR_HOME/embeddings"

echo -e "${BLUE}📂 Victor home: $VICTOR_HOME${NC}"
echo ""

# Copy default profiles if not exists
if [ ! -f "$VICTOR_HOME/profiles.yaml" ]; then
    echo -e "${BLUE}📋 Installing default profiles...${NC}"
    cp /app/docker/profiles.yaml "$VICTOR_HOME/profiles.yaml"
    echo -e "${GREEN}✓ Profiles installed${NC}"
else
    echo -e "${YELLOW}⚠ Profiles already exist, skipping${NC}"
fi

echo ""

# Initialize tool embeddings cache
echo -e "${BLUE}🧠 Initializing tool embeddings cache...${NC}"
echo "   This is a one-time operation that pre-computes semantic embeddings"
echo "   for all 31 tools using the all-MiniLM-L12-v2 model."
echo ""

python3 <<EOF
import asyncio
import sys
from pathlib import Path

# Add /app to path to import victor
sys.path.insert(0, '/app')

from victor.config.settings import Settings
from victor.tools.base import ToolRegistry
from victor.tools.semantic_selector import SemanticToolSelector

# Import all tools
from victor.tools.filesystem import read_file, write_file, list_directory
from victor.tools.bash import execute_bash
from victor.tools.file_editor_tool import edit_files
from victor.tools.git_tool import git, git_suggest_commit, git_create_pr
from victor.tools.code_executor_tool import (
    execute_python_in_sandbox,
    upload_files_to_sandbox,
)
from victor.tools.code_intelligence_tool import find_symbol, rename_symbol
from victor.tools.code_review_tool import code_review
from victor.tools.refactor_tool import (
    refactor_extract_function,
    refactor_inline_variable,
    refactor_organize_imports,
)
from victor.tools.testing_tool import (
    testing_generate,
    testing_run,
    testing_coverage,
    run_tests,
)
from victor.tools.security_scanner_tool import security_scan
from victor.tools.documentation_tool import generate_docs, analyze_docs
from victor.tools.batch_processor_tool import batch
from victor.tools.dependency_tool import dependency
from victor.tools.metrics_tool import analyze_metrics
from victor.tools.cicd_tool import cicd
from victor.tools.database_tool import database
from victor.tools.docker_tool import docker
from victor.tools.http_tool import http
from victor.tools.workflow_tool import run_workflow
from victor.tools.cache_tool import cache_clear, cache_stats

async def initialize_embeddings():
    """Initialize tool embeddings cache."""
    print("   Loading settings...")
    settings = Settings()

    print(f"   Embedding model: {settings.embedding_model}")
    print(f"   Embedding provider: {settings.embedding_provider}")
    print("")

    # Create semantic selector
    selector = SemanticToolSelector(
        embedding_model=settings.embedding_model,
        embedding_provider=settings.embedding_provider,
        cache_embeddings=True,
    )

    # Register all tools
    tools = ToolRegistry()
    all_tools = [
        read_file, write_file, list_directory,
        execute_bash,
        edit_files,
        git, git_suggest_commit, git_create_pr,
        execute_python_in_sandbox, upload_files_to_sandbox,
        find_symbol, rename_symbol,
        code_review,
        refactor_extract_function, refactor_inline_variable, refactor_organize_imports,
        testing_generate, testing_run, testing_coverage, run_tests,
        security_scan,
        generate_docs, analyze_docs,
        batch,
        dependency,
        analyze_metrics,
        cicd,
        database,
        docker,
        http,
        run_workflow,
        cache_clear, cache_stats,
    ]

    for tool in all_tools:
        tools.register(tool)

    print(f"   Registered {len(tools.list_tools())} tools")
    print("")

    # Initialize embeddings
    print("   Computing embeddings...")
    await selector.initialize_tool_embeddings(tools)

    # Verify cache was created
    cache_file = Path.home() / ".victor" / "embeddings" / f"tool_embeddings_{settings.embedding_model}.pkl"
    if cache_file.exists():
        print(f"   ✓ Cache created: {cache_file}")
        print(f"   ✓ Size: {cache_file.stat().st_size / 1024:.2f} KB")
        return True
    else:
        print(f"   ✗ Cache not found at {cache_file}")
        return False

# Run initialization
success = asyncio.run(initialize_embeddings())
sys.exit(0 if success else 1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Tool embeddings cache initialized successfully${NC}"
    echo ""
else
    echo ""
    echo -e "${YELLOW}⚠ Warning: Tool embeddings cache initialization failed${NC}"
    echo -e "${YELLOW}  Victor will still work but may be slower on first use${NC}"
    echo ""
fi

# Display summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}✅ Victor initialization complete!${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Air-Gapped Features:"
echo "  • Embedding Model: all-MiniLM-L12-v2 (120MB, 384-dim)"
echo "  • Tool Embeddings: Pre-computed (31 tools)"
echo "  • Semantic Selection: Enabled (threshold: 0.15, top-5)"
echo "  • Default Model: qwen3-coder:30b"
echo "  • Internet Required: No (100% offline)"
echo ""
echo "Quick Start:"
echo "  victor                           # Interactive mode"
echo "  victor \"Write hello world\"        # One-shot command"
echo "  victor --profile fast \"...\"       # Use fast profile (7B model)"
echo "  victor profiles                  # List all profiles"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
