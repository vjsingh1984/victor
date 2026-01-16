#!/bin/bash
# Victor Workflow Editor - End-to-End Demo Script
# This script demonstrates the complete workflow editor functionality

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOW_DIR="$SCRIPT_DIR/tools/workflow_editor"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  🎨 Victor Workflow Editor - End-to-End Demo                  ║"
echo "║  Team Node Support with Visual Connections & YAML Export      ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Start the backend server
echo -e "${BLUE}📡 Step 1: Starting Backend Server${NC}"
echo "─────────────────────────────────────────────────────────────────"
cd "$WORKFLOW_DIR"

# Check if dependencies are installed
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Installing dependencies...${NC}"
    pip install fastapi uvicorn pydantic -q
fi

# Start backend in background
echo "Starting FastAPI backend on http://localhost:8000..."
python3 backend/api.py &
BACKEND_PID=$!

# Wait for server to start
sleep 3

# Check if server is running
if curl -s http://localhost:8000/api/health > /dev/null; then
    echo -e "${GREEN}✅ Backend server started successfully (PID: $BACKEND_PID)${NC}"
else
    echo -e "${YELLOW}⚠️  Backend server may not be responding${NC}"
fi
echo ""

# Step 2: Verify API endpoints
echo -e "${BLUE}🔍 Step 2: Verifying API Endpoints${NC}"
echo "─────────────────────────────────────────────────────────────────"

echo "Testing /api/health..."
HEALTH=$(curl -s http://localhost:8000/api/health)
echo "$HEALTH" | python3 -m json.tool
echo ""

echo "Testing /api/node-types..."
NODE_TYPES=$(curl -s http://localhost:8000/api/node-types)
echo "$NODE_TYPES" | python3 -m json.tool
echo ""

echo "Testing /api/formations..."
FORMATIONS=$(curl -s http://localhost:8000/api/formations)
echo "$FORMATIONS" | python3 -m json.tool
echo ""

# Step 3: Feature showcase
echo -e "${PURPLE}🎯 Step 3: Workflow Editor Features${NC}"
echo "─────────────────────────────────────────────────────────────────"
echo ""
echo "The visual workflow editor includes the following features:"
echo ""
echo -e "${GREEN}✓ Node Types (7 total):${NC}"
echo "  • 🤖 Agent       - LLM-powered agent with role/goal"
echo "  • 👥 Team        - Multi-agent team with 8 formations"
echo "  • ⚙️  Compute     - Execute handler functions"
echo "  • 🔀 Condition   - Conditional branching"
echo "  • 🔀 Parallel    - Parallel execution"
echo "  • 🔄 Transform   - State transformation"
echo "  • 👤 HITL        - Human-in-the-loop approval"
echo ""
echo -e "${GREEN}✓ Team Formations (8 types):${NC}"
echo "  1. Sequential   - Execute members one by one"
echo "  2. Parallel     - All members work simultaneously"
echo "  3. Pipeline     - Stage-wise refinement"
echo "  4. Hierarchical - Manager-coordinated"
echo "  5. Consensus    - Agreement-based"
echo "  6. Dynamic      - Adaptive switching"
echo "  7. Adaptive     - ML-powered selection"
echo "  8. Hybrid       - Multi-phase execution"
echo ""
echo -e "${GREEN}✓ Visual Features:${NC}"
echo "  • Drag-and-drop node placement"
echo "  • Visual connection lines between nodes"
echo "  • Real-time YAML preview"
echo "  • Node property configuration panel"
echo "  • Team node configuration with all formations"
echo "  • Agent node configuration with role/goal"
echo "  • Connection management (add/remove)"
echo "  • Export complete workflow to YAML"
echo "  • Copy workflow to clipboard"
echo ""

# Step 4: Create example workflow
echo -e "${BLUE}📝 Step 4: Creating Example Workflow${NC}"
echo "─────────────────────────────────────────────────────────────────"
echo ""
echo "Example workflow demonstrating team node connections:"
echo ""
cat << 'EOF'
workflows:
  team_workflow_demo:
    description: "Team workflow with parallel formation"
    metadata:
      version: "1.0"
      vertical: "coding"
    nodes:
      - id: task_analyzer
        type: agent
        name: "Task Analyzer"
        role: planner
        goal: "Analyze the user task"
        tool_budget: 5
        next: [research_team]

      - id: research_team
        type: team
        name: "Research Team"
        goal: "Research and analyze the codebase"
        team_formation: parallel
        total_tool_budget: 50
        max_recursion_depth: 3
        members:
          - id: researcher
            role: researcher
            goal: "Conduct research"
          - id: analyst
            role: analyst
            goal: "Analyze findings"
          - id: synthesizer
            role: synthesizer
            goal: "Synthesize results"
        next: [finalize]

      - id: finalize
        type: agent
        name: "Result Finalizer"
        role: writer
        goal: "Prepare final report"
        next: []
EOF
echo ""

# Step 5: Instructions
echo -e "${PURPLE}🚀 Step 5: Launching Workflow Editor${NC}"
echo "─────────────────────────────────────────────────────────────────"
echo ""
echo -e "${GREEN}✨ Workflow Editor is now running!${NC}"
echo ""
echo "Access the editor at:"
echo -e "  ${BLUE}http://localhost:8000${NC}"
echo ""
echo "Demo Instructions:"
echo ""
echo "1. ${GREEN}Drag nodes${NC} from the left sidebar onto the canvas"
echo "   • Try dragging an Agent node and a Team node"
echo ""
echo "2. ${GREEN}Configure nodes${NC} by clicking on them"
echo "   • For Team nodes: Select formation (Sequential, Parallel, etc.)"
echo "   • Set goal, member count, tool budget, recursion depth"
echo "   • For Agent nodes: Set role, goal, tool budget"
echo ""
echo "3. ${GREEN}Connect nodes${NC} to create workflows"
echo "   • Method 1: Click '🔗 Connect Mode' button"
echo "     - Click source node, then destination node"
echo "   • Method 2: Use Properties Panel → Connections tab"
echo "     - Select destination node from dropdown"
echo "     - Click '➕ Add Connection'"
echo ""
echo "4. ${GREEN}View YAML${NC} in the Properties Panel → YAML tab"
echo "   • See real-time YAML for selected node"
echo "   • Copy node YAML to clipboard"
echo ""
echo "5. ${GREEN}Export workflow${NC} to YAML file"
echo "   • Click '📥 Export YAML' button (top right)"
echo "   • Saves as 'workflow.yaml' in your Downloads"
echo "   • Click '📋 Copy to Clipboard' for quick copy"
echo ""
echo "6. ${GREEN}Clear canvas${NC} to start over"
echo "   • Click '🗑️ Clear' button (top right)"
echo ""
echo -e "${YELLOW}💡 Tips:${NC}"
echo "  • Nodes can be dragged around the canvas"
echo "  • Connections update automatically as you move nodes"
echo "  • Click empty canvas space to deselect nodes"
echo "  • Team nodes support all 8 formation types"
echo "  • Agent nodes support role and goal configuration"
echo "  • All workflows include recursion depth tracking"
echo ""

# Step 6: Open browser
echo -e "${BLUE}🌐 Step 6: Opening Browser${NC}"
echo "─────────────────────────────────────────────────────────────────"
echo ""

# Detect OS and open browser
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Opening on macOS..."
    open "http://localhost:8000"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Opening on Linux..."
    if command -v xdg-open > /dev/null; then
        xdg-open "http://localhost:8000"
    else
        echo "Please open manually: http://localhost:8000"
    fi
else
    echo "Please open your browser to: http://localhost:8000"
fi
echo ""

# Step 7: Monitoring
echo -e "${PURPLE}⚙️  Server Monitoring${NC}"
echo "─────────────────────────────────────────────────────────────────"
echo ""
echo "Backend server is running with PID: $BACKEND_PID"
echo ""
echo "Server logs will appear below:"
echo "Press Ctrl+C to stop the server"
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""

# Wait for backend process
wait $BACKEND_PID

# Cleanup on exit
trap "echo ''; echo 'Stopping server...'; kill $BACKEND_PID 2>/dev/null; exit 0" INT TERM
