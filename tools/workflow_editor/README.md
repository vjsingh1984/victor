# Victor Workflow Editor

A visual, drag-and-drop workflow editor for Victor AI workflows with special support for team nodes.

## Features

- **Visual Canvas**: Drag-and-drop interface using React Flow
- **Team Node Builder**: Visual team configuration with member management
- **Real-time Validation**: Live feedback as you build workflows
- **YAML Sync**: Bidirectional sync between visual editor and YAML
- **Formation Selector**: Visual diagrams for team formation types
- **Node Palette**: Pre-configured node templates (agent, compute, team, condition, parallel, etc.)
- **Import/Export**: Load and save workflows from/to YAML files
- **Live Preview**: See workflow structure update in real-time

## Quick Start

```bash
# Install dependencies
cd tools/workflow_editor
./install.sh

# Start development server
./run.sh

# Open browser
open http://localhost:3000
```

## Project Structure

```
tools/workflow_editor/
├── frontend/
│   └── src/
│       ├── components/
│       │   ├── Canvas/          # Main workflow canvas
│       │   ├── NodePalette/     # Node type selector
│       │   ├── TeamNodeEditor/  # Team configuration panel
│       │   ├── FormationSelector/ # Team formation visualizer
│       │   ├── YAMLPreview/     # Real-time YAML preview
│       │   └── PropertyPanel/   # Node property editor
│       ├── nodes/
│       │   ├── AgentNode.tsx
│       │   ├── ComputeNode.tsx
│       │   ├── TeamNode.tsx
│       │   └── ...
│       ├── hooks/
│       ├── utils/
│       └── App.tsx
├── backend/
│   ├── api.py                   # FastAPI REST API
│   ├── compiler.py              # UnifiedWorkflowCompiler integration
│   ├── validator.py             # Real-time validation
│   └── yaml_manager.py          # YAML file operations
├── examples/
│   ├── code_review_workflow.json
│   └── team_node_examples.json
├── docs/
│   ├── installation.md
│   ├── user_manual.md
│   └── api_reference.md
├── install.sh
├── run.sh
├── build.sh
└── package.sh
```

## Usage

### Creating a Basic Workflow

1. Drag nodes from the palette to the canvas
2. Connect nodes by dragging from output ports to input ports
3. Click nodes to configure properties
4. View real-time YAML preview
5. Save workflow to file

### Building Team Nodes

1. Drag a "Team" node from the palette
2. Click the node to open the Team Editor
3. Add team members with roles, goals, and tool budgets
4. Select team formation (parallel, sequential, pipeline, etc.)
5. Configure recursion depth and timeout
6. Visualize team structure with formation diagrams

### Importing Existing Workflows

```bash
# Import from YAML file
victor workflow edit path/to/workflow.yaml

# Or use the web UI
# Click "Import YAML" and select file
```

## Development

### Backend API

```bash
cd backend
pip install -r requirements.txt
python api.py
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Node Types

- **Agent**: LLM-powered agent with role and goal
- **Compute**: Tool execution without LLM
- **Team**: Multi-agent team with configurable formation
- **Condition**: Branching logic based on state
- **Parallel**: Execute multiple branches concurrently
- **Transform**: State transformation
- **HITL**: Human-in-the-loop interaction

## Team Formations

- **Parallel**: All members work simultaneously
- **Sequential**: Members work in sequence
- **Pipeline**: Output passes through stages
- **Hierarchical**: Manager-worker structure
- **Consensus**: Vote-based decision making

## Keyboard Shortcuts

- `Ctrl+S`: Save workflow
- `Ctrl+Z`: Undo
- `Ctrl+Y`: Redo
- `Ctrl+C`: Copy selected node
- `Ctrl+V`: Paste node
- `Delete`: Remove selected node
- `Ctrl+D`: Duplicate node
- `Ctrl+E`: Export YAML
- `Ctrl+I`: Import YAML

## Contributing

Contributions welcome! Please see `docs/contributing.md` for guidelines.

## License

Apache License 2.0
