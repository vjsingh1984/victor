# User Manual - Victor Workflow Editor

## Table of Contents

1. [Getting Started](#getting-started)
2. [Interface Overview](#interface-overview)
3. [Creating Workflows](#creating-workflows)
4. [Team Node Editor](#team-node-editor)
5. [Import/Export](#importexport)
6. [Keyboard Shortcuts](#keyboard-shortcuts)
7. [Tips and Tricks](#tips-and-tricks)

## Getting Started

### Launch the Editor

```bash
cd tools/workflow_editor
./run.sh
```

Open http://localhost:3000 in your browser.

### First Time

1. You'll see a blank canvas with node palette on the left
2. Drag nodes from the palette to the canvas
3. Connect nodes by dragging from output handles to input handles
4. Click nodes to configure properties
5. Export your workflow as YAML

## Interface Overview

```
┌─────────────────────────────────────────────────────────┐
│ Header: Title | Import | Export | Clear                 │
├──────────┬──────────────────────────┬────────────────────┤
│ Node     │                          │  Property Panel   │
│ Palette  │      Canvas               │  (when node       │
│          │                          │   selected)       │
│ - Agent  │                          │                   │
│ - Compute│     [Your Nodes]         │  - ID             │
│ - Team   │                          │  - Type           │
│ - Cond   │                          │  - Label          │
│ - Trans  │                          │  - Config         │
└──────────┴──────────────────────────┴────────────────────┘
```

### Components

- **Node Palette** (left): Drag-and-drop node templates
- **Canvas** (center): Visual workflow editing area
- **Property Panel** (right): Edit selected node properties
- **YAML Preview**: Toggle with "YAML" button in header

## Creating Workflows

### Adding Nodes

1. Drag a node type from the palette
2. Drop it onto the canvas
3. The node is created with default settings

### Connecting Nodes

1. Hover over a node to see connection handles
2. Drag from the bottom handle (output) to another node's top handle (input)
3. Release to create the connection

### Configuring Nodes

1. Click a node to select it
2. Property panel shows on the right
3. Edit properties:
   - **Label**: Display name
   - **Type**: Node type (read-only)
   - **Custom properties**: Vary by node type

### Deleting Nodes

1. Select a node
2. Click "Delete" in property panel, or
3. Press Delete/Backspace key

## Team Node Editor

Team nodes are special - they have a dedicated editor.

### Creating a Team

1. Drag "Team" node from palette to canvas
2. Team editor automatically opens
3. Configure team settings

### Team Formation Types

**Parallel** (||)
- All members work simultaneously
- Best for: Independent analysis, multi-perspective review

**Sequential** (→)
- Members work in sequence
- Best for: Step-by-step processing, refinement loops

**Pipeline** (⇒)
- Assembly-line processing
- Best for: Multi-stage workflows, specialized tasks

**Hierarchical** (⬗)
- Manager coordinates workers
- Best for: Complex coordination, task delegation

**Consensus** (◊)
- Members vote on decisions
- Best for: Decision making, quality assurance

### Adding Team Members

1. In team editor, click "Add Member"
2. Configure member:
   - **Role**: assistant, researcher, planner, executor, reviewer, writer
   - **Goal**: What this member should accomplish
   - **Tool Budget**: Max tool calls (default: 25)
   - **Backstory**: Background context (optional)
   - **Expertise**: Areas of specialization (optional)
   - **Personality**: Behavioral traits (optional)
3. Click "Save Member"

### Editing Team Members

1. In team editor, click the edit (pencil) icon next to a member
2. Update member properties
3. Click "Save Member"

### Deleting Team Members

1. In team editor, click the delete (trash) icon next to a member
2. Confirm deletion

## Import/Export

### Export Workflow

1. Click "Export" in header
2. Workflow downloads as JSON file
3. For YAML, click "YAML" button, then "Download"

### Import Workflow

1. Click "Import" in header
2. Select workflow file (JSON or YAML)
3. Workflow loads onto canvas

### Copy YAML to Clipboard

1. Click "YAML" button in header
2. YAML preview panel opens
3. Click "Copy" button

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+S` | Save workflow (to browser local storage) |
| `Ctrl+Z` | Undo last action |
| `Ctrl+Y` | Redo last action |
| `Ctrl+C` | Copy selected node |
| `Ctrl+V` | Paste node |
| `Ctrl+D` | Duplicate selected node |
| `Delete` | Delete selected node |
| `Ctrl+E` | Export workflow |
| `Ctrl+I` | Import workflow |
| `Escape` | Close modal panels |
| `Ctrl+Shift+Y` | Toggle YAML preview |

## Tips and Tricks

### Canvas Navigation

- **Pan**: Click and drag on empty canvas space
- **Zoom**: Use mouse wheel or pinch gesture
- **Fit view**: Click the fit-view button in controls (bottom-right)

### Node Organization

- **Grid snap**: Nodes automatically align to grid
- **Auto-layout**: Select nodes and press `L` to auto-arrange
- **Grouping**: Hold `Shift` to select multiple nodes

### Workflow Best Practices

1. **Start with a goal**: Define what the workflow should accomplish
2. **Use team nodes wisely**: Teams are powerful but resource-intensive
3. **Test incrementally**: Build and test small sections
4. **Document nodes**: Use clear labels and descriptions
5. **Validate often**: Check for errors in YAML preview

### Common Patterns

**Sequential Processing:**
```
Agent → Compute → Transform → Agent
```

**Parallel Analysis:**
```
      → Agent (1) →
Agent                → Transform
      → Agent (2) →
```

**Team Review:**
```
Agent → Team (parallel formation) → Transform
```

## Troubleshooting

### Nodes Won't Connect

- Check that you're dragging from output (bottom) to input (top)
- Ensure both handles are visible (hover over node)
- Try zooming in if handles are too small

### Workflow Won't Export

- Check for validation errors (red banner at top of canvas)
- Ensure all required node properties are set
- Look for circular dependencies

### Team Editor Won't Open

- Double-click the team node instead of single-click
- Check that the node type is "team" (not "agent" or other)
- Try selecting the node first, then double-clicking

## Getting Help

- GitHub Issues: https://github.com/your-repo/issues
- Documentation: See other docs in `docs/` folder
- Examples: Check `examples/` folder for sample workflows
