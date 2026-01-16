# Victor Workflow Editor - Advanced Features Guide

## Overview

The Victor Workflow Editor v2.0 introduces advanced features designed for power users and professionals working with complex workflows. This guide covers all new capabilities in detail.

**Version:** 2.0.0
**Last Updated:** 2026-01-15

---

## Table of Contents

1. [Undo/Redo System](#1-undoredo-system)
2. [Zoom and Pan Controls](#2-zoom-and-pan-controls)
3. [Keyboard Shortcuts](#3-keyboard-shortcuts)
4. [Mini-Map Navigation](#4-mini-map-navigation)
5. [Search and Filter](#5-search-and-filter)
6. [Node Grouping](#6-node-grouping)
7. [Workflow Templates](#7-workflow-templates)
8. [Auto-Layout Algorithms](#8-auto-layout-algorithms)
9. [Performance Optimization](#9-performance-optimization)
10. [Customization](#10-customization)

---

## 1. Undo/Redo System

### Overview

The undo/redo system tracks all canvas operations, allowing you to revert mistakes or reapply changes.

### Features

- **History Limit:** 100 operations
- **Operations Tracked:**
  - Add node
  - Delete node
  - Move node
  - Create connection
  - Delete connection
  - Update node data

### Usage

#### Keyboard Shortcuts
- `Ctrl + Z` - Undo
- `Ctrl + Y` - Redo
- `Ctrl + Shift + Z` - Redo (alternative)

#### Toolbar Buttons
- Click the **Undo** button (←) to revert the last operation
- Click the **Redo** button (→) to reapply the undone operation
- Buttons are disabled when no history is available

### Technical Details

**State Management:**
- Deep cloning of nodes and edges for each history state
- Timestamps for each operation
- Automatic history pruning when limit reached
- Future history removed when new action performed

**Memory Usage:**
- Approximately 0.65 KB per history entry
- 100 operations ≈ 65 KB memory usage

### Best Practices

1. **Frequent Undos:** Use `Ctrl + Z` frequently to experiment without fear
2. **History Navigation:** Undo multiple steps to reach desired state
3. **Branch Exploration:** Try different approaches using undo/redo

### Example Workflow

```
1. Add agent node → Ctrl+Z (undo)
2. Add team node instead
3. Connect nodes
4. Realize team formation is wrong → Ctrl+Z (undo connection)
5. Change formation to "parallel"
6. Reconnect → Ctrl+Y (redo if needed)
```

---

## 2. Zoom and Pan Controls

### Overview

Navigate large workflows easily with zoom (10%-200%) and pan controls.

### Features

- **Zoom Range:** 10% to 200%
- **Mouse Wheel:** Zoom in/out
- **Click and Drag:** Pan canvas
- **Fit to Screen:** Automatically zoom to fit all nodes

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl + 0` | Reset zoom to 100% |
| `Ctrl + +` | Zoom in (+10%) |
| `Ctrl + -` | Zoom out (-10%) |
| `Ctrl + 9` | Fit to screen |
| `Mouse Wheel` | Zoom in/out |

### Toolbar Controls

```
[-] 100% [+] [Reset] [Fit to Screen]
```

- **(-)** Button: Zoom out
- **Percentage Display:** Current zoom level
- **(+)** Button: Zoom in
- **Reset:** Return to 100%
- **Fit to Screen:** Zoom to fit all nodes

### Usage Scenarios

#### Working with Large Workflows (50+ nodes)
1. Zoom out to 50% to see overview
2. Use mini-map to navigate
3. Zoom in to 150% for detailed editing
4. Use `Ctrl + 9` to reset when done

#### Fine-Tuning Node Positions
1. Zoom to 150% for precision
2. Drag nodes to exact positions
3. Zoom out to verify overall layout

#### Presenting Workflows
1. Use "Fit to Screen" for full view
2. Zoom to specific sections during discussion
3. Reset zoom when finished

### Technical Details

**Viewport State:**
```typescript
{
  x: number,  // Pan offset X
  y: number,  // Pan offset Y
  zoom: number  // Zoom level (0.1 - 2.0)
}
```

**Performance:**
- 60fps smooth zoom/pan
- <16ms render time
- Hardware acceleration via CSS transforms

---

## 3. Keyboard Shortcuts

### Overview

Comprehensive keyboard shortcuts for efficient, mouse-free workflow building.

### Complete Reference

#### Edit Operations
| Shortcut | Action | Notes |
|----------|--------|-------|
| `Ctrl + Z` | Undo | Revert last operation |
| `Ctrl + Y` | Redo | Reapply undone operation |
| `Ctrl + C` | Copy | Copy selected nodes |
| `Ctrl + V` | Paste | Paste copied nodes |
| `Ctrl + D` | Duplicate | Duplicate selected node |
| `Ctrl + A` | Select All | Select all nodes on canvas |
| `Delete` | Delete | Delete selected nodes |
| `Backspace` | Delete | Delete selected nodes (alternative) |
| `Escape` | Deselect | Clear selection / Close panels |

#### Zoom Controls
| Shortcut | Action |
|----------|--------|
| `Ctrl + 0` | Reset zoom to 100% |
| `Ctrl + +` | Zoom in |
| `Ctrl + -` | Zoom out |
| `Ctrl + 9` | Fit to screen |

#### Search
| Shortcut | Action |
|----------|--------|
| `Ctrl + F` | Focus search bar |
| `F3` | Next search result |
| `Ctrl + G` | Next search result (alternative) |
| `Shift + F3` | Previous search result |
| `Ctrl + Shift + G` | Previous search result (alternative) |

#### File Operations
| Shortcut | Action |
|----------|--------|
| `Ctrl + S` | Save/Export workflow |
| `Ctrl + O` | Import workflow |

#### Help
| Shortcut | Action |
|----------|--------|
| `Ctrl + /` | Show keyboard shortcuts help |

### Help Modal

Press `Ctrl + /` to see all keyboard shortcuts organized by category:
- Edit Operations
- Selection
- Zoom & Pan
- Search
- File Operations
- Help

### Power User Workflow

Build a complete workflow without touching the mouse:

```
1. Ctrl+F → Type "bugfix" → Enter
2. Tab → Enter (insert template)
3. Ctrl+C (select agent) → Ctrl+V (paste)
4. Ctrl+D (duplicate agent)
5. Click to select → Drag to position
6. Ctrl+S (save)
```

### Customization

Keyboard shortcuts are hardcoded in `KeyboardShortcuts.tsx`. To customize:

1. Open `frontend/src/components/KeyboardShortcuts/KeyboardShortcuts.tsx`
2. Find the `handleKeyDown` function
3. Modify shortcut combinations
4. Update help modal text

---

## 4. Mini-Map Navigation

### Overview

Mini-map provides a bird's-eye view of the entire workflow for quick navigation.

### Features

- **Position:** Bottom-right corner
- **Size:** 200x150 pixels
- **Auto-Scaling:** Fits all nodes in view
- **Color-Coding:** Nodes colored by type
- **Viewport Indicator:** Blue rectangle shows current view
- **Click Navigation:** Click to jump to area

### Visual Design

```
┌─────────────────────┐
│  Mini Map           │
│  ┌────────┐         │
│  │ ┌────┐ │         │
│  │ │Agent│ │  ← Viewport
│  │ └────┘ │         │
│  └────────┘         │
│  12 nodes           │
└─────────────────────┘
```

### Node Colors

| Type | Color | Hex |
|------|-------|-----|
| Agent | Blue | #3b82f6 |
| Team | Purple | #8b5cf6 |
| Compute | Green | #10b981 |
| Condition | Amber | #f59e0b |
| Parallel | Pink | #ec4899 |
| Transform | Indigo | #6366f1 |
| HITL | Red | #ef4444 |

### Usage

#### Navigating Large Workflows
1. Look at mini-map to see workflow structure
2. Click on area you want to view
3. Viewport centers on clicked location
4. Zoom in/out for detail

#### Orientation
1. Mini-map shows all nodes at once
2. Identify where you are (blue rectangle)
3. Click to jump to different section
4. Use when lost in large workflow

### Technical Details

**Rendering:**
- Canvas-based rendering for performance
- Updates in real-time as nodes move
- <50ms render time for 100 nodes

**Coordinate System:**
- Auto-scales to fit all nodes
- Maintains aspect ratio
- Click position maps to canvas coordinates

### Performance

- Updates: Real-time (<50ms)
- Memory: ~5 KB
- Canvas-based: Hardware accelerated

---

## 5. Search and Filter

### Overview

Quickly find nodes by ID, name, or type with real-time search.

### Features

- **Real-time Search:** Results update as you type
- **Search Fields:** ID, name, type
- **Case-Insensitive:** "Agent" matches "agent"
- **Match Counter:** "3 of 12 matches"
- **Navigation:** Next/Previous buttons
- **Keyboard Support:** F3 for next match

### Search Bar

```
[🔍 Search nodes...] [3 of 12] [←] [→] [✕]
```

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl + F` | Focus search bar |
| `Enter` | Submit search |
| `F3` | Next match |
| `Shift + F3` | Previous match |
| `Escape` | Clear search |

### Search Examples

#### Find All Agent Nodes
1. Press `Ctrl + F`
2. Type "agent"
3. See all agent nodes highlighted
4. Press `F3` to navigate between them

#### Find by Name
1. Press `Ctrl + F`
2. Type "research"
2. Finds "Research Planning", "Data Collection", etc.

#### Find by Type
1. Press `Ctrl + F`
2. Type "team"
3. Finds all team nodes

### Technical Details

**Search Algorithm:**
```typescript
nodes.filter(node =>
  node.id.toLowerCase().includes(query) ||
  node.data.label?.toLowerCase().includes(query) ||
  node.type.toLowerCase().includes(query)
)
```

**Performance:**
- 100 nodes: <100ms
- 50 nodes: <50ms
- Real-time filtering

### Best Practices

1. **Specific Terms:** Use unique parts of node names
2. **Type Search:** Search by type to find all nodes of a kind
3. **Partial Matches:** Use partial terms for broader results
4. **Clear Often:** Press `Escape` to clear search when done

---

## 6. Node Grouping

### Overview

Organize related nodes into color-coded groups for better workflow organization.

### Features

- **Multi-Select:** Select 2+ nodes to group
- **Custom Labels:** Name your groups
- **Color-Coded:** 8 colors to choose from
- **Collapse/Expand:** Hide/show grouped nodes
- **Visual Badges:** Group labels on nodes

### Creating Groups

#### Step-by-Step
1. Select 2 or more nodes (click nodes or `Ctrl + A`)
2. Click "Group" button (top-right of canvas)
3. Enter group label (e.g., "Authentication Flow")
4. Select group color
5. Click "Create Group"

#### Group Colors
- Purple (default)
- Blue
- Green
- Amber
- Red
- Pink
- Indigo
- Teal

### Group Management

#### View Groups
- Group counter in toolbar: "2 groups"
- Visual badges on grouped nodes
- Color-coded node borders

#### Collapse/Expand
- Click group badge to toggle
- Collapsed groups show summary
- Expanded groups show all nodes

### Use Cases

#### Organize by Feature
```
Group: "Authentication"
- Agent: "Validate User"
- Team: "Auth Review"
- Agent: "Generate Token"
```

#### Organize by Stage
```
Group: "Data Processing"
- Compute: "Extract"
- Transform: "Clean"
- Compute: "Load"
```

#### Organize by Team
```
Group: "Frontend Tasks"
- Agent: "UI Developer"
- Agent: "UX Designer"
- HITL: "Design Review"
```

### Limitations (v2.0)

- Cannot ungroup nodes
- Cannot add/remove nodes from groups
- No nested groups
- Workaround: Delete group and recreate

### Future Enhancements (v2.1)

- Ungroup functionality
- Add/remove nodes from groups
- Nested groups
- Group-based operations (move, delete together)

---

## 7. Workflow Templates

### Overview

Pre-built workflow templates for common use cases. Insert complete workflows with one click.

### Available Templates

#### 🐛 Bug Fix Workflow
**Category:** Coding
**Nodes:** 4
**Flow:** Analyze Bug → Implement Fix → Code Review Team → Test Fix

#### ✨ Feature Development
**Category:** Coding
**Nodes:** 5
**Flow:** Requirement Analysis → Design Team → Implementation Team → Quality Check → Deploy

#### 🔬 Research Workflow
**Category:** Research
**Nodes:** 4
**Flow:** Research Planning → Data Collection → Analysis Team → Synthesize Findings

#### 👁️ Code Review Workflow
**Category:** Coding
**Nodes:** 5
**Flow:** Receive PR → Review Team + Security + Performance → Synthesize Feedback

#### 📊 Data Pipeline Workflow
**Category:** Data
**Nodes:** 4
**Flow:** Extract → Validate → Transform → Load

#### 🚀 CI/CD Pipeline
**Category:** DevOps
**Nodes:** 6
**Flow:** Trigger Build → Run Tests → Check Results → Deploy Staging → Approval → Deploy Production

### Using Templates

#### Step-by-Step
1. Click "Templates" button in header
2. Browse categories (coding, research, data, devops)
3. Click template card to see details
4. Click "Insert" button
5. Template added to canvas (offset from existing nodes)

#### Template Card
```
┌─────────────────────┐
│ 🐛 Bug Fix Workflow │
│ Standard workflow   │
│ for fixing bugs...  │
│ 4 nodes             │
│ [Insert] [Details]  │
└─────────────────────┘
```

### Creating Custom Templates

#### Edit Templates JSON
1. Open `frontend/src/templates/templates.json`
2. Add template object:
```json
{
  "id": "my_template",
  "name": "My Custom Workflow",
  "description": "Does something amazing",
  "category": "coding",
  "icon": "🎯",
  "nodes": [
    {
      "id": "node1",
      "type": "agent",
      "position": { "x": 100, "y": 0 },
      "data": {
        "label": "My Agent",
        "role": "developer",
        "goal": "Do something"
      }
    }
  ],
  "edges": []
}
```
3. Save file
4. Templates auto-load in sidebar

### Best Practices

1. **Start Simple:** Use templates as starting points
2. **Customize:** Modify templates to fit your needs
3. **Save Patterns:** Create templates for recurring patterns
4. **Share:** Export custom templates for team use

---

## 8. Auto-Layout Algorithms

### Overview

Automatically arrange nodes using three different algorithms for optimal visualization.

### Available Algorithms

#### 1. Hierarchical Layout 🌲
**Best For:** Sequential workflows, pipelines, DAGs

**Algorithm:**
- Topological sort for levels
- Nodes grouped by dependency depth
- Horizontal spacing within levels
- Vertical spacing between levels

**Parameters:**
- Level height: 200px
- Node width: 250px

**Example:**
```
Level 0:  [Start]
Level 1:  [Process1] [Process2]
Level 2:    [Review]
Level 3:      [End]
```

#### 2. Force-Directed Layout 🕸️
**Best For:** Complex workflows, networks, cyclic graphs

**Algorithm:**
- Repulsion between all nodes
- Attraction along edges
- Iterative convergence
- Random initial positioning

**Parameters:**
- Repulsion: 5000 force
- Attraction: 0.01 coefficient
- Iterations: 50
- Damping: 0.9

**Example:**
```
     [A]
    / | \
  [B][C][D]
    \ | /
     [E]
```

#### 3. Grid Layout 📐
**Best For:** Parallel workflows, equal-status nodes

**Algorithm:**
- Square root of node count for columns
- Left-to-right, top-to-bottom
- Fixed spacing

**Parameters:**
- Node width: 250px
- Node height: 150px

**Example:**
```
[Node1] [Node2] [Node3]
[Node4] [Node5] [Node6]
[Node7] [Node8] [Node9]
```

### Using Auto-Layout

#### Step-by-Step
1. Add nodes to canvas
2. Click layout button in toolbar (top-center)
3. Confirm layout application
4. Nodes reposition automatically
5. Use `Ctrl + Z` to undo if needed

#### Layout Toolbar
```
[Auto-Layout:] [🌲 Hierarchical] [🕸️ Force-Directed] [📐 Grid]
```

### Comparison

| Algorithm | Speed | Quality | Best For |
|-----------|-------|---------|----------|
| Hierarchical | Fast | Best for DAGs | Sequential workflows |
| Force-Directed | Medium | Organic | Complex networks |
| Grid | Fast | Organized | Parallel tasks |

### Tips

1. **Hierarchical:** Use for workflows with clear direction
2. **Force-Directed:** Use for interconnected nodes
3. **Grid:** Use for independent parallel tasks
4. **Experiment:** Try all three to find best fit
5. **Undo:** Always use `Ctrl + Z` to revert if needed

---

## 9. Performance Optimization

### Large Workflows (50+ nodes)

#### Mini-Map Navigation
- Use mini-map for overview
- Click to jump to sections
- Reduces manual scrolling

#### Search and Filter
- Search instead of manual hunting
- F3 to navigate between matches
- Fast node location

#### Zoom and Pan
- Zoom out for overview
- Zoom in for details
- Fit to screen for context

### Memory Management

#### Undo/Redo
- 100 operation limit
- Automatic pruning
- ~65 KB memory usage

#### Canvas Rendering
- Hardware acceleration
- Virtual scrolling (planned)
- Lazy loading (planned)

### Performance Metrics

| Operation | Time | Notes |
|-----------|------|-------|
| Undo/Redo | <50ms | State restoration |
| Search (100 nodes) | <100ms | Real-time |
| Layout (50 nodes) | <500ms | Force-directed |
| Mini-Map Render | <50ms | Canvas-based |
| Zoom/Pan | <16ms | 60fps |

---

## 10. Customization

### Adding Custom Templates

#### Location
`frontend/src/templates/templates.json`

#### Format
```json
{
  "templates": [
    {
      "id": "unique_id",
      "name": "Template Name",
      "description": "What it does",
      "category": "coding|research|data|devops",
      "icon": "🎯",
      "nodes": [...],
      "edges": [...]
    }
  ]
}
```

### Modifying Keyboard Shortcuts

#### Location
`frontend/src/components/KeyboardShortcuts/KeyboardShortcuts.tsx`

#### Steps
1. Find `handleKeyDown` function
2. Modify event.key combinations
3. Update help modal text
4. Test changes

### Adjusting Layout Algorithms

#### Location
`frontend/src/store/useWorkflowStore.ts`

#### Parameters
```typescript
// Hierarchical
levelHeight: 200
nodeWidth: 250

// Force-Directed
repulsion: 5000
attraction: 0.01
iterations: 50
damping: 0.9

// Grid
nodeWidth: 250
nodeHeight: 150
```

### Custom Colors

#### Location
CSS variables in `frontend/src/index.css`

#### Modify
```css
--agent-color: #3b82f6;
--team-color: #8b5cf6;
--compute-color: #10b981;
```

---

## Troubleshooting

### Undo/Redo Not Working

**Problem:** Ctrl+Z has no effect

**Solutions:**
1. Check if at start of history (no operations to undo)
2. Look for disabled undo button in toolbar
3. Try clicking undo button instead of keyboard

### Mini-Map Not Showing

**Problem:** Mini-map is empty

**Solutions:**
1. Add nodes to canvas first
2. Check if nodes are off-screen
3. Use "Fit to Screen" to reset view

### Search Not Finding Nodes

**Problem:** Search returns no results

**Solutions:**
1. Check spelling
2. Try partial matches
3. Search by node type instead of name
4. Clear search and try again

### Layout Looks Wrong

**Problem:** Nodes overlap or are scattered

**Solutions:**
1. Press `Ctrl + Z` to undo
2. Try a different layout algorithm
3. Manually adjust positions
4. Check for circular dependencies (hierarchical)

---

## FAQ

**Q: How many nodes can the editor handle?**
A: Tested up to 200 nodes. Performance remains good with mini-map and search.

**Q: Can I save my own templates?**
A: Currently, you must edit `templates.json` directly. UI template editor planned for v2.1.

**Q: How do I ungroup nodes?**
A: Delete the group and recreate. Ungroup functionality planned for v2.1.

**Q: Can I nest groups?**
A: Not in v2.0. Nested groups planned for v2.1.

**Q: Does search support regex?**
A: No, only simple substring search. Regex support planned for v2.1.

**Q: Can I resize the mini-map?**
A: Not in v2.0 (fixed at 200x150). Resizable mini-map planned for v2.1.

**Q: How do I share custom templates?**
A: Share your `templates.json` file with teammates.

---

## Summary

The Victor Workflow Editor v2.0 provides comprehensive advanced features:

### ✅ Implemented
- Undo/Redo (100 operations)
- Zoom/Pan (10%-200%)
- 20+ Keyboard shortcuts
- Mini-Map navigation
- Real-time search
- Node grouping (8 colors)
- 6 Pre-built templates
- 3 Auto-layout algorithms

### 🚀 Coming Soon (v2.1)
- Ungroup functionality
- Template editor UI
- Regex search
- Resizable mini-map
- Nested groups
- Multi-user collaboration

### 📖 Resources
- FEATURES.md - Complete feature documentation
- templates.json - Template definitions
- useWorkflowStore.ts - State management
- KeyboardShortcuts.tsx - Shortcut definitions

---

**Version:** 2.0.0
**Last Updated:** 2026-01-15
**Status:** Production Ready ✅
