# Placeholder Assets List

This document lists the recommended placeholder assets to be created for the Victor documentation.

## Priority Placeholders

### 1. Branding Assets

#### `victor-banner.png` (1200x200px)
- **Purpose**: Landing page header, README header
- **Content**: Victor branding with tagline
- **Style**: Clean, professional, dark theme compatible
- **Location**: `docs/assets/images/victor-banner.png`

#### `victor-logo.png` (512x512px)
- **Purpose**: Main logo for documentation headers, favicon
- **Content**: Victor icon/logo mark
- **Style**: Scalable, works on light/dark backgrounds
- **Location**: `docs/assets/images/victor-logo.png`

#### `victor-logo-icon.png` (128x128px)
- **Purpose**: Smaller icon for inline use, badges
- **Content**: Simplified version of main logo
- **Style**: Minimal, recognizable at small sizes
- **Location**: `docs/assets/images/victor-logo-icon.png`

### 2. Application Screenshots

#### `tui-screenshot.png` (1200x800px)
- **Purpose**: Demonstrate TUI mode interface
- **Content**: Main TUI interface with panels, chat input, tool output
- **Style**: Dark terminal theme, clear readable text
- **Location**: `docs/assets/screenshots/tui-screenshot.png`

#### `tui-workflow-execution.png` (1200x800px)
- **Purpose**: Show workflow execution in TUI
- **Content**: Workflow progress, node execution, streaming output
- **Style**: Active workflow with visual feedback
- **Location**: `docs/assets/screenshots/tui-workflow-execution.png`

#### `cli-chat-mode.png` (1200x800px)
- **Purpose**: Demonstrate CLI chat mode
- **Content**: Terminal showing interactive chat session
- **Style**: Clean terminal output with example conversation
- **Location**: `docs/assets/screenshots/cli-chat-mode.png`

#### `cli-workflow-mode.png` (1200x800px)
- **Purpose**: Show workflow CLI commands
- **Content**: Terminal with workflow validation and execution
- **Style**: Command-line interface examples
- **Location**: `docs/assets/screenshots/cli-workflow-mode.png`

#### `provider-switching.png` (1200x800px)
- **Purpose**: Demonstrate provider switching feature
- **Content**: Terminal showing `/provider` command and response
- **Style**: Clear example of switching between providers
- **Location**: `docs/assets/screenshots/provider-switching.png`

### 3. Architecture Diagrams

#### `architecture-overview.png` (1600x900px)
- **Purpose**: High-level system architecture
- **Content**: Client layer, orchestrator, providers, tools, workflows, verticals
- **Style**: Clean flow diagram with clear component separation
- **Location**: `docs/assets/diagrams/architecture-overview.png`

#### `provider-system-diagram.png` (1600x900px)
- **Purpose**: Provider architecture and inheritance
- **Content**: BaseProvider, concrete providers, tool calling adapters
- **Style**: Class hierarchy or component diagram
- **Location**: `docs/assets/diagrams/provider-system-diagram.png`

#### `workflow-system-diagram.png` (1600x900px)
- **Purpose**: Workflow compilation and execution flow
- **Content**: YAML → Compiler → StateGraph → Executor
- **Style**: Flow diagram showing transformation steps
- **Location**: `docs/assets/diagrams/workflow-system-diagram.png`

#### `tool-pipeline-diagram.png` (1600x900px)
- **Purpose**: Tool calling and execution pipeline
- **Content**: Tool selection, validation, execution, result aggregation
- **Style**: Sequence diagram or flow chart
- **Location**: `docs/assets/diagrams/tool-pipeline-diagram.png`

#### `vertical-architecture-diagram.png` (1600x900px)
- **Purpose**: Vertical system architecture
- **Content**: Base vertical, concrete verticals, tools, workflows
- **Style**: Layered diagram showing vertical structure
- **Location**: `docs/assets/diagrams/vertical-architecture-diagram.png`

#### `multi-agent-coordination.png` (1600x900px)
- **Purpose**: Team formation and coordination
- **Content**: Team formations, roles, communication patterns
- **Style**: Network diagram or sequence diagram
- **Location**: `docs/assets/diagrams/multi-agent-coordination.png`

### 4. Feature Screenshots

#### `codebase-search.png` (1200x800px)
- **Purpose**: Demonstrate semantic codebase search
- **Content**: Search query and results display
- **Style**: Clear example of search functionality
- **Location**: `docs/assets/screenshots/codebase-search.png`

#### `workflow-builder.png` (1200x800px)
- **Purpose**: Show workflow YAML editor/validation
- **Content**: Workflow file with validation output
- **Style**: Code editor with workflow definition
- **Location**: `docs/assets/screenshots/workflow-builder.png`

#### `benchmark-results.png` (1200x800px)
- **Purpose**: Display benchmark execution and results
- **Content**: Benchmark command with results table
- **Style**: Terminal output with formatted results
- **Location**: `docs/assets/screenshots/benchmark-results.png`

## Secondary Placeholders

### Tutorial Assets

#### `quickstart-terminal.png` (1200x800px)
- **Purpose**: Quickstart guide installation steps
- **Location**: `docs/assets/screenshots/quickstart-terminal.png`

#### `first-workflow.png` (1200x800px)
- **Purpose**: First workflow creation tutorial
- **Location**: `docs/assets/screenshots/first-workflow.png`

### Vertical-Specific Assets

#### `coding-vertical-diagram.png` (1600x900px)
- **Purpose**: Coding vertical architecture (AST, LSP, etc.)
- **Location**: `docs/assets/diagrams/coding-vertical-diagram.png`

#### `research-vertical-diagram.png` (1600x900px)
- **Purpose**: Research vertical architecture
- **Location**: `docs/assets/diagrams/research-vertical-diagram.png`

## Creation Guidelines

### Screenshots
1. Use consistent terminal theme (preferably dark)
2. Set terminal font to monospace at 14-16pt
3. Capture at 2x resolution for crispness
4. Trim unnecessary whitespace
5. Add subtle drop shadows for depth
6. Consider adding annotations for clarity

### Diagrams
1. Use consistent color scheme matching Victor brand
2. Include clear labels and legends
3. Maintain good contrast for accessibility
4. Use arrows/lines to show data flow
5. Keep layout clean and uncluttered
6. Consider SVG format for scalability

### Logos/Branding
1. Ensure readability at various sizes
2. Test on both light and dark backgrounds
3. Maintain brand consistency
4. Provide multiple formats (PNG, SVG)
5. Consider creating a favicon version

## Tools Recommended for Creation

### Screenshots
- **macOS**: Built-in screenshot (Cmd+Shift+4), CleanShot X
- **Linux**: GNOME Screenshot, Spectacle
- **Windows**: Snipping Tool, ShareX
- **Cross-platform**: Flameshot (recommended for annotations)

### Diagrams
- **Vector**: Draw.io (diagrams.net), Lucidchart, Excalidraw
- **Code**: Mermaid.js, PlantUML, GraphViz
- **Professional**: Adobe Illustrator, Inkscape

### Image Editing
- **Simple**: Preview (macOS), Paint.NET
- **Advanced**: GIMP, Photoshop, Affinity Photo
- **Optimization**: ImageOptim (macOS), FileOptimizer (cross-platform)

## Asset Status Tracking

Use the following checklist to track asset creation:

- [ ] victor-banner.png
- [ ] victor-logo.png
- [ ] victor-logo-icon.png
- [ ] tui-screenshot.png
- [ ] tui-workflow-execution.png
- [ ] cli-chat-mode.png
- [ ] cli-workflow-mode.png
- [ ] provider-switching.png
- [ ] architecture-overview.png
- [ ] provider-system-diagram.png
- [ ] workflow-system-diagram.png
- [ ] tool-pipeline-diagram.png
- [ ] vertical-architecture-diagram.png
- [ ] multi-agent-coordination.png
- [ ] codebase-search.png
- [ ] workflow-builder.png
- [ ] benchmark-results.png
- [ ] quickstart-terminal.png
- [ ] first-workflow.png
- [ ] coding-vertical-diagram.png
- [ ] research-vertical-diagram.png

## Next Steps

1. Create branding assets first (logo, banner, icon)
2. Capture core screenshots (TUI, CLI modes)
3. Create architecture diagrams (start with high-level overview)
4. Add feature-specific screenshots as documentation is written
5. Optimize all assets for web delivery
6. Test all assets in both light and dark documentation themes

---

**Reading Time:** 4 min
**Last Updated:** February 08, 2026**
