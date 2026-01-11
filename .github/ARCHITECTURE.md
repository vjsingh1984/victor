# GitHub Actions Validation Architecture

## Workflow Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Pull Request Event                          │
│                    (opened, synchronize, reopened)                  │
└──────────────────────────┬──────────────────────────────────────────┘
                            │
                            ├─────────────────────────────────────────────┐
                            │                                             │
                            ▼                                             ▼
            ┌───────────────────────────┐               ┌───────────────────────────┐
            │  Path: feps/**            │               │  Path: victor/**          │
            │  Trigger FEP Validation   │               │  Trigger Vertical Valid.  │
            └───────────┬───────────────┘               └───────────┬───────────────┘
                        │                                           │
                        ▼                                           ▼
    ┌───────────────────────────────────┐       ┌───────────────────────────────────┐
    │   FEP Validation Workflow         │       │   Vertical Validation Workflow   │
    │   (.github/workflows/             │       │   (.github/workflows/             │
    │    fep-validation.yml)            │       │    vertical-validation.yml)      │
    └───────────┬───────────────────────┘       └───────────┬───────────────────────┘
                │                                           │
                │                                           │
                ▼                                           ▼
    ┌───────────────────────────────────┐       ┌───────────────────────────────────┐
    │ 1. Find modified FEP files        │       │ 1. Find modified vertical TOMLs   │
    │ 2. Run victor fep validate        │       │ 2. Validate TOML schema           │
    │ 3. Check YAML frontmatter         │       │ 3. Verify class exists            │
    │ 4. Verify required sections       │       │ 4. Check inheritance              │
    │ 5. Validate content quality       │       │ 5. Validate entry points          │
    │ 6. Check numbering consistency    │       │ 6. Verify provided tools          │
    │ 7. Generate validation report     │       │ 7. Generate validation report     │
    └───────────┬───────────────────────┘       └───────────┬───────────────────────┘
                │                                           │
                │                                           │
                ▼                                           ▼
    ┌───────────────────────────────────┐       ┌───────────────────────────────────┐
    │  Upload Artifact                  │       │  Upload Artifact                  │
    │  (fep-validation-report.md)       │       │  (vertical-validation-report.md)  │
    └───────────┬───────────────────────┘       └───────────┬───────────────────────┘
                │                                           │
                │                                           │
                ▼                                           ▼
    ┌───────────────────────────────────┐       ┌───────────────────────────────────┐
    │  Post PR Comment with Results     │       │  Post PR Comment with Results     │
    │  - Validation status              │       │  - Validation status              │
    │  - Error details                  │       │  - Error details                  │
    │  - Next steps                     │       │  - Next steps                     │
    └───────────┬───────────────────────┘       └───────────┬───────────────────────┘
                │                                           │
                └───────────────────┬───────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────────┐
                    │   PR Comment Helper Workflow      │
                    │   (.github/workflows/             │
                    │    pr-comment.yml)                │
                    └───────────┬───────────────────────┘
                                │
                                ▼
                    ┌───────────────────────────────────┐
                    │ 1. Detect PR type                 │
                    │    (FEP, vertical, or code)       │
                    │                                   │
                    │ 2. Post welcome comment           │
                    │    (if new PR)                    │
                    │                                   │
                    │ 3. Provide context-specific       │
                    │    guidance                       │
                    │                                   │
                    │ 4. Update validation status       │
                    │    (if PR updated)                │
                    │                                   │
                    │ 5. Combine results from all       │
                    │    validation workflows           │
                    └───────────┬───────────────────────┘
                                │
                                ▼
                    ┌───────────────────────────────────┐
                    │  Final PR Comment                 │
                    │  - Welcome message                │
                    │  - Validation results             │
                    │  - Context-specific guidance      │
                    │  - Next steps                     │
                    │  - Documentation links            │
                    └───────────────────────────────────┘
```

## Validation Pipeline

### FEP Validation Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│ Step 1: File Detection                                              │
├─────────────────────────────────────────────────────────────────────┤
│ git diff --name-only --diff-filter=AM origin/main...HEAD           │
│ grep '^feps/fep-.*\.md$'                                            │
└────────────────────────────┬───────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 2: Schema Validation                                           │
├─────────────────────────────────────────────────────────────────────┤
│ from victor.feps import parse_fep_metadata                          │
│ metadata = parse_fep_metadata(content)                              │
│ ✓ Check required fields: fep, title, type, status, dates, authors  │
└────────────────────────────┬───────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 3: Section Validation                                          │
├─────────────────────────────────────────────────────────────────────┤
│ REQUIRED_SECTIONS = {                                               │
│   "Summary", "Motivation", "Proposed Change", ...                   │
│ }                                                                    │
│ ✓ Check all sections present                                        │
│ ✓ Check minimum word counts                                         │
└────────────────────────────┬───────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 4: Number Consistency                                          │
├─────────────────────────────────────────────────────────────────────┤
│ filename: fep-0002-feature.md                                       │
│ frontmatter: fep: 2                                                │
│ ✓ Parse filename number                                            │
│ ✓ Parse frontmatter number                                         │
│ ✓ Verify they match                                                 │
└────────────────────────────┬───────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 5: Report Generation                                           │
├─────────────────────────────────────────────────────────────────────┤
│ ✓ Create validation report                                          │
│ ✓ Upload as artifact (30-day retention)                             │
│ ✓ Post as PR comment                                                │
│ ✓ Set CI status (pass/fail)                                         │
└─────────────────────────────────────────────────────────────────────┘
```

### Vertical Validation Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│ Step 1: Directory Detection                                         │
├─────────────────────────────────────────────────────────────────────┤
│ git diff --name-only origin/main...HEAD                             │
│ grep '^victor/.*/victor-vertical.toml$'                             │
└────────────────────────────┬───────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 2: TOML Schema Validation                                      │
├─────────────────────────────────────────────────────────────────────┤
│ from victor.core.verticals.package_schema import VerticalPackageMetadata│
│ metadata = VerticalPackageMetadata.from_toml(path)                  │
│ ✓ Validate required fields                                          │
│ ✓ Check name pattern (lowercase, alphanumeric)                      │
│ ✓ Validate semantic version                                         │
│ ✓ Check authors list                                                │
└────────────────────────────┬───────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 3: Class Implementation Validation                             │
├─────────────────────────────────────────────────────────────────────┤
│ module = __import__(metadata.class_spec.module)                     │
│ cls = getattr(module, metadata.class_spec.class_name)               │
│ ✓ Check class exists                                                │
│ ✓ Verify inherits from VerticalBase                                 │
│ ✓ Test instantiation                                                │
└────────────────────────────┬───────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 4: Entry Point Validation                                      │
├─────────────────────────────────────────────────────────────────────┤
│ [project.entry-points."victor.verticals"]                           │
│ myvertical = "victor_myvertical:MyVertical"                         │
│ ✓ Check entry point exists in pyproject.toml                        │
│ ✓ Verify entry point matches TOML spec                              │
└────────────────────────────┬───────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 5: Tool Validation                                             │
├─────────────────────────────────────────────────────────────────────┤
│ for tool in metadata.class_spec.provides_tools:                     │
│   registry.get_tool(tool)                                           │
│ ✓ Verify tools exist in registry                                    │
│ ⚠ Warn if not found (may be dynamically registered)                 │
└────────────────────────────┬───────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 6: Report Generation                                           │
├─────────────────────────────────────────────────────────────────────┤
│ ✓ Create validation report                                          │
│ ✓ Upload as artifact (30-day retention)                             │
│ ✓ Post as PR comment                                                │
│ ✓ Set CI status (pass/fail)                                         │
└─────────────────────────────────────────────────────────────────────┘
```

## CI/CD Integration

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Pull Request                                 │
└──────┬──────────────────────────────────────────────┬───────────────┘
       │                                              │
       ├─────────────────────┬────────────────────────┤
       │                     │                        │
       ▼                     ▼                        ▼
┌──────────────┐    ┌──────────────┐       ┌──────────────┐
│ FEP          │    │ Vertical     │       │ Code         │
│ Validation   │    │ Validation   │       │ CI           │
└──────┬───────┘    └──────┬───────┘       └──────┬───────┘
       │                   │                       │
       └─────────┬─────────┴───────────────────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ PR Status Check  │
        │ (All must pass)  │
        └────────┬─────────┘
                 │
        ┌────────▼─────────┐
        │ Mergeable?       │
        └────────┬─────────┘
                 │
        ┌────────▼─────────┐
        │ PR Approved &    │
        │ CI Passed        │
        └────────┬─────────┘
                 │
        ┌────────▼─────────┐
        │ Ready to Merge   │
        └──────────────────┘
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                          GitHub Events                              │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Workflow Execution                               │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  1. Checkout Code                                             │ │
│  │  2. Setup Python (with caching)                               │ │
│  │  3. Install Dependencies (CPU-only torch)                     │ │
│  │  4. Run Validation                                            │ │
│  │  5. Generate Report                                           │ │
│  │  6. Upload Artifact                                           │ │
│  │  7. Post PR Comment                                           │ │
│  └───────────────────────────────────────────────────────────────┘ │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          Outputs                                    │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  • PR Comments (immediate feedback)                           │ │
│  │  • Artifacts (detailed reports, 30-day retention)            │ │
│  │  • CI Status (pass/fail)                                      │ │
│  │  • Workflow Logs (debugging)                                  │ │
│  └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

## Error Handling Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Validation Runs                             │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                    ┌────────┴────────┐
                    │                 │
                    ▼                 ▼
              ┌──────────┐      ┌──────────┐
              │ Success  │      │ Failure  │
              └─────┬────┘      └─────┬────┘
                    │                 │
                    ▼                 ▼
        ┌───────────────────┐  ┌───────────────────┐
        │ ✓ Post Success    │  │ ✗ Post Failure    │
        │   Comment         │  │   Comment with     │
        │ ✓ Set CI = Pass   │  │   Error Details   │
        │ ✓ Upload Report   │  │ ✗ Set CI = Fail   │
        └───────────────────┘  │ ✓ Upload Report   │
                               └───────────────────┘

                    │                 │
                    └────────┬────────┘
                             ▼
                    ┌───────────────────┐
                    │ Contributor Reviews│
                    │ Errors & Fixes    │
                    └────────┬───────────┘
                             │
                             ▼
                    ┌───────────────────┐
                    │ Push Fix Commit   │
                    │ (triggers re-run) │
                    └───────────────────┘
```

## Performance Optimizations

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Caching Strategy                               │
├─────────────────────────────────────────────────────────────────────┤
│  • pip cache: Python dependencies                                   │
│  • CPU-only torch: Avoid 800MB GPU download                         │
│  • fetch-depth: 0: Full history for diff only                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                   Concurrency Control                               │
├─────────────────────────────────────────────────────────────────────┤
│  concurrency:                                                       │
│    group: ${{ github.workflow }}-${{ github.ref }}                 │
│    cancel-in-progress: true                                         │
│                                                                     │
│  Effect: Cancel old runs when new commits pushed                    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      Targeted Execution                             │
├─────────────────────────────────────────────────────────────────────┤
│  on:                                                                │
│    pull_request:                                                    │
│      paths:                                                         │
│        - 'feps/**'      # FEP validation only                       │
│        - 'victor/**'     # Vertical validation only                │
│                                                                     │
│  Effect: Don't run when files unchanged                             │
└─────────────────────────────────────────────────────────────────────┘
```

## Security Model

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Minimal Permissions                            │
├─────────────────────────────────────────────────────────────────────┤
│  permissions:                                                       │
│    pull-requests: write  # Post comments                            │
│    contents: read        # Read code                                │
│                                                                     │
│  No secrets required                                               │
│  Uses GITHUB_TOKEN (automatically provided)                        │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      Dependency Management                          │
├─────────────────────────────────────────────────────────────────────┤
│  • Install only what's needed: pip install -e ".[dev]"            │
│  • Use CPU-only torch for faster installs                          │
│  • No external API keys or credentials                             │
└─────────────────────────────────────────────────────────────────────┘
```
