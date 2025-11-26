# Victor Docker - Air-gapped Deployment

This Docker image includes **pre-downloaded embedding models** for 100% offline operation.

## Features

✅ **Air-gapped Capable**: Works without internet after build
✅ **Pre-bundled Model**: all-MiniLM-L12-v2 (120MB) included
✅ **Fast Startup**: No model download on first run
✅ **Unified Model**: Same model for tool selection and codebase search
✅ **Memory Optimized**: 120MB model (40% reduction from separate models)

## Quick Start

### Build Air-gapped Image

```bash
# Build Docker image with embedded model
docker build -t victor:airgapped .
```

### Test Air-gapped Setup

```bash
# Run air-gapped test
docker run --rm victor:airgapped bash /app/docker/scripts/test_airgapped.sh
```

### Run Air-gapped Demo

```bash
# Run codebase search demo (100% offline)
docker run --rm victor:airgapped python3 /app/examples/airgapped_codebase_search.py
```

See full documentation in this file for more details.
