# Victor Docker Quick Reference

Ultra-fast reference for Docker deployment with air-gapped semantic tool selection.

## 🚀 One-Command Setup

```bash
./docker-quickstart.sh
```

**What it does:**
- Builds Victor image (~1.5 GB)
- Starts Ollama server
- Pulls lightweight model (~1 GB)
- Verifies setup
- Total time: ~5 minutes
- Total size: ~2.5 GB

## 📦 What's Included

- ✅ Victor with air-gapped semantic tool selection
- ✅ all-MiniLM-L12-v2 embedding model (pre-downloaded)
- ✅ Tool embeddings cache (pre-computed, 31 tools)
- ✅ qwen2.5-coder:1.5b (default, ~1 GB)
- ✅ Fallback JSON parser for all Ollama models
- ✅ 100% offline capable after setup

## ⚡ Quick Commands

```bash
# Interactive mode
docker-compose run --rm victor

# One-shot command
docker-compose run --rm victor victor main "Write a function to sort arrays"

# Run demo (4 test cases)
docker-compose run --rm victor bash /app/docker/demos/semantic-tools.sh

# List profiles
docker-compose run --rm victor victor profiles

# Use better quality model
docker-compose run --rm victor victor --profile code "Complex task"
```

## 🎯 Profiles

| Profile | Model | Size | Use Case |
|---------|-------|------|----------|
| **default** | qwen2.5-coder:1.5b | ~1 GB | Fast distribution, demos |
| **code** | qwen2.5-coder:7b | ~4.7 GB | Better quality code |
| **advanced** | qwen3-coder:30b | ~18 GB | Production, max quality |
| **general** | llama3.1:8b | ~4.9 GB | Non-coding tasks |

## 🔧 Optional: Add More Models

```bash
# Better code quality (recommended for real work)
docker-compose exec ollama ollama pull qwen2.5-coder:7b

# Maximum quality (for production)
docker-compose exec ollama ollama pull qwen3-coder:30b

# Use them
docker-compose run --rm victor victor --profile code "Your task"
docker-compose run --rm victor victor --profile advanced "Complex task"
```

## 📊 Size Comparison

```
Lightweight Distribution (Default):
├── Victor Image ............. 1.5 GB
├── Ollama ................... 0.5 GB
├── qwen2.5-coder:1.5b ....... 1.0 GB
└── Total .................... 3.0 GB ✅

Production Distribution (Optional):
├── Victor Image ............. 1.5 GB
├── Ollama ................... 0.5 GB
├── qwen3-coder:30b .......... 18 GB
└── Total .................... 20 GB
```

## 🌐 Air-Gapped Deployment

### Export (on online machine)

```bash
# 1. Build and setup
./docker-quickstart.sh

# 2. Save images
docker save -o victor.tar $(docker images -q victor-ai)
docker save -o ollama.tar ollama/ollama:latest

# 3. Export models
docker run --rm -v victor_ollama_data:/data -v $(pwd):/backup \
  alpine tar czf /backup/ollama-models.tar.gz -C /data .

# 4. Copy to offline machine:
#    - victor.tar (1.5 GB)
#    - ollama.tar (0.5 GB)
#    - ollama-models.tar.gz (1 GB)
#    - docker-compose.yml
```

### Import (on offline machine)

```bash
# 1. Load images
docker load -i victor.tar
docker load -i ollama.tar

# 2. Restore models
docker volume create victor_ollama_data
docker run --rm -v victor_ollama_data:/data -v $(pwd):/backup \
  alpine tar xzf /backup/ollama-models.tar.gz -C /data

# 3. Start
docker-compose --profile demo up -d

# 4. Test
docker-compose run --rm victor victor main "Write hello world"
```

## 🐛 Troubleshooting

### Ollama not responding
```bash
docker-compose restart ollama
docker-compose logs ollama

# Or use shared utility
docker-compose run --rm victor bash /app/docker/scripts/wait-for-ollama.sh
```

### Model not found
```bash
docker-compose exec ollama ollama list
docker-compose exec ollama ollama pull qwen2.5-coder:1.5b

# Or use shared utility
bash docker/scripts/ensure-model.sh qwen2.5-coder:1.5b "1 GB"
```

### Out of memory
```bash
# Use lighter model (already default)
docker-compose run --rm victor victor --profile fast "Task"

# Or increase Docker memory
# Docker Desktop → Settings → Resources → Memory: 8 GB
```

### Tool embeddings cache missing
```bash
docker-compose run --rm victor bash /app/docker/scripts/init-embeddings.sh
```

## 📝 Examples

### Simple Function
```bash
docker-compose run --rm victor \
  victor main "Write a function to check if a number is prime"
```

### Multiple Functions
```bash
docker-compose run --rm victor \
  victor main "Write a calculator with add, subtract, multiply, divide"
```

### Email Validator
```bash
docker-compose run --rm victor \
  victor main "Write a function to validate email addresses using regex"
```

### Data Processing
```bash
docker-compose run --rm victor \
  victor main "Write a function to read CSV file and calculate averages"
```

## 📚 Documentation

- **Docker Guide**: `docker/README.md` (coming soon)
- **Embeddings & Air-Gapped**: `docs/embeddings/` directory
- **Tool Calling**: `docs/embeddings/TOOL_CALLING_FORMATS.md`
- **Air-Gapped Deployment**: `docs/embeddings/AIRGAPPED.md`
- **General Docs**: `README.md`

## ⚙️ Configuration

**Profiles**: `docker/profiles.yaml`
**Semantic Selection**: Enabled (threshold: 0.15, top-5 tools)
**Embedding Model**: all-MiniLM-L12-v2 (120MB)
**Tool Embeddings**: Pre-computed (31 tools)

## 🎓 Demo Scripts

Victor includes multiple demonstration scripts:

```bash
# Semantic tool selection demo (4 test cases)
docker-compose run --rm victor bash /app/docker/demos/semantic-tools.sh

# Provider features demo (5 demos: chat, streaming, etc.)
docker-compose run --rm victor python /app/docker/demos/provider-features.py

# FastAPI webapp generation demo
docker-compose run --rm victor bash /app/docker/demos/fastapi-webapp.sh
```

**Demos:**
1. Simple function (factorial)
2. Email validation (regex)
3. Prime checker (with docstrings)
4. Calculator (multiple functions)

**Shows:**
- Semantic tool selection in action
- Similarity scores for each query
- Tool execution (file creation)
- Code quality from qwen2.5-coder:1.5b

## 📈 Performance

| Metric | Value |
|--------|-------|
| Setup Time | ~5 minutes |
| First Run | Instant (cache ready) |
| Inference Speed | ~15 tok/s (1.5B model) |
| RAM Usage | ~2 GB (1.5B model) |
| Disk Space | ~3 GB total |

## 🔄 Upgrade Path

Start lightweight, upgrade as needed:

```bash
# Start: 1.5B model (~1 GB)
./docker-quickstart.sh

# Upgrade to 7B (~4.7 GB) for better quality
docker-compose exec ollama ollama pull qwen2.5-coder:7b
docker-compose run --rm victor victor --profile code "Task"

# Upgrade to 30B (~18 GB) for production
docker-compose exec ollama ollama pull qwen3-coder:30b
docker-compose run --rm victor victor --profile advanced "Task"
```

## 🎯 When to Use Which

**qwen2.5-coder:1.5b (default):**
- Demos and workshops
- Quick evaluation
- Resource-constrained (< 8 GB RAM)
- Simple tasks

**qwen2.5-coder:7b (code profile):**
- Real development work
- Better code quality needed
- Have 16 GB+ RAM
- Complex algorithms

**qwen3-coder:30b (advanced profile):**
- Production code
- Maximum quality required
- Have 32 GB+ RAM
- Mission-critical tasks

## 💡 Pro Tips

1. **Fast demos**: Use default 1.5B model
2. **Real work**: Pull 7B model (`ollama pull qwen2.5-coder:7b`)
3. **Production**: Pull 30B model for critical code
4. **Air-gapped**: Export/import volumes between machines
5. **GPU**: Automatic on Apple Silicon, configure for NVIDIA
6. **Storage**: Use volumes for persistent Ollama models
7. **Speed**: Lighter models = faster inference
8. **Quality**: Larger models = better code

## 🚀 Quick Start Checklist

- [ ] Run `./docker-quickstart.sh`
- [ ] Wait 5 minutes for setup
- [ ] Test: `docker-compose run --rm victor victor main "test"`
- [ ] Try demo: `docker-compose run --rm victor bash /app/docker/demos/semantic-tools.sh`
- [ ] Optional: Pull better models (`docker-compose exec ollama ollama pull qwen2.5-coder:7b`)
- [ ] Start coding!

---

**Total setup time**: 5 minutes
**Total disk space**: ~3 GB (lightweight) or ~6 GB (with 7B model)
**Internet required**: Only for initial setup (then 100% offline)

🎉 **You're ready to code with Victor!**
