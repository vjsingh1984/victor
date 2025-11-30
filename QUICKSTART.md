# Quick Start Guide

Canonical quickstart lives in `docs/guides/QUICKSTART.md` and `docker/README.md`; this file is retained for convenience.

This guide helps you test drive Victor with Docker in under 5 minutes.

## Prerequisites

- Docker Desktop installed and running
- At least 8GB RAM available
- 10GB free disk space

## Step 1: Start Ollama Service

```bash
# Start Ollama in the background
docker-compose --profile ollama up -d

# Wait for Ollama to be ready (30-60 seconds)
docker exec victor-ollama ollama --version

# Pull a small, fast model for testing
docker exec victor-ollama ollama pull qwen2.5-coder:1.5b
```

## Step 2: Initialize Victor Configuration

```bash
# Initialize Victor (creates profiles.yaml)
docker-compose run --rm victor victor init

# Verify Ollama connection
docker-compose run --rm victor victor test-provider ollama
```

## Step 3: Test Victor Interactive Mode

```bash
# Start Victor in interactive mode
docker-compose run --rm victor victor main

# Inside Victor, try these commands:
# "Write a Python function to calculate fibonacci numbers"
# "Create a simple REST API endpoint"
# Type 'exit' to quit
```

## Step 4: Test One-Shot Command

```bash
# Run Victor with a one-shot command
docker-compose run --rm victor victor main "Write a Python function to reverse a string"
```

**Note**: First inference on CPU may take 30-60 seconds as the model loads. Subsequent requests will be faster.

## Step 5: Run the FastAPI Demo (Optional)

This demo shows Victor building a complete production-ready web application:

```bash
# Run the demo (takes ~5 minutes)
docker exec victor-demo bash /app/demos/run_fastapi_demo.sh
```

## Troubleshooting

### Ollama not responding
```bash
# Check Ollama logs
docker logs victor-ollama

# Restart Ollama
docker-compose restart ollama
```

### Port already in use
```bash
# Check what's using port 11434
lsof -i :11434

# Stop conflicting services or change port in docker-compose.yml
```

### Out of memory
```bash
# Use a smaller model
docker exec victor-ollama ollama pull qwen2.5-coder:0.5b
```

### Debugging and Logging

Victor supports runtime logging control for debugging:

```bash
# Enable debug logging for detailed troubleshooting
victor --log-level DEBUG main "your query"

# Set logging level to INFO (default)
victor --log-level INFO main "your query"

# Only show warnings and errors
victor --log-level WARN main "your query"

# Only show errors
victor --log-level ERROR main "your query"
```

You can also set the default logging level via environment variable:

```bash
# Set default logging level
export VICTOR_LOG_LEVEL=DEBUG
victor main "your query"

# CLI argument overrides environment variable
victor --log-level INFO main "your query"  # Uses INFO, not DEBUG
```

Valid logging levels: `DEBUG`, `INFO`, `WARN`, `WARNING`, `ERROR`, `CRITICAL`

### Extended Thinking Mode

Victor supports extended thinking/reasoning mode for compatible models (like Claude):

```bash
# Enable thinking mode for complex reasoning tasks
victor --thinking main "Design a distributed caching system with fault tolerance"

# Combine with logging to see reasoning process
victor --thinking --log-level DEBUG main "your complex query"

# Works in interactive mode too
victor --thinking
```

**What it does:**
- Enables the model's extended reasoning capabilities (Claude's thinking mode)
- Shows the model's step-by-step reasoning process
- Useful for complex tasks requiring careful analysis
- Allocates additional compute budget for deeper reasoning (10,000 tokens)

**When to use:**
- Complex architectural decisions
- Multi-step problem solving
- Code review requiring deep analysis
- Design pattern selection
- Performance optimization strategies

**Note:** Currently supported by Anthropic Claude models. Other providers will ignore this flag gracefully.

## Next Steps

Once you've verified Victor works:

1. **Configure profiles**: Edit `~/.victor/profiles.yaml` to add API keys
2. **Try other models**: `docker exec victor-ollama ollama pull codellama:7b`
3. **Use cloud providers**: Add ANTHROPIC_API_KEY, OPENAI_API_KEY to `.env`
4. **Explore enterprise tools**: See `docs/ENTERPRISE.md` for advanced features

## Stopping Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (clean slate)
docker-compose down -v
```

## Support

- Documentation: `docs/`
- Issues: https://github.com/vjsingh1984/victor/issues
- Email: singhvjd@gmail.com
