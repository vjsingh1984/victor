# Docker Deployment

Run Victor in a containerized environment.

## Quick Start

```bash
docker pull ghcr.io/vjsingh1984/victor:latest
docker run -it -v ~/.victor:/root/.victor ghcr.io/vjsingh1984/victor:latest
```

## Options

### Interactive Shell
```bash
docker run -it ghcr.io/vjsingh1984/victor:latest bash
```

### Mount Project Directory
```bash
docker run -it -v $(pwd):/workspace ghcr.io/vjsingh1984/victor:latest
```

### With API Keys
```bash
docker run -it \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  -v ~/.victor:/root/.victor \
  ghcr.io/vjsingh1984/victor:latest
```

### With Local Model (Ollama)
```bash
# Run Ollama
docker run -d -v ollama:/root/.ollama -p 11434:11434 ollama/ollama

# Run Victor connected to Ollama
docker run -it \
  --network host \
  -v ~/.victor:/root/.victor \
  ghcr.io/vjsingh1984/victor:latest \
  victor chat --provider ollama
```

## Docker Compose

```yaml
version: '3.8'
services:
  victor:
    image: ghcr.io/vjsingh1984/victor:latest
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    volumes:
      - ~/.victor:/root/.victor
      - ./workspace:/workspace
    working_dir: /workspace
    stdin_open: true
    tty: true
```

Run with:
```bash
docker-compose up -d
docker-compose exec victor victor chat
```

## Build from Source

```bash
git clone https://github.com/vjsingh1984/victor
cd victor
docker build -t victor:local .
```

## Troubleshooting

**Permission denied?**
```bash
# Fix volume permissions
docker run -it -v ~/.victor:/root/.victor --user $(id -u):$(id -g) ghcr.io/vjsingh1984/victor:latest
```

**Can't connect to Ollama?**
```bash
# Use host networking
docker run --network host ghcr.io/vjsingh1984/victor:latest
```

## Next Steps

- [Installation](installation.md) - Other installation methods
- [Local Models](local-models.md) - Ollama setup
- [Configuration](../user-guide/configuration.md) - Advanced configuration
