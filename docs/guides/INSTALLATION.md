# Installation

Short install paths for the Victor CLI.

## Quick Install
```bash
pipx install victor-ai
# or
pip install victor-ai
```

Verify:
```bash
victor --version
```

## Local Dev (Source)
```bash
git clone https://github.com/vijayksingh/victor.git
cd victor
pip install -e ".[dev]"
```

## Docker
```bash
docker pull vjsingh1984/victor:latest
docker run -it --rm -v $(pwd):/workspace -v ~/.victor:/home/victor/.victor vjsingh1984/victor:latest victor chat
```

## Post-Install
```bash
victor init
```

Configure a cloud provider (optional):
```bash
victor keys --set anthropic --keyring
victor chat --provider anthropic --model claude-sonnet-4-5
```

## Uninstall
```bash
pipx uninstall victor-ai
# or
pip uninstall victor-ai
```

## Help
- Issues: https://github.com/vijayksingh/victor/issues
- Docs index: `../README.md`
