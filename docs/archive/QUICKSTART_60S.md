# 60-Second Quickstart (Local, No API Key)

This path uses local models and avoids API keys.

## 1) Install Victor

```bash
pipx install victor-ai
```

If you do not use pipx:

```bash
pip install victor-ai
```

## 2) Install a local model (Ollama)

```bash
ollama pull qwen2.5-coder:7b
```

## 3) Initialize

```bash
victor init
```

This creates `~/.victor/profiles.yaml`; review it to pick a local model.

## 4) Chat

```bash
victor chat
```

## Optional: One-shot command

```bash
victor "write unit tests for src/utils.py"
```

## Troubleshooting

- If `victor chat` cannot reach Ollama, verify `ollama serve` is running.
- If you want a different model, update `~/.victor/profiles.yaml`.
