# 60-Second Quickstart (Local, No API Key)

This path runs entirely on your machine.

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

This creates `~/.victor/profiles.yaml` with a working local profile.

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
- The default model is `qwen2.5-coder:7b` in `~/.victor/profiles.yaml`.
