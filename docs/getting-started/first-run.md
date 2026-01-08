# First Run Guide

This guide gets you productive in a few minutes with Victor.

## 1) Initialize

```bash
victor init
```

This creates `~/.victor/profiles.yaml` and a project context file.

## 2) Start a chat

```bash
victor chat
```

## 3) Try these prompts

- "Summarize this repo and list risky areas."
- "Generate unit tests for src/utils.py."
- "Refactor src/api.py to improve readability."

## 4) One-shot mode

```bash
victor "explain src/index.ts and suggest improvements"
```

## 5) Switch models

Edit `~/.victor/profiles.yaml` to change the provider or model.

If you want a local-only setup with no API keys, follow `../QUICKSTART_60S.md`.
