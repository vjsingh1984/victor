<div align="center">

![Victor Banner](assets/victor-banner.png)

# Victor Enterprise Evaluation

*Local‑first AI coding assistance for teams that need control over data and models.*

</div>

This guide is for teams evaluating Victor in enterprise or regulated environments. It focuses on practical rollout, local‑first deployment, and realistic expectations.

## Who This Is For
- Teams that want local models by default, with optional cloud use
- Security‑conscious orgs that need control over data flow
- Engineering leaders running pilots before broader adoption

## Quick Evaluation (1–2 days)
1. Install and run locally with Ollama or vLLM.
2. Try a workflow on a real repo (review, refactor, or tests).
3. Decide where data should live (local only vs. hybrid).

```bash
pipx install victor-ai
victor init
ollama pull qwen2.5-coder:7b
victor chat --provider ollama --model qwen2.5-coder:7b
```

## Deployment Options
### Local‑only (air‑gapped)
- Local models (Ollama / vLLM / LM Studio)
- No network access required for the core workflow
- Good fit for restricted environments

### Hybrid
- Local models for most work
- Optional cloud models for higher quality or long context
- Explicitly configured by profile

### Internal service
- Run Victor in a container or VM
- Integrate with internal auth, logging, and CI

See `../docker/README.md` for container deployment basics.
See `ENTERPRISE_DEPLOYMENT.md` for the full deployment appendix.

## Data & Security Notes
- Local models keep prompts and outputs on your infrastructure.
- Cloud models send data to the provider you choose.
- Disable network tools if you want to enforce local‑only usage.
- Audit and logging hooks are available for integration with your stack.

## Pilot Checklist
- Define a small, representative repo.
- Choose 2–3 tasks (review, refactor, tests).
- Decide on allowed tools and models.
- Measure time saved and quality of outputs.

## Support
Victor is Apache‑2.0 licensed. Commercial support is available on request.

**Contact**: singhvjd@gmail.com
**Subject**: "Victor Enterprise"

## FAQ
**Can Victor run offline?**
Yes, with local models and local tools.

**Can we limit tool access?**
Yes. Tools are gated by mode, configuration, and permissions.

**How do we integrate with CI/CD?**
Use the CLI in pipelines or run Victor as a service and call it via API.
