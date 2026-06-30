---
fep: 0012
title: "Shipped RL-trained edge classifier (replace the Ollama edge default)"
type: Standards Track
status: Draft
created: 2026-06-30
modified: 2026-06-30
authors:
  - name: Vijaykumar Singh
    email: singhvjd@gmail.com
    github: vjsingh1984
reviewers: []
discussion: https://github.com/vjsingh1984/victor/discussions/0012
---

# FEP-0012: Shipped RL-trained edge classifier

## Summary

FEP-0001's edge model defaults to `ollama/qwen3.5:2b` for ~14 micro-decisions
(task-type, completion, loop-detection, tool-necessity, prompt-focus, stage,
…). For client/edge installs that means a separate Ollama server + a ~2.7 GB
model pull just for micro-routing. It is a **soft** dependency (graceful
heuristic fallback exists), and observed logs show **~84 % of decisions already
run on heuristics** vs ~14 % on the LLM — so the Ollama edge model is barely
carrying weight.

This FEP replaces the Ollama edge **default** with a **shipped classic-ML
classifier** trained on the collected decisions + RL outcomes, hosted in the
native/numpy layer (no new heavy runtime dep). Ollama/LLM remains an
**optional auto-detected upgrade**. It removes the 2.7 GB client burden, likely
*improves* micro-decision quality (classic ML beats a 2B model on narrow
classification), and turns 20K+ logged decisions + RL outcomes into a real
asset.

## Motivation

### Problem Statement

- **Client/edge friction:** requiring Ollama + a 2.7 GB model for micro-routing
  is heavy for a client/edge install of Victor.
- **Low utilization:** the LLM edge model serves only ~14 % of decisions; the
  rest already fall back to keyword heuristics.
- **Noisy training data:** `decisions.jsonl` (20K+ records) is mostly
  heuristic-sourced with many null/0-confidence labels, and — critically — is
  **not joinable to outcomes** (no session/turn id), so it cannot today support
  reward-weighted training.

### Goals

1. A shipped, versioned classifier artifact (universal baseline) that runs
   offline with zero external dependency (numpy core; sklearn dev-only).
2. Reward-weighted training (learn from outcomes, not imitation of a noisy log).
3. Optional per-project RL personalization (on-device delta, never uploaded).
4. Non-breaking: Ollama/LLM edge remains an opt-in upgrade; graceful fallback to
   heuristics throughout.

### Non-Goals

- Generative decisions (`multi_skill_decomposition`, `continuation_action`,
  `compaction`) — inherently generative, stay LLM/heuristic.
- A distilled-transformer model (linear hashing is v1).
- Federated aggregation / uploading deltas across users (privacy + infra — later).
- Changing the **main** model default (the user's choice, not the framework's).

## Proposed Change

### Scoping: universal baseline + per-project RL delta

**Hybrid — not purely global, not per-user.**

- **Universal baseline** (released for **all** users, in the package): a
  pretrained classifier on *aggregate, cleaned* decision+outcome data. Solves
  cold-start.
- **Per-project RL delta** (optional, **on-device**): a small reward-weighted
  overlay learned from that project's own outcomes, stored in the **project DB**
  (`.victor/project.db`), never uploaded.
- **Not per-user-global:** a user's cross-project data is less useful for the
  prompt/project-sensitive decisions and raises privacy concerns.

**Decision taxonomy split** — which decisions the classifier owns:

| Universal (classifier-owned) | Project/prompt-sensitive (heuristic / +features) | Generative (out of scope) |
|---|---|---|
| task_type, task_completion, tool_necessity, loop_detection, error_classification, question_classification, intent_classification | prompt_focus, stage_detection, skill_selection, tool_selection | multi_skill_decomposition, continuation_action, compaction |

### Architecture

`caller → TieredDecisionService → local_classifier tier → LocalClassifierDecisionService`
implementing `LLMDecisionServiceProtocol` (`victor/agent/services/protocols/
decision_service.py`). The service: (1) feature-extracts the decision context
into a fixed-size hashed n-gram vector (hashing trick, `2^18` dims, versioned
`feature_spec_version`); (2) scores `W_universal·x + α·W_delta·x + b` per
DecisionType; (3) calibrates confidence; (4) if confidence < τ, returns the
heuristic result. Weights host: `victor_native` Rust (sub-ms) with a pure-numpy
fallback. The delta host is the project DB.

The LLM edge tier remains in the fallback chain — if Ollama is present and a
decision is below τ, the tiered service can still escalate to it (opt-in boost).

### Data model (the unblock)

The current blocker is that decision records carry no session/turn id. The
correlation spine already exists (`victor/core/context.py`:
`session_id`/`turn_id`/`request_id`); `rl_outcome` + `usage.jsonl` already
carry session/turn + success. This FEP closes the last mile:

- **Enrich decision records** (`chain.py`) with `decision_id`, `session_id`,
  `turn_id`, `turn_idx`, `trace_id`, `model_version`, `feature_spec_version`,
  `feature_digest`.
- **`decision_log` + `decision_outcome`** in the **global** DB (consistent with
  `rl_outcome`; enables cross-project training). The junction carries the
  session outcome (`success`, `quality_score`, `segment_rewards`) and the
  per-decision `attributed_reward` via existing credit assignment
  (`trace_to_credit`, `credit_assignment`).
- **`local_classifier_delta`** in the **project** DB: sparse reward-weighted
  overlay `(decision_type, feature_hash) → weight, samples, sum_reward`, top-K
  bounded per type, L2-decayed.
- **Feature provenance:** `feature_spec_version` pins the hasher config so a
  model trained on spec v1 only consumes v1 features; spec changes invalidate
  old deltas (migrated/zeroed).

### Training pipeline (offline, dev-side → universal artifact)

Clean (drop heuristic-noise labels) → join on `decision_outcome` → featurize
(versioned spec) → reward-supervised train (logistic / GBM) → Platt-calibrate →
A/B vs the LLM edge model (held-out, by-session split) → export compressed
sparse `W` + bias + label maps → `victor/models/edge_classifier_v{N}.npz`.
`sklearn`/`torch` are **dev-only** extras; runtime needs only numpy.

### RL personalization (online, per-project)

On session end / periodic flush: join the turn's decisions → session outcome →
reward `r`; reward-weighted logistic-SGD on the touched feature hashes writes
`local_classifier_delta` (bounded, L2-decayed so the universal model
re-asserts). Pure-numpy/Rust; local only.

## API Changes

- New `LocalClassifierDecisionService` implementing `LLMDecisionServiceProtocol`
  (drop-in for the edge model).
- New `local_classifier` tier in `TieredDecisionService`; becomes the default
  when the artifact is present.
- `DecisionResult.source` gains `local_classifier` / `local_classifier+delta`.
- Settings: `edge_classifier_version`, `feature_spec_version`,
  `USE_LOCAL_CLASSIFIER`, `local_learning_enabled`.

## Compatibility

- **Breaking:** No. Ollama/LLM edge remains available; heuristic fallback
  intact. The default *tier* shifts from `edge` to `local_classifier` when the
  artifact is present.
- **Minimum Python:** 3.10 (unchanged). **New runtime deps:** none (numpy
  already core).

## Drawbacks and Alternatives

- **Drawback:** a shipped baseline can be stale vs a fresh LLM. **Mitigation:**
  versioned artifact + CDN updates + per-project RL delta that corrects locally.
- **Alternative — keep Ollama:** rejected (the 2.7 GB client burden + low
  utilization).
- **Alternative — distilled transformer:** deferred to v2 if linear hashing
  underperforms on some decision type.

## Unresolved Questions

- **α (delta blend weight) and τ (confidence gate) defaults:** tune via the A/B
  harness; proposed α=0.3, τ=0.6 to start.
- **CDN artifact distribution vs bundle-only:** proposed bundle-by-default,
  CDN as a later enhancement.

## Implementation Plan

1. **FEP-0012 doc** (this).
2. **Data model:** enrich decision records + `decision_log` /
   `decision_outcome` (global) / `local_classifier_delta` (project) + migration;
   wire per-decision credit assignment.
3. **Training pipeline** (`scripts/train_edge_classifier.py` + `victor/ml/`).
4. **Inference host** (`victor/processing/native/classifier.py` + Rust).
5. **`LocalClassifierDecisionService`** + tier + bootstrap; flip default.
6. **RL delta** (online, project-DB).
7. **Validation:** A/B classifier vs LLM edge; ship only at parity.

## References

- FEP-0001 (edge model system being replaced as default).
- `victor/agent/edge_model.py`, `victor/agent/services/decision_service.py`,
  `victor/agent/services/tiered_decision_service.py`, `victor/core/context.py`.
- Spec + ASCII diagrams: `/Users/vijaysingh/.claude/plans/lively-squishing-snowflake.md`.

## Review Process

- **Submitted by:** Vijaykumar Singh. **Initial review period:** 14 days.

## Acceptance Criteria

1. `victor chat` with no Ollama resolves micro-decisions via the shipped
   classifier (`source=local_classifier`); session completes.
2. A/B parity (or better) vs the LLM edge model on the universal decision types.
3. Per-project delta grows in `project.db` after a successful task; never
   uploaded.
4. No new runtime dependency; heuristic fallback intact when artifact absent.

---

## Copyright

This FEP is licensed under the Apache License 2.0, same as the Victor project.
