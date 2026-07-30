# CLAUDE.md — SageMaker Warm-Up

This repo is a small, cheap learning project: train and serve a pCTR (predicted click-through rate) model on AWS SageMaker, using the Avazu CTR dataset, to get hands-on with AWS/SageMaker basics before starting a role managing an ads-ranking team. The build plan lives in `PHASES.md`. This is deliberately scoped small — clarity and understanding *why* each AWS primitive exists matter more than build speed or completeness.

## Why this project exists

Lyft's ML platform, LyftLearn, splits into two halves:
- **LyftLearn Compute (offline)** — training, batch prediction, notebooks. Runs on **AWS SageMaker**.
- **LyftLearn Serving (online)** — real-time inference. Runs on **Kubernetes**, *not* SageMaker endpoints.

So the real-time endpoint phase here (Phase 4) teaches genuine SageMaker/AWS deployment mechanics, but is **not** how Lyft serves models in production. Don't let it get treated as "this is my team's serving path" — it isn't. Flag this distinction anywhere it's relevant, not just once.

## Standing workflow rules

1. **Work only on the phase named.** Never build ahead into future phases, even if it seems efficient. If a task seems to need future-phase work, stop and say so.
2. **Plan before code.** For any phase or nontrivial change, present a plan and wait for approval before implementing.
3. **Acceptance criteria are the contract.** Run each phase's acceptance criteria yourself and show passing output before considering a phase done. Don't mark a phase complete on the strength of code existing — run it.
4. **Every phase ends with a teardown check.** Nothing should be left running (training jobs, endpoints, notebook instances) between sessions unless mid-phase. Phase 6 exists to formalize this, but the discipline applies throughout — always confirm before ending a session.
5. **Budget guardrail.** The Phase 0 billing alarm must exist before any other AWS action. If it's ever unclear whether an action costs money, say so before doing it.
6. **Never modify a completed phase's code without flagging it explicitly first.**
7. **The repo is public — never commit secrets or the AWS account ID.** `scripts/check_secrets.sh` runs automatically as a pre-commit hook (install with `scripts/install-hooks.sh` after a fresh clone). Run `scripts/check_secrets.sh --all` before any push, and at the end of every phase alongside the teardown check. Never bypass with `--no-verify`. Anything containing the real account ID (rendered IAM policies, `.claude/settings.local.json`) is git-ignored; templates with an `ACCOUNT_ID` placeholder are the committed source of truth. If a credential ever does land in a commit, **rotate it** — removing it from history does not un-leak a pushed secret.
8. **Keep `PHASES.md` checkboxes current.** Tasks and acceptance criteria are checkboxes; tick them as work completes, and update the Progress table at the top when a phase closes. A box gets ticked only after the thing was *run* with output shown — never because the code exists (that's rule 3, enforced in the file). `PHASES.md` is the state of the build; `NOTES.md` is the reasoning behind it.

## Contracts digest (locked decisions)

- **Dataset:** Avazu CTR Prediction, 50k-row Kaggle sample — not the full ~40M-row file. `click` is the label; `id` is dropped (not a feature).
- **Model:** `sklearn.linear_model.LogisticRegression`. Not XGBoost, not a built-in SageMaker algorithm — a custom script the project owns, because that's analogous to what a DS/MLE team actually owns at Lyft (vs. platform-provided infra).
- **Training/serving split:** `train.py` and `inference.py` are the two owned artifacts. SageMaker's SKLearn framework container is generic infra underneath both — don't conflate "the container" with "the model."
- **Feature parity:** the fitted encoder from `train.py` must be saved alongside the model and reused unchanged in `inference.py`. Training and serving must never diverge on how a raw column becomes a feature — this is the same training/serving-skew concern as a real feature store.
- **Model inputs**, grouped (see PHASES.md Phase 2 for the full table):
  - *Context:* `hour` (extract hour-of-day, not the raw `YYMMDDHH` string), `C1`
  - *Ad:* `banner_pos`, `site_*`/`app_*` fields, `C15`, `C16`
  - *User/device:* `device_id`, `device_ip`, `device_model`, `device_type`, `device_conn_type` — `device_id`/`device_ip` are near-unique pseudo-identifiers and must be hashed into a small bucket space (or dropped), never one-hot encoded directly
  - *Anonymized/mixed:* `C14`, `C17`–`C21`
  - All groups concatenate into one flat numeric vector — there is no separate "user model" or "ad model."
- **The model is bid-agnostic.** `inference.py` returns a pCTR score only — it must never know about bids, budgets, or auction rules. Ranking logic (`bid × pCTR`, sorting) lives in `rank_candidates.py`, a separate script that plays the calling/auction-service role. Keep this boundary explicit; don't let ranking logic creep into the model artifact.
- **Batch Transform (Phase 5) reuses `inference.py` unchanged** — it should not need endpoint-specific code, confirming the serving code isn't secretly coupled to the real-time path.

## Global conventions

- **Language:** Python 3.12, `boto3` + the `sagemaker` Python SDK. `uv`-managed local repo.
- **One S3 bucket** for the whole project (data in, artifacts out), one clearly-named IAM role.
- Files stay in the repo, not inline in notebooks — `train.py`, `inference.py`, `rank_candidates.py` are real, reviewable files.

## Session start

Open a new session with: *"Read CLAUDE.md and PHASES.md Phase N. Plan before implementing."*
