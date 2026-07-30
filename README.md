# learn-sage

Train and serve a **predicted click-through rate (pCTR)** model end-to-end on AWS
SageMaker, using the [Avazu CTR dataset](https://www.kaggle.com/c/avazu-ctr-prediction).

A deliberately small learning project: the goal is understanding *why each AWS
primitive exists*, not building something production-grade. Every step was run
for real and its acceptance criteria verified.

**Total AWS cost: about $0.02.**

---

## What it does

```
Avazu 50k sample ──▶ S3 ──▶ SageMaker Training Job ──▶ model.tar.gz in S3
                                                             │
                                    ┌────────────────────────┴───────────────┐
                                    ▼                                        ▼
                          real-time endpoint                        Batch Transform
                          (HTTPS, always-on)                        (bulk, no server)
                                    │
                                    ▼
                          rank_candidates.py
                          (bid × pCTR auction)
```

| Phase | What happens | Cost |
|---|---|---|
| 0 | Billing alarm, least-privilege IAM user | $0 |
| 1 | Sample Avazu → train/test split → S3 | ~$0 |
| 2 | Train `LogisticRegression` as a SageMaker Training Job | $0.005 |
| 3 | Register the artifact as a Model + registry version | $0 |
| 4 | Deploy a real-time endpoint, verify, tear down | $0.006 |
| 5 | Batch Transform the same model, no endpoint | ~$0.009 |
| 6 | One-command teardown (`make aws-down`) | $0 |

**Result:** validation **AUC 0.7322**, log loss 0.4063 vs 0.4575 for a
predict-the-base-rate baseline. Predicted CTR 0.1685 against actual 0.1710 —
i.e. well calibrated, which matters because the ranking step multiplies by it.

---

## Repo layout

**Model code** — the part a data science team would own:

| File | Role |
|---|---|
| [`src/features.py`](src/features.py) | **The single implementation of "raw column → feature."** Imported by both training and serving. |
| [`src/train.py`](src/train.py) | Training entry point. Runs inside the SageMaker container. |
| [`src/inference.py`](src/inference.py) | Serving entry point (`model_fn` / `input_fn` / `predict_fn` / `output_fn`). Used unchanged by both the endpoint and batch. |

**Orchestration** — instructions *to* AWS; no ML logic:

| File | Role |
|---|---|
| [`prepare_data.py`](prepare_data.py) | Sample 50k rows from the 40M-row Avazu file, split, write CSVs |
| [`upload_data.py`](upload_data.py) / [`verify_data.py`](verify_data.py) | Push to S3, then verify by reading *back from S3* |
| [`submit_training.py`](submit_training.py) | Launch the training job |
| [`register_model.py`](register_model.py) | Create the Model + Model Registry entry |
| [`deploy_endpoint.py`](deploy_endpoint.py) | Deploy the real-time endpoint (**this one bills continuously**) |
| [`invoke_endpoint.py`](invoke_endpoint.py) | Verify predictions and training/serving parity |
| [`batch_transform.py`](batch_transform.py) | Bulk scoring with no endpoint |
| [`aws_down.py`](aws_down.py) | One-command teardown across every region touched |

**The auction, deliberately separate:**

| File | Role |
|---|---|
| [`rank_candidates.py`](rank_candidates.py) | Ranks ads by `bid × pCTR`. **The model never sees a bid.** |

**Docs:**

| File | Role |
|---|---|
| [`PHASES.md`](PHASES.md) | The build plan and its acceptance criteria — the state of the build |
| [`NOTES.md`](NOTES.md) | What actually happened and why — every decision, and every mistake |
| [`CLAUDE.md`](CLAUDE.md) | Standing rules and locked contracts |
| [`iam/`](iam/) | Policy documents as reviewable files, not console clicks |

---

## Design choices and tradeoffs

### A custom script, not a built-in algorithm

SageMaker offers ready-made algorithms. This project uses its own `train.py`
instead, because that mirrors the real division of labour: **the platform runs
the job; the team owns what runs.** SageMaker provisions the instance, pulls a
container, runs your code, and captures the output. It writes none of the ML.

*Tradeoff:* more code to own, and you inherit the container's dependency
versions. In exchange, every decision that affects predictions is yours.

### Hashing, not one-hot encoding

All 22 features become `"column=value"` tokens fed through a single
`FeatureHasher` into one 262,144-dimension vector.

Why: `site_id` is unbounded in production. A fitted vocabulary can't represent a
value it never saw, so every new site would need a retrain. A hash function has
no vocabulary — an unseen value lands in a bucket immediately.

It also handles the problem features honestly: `device_ip` has **33,635 distinct
values in 40,000 rows** (84% unique). One-hot encoding that hands the model 33k
columns to memorise. See the reasoning in [`src/features.py`](src/features.py).

*Tradeoff:* hash collisions (rare here — ~48k tokens in 262k slots), and you lose
the ability to inspect a named coefficient.

### One feature module, shared by training and serving

`train.py` and `inference.py` both import `features.py`. Neither reimplements
it. Serving also loads a `feature_config.json` saved at training time and
**refuses to start** if it disagrees — a dead endpoint beats one quietly scoring
a feature space the model never saw.

Measured: endpoint and batch predictions matched local scoring to **5.6e-17**
across 10,000 rows, despite different Python and numpy versions in the
container. That's a consequence of there being one implementation, not luck.

This is the small-scale version of what a feature store provides.

### The model is bid-agnostic

`inference.py` returns a probability and nothing else. All auction logic —
`bid × pCTR`, sorting, picking a winner — lives in
[`rank_candidates.py`](rank_candidates.py), which plays the calling service.

Why the boundary is worth defending: the model can be retrained or replaced
without touching auction rules; reserve prices and pacing can change without
retraining; and pCTR stays interpretable as a probability. Train directly
against revenue and you can no longer ask "is this calibrated?" — which is what
makes `bid × pCTR` mean anything.

In the demo run, `footer` wins at `$4.00 × 0.0413 = 0.1653` despite *not* having
the highest pCTR. Neither relevance nor willingness-to-pay wins alone.

### Endpoint vs. batch: availability vs. work

Both serve the same artifact with the same `inference.py`. The difference is
economic:

- **Endpoint** bills ~$0.065/hr from the moment it starts until deleted,
  regardless of traffic. ~30 requests cost the same as zero.
- **Batch** provisions, runs, terminates. Cost scales with work done.

The question isn't "how fast" — it's **do you know the inputs before the request
arrives?** If yes, batch is usually better: no idle cost, nothing to forget.

### Deliberate deviations

| Spec said | Used | Why |
|---|---|---|
| `ml.t3.medium` endpoint | `ml.t2.medium` | t3 isn't a supported endpoint instance type — `CreateEndpointConfig` rejects it |
| latest `sagemaker` SDK | pinned `>=2,<3` | v3 removed the `SKLearn` estimator this project is built around |

---

## Running it

Requires Python 3.12, [uv](https://docs.astral.sh/uv/), the AWS CLI with
credentials, and a Kaggle API token.

```sh
uv sync
./scripts/install-hooks.sh          # pre-commit secret scan

python prepare_data.py              # sample + split (local)
python upload_data.py --confirm     # → S3
python verify_data.py               # acceptance criteria

python submit_training.py --confirm # ~$0.005, self-terminating
python register_model.py --confirm  # $0

python deploy_endpoint.py --confirm # ⚠️ bills ~$0.065/hr until deleted
python invoke_endpoint.py           # verify + parity check
python rank_candidates.py           # the auction demo
make aws-down                       # stop billing

python batch_transform.py --confirm # bulk scoring, no endpoint
```

Every script that costs money is gated behind `--confirm` and supports
`--dry-run`.

```sh
make aws-status     # what exists, and what of it bills (read-only)
make aws-down       # stop everything billing; keep data and artifacts
make check-secrets  # scan for credentials and account IDs
```

---

## Cost safety

An endpoint is the only thing here that bills continuously — roughly **$1.56/day
if forgotten**. Everything else self-terminates.

- A billing alarm existed before any AWS resource did
- `aws_down.py` was written *before* the first endpoint was ever deployed
- Teardown sweeps **all three regions** the project touched, not just the home one
- It refuses to report success while an endpoint is still `Deleting` — that state
  still bills

Teardown stops the billing; it does **not** delete the project. The bucket, IAM
role, artifact, and registry survive.

---

## Secret hygiene

The repo is public, so [`scripts/check_secrets.sh`](scripts/check_secrets.sh)
runs as a pre-commit hook, scanning for AWS keys, API tokens, private keys, and
the AWS account ID (resolved at runtime, so the scanner never contains the value
it protects).

It's verified against **positive controls** — planted fake credentials that it
must catch. A scanner that has only ever passed is indistinguishable from one
that does nothing.

IAM policies are committed as `.json.template` files with an `ACCOUNT_ID`
placeholder. The extension is deliberate: an unrendered policy is *valid* JSON
that applies cleanly and grants access to a bucket that doesn't exist — breaking
things with no error pointing at the cause.

---

## What this project deliberately doesn't do

- **No retrieval/ranking funnel.** This model is the ranking stage only. Real ad
  systems cut millions of candidates to hundreds first, with a cheaper model.
- **No feature store.** Every feature arrives in the request payload; in
  production, user features are fetched from an online store.
- **No monitoring, no A/B testing, no cross features.**
- **One held-out split** doing double duty as tuning target and reported metric,
  so the AUC is mildly optimistic.
- **`id` was dropped in data prep** — correct for modelling, wrong for the
  pipeline, since batch scoring then produces predictions with no key to join
  back. "Not a feature" and "not needed downstream" are different claims. Left
  as-is and documented rather than silently fixed.

---

## The recurring lesson

Four times in this project, a passing result meant nothing until something was
made to fail:

1. An IAM scoping test passed identically whether or not the scoping worked — a
   negative test showed two permissions had survived.
2. The Model Registry was assumed to validate nothing; testing showed it
   validates one thing and not three others.
3. A batch job "succeeded," but the meaningful result was an empty `git diff` on
   `inference.py`, not the output file.
4. The teardown script printed `CLEAN` while a stale resource sat in the same
   output, because cleanup was accidentally gated behind a billing check.

Four different services, one failure mode: **the happy path passing is not
evidence that the mechanism works.** Full detail in [`NOTES.md`](NOTES.md).
