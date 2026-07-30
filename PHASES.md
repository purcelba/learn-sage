# PHASES.md — SageMaker Warm-Up Build Plan

Phased build spec. Work one phase at a time, in order; `CLAUDE.md` holds the standing rules and locked contracts that apply to every phase — read it first, and re-check it whenever a phase touches the model inputs, the training/serving split, or the bid-agnostic boundary. Each phase ends with acceptance criteria — all must pass, with output shown, before the phase is considered done and the next begins.

**Checkboxes are the running record.** Tasks and acceptance criteria are checked off only after being *run* with output shown — never on the strength of code existing (see `CLAUDE.md` rule 3). `NOTES.md` holds the detail behind each checked box: what actually happened, and why any judgment call went the way it did.

## Progress

| Phase | Status |
|---|---|
| 0 — AWS account setup & guardrails | ✅ complete (2026-07-29) |
| 1 — Data | ✅ complete (2026-07-29) |
| 2 — Train a model on SageMaker | ✅ complete (2026-07-30) — cost $0.005 |
| 3 — Register the model | ✅ complete (2026-07-30) — $0, metadata only |
| 4 — Deploy a real-time endpoint | ✅ complete (2026-07-30) — ~$0.006, endpoint deleted |
| 5 — Batch Transform (optional) | ✅ complete (2026-07-30) — ~$0.01, self-terminating |
| 6 — Teardown & hygiene | ✅ complete (2026-07-30) — `make aws-down` |
| 7 — Map it back to LyftLearn (stretch) | ⬜ not started |
| 8 — Terraform (stretch) | ⬜ not started |

---

## Phase 0 — AWS account setup & guardrails ✅

**Goal:** a safe sandbox you can't accidentally leave running up a bill.

- [x] Use a personal AWS account (free tier if new).
- [x] **Set a billing alarm** ($10–20) via CloudWatch/Budgets — do this before anything else.
- [x] Create an IAM user (or role) scoped to just what's needed: S3 (one bucket), SageMaker full access is fine for a learning sandbox, CloudWatch Logs read.
- [x] Install/configure AWS CLI + `boto3` + `sagemaker` SDK locally; confirm credentials work.

**Acceptance criteria:**
- [x] 1. `aws sts get-caller-identity` returns your sandbox identity.
- [x] 2. Billing alarm is visible and active in the console.

---

## Phase 1 — Data ✅

**Goal:** get a small, clean CTR dataset into S3 in the shape your training script expects.

- [x] Pull the **Avazu CTR Prediction** 50k-row sample (Kaggle) rather than the full dataset — keeps everything in this project fast and cheap.
- [x] `click` is the label (0/1). The rest are categorical: `hour`, `banner_pos`, `site_id`/`site_category`, `app_id`/`app_category`, `device_type`/`device_conn_type`, and several anonymized `C1`/`C14`–`C21` fields.
- [x] Keep this phase light — drop the `id` column, split train/test, upload both as CSV **with headers, unencoded**. Since Phase 2 is now a custom script rather than a built-in algorithm, categorical encoding belongs in your training code, not a one-off data-prep step — that's part of what makes the script yours to own.
- [x] Upload both to S3 under a clear prefix (`s3://<bucket>/data/train/`, `.../test/`).

**Acceptance criteria:**
- [x] 1. `aws s3 ls` on the bucket shows `train.csv` and `test.csv`.
- [x] 2. A quick local read-back confirms row counts match your split and headers are intact.

**Follow-up completed in this phase (carried over from Phase 0):**
- [x] Narrow `AmazonS3FullAccess` to a single-bucket policy now that the bucket name is known. Done via `iam/learn-sage-s3-policy.json`; verified by negative test. Two account-wide permissions survive via `AmazonSageMakerFullAccess` — see `iam/README.md` for why that's accepted rather than a bug.

---

## Phase 2 — Train a model on SageMaker (custom script) ✅

**Goal:** run your own training code as a SageMaker Training Job, not a black-box algorithm — this is the part that's actually analogous to what your team will own at Lyft (a model your DS/MLE org writes and is accountable for, running on shared training infra).

- [x] Write a small `train.py`: reads `train.csv`/`test.csv` from the SageMaker-provided input channels, does the categorical encoding (one-hot or hashing — a hashing trick is worth doing here specifically, since it's what production ads systems use for high-cardinality, unbounded categorical fields like `site_id`), fits **`sklearn.linear_model.LogisticRegression`**, evaluates on the test set, and writes the fitted model (and the fitted encoder, so serving doesn't silently diverge from training — the same training/serving-skew problem your feature store notes flagged) to `/opt/ml/model` via `joblib`.
- [x] Use the `sagemaker` SDK's `SKLearn` estimator (framework/script mode: you supply `train.py`, SageMaker supplies the container). Pass hyperparameters as CLI args your script parses with `argparse`.
- [x] Kick off the job with `estimator.fit({'train': ..., 'test': ...})`, watch it provision, run, and terminate.
- [x] Note where the model artifact lands: `s3://<bucket>/output/.../model.tar.gz` — this now contains *your* joblib-dumped model and encoder, not a SageMaker-native format.
- [x] Create the SageMaker **execution role** the training job runs as. This is a different principal from `learn-sage-dev` — the user submits the job, the role is what the container actually runs as — so it needs its own S3 access to the project bucket. Flagged during Phase 1; see `iam/README.md`.

**Model inputs.** Every Avazu column except `click` (the label) and `id` (a row identifier, not a feature) goes in, mapped onto the same user/ad/context grouping your ad server spec uses:

| Group | Columns | Notes |
|---|---|---|
| **Context** | `hour`, `C1` | `hour` arrives as `YYMMDDHH` — extract hour-of-day (and optionally day-of-week) rather than treating the whole string as one category |
| **Ad** | `banner_pos`, `site_id`/`site_domain`/`site_category` *or* `app_id`/`app_domain`/`app_category` (populated depending on whether the impression was in-site or in-app), `C15`, `C16` | this is the "which candidate" side — the fields that vary per row in your Phase 4 ranking script |
| **User/device** | `device_id`, `device_ip`, `device_model`, `device_type`, `device_conn_type` | worth a deliberate call-out: `device_id`/`device_ip` are near-unique pseudo-identifiers, not real behavioral features — hash them into a small bucket space rather than one-hot encoding directly, or drop them and note in your write-up that real user features (at Lyft, real behavioral aggregates from the feature store) would replace raw IDs entirely |
| **Anonymized (mixed)** | `C14`, `C17`–`C21` | undisclosed semantics; treat as categorical and encode the same way as the rest |

All of these get encoded into a single numeric feature vector per row; that vector is the actual input to `LogisticRegression` — there's no separate "user model" or "ad model," just one flat vector combining all three groups, which is the simplest version of the "cross features" idea from your ad server spec (in a fuller system, interactions between user and ad groups would be engineered explicitly rather than left for logistic regression to find linearly).

**Acceptance criteria:**
- [x] 1. Training job shows `Completed` in the console and in `describe_training_job`.
- [x] 2. `model.tar.gz` exists in S3 and, unpacked, contains your model and encoder artifacts.
- [x] 3. Your script prints validation AUC to the training logs, meaningfully above 0.5 — your pCTR model is actually discriminating clicks from non-clicks, not just running.
- [x] 4. You can list the model's input columns by group (context / ad / user-device) and explain why `device_id`/`device_ip` needed different handling than the rest.
- [x] 5. You can articulate, in your own words, what SageMaker did on your behalf (provisioning, pulling your code into its container, running it, capturing `/opt/ml/model` to S3) vs. what your script owns (all the actual ML logic) — that split is the LyftLearn Compute analog: platform runs it, your team owns what runs.

---

## Phase 3 — Register the model ✅

**Goal:** the artifact becomes a first-class "Model" object, not just a file.

- [x] Create a SageMaker `Model` pointing at your artifact and the **SKLearn framework container** SageMaker used to run `train.py` (the same image family, now paired with your `inference.py` from Phase 4 for serving).
- [x] Optional stretch: register it in the SageMaker Model Registry (a light touch — this is the closest AWS analog to "versioned model artifact gated by an eval report," which is how LyftLearn's CI/CD treats models).

**Acceptance criteria:**
- [x] 1. `aws sagemaker list-models` shows your model.
- [x] 2. You can state which container/image it's tied to and why that matters (the container is generic infra; your code — training and inference scripts — is the actual owned artifact, same split as Phase 2).

---

## Phase 4 — Deploy a real-time endpoint ✅

**Goal:** feel what "always-on inference infra" costs and requires — this is the part LyftLearn deliberately keeps off SageMaker in production, so pay attention to *why* it might.

- [x] Because Phase 2 used a custom script rather than a built-in algorithm, serving needs its own small `inference.py` alongside `train.py`: a `model_fn` (load your joblib model + encoder), `input_fn`/`output_fn` (parse the request, format the response), and `predict_fn` (encode inputs using the *same* fitted encoder from training, then call `model.predict_proba`). This pairing — training code and serving code sharing the same encoder artifact — is the same feature-parity discipline your ad server project's registry-as-contract is built around, just at a much smaller scale.
- [x] Deploy to a small instance type (`ml.t3.medium`) — or use **Serverless Inference** if you want to see the cheaper, newer alternative (no idle cost, cold-start latency instead).
- [x] Invoke the endpoint via `boto3` (`invoke_endpoint`) with a held-out test row and confirm a sane pCTR score (a probability between 0 and 1) comes back.
- [x] **Toy auction service:** write a small local script (`rank_candidates.py`) that plays the caller's role rather than the model's — the model endpoint should never know about bids or auctions. Take N candidate rows (a mini candidate set — vary a couple of ad-side fields like `banner_pos` or `site_category` across them, holding user/context fields fixed to simulate "same request, different candidate ads"), call `invoke_endpoint` once per candidate (or as a batch if your `inference.py` supports it) to get pCTR per candidate, assign each candidate a fake `bid` value, compute `bid × pCTR`, sort, and print the ranked list with the winner on top. This is the actual mechanic underneath ads ranking — scoring and ranking are separate concerns, and this script is where they're deliberately stitched together *outside* the model artifact.

**Acceptance criteria:**
- [x] 1. `invoke_endpoint` returns a probability-shaped prediction (0–1) that moves sensibly when you vary an input feature you'd expect to matter (e.g. `banner_pos`).
- [x] 2. The prediction matches what you'd get running the same row through your model + encoder locally — confirming training/serving parity, not just "an endpoint responds."
- [x] 3. You can name the ongoing cost driver (an always-on instance) and contrast it with how Kubernetes-based serving amortizes cost differently across many models — this contrast is the actual learning goal of this phase.
- [x] 4. `rank_candidates.py` prints a ranked candidate list where the ordering changes sensibly when you change either a candidate's simulated bid or its ad-side features — confirming the ranking logic actually depends on both inputs, not just one.

---

## Phase 5 (optional) — Batch Transform ✅

**Goal:** the other offline pattern — batch prediction without a persistent endpoint, closer to how LyftLearn Compute would run scheduled scoring jobs.

- [x] Run a SageMaker Batch Transform job against your held-out `test.csv`, writing predictions back to S3. No endpoint stays up. It reuses the same `inference.py` as Phase 4 — a useful confirmation that your serving code isn't secretly endpoint-specific.

**Acceptance criteria:**
- [x] 1. Batch Transform job completes; predictions file appears in S3.
- [x] 2. You can articulate when you'd reach for this vs. a real-time endpoint (latency need vs. throughput/cost).

---

## Phase 6 — Teardown & hygiene ✅

**Goal:** nothing keeps billing after you stop working.

- [x] Delete the endpoint and endpoint config (this is the one thing that silently costs money if forgotten).
- [x] Confirm no training jobs, notebook instances, or endpoints are left running.
- [x] Write a one-command teardown (`make aws-down` or a short script) so this is never a manual checklist again.

**Acceptance criteria:**
- [x] 1. `aws sagemaker list-endpoints` returns empty.
- [x] 2. Billing alarm shows spend well under your budget.

---

## Phase 7 (stretch) — Map it back to LyftLearn ⬜

**Goal:** convert hands-on AWS experience into Lyft-specific vocabulary.

- [ ] Write yourself a short one-pager: which piece of what you just did maps to LyftLearn Compute (training), which maps to the model-registry-in-CI/CD concept, and — importantly — where the mapping *breaks down* at the serving step (SageMaker endpoint vs. LyftLearn Serving on Kubernetes).
- [ ] This becomes a good artifact to sanity-check with Ravi's team early on.

---

## Phase 8 (stretch) — Bring the surviving infra under Terraform ⬜

**Goal:** see where the infrastructure/ML-lifecycle boundary actually falls, by discovering which of this project's resources Terraform *should* manage — and which it shouldn't.

The organizing observation: **the resources that survive Phase 6 teardown are exactly the resources that belong in Terraform.** The bucket, the IAM user policy, the SageMaker execution role, and the budget all persist between sessions. The endpoint, the training job, and the transform job are precisely what Phase 6 destroys. That's the same boundary, seen twice. Terraform manages the durable substrate; an orchestrator or ML platform manages the things that run once and exit. A training job in `tfstate` is a category error — you'd be asking a state-reconciliation engine to manage something meant to terminate.

Do this **after** Phase 6, not before. Running it post-teardown is what makes the boundary self-evident: the only things left to import are the things that should have been in Terraform all along.

- [ ] **Set `prevent_destroy = true` on the bucket in the very first commit, before importing anything.** A `terraform destroy` against a state file containing the data bucket would delete Phase 1's work. This guardrail comes first, not last.
- [ ] **Import, don't create.** Write HCL for resources that already exist and use `terraform import` (or `import` blocks) to adopt them. Drive `terraform plan` to a clean no-op diff. Creating greenfield resources would teach less: reaching an empty diff forces you to discover every attribute of infra you *thought* you understood — the bucket you hand-made in Phase 1 has versioning enabled, default encryption, and no lifecycle policy, and `plan` is what will tell you so. It's also the realistic enterprise scenario; almost nobody starts clean, they inherit click-ops infra and bring it under management.
- [ ] Scope: the S3 bucket (+ versioning), the `learn-sage-s3-scoped` inline policy — its JSON already lives in `iam/` under version control, so this is mostly moving *who applies it* — the Phase 2 SageMaker execution role, and optionally the Phase 0 budget.
- [ ] Explicitly out of scope: training jobs, transform jobs, endpoints, model versions. Note in your write-up *why* each one is excluded — that reasoning is the actual deliverable.

**Acceptance criteria:**
- [ ] 1. `terraform plan` reports no changes against the real, hand-built infrastructure — proving your HCL describes what actually exists rather than what you assumed exists.
- [ ] 2. You can name at least one attribute `plan` revealed that you hadn't consciously decided when you created the resource by hand.
- [ ] 3. You can state which resources you deliberately left out of Terraform and why, in lifecycle terms (durable substrate vs. ephemeral job).
- [ ] 4. You can explain why Terraform would **not** have caught the Phase 1 IAM finding — that `s3:CreateBucket` leaked in via `AmazonSageMakerFullAccess` is a policy-*union* property, and a perfectly-applied config produces identical effective permissions. IaC guarantees the state you declared; it doesn't tell you the state you declared means what you think. Catching that needs policy analysis (IAM Access Analyzer, `simulate-principal-policy`) or negative tests.
- [ ] 5. You can articulate why, at Lyft, you probably wouldn't write this Terraform at all — LyftLearn *is* the abstraction over it, and the platform team owns the IaC while your team owns model code and eval gates. Knowing where that seam is, is what lets you tell "our model is broken" from "the platform is broken."

**Note on ROI.** This phase teaches what the *platform* team owns, not what *your* team owns. That's real but second-order for an ads-ranking manager — model code, eval gates, and training/serving parity are your actual surface area. Hence: bottom of the stretch list, first to cut.

---

## Suggested pacing

Phases 0–1 in one sitting; 2–3 the next; 4 one sitting (this is the one worth lingering on — the cost/latency tradeoffs are the point); 5–6 together; 7 whenever you feel like writing it up; 8 only if the Terraform question is still itching after 7.

## Trim order if scope must shrink

Cut in this order: Phase 8 (Terraform) → Phase 5 (Batch Transform) → Phase 3's Model Registry stretch → Phase 7 write-up (do it mentally instead). Never cut: the Phase 0 billing alarm, or Phase 6 teardown.
