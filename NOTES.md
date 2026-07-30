# NOTES.md — running state & decisions

Scratch log of decisions and environment state that aren't obvious from the
code. `CLAUDE.md` holds the locked contracts; `PHASES.md` holds the build plan.
This file holds "what actually happened and why."

---

## Status: Phase 0 — COMPLETE (2026-07-29)

### Passing

- **Criterion 1** — `aws sts get-caller-identity` resolves to
  `arn:aws:iam::ACCOUNT_ID:user/learn-sage-dev` (the scoped IAM user, not
  root).
- S3, SageMaker, and CloudWatch Logs calls all authorize.
- Region `us-east-1`, output `json`.
- **Criterion 2** — `learn-sage-budget` confirmed in the Billing console:
  $15.00 budget, **Thresholds: OK**, **Health status: Healthy**, $0.00 spend.
  Verified by screenshot, because this can't be checked from the CLI:
  `aws budgets describe-budgets` returns `AccessDeniedException` — the IAM user
  has no `budgets:ViewBudget`. **That's the least-privilege scoping working as
  intended, not a bug.** Don't widen the IAM user just to make this
  CLI-checkable.

  Note on reading that console row: `Thresholds: OK` (rather than a dash) means
  an alert threshold is defined and unbreached. $0.00 spend and a `-` forecast
  are the expected empty-history state, not a misconfiguration — Budgets lags
  ~24h and needs several days before it will forecast.
- Teardown check: 0 endpoints, 0 training jobs, 0 notebook instances. Nothing
  is billing.

### Soft follow-up (not blocking)

- The budgets **list** view shows the threshold but not its **recipient**.
  Click into `learn-sage-budget` sometime and confirm an email address is
  attached. A threshold with no subscriber tracks spend silently and notifies
  nobody — that's the actual guardrail failure mode, and it looks identical to
  a working budget from the list view.

---

## Status: Phase 1 — COMPLETE (2026-07-29), one console follow-up open

### Passing

- **Criterion 1** — `aws s3 ls s3://learn-sage-ACCOUNT_ID --recursive` shows
  `data/train/train.csv` (5.1 MiB) and `data/test/test.csv` (1.3 MiB).
- **Criterion 2** — `verify_data.py` reads both objects **back from S3** and
  asserts 40,000 / 10,000 rows, 23 identical columns, `id` absent, `click`
  present. Passes.

  The read-back deliberately fetches from S3 rather than re-reading
  `data/*.csv`. Reading the local files would only prove pandas can read what
  pandas just wrote — it says nothing about whether the upload landed intact,
  which is what the criterion is actually about.

### IAM scoping — done, with a caveat worth knowing

`AmazonS3FullAccess` is **detached**. `learn-sage-dev` now carries an inline
`learn-sage-s3-scoped` policy (source of truth: `iam/learn-sage-s3-policy.json.template`)
granting S3 only on `learn-sage-ACCOUNT_ID`.

Verified by negative test, which is the only reason the caveat was found:
reading a third-party bucket (`s3://nyc-tlc/`) is now **denied** — that was the
real exposure and it's closed. But `s3:CreateBucket` and `s3:ListAllMyBuckets`
are **still allowed account-wide**, inherited from `AmazonSageMakerFullAccess`
(SageMaker needs to create its own default bucket). IAM unions Allows, so the
scoped policy can't revoke them. Accepted for a learning sandbox; see
`iam/README.md` for the reasoning and what closing it would cost.

The positive check (`verify_data.py` passes) proved nothing by itself — it would
pass identically with a broken policy, since it only touches the bucket that's
meant to work. **When tightening permissions, test what should now fail.**

Also: the `create-bucket` negative test left a stray empty bucket
`learn-sage-scope-test-ACCOUNT_ID` that the scoped user couldn't delete
(`DeleteBucket` not granted); removed from the console as root. Prefer
non-mutating probes next time.

### The data

| Thing | Value |
|---|---|
| Bucket | `learn-sage-ACCOUNT_ID` (`us-east-1`, versioning **enabled**) |
| Train | `s3://learn-sage-ACCOUNT_ID/data/train/train.csv` — 40,000 rows |
| Test | `s3://learn-sage-ACCOUNT_ID/data/test/test.csv` — 10,000 rows |
| Columns | 23 = 1 label (`click`) + 22 features; `id` dropped |
| Click rate | 0.1710 in **both** splits (stratified, seed 42, 80/20) |
| Raw file | `data/raw/train.gz`, 1.04 GB, git-ignored — deletable |

Sanity checks that actually validated the sample, not just "it ran":

- Scanned **40,428,967 rows** — exactly the canonical Avazu train count, so the
  streaming pass saw the whole file with nothing silently truncated.
- Click rate **0.1710** matches Avazu's known ~17% base CTR. Identical across
  train and test confirms stratification took.
- **240 distinct `hour` values = 10 days x 24 hours, all present.**

### Decision: systematic sampling, not head-50k

Avazu's train file is **time-ordered** across 10 days. Taking the first 50k rows
would yield roughly two hours of Oct 21 — which would leave Phase 2's
hour-of-day context feature with almost no variance, i.e. a feature the spec
asks for that couldn't possibly carry signal. So `prepare_data.py` streams the
whole file and samples across it.

Two-stage, not one: keep every 400th row (pool = 101,073), then thin to exactly
50,000 evenly spaced. A single stride tuned to land on 50k directly would come
up **short** if the file had fewer rows than expected. Over-collect then thin is
robust in both directions. Fully deterministic — no RNG in the sampling.

### Decision: `dtype=str` on every column when reading raw

`site_id`, `device_ip`, `device_model` etc. are hex strings whose leading zeros
are significant (`08ac11ab`), and the anonymized `C*` columns are categorical
despite looking numeric. Letting pandas infer dtypes coerces some of them to
ints and loses information. `click` is cast to int only where it's needed as a
number (stratification, click-rate reporting).

### Decision: no encoding in Phase 1

`hour` stays as the raw `YYMMDDHH` string in the uploaded CSVs. Extracting
hour-of-day is a modeling decision and belongs in `train.py`, so that training
and serving share exactly **one** implementation of "raw column -> feature."
Doing it here would split feature logic across two files — the training/serving
skew problem the feature-parity contract exists to prevent.

### Deps added

`pandas` and `scikit-learn` are now **explicit** in `pyproject.toml`. Both were
already installed transitively via `sagemaker`; relying on that would have
broken silently the first time that dependency tree shifted. `kaggle` (2.2.4)
added for the dataset download.

### Kaggle auth

Works via `~/.kaggle/access_token` holding a bare `KGAT_...` token — the newer
scheme, honored by `kaggle` 2.2.4. No `kaggle.json` needed. **Rotate that token
when the project is done** (Kaggle → Settings → API): it was pasted into a
Claude Code session, so it's in plaintext in the local transcript under
`~/.claude/projects/`.

---

## Status: Phase 2 — COMPLETE (2026-07-30)

Job `learn-sage-pctr-2026-07-30-19-35-01-384`, `Completed`, **95 billable
seconds on ml.m5.large = $0.003**. Total Phase 2 spend including three failed
attempts: **~$0.005**.

| Criterion | Result |
|---|---|
| 1. `describe_training_job` = Completed | PASS |
| 2. `model.tar.gz` unpacks to model + encoder | PASS — model.joblib, featurizer.joblib, feature_config.json, metrics.json |
| 3. Validation AUC meaningfully > 0.5 | PASS — **0.7322** |
| 4. Columns by group; why device_id/ip differ | see below |
| 5. What SageMaker did vs. what the script owns | see below |

Artifact: `s3://learn-sage-ACCOUNT_ID/output/learn-sage-pctr-2026-07-30-19-35-01-384/output/model.tar.gz`
Container: `sagemaker-scikit-learn:1.4-2-cpu-py3`

Results (identical local and in-container, which is itself reassuring):
AUC 0.7322 | log loss 0.4063 vs 0.4575 baseline (11.2% better) |
predicted CTR 0.1685 vs actual 0.1710.

Verified the artifact **loads and scores**, not merely that it exists:
`assert_config_matches` passes and five test rows produce sane probabilities.

### Three failures, each worth more than the success

**1. Training quota was 0 account-wide.** Not regional — probed `us-west-2`
too. New AWS accounts start at zero SageMaker training instances and need a
Service Quotas increase (granted in ~1h here). A rejected job is free: the
limit check happens *after* authorization, which incidentally proved the IAM
work was correct before anything ran.

**2. The execution role couldn't read the bucket.** The failure that the
two-principals distinction predicts: `learn-sage-dev` reads
`data/test/test.csv` fine from the CLI, but the container runs as the role, and
the role got `AccessDenied` on the same object.

Root cause is a naming coincidence: `AmazonSageMakerFullAccess` sounds
blanket but its S3 grant is scoped to `arn:aws:s3:::*SageMaker*` /
`*Sagemaker*` / `*sagemaker*`. Our bucket is `learn-sage-...` — no match. Had
the bucket been named `sagemaker-learn-sage` this would have silently worked
and taught nothing. Fixed by attaching `learn-sage-s3-access` (rendered from
`iam/sagemaker-execution-policy.json.template`) **to the role**.

Sub-failure worth its own line: the first attempt at that policy was pasted
**unrendered**, with the literal string `ACCOUNT_ID` still in the ARNs. It
attached without error and produced a byte-identical `AccessDenied` — a valid
policy granting access to a bucket that does not exist. This is exactly the
failure mode the `.json.template` extension exists to prevent, and it still
happened, because the render step was done by hand.

**3. `-C` vs `--C`.** SageMaker renders each hyperparameter as a CLI flag, and a
**single-character** name gets a *single* dash: it invoked `train.py -C 0.1`,
which argparse rejects against a `--C` declaration. `--max-iter` and
`--n-features` were unaffected. Renamed to `--reg-c` with `dest="C"`.
**Keep SageMaker hyperparameter names multi-character.**

### Container vs. local library versions

`train.py` logs the container environment on every run, because Phase 4 must
load this pickle locally and compare predictions:

| | container | local |
|---|---|---|
| python | 3.10.20 | 3.12.13 |
| scikit-learn | 1.4.2 | 1.4.2 (pinned to match) |
| numpy | 2.1.0 | 2.5.1 |
| scipy | 1.15.3 | 1.18.0 |

Both numpy are 2.x, so the pickle should load cleanly in Phase 4 — the
dangerous case would have been a 1.x/2.x split. Worth re-checking if Phase 4's
parity comparison ever disagrees.

### Decision: hashing, and where the skew risk actually lives

All 22 features become `"col=value"` tokens through one
`FeatureHasher(2**18)`. Chosen over one-hot because `site_id` is unbounded in
production: a fitted vocabulary needs a retrain before a new site can be
represented at all, while a hash bucket absorbs it immediately.

Measured cardinalities that motivated it: `device_ip` 33,635 distinct in 40,000
rows (84% unique), `device_id` 6,926, `device_model` 2,360, `site_id` 1,053 —
~48,000 distinct tokens total, against 262,144 slots, so collisions are rare.

`hour` is **dropped** after deriving `hour_of_day` and `day_of_week`. All 240
raw `YYMMDDHH` values are specific October 2014 dates; as a category the model
would memorize dates that can never recur.

**The honest wrinkle:** `FeatureHasher` is stateless — it fits nothing. So
"save the fitted encoder" is nearly vacuous here, and that's precisely why
hashing is attractive (no vocabulary to keep in sync). But the skew risk moves
rather than vanishing: parity now depends on `src/features.py` and
`n_features` being identical at train and serve time. Hence
`feature_config.json` and `assert_config_matches()`, which Phase 4's
`inference.py` calls at load time so a mismatch crashes loudly instead of
silently scoring a feature space the model never saw.

### Local-first was worth it

The hyperparameter sweep (C in {0.1, 0.3, 1.0}) ran locally for $0 and picked
C=0.1. The flag-rename fix was also verified locally before resubmitting. Every
paid job that ran was a configuration already known to work — the three
failures were all infrastructure, never modeling.

---

## Status: Phase 3 — COMPLETE (2026-07-30). Cost: $0.

Metadata only — nothing provisioned, nothing executed.

| Criterion | Result |
|---|---|
| 1. `list-models` shows the model | PASS — `learn-sage-pctr-2026-07-30-19-35-01-384` |
| 2. Which image, and why it matters | see below |

Created: a `Model`, plus Model Package Group `learn-sage-pctr` with version 1,
`Approved`, metrics pointing at
`s3://learn-sage-ACCOUNT_ID/eval/<job>/metrics.json`.

### What the registry validates — measured, not assumed

I claimed the registry "validates nothing." **That was wrong**, and only a
negative test caught it. Registering deliberately-broken versions shows:

| Thing | Validated at registration? |
|---|---|
| Model artifact S3 object exists | **YES** — `ValidationException: Cannot find S3 object` |
| `ModelMetrics` S3 URI exists | **NO** — a nonexistent path registers fine |
| The metric values themselves | **NO** — nothing opens the file |
| Container can load the artifact | **NO** — never attempted |
| `ModelApprovalStatus` justified | **NO** — set straight to `Approved` |

Accurate summary: it verifies the artifact is **present**, nothing about whether
it's **good**. Both bogus test versions were deleted; the group holds version 1
only.

The gate is really four separate things, and the registry is only the third:
you run the evaluation; you decide whether it passes; the registry records the
artifact, the metrics *location*, and the verdict; your deployment pipeline
refuses anything not `Approved`. Miss the fourth and "Approved" means somebody
set a string. Automating the third-to-fourth link is SageMaker Pipelines'
`ConditionStep` — different machinery, out of scope here.

**The distinction that matters for an ads-ranking manager:** "we have a model
registry" and "we have an enforced quality gate" are different claims. Worth
asking Ravi's team where LyftLearn's enforcement actually lives — automated
threshold in CI, or a human reading a dashboard.

### The image is pinned to the one that produced the artifact

`sagemaker-scikit-learn:1.4-2-cpu-py3` — identical to
`AlgorithmSpecification.TrainingImage` from the training job.

Getting there needed a workaround worth recording. sklearn uses **one image
family** for training and inference (same ECR repo, different tags), but this
SDK version's *inference* version list is stale — it stops at `1.2-1` while
training reaches `1.4-2`, so `image_scope="inference"` raises
`ValueError: Unsupported sklearn version: 1.4-2`.

Falling back to `1.2-1` would be a **correctness bug, not a cosmetic one**: the
artifact was pickled by scikit-learn 1.4.2, and sklearn does not support loading
a newer version's pickle in an older runtime. So `register_model.py` retrieves
with `image_scope="training"` and uses that URI for serving. The comment there
explains why, because it looks like a mistake otherwise.

Also: the ECR account hosting AWS images differs per region (`683313688378` in
us-east-1), so the URI is resolved via `image_uris.retrieve`, never hardcoded.

### Deliberately NOT done: `inference.py`

Phase 3's text mentions the container being "paired with your `inference.py`
from Phase 4." That's forward context, not a Phase 3 task, and writing it here
would violate the work-only-on-the-named-phase rule.

Consequence, stated plainly: **the Model created here would not serve correctly
if deployed today.** The container's default handler loads `model.joblib` but
knows nothing about `featurizer.joblib` or the `col=value` tokenisation, so it
would score raw strings and fail. Phase 4 supplies `inference.py` and, on
deploy, creates its own Model including it — superseding this one.

That SageMaker happily created an unservable Model is the phase's real lesson:
"first-class object" means addressable and versioned, not validated.

---

## Status: Phase 4 — COMPLETE (2026-07-30). Cost: ~$0.006. Endpoint DELETED.

Live window: provisioning 16:25:37 → InService 16:30:41 → deleted 16:31:14.
About 5.5 minutes on `ml.t2.medium` at ~$0.065/hr.

| Criterion | Result |
|---|---|
| 1. Probability-shaped, responsive to inputs | PASS — 0.0218 baseline; `banner_pos` 0→7 moves it 0.0218→0.0345 |
| 2. Endpoint matches local scoring | PASS — **max abs diff 3.5e-18** over 10 rows |
| 3. Cost driver, and the Kubernetes contrast | see below |
| 4. Ranking responds to bid *and* ad features | PASS — 3 auctions |

### Deviation: ml.t2.medium, not ml.t3.medium

`ml.t3.medium` **is not a supported real-time endpoint instance type.**
`CreateEndpointConfig` rejects it outright — and that also finally explains the
missing quota: there's no `ml.t3.medium for endpoint usage` entry in Service
Quotas because the instance isn't offered for endpoints at all. Earlier I
reasoned "not listed, therefore untracked, therefore fine." Wrong inference;
the right one was "not listed, therefore find out why."

The rejection was free (validation precedes provisioning), consistent with the
quota rejections in Phase 2. Supported small types are `ml.t2.medium`,
`ml.t2.large`, `ml.m5.large`. Chose `ml.t2.medium`: same burstable family, one
generation older, cheaper than m5.large.

Side effect worth knowing: the failed deploy still created a `Model` object
before validation failed, which then collided on retry. Deleted it manually.
`.deploy()` creates Model → EndpointConfig → Endpoint in sequence, so a
mid-sequence failure leaves earlier objects behind.

### Criterion 2 is the one that matters, and it passed hard

**Max absolute difference 3.5e-18** between endpoint and local scoring across 10
rows — 9 of 10 bit-identical. That's despite genuinely different runtimes:
container Python 3.10.20 / numpy 2.1.0, local Python 3.12.13 / numpy 2.5.1.

That result is *earned*, not lucky. It follows from `source_dir="src"` shipping
`features.py` into the container, so serving imports the same module training
did rather than a copy. Had `inference.py` reimplemented tokenisation "just to
keep serving self-contained," this number would be the first place the drift
showed — and only if someone thought to check.

An endpoint returning 0.34 looks identical whether or not it agrees with
training. Only the comparison tells you.

### Criterion 3 — the cost driver, and why Lyft doesn't do this

**The cost driver is the instance, not the traffic.** `ml.t2.medium` bills
~$0.065/hr — ~$1.56/day, ~$47/month — from InService until deleted, whether it
serves a million requests or zero. We sent roughly 30 requests total. The
per-request cost was absurd; the per-hour cost was the same as if we'd sent
none.

Three consequences that scale badly:

1. **Cost is per model, not per request.** 50 models = 50 always-on instances,
   each sized for that model's peak, each idle most of the time.
2. **Utilisation is invisible.** Nothing in the bill distinguishes a saturated
   endpoint from an idle one.
3. **You pay for peak, continuously.** Sizing for the busy hour means
   overpaying for the other 23.

**How Kubernetes-based serving differs** — and this is why LyftLearn Serving is
on Kubernetes rather than SageMaker endpoints:

- **Bin-packing.** Many models share a node pool. A small model gets a pod, not
  a machine; 50 models may fit on a handful of nodes.
- **Elasticity.** HPA scales replicas with load; cluster autoscaler adds and
  removes nodes. Capacity follows demand instead of being pinned to peak.
- **One platform, not one per model.** Rollouts, canaries, mTLS, tracing,
  service mesh, on-call runbooks — shared infrastructure every model inherits,
  rather than per-endpoint configuration.
- **No AWS-managed-service premium** on top of the compute.

The tradeoff SageMaker endpoints buy in exchange: no cluster to operate. For one
model, or a team without platform engineers, that's a genuinely good deal — it's
why this phase is worth doing. At Lyft's scale, with a platform team already
running Kubernetes, the arithmetic inverts.

**Restating the standing caveat: Phase 4 is not the Lyft path.** Only LyftLearn
Compute (training, batch, notebooks) maps to SageMaker. Phase 5's Batch
Transform is much closer to how LyftLearn would run scheduled scoring.

### The bid-agnostic boundary, demonstrated

`rank_candidates.py` sends the endpoint raw ad rows and receives pCTR. It never
sends a bid. Every multiplication, comparison and sort happens caller-side.

Auction 1: `footer` wins at 0.1653 (bid $4.00 x pCTR 0.0413) despite *not*
having the highest pCTR — `top-banner` scores 0.0572 but only bids $2.50. That's
the mechanic in one line: neither pure relevance nor pure willingness-to-pay
wins, the product does.

Auction 2 (only a bid changed): `interstitial` goes last → first on a $12 bid,
with pCTR unchanged at 0.0345. Auction 3 (only ad features changed): moving
`interstitial` to `banner_pos=1` drops its pCTR 0.0345 → 0.0247, no bid change.
Ordering responds to both levers, independently.

Why the boundary is worth defending: the model can be retrained or replaced
without touching auction logic; reserve prices and pacing can change hourly
without retraining; and pCTR stays interpretable as a probability. Train
directly against revenue and you can no longer ask "is this calibrated?" —
which is what makes `bid x pCTR` mean anything.

### Models kept on purpose

`learn-sage-pctr-model` (bound to `inference.py`) is retained — Phase 5's Batch
Transform needs exactly that binding. `learn-sage-pctr-2026-07-30-19-35-01-384`
is the Phase 3 object. Both are metadata, $0.

### Teardown written before deploy

`teardown_endpoint.py` was written and exercised *before* `deploy_endpoint.py`
ever ran, so the means to stop billing existed before the billing did. It
deletes the endpoint **and** the endpoint config — deleting only the endpoint
leaves a config behind, which is free but makes "no endpoints listed" a
misleading all-clear.

It also refuses to report success while an endpoint is in `Deleting`: that state
still bills, so an optimistic "done" would be wrong. First run said
`STILL DELETING`; a second confirmed `none`.

---

## Status: Phase 5 — COMPLETE (2026-07-30). Cost: ~$0.01. Self-terminating.

Job `learn-sage-pctr-batch-model-2026-07-30-21-08-32-250`, `Completed`,
`ml.m5.large`, ~4.5 min wall clock. No endpoint involved; nothing left running.

| Criterion | Result |
|---|---|
| 1. Job completes, predictions in S3 | PASS — `predictions/test.csv.out`, 193 KiB |
| 2. Batch vs. real-time articulation | see below |

### The real result: an empty diff

`git diff HEAD -- src/inference.py src/features.py` is **empty**. Both files are
byte-identical to commit `eb58c6f`, which deployed the Phase 4 endpoint. The
serving code carried no endpoint-specific assumptions — the same file served
HTTP requests and a bulk file with no edits.

The only difference between the two phases is the method called on an
identically-constructed `SKLearnModel`: `.deploy()` versus `.transformer()`.

Verification went past "a file appeared":

- 10,000 predictions for 10,000 input rows
- **max abs diff vs local scoring 5.6e-17**; 9,985 of 10,000 bit-identical
- AUC recomputed from the batch output: **0.7322**, matching what `train.py`
  logged during training
- First three values match the Phase 4 endpoint exactly

So training, the real-time endpoint, batch, and local scoring all agree. That's
the feature-parity contract holding across every path.

### I broke a Model between phases

Phase 4's cleanup deleted `s3://<bucket>/code/` as "SDK scratch". **It is not
scratch** — it is where a deployed Model's source permanently lives
(`SAGEMAKER_SUBMIT_DIRECTORY`). The `learn-sage-pctr-model` object survived,
still pointing at a tarball that no longer existed, and nothing anywhere
reported the dangling reference.

Extends the Phase 3 finding: `CreateModel` validates that `ModelDataUrl` exists,
but evidently applies no such check to the submit directory. Two pointers, one
checked.

**Phase 6 hazard:** a teardown script that blanket-deletes `code/` silently
breaks every Model in the account, and nothing surfaces it until the next job
fails. Either keep `code/` or delete the Models that depend on it — the two are
coupled, and treating one as disposable while keeping the other is what caused
this.

Fixed by having `batch_transform.py` construct its own `SKLearnModel` rather
than reusing an artifact another phase happened to leave behind. Better design
regardless: phases shouldn't depend on each other's incidental leftovers.

### Coupling this phase surfaced (the point of the exercise)

`input_fn` **requires a header row** for `text/csv`. Batch Transform can split
input into line-chunks, and only the first chunk would carry the header — the
rest would misparse silently as data. So the job ran with `split_type=None`:
whole file as one payload, header intact. The container log confirms it — a
single `POST /invocations` returning 197,824 bytes.

That works here and does not scale: it caps input at `MaxPayloadInMB` (6 MB
default, 100 max). A production handler would take headerless positional CSV
with an explicit column list, or JSON Lines, and split freely.

Naming this rather than hiding it, because surfacing exactly this sort of
coupling is what the phase is for. `inference.py` is endpoint-agnostic —
confirmed — but it is not payload-size-agnostic.

### The `id` mistake, stated plainly

Output is scores only, one per line, positionally aligned to the input. Keying
them to records needs `join_source="Input"`, which needs line-splitting, which
the header requirement rules out. The two constraints interact.

It would not have helped anyway: **Phase 1 dropped the `id` column.** That was
right for modelling — `id` is a row identifier, not a feature — but wrong for
the pipeline, and the two got conflated. The production shape keeps the
identifier in the file, uses `input_filter` to withhold it from the model, and
`join_source` to re-attach it to the prediction. The identifier must travel with
the data without entering the model.

Not fixing it: regenerating the Phase 1 CSVs would modify completed work and
break artifact lineage for a lesson already captured here. Cheap to learn now,
expensive when a production job returns a million scores that can't be keyed
back to anything.

### Criterion 2 — batch vs. real-time

**The question is not "how fast" but "do you know the inputs before the request
arrives?"**

Reach for **Batch Transform** when inputs are enumerable ahead of time: nightly
scoring, email and push campaigns, daily recommendations, risk scores refreshed
on a schedule, backfilling a new model version over history, or replaying logged
traffic to compare a candidate model before shipping. It provisions, runs, and
terminates — no idle cost, no capacity planning, no resource to forget. Cost is
proportional to work done.

Reach for a **real-time endpoint** when the input only exists at request time.
Ad ranking is the canonical case: the candidate set depends on the request, so
nothing can be precomputed. You pay for an always-on instance regardless of
traffic, and that is the price of answering questions you couldn't anticipate.

The economics differ in kind, not degree. Phase 4 cost ~$0.006 for ~30 requests
and would have cost the same for zero. This job cost ~$0.01 for 10,000 rows and
would cost nothing tomorrow if not run. **Endpoints bill for availability;
batch bills for work.**

Worth noting: our model is logistic regression, so it is additively separable —
its score genuinely *could* be decomposed into per-user and per-ad partial dot
products computed offline. That is why early ad systems were linear. The moment
cross features or a hidden layer appear, separability is lost, and that is
exactly the trade modern rankers make: give up cheap precomputation to buy the
interactions that carry the accuracy.

**This phase is the one that maps to Lyft.** Scheduled batch scoring is
LyftLearn Compute's job, alongside training and notebooks. Phase 4's endpoint
corresponds to nothing in their stack — LyftLearn Serving is Kubernetes. Of the
two serving phases, the optional one is the realistic one.

---

## Status: Phase 6 — criterion 1 PASS, criterion 2 pending console check

| Criterion | Result |
|---|---|
| 1. `list-endpoints` returns empty | PASS — `[]` in us-east-1, us-west-2, us-east-2 |
| 2. Spend well under budget | pending — needs a browser check (see below) |

`make aws-down` is the one command. `make aws-status` is the read-only version,
which is the right thing to run before closing the laptop.

### Total project spend: ~$0.02 of $15 (0.15%)

| | seconds | instance | cost |
|---|---|---|---|
| training (4 jobs, 3 failed) | 227 | ml.m5.large | $0.0073 |
| endpoint | 330 | ml.t2.medium | $0.0060 |
| batch transform | ≤270 | ml.m5.large | ≤$0.0086 |
| S3 (~7 MB) | — | — | ~$0.0002/mo |

Transform is an upper bound: the job's own start→end span was 73s, but SageMaker
bills instance time including provisioning, so actual is $0.002–$0.009.

Criterion 2 can't be checked from the CLI — **both** `budgets:ViewBudget` and
`ce:GetCostAndUsage` are denied to `learn-sage-dev`. That's the Phase 0
least-privilege scoping still holding, same as it did in Phase 0 itself. Don't
widen the user to make a check convenient.

### The bug the free test found

`aws_down.py` initially gated cleanup behind `if billing:`. Endpoint configs are
free, so whenever nothing happened to be billing the cleanup was **skipped
entirely** — and the script then printed `CLEAN` with a stale config listed
directly above it in the same output. Technically true about billing, and
thoroughly misleading.

Found by creating a throwaway endpoint config (free — no instance) and watching
`make aws-down` fail to delete it. A teardown script that has only ever reported
"0 billing" is unproven; it has to be shown deleting something.

This is the third time in this project that a green result meant nothing until
something was made to fail: Phase 1's IAM scoping, Phase 3's registry
validation, now this. **The pattern is the lesson.**

Fixed twice over: cleanup now runs unconditionally under `--confirm`, and the
final verdict distinguishes "nothing is billing" from "nothing is stale."

### Teardown means stopping the billing, not deleting the project

`make aws-down` keeps the bucket, the IAM role and policies, the model artifact,
the predictions, the registry, and the Models. All are free or ~$0.0002/month.

Deliberate: **Phase 8 imports exactly the resources that survive this script**,
and the observation that those are precisely the resources belonging in
Terraform is that phase's organizing idea. A teardown that nuked the bucket
would delete Phase 8's subject matter along with Phase 1's data.

`make aws-purge` exists for genuinely finishing, and is not run.

### Hazards the script handles, each learned by hitting it

1. **`code/` is coupled to Models.** Purging deletes the Models *first*, because
   removing `s3://<bucket>/code/` while a Model references it leaves the Model
   intact but silently broken — exactly what I did between Phases 4 and 5.
2. **Deletion is asynchronous.** An endpoint in `Deleting` still bills, so the
   script polls until the count is actually zero rather than trusting that the
   API call returned.
3. **Resources hide in other regions.** The Phase 2 quota hunt touched
   us-west-2 and us-east-2. A single-region teardown would miss anything
   stranded there.
4. **Studio domains.** Checked but not auto-deleted — removing one requires
   deleting user profiles, apps and spaces first, which is too destructive to do
   implicitly. It reports loudly instead. None exist; the check is there because
   an EFS volume is the classic thing a teardown written before it existed will
   not know to look for.

### `teardown_endpoint.py` superseded (flagged per rule 6)

Phase 4's script is superseded by `aws_down.py`, which does everything it did
plus multi-region sweep, job stopping, Studio detection, and the
bills-vs-free distinction. Kept rather than deleted — it is a completed phase's
artifact and still works for the single-region case — with a pointer added at
the top of its docstring. That docstring edit is the only change to Phase 4 code.

---

## Environment

| Thing | Value |
|---|---|
| AWS account | `ACCOUNT_ID` |
| IAM user | `learn-sage-dev` |
| Region | `us-east-1` |
| AWS CLI | 2.36.10 (Homebrew) |
| Python | 3.12.13 (uv-managed, `.venv/`) |
| `boto3` | 1.43.58 |
| `sagemaker` | 2.257.5 — **pinned to v2 deliberately, see below** |

IAM policies attached to `learn-sage-dev`: `AmazonSageMakerFullAccess`,
`AmazonS3FullAccess`, `CloudWatchLogsReadOnlyAccess`.

---

## Decisions

### `sagemaker` is pinned to `>=2,<3` — do not casually unpin

`uv add sagemaker` resolves to **v3**, which **removed the `SKLearn`
estimator** (`sagemaker.sklearn.estimator.SKLearn`) that PHASES.md Phase 2
explicitly specifies. v3 replaces it with `ModelTrainer`
(`from sagemaker.train import ModelTrainer`).

Pinned to v2 (2.257.5) so Phases 2–5 work as written. Verified importable on
the pin: `SKLearn` (Phase 2 training) and `SKLearnModel` (Phases 3–4 serving).

v2 prints a deprecation warning on import; set `SAGEMAKER_SUPPRESS_V2_WARNING=1`
to silence. The tradeoff was accepted knowingly: v2 matches the written spec
and most existing docs/tutorials, but is on a deprecation path, so the wrapper
API learned here is one AWS is moving away from. The underlying *primitives*
(training jobs, model artifacts, endpoints, batch transform) are identical
either way, and those are the actual learning goal.

If this ever needs revisiting, it's a PHASES.md amendment across Phases 2–5,
not a one-line dependency bump.

### `AmazonS3FullAccess` is broader than it should be

It grants every bucket in the account, not just this project's. Left broad
because the bucket doesn't exist until Phase 1. **Phase 1 follow-up: narrow to
a single-bucket policy** once the bucket name is known. Flagged rather than
built ahead, per the "work only on the phase named" rule.

---

## Secrets handling

**The AWS access-key CSV lives in `secrets/`, which is git-ignored. It must
never be committed.**

- `secrets/` is ignored via `.gitignore` (a directory rule — harder to get
  wrong than per-file patterns, which are also present as a backstop).
- Verified: `git check-ignore` matches, and `git add -A --dry-run` does not
  pick it up.
- History is clean — the CSV was never staged and never committed. The repo
  has **no remote** configured as of this writing.

**Before ever adding a GitHub remote**, re-verify with:

```sh
git add -A --dry-run          # nothing from secrets/ should appear
git check-ignore -v secrets/  # should print a matching rule
git log --all --oneline -- secrets/   # should be empty
```

The CSV is **redundant** — its contents already live in `~/.aws/credentials`,
which is what the CLI actually reads. Deleting it entirely is the safest end
state; it's kept only as a convenience copy.

**If a key ever does leak:** rotate, don't just delete. IAM → the user →
Security credentials → deactivate and delete the exposed key, then create a new
one. Removing a file from git history does not un-leak a pushed secret.

---

## Reminder: the serving phase is not the Lyft path

Per `CLAUDE.md` — Phase 4's SageMaker real-time endpoint teaches genuine AWS
deployment mechanics but is **not** how Lyft serves models. LyftLearn Serving
runs on Kubernetes; only LyftLearn Compute (training, batch, notebooks) maps to
SageMaker. Keep that distinction live rather than noting it once.
