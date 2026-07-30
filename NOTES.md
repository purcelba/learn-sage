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
