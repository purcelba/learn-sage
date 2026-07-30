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
