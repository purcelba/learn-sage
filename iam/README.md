# iam/

Policy documents for this project, kept as files so they're reviewable and
diffable rather than living only in the AWS console.

## `learn-sage-s3-policy.json.template`

Replaces `AmazonS3FullAccess` on the `learn-sage-dev` IAM user, narrowing it
from "every bucket in the account" to just `learn-sage-<your account id>`. This
was flagged as a Phase 1 follow-up in `NOTES.md` — the policy couldn't be
written earlier because the bucket name didn't exist until Phase 1.

### Rendering it

The file is a **template**: the bucket ARNs contain the literal string
`ACCOUNT_ID`, because this repo is public and the real ID shouldn't be in it.
Render it with your own account ID before use:

```sh
sed "s/ACCOUNT_ID/$(aws sts get-caller-identity --query Account --output text)/g" \
  iam/learn-sage-s3-policy.json.template | pbcopy
```

That puts the finished policy on your clipboard, ready to paste into the IAM
console. If you'd rather write it to disk than pipe it, the rendered
`iam/learn-sage-s3-policy.json` is git-ignored precisely so it can't be
committed by accident.

It's a `.template` rather than a `.json` with a placeholder for a specific
reason: `"arn:aws:s3:::learn-sage-ACCOUNT_ID"` is *perfectly valid* JSON and a
*perfectly valid* policy. Applied unrendered, it would attach without error and
silently grant access to a bucket that doesn't exist — breaking S3 access with
no message pointing at the cause. The extension makes it impossible to paste by
accident.

### Why each action is here

| Action | Needed for |
|---|---|
| `s3:ListBucket` | `aws s3 ls`, and the `list_objects_v2` call in `verify_data.py` |
| `s3:ListBucketVersions` | the bucket has versioning enabled; listing versions without this returns `AccessDenied` |
| `s3:GetBucketLocation` | the `sagemaker` SDK resolves a bucket's region before using it |
| `s3:GetObject` | reading the CSVs; Phase 2's training job reading its input channels |
| `s3:GetObjectVersion` | **not needed yet.** Included because the bucket is versioned, so any later phase that reads a non-current version would otherwise fail with a confusing `AccessDenied` rather than a clear "no such version" |
| `s3:PutObject` | uploads; Phase 2 writing `model.tar.gz` to `output/` |
| `s3:DeleteObject` | Phase 6 teardown |

This policy deliberately does **not** grant `s3:CreateBucket`,
`s3:DeleteBucket`, or `s3:ListAllMyBuckets`. The bucket already exists and the
project only ever needs one.

### What this actually achieved — measured, not assumed

Verified after detaching `AmazonS3FullAccess`:

| Check | Result |
|---|---|
| Read/list the project bucket | allowed (intended) |
| Read a third-party bucket (`s3://nyc-tlc/`) | **denied** — this is the win |
| `aws s3 ls` with no args (list all buckets) | still **allowed** |
| Create a *different* bucket | still **allowed** |

The last two leak in from **`AmazonSageMakerFullAccess`**, which is still
attached and grants `s3:CreateBucket` and `s3:ListAllMyBuckets` account-wide —
SageMaker needs to create its own default `sagemaker-{region}-{account}` bucket,
so AWS bakes those into the managed policy. Nothing in this inline policy can
take them away; IAM unions Allows and only an explicit Deny overrides them.

**This is accepted, not a bug to fix.** `CLAUDE.md` Phase 0 says SageMaker full
access is fine for a learning sandbox, and Phase 2 needs it. The meaningful
exposure — read/write on every bucket in the account — is closed. What remains
is the ability to create empty buckets and enumerate bucket names.

If it ever needed closing, the tool would be an explicit `Deny` statement on
`s3:CreateBucket` scoped to exclude SageMaker's own default bucket pattern. That
is more IAM complexity than a learning sandbox warrants.

### Lesson worth keeping

The positive test (`verify_data.py` still passes) proved almost nothing on its
own — it passes identically whether or not the scoping worked, because it only
ever exercises the one bucket that's supposed to work. Only the **negative**
tests revealed that two permissions survived. When tightening a permission, test
what should now *fail*; "the thing I care about still works" is not evidence
that anything was actually restricted.

Caveat on writing those negative tests: prefer non-mutating probes. The
`create-bucket` check above succeeded and left a real stray bucket behind that
the scoped user then couldn't delete (`DeleteBucket` isn't granted) — it had to
be removed from the console as root.

### Applying it

`learn-sage-dev` has no IAM write permissions on itself — by design — so this is
a console operation:

1. IAM → Users → `learn-sage-dev` → Permissions
2. Add permissions → Create inline policy → JSON → paste the *rendered* policy (see "Rendering it" above)
3. Name it `learn-sage-s3-scoped`
4. **Then** detach `AmazonS3FullAccess` (in that order — detaching first would
   briefly leave the user with no S3 access at all)

Afterwards run `python verify_data.py` again. It should still pass. If it
doesn't, the policy is wrong and re-attaching `AmazonS3FullAccess` restores the
previous state.

## Still to come (Phase 2, not built yet)

Phase 2's training job runs as a **SageMaker execution role**, which is a
different principal from `learn-sage-dev` — the user submits the job, the role
is what the container actually runs as. That role needs its own S3 access to
this bucket. Flagged here so it isn't a surprise; creating it belongs to
Phase 2, not this phase.
