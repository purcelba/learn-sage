# SageMaker execution role

Phase 2 needs a role for the training job to run as. This is **a different
principal from `learn-sage-dev`**, and that distinction is the whole point:

- **`learn-sage-dev` (your IAM user)** *submits* the job. It needs
  `sagemaker:CreateTrainingJob` and `iam:PassRole`.
- **The execution role** is what the *container* runs as. It needs to read the
  training data from S3, write the model artifact back, and ship logs to
  CloudWatch.

Consequence worth internalizing: a training job can fail with `AccessDenied` on
S3 while your CLI reads the exact same bucket perfectly. Nothing is wrong with
your credentials — the container isn't using them. Two principals, two sets of
permissions.

In a company with a platform team, the equivalent role is platform-owned and
you would never create it yourself. Knowing it exists is what lets you tell "my
job is broken" from "the platform's role is misconfigured."

## The role name is NOT arbitrary

**The role name must contain the string `AmazonSageMaker`.**

`AmazonSageMakerFullAccess` — attached to `learn-sage-dev` — grants `iam:PassRole`
only for roles matching `arn:aws:iam::*:role/*AmazonSageMaker*`:

```json
{
  "Effect": "Allow",
  "Action": ["iam:PassRole"],
  "Resource": "arn:aws:iam::*:role/*AmazonSageMaker*",
  "Condition": {"StringEquals": {"iam:PassedToService": "sagemaker.amazonaws.com"}}
}
```

Name the role `learn-sage-execution-role` and job submission fails with a
`PassRole` denial that points at *your user's* permissions, not at the role —
sending you to debug the wrong principal entirely.

Use: **`AmazonSageMaker-learn-sage-ExecutionRole`**

## Creating it (console, as root)

`learn-sage-dev` can list roles but not create them (`iam:CreateRole` isn't in
`AmazonSageMakerFullAccess`), so this is a root console step.

1. IAM → **Roles** → **Create role**
2. Trusted entity type: **AWS service**
3. Use case: **SageMaker** → pick plain **SageMaker** → Next
   - This attaches `AmazonSageMakerFullAccess` to the role and sets the trust
     policy to `sagemaker.amazonaws.com` automatically.
4. Role name: **`AmazonSageMaker-learn-sage-ExecutionRole`**
5. Create role

That is the fast path and it's fine for a sandbox. It gives the role broader S3
access than it needs — the same tradeoff already documented in `README.md` for
the user, and accepted for the same reason.

### Tighter alternative

To scope the role the way the user was scoped: create the role as above, then
detach `AmazonSageMakerFullAccess` from **the role** and attach an inline policy
rendered from `sagemaker-execution-policy.json.template`. Follow the same
add-then-remove order used in Phase 1, and re-run the training job afterwards to
confirm it still works. Note the negative-test lesson applies here too: a job
that succeeds proves the role has *enough* permission, never that it lacks
anything extra.

## Trust policy (for reference)

The console sets this for you. It says "SageMaker may assume this role" — the
role is useless to anyone else, which is why a role is safer than handing the
container long-lived keys.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {"Service": "sagemaker.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }
  ]
}
```

## After creating it

Verify from the CLI — `learn-sage-dev` can read roles even though it can't
create them:

```sh
aws iam list-roles \
  --query 'Roles[?contains(RoleName,`learn-sage`)].[RoleName,Arn]' --output table
```

`submit_training.py` looks the role up by name, so nothing needs hardcoding.
