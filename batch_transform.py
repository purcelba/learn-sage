"""Phase 5: score test.csv in bulk with no endpoint running.

The other offline pattern. Like a training job and unlike an endpoint, a
Transform job provisions, runs, and terminates itself -- there is nothing to
forget to delete. This is much closer to how a platform would run a scheduled
scoring job than Phase 4's endpoint is.

## What this is really testing

It reuses `src/inference.py` **unchanged**. That's a contract in CLAUDE.md, and
the point is architectural: if the serving code had picked up any
endpoint-specific assumption, this job would fail. Verify with
`git diff src/inference.py` afterwards -- an empty diff is the actual result of
this phase, not the predictions file.

## A coupling this surfaced

`input_fn` requires a header row for text/csv. Batch Transform can split a file
into line-chunks (`split_type="Line"`), and only the FIRST chunk would carry the
header -- the rest would misparse, silently, as data. So we send the whole file
as one payload (`split_type=None`, the SDK default), which keeps the header
intact and is fine for 1.35 MB.

That is a real constraint, not a non-issue: it caps input at what fits in one
payload (MaxPayloadInMB, default 6 MB, max 100). A production handler would
accept headerless positional CSV plus an explicit column list, or JSON Lines,
and would then scale to arbitrarily large inputs. Worth naming rather than
hiding, since surfacing exactly this kind of coupling is why the phase exists.

## Why the output is scores-only

Predictions come back one per line, positionally aligned to the input. Keying
them to records would use `join_source="Input"` -- but that needs line-splitting,
which the header requirement above rules out. The two constraints interact.

It would not help anyway: Phase 1 dropped the `id` column, so there is no key to
join on. Dropping it was right for modelling (`id` is not a feature) but wrong
for the pipeline (it is still needed downstream). The production shape keeps the
identifier in the file, uses `input_filter` to withhold it from the model, and
`join_source` to re-attach it. See NOTES.md.

Usage:
    python batch_transform.py --dry-run
    python batch_transform.py --confirm
"""

from __future__ import annotations

import argparse
import sys

import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel

REGION = "us-east-1"
BUCKET_TEMPLATE = "learn-sage-{account_id}"
ROLE_NAME = "AmazonSageMaker-learn-sage-ExecutionRole"
TRAINING_JOB = "learn-sage-pctr-2026-07-30-19-35-01-384"
FRAMEWORK_VERSION = "1.4-2"

# This script builds its own SKLearnModel rather than reusing the one Phase 4's
# deploy left behind, for a reason learned the hard way: Phase 4's cleanup
# deleted the s3://<bucket>/code/ prefix as "SDK scratch". It is not scratch --
# it is where a deployed Model's source code permanently lives
# (SAGEMAKER_SUBMIT_DIRECTORY). Deleting it left the Model object intact and
# pointing at a tarball that no longer existed, with nothing reporting the
# dangling reference. (Note CreateModel *does* validate ModelDataUrl exists but
# evidently not the submit directory.)
#
# Constructing the model here re-uploads src/ and makes this phase independent
# of another phase's leftovers.
MODEL_NAME = "learn-sage-pctr-batch-model"

INSTANCE_TYPE = "ml.m5.large"
USD_PER_HOUR = 0.115


def resolve_role_arn(session: boto3.Session) -> str:
    iam = session.client("iam")
    for page in iam.get_paginator("list_roles").paginate():
        for role in page["Roles"]:
            if role["RoleName"] == ROLE_NAME:
                return role["Arn"]
    raise SystemExit(f"Role {ROLE_NAME!r} not found -- see iam/sagemaker-execution-role.md")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--confirm", action="store_true",
                    help="required to launch a billable job")
    args = ap.parse_args()

    session = boto3.Session(region_name=REGION)
    account_id = session.client("sts").get_caller_identity()["Account"]
    bucket = BUCKET_TEMPLATE.format(account_id=account_id)
    role_arn = resolve_role_arn(session)

    input_uri = f"s3://{bucket}/data/test/test.csv"
    output_uri = f"s3://{bucket}/predictions"
    artifact = f"s3://{bucket}/output/{TRAINING_JOB}/output/model.tar.gz"

    print(f"model       : {MODEL_NAME}  (bound to src/inference.py)")
    print(f"input       : {input_uri}")
    print(f"output      : {output_uri}")
    print(f"instance    : {INSTANCE_TYPE}  (${USD_PER_HOUR}/hr, per-second)")
    print(f"split_type  : None -- whole file as one payload, header preserved")
    print(f"\nestimated cost: ~${USD_PER_HOUR * 5 / 60:.3f} for a ~5-minute job")
    print("Self-terminating: no endpoint, nothing left running afterwards.")

    if args.dry_run or not args.confirm:
        print("\nNo job submitted. Re-run with --confirm.")
        sys.exit(0)

    sm_session = sagemaker.Session(boto_session=session)

    # Identical construction to Phase 4's deploy -- same artifact, same image,
    # same entry point, same source_dir. The ONLY difference between real-time
    # and batch is what we call next: .deploy() versus .transformer(). That
    # symmetry is the phase's actual claim.
    model = SKLearnModel(
        model_data=artifact,
        role=role_arn,
        entry_point="inference.py",
        source_dir="src",
        framework_version=FRAMEWORK_VERSION,
        py_version="py3",
        sagemaker_session=sm_session,
        code_location=f"s3://{bucket}/code",
        name=MODEL_NAME,
    )

    transformer = model.transformer(
        instance_count=1,
        instance_type=INSTANCE_TYPE,
        output_path=output_uri,
        accept="text/csv",          # output_fn returns one float per line
        assemble_with="Line",
    )

    print("\nSubmitting transform job...")
    transformer.transform(
        data=input_uri,
        content_type="text/csv",    # input_fn's CSV branch, header row required
        split_type=None,            # see module docstring
        logs=True,
        wait=True,
    )

    print(f"\njob name   : {transformer.latest_transform_job.name}")
    print(f"predictions: {output_uri}/test.csv.out")


if __name__ == "__main__":
    main()
