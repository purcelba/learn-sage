"""Submit src/train.py to SageMaker as a Training Job.

This is the boundary between what the platform does and what we own. Everything
here is instructions *to* SageMaker: which container, which instance, where the
data is, where to put the result. None of it is machine learning -- all of that
lives in src/train.py and src/features.py.

At Lyft this layer is roughly what LyftLearn Compute provides; a DS/MLE team
writes the training script and the platform handles submission, provisioning,
and artifact capture.

Costs money. Guarded behind --confirm.

Usage:
    python submit_training.py --dry-run
    python submit_training.py --confirm
"""

from __future__ import annotations

import argparse
import sys

import boto3
import sagemaker
from sagemaker.sklearn.estimator import SKLearn

REGION = "us-east-1"
BUCKET_TEMPLATE = "learn-sage-{account_id}"
ROLE_NAME = "AmazonSageMaker-learn-sage-ExecutionRole"

# Must match the local scikit-learn pin in pyproject.toml. Phase 4 has to load
# this job's artifact locally and reproduce the endpoint's prediction exactly;
# a version gap between the container that writes the pickle and the interpreter
# that reads it is a needless way to make that comparison fail.
FRAMEWORK_VERSION = "1.4-2"
PY_VERSION = "py3"

INSTANCE_TYPE = "ml.m5.large"
USD_PER_HOUR = 0.115  # us-east-1 on-demand, billed per second

HYPERPARAMETERS = {
    # Chosen by a local sweep over C in {0.1, 0.3, 1.0} -- free, and it meant the
    # paid job runs a configuration we already know converges and performs.
    #
    # "reg-c", not "C": SageMaker renders a single-character hyperparameter name
    # with a single dash (`-C 0.1`), which argparse won't match against a `--C`
    # declaration. Keep hyperparameter names multi-character.
    "reg-c": 0.1,
    "max-iter": 1000,
    "n-features": 2 ** 18,
}


def resolve_role_arn(session: boto3.Session) -> str:
    """Look the role up by name rather than hardcoding its ARN.

    The ARN embeds the AWS account ID, and this repo is public (CLAUDE.md rule
    7). Resolving at runtime keeps the ID out of version control entirely.
    """
    iam = session.client("iam")
    paginator = iam.get_paginator("list_roles")
    for page in paginator.paginate():
        for role in page["Roles"]:
            if role["RoleName"] == ROLE_NAME:
                return role["Arn"]
    raise SystemExit(
        f"Role {ROLE_NAME!r} not found. Create it first -- see "
        f"iam/sagemaker-execution-role.md. The name must contain "
        f"'AmazonSageMaker' or iam:PassRole will be denied."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--confirm", action="store_true",
                    help="required to actually submit a billable job")
    ap.add_argument("--instance-type", default=INSTANCE_TYPE)
    # SageMaker quotas are per-region. The bucket stays in us-east-1 regardless;
    # cross-region reads of a 6.7 MB dataset cost fractions of a cent, so this is
    # a viable escape hatch if one region's quota is stuck at zero. Note that a
    # job launched here must be torn down / checked in THAT region.
    ap.add_argument("--region", default=REGION)
    args = ap.parse_args()

    session = boto3.Session(region_name=args.region)
    account_id = session.client("sts").get_caller_identity()["Account"]
    bucket = BUCKET_TEMPLATE.format(account_id=account_id)
    role_arn = resolve_role_arn(session)

    train_uri = f"s3://{bucket}/data/train/"
    test_uri = f"s3://{bucket}/data/test/"
    output_uri = f"s3://{bucket}/output"
    code_uri = f"s3://{bucket}/code"

    print(f"region        : {args.region}")
    print(f"role          : {ROLE_NAME}")
    print(f"container     : SKLearn {FRAMEWORK_VERSION} ({PY_VERSION})")
    print(f"instance      : {args.instance_type}  (${USD_PER_HOUR}/hr, per-second billing)")
    print(f"entry point   : src/train.py")
    print(f"hyperparams   : {HYPERPARAMETERS}")
    print(f"train channel : {train_uri}")
    print(f"test channel  : {test_uri}")
    print(f"output        : {output_uri}")
    print(f"\nestimated cost: ~${USD_PER_HOUR * 5 / 60:.3f} for a 5-minute job "
          f"(provisioning included; may be $0 under the SageMaker free tier)")

    if args.dry_run or not args.confirm:
        print("\nNo job submitted. Re-run with --confirm to launch.")
        sys.exit(0)

    sm_session = sagemaker.Session(boto_session=session)

    estimator = SKLearn(
        entry_point="train.py",
        source_dir="src",          # ships features.py alongside train.py, so
                                   # serving can import the identical module
        role=role_arn,
        instance_type=args.instance_type,
        instance_count=1,
        framework_version=FRAMEWORK_VERSION,
        py_version=PY_VERSION,
        hyperparameters=HYPERPARAMETERS,
        # Both point at the project bucket on purpose. Left unset, the SDK
        # creates and uses its own sagemaker-{region}-{account} bucket, which
        # would quietly break the "one bucket for the whole project" convention
        # and scatter artifacts somewhere teardown doesn't look.
        output_path=output_uri,
        code_location=code_uri,
        sagemaker_session=sm_session,
        base_job_name="learn-sage-pctr",
    )

    print("\nSubmitting... (streams container logs; Ctrl-C detaches but does NOT stop the job)")
    estimator.fit({"train": train_uri, "test": test_uri}, logs="All")

    print(f"\njob name      : {estimator.latest_training_job.name}")
    print(f"model artifact: {estimator.model_data}")


if __name__ == "__main__":
    main()
