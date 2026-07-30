"""Phase 3: make the trained artifact a first-class SageMaker object.

Creates two things, both of which are **pure metadata** -- no compute, no cost,
nothing loaded or executed:

1. A SageMaker `Model`: a binding of {container image, model artifact, IAM role}.
2. A Model Package in the Model Registry: the same binding, plus a version
   number and an approval status.

## What these do and don't validate -- measured, not assumed

Tested by deliberately registering bad versions and observing what SageMaker
rejects:

| Thing | Validated? |
|---|---|
| Model artifact S3 object exists | **YES** -- `ValidationException: Cannot find S3 object` |
| `ModelMetrics` S3 URI exists | **NO** -- a nonexistent path registers fine |
| The metrics themselves | **NO** -- nothing opens the file |
| Container can actually load the artifact | **NO** -- never attempted |
| `ModelApprovalStatus` justified by anything | **NO** -- set it to `Approved` directly |

So it verifies the artifact is *present*, and nothing about whether it is *good*.
`ModelMetrics` stores only a pointer; point it at a file claiming AUC 0.99, or at
no file at all, and registration succeeds either way.

The registry is a ledger, not a judge. The actual gate is three separate pieces:
you run the evaluation, you decide whether it passes, and your deployment
pipeline refuses to ship anything not marked Approved. The registry only records
the verdict. Automating the decision is SageMaker Pipelines' ConditionStep --
different machinery, out of scope here.

Worth internalising the distinction: "we have a model registry" and "we have an
enforced quality gate" are not the same claim.

Usage:
    python register_model.py --dry-run
    python register_model.py --confirm
"""

from __future__ import annotations

import argparse
import sys

import boto3
import sagemaker
from sagemaker.image_uris import retrieve as retrieve_image_uri

REGION = "us-east-1"
BUCKET_TEMPLATE = "learn-sage-{account_id}"
ROLE_NAME = "AmazonSageMaker-learn-sage-ExecutionRole"

# The training job whose artifact we're registering. Naming the Model after it
# keeps lineage readable: given a Model, you can tell which job produced it.
TRAINING_JOB = "learn-sage-pctr-2026-07-30-19-35-01-384"

# Same container family that ran train.py. Phase 4 pairs this image with
# inference.py to serve; the image itself is identical either way -- it's a
# runtime, not a model.
FRAMEWORK_VERSION = "1.4-2"

MODEL_NAME = f"learn-sage-pctr-{TRAINING_JOB.split('-', 3)[-1]}"
PACKAGE_GROUP = "learn-sage-pctr"


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
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--skip-registry", action="store_true",
                    help="create only the Model, not the Model Package")
    args = ap.parse_args()

    session = boto3.Session(region_name=REGION)
    account_id = session.client("sts").get_caller_identity()["Account"]
    bucket = BUCKET_TEMPLATE.format(account_id=account_id)
    role_arn = resolve_role_arn(session)
    sm = session.client("sagemaker")

    artifact = f"s3://{bucket}/output/{TRAINING_JOB}/output/model.tar.gz"
    metrics = f"s3://{bucket}/eval/{TRAINING_JOB}/metrics.json"

    # Resolved rather than hardcoded. The ECR account hosting AWS's prebuilt
    # images differs per region (us-east-1 is 683313688378, others differ), so a
    # copy-pasted URI silently breaks everywhere but the region you copied it
    # from.
    #
    # image_scope="training" even though this Model is for serving. Not a
    # mistake -- sklearn uses ONE image family for both (same ECR repository,
    # `sagemaker-scikit-learn`, just different tags), but this SDK version's
    # *inference* version list is stale: it stops at 1.2-1 while training goes
    # to 1.4-2. Asking for inference scope raises ValueError on 1.4-2.
    #
    # Taking 1.2-1 instead would be a correctness bug, not a cosmetic one: the
    # artifact was pickled by scikit-learn 1.4.2 inside the 1.4-2 container, and
    # sklearn does not support loading a newer version's pickle in an older
    # runtime. The Model must point at the image that produced the artifact.
    image_uri = retrieve_image_uri(
        framework="sklearn", region=REGION, version=FRAMEWORK_VERSION,
        py_version="py3", instance_type="ml.m5.large", image_scope="training",
    )

    print(f"model name  : {MODEL_NAME}")
    print(f"image       : {image_uri}")
    print(f"artifact    : {artifact}")
    print(f"role        : {ROLE_NAME}")
    print(f"registry    : {'skipped' if args.skip_registry else PACKAGE_GROUP}")
    print(f"metrics     : {metrics}")
    print("\ncost: $0 -- metadata only, nothing is provisioned or executed.")

    if args.dry_run or not args.confirm:
        print("\nNothing created. Re-run with --confirm.")
        sys.exit(0)

    # --- 1. The Model object -------------------------------------------------
    # No inference.py: that's Phase 4's first task and building it here would be
    # working ahead. Consequence, stated plainly -- this Model would NOT serve
    # correctly if deployed today. The container's default handler loads
    # model.joblib but knows nothing about featurizer.joblib or the "col=value"
    # tokenisation, so it would score raw strings and fail. That SageMaker lets
    # us create it anyway is the lesson: a Model is a metadata binding, not a
    # validated serving unit.
    sm.create_model(
        ModelName=MODEL_NAME,
        PrimaryContainer={"Image": image_uri, "ModelDataUrl": artifact},
        ExecutionRoleArn=role_arn,
    )
    print(f"\ncreated Model: {MODEL_NAME}")

    if args.skip_registry:
        return

    # --- 2. The Model Registry ----------------------------------------------
    try:
        sm.create_model_package_group(
            ModelPackageGroupName=PACKAGE_GROUP,
            ModelPackageGroupDescription=(
                "pCTR model on Avazu. Versions are LogisticRegression over "
                "hashed categorical features; see metrics.json per version."
            ),
        )
        print(f"created Model Package Group: {PACKAGE_GROUP}")
    except sm.exceptions.ResourceInUse:
        print(f"Model Package Group {PACKAGE_GROUP} already exists -- reusing")

    resp = sm.create_model_package(
        ModelPackageGroupName=PACKAGE_GROUP,
        ModelPackageDescription=f"From training job {TRAINING_JOB}",
        InferenceSpecification={
            "Containers": [{"Image": image_uri, "ModelDataUrl": artifact}],
            "SupportedContentTypes": ["text/csv", "application/json"],
            "SupportedResponseMIMETypes": ["application/json"],
        },
        ModelMetrics={
            # A POINTER. SageMaker stores this URI and never reads the file.
            "ModelQuality": {
                "Statistics": {"ContentType": "application/json", "S3Uri": metrics}
            }
        },
        # Deliberately not Approved. The transition below is a separate,
        # explicit act -- which is the whole point.
        ModelApprovalStatus="PendingManualApproval",
    )
    arn = resp["ModelPackageArn"]
    version = arn.rsplit("/", 1)[-1]
    print(f"registered version {version}, status=PendingManualApproval")

    # Approving is one API call with nothing checking it. If no pipeline gates
    # on this status, "Approved" means only that somebody set a string.
    sm.update_model_package(
        ModelPackageArn=arn,
        ModelApprovalStatus="Approved",
        ApprovalDescription=(
            f"AUC 0.7322 vs 0.5 chance; log loss 0.4063 vs 0.4575 baseline. "
            f"Approved manually -- no automated threshold check exists."
        ),
    )
    print(f"approved version {version}")
    print(f"\nmodel package ARN: {arn}")


if __name__ == "__main__":
    main()
